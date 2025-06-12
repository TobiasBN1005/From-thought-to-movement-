import logging
import time
import serial
import csv
import os
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore
from threading import Thread
from collections import deque

channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
alpha_channel_name = 'Fp1'

band_definitions = {
    'Delta': (0.5, 4.0, 'g'),
    'Theta': (4.0, 8.0, 'b'),
    'Alpha': (8.0, 13.0, 'r'),
    'Beta':  (13.0, 30.0, 'y')
}

class Graph:
    def __init__(self, board_shim):
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.teensy = serial.Serial('/dev/cu.usbmodem170452301', 9600)
        time.sleep(2)
        self.alpha_threshold = 10
        self.alpha_duration_required = 2.0
        self.alpha_above_start_time = None
        self.led_on = False

        # === EMG data buffer ===
        self.emg_buffer = deque(maxlen=500)
        self.emg_plot = None
        self.emg_curve = None

        # === CSV log setup
        self.session_start = time.time()
        log_dir = "eeg_emg_logs"
        os.makedirs(log_dir, exist_ok=True)
        filename = time.strftime("eeg_emg_log_%Y%m%d-%H%M%S.csv")
        self.csv_path = os.path.join(log_dir, filename)
        print(f"📁 Data logges til: {self.csv_path}")

        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['absolute_time', 'session_time (s)'] + channel_names + ['EMG']
            writer.writerow(header)

        self.band_values = {band: [] for band in band_definitions}
        self.band_timestamps = []
        self.max_band_points = self.sampling_rate * 5
        self.band_curves = {}

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='EEG & EMG Live Plot', size=(1000, 800), show=True)
        self._init_timeseries()

        # Start EMG thread
        self.emg_thread = Thread(target=self.read_emg_from_teensy)
        self.emg_thread.daemon = True
        self.emg_thread.start()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []

        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            if i < len(channel_names):
                p.setLabel('left', channel_names[i])
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

        # === EMG plot ===
        row_index = len(self.exg_channels)
        self.emg_plot = self.win.addPlot(row=row_index, col=0)
        self.emg_plot.setTitle("Live EMG Signal")
        self.emg_plot.setLabel("left", "EMG (ADC units)")
        self.emg_plot.setLabel("bottom", "Tid")
        self.emg_curve = self.emg_plot.plot(pen='m')

        # === EEG Band Power ===
        self.band_plot = self.win.addPlot(row=row_index+1, col=0)
        self.band_plot.setTitle('EEG Band Power – Last 5 Seconds')
        self.band_plot.setLabel('left', 'Power (µV²/Hz)')
        self.band_plot.setLabel('bottom', 'Tid (sekunder)')
        self.band_plot.setYRange(0, 50)
        self.band_plot.enableAutoRange(axis='x', enable=False)

        legend = pg.LegendItem(offset=(-20, 10))
        legend.setParentItem(self.band_plot.graphicsItem())

        for band, (_, _, color) in band_definitions.items():
            curve = self.band_plot.plot(pen=color, name=band)
            self.band_curves[band] = curve
            legend.addItem(curve, band)

    def read_emg_from_teensy(self):
        while True:
            try:
                line = self.teensy.readline().decode().strip()
                if line.startswith("EMG:"):
                    try:
                        emg_value = int(line.split(":")[1])
                        self.emg_buffer.append(emg_value)
                    except ValueError:
                        pass

            except Exception:
                continue

    def update(self):
        timestamp = time.time()  # Fælles timestamp for både EEG og EMG
        session_time = timestamp - self.session_start
    
        # === EMG: hent seneste værdi
        latest_emg = self.emg_buffer[-1] if self.emg_buffer else 0
    
        # === EEG: hent seneste data
        data = self.board_shim.get_current_board_data(self.num_points)
        latest_values = [data[ch][-1] * 0.02235 for ch in self.exg_channels]
    
        # === Gem data i CSV-fil med synkroniseret tid
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, round(session_time, 3)] + latest_values + [latest_emg])
    
        band_powers = {}
    
        for count, channel in enumerate(self.exg_channels):
            try:
                signal = data[channel]
                DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
                self.curves[count].setData(signal.tolist())
    
                if count < len(channel_names) and channel_names[count] == alpha_channel_name:
                    if len(signal) >= 256:
                        psd = DataFilter.get_psd_welch(signal, 256, 128,
                                                       self.sampling_rate, WindowOperations.HANNING.value)
                        for band, (low, high, _) in band_definitions.items():
                            try:
                                band_power = DataFilter.get_band_power(psd, low, high)
                                band_powers[band] = band_power
                            except:
                                band_powers[band] = 0.0
            except Exception as e:
                print(f"EEG error channel {channel}: {e}")
    
        # === LED kontrol via alpha-aktivitet
        alpha_value = band_powers.get('Alpha', 0.0)
        self.win.setWindowTitle(f"Alpha Power: {alpha_value:.2f} µV²/Hz")
        now = timestamp  # Brug samme timestamp
    
        if alpha_value > self.alpha_threshold:
            if self.alpha_above_start_time is None:
                self.alpha_above_start_time = now
            elif now - self.alpha_above_start_time >= self.alpha_duration_required and not self.led_on:
                self.teensy.write(b'1')
                self.led_on = True
        else:
            self.alpha_above_start_time = None
            if self.led_on:
                self.teensy.write(b'0')
                self.led_on = False
    
        # === Band power plot
        if 'Alpha' in band_powers:
            self.band_timestamps.append(timestamp)
            if len(self.band_timestamps) > self.max_band_points:
                self.band_timestamps.pop(0)
    
            for band, value in band_powers.items():
                self.band_values[band].append(value)
                if len(self.band_values[band]) > self.max_band_points:
                    self.band_values[band].pop(0)
    
            t0 = self.band_timestamps[0]
            x_values = [t - t0 for t in self.band_timestamps]
    
            for band, curve in self.band_curves.items():
                y = self.band_values[band]
                if len(y) == len(x_values):
                    curve.setData(x=x_values, y=y)
    
            if x_values:
                self.band_plot.setXRange(max(0, x_values[-1] - 5), x_values[-1])
    
        # === EMG-plot
        if len(self.emg_buffer) > 0:
            y_emg = list(self.emg_buffer)
            x_emg = list(range(len(y_emg)))
            self.emg_curve.setData(x=x_emg, y=y_emg)
    
        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"

    board = BoardShim(BoardIds.CYTON_BOARD, params)

    try:
        print("👉 Forbereder session...")
        board.prepare_session()
        board.start_stream(450000, "")
        print("🚀 Stream started.")
        Graph(board)
    except Exception as e:
        print(f"⚠️ Fejl: {e}")
    finally:
        if board.is_prepared():
            board.release_session()
            print("🔚 Session released.")

if __name__ == "__main__":
    main()
