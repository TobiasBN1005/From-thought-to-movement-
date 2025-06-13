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
        self.num_points = 4 * self.sampling_rate

        self.teensy = serial.Serial('/dev/cu.usbmodem170452301', 115200)
        time.sleep(2)
        self.alpha_threshold = 10
        self.alpha_duration_required = 2.0
        self.alpha_above_start_time = None
        self.led_on = False

        self.emg_buffer = deque(maxlen=500)
        self.imu1_buffer = {'ax': deque(maxlen=500), 'ay': deque(maxlen=500), 'az': deque(maxlen=500)}
        self.session_start = time.time()

        log_dir = "eeg_emg_imu_logs"
        os.makedirs(log_dir, exist_ok=True)
        filename = time.strftime("eeg_emg_imu_log_%Y%m%d-%H%M%S.csv")
        self.csv_path = os.path.join(log_dir, filename)
        print(f"📁 Logging til: {self.csv_path}")

        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'session_time'] + channel_names + ['EMG'] + ['IMU1_ax', 'IMU1_ay', 'IMU1_az']
            writer.writerow(header)
        self.max_band_points = int(self.sampling_rate * 5)  # 5 sekunder historik
        self.band_values = {band: [] for band in band_definitions}
        self.band_timestamps = []
        self.band_curves = {}

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='EEG + EMG + IMU Live Plot', size=(1000, 900), show=True)
        self._init_plots()

        # Start Teensy reader thread
        self.teensy_thread = Thread(target=self.read_teensy_data)
        self.teensy_thread.daemon = True
        self.teensy_thread.start()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec()

    def _init_plots(self):
        self.plots = []
        self.curves = []

        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            if i < len(channel_names):
                p.setLabel('left', channel_names[i])
            self.plots.append(p)
            self.curves.append(p.plot())

        row = len(self.exg_channels)
        self.emg_plot = self.win.addPlot(row=row, col=0)
        self.emg_plot.setTitle("Live EMG Signal")
        self.emg_plot.setLabel("left", "EMG (ADC)")
        self.emg_plot.setLabel("bottom", "Tid")
        self.emg_curve = self.emg_plot.plot(pen='m')

        self.imu1_plot = self.win.addPlot(row=row+1, col=0)
        self.imu1_plot.setTitle("IMU1 Acceleration (X, Y, Z)")
        self.imu1_plot.setLabel("left", "Accel (g)")
        self.imu1_plot.setLabel("bottom", "Tid")
        self.imu1_curves = {
            'ax': self.imu1_plot.plot(pen='c', name='X'),
            'ay': self.imu1_plot.plot(pen='y', name='Y'),
            'az': self.imu1_plot.plot(pen='g', name='Z'),
        }

        self.band_plot = self.win.addPlot(row=row+2, col=0)
        self.band_plot.setTitle("EEG Band Power – Last 5 Seconds")
        self.band_plot.setLabel('left', 'µV²/Hz')
        self.band_plot.setLabel('bottom', 'Tid')
        self.band_plot.setYRange(0, 50)
        self.band_plot.enableAutoRange(axis='x', enable=False)

        legend = pg.LegendItem(offset=(-20, 10))
        legend.setParentItem(self.band_plot.graphicsItem())
        for band, (_, _, color) in band_definitions.items():
            curve = self.band_plot.plot(pen=color, name=band)
            self.band_curves[band] = curve
            legend.addItem(curve, band)

    def read_teensy_data(self):
        while True:
            try:
                line = self.teensy.readline().decode().strip()
                if line.startswith("EMG:"):
                    emg_val = int(line.split(":")[1])
                    self.emg_buffer.append(emg_val)
                elif line.startswith("IMU1:"):
                    parts = line[5:].split(",")
                    if len(parts) == 6:
                        ax = float(parts[0])
                        self.imu1_buffer['ax'].append(ax)
                        self.imu1_buffer['ay'].append(float(parts[1]))
                        self.imu1_buffer['az'].append(float(parts[2]))
            except Exception:
                continue

    def update(self):
        now = time.time()
        session_time = now - self.session_start

        # CSV logging: EEG + EMG + IMU1
        data = self.board_shim.get_current_board_data(self.num_points)
        latest_eeg = [data[ch][-1] * 0.02235 for ch in self.exg_channels]
        latest_emg = self.emg_buffer[-1] if self.emg_buffer else 0
        imu_ax = self.imu1_buffer['ax'][-1] if self.imu1_buffer['ax'] else 0
        imu_ay = self.imu1_buffer['ay'][-1] if self.imu1_buffer['ay'] else 0
        imu_az = self.imu1_buffer['az'][-1] if self.imu1_buffer['az'] else 0

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, round(session_time, 3)] + latest_eeg + [latest_emg, imu_ax, imu_ay, imu_az])

        # EEG plot
        for i, channel in enumerate(self.exg_channels):
            signal = data[channel]
            DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
            self.curves[i].setData(signal.tolist())

        # Band Power
        alpha_value = 0.0
        for count, channel in enumerate(self.exg_channels):
            if channel_names[count] == alpha_channel_name:
                signal = data[channel]
                if len(signal) >= 256:
                    psd = DataFilter.get_psd_welch(signal, 256, 128,
                                                   self.sampling_rate, WindowOperations.HANNING.value)
                    for band, (low, high, _) in band_definitions.items():
                        band_power = DataFilter.get_band_power(psd, low, high)
                        self.band_values[band].append(band_power)
                        if band == 'Alpha':
                            alpha_value = band_power

        self.win.setWindowTitle(f"Alpha Power: {alpha_value:.2f} µV²/Hz")
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

        # Band power plot
        self.band_timestamps.append(now)
        if len(self.band_timestamps) > self.max_band_points:
            self.band_timestamps.pop(0)
        for band in self.band_values:
            if len(self.band_values[band]) > self.max_band_points:
                self.band_values[band].pop(0)

        if self.band_timestamps:
            x_vals = [t - self.band_timestamps[0] for t in self.band_timestamps]
            for band, curve in self.band_curves.items():
                y_vals = self.band_values[band]
                if len(x_vals) == len(y_vals):
                    curve.setData(x=x_vals, y=y_vals)
            self.band_plot.setXRange(max(0, x_vals[-1] - 5), x_vals[-1])

        # EMG plot
        if self.emg_buffer:
            self.emg_curve.setData(list(self.emg_buffer))

        # IMU1 plot
        if self.imu1_buffer['ax']:
            self.imu1_curves['ax'].setData(list(self.imu1_buffer['ax']))
            self.imu1_curves['ay'].setData(list(self.imu1_buffer['ay']))
            self.imu1_curves['az'].setData(list(self.imu1_buffer['az']))

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"
    board = BoardShim(BoardIds.CYTON_BOARD, params)

    try:
        board.prepare_session()
        board.start_stream()
        Graph(board)
    finally:
        if board.is_prepared():
            board.release_session()

if __name__ == "__main__":
    main()
