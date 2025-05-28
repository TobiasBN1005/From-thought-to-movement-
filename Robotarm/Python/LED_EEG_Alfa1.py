import logging
import time
import serial
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore

channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
alpha_channel_name = 'Fp1'

# EEG-bånddefinitioner: (lav, høj, farve)
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

        # Serial til Teensy (ret portnavn om nødvendigt)
        self.teensy = serial.Serial('/dev/cu.usbmodem170452301', 9600)
        time.sleep(2)  # Giv tid til at initialisere

        self.alpha_threshold = 10
        self.alpha_duration_required = 2.0  # sekunder
        self.alpha_above_start_time = None
        self.led_on = False

        self.band_values = {band: [] for band in band_definitions}
        self.max_band_points = self.sampling_rate * 5
        self.band_curves = {}

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow EEG Plot', size=(1000, 700), show=True)
        self._init_timeseries()

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
            else:
                p.setLabel('left', f'Channel {self.exg_channels[i]}')
            if i == 0:
                p.setTitle('Live EEG Signals')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

        row_index = len(self.exg_channels) + 1
        self.band_plot = self.win.addPlot(row=row_index, col=0)
        self.band_plot.setTitle('EEG Band Power – Last 5 Seconds')
        self.band_plot.setLabel('left', 'Power (µV²/Hz)')
        self.band_plot.setLabel('bottom', 'Tid (sekunder)')
        self.band_plot.showGrid(x=True, y=True)
        self.band_plot.setYRange(0, 50)
        self.band_plot.enableAutoRange(axis='x', enable=False)

        legend = pg.LegendItem(offset=(-20, 10))
        legend.setParentItem(self.band_plot.graphicsItem())

        for band, (_, _, color) in band_definitions.items():
            curve = self.band_plot.plot(pen=color, name=band)
            self.band_curves[band] = curve
            legend.addItem(curve, band)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        band_powers = {}

        for count, channel in enumerate(self.exg_channels):
            try:
                signal = data[channel]
                DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
                DataFilter.perform_bandstop(signal, self.sampling_rate, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
                DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
                self.curves[count].setData(signal.tolist())

                if count < len(channel_names) and channel_names[count] == alpha_channel_name:
                    if len(signal) >= 256:
                        nfft = 256
                        psd = DataFilter.get_psd_welch(signal, nfft, nfft // 2,
                                                       self.sampling_rate, WindowOperations.HANNING.value)
                        for band, (low, high, _) in band_definitions.items():
                            try:
                                band_powers[band] = DataFilter.get_band_power(psd, low, high)
                            except Exception:
                                band_powers[band] = 0.0
            except Exception as e:
                print(f"Error processing channel {channel}: {e}")

        # === Teensy LED kontrol ===
        alpha_value = band_powers.get('Alpha', 0.0)
        self.win.setWindowTitle(
            f"Alpha Power ({alpha_channel_name}): {alpha_value:.2f} µV²/Hz"
        )

        current_time = time.time()
        if alpha_value > self.alpha_threshold:
            if self.alpha_above_start_time is None:
                self.alpha_above_start_time = current_time
            elif current_time - self.alpha_above_start_time >= self.alpha_duration_required and not self.led_on:
                self.teensy.write(b'1')
                self.led_on = True
        else:
            self.alpha_above_start_time = None
            if self.led_on:
                self.teensy.write(b'0')
                self.led_on = False

        # === Opdater graf ===
        reference_band = 'Alpha'
        if reference_band in band_powers:
            for band, value in band_powers.items():
                self.band_values[band].append(value)
                if len(self.band_values[band]) > self.max_band_points:
                    self.band_values[band].pop(0)

            x_values = [i / self.sampling_rate for i in range(len(self.band_values[reference_band]))]

            for band, curve in self.band_curves.items():
                y = self.band_values[band]
                if len(y) == len(x_values):
                    curve.setData(x=x_values, y=y)

            if len(x_values) > 0:
                self.band_plot.setXRange(max(0, x_values[-1] - 5), x_values[-1])

        self.band_plot.setYRange(0, 50)
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
        print(f"⚠️ Fejl under kørsel: {e}")
    finally:
        if board.is_prepared():
            board.release_session()
            print("🔚 Session released.")

if __name__ == "__main__":
    main()
