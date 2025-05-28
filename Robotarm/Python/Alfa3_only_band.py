import numpy as np
import time
import pyqtgraph as pg
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore

class EEGVisualizer:
    def __init__(self, board_shim):
        self.board = board_shim
        self.sampling_rate = BoardShim.get_sampling_rate(self.board.get_board_id())
        self.band_window = 5  # sekunder
        self.num_points = self.band_window * self.sampling_rate
        self.eeg_channels = BoardShim.get_eeg_channels(self.board.get_board_id())
        self.target_channel = self.eeg_channels[1]

        self.alpha_thresh = 20

        self.time_buffer = deque(maxlen=self.num_points)
        self.band_buffers = {
            'Delta': deque(maxlen=self.num_points),
            'Theta': deque(maxlen=self.num_points),
            'Alpha': deque(maxlen=self.num_points),
            'Beta': deque(maxlen=self.num_points),
        }

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='EEG Band Power', size=(1000, 600), show=True)
        self.plot = self.win.addPlot(title="EEG Band Power (last 5 sec)", row=1, col=0)
        self.plot.showGrid(x=True, y=True)
        self.plot.setYRange(0, 50)
        self.plot.setLabel('bottom', 'Tid (s)')
        self.plot.setLabel('left', 'Power (µV²/Hz)')
        self.plot.setXRange(0, self.band_window)

        self.curves = {
            'Delta': self.plot.plot(pen=pg.mkPen('c', width=2), name='Delta'),
            'Theta': self.plot.plot(pen=pg.mkPen('g', width=2), name='Theta'),
            'Alpha': self.plot.plot(pen=pg.mkPen('y', width=2), name='Alpha'),
            'Beta': self.plot.plot(pen=pg.mkPen('m', width=2), name='Beta'),
        }

        self.alpha_thresh_curve = self.plot.plot(pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine), name='Alpha > 20')

        legend = pg.LegendItem(offset=(70, 10))
        legend.setParentItem(self.plot.graphicsItem())
        legend.addItem(self.curves['Delta'], 'Delta (0.5-4 Hz)')
        legend.addItem(self.curves['Theta'], 'Theta (4-7 Hz)')
        legend.addItem(self.curves['Alpha'], 'Alpha (8-13 Hz)')
        legend.addItem(self.curves['Beta'], 'Beta (13-30 Hz)')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        QtWidgets.QApplication.instance().exec()

    def update(self):
        data = self.board.get_current_board_data(self.num_points)
        signal = data[self.target_channel]

        if signal.shape[0] < self.num_points:
            return

        DataFilter.detrend(signal, DetrendOperations.LINEAR.value)
        nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        try:
            psd = DataFilter.get_psd_welch(signal, nfft, nfft // 2, self.sampling_rate, WindowOperations.HANNING.value)
        except Exception as e:
            print(f"PSD-fejl: {e}")
            return
       
        timestamp = time.time()

        self.time_buffer.append(timestamp)

        # Udregn båndstyrker
        band_values = {
            'Delta': DataFilter.get_band_power(psd, 0.5, 4.0),
            'Theta': DataFilter.get_band_power(psd, 4.0, 7.0),
            'Alpha': DataFilter.get_band_power(psd, 8.0, 13.0),
            'Beta':  DataFilter.get_band_power(psd, 13.0, 30.0),
        }

        for band, value in band_values.items():
            self.band_buffers[band].append(value)

        # Tid i sekunder (seneste 5 sek)
        t0 = self.time_buffer[0] if self.time_buffer else timestamp
        x_vals = [t - t0 for t in self.time_buffer]

        # Opdater plot
        for band in self.band_buffers:
            self.curves[band].setData(x_vals, list(self.band_buffers[band]))

        # Highlight alpha hvis over 20
        alpha_curve = np.array(self.band_buffers['Alpha'])
        alpha_highlight = np.where(alpha_curve > self.alpha_thresh, alpha_curve, np.nan)
        self.alpha_thresh_curve.setData(x_vals, alpha_highlight)

        self.plot.setXRange(max(x_vals) - self.band_window, max(x_vals))
        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    try:
        print("👉 Forbereder session...")
        board.prepare_session()
        board.start_stream(450000, "")
        print("🚀 Stream started.")
        EEGVisualizer(board)
    except Exception as e:
        print(f"⚠️ Fejl under kørsel: {e}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("🔚 Session released.")

if __name__ == "__main__":
    main()
