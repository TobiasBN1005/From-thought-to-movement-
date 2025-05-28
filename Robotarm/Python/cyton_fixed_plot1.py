import logging
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4  # seconds
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow EEG Plot', size=(800, 600), show=True)

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
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('EEG Timeseries')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            try:
                DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                self.curves[count].setData(data[channel].tolist())
            except Exception as e:
                print(f"Error in channel {channel}: {e}")

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    # === DINE INDSTILLINGER HER ===
    board_id = BoardIds.CYTON_BOARD.value  # Cyton board ID = 0
    serial_port = "/dev/tty.usbserial-DM0258MO"  # Sørg for at denne port er korrekt

    params = BrainFlowInputParams()
    params.serial_port = serial_port

    board_shim = BoardShim(board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, "")
        print("✅ Stream started – plotting EEG live...")
        Graph(board_shim)
    except BaseException:
        logging.warning('❌ Exception occurred:', exc_info=True)
    finally:
        logging.info('🔁 Releasing session...')
        if board_shim.is_prepared():
            board_shim.release_session()
            logging.info('✅ Session released')


if __name__ == '__main__':
    main()
