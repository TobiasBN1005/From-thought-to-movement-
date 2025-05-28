import numpy as np
import logging
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore

# Kanalnavne for standard Cyton-opsætning
channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']


def zscore_normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size = 5  # Sekunder
        self.num_points = self.window_size * self.sampling_rate

        pg.setConfigOptions(antialias=True)
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow EEG Plot', size=(1000, 600), show=True)
        self.win.setBackground('k')  # Sort baggrund

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
            p.setLabel('left', channel_names[i], color='white')
            p.setYRange(-5, 5)  # Z-score skala
            if i == 0:
                p.setTitle('Live EEG Signals (Z-score | 1–49Hz bandpass | 50Hz notch)', color='white')
            p.getAxis('left').setPen(pg.mkPen('w'))
            p.getAxis('bottom').setPen(pg.mkPen('w'))
            self.plots.append(p)
            curve = p.plot(pen=pg.mkPen('c'))  # cyan linje
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            try:
                signal = data[channel].copy()

                # --- FILTERING ---
                DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(
                    signal, self.sampling_rate, 1.0, 49.0, 2,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0
                )
                DataFilter.perform_bandstop(
                signal, self.sampling_rate, 49.0, 51.0, 2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0
                )


                # --- NORMALISERING ---
                signal = zscore_normalize(signal)

                # --- TJEK FOR DØD KANAL ---
                if np.std(signal) < 0.1:
                    print(f"⚠️ Advarsel: Kanal {channel_names[count]} (ch {channel}) kan være inaktiv.")

                # --- PLOT ---
                self.curves[count].setData(signal.tolist())

            except Exception as e:
                print(f"⚠️ Fejl i kanal {channel_names[count]}: {e}")

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"  # Opdater om nødvendigt

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
