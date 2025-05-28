import logging
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore

# Navne til dine EEG-kanaler
channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
alpha_channel_name = 'Fp1'  # ← Skift til f.eks. 'C4' eller 'O1' hvis ønsket

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4  # sekunder
        self.num_points = self.window_size * self.sampling_rate

        # Alpha tidsserie
        self.alpha_values = []
        self.max_alpha_points = 300  # ca. 5 min ved 1s opdatering
        self.alpha_curve = None

        # GUI setup
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow EEG Plot', size=(1000, 600), show=True)

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

        # Tilføj alpha power plot nederst
        p = self.win.addPlot(row=len(self.exg_channels), col=0)
        p.setLabel('left', 'Alpha Power (µV²/Hz)')
        p.setLabel('bottom', 'Tid (målinger)')
        p.setTitle('Live Alpha Power')
        self.alpha_curve = p.plot(pen='y')

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        alpha_power = None

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

                # Alpha power beregning for udvalgt kanal
                if count < len(channel_names) and channel_names[count] == alpha_channel_name:
                    if len(signal) >= 64:
                        nfft = min(256, DataFilter.get_nearest_power_of_two(len(signal)))
                        if nfft < len(signal):
                            psd = DataFilter.get_psd_welch(signal, nfft, nfft // 2,
                                                           self.sampling_rate, WindowOperations.HANNING.value)
                            alpha_power = DataFilter.get_band_power(psd, 8.0, 13.0)

            except Exception as e:
                print(f"Error processing channel {channel}: {e}")

        # GUI feedback og alpha graf
        if alpha_power is not None:
            self.win.setWindowTitle(
                f"BrainFlow EEG Plot – Alpha Power ({alpha_channel_name}): {alpha_power:.2f} µV²/Hz"
            )
            self.alpha_values.append(alpha_power)
            if len(self.alpha_values) > self.max_alpha_points:
                self.alpha_values.pop(0)
            if self.alpha_curve is not None:
                self.alpha_curve.setData(self.alpha_values)

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"  # ← Brug "cu.*" på Mac, ikke "tty.*"

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
