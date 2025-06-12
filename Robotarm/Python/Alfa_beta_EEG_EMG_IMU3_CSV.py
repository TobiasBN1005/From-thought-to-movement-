import sys, time, csv, os, serial, threading
from collections import deque
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations

# === Konfiguration ===
channel_names = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2']
alpha_channel_name = 'Fp1'
band_definitions = {
    'Delta': (0.5, 4.0, 'g'), 'Theta': (4.0, 8.0, 'b'),
    'Alpha': (8.0, 13.0, 'r'), 'Beta': (13.0, 30.0, 'y')
}
emg_buffer = deque(maxlen=500)
imu_buffer = {f'{axis}{i}': deque(maxlen=500) for i in range(1, 4) for axis in ['ax','ay','az','gx','gy','gz']}
lock = threading.Lock()

# === CSV setup ===
log_dir = "alfa_eeg_emg_imu_logs"
os.makedirs(log_dir, exist_ok=True)
filename = time.strftime("alfa_eeg_emg_imu_log_%Y%m%d-%H%M%S.csv")
csv_path = os.path.join(log_dir, filename)
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['timestamp', 'session_time'] + channel_names + ['EMG'] + \
             [f'IMU{i}_{axis}' for i in range(1, 4) for axis in ['ax','ay','az','gx','gy','gz']]
    writer.writerow(header)

# === Teensy Reader i baggrundstråd ===
class TeensyReader(threading.Thread):
    def __init__(self, port='/dev/cu.usbmodem170452301'):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, 115200)
        time.sleep(2)

    def run(self):
        while True:
            try:
                line = self.ser.readline().decode().strip()
                with lock:
                    if line.startswith("EMG:"):
                        emg_buffer.append(int(line.split(":")[1]))
                    for i in range(1, 4):
                        if line.startswith(f"IMU{i}:"):
                            parts = line[len(f"IMU{i}:"):].split(",")
                            if len(parts) == 6:
                                for j, axis in enumerate(['ax','ay','az','gx','gy','gz']):
                                    key = f"{axis}{i}"
                                    imu_buffer[key].append(float(parts[j]))
            except:
                continue

# === EEG GUI + logging ===
class EEGWindow(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_shim = board_shim
        self.setWindowTitle("EEG GUI")
        self.widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.widget)
        self.curves = []
        self.band_curves = {}
        self.band_vals = {b: [] for b in band_definitions}
        self.band_timestamps = []
        self.sampling_rate = BoardShim.get_sampling_rate(board_shim.get_board_id())
        self.session_start = time.time()
        self.num_points = 4 * self.sampling_rate

        for i, ch in enumerate(BoardShim.get_exg_channels(board_shim.get_board_id())):
            p = self.widget.addPlot(row=i, col=0)
            p.setLabel('left', channel_names[i])
            c = p.plot()
            self.curves.append(c)

        band_plot = self.widget.addPlot(row=len(self.curves), col=0)
        band_plot.setTitle("Band Power")
        band_plot.setYRange(0, 50)
        legend = pg.LegendItem(offset=(-20, 10))
        legend.setParentItem(band_plot.graphicsItem())
        for band, (_, _, color) in band_definitions.items():
            curve = band_plot.plot(pen=color)
            self.band_curves[band] = curve
            legend.addItem(curve, band)

    def update(self):
        now = time.time()
        session_time = now - self.session_start
        data = self.board_shim.get_current_board_data(self.num_points)
        eeg_vals = [data[ch][-1] * 0.02235 for ch in BoardShim.get_exg_channels(self.board_shim.get_board_id())]

        with lock:
            emg = emg_buffer[-1] if emg_buffer else 0
            imu = [imu_buffer[k][-1] if imu_buffer[k] else 0 for k in sorted(imu_buffer)]
            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow([now, round(session_time, 3)] + eeg_vals + [emg] + imu)

        for i, ch in enumerate(BoardShim.get_exg_channels(self.board_shim.get_board_id())):
            signal = data[ch]
            DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
            self.curves[i].setData(signal.tolist())

        for count, ch in enumerate(BoardShim.get_exg_channels(self.board_shim.get_board_id())):
            if channel_names[count] == alpha_channel_name:
                signal = data[ch]
                if len(signal) >= 256:
                    psd = DataFilter.get_psd_welch(signal, 256, 128,
                                                   self.sampling_rate, WindowOperations.HANNING.value)
                    for band, (low, high, _) in band_definitions.items():
                        power = DataFilter.get_band_power(psd, low, high)
                        self.band_vals[band].append(power)
        self.band_timestamps.append(now)
        if len(self.band_timestamps) > self.sampling_rate * 5:
            self.band_timestamps.pop(0)
        for band in self.band_vals:
            if len(self.band_vals[band]) > self.sampling_rate * 5:
                self.band_vals[band].pop(0)
        if self.band_timestamps:
            x = [t - self.band_timestamps[0] for t in self.band_timestamps]
            for band in self.band_vals:
                if len(self.band_vals[band]) == len(x):
                    self.band_curves[band].setData(x=x, y=self.band_vals[band])

# === EMG GUI ===
class EMGWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG GUI")
        self.widget = pg.PlotWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLabel('left', 'EMG')
        self.curve = self.widget.plot(pen='m')

    def update(self):
        with lock:
            self.curve.setData(list(emg_buffer))

# === IMU GUI ===
class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU1–3 GUI")
        self.widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.widget)
        self.acc_curves = {}
        self.gyro_curves = {}

        for i in range(1, 4):
            acc_plot = self.widget.addPlot(row=(i-1)*2, col=0)
            acc_plot.setTitle(f"IMU{i} Acceleration")
            acc_plot.enableAutoRange(axis='y', enable=True)

            gyro_plot = self.widget.addPlot(row=(i-1)*2+1, col=0)
            gyro_plot.setTitle(f"IMU{i} Gyroscope")
            gyro_plot.enableAutoRange(axis='y', enable=True)

            self.acc_curves[f'IMU{i}'] = {
                a: acc_plot.plot(pen=p) for a, p in zip(['ax','ay','az'], ['c','y','g'])
            }
            self.gyro_curves[f'IMU{i}'] = {
                a: gyro_plot.plot(pen=p) for a, p in zip(['gx','gy','gz'], ['c','y','g'])
            }

            if i == 1:
                acc_legend = pg.LegendItem(offset=(10, 10))
                acc_legend.setParentItem(acc_plot.graphicsItem())
                acc_legend.addItem(self.acc_curves[f'IMU1']['ax'], 'X (cyan)')
                acc_legend.addItem(self.acc_curves[f'IMU1']['ay'], 'Y (yellow)')
                acc_legend.addItem(self.acc_curves[f'IMU1']['az'], 'Z (green)')

                gyro_legend = pg.LegendItem(offset=(10, 10))
                gyro_legend.setParentItem(gyro_plot.graphicsItem())
                gyro_legend.addItem(self.gyro_curves[f'IMU1']['gx'], 'X (cyan)')
                gyro_legend.addItem(self.gyro_curves[f'IMU1']['gy'], 'Y (yellow)')
                gyro_legend.addItem(self.gyro_curves[f'IMU1']['gz'], 'Z (green)')

    def update(self):
        with lock:
            for i in range(1, 4):
                for a in ['ax','ay','az']:
                    self.acc_curves[f'IMU{i}'][a].setData(list(imu_buffer.get(f'{a}{i}', [])))
                for a in ['gx','gy','gz']:
                    self.gyro_curves[f'IMU{i}'][a].setData(list(imu_buffer.get(f'{a}{i}', [])))

# === Main ===
def main():
    import logging
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM0258MO"
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    teensy = TeensyReader()
    teensy.start()

    app = QtWidgets.QApplication(sys.argv)
    eeg = EEGWindow(board); eeg.show()
    emg = EMGWindow(); emg.show()
    imu = IMUWindow(); imu.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(eeg.update)
    timer.timeout.connect(emg.update)
    timer.timeout.connect(imu.update)
    timer.start(50)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
