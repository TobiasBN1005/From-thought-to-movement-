import time
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations

# === KONFIGURATION ===
params = BrainFlowInputParams()
params.serial_port = '/dev/tty.usbserial-DM0258MO'  # Cyton dongle
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

teensy = serial.Serial('/dev/tty.usbmodem170452301', 9600)  # ← Tilpas denne port
time.sleep(2)  # Giv Teensy tid til at initialisere

try:
    BoardShim.enable_dev_board_logger()
    print("🧠 Logger aktiveret")

    board.prepare_session()
    print("✅ BrainFlow-session forberedt")

    board.start_stream()
    print("📶 EEG stream startet – venter på data...")

    while True:
        data = board.get_current_board_data(256)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        signal = data[eeg_channels[1]]

        if len(signal) < 10:
            print("⏳ Ikke nok data endnu")
            time.sleep(1)
            continue

        DataFilter.detrend(signal, DetrendOperations.LINEAR.value)
        sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

        psd = DataFilter.get_psd_welch(signal, nfft, nfft // 2, sampling_rate, WindowOperations.HANNING.value)
        alpha_power = DataFilter.get_band_power(psd, 7.0, 13.0)

        print(f"⚡ Alpha power: {alpha_power:.2f} µV²/Hz")

        if alpha_power > 20:
            teensy.write(b'1')
            print("📤 SENDT: b'1' → LED ON")
        else:
            teensy.write(b'0')
            print("📤 SENDT: b'0' → LED OFF")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("🛑 Stopper...")
finally:
    board.stop_stream()
    board.release_session()
    teensy.close()
    print("🔒 Lukket ned")
