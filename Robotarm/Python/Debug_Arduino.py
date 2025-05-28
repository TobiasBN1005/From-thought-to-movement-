import time
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations

# === KONFIGURATION ===
params = BrainFlowInputParams()
params.serial_port = '/dev/tty.usbserial-DM0258MO'  # Cyton dongle
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

print("🔌 Åbner forbindelse til Arduino...")
arduino = serial.Serial('/dev/tty.usbmodem1101', 9600)
time.sleep(2)  # Giv Arduino tid til at genstarte
print("✅ Arduino tilsluttet")

try:
    BoardShim.enable_dev_board_logger()
    print("🧠 Logger aktiveret")

    board.prepare_session()
    print("✅ BrainFlow-session forberedt")

    board.start_stream()
    print("📶 Stream startet – afventer EEG-data...")

    while True:
        data = board.get_current_board_data(256)  # Ca. 1 sekunds data
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        channel = eeg_channels[1]  # Fx C4

        signal = data[channel]
        print(f"📊 Signal længde: {len(signal)}")
        if len(signal) < 10:
            print("⏳ For lidt data – venter...")
            time.sleep(1)
            continue

        try:
            DataFilter.detrend(signal, DetrendOperations.LINEAR.value)
            sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
            nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
            psd = DataFilter.get_psd_welch(signal, nfft, nfft // 2, sampling_rate, WindowOperations.HANNING.value)

            print(f"🔬 Første 5 PSD-værdier: {psd[1][:5]}")

            alpha_power = DataFilter.get_band_power(psd, 7.0, 13.0)
            print(f"⚡ Alpha power: {alpha_power:.2f} µV²/Hz")

            if alpha_power > 20:
                arduino.write(b'1')
                print("📤 SENDT: b'1' → LED ON")
            else:
                arduino.write(b'0')
                print("📤 SENDT: b'0' → LED OFF")

        except Exception as e:
            print("❌ FEJL UNDER BEREGNING:", e)
            continue

        time.sleep(0.5)

except KeyboardInterrupt:
    print("🛑 Afslutter...")

finally:
    board.stop_stream()
    board.release_session()
    arduino.close()
    print("🔒 Forbindelser lukket")
