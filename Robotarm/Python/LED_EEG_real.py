import time
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations

# Setup for BrainFlow
params = BrainFlowInputParams()
params.serial_port = '/dev/tty.usbserial-DM0258MO'  # ← Ret til din port
board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# Setup for Serial Communication to Arduino
arduino = serial.Serial('/dev/cu.usbmodem1401', 57600)  # ← Ret til din Arduino-port
time.sleep(2)  # Vent på at Arduino bliver klar

try:
    while True:
        data = board.get_current_board_data(256)  # Typisk 1 sek data
        eeg_channels = BoardShim.get_eeg_channels(board_id)

        # Vælg kun én EEG-kanal – fx Cz eller C3
        alpha_power = 0
        for ch in eeg_channels:
            DataFilter.detrend(data[ch], 0)
            psd = DataFilter.get_psd(data[ch], BoardShim.get_sampling_rate(board_id), WindowOperations.HAMMING.value)
            alpha_power += DataFilter.get_band_power(psd, 8.0, 13.0)

        alpha_power /= len(eeg_channels)  # Gennemsnitlig alpha power

        print(f"Alpha Power: {alpha_power:.2f} µV^2/Hz")

        if alpha_power > 20.0:
            arduino.write(b'1\n')
        else:
            arduino.write(b'0\n')

        time.sleep(1)  # Opdater hvert sekund

except KeyboardInterrupt:
    board.stop_stream()
    board.release_session()
    arduino.close()
