#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:56:24 2025
@author: tobiasbendix
"""

import logging
import time
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pyqtgraph.Qt import QtWidgets, QtCore


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600), show=True)
        self.win.raise_()
        self.win.activateWindow()

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
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            try:
                # === Debug print af rå kanaldata ===
                print(f"Kanal {channel}, sidste 5 samples:", data[channel][-5:])

                # === Midlertidigt UDEN filter for at tjekke visning ===
                self.curves[count].setData(data[channel].tolist())

                # === Hvis du vil slå filter til igen, brug disse i stedet ===
                # DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                #                             FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                #                             FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                #                             FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0)
                # self.curves[count].setData(data[channel].tolist())
            except Exception as e:
                print(f"Error in channel {channel}: {e}")

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    # === DINE PARAMETRE HER ===
    board_id = BoardIds.CYTON_BOARD  # = 0
    serial_port = "/dev/tty.usbserial-DM0258MO"

    params = BrainFlowInputParams()
    params.serial_port = serial_port

    board_shim = BoardShim(board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, "")
        print("Stream started – venter 3 sekunder på buffer ...")
        time.sleep(3)

        # === Debug: Hvor mange samples er tilgængelige? ===
        count = board_shim.get_board_data_count()
        print(f"Samples i buffer: {count}")

        if count == 0:
            print("⚠️  Ingen data modtages – tjek at boardet er tændt og korrekt tilsluttet!")

        Graph(board_shim)

    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
