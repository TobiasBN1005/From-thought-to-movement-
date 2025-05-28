#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:25:12 2025

@author: tobiasbendix
"""

import serial
import time

arduino = serial.Serial('/dev/tty.usbmodem1101', 9600)
time.sleep(2)  # Vent til Arduino er klar

for i in range(5):
    arduino.write(b'1')
    print("SEND: 1 (LED ON)")
    time.sleep(2)
    arduino.write(b'0')
    print("SEND: 0 (LED OFF)")
    time.sleep(2)

arduino.close()
