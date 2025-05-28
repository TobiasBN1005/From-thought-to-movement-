#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:56:51 2025

@author: tobiasbendix
"""

import serial
import time

# Opret forbindelse til Arduino
arduino = serial.Serial('/dev/cu.usbmodem1401', 57600)  # <-- Erstat med din Arduino-port
time.sleep(2)  # Vent på Arduino reset

# Dummy alpha power værdi fra f.eks. BrainFlow
alpha_power = 21  # μV²/Hz

# Send kommando baseret på alpha power
if alpha_power > 20:
    arduino.write(b'1\n')
else:
    arduino.write(b'0\n')
