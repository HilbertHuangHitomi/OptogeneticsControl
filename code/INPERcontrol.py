# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:06:44 2020

@author: Hilbert Huang Hitomi
"""


import serial
import time
import simplejson
import struct
from serial.tools import list_ports
from PARAMS import INPERparameters


# crc
def crc16(data: bytes) -> bytes:
    data_len = len(data)
    if data_len <= 0:
        return bytes([0, 0])
    crc = 0xFFFF
    for byte in data:
        crc = crc ^ byte
        for _ in range(8):
            if crc % 2 == 0:
                crc = crc >> 1
            else:
                crc = (crc >> 1) ^ 0xA001
    return crc.to_bytes(length=2, byteorder='big', signed=False)


# commands
class as_cmd:
    def report() -> bytes:
        header = bytes('NRT'.ljust(72, 'F'), 'ASCII')
        crc = crc16(header)
        frame = header + crc
        return frame
    def set_wf(
            cha: str,
            freq_hz: float,
            duty_cycle: float,
            power_percent: float,
            start_delay_s: float,
            burst_duration_s: float,
            burst_interval_s: float,
            sweep_len_s: float,
            sweep_interval_s: float,
            sweep_count: int):
        header = bytes('NST' + cha, 'ASCII')
        freq_bytes = struct.pack('d', freq_hz)
        duty_cycle_bytes = struct.pack('d', duty_cycle)
        power_bytes = struct.pack('d', power_percent)
        start_delay_bytes = struct.pack('d', start_delay_s)
        burst_duration_bytes = struct.pack('d', burst_duration_s)
        burst_interval_bytes = struct.pack('d', burst_interval_s)
        sweep_length_bytes = struct.pack('d', sweep_len_s)
        sweep_interval_bytes = struct.pack('d', sweep_interval_s)
        sweep_count_bytes = struct.pack('i', sweep_count)
        frame = header + freq_bytes + duty_cycle_bytes + power_bytes + start_delay_bytes + \
            burst_duration_bytes + burst_interval_bytes + sweep_length_bytes + sweep_interval_bytes + sweep_count_bytes
        crc = crc16(frame)
        frame += crc
        return frame
    def start(channel: str) -> bytes:
        header = bytes('NIN' + channel, 'ASCII')
        fill = bytes([0xFF] * (72 - len(header)))
        frame = header + fill
        crc = crc16(frame)
        frame += crc
        return frame
    def stop(channel: str) -> bytes:
        header = bytes('NSU' + channel, 'ASCII')
        fill = bytes([0xFF] * (72 - len(header)))
        frame = header + fill
        crc = crc16(frame)
        frame += crc
        return frame


# search device

def find_inper_device(sp_ls=list(list_ports.comports())):
    inper_devs = []
    cmd_rpt = as_cmd.report()
    for p in sp_ls:
        port = serial.Serial(port=p[0],
                             baudrate=9600,
                             bytesize=serial.EIGHTBITS,
                             parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE,
                             timeout=5)
        port.flushInput()
        port.write(cmd_rpt)
        recv = ''
        time.sleep(0.1)
        while True:
            wait_len = port.inWaiting()
            if wait_len == 0:
                break
            recv += port.read().decode('ASCII')

        if len(recv) > 0 and recv[0] == '{':
            try:
                json_recv = simplejson.loads(recv)
            except simplejson.errors.JSONDecodeError:
                port.close()
            else:
                if json_recv['system']['name'] == 'NWI/EMB':
                    inper_devs.append(port)
        else:
            port.close()
    if len(inper_devs) >0 :
        return inper_devs[0]
    else:
        ValueError('device not found')


# control
class INPERdeviceControl:

    inper_dev = []
    set_wf_cmd = []
    start_cmd = []
    stop_cmd = []

    def __init__(self):
        self.channel = INPERparameters['channel']
        self.d_cycle = INPERparameters['d_cycle']
        self.frequency = INPERparameters['frequency']
        self.power = INPERparameters['power']
        self.start_delay = INPERparameters['start_delay']
        self.burst_len = INPERparameters['burst_len']
        self.burst_int = INPERparameters['burst_int']
        self.sweep_len = INPERparameters['sweep_len']
        self.sweep_int = INPERparameters['sweep_int']

    def InitiateDevice(self):
        self.inper_dev = find_inper_device()

    def InitiateCommand(self):
        self.set_wf_cmd = as_cmd.set_wf(cha=self.channel,
                                       freq_hz = self.frequency,
                                       duty_cycle = self.d_cycle,
                                       power_percent = self.power,
                                       start_delay_s = self.start_delay,
                                       burst_duration_s = self.burst_len,
                                       burst_interval_s = self.burst_int,
                                       sweep_len_s = self.sweep_len,
                                       sweep_interval_s = self.sweep_int,
                                       sweep_count = 10)
        self.start_cmd = as_cmd.start('11')
        self.stop_cmd = as_cmd.stop(self.channel)

    def SendCommand(self, command:str):
        if command == 'setting' :
            self.inper_dev.write(self.set_wf_cmd)
        if command == 'start' :
            self.inper_dev.write(self.start_cmd)
        if command == 'stop' :
            self.inper_dev.write(self.stop_cmd)

    def CloseDevice(self):
        self.inper_dev.close()


'''
def RunDevice(INPERparameters):
    I = INPERdeviceControl(INPERparameters)
    I.InitiateDevice()
    I.InitiateCommand()
    I.SendCommand('stop')
    time.sleep(0.05)
    I.SendCommand('setting')
    time.sleep(0.05)
    I.SendCommand('start')
    time.sleep(20)
    I.SendCommand('stop')
    time.sleep(0.5)
    I.CloseDevice()
'''


# firing judgement
def IFfire(state, inper_count):

    def ENGAGE():
        I = INPERdeviceControl()
        I.InitiateDevice()
        I.InitiateCommand()
        I.SendCommand('stop')
        time.sleep(0.05)
        I.SendCommand('setting')
        time.sleep(0.05)
        I.SendCommand('start')
        time.sleep(0.05)
        I.CloseDevice()

    def HOLD():
        I = INPERdeviceControl()
        I.InitiateDevice()
        I.InitiateCommand()
        I.SendCommand('stop')
        time.sleep(0.05)
        I.SendCommand('setting')
        time.sleep(0.05)
        I.SendCommand('start')
        time.sleep(0.05)
        I.SendCommand('stop')
        time.sleep(0.05)
        I.CloseDevice()

    # start
    if state == 1 :
        inper_count = 0
        ENGAGE()
        inper_count = 1
        return inper_count
    else:
        # normal
        if inper_count == 0 :
            return inper_count
        # ongoing
        if inper_count > 0 :
            # hold
            if inper_count >= INPERparameters['firing_duration'] :
                inper_count = 0
                HOLD()
                return inper_count
            # continue
            else:
                inper_count += 1
                return inper_count
        return inper_count
