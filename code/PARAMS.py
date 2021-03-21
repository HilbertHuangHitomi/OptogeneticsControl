# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:19:37 2020

@author: Hilbert Huang Hitomi
"""

hyperparameters = {
    'path':             'C:\\Users\\dell\\anaconda3\\OptogeneticsControl\\',
    'subject':          '20210222-8',
    'train_flag':       False,
    'sample_size':      2048,
    'duration':         2048,
    'sampling_rate':    500,
    'record_days':      5,
    'threshold':        0.5,
    'waiting_times':    [0.25,0.50,0.60,0.70],
    'ema':              0.5,
    'weights':          [1,1,1]}

'''
'path':             'H:\\Projects\\OptogeneticsControl\\',
'path':             'C:\\Users\\dell\\anaconda3\\OptogeneticsControl\\',
'''

INPERparameters = {
    'firing_duration':  4,          # terms
    'channel':          '1',        # channel
    'd_cycle':          0.05,       # duty_cycle, ratio
    'frequency':        0,          # hertz
    'power':            50,         # percent
    'start_delay':      0,          # seconds
    'burst_len':        50,         # seconds
    'burst_int':        10,         # seconds
    'sweep_len':        2000,       # seconds
    'sweep_int':        20}         # seconds