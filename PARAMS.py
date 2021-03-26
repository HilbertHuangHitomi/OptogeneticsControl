# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:19:37 2020

@author: Hilbert Huang Hitomi
"""


import os


hyperparameters = {

    'path':             os.path.dirname(os.path.abspath(__file__)),
    'subject':          'ICR20210222-8',

    'train_new_model':  False,
    'split':            [7,1],
    'duration':         2048,
    'batch_size':       64,
    'lr':               1e-3,
    'epochs':           16,
    'focal_gamma':      8,
    'gamma':            1e-1,
    'weight_decay':     1e-3,
    'working_device':   'cpu',

    'sampling_rate':    500,
    'record_days':      5,
    'waiting_times':    [0.30,0.40],
    'ema':              0.

    }


'''
'path':             '~/hilberthitomi05/Projects/OptogeneticsControl/',
'path':             'H:/Projects/OptogeneticsControl/',
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
    'sweep_int':        20         # seconds

    }