# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:16:49 2020

@author: Hilbert Huang Hitomi
"""

import torch
import time
import WatchDog
import DataIO
import Model
import INPERcontrol
from PARAMS import hyperparameters


if __name__== '__main__':

    # watch dog
    hDir = WatchDog.WatchPath()

    # load model
    model = Model.LoadModel()
    model.to(torch.device('cpu'))

    # initialization
    ema_proba = 0
    runtime = 0
    runtime_max = int(6*60*60//(hyperparameters['duration']*0.002)+1)
    flag = [0]*len(hyperparameters['waiting_times'])
    state = 0
    inper_count = 0

    print('\n------------------------------------------------------')
    print('MODEL START')
    print('------------------------------------------------------\n')

    # call to read and predict txt data

    for quaters in range(hyperparameters['record_days']*4):
        runtime = 0
        floder = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        DataIO.CreateTXT(floder)

        while runtime < runtime_max :
            runtime += 1

            WatchDog.Watching(hDir)
            inputdata, seizure_proba = DataIO.QueryPredict(model)
            ema_proba = Model.EmaPredict(ema_proba, seizure_proba)
            print(time.strftime("%Y-%m-%d %H:%M:%S - ", time.localtime()) + '{:.2%}'.format(ema_proba) + '\n')

            DataIO.Report(ema_proba)
            DataIO.events(state)
            DataIO.SaveTrace(inputdata[0], floder)

            flag, state = Model.StartFlag(flag, ema_proba)
            inper_count = INPERcontrol.IFfire(state, inper_count)