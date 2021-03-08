# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:30:22 2020

@author: Hilbert Huang Hitomi
"""


import pandas as pd
import numpy as np
import time
import Model
from PARAMS import hyperparameters
import os


# read text data and predict
def QueryPredict(models:list):
    datapath='spike2data\\record\\ele_data.txt'
    inputdata = pd.read_table(hyperparameters['path']+datapath, header=0, encoding='ANSI')
    inputdata.columns = ['time','channel']
    inputdata.fillna(inputdata['channel'].mean(),inplace=True)
    inputdata = np.array(inputdata['channel'])[:hyperparameters['duration']]
    feature = Model.FeatureProcess(inputdata)
    seizure_proba = Model.PREDICT_PROBA(models, feature.reshape(1,-1))[0]
    return inputdata, seizure_proba


# create null txt
def CreateTXT(floder):
    path = hyperparameters['path']+'spike2data\\trace_data\\'
    if not os.path.exists(path) :
        os.mkdir(path)
    path = hyperparameters['path']+'spike2data\\trace_data\\'
    filename = floder + '.txt'
    if not os.path.exists(path) :
        os.mkdir(path)
    f = open(path+filename,'w')
    f.close()


# save data
def SaveTrace(inputdata, floder):
    path = hyperparameters['path']+'spike2data\\trace_data\\'
    filename = floder + '.txt'
    f = open(path+filename,'a+')
    for i in range(hyperparameters['duration']):
        f.writelines(str(np.around(inputdata[i],decimals=5))+'\n')
    for i in range(3):
        f.writelines(str(np.around(0,decimals=5))+'\n')
    f.close()


# manually combine traces
def ManualCombine(path):
    data_buff = []
    files = os.listdir(path)
    np.sort(files)
    for filename in files:
        inputdata = np.loadtxt(path+filename)
        data_buff.append(inputdata)
    trace_data = np.concatenate((data_buff))
    np.savetxt(path+'ManualOverall.txt', trace_data, '%10.5f')


# record report and events
def Report(ema_proba):
    path = hyperparameters['path']+'spike2data\\report\\'
    if not os.path.exists(path) :
        os.mkdir(path)
    text = time.strftime("%Y-%m-%d %H:%M:%S - ", time.localtime()) + '{:.2%}'.format(ema_proba) + '\n'
    filename = 'Report.txt'
    f = open(path+filename,'a+')
    f.writelines(text)
    f.close()
def events(state:int):
    path = hyperparameters['path']+'spike2data\\report\\'
    if not os.path.exists(path) :
        os.mkdir(path)
    if state == 1:
        text = time.strftime("%Y-%m-%d %H:%M:%S - ", time.localtime()) + 'LIGHT ON' + '\n'
        filename = 'EventsRecord.txt'
        f = open(path+filename,'a+')
        f.writelines(text)
        f.close()