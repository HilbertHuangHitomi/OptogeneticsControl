# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:06:44 2020

@author: Hilbert Huang Hitomi
"""

import sys
sys.path.append('H:\\Projects\\OptogeneticsControl\\code\\')
sys.path.append('C:\\Users\\dell\\anaconda3\\OptogeneticsControl\\code\\')

import numpy as np
import pywt
import scipy
import matplotlib.pyplot as plt
import gc
import os
import pickle
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from pyentrp import entropy
from PARAMS import hyperparameters


#%%

# read data and sample
def DatasetGenerate(split=[7,1], ratio=[4,1]):

    sbj = hyperparameters['subject']
    path = hyperparameters['path'] + 'TrainData\\' + hyperparameters['subject'] + '\\'
    sample_size = hyperparameters['sample_size']
    duration = hyperparameters['duration']

    def ReadData():
        print('data for subject {} is loading'.format(sbj))
        normal_txt_files = os.listdir(path+'normal\\')
        seizure_txt_files = os.listdir(path+'seizure\\')
        raw_normal = np.zeros((1))
        raw_seizure = np.zeros((1))
        for filename in normal_txt_files :
            raw_normal = np.concatenate((raw_normal,np.loadtxt(path+'normal\\'+filename))).astype(np.float32)
        for filename in seizure_txt_files :
            raw_seizure = np.concatenate((raw_seizure,np.loadtxt(path+'seizure\\'+filename))).astype(np.float32)
        raw_normal = raw_normal[1:]
        raw_seizure = raw_seizure[1:]
        return raw_normal, raw_seizure

    def Sampling(raw_data, duration, sample_size):
        samples = np.zeros([sample_size, duration])
        index = list(np.random.randint(0, int(len(raw_data))-duration, sample_size))
        for i in tqdm(range(sample_size)):
            samples[i,:] = raw_data[index[i] : index[i]+duration]
        np.random.shuffle(samples)
        return samples

    def OverSampling(normal,seizure):
        over_sampler = SMOTE()
        resample_data, _ = over_sampler.fit_resample(
            np.concatenate((normal, seizure)),
            np.concatenate((np.array([0]*len(normal[:,0])), np.array([1]*len(seizure[:,0])))))
        normal = resample_data[:len(normal[:,0])]
        seizure = resample_data[len(normal[:,0]):]
        return normal, seizure

    # read
    raw_normal, raw_seizure = ReadData()

    normal = Sampling(raw_normal, duration, sample_size*ratio[0])
    seizure = Sampling(raw_seizure, duration, sample_size*ratio[1])
    del raw_normal, raw_seizure
    gc.collect()

    # over sampling
    normal, seizure = OverSampling(normal,seizure)

    # split
    index = int((split[0]/sum(split)) * (sample_size*ratio[0]))
    train_normal = normal[:index,:]
    test_normal = normal[index:,:]
    index = int((split[0]/sum(split)) * (sample_size*ratio[0]))
    train_seizure = seizure[:index,:]
    test_seizure = seizure[index:,:]
    del normal, seizure
    gc.collect()

    # construct
    train_data = np.concatenate((train_normal, train_seizure))
    train_label = np.concatenate((np.array([0]*len(train_normal[:,0])), np.array([1]*len(train_seizure[:,0]))))

    test_data = np.concatenate((test_normal, test_seizure))
    test_label = np.concatenate((np.array([0]*len(test_normal[:,0])), np.array([1]*len(test_seizure[:,0]))))

    del train_normal, train_seizure, test_normal, test_seizure
    gc.collect()

    print('------------------------------------------------------')
    print('data processing')

    return train_data, train_label, test_data, test_label


#%%

# feature process
def FeatureCalculate(trace):
    # feature : subband standard variance distribution & entropy
    def StandardVariance(dwt_data):
        V = np.zeros(len(dwt_data))
        for i in range(len(dwt_data)):
            V[i] = np.std(dwt_data[i])
        V = V / np.sum(V)
        V = np.concatenate((V,np.array(scipy.stats.entropy(pk=V, base=2))))
        return V
    # feature : subband maximum value
    def MaximumValue(dwt_data):
        V = np.zeros(len(dwt_data))
        for i in range(len(dwt_data)):
            V[i] = np.max(dwt_data[i])
        V = V / np.sum(V)
        V = np.concatenate((V,np.array(scipy.stats.entropy(pk=V, base=2))))
        return V
    # feature : subband relative wavelet energy distribution & entropy
    def RelativeEnergy(dwt_data):
        V = np.zeros(len(dwt_data))
        for i in range(len(dwt_data)):
            V[i] = np.sum(dwt_data[i]**2)
        V = V / np.sum(V)
        V = np.concatenate((V,np.array(scipy.stats.entropy(pk=V, base=2))))
        return V
    # feature : subband Shanon entropy distribution & entropy
    def ShanonEntropy(dwt_data):
        V = np.zeros(len(dwt_data))
        for i in range(len(dwt_data)):
            V[i] = entropy.shannon_entropy(dwt_data[i])
        V = V / np.sum(V)
        V = np.concatenate((V,np.array(scipy.stats.entropy(pk=V, base=2))))
        return V
    # feature : subband permutation entropy distribution & entropy
    def PermutationEntropy(dwt_data):
        V = np.zeros(len(dwt_data))
        for i in range(len(dwt_data)):
            V[i] = entropy.permutation_entropy(dwt_data[i],normalize=True)
        V = V / np.sum(V)
        V = np.concatenate((V,np.array(scipy.stats.entropy(pk=V, base=2))))
        return V
    # calculate feature
    dwt_data = pywt.wavedec(trace, 'db4', level=6)
    x = np.concatenate((StandardVariance(dwt_data),
                        MaximumValue(dwt_data),
                        RelativeEnergy(dwt_data),
                        ShanonEntropy(dwt_data),
                        PermutationEntropy(dwt_data)))
    return x


# process for training & predcition
def FeatureProcess(trace_data):
    # predict
    if len(trace_data.shape) == 1 :
        return FeatureCalculate(trace_data)
    # train & test
    if len(trace_data.shape) == 2 :
        return np.apply_along_axis(FeatureCalculate, 1, trace_data)


#%%

# predict probability
def PREDICT_PROBA(models, x, thres=False):
    weights = hyperparameters['weights']
    prediction = []
    for M in models:
        prediction_one_model = M.predict_proba(x)[:,1]
        prediction.append(prediction_one_model)
    prediction = np.array(prediction)
    prediction = ( weights[0]*prediction[0] + weights[1]*prediction[1] + weights[2]*prediction[2] ) / np.sum(weights)
    if thres:
        prediction[prediction > hyperparameters['threshold']] = 1
        prediction[prediction < hyperparameters['threshold']] = 0
    return prediction


# EMA smoothing
def EmaPredict(ema_proba, seizure_proba):
    e = hyperparameters['ema']
    ema_proba = ema_proba*e + seizure_proba*(1-e)
    return ema_proba


# queue for delay
def StartFlag(flag:list, ema_proba):
    ValueError(len(flag) != hyperparameters['waiting_times'])
    flag.pop(0)
    flag.append(ema_proba)
    state = 0
    if sum(np.array(flag) >= np.array(hyperparameters['waiting_times'])) >= len(hyperparameters['waiting_times']):
        state = 1
        flag = [0]*len(hyperparameters['waiting_times'])
    return flag, state


#%% train new model
def train_model():

    if hyperparameters['train_flag'] == True :

        sbj = hyperparameters['subject']

        train_data, train_label, test_data, test_label = DatasetGenerate()

        train_data_feature = FeatureProcess(train_data)
        test_data_feature = FeatureProcess(test_data)

        del train_data, test_data
        gc.collect()

        model_1 = DecisionTreeClassifier(min_samples_leaf = 128)
        model_2 = MLPClassifier(hidden_layer_sizes = (32),
                                verbose = True,
                                early_stopping = True,
                                n_iter_no_change = 10,
                                batch_size = 256,
                                learning_rate = 'adaptive',
                                alpha = 1e-2,
                                activation = 'relu')
        model_3 = MLPClassifier(hidden_layer_sizes = (16,8),
                                verbose = True,
                                early_stopping = True,
                                n_iter_no_change = 10,
                                batch_size = 256,
                                learning_rate = 'adaptive',
                                alpha = 1e-3,
                                activation = 'relu')

        print('------------------------------------------------------')
        print('start training model for subject {}'.format(sbj))

        model_1.fit(train_data_feature, train_label)
        model_2.fit(train_data_feature, train_label)
        model_3.fit(train_data_feature, train_label)

        models = [model_1, model_2, model_3]

        Evaluation(models, train_data_feature, train_label, test_data_feature, test_label)

        SaveModel(models)

        del train_data_feature, test_data_feature
        del model_1, model_2, model_3
        gc.collect()

        return models


#%% evaluate trained model

def Evaluation(models, train_data_feature, train_label, test_data_feature, test_label):

    T = PrettyTable()
    T.field_names = ["model #", "train ACC", "test ACC"]

    train_acc = '{:.2%}'.format(np.sum(models[0].predict(train_data_feature) == train_label)/len(train_label))
    test_acc = '{:.2%}'.format(np.sum(models[0].predict(test_data_feature) == test_label)/len(test_label))
    T.add_row(["model 1", train_acc, test_acc])

    train_acc = '{:.2%}'.format(np.sum(models[1].predict(train_data_feature) == train_label)/len(train_label))
    test_acc = '{:.2%}'.format(np.sum(models[1].predict(test_data_feature) == test_label)/len(test_label))
    T.add_row(["model 2", train_acc, test_acc])

    train_acc = '{:.2%}'.format(np.sum(models[2].predict(train_data_feature) == train_label)/len(train_label))
    test_acc = '{:.2%}'.format(np.sum(models[2].predict(test_data_feature) == test_label)/len(test_label))
    T.add_row(["model 3", train_acc, test_acc])

    train_acc = '{:.2%}'.format(np.sum(PREDICT_PROBA(models,train_data_feature, thres=True) == train_label)/len(train_label))
    test_acc = '{:.2%}'.format(np.sum(PREDICT_PROBA(models,test_data_feature, thres=True) == test_label)/len(test_label))
    T.add_row(["Resemble MODEL", train_acc, test_acc])

    print(T)


#%% moddel sl
def SaveModel(models):
    sbj = hyperparameters['subject']
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_1.pickle', 'wb') as f:
        pickle.dump(models[0],f)
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_2.pickle', 'wb') as f:
        pickle.dump(models[1],f)
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_3.pickle', 'wb') as f:
        pickle.dump(models[2],f)
    print('\n model for subject {} completed and saved'.format(sbj))
def LoadModel():
    sbj = hyperparameters['subject']
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_1.pickle', 'rb') as f:
        model_1 = pickle.load(f)
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_2.pickle', 'rb') as f:
        model_2 = pickle.load(f)
    with open (hyperparameters['path'] + 'model\\' + sbj + '\\' + 'model_3.pickle', 'rb') as f:
        model_3 = pickle.load(f)
    models = [model_1, model_2, model_3]
    print('\n model for subject {} successfully loaded'.format(sbj))
    return models


#%% test
def TestModel():

    # read test data
    def ReadData(filename):
        duration = hyperparameters['duration']
        # read
        D = np.loadtxt(hyperparameters['path']+'TrainData\\'+hyperparameters['subject']+'\\test\\'+filename).astype(np.float32)
        # reshape
        num_sample = len(D)//duration
        D = D[:num_sample*duration].reshape(num_sample,duration)
        # features
        D = FeatureProcess(D)
        return D

    # visulize
    def PlotPrediction(ema_prediction):
        plt.figure(1, figsize=(64, 2))
        plt.xlim(0,len(ema_prediction))
        plt.ylim(0,1)
        plt.yticks(np.arange(0,1,0.2))
        plt.xticks(np.arange(0,len(ema_prediction),100))
        plt.grid(True)
        plt.scatter(np.arange(0,len(ema_prediction),1),ema_prediction,s=5,color='red')
        plt.plot(ema_prediction,color='coral')
        plt.show()

    files = os.listdir(hyperparameters['path']+'TrainData\\' + hyperparameters['subject'] + '\\test\\')

    models = LoadModel()

    for filename in files:
        D = ReadData(filename)

        ema_proba = 0
        flag = [0]*len(hyperparameters['waiting_times'])
        state = 0

        ema_prediction = []
        states = []
        prediction = PREDICT_PROBA(models,D)

        for p in prediction :
            ema_proba = EmaPredict(ema_proba, p)
            ema_prediction.append(ema_proba)

        for e in ema_prediction :
            flag, state = StartFlag(flag, e)
            states.append(state)

        PlotPrediction(ema_prediction)
        print(filename + ' siezure points are {}'.format(np.where(np.array(states)==1)[0]))


#%%

if __name__== '__main__':

    #train_model()
    TestModel()




