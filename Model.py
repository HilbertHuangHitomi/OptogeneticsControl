# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:06:44 2020

@author: Hilbert Huang Hitomi
"""

import sys
sys.path.append('H:\\Projects\\OptogeneticsControl\\')
sys.path.append('C:\\Users\\dell\\anaconda3\\OptogeneticsControl\\')

import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import pywt
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from PARAMS import hyperparameters

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('H:/Projects/GrauralNetwork/tensorboard/')


#%% predict


# predict probability
def PREDICT_PROBA(model, inputdata, thres=False):
    with torch.no_grad():
        inputdata = Normalizing(inputdata)
        inputdata = torch.tensor(inputdata, dtype=torch.float32)
        inputdata = inputdata.reshape(-1,1,hyperparameters['duration'])
        inputdata = inputdata.to(torch.device(hyperparameters['working_device']))
        model.to(torch.device(hyperparameters['working_device']))
        prediction = model(inputdata)
        label = torch.argmax(prediction).item()
        prob = prediction[0,1]
        prob = prediction.detach().cpu().numpy()
    if thres :
        return label
    else :
        return prob


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


#%% read data


def ReadData():
    sbj = hyperparameters['subject']
    path = os.path.join(hyperparameters['path'], 'TrainData', hyperparameters['subject'])
    print('data for subject {} is loading'.format(sbj))
    normal_txt_files = os.listdir(os.path.join(path, 'normal'))
    seizure_txt_files = os.listdir(os.path.join(path, 'seizure'))
    raw_normal = np.zeros((1))
    raw_seizure = np.zeros((1))
    for filename in normal_txt_files :
        data = np.loadtxt(os.path.join(path, 'normal', filename))
        raw_normal = np.concatenate((raw_normal,data)).astype(np.float32)
    for filename in seizure_txt_files :
        data = np.loadtxt(os.path.join(path, 'seizure', filename))
        raw_seizure = np.concatenate((raw_seizure,data)).astype(np.float32)
    raw_normal = raw_normal[1:]
    raw_seizure = raw_seizure[1:]
    print('data loading complete')
    return raw_normal, raw_seizure


#%% process data


def Sampling(raw_data):
    duration = hyperparameters['duration']
    sample_size = len(raw_data)//duration
    data = raw_data[:sample_size*duration]
    data = data.reshape(sample_size,duration)
    return data


def OverSampling(normal,seizure):
    data = np.concatenate((normal, seizure))
    target = np.concatenate((np.array([0]*len(normal[:,0])), np.array([1]*len(seizure[:,0]))))
    dwt_data = pywt.wavedec(data, 'db4', level=7)
    over_sampler = SMOTE()
    resample_dwt_data =[]
    for subband in dwt_data :
        resample_subband, _ = over_sampler.fit_resample(subband, target)
        resample_dwt_data.append(resample_subband)
    resample_data = pywt.waverec(resample_dwt_data, 'db4')
    normal = resample_data[:len(normal[:,0])]
    seizure = resample_data[len(normal[:,0]):]
    del data, dwt_data, subband, resample_subband, target, resample_data
    gc.collect()
    return normal, seizure


def Splitting(data):
    split = hyperparameters['split']
    index = int((split[0]/sum(split)) * data.shape[0])
    train_data = data[:index,:]
    test_data = data[index:,:]
    return train_data, test_data


def Normalizing(data):
    data = preprocessing.StandardScaler().fit_transform(data)
    return data


def Concating(train_normal,test_normal,train_seizure,test_seizure):
    train_data = np.concatenate((train_normal, train_seizure))
    train_label = np.concatenate((np.array([0]*len(train_normal[:,0])), np.array([1]*len(train_seizure[:,0]))))
    test_data = np.concatenate((test_normal, test_seizure))
    test_label = np.concatenate((np.array([0]*len(test_normal[:,0])), np.array([1]*len(test_seizure[:,0]))))
    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.int64)
    test_data = test_data.astype(np.float32)
    test_label = test_label.astype(np.int64)
    return train_data, train_label, test_data, test_label


# process main
def ProcessData(raw_normal, raw_seizure):

    normal = Sampling(raw_normal)
    seizure = Sampling(raw_seizure)

    normal,seizure = OverSampling(normal,seizure)
    #seizure = np.tile(seizure, (int(normal.shape[0]/seizure.shape[0]),1))

    train_normal, test_normal = Splitting(normal)
    train_seizure, test_seizure = Splitting(seizure)

    del normal,seizure
    gc.collect()

    train_normal = Normalizing(train_normal)
    test_normal = Normalizing(test_normal)
    train_seizure = Normalizing(train_seizure)
    test_seizure = Normalizing(test_seizure)

    train_data, train_label, test_data, test_label = Concating(train_normal,test_normal,train_seizure,test_seizure)

    del train_normal,test_normal,train_seizure,test_seizure
    gc.collect()

    return train_data, train_label, test_data, test_label


# rerange as pytorch datasets
class SRSdataset(Dataset):
    def __init__(self,
                 data, label,
                 transform=None, target_transform=None):
        self.data = data
        self.labels = label
        self.data = self.data.reshape(-1, 1, hyperparameters['duration'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trace = self.data[idx,:,:]
        label = self.labels[idx]
        if self.transform:
            trace = self.transform(trace)
        if self.target_transform:
            label = self.target_transform(label)
        return trace, label


#%% SRS classification model


# 1D Inception block
class Inception(nn.Module):

    def __init__(self, in_channels):
        super(Inception, self).__init__()

        if in_channels < 4 :
            raise ValueError('Too less channels for the Inception block')
        else:
            C = int(in_channels/4)
            self.branch1_0 = nn.Conv1d(in_channels, C, kernel_size=1)

            self.branch2_0 = nn.Conv1d(in_channels, C, kernel_size=1)
            self.branch2_1 = nn.Conv1d(C, C, kernel_size=3, padding=1)
            self.branch2_2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
            self.branch2_3 = nn.Conv1d(C, C, kernel_size=3, padding=1)
            self.branch2_4 = nn.Conv1d(C, C, kernel_size=3, padding=1)

            self.branch3_0 = nn.Conv1d(in_channels, C, kernel_size=1)
            self.branch3_1 = nn.Conv1d(C, C, kernel_size=5, padding=2)
            self.branch3_2 = nn.Conv1d(C, C, kernel_size=5, padding=2)

            self.branch4_0 = nn.Conv1d(in_channels, C, kernel_size=1)
            self.branch4_1 = nn.Conv1d(C, C, kernel_size=7, padding=3)

            self.Norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):

        branch1 = self.branch1_0(x)

        branch2 = self.branch2_0(x)
        branch2 = self.branch2_1(branch2)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)
        branch2 = self.branch2_4(branch2)

        branch3 = self.branch3_0(x)
        branch3 = self.branch3_1(branch3)
        branch3 = self.branch3_2(branch3)

        branch4 = self.branch4_0(x)
        branch4 = self.branch4_1(branch4)

        concat = torch.cat([branch1, branch2, branch3, branch4], 1)

        concat = self.Norm(concat)
        concat = F.elu(concat)

        outputs = x + concat

        return outputs


# main model
class SRSmodel(nn.Module):

    def __init__(self,
                 in_channels = 32,
                 ):
        super(SRSmodel, self).__init__()

        self.conv_in = nn.Conv1d(1, in_channels, kernel_size=1)

        self.inception1 = Inception(in_channels=in_channels)
        self.inception2 = Inception(in_channels=in_channels)
        self.inception3 = Inception(in_channels=in_channels)
        self.inception4 = Inception(in_channels=in_channels)

        self.conv_out = nn.Conv1d(in_channels, 1, kernel_size=1)

        self.fc = nn.Linear(int(hyperparameters['duration']),2)

    def forward(self, x):

        net = self.conv_in(x) # [batch_size, in_channels, duration]
        net = self.inception1(net) # [batch_size, in_channels, duration]
        net = self.inception2(net) # [batch_size, in_channels, duration]
        net = self.inception3(net) # [batch_size, in_channels, duration]
        net = self.inception4(net) # [batch_size, in_channels, duration]
        net = self.conv_out(net) # [batch_size, 1, duration]

        net = torch.flatten(net, 1) # [batch_size, duration]
        net = self.fc(net) # [batch_size, 2]
        net = F.softmax(net, dim=1)

        return net


# loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=hyperparameters['focal_gamma'], alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.log(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.sum()

LossFunction = FocalLoss()


#%% training process


# model summary
def summary(model):
    modules = []
    params = []
    for m in model.named_children():
        modules.append(m[0])
    for m in model.children():
        params.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
    T = PrettyTable()
    T.field_names = ["Total Parameters", "{:,}".format(sum(params))]
    for i in range(len(modules)):
        T.add_row([modules[i], '{:,}'.format(params[i])])
    print(T)


# train model
def train(model, optimizer, train_loader, device, epoch):
    with tqdm(train_loader) as T:
        model.train()

        for i, (data, target) in enumerate(T):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = LossFunction(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            TrainACC = pred.eq(target.view_as(pred)).sum().item()/len(target)
            T.set_postfix({
                'epoch'    : epoch,
                'loss'     : '{0:1.5f}'.format(loss),
                'TrainACC' : '{:.2%}'.format(TrainACC),
                })


# test model
def test(model, test_loader, device):
    AUC_output = []
    AUC_target = []
    correct = 0

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            AUC_output.append(output.exp()[:,1].cpu().numpy())
            AUC_target.append(target.cpu().numpy())

    AUC_output = np.concatenate(AUC_output)
    AUC_target = np.concatenate(AUC_target)
    TestAUC = roc_auc_score(AUC_target, AUC_output)

    TestACC = correct/len(test_loader.dataset)

    with tqdm(range(1)) as T:
        T.set_postfix({
           'TestAUC' : '{:.2%}'.format(TestAUC),
           'TestACC' : '{:.2%}'.format(TestACC),
           })

    return TestAUC, TestACC


# configure model and start training process
def run(model, device, train_loader, test_loader):

    model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr = hyperparameters['lr'],
                            weight_decay = hyperparameters['weight_decay'])

    scheduler = StepLR(optimizer,
                       step_size = 1,
                       gamma = hyperparameters['gamma'])

    for epoch in range(hyperparameters['epochs']):
        train(model, optimizer, train_loader, device, epoch)
        test(model, test_loader, device)
        scheduler.step()
    return model


#%% moddel sl


def SaveModel(model):
    sbj = hyperparameters['subject']
    path = os.path.join(hyperparameters['path'], 'model', sbj, 'model.weights')
    model.to(torch.device(hyperparameters['working_device']))
    torch.save(model.state_dict(), path)
    print('\n model for subject {} completed and saved'.format(sbj))


def LoadModel():
    sbj = hyperparameters['subject']
    path = os.path.join(hyperparameters['path'], 'model', sbj, 'model.weights')
    model = SRSmodel()
    model.load_state_dict(torch.load(path, map_location=torch.device(hyperparameters['working_device'])))
    model.eval()
    print('\n model on {} for subject {} successfully loaded'.format(hyperparameters['working_device'],sbj))
    return model


#%% test


def TestModel():

    def Read(filename):
        duration = hyperparameters['duration']
        path = os.path.join(hyperparameters['path'], 'TrainData', hyperparameters['subject'], 'test', filename)
        inputdata = np.loadtxt(path).astype(np.float32)
        sample_size = len(inputdata)//duration
        inputdata = inputdata[:sample_size*duration]
        inputdata = inputdata.reshape(sample_size,duration)
        return inputdata

    def predict(model,inputdata):
        with torch.no_grad():
            inputdata = Normalizing(inputdata)
            inputdata = torch.tensor(inputdata, dtype=torch.float32)
            inputdata = inputdata.reshape(-1,1,hyperparameters['duration'])
            inputdata = inputdata.to(torch.device("cpu"))
            prediction = model(inputdata)
            prediction = prediction[:,1]
            prediction = prediction.detach().cpu().numpy()
        return prediction

    def Plot(ema_prediction):
        plt.figure(1, figsize=(64, 2))
        plt.xlim(0,len(ema_prediction))
        plt.ylim(0,1)
        plt.yticks(np.arange(0,1,0.2))
        plt.xticks(np.arange(0,len(ema_prediction),100))
        plt.grid(True)
        plt.scatter(np.arange(0,len(ema_prediction),1),ema_prediction,s=5,color='red')
        plt.plot(ema_prediction,color='coral')
        plt.show()

    files = os.listdir(os.path.join(hyperparameters['path'], 'TrainData', hyperparameters['subject'], 'test'))
    files = sort(files)

    model = LoadModel()
    model.to(torch.device("cpu"))

    for filename in files:

        ema_proba = 0
        flag = [0]*len(hyperparameters['waiting_times'])
        state = 0
        ema_prediction = []
        states = []

        inputdata = Read(filename)
        prediction = predict(model,inputdata)

        for pred in prediction :
            ema_proba = EmaPredict(ema_proba, pred)
            ema_prediction.append(ema_proba)

        for e in ema_prediction :
            flag, state = StartFlag(flag, e)
            states.append(state)

        Plot(ema_prediction)
        print(filename + ' siezure points are {}'.format(np.where(np.array(states)==1)[0]))

        del inputdata, prediction
        gc.collect()


#%% train new model


def main():

    if hyperparameters['train_new_model']:

        raw_normal, raw_seizure = ReadData()

        train_data, train_label, test_data, test_label = ProcessData(raw_normal, raw_seizure)

        train_torch_dataset = SRSdataset(train_data, train_label)
        test_torch_dataset = SRSdataset(test_data, test_label)

        train_dataloader = DataLoader(train_torch_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_torch_dataset, batch_size=int(hyperparameters['batch_size']*4), shuffle=False)

        model = SRSmodel()
        summary(model)

        model = run(model, torch.device("cuda"), train_dataloader, test_dataloader)

        SaveModel(model)


#%% test traces


if __name__== '__main__':

    main()

    TestModel()