import numpy as np
import scipy.io as scio

import pandas as pd
import torch
import os
from scipy.io import loadmat
from sklearn import preprocessing


def sliding_windows(data,sw_width,covariates_idx,y_idx,pred_len=1):
    datalen = data.shape[0]
    covariates_num, y_num = len(covariates_idx), len(y_idx)
    samples_num = datalen-sw_width-pred_len

    X = np.zeros((samples_num,sw_width,covariates_num+y_num))
    y = np.zeros((samples_num, pred_len, y_num))

    for i in range(samples_num):
        start, end = i, i + sw_width
        X[i, :, :covariates_num] = data[start:end, covariates_idx]
        X[i, :, covariates_num:covariates_num+y_num] = data[start:end, y_idx]
        y[i, :, :] = data[end:end + pred_len, y_idx]
    return (X,y)

def sliding_window_non_overlap(data, sw_width,covariates_idx,y_idx):
    datalen = data.shape[0]
    covariates_num, y_num = len(covariates_idx), len(y_idx)
    samples_num = datalen // sw_width
    X = np.zeros((samples_num, sw_width, covariates_num))
    y = np.zeros((samples_num, 1, y_num))
    for i in range(samples_num):
        start, end = i*sw_width, (i+1)*sw_width
        X[i, :, :covariates_num] = data[start:end, covariates_idx]
        y[i, :, :] = data[end-1:end, y_idx]
    return X,y

if __name__ == '__main__':
    Path = 'TE'
    data = loadmat(f'{Path}/TEP.mat')
    output_dir = '../datasets/TEP'
    control = np.array(data['xmv'])
    measure = np.array(data['simout'])
    opCost = np.array(data['OpCost'])
    control_del = np.delete(control, 0, 0)[::5, :]
    measure_del = np.delete(measure, 0, 0)[::5, :]
    opCost_del = np.delete(opCost, 0, 0)[::5, :]
    all_data = np.hstack((measure_del, control_del))

    scaler = preprocessing.StandardScaler().fit(all_data)  # scaler保存方差和均值
    scorelabelData = preprocessing.scale(all_data)

    #choose suitable variables
    all_idx = np.arange(41)
    y_idx = np.array([29, 30, 38]) - 1
    del_idx = np.array([23,24,25,26,27,28,31,32,33,34,35,36,37,39,40,41])-1

    covariates_idx = np.setdiff1d(all_idx, del_idx)
    covariates_idx = np.setdiff1d(covariates_idx, y_idx)


    initTrainAllDataSet = scorelabelData[:10000, :]
    initTrainDataSet = scorelabelData[:8000, :]
    initTestDataSet = scorelabelData[10000:12000, :]

    train_data_all,train_labels_all = sliding_window_non_overlap(initTrainAllDataSet,24,covariates_idx,y_idx)
    train_data,train_labels = sliding_window_non_overlap(initTrainDataSet,24,covariates_idx,y_idx)
    test_data,test_labels = sliding_window_non_overlap(initTestDataSet,24,covariates_idx,y_idx)

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(train_data_all).transpose(1,2)
    dat_dict["labels"] = torch.from_numpy(train_labels_all).transpose(1,2)
    torch.save(dat_dict, os.path.join(output_dir, "train_all.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(train_data).transpose(1,2)
    dat_dict["labels"] = torch.from_numpy(train_labels).transpose(1,2)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))


    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(test_data).transpose(1,2)
    dat_dict["labels"] = torch.from_numpy(test_labels).transpose(1,2)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

    print()
    # data = pd.DataFrame(all_data)
