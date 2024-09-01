import argparse
import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump
import torch
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
# from modeling.utils import setup_logger
from dataset_generation.data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)

# def parse_data(line):
#     # 分割数据和类别
#     if ':' in line:
#         print("字符串中包含冒号")
#     data = line.strip().split(':')
#     # 分割数据并返回
#     data,label =data[0],data[1]
#     return data.split(','),label
#
if __name__ == '__main__':
    Path = 'SAD'
    output_dir = '../datasets/SAD'
    m = loadmat(f'{Path}/ArabicDigits.mat')
    mts_data = m['mts'][0][0]
    trainlabels_value = mts_data['trainlabels']
    train_value = mts_data['train']
    testlabels_value = mts_data['testlabels']
    test_value = mts_data['test']

    trainlabels_value = trainlabels_value.squeeze()
    train_value = train_value.squeeze()
    testlabels_value = testlabels_value.squeeze()
    test_value = test_value.squeeze()
    length_list = []
    length_list_test = []
    train_dataset = []
    test_dataset = []

    max_lenth = 0  # 93
    for item in train_value:
        item = torch.as_tensor(item).float()
        length_list.append(item.shape[1])
        if item.shape[1] > max_lenth:
            max_lenth = item.shape[1]
    for item in test_value:
        item = torch.as_tensor(item).float()
        length_list_test.append(item.shape[1])
        if item.shape[1] > max_lenth:
            max_lenth = item.shape[1]

    for x1 in train_value:
        x1 = torch.as_tensor(x1).float()
        if x1.shape[1] != max_lenth:
            padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
            x1 = torch.cat((x1, padding), dim=1)
        train_dataset.append(x1)

    for x2 in test_value:
        x2 = torch.as_tensor(x2).float()
        if x2.shape[1] != max_lenth:
            padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
            x2 = torch.cat((x2, padding), dim=1)
        test_dataset.append(x2)
    trainlabels_value -= np.min(trainlabels_value)
    testlabels_value -= np.min(testlabels_value)

    thre = 40
    indices_train = [idx for idx, value in enumerate(length_list) if value >= thre]
    indices_test = [idx for idx, value in enumerate(length_list_test) if value > thre]

    train_data = torch.stack(train_dataset, dim=0)[indices_train]
    test_data = torch.stack(test_dataset, dim=0)[indices_test]

    train_labels = torch.Tensor(trainlabels_value)[indices_train]
    test_labels = torch.Tensor(testlabels_value)[indices_test]

    dat_dict = dict()
    dat_dict["samples"] = train_data
    dat_dict["labels"] = train_labels
    torch.save(dat_dict, os.path.join(output_dir, "train_all.pt"))
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = test_data
    dat_dict["labels"] = test_labels
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

    # length_list.sort(reverse=True)
    # length_list_test.sort(reverse=True)

    count = sum(1 for x in length_list if x >= thre)
    count_test = sum(1 for x in length_list_test if x >= thre)
    print()


    # mts_data = data['mts'][0, 0]
    #
    #
    #
    # for i in range(len(train_value)):
    #     a = train_value[i]
    #     print(a.shape)
    #
    # print()
#
#     data = pd.read_csv(f'{Path}/SpokenArabicDigits_TRAIN.ts', delimiter='\n', header=None, index_col=None, skiprows=28)
#     a =data.iloc[0]
#     a_string = a.to_string(index=False,header=False)
#     print(a_string)
#     b,c = parse_data(a_string)
#     # df = pd.DataFrame([parse_data(line) for line in data.values], columns=['Data', 'Category'])
#     print()
