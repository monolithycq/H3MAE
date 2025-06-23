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

# from modeling.utils import setup_logger
from dataset_generation.data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == '__main__':
    Path = 'HAR/UCI HAR Dataset'
    output_dir = '../datasets/HAR'


    #channels : 9    classes: 6
    train_acc_x = np.loadtxt(f'{Path}/train/Inertial Signals/body_acc_x_train.txt')
    train_acc_y = np.loadtxt(f'{Path}/train/Inertial Signals/body_acc_y_train.txt')
    train_acc_z = np.loadtxt(f'{Path}/train/Inertial Signals/body_acc_z_train.txt')
    train_gyro_x = np.loadtxt(f'{Path}/train/Inertial Signals/body_gyro_x_train.txt')
    train_gyro_y = np.loadtxt(f'{Path}/train/Inertial Signals/body_gyro_y_train.txt')
    train_gyro_z = np.loadtxt(f'{Path}/train/Inertial Signals/body_gyro_z_train.txt')
    train_tot_acc_x = np.loadtxt(f'{Path}/train/Inertial Signals/total_acc_x_train.txt')
    train_tot_acc_y = np.loadtxt(f'{Path}/train/Inertial Signals/total_acc_y_train.txt')
    train_tot_acc_z = np.loadtxt(f'{Path}/train/Inertial Signals/total_acc_z_train.txt')

    test_acc_x = np.loadtxt(f'{Path}/test/Inertial Signals/body_acc_x_test.txt')
    test_acc_y = np.loadtxt(f'{Path}/test/Inertial Signals/body_acc_y_test.txt')
    test_acc_z = np.loadtxt(f'{Path}/test/Inertial Signals/body_acc_z_test.txt')
    test_gyro_x = np.loadtxt(f'{Path}/test/Inertial Signals/body_gyro_x_test.txt')
    test_gyro_y = np.loadtxt(f'{Path}/test/Inertial Signals/body_gyro_y_test.txt')
    test_gyro_z = np.loadtxt(f'{Path}/test/Inertial Signals/body_gyro_z_test.txt')
    test_tot_acc_x = np.loadtxt(f'{Path}/test/Inertial Signals/total_acc_x_test.txt')
    test_tot_acc_y = np.loadtxt(f'{Path}/test/Inertial Signals/total_acc_y_test.txt')
    test_tot_acc_z = np.loadtxt(f'{Path}/test/Inertial Signals/total_acc_z_test.txt')



    # Stacking channels together data  (N,C,L)
    train_data = np.stack((train_acc_x, train_acc_y, train_acc_z,
                           train_gyro_x, train_gyro_y, train_gyro_z,
                           train_tot_acc_x, train_tot_acc_y, train_tot_acc_z), axis=1)
    test = np.stack((test_acc_x, test_acc_y, test_acc_z,
                       test_gyro_x, test_gyro_y, test_gyro_z,
                       test_tot_acc_x, test_tot_acc_y, test_tot_acc_z), axis=1)


    train_labels = np.loadtxt(f'{Path}/train/y_train.txt')
    train_labels -= np.min(train_labels)
    y_test = np.loadtxt(f'{Path}/test/y_test.txt')
    y_test -= np.min(y_test)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    part = 2.5
    file_name = "train" + str(part) + ".pt"
    _, X_train_part, _, y_part = train_test_split(X_train, y_train, test_size=part/100, random_state=42)


    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(train_data)
    dat_dict["labels"] = torch.from_numpy(train_labels)
    torch.save(dat_dict, os.path.join(output_dir, "train_all.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_val)
    dat_dict["labels"] = torch.from_numpy(y_val)
    torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(test)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train_part)
    dat_dict["labels"] = torch.from_numpy(y_part)

    torch.save(dat_dict, os.path.join(output_dir, file_name))




