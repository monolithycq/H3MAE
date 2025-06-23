from scipy.io import arff
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
Path = 'NATOPS'
output_dir = '../datasets/NATOPS'

train_data = arff.loadarff(f'{Path}/NATOPS_TRAIN.arff')[0]
test_data = arff.loadarff(f'{Path}/NATOPS_TEST.arff')[0]


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


X_train_all, y_train_all = extract_data(train_data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all)
X_test, y_test = extract_data(test_data)

labels = np.unique(y_train_all)
transform = { k : i for i, k in enumerate(labels)}
y_train_all = np.vectorize(transform.get)(y_train_all)
y_train = np.vectorize(transform.get)(y_train)
y_valid = np.vectorize(transform.get)(y_valid)
y_test = np.vectorize(transform.get)(y_test)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train_all).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_train_all)
torch.save(dat_dict, os.path.join(output_dir, "train_all.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_valid).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_valid)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
