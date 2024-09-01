import os
import logging
import sys
import numpy as np
import torch
import ast

def binary_select(raw_sampling_rate,multi_sampling_rates,L):
    sampling_rates = np.concatenate(multi_sampling_rates,axis=0) # sample point list 1:sample 0:non
    real_index = []
    binary_sequence = np.zeros((len(sampling_rates), L))
    for i in range(len(sampling_rates)):
        sample_interval = round(raw_sampling_rate/sampling_rates[i])
        binary_sequence[i,sample_interval-1::sample_interval] = 1
        index = np.arange(sample_interval-1, L, sample_interval)
        real_index.append(index)
    return binary_sequence,real_index
# a,b = binary_select(50,[[50,50,50],[10,10,10],[5,5,5]],128)
# print(a[1][b[1]],b[6:9])

def set_requires_grad(model,pretrain_model_dict,requires_grad = True):
    for param in model.named_parameters():
        if param[0] in pretrain_model_dict:
            param[1].requires_grad = requires_grad


def read_arguments(arg_parser, cfg_parser):
    # dataset info
    arg_parser.drop_last = cfg_parser.getboolean("dataset", "drop_last")
    arg_parser.seq_len = cfg_parser.getint("dataset", "seq_len")
    arg_parser.batch_size = cfg_parser.getint("dataset", "batch_size")
    arg_parser.num_workers = cfg_parser.getint("dataset", "num_workers")
    arg_parser.feature_num = cfg_parser.getint("dataset", "feature_num")
    arg_parser.num_targets = cfg_parser.getint("dataset", "num_targets")
    arg_parser.raw_sampling_rate = cfg_parser.getint("dataset", "raw_sampling_rate")
    samples_list = cfg_parser.get("dataset", "multi_sampling_rates")
    arg_parser.multi_sampling_rates = ast.literal_eval(samples_list)
    arg_parser.task = cfg_parser.get("dataset", "task")

    # model settings
    arg_parser.multi_rate_groups = cfg_parser.getint("model", "multi_rate_groups")
    arg_parser.pooling_type = cfg_parser.get("model", "pooling_type")

    d_model_str = cfg_parser.get("model", "d_model")
    arg_parser.d_model = [int(item) for item in d_model_str.split(',')]
    kernel_size_str = cfg_parser.get("model", "kernel_size")
    arg_parser.kernel_size = [int(item) for item in kernel_size_str.split(',')]

    arg_parser.layers_per_encoder = cfg_parser.getint("model", "layers_per_encoder")
    arg_parser.attn_heads = cfg_parser.getint("model", "attn_heads")
    arg_parser.d_ffn = cfg_parser.getint("model", "d_ffn")
    arg_parser.dropout = cfg_parser.getfloat("model", "dropout")
    arg_parser.enable_res_parameter = cfg_parser.getint("model", "enable_res_parameter")
    arg_parser.momentum = cfg_parser.getfloat("model", "momentum")

    lambda_str = cfg_parser.get("model", "lambda")
    arg_parser.Lambda = [float(item) for item in lambda_str.split(',')]

    # training settings
    arg_parser.device = cfg_parser.get("training", "device")
    arg_parser.lr = cfg_parser.getfloat("training", "lr")
    arg_parser.weight_decay = cfg_parser.getfloat("training", "weight_decay")
    arg_parser.epochs = cfg_parser.getint("training", "epochs")
    arg_parser.num_epoch_pretrain = cfg_parser.getint("training", "num_epoch_pretrain")
    arg_parser.optimizer_type = cfg_parser.get("training", "optimizer_type")

    return arg_parser

def _logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def DataTransform(sample):

    weak_aug = scaling(sample, 1.1)
    strong_aug = jitter(permutation(sample, max_segments=8), 0.8)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)