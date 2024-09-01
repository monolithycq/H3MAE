import argparse
import math
import random
import os
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from model.H3MAE import *
from configparser import ConfigParser, ExtendedInterpolation
from utils import _logger,set_requires_grad,read_arguments,binary_select
from dataLoader.dataloader import *
from trainer.trainer import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#python main.py --config_path configs/HAR.ini
OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random_seed", default=3407)
    parser.add_argument("--config_path", type=str, help="path of config file", default='configs/SAD.ini')
    parser.add_argument('--selected_dataset', default='SAD', type=str, help='Dataset of choice: SAD, HAR, JapVo, TEP')
    parser.add_argument('--multi_rate_groups', default='3', type=int, help='multi_sample_rating')
    parser.add_argument('--result_saving_base_dir', default='experiments_logs&models', type=str, help='saving directory')
    parser.add_argument('--pretrain_dic', default='experiments_logs&models/SAD/_seed_3407/saved_models/2024-08-09_T17-21-57', type=str, help='pretrain model directory')
    parser.add_argument('--fine_tune', default= False, type=bool)

    args = parser.parse_args()

    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)
    args.model_type = 'transformer'

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    warnings.filterwarnings("ignore")  # if to ignore warnings

    #create log directory
    time_now = datetime.now().__format__("%Y-%m-%d_T%H-%M-%S")
    experiment_log_dir = os.path.join(args.result_saving_base_dir, args.selected_dataset, f"_seed_{args.seed}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    if args.fine_tune == False:
        log_file_name = os.path.join(experiment_log_dir, 'logs' + '_' + time_now)
        saved_model = os.path.join(experiment_log_dir, 'saved_models', time_now)
        os.makedirs(saved_model, exist_ok=True)
    else:
        time_pre_train = os.path.basename(args.pretrain_dic)
        log_file_name = os.path.join(experiment_log_dir, 'logs' + '_' + time_pre_train+'xxx'+time_now)
        saved_model = os.path.join(experiment_log_dir, 'saved_models', time_now)

    logger = _logger(log_file_name)
    logger.info(f"args: {args}")
    logger.info(f'Dataset: {args.selected_dataset}')

    # Load datasets
    dataset_name = args.selected_dataset
    train_dl, test_dl,trainAll_dl = data_generator(dataset_name, args) #[b,c,l]
    logger.info("Data loaded ...")
    binary_sequence, indexes = binary_select(args.raw_sampling_rate,args.multi_sampling_rates,args.seq_len)  #[c,l]

    input_channels_per_layer = [len(group) for group in args.multi_sampling_rates]
    colume_index =  np.cumsum(input_channels_per_layer)
    # indexes = [indexes[i] for i in colume_index - 1]
    # logger.info(f"indexes:{indexes}")

    # Load model
    model = H3MAE(args, input_channels_per_layer).to(args.device)
    logger.info("Model initial ends ...")

    optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr)

    trainer = Trainer(args, model, train_dl, test_dl,trainAll_dl,logger,optimizer,saved_model,indexes)

    if args.fine_tune:
        trainer.fine_tune(args.pretrain_dic)
    else:
        trainer.pretrain_lp(experiment_log_dir, colume_index)

    # training with limited labels

    # train_part_dl = data_generator_part(dataset_name,2.5, args)
    # trainer_part = Trainer(args, model, train_part_dl, test_dl, trainAll_dl, logger, optimizer, saved_model, indexes)
    # trainer_part.fine_tune(args.pretrain_dic)
    # trainer_part.fine_tune(None)




