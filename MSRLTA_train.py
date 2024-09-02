import argparse
import numpy as np
import random
import torch
import warnings
from utils import _logger,set_requires_grad,read_arguments,binary_select
from model.MAE import *
from tqdm import tqdm
from dataLoader.dataloader import *
from torch.multiprocessing import Process, set_start_method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error,r2_score

def train_mae(model_now,train_loader, model_index, indexes, ic_per_layer, num_epochs,args):
    optimizer = torch.optim.Adam(model_now.parameters(), lr=0.001)
    if model_index==0:
        start = 0
        end = sum(ic_per_layer[:(model_index+1)])
    else:
        start = sum(ic_per_layer[:model_index])
        end = sum(ic_per_layer[:(model_index+1)])
    indexes_now = indexes[start]
    for epoch in range(num_epochs):
        loss_sum = 0
        tqdm_train_dataloader = tqdm(train_loader)
        model_now.train()
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            input_now = input[:, start:end, indexes_now]
            optimizer.zero_grad()
            loss = model_now(input_now.float(),0.2,indexes_now)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')

def index_for_conv1d(index,colume_index,dataset_name):
    colume_index = [i-1 for i in colume_index]
    remove_duplicates = [index[i] for i in colume_index]
    indexes_for_conv1d = []
    if dataset_name=='HAR':
        list2 = remove_duplicates[-1]
        list1 = remove_duplicates[1][~np.isin(remove_duplicates[1], list2)]
        list0 = remove_duplicates[0][~np.isin(remove_duplicates[0], remove_duplicates[1])]
        indexes_for_conv1d= [list0,list1,list2]
        conv0_0 = list0
        conv0 = [conv0_0]
        conv1_0 = list1
        conv1_1 =  np.searchsorted(remove_duplicates[1],list1)
        conv1 = [conv1_0,conv1_1]
        conv2_0 = list2
        conv2_1 = np.searchsorted(remove_duplicates[1],list2)
        conv2_2 = np.searchsorted(remove_duplicates[2],list2)
        conv2 = [conv2_0,conv2_1,conv2_2]
        conv = [conv0,conv1,conv2]
    elif dataset_name == 'SAD' or dataset_name == 'TEP':
        list3 = np.array([item for item in remove_duplicates[-1] if item in remove_duplicates[-2]])
        list2_raw = np.array([item for item in remove_duplicates[-1] if item in remove_duplicates[0]])
        list2 = list2_raw[~np.isin(list2_raw, list3)]
        list1_raw = np.array([item for item in remove_duplicates[-2] if item in remove_duplicates[0]])
        list1 = list1_raw[~np.isin(list1_raw, list3)]
        list0 = remove_duplicates[0][~np.isin(remove_duplicates[0], list1_raw)&~np.isin(remove_duplicates[0], list2_raw)]
        if dataset_name == 'TEP':
            indexes_for_conv1d = [list0, list1, list2, list3]
        else:
            indexes_for_conv1d = [((list0+1)/6-1).astype(int), ((list1+1)/6-1).astype(int), ((list2+1)/6-1).astype(int), ((list3+1)/6-1).astype(int)]
        conv0_0 = np.searchsorted(remove_duplicates[0],list0)

        conv1_0 = np.searchsorted(remove_duplicates[0],list1)
        conv1_1 = np.searchsorted(remove_duplicates[1], list1)

        conv2_0 = np.searchsorted(remove_duplicates[0], list2)
        conv2_1 = np.searchsorted(remove_duplicates[2], list2)

        conv3_0 = np.searchsorted(remove_duplicates[0], list3)
        conv3_1 = np.searchsorted(remove_duplicates[1], list3)
        conv3_2 = np.searchsorted(remove_duplicates[2], list3)
        conv0 = [conv0_0]
        conv1 = [conv1_0, conv1_1]
        conv2 = [conv2_0, conv2_1]
        conv3 = [conv3_0, conv3_1, conv3_2]
        conv = [conv0, conv1, conv2, conv3]


    return indexes_for_conv1d,conv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random_seed", default=3407)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--drop_last", type=bool, help="drop_last", default=True)
    parser.add_argument("--device", type=str, help="cuda or cpu", default='cuda:1')
    parser.add_argument('--selected_dataset', default='HAR', type=str, help='Dataset of choice: HAR, SAD, TEP')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    warnings.filterwarnings("ignore")  # if to ignore warnings

    dataset_name = args.selected_dataset
    data_path = f"./datasets/{dataset_name}"
    if dataset_name == 'HAR':
        max_len = 120
        raw_sampling_rate = 50
        num_class = 6
        multi_sampling_rates = [[50, 50, 50], [10, 10, 10], [5, 5, 5]]
        embed_dim = [32,32,32]
        con_in_chans = [32,32*2,32*3]
        task = 'classification'


    elif dataset_name == 'SAD':
        max_len = 90
        raw_sampling_rate = 4410 * 3
        multi_sampling_rates = [[2205, 2205, 2205, 2205, 2205, 2205], [735, 735, 735, 735], [441, 441, 441]]
        embed_dim = [32, 32, 32]
        num_class = 10
        con_in_chans = [32, 32 * 2, 32 * 2,32 * 3]
        task = 'classification'
    elif dataset_name=='TEP':
        max_len = 24
        features_num, num_class = 22, 3
        raw_sampling_rate = 24
        multi_sampling_rates = [[24, 24, 24, 24, 24, 24, 24, 24], [12, 12, 12, 12, 12, 12, 12], [8, 8, 8, 8, 8, 8, 8]]
        embed_dim = [32, 32, 32]
        con_in_chans = [32, 32 * 2, 32 * 2, 32 * 3]
        task = 'prediction'

    train_dl, test_dl, _ = data_generator(dataset_name, args)
    train_part_dl = data_generator_part(dataset_name,25, args)

    binary_sequence, indexes = binary_select(raw_sampling_rate, multi_sampling_rates, max_len)  # [c,l]
    input_channels_per_layer = [len(group) for group in multi_sampling_rates]
    # print(sum(input_channels_per_layer[:1+1]))
    colume_index = np.cumsum(input_channels_per_layer)

    indexes_for_conv1d,indexes_in_enc = index_for_conv1d(indexes,colume_index,dataset_name)

    model_num = len(multi_sampling_rates)
    model = MSRLTA(seq_len=max_len, in_chans=input_channels_per_layer,embed_dim=embed_dim, model_num=model_num,con_in_chans=con_in_chans,conv_num=len(con_in_chans),out_chans=32,class_num=num_class).to(args.device)

    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()

    for i in range(model_num):
        train_mae(model.models[i], train_dl, i, indexes, input_channels_per_layer, 100,args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        loss_sum = 0
        # tqdm_train_dataloader = tqdm(train_dl)
        tqdm_train_dataloader = tqdm(train_part_dl)
        model.train()
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            output = model(input.float(), indexes,input_channels_per_layer,indexes_for_conv1d,indexes_in_enc,dataset_name=dataset_name)
            if task!='prediction':
                loss = CEloss(output, label.view(-1).long())
                loss_sum += loss.item()
            else:
                label = label[:, :, -1].float()
                loss = MSEloss(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
        if (epoch + 1) % 5 == 0:
            tqdm_data_loader = tqdm(test_dl)
            predicts = []
            labels = []
            test_loss_sum = 0
            if task == 'classification':
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model(input.float(), indexes, input_channels_per_layer, indexes_for_conv1d, indexes_in_enc, dataset_name=dataset_name)
                        _, pred = torch.topk(output, 1)
                        test_loss = CEloss(output, label.view(-1).long())
                        labels += label.cpu().numpy().tolist()
                        predicts += pred.view(-1).cpu().numpy().tolist()
                        test_loss_sum += test_loss.cpu().numpy().item()
                f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
                micro_f1 = f1_score(y_true=labels, y_pred=predicts, average='micro')
                acc = accuracy_score(y_true=labels, y_pred=predicts)
                print(f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}')
            else:
                rmse, mae, r2 = [], [], []
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model(input.float(), indexes,input_channels_per_layer,indexes_for_conv1d,indexes_in_enc,dataset_name=dataset_name)
                        label = label[:, :, -1].float()
                        test_loss = MSEloss(output, label)

                        predicts.append(output)
                        labels.append(label)

                labels = torch.cat(labels, dim=0).cpu().numpy()
                predicts = torch.cat(predicts, dim=0).cpu().numpy()
                num = labels.shape[-1]
                rmse_total = np.sqrt(mean_squared_error(labels, predicts))
                mae_total = mean_absolute_error(labels, predicts)
                r2_total = r2_score(labels, predicts)
                for j in range(num):
                    rmse.append(np.sqrt(mean_squared_error(labels[:, j], predicts[:, j])))
                    mae.append(mean_absolute_error(labels[:, j], predicts[:, j]))
                    r2.append(r2_score(labels[:, j], predicts[:, j]))

                print((f'rmse : {rmse_total}\t| \t'f' mae  : {mae_total:.6f} \t| \t'f'r2 : {r2_total}\t| \t'))
                print((f'rmse_29 : {rmse[0]}\t| \t'f' mae_29  : {mae[0]:.6f} \t| \t'f'r2_29 : {r2[0]}\t| \t'))
                print((f'rmse_30 : {rmse[1]}\t| \t'f' mae_30  : {mae[1]:.6f} \t| \t'f'r2_30 : {r2[1]}\t| \t'))
                print((f'rmse_38 : {rmse[2]}\t| \t'f' mae_38  : {mae[2]:.6f} \t| \t'f'r2_38 : {r2[2]}\t| \t'))




