from utils import _logger,set_requires_grad,read_arguments,binary_select
from dataLoader.dataloader import *
import numpy as np
import random
import torch
import warnings
import argparse
from model.TCN import *
from model.LSTM import *
from model.iTransformer import *
from model.Transformer import *
from model.DCAFormer import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error,r2_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random_seed", default=3407)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--drop_last", type=bool, help="drop_last", default=True)
    parser.add_argument("--device", type=str, help="cuda or cpu", default='cuda:1')
    parser.add_argument('--selected_dataset', default='HAR', type=str, help='Dataset of choice: SAD, HAR, JapVo, TEP')
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
        features_num, num_classes = 9, 6
        raw_sampling_rate = 50
        multi_sampling_rates = [[50, 50, 50], [10, 10, 10], [5, 5, 5]]
        task = 'classification'


    elif dataset_name == 'SAD':
        max_len = 90
        features_num, num_classes = 13, 10
        raw_sampling_rate = 4410 * 3
        multi_sampling_rates = [[2205, 2205, 2205, 2205, 2205, 2205], [735, 735, 735, 735], [441, 441, 441]]
        task = 'classification'
        # multi_sampling_rates = [[2205*2, 2205*2, 2205*2, 2205*2, 2205*2, 2205*2], [735*2, 735*2, 735*2, 735*2], [441*2, 441*2, 441*2]]
    elif dataset_name == 'JapVo':
        max_len = 20
        features_num, num_classes = 12, 9
        raw_sampling_rate = 10000
        multi_sampling_rates = [[10000, 10000, 10000, 10000, 10000, 10000], [2000, 2000, 2000, 2000,2000,2000]]
        task = 'classification'
    elif dataset_name=='TEP':
        max_len = 24
        features_num, target_features_num = 22, 3
        raw_sampling_rate = 24
        multi_sampling_rates = [[24, 24, 24, 24, 24, 24, 24, 24], [12, 12, 12, 12, 12, 12, 12], [8, 8, 8, 8, 8, 8, 8]]
        task = 'prediction'


    binary_sequence, indexes = binary_select(raw_sampling_rate, multi_sampling_rates, max_len)  # [c,l]
    input_channels_per_layer = [len(group) for group in multi_sampling_rates]
    colume_index = np.cumsum(input_channels_per_layer)

    pre_operation = 'down'
    print(pre_operation)
    if pre_operation == 'up':
        train_dl, test_dl = up_sampling_generator(data_path, max_len, multi_sampling_rates, indexes, args)
        train_dl = up_sampling_generator_part(data_path, max_len, multi_sampling_rates, indexes, 25, args)
    elif pre_operation == 'down':
        max_len_copy = max_len
        train_dl, test_dl, max_len = down_sampling_generator(data_path, max_len, multi_sampling_rates, indexes, args)
        train_dl = down_sampling_generator_part(data_path, max_len_copy, multi_sampling_rates, indexes,25, args)

    if dataset_name == "HAR":
        # model = RNN(features_num,256,2,num_classes).to(args.device)
        model = TCN(num_classes, features_num, [64, 64], 3).to(args.device)
        # model = BiLSTM(features_num,128,2,num_classes).to(args.device)
        # model = Transformer(input_dim=9,seq_len=120,d_model=256,e_layers=2,num_class=6).to(args.device)
        # model = iTransformer(input_dim=9, seq_len=max_len, d_model=64, e_layers=2, num_class=6).to(args.device)
        # model = DCAFormerModel(seq_len=max_len,d_model1=max_len,d_model2=features_num,e_layers=2,enc_in=features_num, num_classes=num_classes,num_MLPlayer=1, hidden_dims = [8]).to(args.device)
        print(model)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model,device_ids=[0,1])
    elif dataset_name == 'SAD':
        model = TCN(num_classes, features_num, [128, 128,128], 3).to(args.device)
        # model = BiLSTM(features_num, 128, 2, num_classes).to(args.device)
        # model = RNN(13, 256, 2, 10).to(args.device)
        # model = iTransformer(input_dim=features_num, seq_len=max_len, d_model=128, e_layers=3, num_class=num_classes).to(args.device)
        # model = DCAFormerModel(seq_len=max_len,d_model1=max_len,d_model2=features_num,e_layers=2,enc_in=features_num, num_classes=num_classes,num_MLPlayer=3, hidden_dims = [8, 8, 8]).to(args.device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model,device_ids=[0,1])
    elif dataset_name == 'TEP':
        # model = TCN(target_features_num, features_num, [64, 64], 4).to(args.device)
        # model = iTransformer(input_dim=features_num, seq_len=max_len, d_model=64, e_layers=2, num_class=target_features_num).to(args.device)
        model = DCAFormerModel(seq_len=max_len, d_model1=max_len, d_model2=features_num, e_layers=1, enc_in=features_num, num_classes=target_features_num,num_MLPlayer=1, hidden_dims=[8]).to(args.device)
        # model = BiLSTM(features_num, 128, 2, target_features_num).to(args.device)


    OPTIMIZER = {"adam": torch.optim.Adam, "adamw":torch.optim.AdamW}
    optimizer = OPTIMIZER["adamw"](model.parameters(), lr=0.001)

    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()

    for epoch in range(100):
        model.train()
        tqdm_train_dataloader = tqdm(train_dl)
        loss_sum = 0
        eval_rmse = 1e2
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            output = model(input.float(),task)

            if task == 'classification':
                label = label.view(-1).long()
                loss = CEloss(output, label)
            else:
                label = label[:,:,-1].float()
                loss = MSEloss(output, label)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
        if (epoch + 1) % 5 == 0:
            model.eval()
            tqdm_data_loader = tqdm(test_dl)
            predicts = []
            labels = []
            test_loss_sum = 0
            if task == 'classification':
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model(input.float(), task)

                        _, pred = torch.topk(output, 1)
                        test_loss = CEloss(output, label.view(-1).long())
                        labels += label.cpu().numpy().tolist()
                        predicts += pred.view(-1).cpu().numpy().tolist()
                        test_loss_sum += test_loss.cpu().numpy().item()
                f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
                micro_f1 = f1_score(y_true=labels, y_pred=predicts, average='micro')
                acc = accuracy_score(y_true=labels, y_pred=predicts)
                print((f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}'))

            else:
                rmse, mae, r2 = [], [], []
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model(input.float(), task)
                        label = label[:, :, -1].float()
                        test_loss = MSEloss(output,label)

                        predicts.append(output)
                        labels.append(label)

                labels = torch.cat(labels, dim=0).cpu().numpy()
                predicts = torch.cat(predicts, dim=0).cpu().numpy()
                num = labels.shape[-1]
                rmse_total = np.sqrt(mean_squared_error(labels, predicts))
                mae_total = mean_absolute_error(labels, predicts)
                r2_total = r2_score(labels, predicts)
                for j in range(num):
                    rmse.append(np.sqrt(mean_squared_error(labels[:,j], predicts[:,j])))
                    mae.append(mean_absolute_error(labels[:,j], predicts[:,j]))
                    r2.append(r2_score(labels[:,j], predicts[:,j]))

                print((f'rmse : {rmse_total}\t| \t'f' mae  : {mae_total:.6f} \t| \t'f'r2 : {r2_total}\t| \t'))
                print((f'rmse_29 : {rmse[0]}\t| \t'f' mae_29  : {mae[0]:.6f} \t| \t'f'r2_29 : {r2[0]}\t| \t'))
                print((f'rmse_30 : {rmse[1]}\t| \t'f' mae_30  : {mae[1]:.6f} \t| \t'f'r2_30 : {r2[1]}\t| \t'))
                print((f'rmse_38 : {rmse[2]}\t| \t'f' mae_38  : {mae[2]:.6f} \t| \t'f'r2_38 : {r2[2]}\t| \t'))
                if rmse_total < eval_rmse:
                    eval_rmse = rmse_total
                    best_pred = predicts
    if task == 'prediction':
        np.savetxt('pred_results/DCAFormer.txt', best_pred)








