from utils import _logger,set_requires_grad,read_arguments,binary_select
from dataLoader.dataloader import *
import numpy as np
import random
import torch
import torch.nn.functional as F
import warnings
import argparse
from model.TS2Vec import *
from model.TS_TCC import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,mean_squared_error,mean_absolute_error,r2_score
from trainer.classification import fit_lr
from model.loss import NTXentLoss
from model.MAE import *
import torch.nn.functional as F

def TiMAE_pre(model,train_dl,mask_ratio,optimizer,args):
    for epoch in range(100):
        model.train()
        tqdm_train_dataloader = tqdm(train_dl)
        loss_sum = 0
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            loss = model(input.float(),mask_ratio=0.4)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
    print(("Fine-tune started ...."))

def TiMAE_fine_tune(model,train_dl,test_dl,optimizer,task,args):
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    model.fine_tune = True
    if task == 'classification':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label = map(lambda x: x.to(args.device), data)
                optimizer.zero_grad()
                output = model.TiMAE_fine_tune(input.float())  # [b,classes]
                loss = CEloss(output, label.view(-1).long())
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
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model.TiMAE_fine_tune(input.float())
                        _, pred = torch.topk(output, 1)
                        test_loss = CEloss(output, label.view(-1).long())
                        labels += label.cpu().numpy().tolist()
                        predicts += pred.view(-1).cpu().numpy().tolist()
                        test_loss_sum += test_loss.cpu().numpy().item()
                f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
                micro_f1 = f1_score(y_true=labels, y_pred=predicts, average='micro')
                acc = accuracy_score(y_true=labels, y_pred=predicts)
                print(f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}')
    elif task == 'prediction':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label = map(lambda x: x.to(args.device), data)
                optimizer.zero_grad()
                output = model.TiMAE_fine_tune(input.float())  # [b,classes]
                label = label[:, :, -1].float()
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
                rmse, mae, r2 = [], [], []
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model.TiMAE_fine_tune(input.float())
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

def TS2Vec_pre(model,train_dl,test_dl,optimizer,args):
    for epoch in range(100):
        model.train()
        tqdm_train_dataloader = tqdm(train_dl)
        loss_sum = 0
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            loss = model.fit(input.float())
            loss.backward()
            optimizer.step()
            model.net.update_parameters(model._net)
            loss_sum += loss.item()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
        # if (epoch + 1) % 5 == 0:
        #     train_results, train_labels = [], []
        #     test_results, test_labels = [], []
        #     model.eval()
        #     with torch.no_grad():
        #         tqdm_train_dataloader = tqdm(train_dl)
        #         for i, data in enumerate(tqdm_train_dataloader):
        #             input, label = map(lambda x: x.to(args.device), data)
        #
        #             train_labels += label.cpu().numpy().tolist()
        #             result = model.encoder(input.float())
        #             train_results += result.cpu().numpy().tolist()
        #
        #         tqdm_test_dataloader = tqdm(test_dl)
        #         for i, data in enumerate(tqdm_test_dataloader):
        #             input, label = map(lambda x: x.to(args.device), data)
        #
        #             test_labels += label.cpu().numpy().tolist()
        #             result = model.encoder(input.float())
        #             test_results += result.cpu().numpy().tolist()
        #     clf = fit_lr(train_results, train_labels, args.seed)
        #     acc = clf.score(test_results, test_labels)
        #     print(f'Epoch : {epoch}\t| \t' f'ACC     : {acc:.6f} ')
    print(("Fine-tune started ...."))

def TS2Vec_fine(model,train_dl,test_dl,optimizer,task,args):
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    model.fine_tune = True
    if task == 'classification':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label = map(lambda x: x.to(args.device), data)
                optimizer.zero_grad()
                output = model.encoder(input.float())  # [b,classes]
                loss = CEloss(output, label.view(-1).long())
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
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model.encoder(input.float())
                        _, pred = torch.topk(output, 1)
                        test_loss = CEloss(output, label.view(-1).long())
                        labels += label.cpu().numpy().tolist()
                        predicts += pred.view(-1).cpu().numpy().tolist()
                        test_loss_sum += test_loss.cpu().numpy().item()
                f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
                micro_f1 = f1_score(y_true=labels, y_pred=predicts, average='micro')
                acc = accuracy_score(y_true=labels, y_pred=predicts)
                print(f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}')
    elif task == 'prediction':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label = map(lambda x: x.to(args.device), data)
                optimizer.zero_grad()
                output = model.encoder(input.float())  # [b,classes]
                label = label[:, :, -1].float()
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
                rmse, mae, r2 = [], [], []
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(args.device), data)
                        output = model.encoder(input.float())
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

def TS_TCC_pre(model,temporal_contr_model,train_dl,test_dl,optimizer,temp_cont_optimizer,args):
    total_loss = []
    for epoch in range(100):
        model.train()
        temporal_contr_model.train()
        tqdm_train_dataloader = tqdm(train_dl)
        loss_sum = 0
        for i, data in enumerate(tqdm_train_dataloader):
            input, label, aug1, aug2 = map(lambda x: x.to(args.device), data)

            optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()

            predictions1, features1 = model(aug1.float())
            predictions2, features2 = model(aug2.float())

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)
            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2

            nt_xent_criterion = NTXentLoss(args.device, args.batch_size, temperature=0.2,use_cosine_similarity=True)
            lambda1 = 1
            lambda2 = 0.7
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            temp_cont_optimizer.step()
            loss_sum += loss.item()
        print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
def TS_TCC_fine(model,train_dl,test_dl,optimizer,task,args):
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    total_loss = []
    total_acc = []
    eval_rmse = 1e2
    rmse, mae, r2 = [], [], []
    if task == 'classification':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label, aug1, aug2 = map(lambda x: x.to(args.device), data)
                output = model(input.float())
                predictions, features = output
                loss = CEloss(predictions, label.view(-1).long())
                total_acc.append(label.eq(predictions.detach().argmax(dim=1)).float().mean())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
            if (epoch + 1) % 5 == 0:
                model.eval()
                predicts = []
                labels = []
                test_loss_sum = 0
                tqdm_data_loader = tqdm(test_dl)
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label, _, _ = map(lambda x: x.to(args.device), data)
                        output = model(input.float())
                        predictions, features = output
                        test_loss = CEloss(predictions, label.view(-1).long())
                        _, pred = torch.topk(predictions, 1)
                        labels += label.cpu().numpy().tolist()
                        predicts += pred.view(-1).cpu().numpy().tolist()
                        test_loss_sum += test_loss.cpu().numpy().item()
                        #
                        # test_acc.append(label.eq(predictions.detach().argmax(dim=1)).float().mean())
                        # test_loss.append(loss.item())
                        #
                        # pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        # outs = np.append(outs, pred.cpu().numpy())
                        # trgs = np.append(trgs, label.data.cpu().numpy())
                    # test_loss = torch.tensor(test_loss).mean()
                    # test_acc = torch.tensor(total_acc).mean()
                    f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
                    acc = accuracy_score(y_true=labels, y_pred=predicts)
                    print(f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t')
    elif task == 'prediction':
        for epoch in range(100):
            model.train()
            tqdm_train_dataloader = tqdm(train_dl)
            loss_sum = 0
            for i, data in enumerate(tqdm_train_dataloader):
                input, label, aug1, aug2 = map(lambda x: x.to(args.device), data)
                output = model(input.float())
                predictions, features = output
                label = label[:, :, -1].float()
                loss = MSEloss(predictions, label)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
            if (epoch + 1) % 5 == 0:
                model.eval()
                predicts = []
                labels = []
                test_loss_sum = 0
                tqdm_data_loader = tqdm(test_dl)
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label, _, _ = map(lambda x: x.to(args.device), data)
                        output = model(input.float())
                        predictions, features = output
                        label = label[:, :, -1].float()
                        test_loss = MSEloss(predictions, label)
                        predicts.append(predictions)
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
                    if rmse_total < eval_rmse:
                        eval_rmse = rmse_total
                        best_pred = predicts
        if task == 'prediction':
            np.savetxt('pred_results/TS_TCC.txt', best_pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random_seed", default=3407)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--drop_last", type=bool, help="drop_last", default=True)
    parser.add_argument("--device", type=str, help="cuda or cpu", default='cuda:1')
    parser.add_argument('--selected_dataset', default='HAR', type=str, help='Dataset of choice: SAD, HAR, JapVo, TEP')
    parser.add_argument('--selected_ssl_model', default='TS2Vec', type=str, help='self supervised learning model of choice: TS2Vec, TS-TCC, Ti-MAE')
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
        multi_sampling_rates = [[50, 50, 50], [10, 10, 10], [5, 5, 5]]
        raw_sampling_rate = 50
        features_num,num_classes = 9,6
        hidden_dims, layers, output_dims = 32,1,32 #TS2Vec
        # h_d, final_out_channels, timesteps,logits_p = 100, 64, 6, 122 #122#TS-TCC
        h_d, final_out_channels, timesteps,logits_p = 100, 128, 6, 14 #122#TS-TCC
        embed_dim, mask_ratio, depth, decoder_depth = 64, 0.5, 2, 2 #TiMAE
        task = 'classification'
    elif dataset_name == 'SAD':
        max_len = 90
        raw_sampling_rate = 13230
        multi_sampling_rates = [[2205, 2205, 2205, 2205, 2205, 2205], [735, 735, 735, 735], [441, 441, 441]]
        features_num, num_classes = 13, 10
        hidden_dims, layers, output_dims = 64, 1, 64  # TS2Vec
        h_d, final_out_channels, timesteps, logits_p = 64, 64, 6, 92  # 122#TS-TCC
        embed_dim, mask_ratio, depth, decoder_depth = 64, 0.6, 3, 3  # TiMAE
        task = 'classification'
    elif dataset_name == 'JapVo':
        max_len = 20
        features_num, num_classes = 12, 9
        raw_sampling_rate = 10000
        multi_sampling_rates = [[10000, 10000, 10000, 10000, 10000, 10000], [5000, 5000, 5000, 5000,5000,5000]]
        hidden_dims, layers, output_dims = 64, 1, 64  # TS2Vec
        task = 'classification'
    elif dataset_name=='TEP':
        max_len = 24
        features_num, num_classes = 22, 3
        raw_sampling_rate = 24
        multi_sampling_rates = [[24, 24, 24, 24, 24, 24, 24, 24], [12, 12, 12, 12, 12, 12, 12], [8, 8, 8, 8, 8, 8, 8]]
        hidden_dims, layers, output_dims = 64, 1, 64  # TS2Vec
        embed_dim, mask_ratio, depth, decoder_depth = 64, 0.4, 2, 2  # TiMAE
        h_d, final_out_channels, timesteps, logits_p = 64, 64, 6, 26  # 122#TS-TCC
        task = 'prediction'

    binary_sequence, indexes = binary_select(raw_sampling_rate, multi_sampling_rates, max_len)  # [c,l]
    input_channels_per_layer = [len(group) for group in multi_sampling_rates]
    colume_index = np.cumsum(input_channels_per_layer)
    pre_operation = 'down'
    print(pre_operation)

    if args.selected_ssl_model == 'TS-TCC': augment = True
    else: augment = False

    # train_dl, valid_dl, test_dl,trainAll_dl = data_generator(data_path, args,augment)
    if pre_operation == 'up':
        train_dl, test_dl = up_sampling_generator(data_path, max_len, multi_sampling_rates, indexes, args, augment)
        train_part_dl = up_sampling_generator_part(data_path, max_len, multi_sampling_rates, indexes,25, args, augment)
    elif pre_operation == 'down':
        max_len_copy = max_len
        train_dl, test_dl, max_len = down_sampling_generator(data_path, max_len, multi_sampling_rates, indexes, args, augment)
        train_part_dl = down_sampling_generator_part(data_path, max_len_copy, multi_sampling_rates, indexes, 25, args, augment)

    OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    # model = TS2Vec(input_dims=9,hidden_dims=64,layers=1,output_dims=32).to(args.device)
    if args.selected_ssl_model == 'TS2Vec':
        model = TS2Vec(input_dims=features_num, hidden_dims=hidden_dims, layers=layers, output_dims=output_dims,num_classes=num_classes).to(args.device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model,device_ids=[0,1])
        optimizer = OPTIMIZER["adamw"](model.parameters(), betas=(0.9, 0.99), lr=0.001)
        TS2Vec_pre(model, train_dl, test_dl, optimizer, args)
        # TS2Vec_fine(model, train_dl, test_dl, optimizer,task, args)
        TS2Vec_fine(model, train_part_dl, test_dl, optimizer,task, args)
    elif args.selected_ssl_model == 'TS-TCC':
        model = base_Model(input_channels=features_num, kernel_size=1, stride=1, final_out_channels=final_out_channels, num_classes=num_classes,logits_p=logits_p,task=task).to(args.device)
        temporal_contr_model = TC(timesteps=timesteps, hidden_dim=h_d, final_out_channels=final_out_channels).to(args.device)
        temporal_contr_optimizer = OPTIMIZER["adam"](temporal_contr_model.parameters(), betas=(0.9, 0.99), lr=0.0005)
        optimizer = OPTIMIZER["adam"](model.parameters(), betas=(0.9, 0.99), lr=0.0005)
        TS_TCC_pre(model,temporal_contr_model,train_dl,test_dl,optimizer,temporal_contr_optimizer,args)
        # TS_TCC_fine(model,train_dl,test_dl,optimizer,task,args)
        TS_TCC_fine(model,train_part_dl,test_dl,optimizer,task,args)
    elif args.selected_ssl_model == 'Ti-MAE':
        model = MaskedAutoencoder(seq_len=max_len,in_chans=features_num,embed_dim=embed_dim,out_chans=num_classes,depth=depth,decoder_depth=decoder_depth).to(args.device)
        optimizer = OPTIMIZER["adamw"](model.parameters(), betas=(0.9, 0.99), lr=0.001)
        TiMAE_pre(model, train_dl, mask_ratio, optimizer, args)
        # TiMAE_fine_tune(model, train_dl, test_dl, optimizer,task, args)
        TiMAE_fine_tune(model, train_part_dl, test_dl, optimizer,task, args)
    # TS2Vec_pre(model, train_dl, test_dl, optimizer, args)
    # TS2Vec_fine(model, train_dl, test_dl, optimizer, args)

    # if dataset_name == 'HAR':
    #     # model = base_Model(input_channels=9, kernel_size=1, stride=1, final_out_channels=128, num_classes=6).to(args.device)
    #     # temporal_contr_model = TC(timesteps=6, hidden_dim=100, final_out_channels=128).to(args.device)
    #     # temporal_contr_optimizer = OPTIMIZER["adam"](temporal_contr_model.parameters(), betas=(0.9, 0.99), lr=0.0003)
    #     model = TS2Vec(input_dims=9, hidden_dims=64, layers=1, output_dims=32).to(args.device)
    #     optimizer = OPTIMIZER["adam"](model.parameters(), betas=(0.9, 0.99), lr=0.0003)
    # elif dataset_name == 'SAD':
    #     # model = base_Model(input_channels=13, kernel_size=1, stride=1, final_out_channels=128, num_classes=10).to(args.device)
    #     # temporal_contr_model = TC(timesteps=6, hidden_dim=100, final_out_channels=128).to(args.device)
    #     # temporal_contr_optimizer = OPTIMIZER["adam"](temporal_contr_model.parameters(), betas=(0.9, 0.99), lr=0.0003)
    #     model = TS2Vec(input_dims=13, hidden_dims=64, layers=1, output_dims=32, num_classes=10).to(args.device)
    #     optimizer = OPTIMIZER["adam"](model.parameters(), betas=(0.9, 0.99), lr=0.0003)

    # TS2Vec_pre(model,train_dl,test_dl,optimizer,args)

# TS_TCC_pre(model,temporal_contr_model,train_dl,test_dl,optimizer,temporal_contr_optimizer,args)
# TS_TCC_fine(model,train_dl,test_dl,optimizer,args)


