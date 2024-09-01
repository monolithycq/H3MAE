import torch
import torch.nn as nn
import os
from tqdm import tqdm
from .classification import fit_lr, get_rep_with_label
from .prediction import fit_lr_pred,get_pred_with_label
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error,r2_score
from torch.optim.lr_scheduler import LambdaLR


class Trainer():
    def __init__(self,args,model,train_loader,test_loader,trainAll_dl,logger,optimizer,saved_models,indexes):
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainAll_dl = trainAll_dl
        self.model = model
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.device = args.device
        self.max_len = args.seq_len
        self.contrastiveloss = nn.MSELoss(reduction='mean')
        self.saved_model = saved_models
        self.num_epochs = args.epochs
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()
        self.indexes = indexes
        self.step = 0
        self.task = args.task

    def pretrain_lp(self,experiment_log_dir,colume_index):
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        self.logger.info("Training started ....")
        # self.model.copy_weight()
        eval_acc = 0
        eval_rmse = 1e2
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_train_dataloader = tqdm(self.trainAll_dl)
            loss_sum = 0
            for i,data in enumerate(tqdm_train_dataloader):
                input,label = map(lambda x: x.to(self.device), data)
                self.optimizer.zero_grad()
                loss = self.model.pretrain_forward(input[:,:,:self.max_len].float(),self.indexes,colume_index,epoch)
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            self.logger.info(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum/(i+1):.6f}')
            if (epoch+1)%5 == 0:
                self.model.eval()
                if self.task == 'classification':
                    train_rep, train_label = get_rep_with_label(self.model, self.train_loader, self.indexes, self.args)
                    test_rep, test_label = get_rep_with_label(self.model, self.test_loader, self.indexes, self.args)
                    clf = fit_lr(train_rep, train_label, self.args.seed)
                    acc = clf.score(test_rep, test_label)
                    self.logger.info(f'Epoch : {epoch}\t| \t'
                                     f'ACC     : {acc:.6f} ')
                    if acc > eval_acc:
                        eval_acc = acc
                        torch.save(self.model.state_dict(), self.saved_model + '/pretrain_model.pkl')
                elif self.task=='prediction':
                    train_pred_rep, train_label = get_pred_with_label(self.model, self.train_loader, self.indexes, self.args)
                    test_pred_rep, test_label = get_pred_with_label(self.model, self.test_loader, self.indexes, self.args)
                    clf = fit_lr_pred(train_pred_rep, train_label)
                    predicts = clf.predict(test_pred_rep)
                    rmse = np.sqrt(mean_squared_error(test_label, predicts))
                    mae = mean_absolute_error(test_label, predicts)
                    r2 = r2_score(test_label, predicts)
                    print((f'rmse : {rmse}\t| \t'f' mae  : {mae:.6f} \t| \t'f'r2 : {r2}\t| \t'))
                    if rmse < eval_rmse:
                        eval_rmse = rmse
                        torch.save(self.model.state_dict(), self.saved_model + '/pretrain_model.pkl')



    # def pretrain(self, experiment_log_dir): #forecast
    #     os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    #     self.logger.info("Training started ....")
    #     for epoch in range(self.num_epoch_pretrain):
    #         self.model.train()
    #         tqdm_train_dataloader = tqdm(self.trainAll_dl)
    #         loss_sum = 0
    #         for i, data in enumerate(tqdm_train_dataloader):
    #             self.optimizer.zero_grad()

    def fine_tune(self,pretrain_dic):
        if self.task == 'classification':
            self.fine_tune_cls(pretrain_dic)
        elif self.task == 'prediction':
            self.fine_tune_pred(pretrain_dic)


    def fine_tune_pred(self,pretrain_dic):
        self.logger.info("Fine-tune started ....")
        self.logger.info("load pretrained model   " + pretrain_dic)
        pretrain_loc = pretrain_dic + '/pretrain_model.pkl'
        state_dict = torch.load(pretrain_loc, map_location=self.device)
        eval_rmse = 1e2

        self.model.load_state_dict(state_dict, False)
        self.model.fine_tune = True
        for epoch in range(self.num_epochs):
            self.model.train()
            tqdm_train_dataloader = tqdm(self.train_loader)
            loss_sum= 0
            for i,data in enumerate(tqdm_train_dataloader):
                input,label = map(lambda x: x.to(self.device), data)
                self.optimizer.zero_grad()
                pred = self.model.predict(input.float(),self.indexes) # [b,classes]
                label = label[:, :, -1].float()

                loss = self.MSEloss(pred, label)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                self.step += 1
            self.logger.info(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                tqdm_data_loader = tqdm(self.test_loader)
                predicts = []
                labels = []
                test_loss_sum = 0
                rmse, mae, r2 = [], [], []
                with torch.no_grad():
                    for i, data in enumerate(tqdm_data_loader):
                        input, label = map(lambda x: x.to(self.device), data)
                        pred = self.model.predict(input.float(),self.indexes)
                        label = label[:, :, -1].float()
                        test_loss = self.MSEloss(pred,label)

                        predicts.append(pred)
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
                self.logger.info((f'rmse : {rmse_total}\t| \t'f' mae  : {mae_total:.6f} \t| \t'f'r2 : {r2_total}\t| \t'))
                self.logger.info((f'rmse_29 : {rmse[0]}\t| \t'f' mae_29  : {mae[0]:.6f} \t| \t'f'r2_29 : {r2[0]}\t| \t'))
                self.logger.info((f'rmse_30 : {rmse[1]}\t| \t'f' mae_30  : {mae[1]:.6f} \t| \t'f'r2_30 : {r2[1]}\t| \t'))
                self.logger.info((f'rmse_38 : {rmse[2]}\t| \t'f' mae_38  : {mae[2]:.6f} \t| \t'f'r2_38 : {r2[2]}\t| \t'))
                if rmse_total<eval_rmse:
                    eval_rmse = rmse_total
                    best_pred = predicts

        np.savetxt('pred_results/H3MAE.txt', best_pred)
        np.savetxt('pred_results/real.txt', labels)



    def fine_tune_cls(self,pretrain_dic):
        if pretrain_dic != None:
            self.logger.info("Fine-tune started ....")
            self.logger.info("load pretrained model   " + pretrain_dic)
            pretrain_loc = pretrain_dic + '/pretrain_model.pkl'
            state_dict = torch.load(pretrain_loc, map_location=self.device)
            del state_dict['predict_head.weight']
            del state_dict['predict_head.bias']

            self.model.load_state_dict(state_dict, False)
            self.model.eval()
            train_rep, train_label = get_rep_with_label(self.model, self.train_loader, self.indexes, self.args)
            test_rep, test_label = get_rep_with_label(self.model, self.test_loader, self.indexes, self.args)
            clf = fit_lr(train_rep, train_label, self.args.seed)
            pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
            f1 = f1_score(test_label, pred_label, average='macro')
            acc = clf.score(test_rep, test_label)
            self.logger.info(f'Acc : {acc:.6f}\t| \t'  f'F1_score : {f1}')




        self.model.fine_tune = True
        eval_acc = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            tqdm_train_dataloader = tqdm(self.train_loader)
            loss_sum= 0
            for i,data in enumerate(tqdm_train_dataloader):
                input,label = map(lambda x: x.to(self.device), data)
                self.optimizer.zero_grad()
                output = self.model(input.float(),self.indexes) # [b,classes]
                label = label.view(-1).long()

                loss = self.CEloss(output,label)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                self.step += 1
            self.logger.info(f'Epoch : {epoch}\t| \t'f'Train Loss     : {loss_sum / (i + 1):.6f}')
            if (epoch+1)%5 == 0:
                acc, _, _ = self.eval_model()
                if acc > eval_acc and pretrain_dic != None:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), pretrain_dic + '/fine_tune_model.pkl')

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader)
        predicts = []
        labels = []
        test_loss_sum = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm_data_loader):
                input, label = map(lambda x: x.to(self.device), data)
                output = self.model(input.float(), self.indexes)
                _, pred = torch.topk(output, 1)
                test_loss = self.CEloss(output, label.view(-1).long())
                labels += label.cpu().numpy().tolist()
                predicts += pred.view(-1).cpu().numpy().tolist()
                test_loss_sum += test_loss.cpu().numpy().item()

        f1 = f1_score(y_true=labels, y_pred=predicts, average='macro')
        micro_f1 = f1_score(y_true=labels, y_pred=predicts, average='micro')
        acc = accuracy_score(y_true=labels, y_pred=predicts)
        # self.logger.info(f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}\t| \t' f'acc : {acc}')
        self.logger.info(f'acc : {acc}\t| \t'f' Test Loss  : {test_loss_sum / (i + 1):.6f} \t| \t'f'f1 : {f1}\t| \t'f'micro_f1  :{micro_f1}' )

        return acc, labels, predicts











