import math
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .attention import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
from .loss import Contrastiveloss
from .layer import Layer
from torch.nn.init import xavier_normal_, uniform_, constant_

class H3MAE(nn.Module):
    def __init__(self,args,input_channels_per_layer):
        super(H3MAE, self).__init__()
        self.layer_num = args.multi_rate_groups
        self.ic_per_layer = input_channels_per_layer
        self.d_model = args.d_model
        self.fine_tune = False
        self.device = args.device
        # self.data_shape = args.data_shape  # [c,l]
        self.kernel_size = args.kernel_size
        self.layers_per_encoder = args.layers_per_encoder
        self.momentum = args.momentum
        self.lamada = np.array(args.Lambda)
        self.layers_enc = nn.ModuleList(
            [Layer(args, input_channels_per_layer[i], self.d_model, self.kernel_size[i], self.layers_per_encoder, i) for i in range(self.layer_num)])

        # self.final_dim = int(sum(self.d_model[i]/(i+1) for i in range(self.layer_num)))
        self.final_dim = sum(self.d_model)
        # self.final_dim = self.d_model[0]*self.layer_num
        self.task = args.task
        if self.task == 'classification':
            self.predict_head = nn.Linear(self.final_dim, args.num_targets)
        elif self.task == 'prediction':
            self.predictor_bone = BiLSTM(input_size=self.final_dim, hidden_size=64, num_layers=1, output_size=args.num_targets)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for layer in self.layers_enc:
                for (param_a, param_b) in zip(layer.encoder.parameters(),layer.momentum_encoder.parameters()):
                    param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for layer in self.layers_enc:
                for (param_a, param_b) in zip(layer.encoder.parameters(),layer.momentum_encoder.parameters()):
                    param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data


    def update_loss_weights(self,epoch_now,initial_weights, target_weights,k):
        return target_weights + (initial_weights - target_weights) * np.exp(-k * epoch_now)

    def pretrain_forward(self, input,indexes,colum_index,epoch):
        start = 0
        loss_sum = 0.0
        last_layer_out = torch.empty(0)
        lamada = self.update_loss_weights(epoch,self.lamada,np.array([0.7,0.5,0.3]),0.03)


        for i in range(self.layer_num):
            end = start + self.ic_per_layer[i]
            x_now_layer = input[:, start:end, indexes[start]]
            index_now = indexes[start]
            index_next = indexes[end-1] if i==self.layer_num-1 else indexes[end]
            last_layer_out_copy = last_layer_out.clone()
            loss, last_layer_out = self.layers_enc[i].pretrain_forward(last_layer_out_copy,x_now_layer,index_now,index_next)
            start = end
            loss_copy = loss.clone()
            loss_sum += loss_copy * lamada[i]
            # loss_sum += loss_copy * self.lamada[i]
        return loss_sum

    def forward(self,input,indexes):
        start = 0
        loss_sum = 0.0
        last_layer_out = torch.empty(0)
        list = []
        if self.fine_tune:
            for i in range(self.layer_num):
                end = start + self.ic_per_layer[i]
                x_now_layer = input[:, start:end, indexes[start]]
                index_now = indexes[start]
                index_next = indexes[end - 1] if i == self.layer_num - 1 else indexes[end]
                last_layer_out_copy = last_layer_out.clone()
                output, last_layer_out, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, index_next)
                # list.append(torch.mean(output[:,:,:int(self.d_model[i]/(i+1))],dim=1))
                # list.append(torch.mean(output[:, :, :self.d_model[i]], dim=1))
                # list.append(torch.mean(output[:, :, :self.d_model[i]], dim=1))
                list.append(F.max_pool1d(output[:, :, :self.d_model[i]].transpose(1,2), kernel_size=output.size(1)).squeeze())
                start = end
            all_representation = torch.cat(list,dim=1)
            # for layer in self.predict_head:
            #     all_representation = layer(all_representation)
            return self.predict_head(all_representation)
        else:
            with torch.no_grad():
                for i in range(self.layer_num):
                    end = start + self.ic_per_layer[i]
                    x_now_layer = input[:, start:end, indexes[start]]
                    index_now = indexes[start]
                    index_next = indexes[end - 1] if i == self.layer_num - 1 else indexes[end]
                    last_layer_out_copy = last_layer_out.clone()
                    output, last_layer_out, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, index_next)

                    # list.append(torch.mean(output[:, :, :int(self.d_model[i] / (i + 1))], dim=1))
                    # list.append(torch.mean(output[:, :, :self.d_model[i]], dim=1))
                    list.append(F.max_pool1d(output[:, :, :self.d_model[i]].transpose(1, 2), kernel_size=output.size(1)).squeeze())
                    start = end
            # return torch.mean(output,dim=1)
            return torch.cat(list, dim=1)

    def predict(self,input,indexes):
        start = 0
        loss_sum = 0.0
        last_layer_out = torch.empty(0)
        list = []
        if self.fine_tune:
            for i in range(self.layer_num):
                end = start + self.ic_per_layer[i]
                x_now_layer = input[:, start:end, indexes[start]]
                index_now = indexes[start]
                index_next = indexes[end - 1] if i == self.layer_num - 1 else indexes[end]
                last_layer_out_copy = last_layer_out.clone()
                _, last_layer_out, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, index_next)
                _, output, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, indexes[0])
                list.append(output)
                start = end
            all_representation = torch.cat(list,dim=-1)
            predict = self.predictor_bone(all_representation)
            return predict
        else:
            with torch.no_grad():
                for i in range(self.layer_num):
                    end = start + self.ic_per_layer[i]
                    x_now_layer = input[:, start:end, indexes[start]]
                    index_now = indexes[start]
                    index_next = indexes[end - 1] if i == self.layer_num - 1 else indexes[end]
                    last_layer_out_copy = last_layer_out.clone()
                    _, last_layer_out, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, index_next)
                    output, _, _ = self.layers_enc[i](last_layer_out_copy, x_now_layer, index_now, indexes[0])
                    # list.append(F.max_pool1d(output[:, :, :self.d_model[i]].transpose(1, 2), kernel_size=output.size(1)).squeeze())
                    start = end
                    # return torch.mean(output[:, :, :int(self.d_model[i]/(i+1))],dim=1)
                    list.append(torch.mean(output,dim=1))
                return torch.cat(list, dim=-1)

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_size, self.output_size)
    def forward(self,x):
        r_out, state = self.bilstm(x)

        output = self.linear(r_out[:,-1,:])

        return output







