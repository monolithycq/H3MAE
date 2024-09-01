import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .attention import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
from torch.nn.init import xavier_normal_, uniform_, constant_

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter [1,n_position,d_hid]
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, index):
        return self.pos_table[:, index] #[1,len(index),d]
        # return x + self.pos_table[:, : x.size(1)].clone().detach()

class Layer(nn.Module):
    def __init__(self,args,input_channel,d_model,kernel_size,layers,idx):
        super(Layer, self).__init__()
        self.max_len = args.seq_len
        self.idx = idx
        self.momentum = args.momentum
        self.d_model = d_model[idx]
        if idx!=0:
            self.last_projection = nn.Conv1d(d_model[idx-1],int(self.d_model/3),kernel_size=1)
            out_channel = int(2*self.d_model/3)
        else:
            out_channel = self.d_model

        self.ks = kernel_size
        self.input_projection = nn.Conv1d(input_channel,out_channel,kernel_size=self.ks,stride=self.ks)


        # self.position =  PositionalEmbedding(self.max_len, d_model)
        self.position = PositionalEncoding(self.d_model,self.max_len)

        self.imputation_token = nn.Parameter(torch.randn(self.d_model, ))

        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        dropout = args.dropout

        enable_res_parameter = args.enable_res_parameter

        self.encoder = Encoder(self.d_model, attn_heads, d_ffn, enable_res_parameter, dropout,layers)
        self.momentum_encoder = Encoder(self.d_model, attn_heads, d_ffn, enable_res_parameter, dropout,layers)
        self.imputation = Imputation(self.d_model, attn_heads, d_ffn, 1,4)

        self.imputationloss = nn.MSELoss(reduction='mean')

    def pretrain_forward(self,last_layer_out,x_now,index_now, index_next):
        '''

        :param last_layer_out: 上一层生成的当前层位置索引的值 [b,l,d]
        :param x_now: 仅保留采样点的值 [b,c,l]
        :param index_now: 当前层采样点索引  [l]
        :param index_next: 下一层采样点索引  [l_next]
        :return: loss和生成的下层位置索引的值 [b,l_next,d]
        '''
        if self.idx == 0:
            input = self.input_projection(x_now).transpose(1, 2).contiguous()  # [b,l,c]
        else:
            x_now = self.input_projection(x_now).transpose(1, 2).contiguous()
            last_layer_out = self.last_projection(last_layer_out.transpose(1, 2)).transpose(1, 2).contiguous()
            input = torch.cat((x_now,last_layer_out),dim=2)

        index_now = index_now[self.ks-1::self.ks]
        input += self.position(index_now).clone().detach()

        token = self.imputation_token.repeat(input.shape[0], input.shape[1], 1) + self.position(index_now)
        token_next = self.imputation_token.repeat(input.shape[0], len(index_next), 1) + self.position(index_next)

        mask_len = int(0.2*len(index_now)) #SAD 0.6
        # mask_len = round(0.4*len(index_now))
        if mask_len == 0:
            mask_len=1
        re_index = np.arange(len(index_now))
        random.shuffle(re_index)
        v_index = re_index[:-mask_len]
        m_index = re_index[-mask_len:]
        visible = input[:, v_index, :]
        mask = input[:, m_index, :]

        mask_token_now = token[:, m_index, :]
        all_imputation_token = torch.cat((mask_token_now,token_next),dim=1)

        visible_enc = self.encoder(visible)
        with torch.no_grad():
            mask_imputation_enc = self.momentum_encoder(mask)

        all_imputation = self.imputation(visible_enc,all_imputation_token)
        loss = self.imputationloss(mask_imputation_enc, all_imputation[:,:mask_len,:]) #correspond
        return loss, all_imputation[:,mask_len:,:] #imputation for next layer

    def forward(self,last_layer_out,x_now,index_now, index_next):
        if self.idx == 0:
            input = self.input_projection(x_now).transpose(1, 2).contiguous()  # [b,l,c]
        else:
            x_now = self.input_projection(x_now).transpose(1, 2).contiguous()
            last_layer_out = self.last_projection(last_layer_out.transpose(1, 2)).transpose(1, 2).contiguous()
            input = torch.cat((x_now, last_layer_out), dim=2)
        index_now = index_now[self.ks - 1::self.ks]
        input += self.position(index_now).clone().detach()
        token_next = self.imputation_token.repeat(input.shape[0], len(index_next), 1) + self.position(index_next)
        output = self.encoder(input)
        next_layer_imputation = self.imputation(output,token_next)
        return output,next_layer_imputation,index_now




class Encoder(nn.Module):
    def __init__(self,d_model, attn_heads, d_ffn, enable_res_parameter, dropout,layers):
        super(Encoder, self).__init__()
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for _ in range(layers)])

    def forward(self,input):
        for trm in self.TRMs:
            input = trm(input,mask=None)
        return input


class Imputation(nn.Module):
    def __init__(self,d_model, attn_heads, d_ffn, enable_res_parameter,layers):
        super(Imputation, self).__init__()
        self.imputation_layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for _ in range(layers)])
    def forward(self,visible_enc,all_imputation_token):
        for imputation_layer in self.imputation_layers:
            all_imputation_token = imputation_layer(visible_enc, all_imputation_token)
        return all_imputation_token