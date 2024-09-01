import torch
from torch import nn, Tensor
import math
import numpy as np
import torch.nn.functional as F

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

class MaskedAutoencoder(nn.Module):
    def __init__(self,seq_len,in_chans,embed_dim,num_heads=8,d_hid = 128,dropout = 0.1,depth=3,decoder_embed_dim=32,decoder_depth=3,decoder_num_heads=8,out_chans=6):
        super(MaskedAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.pos_encoder_e = PositionalEncoding(embed_dim, seq_len)
        self.embedder = nn.Linear(in_chans,embed_dim,bias=False)
        self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(embed_dim, num_heads, d_hid,dropout,)
                for _ in range(depth) ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.pos_decoder_d = PositionalEncoding(decoder_embed_dim, seq_len)
        self.decoder_blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(decoder_embed_dim,decoder_num_heads, d_hid,dropout) for _ in range(decoder_depth) ])
        self.decoder_norm  = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,in_chans, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.proj = nn.Linear(embed_dim,out_chans)



    def rand_masking(self,x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self,x,index,mask_ratio):
        # forward  encoder
        x = self.embedder(x.transpose(1, 2))  # embed patches
        x += self.pos_encoder_e(index).clone().detach()

        x, mask, ids_restore, _ = self.rand_masking(x, mask_ratio)
        for blk in self.blocks:  # apply Transformer blocks
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, index, ids_restore):
        x = self.decoder_embed(x)
        C = x.shape[-1]
        mask_tokens = self.mask_token.repeat(x.shape[0], self.seq_len - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x_
        # x = x_.view([x.shape[0], self.seq_len, C])
        x += self.pos_decoder_d(index).clone().detach()

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss(self,x,pred,mask):
        # x = x.transpose(1,2)
        loss = torch.abs(pred - x.permute(0,2,1))
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self,x,mask_ratio,index=[]):
        N, _, L = x.shape
        if len(index) == 0:
            index = np.arange(L)
        latent, mask, ids_restore = self.forward_encoder(x, index, mask_ratio)
        pred = self.forward_decoder(latent, index, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss

    def fine_tune(self,x, index=None):
        N, _, L = x.shape
        if len(index)==0:
            index = np.arange(L)
        x = self.embedder(x.transpose(1, 2))  # embed patches
        x += self.pos_encoder_e(index).clone().detach()
        for blk in self.blocks: # apply Transformer blocks
            x = blk(x)
        x = self.norm(x)
        return x

    def TiMAE_fine_tune(self,x, index=[]):
        N, _, L = x.shape
        if len(index) == 0:
            index = np.arange(L)
        x = self.embedder(x.transpose(1, 2))  # embed patches
        x += self.pos_encoder_e(index).clone().detach()
        for blk in self.blocks:  # apply Transformer blocks
            x = blk(x)
        x = self.norm(x)
        x = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze()
        x = self.proj(x)
        return x



class MSRLTA(nn.Module):
    def __init__(self,seq_len, in_chans,embed_dim, model_num,con_in_chans,conv_num,out_chans,class_num):
        super(MSRLTA, self).__init__()
        self.mode_num = model_num
        self.conv_num = conv_num
        self.models = nn.ModuleList([MaskedAutoencoder(seq_len, in_chans[i], embed_dim[i]) for i in range(model_num)])
        self.conv1d_list = nn.ModuleList(nn.Conv1d(con_in_chans[i],out_chans,kernel_size=1) for i in range(conv_num))
        self.proj = nn.Linear(out_chans,class_num)

    def forward(self,input,indexes,input_channels_per_layer,conv_index,conv_index_in_enc,dataset_name='HAR'):
        start = 0
        enc_out  = []
        index_model = []
        for i in range(self.mode_num):
            end = start + input_channels_per_layer[i]
            x_now_layer = input[:, start:end, indexes[start]]
            index_now = indexes[start]
            index_model.append(index_now)
            enc_out.append(self.models[i].fine_tune(x_now_layer,index_now))
            start = end

        if dataset_name=='HAR':
            enc_for_conv0 = enc_out[0][:,conv_index_in_enc[0][0],:]
            conv0_out = self.conv1d_list[0](enc_for_conv0.transpose(1,2))

            enc0_for_conv1 = enc_out[0][:,conv_index_in_enc[1][0],:]
            enc1_for_conv1 = enc_out[1][:,conv_index_in_enc[1][1],:]
            enc_for_conv1 = torch.cat((enc0_for_conv1, enc1_for_conv1), dim=-1)
            conv1_out = self.conv1d_list[1](enc_for_conv1.transpose(1, 2))


            enc0_for_conv2 = enc_out[0][:, conv_index_in_enc[2][0], :]
            enc1_for_conv2 = enc_out[1][:, conv_index_in_enc[2][1], :]
            enc2_for_conv2 = enc_out[2][:, conv_index_in_enc[2][2], :]
            enc_for_conv2 = torch.cat((enc0_for_conv2, enc1_for_conv2,enc2_for_conv2), dim=-1)
            conv2_out = self.conv1d_list[2](enc_for_conv2.transpose(1, 2))

            shuffle_out = torch.cat((conv0_out, conv1_out,conv2_out), dim=-1)
            shuffle_inedx = np.concatenate(conv_index)
            restored_out = torch.index_select(shuffle_out, dim=-1, index=torch.tensor(shuffle_inedx).to('cuda:1'))
        elif dataset_name=='SAD' or dataset_name=='TEP':
            enc_for_conv0 = enc_out[0][:,conv_index_in_enc[0][0],:]
            conv0_out = self.conv1d_list[0](enc_for_conv0.transpose(1, 2))

            enc0_for_conv1 = enc_out[0][:, conv_index_in_enc[1][0], :]
            enc1_for_conv1 = enc_out[1][:, conv_index_in_enc[1][1], :]
            enc_for_conv1 = torch.cat((enc0_for_conv1, enc1_for_conv1), dim=-1)
            conv1_out = self.conv1d_list[1](enc_for_conv1.transpose(1, 2))

            enc0_for_conv2 = enc_out[0][:, conv_index_in_enc[2][0], :]
            enc2_for_conv2 = enc_out[2][:, conv_index_in_enc[2][1], :]
            enc_for_conv2 = torch.cat((enc0_for_conv2, enc2_for_conv2), dim=-1)
            conv2_out = self.conv1d_list[2](enc_for_conv2.transpose(1, 2))

            enc0_for_conv3 = enc_out[0][:, conv_index_in_enc[3][0], :]
            enc1_for_conv3 = enc_out[1][:, conv_index_in_enc[3][1], :]
            enc2_for_conv3 = enc_out[2][:, conv_index_in_enc[3][2], :]
            enc_for_conv3 = torch.cat((enc0_for_conv3, enc1_for_conv3, enc2_for_conv3), dim=-1)
            conv3_out = self.conv1d_list[3](enc_for_conv3.transpose(1, 2))
            shuffle_out = torch.cat((conv0_out, conv1_out, conv2_out,conv3_out), dim=-1)
            shuffle_inedx = np.concatenate(conv_index)
            restored_out = torch.index_select(shuffle_out, dim=-1, index=torch.tensor(shuffle_inedx).to('cuda:1'))


        final_out = self.proj(F.max_pool1d(restored_out,kernel_size=restored_out.size(-1)).squeeze())
        # final_out = self.proj(restored_out.transpose(1,2))[:,-1,:]
        return final_out








