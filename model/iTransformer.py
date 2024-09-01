import torch
import torch.nn as nn
import torch.nn.functional as F
from iTransformer_layers.Transformer_EncDec import Encoder, EncoderLayer
from iTransformer_layers.SelfAttention_Family import FullAttention, AttentionLayer
from iTransformer_layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self,input_dim, seq_len,d_model,e_layers,num_class):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = 1
        # self.output_attention = configs.output_attention
        self.use_norm = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=0.1,
                                      output_attention=True), d_model, 8),
                    d_model,
                    512,
                    dropout=0.1,
                    activation='gelu'
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, 1, bias=True)
        self.projector2 = nn.Linear(input_dim, num_class, bias=True)

    def forecast(self, x_enc):
        x_enc = x_enc.transpose(1,2)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  #
        #dec_out = enc_out.permute(0, 2, 1)[:, :, :N]
        # filter the covariates
        dec_out = self.projector2(dec_out)  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        # dec_out = F.max_pool1d(dec_out.transpose(1,2),kernel_size = dec_out.size(1)).squeeze()
        return dec_out[:, -self.pred_len, :]  # [B, nc]
        # return dec_out  # [B, nc]