import torch
import torch.nn as nn
import torch.nn.functional as F
from iTransformer_layers.Transformer_EncDec import  Encoder, EncoderLayer
from iTransformer_layers.SelfAttention_Family import DSAttention,FullAttention, AttentionLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_in: int, max_seq_len=5000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_in, 2)) * (-math.log(10000) / dim_in)
        PE = torch.zeros(1, max_seq_len, dim_in)
        PE[0, :, 0::2] = torch.sin(position * div_term)
        if dim_in %2 ==0:
            PE[0, :, 1::2] = torch.cos(position * div_term)
        else:
            PE[0, :, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('PE', PE)

    def forward(self, x):
        x = x + self.PE[:, :x.size(1)]
        return x


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    # 由于平稳化的序列x’没有非平稳性信息，所以需要加入原始序列x，以便更好学习平稳化因子

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class DCAFormerModel(nn.Module):
    # 模型
    # pred_len 预测长度
    # output_attention 是否输出 attention
    # dropout          一般设置为 0.1
    # d_model          输入维度
    # n_heads          头数
    # activation       激活函数
    # d_ff             feedforward大小
    # e_layers         encoder的层数
    # encoder_in       输入维度大小

    def __init__(self,seq_len,d_model1,d_model2,e_layers,enc_in,num_classes, num_MLPlayer=1, hidden_dims = [4]):
        super(DCAFormerModel, self).__init__()
        self.pred_len = 1
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.src_raw_in = enc_in
        self.output_attention = False
        # 变量注意力
        self.crossVariable_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=0.1, output_attention=True),
                        # 这里面mask_flag=False, 就不要mask 掩码了
                        d_model1, 4),  # Attention_layer, d_model1是输入序列的最后一个维度
                    d_model1,
                    256,
                    dropout=0.1,
                    activation='relu'
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model1)
        )

        # 时间注意力
        self.src_encoder = torch.nn.Linear(self.src_raw_in, d_model2)
        self.positionEncoder = PositionalEncoding(self.d_model2)
        self.crossTime_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, 1, attention_dropout=0.1, output_attention=True),
                        # 这里面mask_flag=False, 就不要mask 掩码了
                        d_model2, 4),
                    d_model2,
                    256,
                    dropout=0.1,
                    activation='relu'
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model2)
        )

        self.tau_learner = Projector(enc_in=self.src_raw_in, seq_len=self.seq_len, hidden_dims=hidden_dims,
                                     hidden_layers=num_MLPlayer, output_dim=1)
        self.delta_learner = Projector(enc_in=self.src_raw_in, seq_len=seq_len,
                                       hidden_dims=hidden_dims, hidden_layers=num_MLPlayer,
                                       output_dim=seq_len)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.belta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.proj = nn.Linear((self.enc_in + self.d_model2), num_classes, bias=True)
        self.projector2 = nn.Linear(seq_len, 1, bias=True)

    # 推理
    def forward(self, src, enc_self_mask=None):
        # 协变量注意力 输出
        x_enc1 = src  # batch_size *(enc_in)*seq_len  变量输入
        enc_out1, attns1 = self.crossVariable_encoder(x_enc1, attn_mask=enc_self_mask)  # batch_size * (enc_in) * seq_len  变量注意力

        # 时间注意力 输出
        # 1. 平稳化
        src_raw = src.permute(0, 2, 1)
        mean_src = src_raw.mean(1, keepdim=True).detach()  # B x 1 x E
        std_src = torch.sqrt(torch.var(src_raw, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc2 = (src_raw - mean_src) / std_src  # B* S * E   时间输入

        # 学习出 Delta 和 tau
        tau = self.tau_learner(src_raw, std_src).exp()
        delta = self.delta_learner(src_raw, mean_src)

        # Cross_Time注意力机制，去过度平稳化
        x_enc2 = self.src_encoder(x_enc2)  # batch_size *  seq_len * d_model2
        x_enc2_pe = self.positionEncoder(x_enc2)
        enc_out2, attns2 = self.crossTime_encoder(x_enc2_pe, attn_mask=enc_self_mask, tau=tau, delta=delta)  # batch_size *seq_len * d_model2 时间注意力

        # 回复原始序列
        enc_out2 = enc_out2 * std_src + mean_src

        # 特征融合
        enc_out1 = enc_out1.permute(0, 2, 1)

        enc_out = torch.cat([self.alpha * enc_out1, enc_out2], dim=2)  # 1*7*40

        # 投影
        enc_out = self.proj(enc_out)
        y_pred = enc_out[:, -1, :]
        # y_pred = F.max_pool1d(enc_out.transpose(1, 2), kernel_size=enc_out.size(1)).squeeze()


        if self.output_attention:
            return y_pred, attns1, attns2
        return y_pred