import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from position_encoder import PositionalEncoding

class AudioEncoder(nn.Module):

    def __init__(self, audio_dim, audio_size, tied_output, nhead, nlayer, d_model, d_feedforward, dropout, down_sampling_factor):
        super(AudioEncoder, self).__init__()

        self.fc1 = nn.Linear(audio_dim, tied_output)

        self.pos_enc = PositionalEncoding(d_model = tied_output, max_len = audio_size, dropout = dropout)

        self.down_sampling = DownSampling(a=down_sampling_factor)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = d_feedforward, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = nlayer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.pos_enc(x)
        x = self.down_sampling(x)
        x = self.transformer_encoder(x)
        return x

class VideoEncoder(nn.Module):

    def __init__(self, video_dim, nhead, nlayer, d_model, d_feedforward, dropout):
        super(VideoEncoder, self).__init__()

        self.fc1 = nn.Linear(video_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = d_feedforward, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = nlayer)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.transformer_encoder(x)
        return x

class DotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v):
        super(DotProductAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.alpha = Variable(torch.rand(1), requires_grad=True)

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)


    def forward(self, q , k, v):

        d_k, d_v =  self.d_k, self.d_v

        sz_batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x dv

        q = self.w_qs(q).view(sz_batch, len_q, d_k)
        k = self.w_ks(k).view(sz_batch, len_k, d_k)
        v = self.w_vs(v).view(sz_batch, len_v, d_v)

        # Transpose for attention dot product: b x lq x dv

        attn = torch.matmul(q, k.transpose(1, 2))

        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        output = output* self.alpha + (1.0 - self.alpha) * residual

        return output

class DownSampling(nn.Module):

    def __init__(self, a: int):
        super().__init__()

        self.a = a # downsampling factor

    def forward(self, x):

        b_size, seq_len, feat_size = x.size() # batch, sequence length, feature dimensions
        if seq_len % self.a != 0:
            raise ValueError("Parameter a should be a divisor of sequence length.")

        return x.view(b_size, int(seq_len/self.a), feat_size*self.a)

