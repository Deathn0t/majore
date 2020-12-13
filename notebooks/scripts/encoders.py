import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .position_encoder import PositionalEncoding


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

        batch, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(seq_len, batch, dim)

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

        batch, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(seq_len, batch, dim)

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

        #IN
        #query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
        # the embedding dimension.
        #- key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          #the embedding dimension.
        #- value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          #the embedding dimension.

        #OUT
        #attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
        #E is the embedding dimension.

        # IN OUR CASE
        #torch.Size([1081, 10, 480])
        #torch.Size([1, 10, 480])
        #torch.Size([1, 10, 480])

        d_k, d_v =  self.d_k, self.d_v

        len_q, sz_batch, emb_dim =  q.size(0), q.size(1), q.size(2)

        residual = q

        q = self.w_qs(q).view(len_q, sz_batch, d_k).transpose(0,1)
        k = self.w_ks(k).view(-1, sz_batch, d_k).transpose(0,1)
        v = self.w_vs(v).view(-1, sz_batch, d_v).transpose(0,1)

        #torch.Size([10, 1081 480])
        #torch.Size([10,1, 480])
        #torch.Size([10,1, 480])

        attn = torch.matmul(q, k.transpose(1,2))

        #torch.Size([10, 1081, 1])

        attn = F.softmax(attn, dim=-1)

        #torch.Size([10, 1081, 1])

        output = torch.matmul(attn, v)

        #torch.Size([10, 1081, 480])

        output = output.transpose(0, 1).contiguous().view(len_q, sz_batch, emb_dim)

        #torch.Size([1081, 10, 480])

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

