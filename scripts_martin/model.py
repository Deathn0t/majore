import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class MultimodalTransformer(nn.Module):

    def __init__(self, a_dim, v_dim, d_model, num_encoder_layers, num_decoder_layers):
        super(MultimodalTransformer, self).__init__()
        
        self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, dim_feedforward=1920, dropout=0.2, activation='relu')
        self.audio_encoder = nn.TransformerEncoder(encoder_layer=self.audio_encoder_layer, num_layers=6)

        self.crossmodal = MultiheadAttention(embed_dim=d_model, num_heads=1)

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, dim_feedforward=1920, dropout=0.2, activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=4)

        self.W_q = nn.Parameter(np.random.rand((d_model, d_model))
        self.W_k = nn.Parameter(np.random.rand((v_dim, d_model))
        self.W_v = nn.Parameter(np.random.rand((v_dim, d_model))



    def forward(self, a_encoded, v_encoded, tgt, tgt_mask):

        a_encoded = self.audio_encoder.forward(a_encoded)

        K = torch.matmul(v_encoded, self.W_k)
        V = torch.matmul(v_encoded, self.W_v)
        Q = torch.matmul(a_encoded, self.W_q)

        mm_attention = nn.Softmax(torch.matmul(Q, torch.t(K)))
        mm_attention = torch.matmul(mm_attention, V)




