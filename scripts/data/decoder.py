import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from position_encoder import PositionalEncoding



class MultimodalDecoder(nn.Module):

    def __init__(self, text_dim, text_size, vocab_size, nhead, nlayer, d_model, d_feedforward, dropout):
        super(MultimodalDecoder, self).__init__()
        
        self.fc1 = nn.Linear(text_dim, d_model)
        
        self.pos_enc = PositionalEncoding(d_model = d_model, max_len = text_size, dropout = dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = d_feedforward, dropout = dropout)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = nlayer)
        
        self.fc2 = nn.Linear(d_model, vocab_size)
    

    def forward(self, text, encoder_output, tgt_mask):
        
        x = F.relu(self.fc1(text))
        
        x = self.pos_enc(x)
        
        batch_size, len_emb, dim_emb = x.shape[0], x.shape[1], x.shape[2]
        
        # Target must be (len, batch_sise, dim_emb) see doc....
        x = x.view(len_emb, batch_size, dim_emb)
        
        x = self.transformer_decoder(x, encoder_output, tgt_mask = tgt_mask)
        
        x = F.softmax(self.fc2(x))
        
        return x
    
   
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
