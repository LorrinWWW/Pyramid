
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from utils import *
from functions import *
        
        
class TransformerEncoding(nn.Module):
    
    def __init__(self, config, nhead=4, num_layers=2, norm_output=True):
        super().__init__()
        self.config = config
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_dim, nhead=nhead)
        if norm_output:
            norm = nn.LayerNorm(self.config.hidden_dim)
        else:
            norm = None
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        
    def forward(self, inputs, mask=None, attn_mask=None):
        inputs = inputs.permute(1,0,2)
        src_key_padding_mask = None if mask is None else ~mask
        outputs = self.attn(inputs, src_key_padding_mask=src_key_padding_mask, mask=attn_mask)
        outputs = outputs.permute(1,0,2)
        return outputs
    
class TransformerDecoding(nn.Module):
    
    def __init__(self, config, nhead=4, num_layers=2, norm_output=True):
        super().__init__()
        self.config = config
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_dim, nhead=nhead)
        if norm_output:
            norm = nn.LayerNorm(self.config.hidden_dim)
        else:
            norm = None
        self.attn = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        
    def forward(self, tgt_inputs, memory_inputs, tgt_mask=None, memory_mask=None):
        tgt_inputs = tgt_inputs.permute(1,0,2)
        memory_inputs = memory_inputs.permute(1,0,2)
        tgt_key_padding_mask = None if tgt_mask is None else ~tgt_mask
        memory_key_padding_mask = None if memory_mask is None else ~memory_mask
        outputs = self.attn(
            tgt_inputs, memory_inputs,
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask)
        outputs = outputs.permute(1,0,2)
        return outputs

    
class AttentionEncoding(nn.Module):
    ''' n to 1 '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = nn.Linear(config.hidden_dim, 1)
        init_linear(self.attention)
        
    def forward(self, inputs, mask=None):
        a = self.attention(inputs) # (B, T, H) => (B, T, 1)
        if mask is not None:
            a -= 999*(~mask).float()[:, :, None]
        a = F.softmax(a, dim=1) # (B, T, 1)
        outputs = (a*inputs).sum(1) # (B, H)
        return outputs