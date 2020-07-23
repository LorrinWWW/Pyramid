
import math
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from transformers import *

from utils import *
from functions import *

from layers.indexings import *


class PreEmbeddedLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.device = config.device
        self.lm_emb_path = config.lm_emb_path
        
        with open(self.lm_emb_path, 'rb') as f:
            self.emb_dict = pickle.load(f)
            
    def forward(self, batch_tokens):
        
        embs = [self.emb_tokens(tokens) for tokens in batch_tokens]
        #embs = np.stack(embs, axis=0) # (B, T, H)
        embs_padded = pad_sequences(embs, maxlen=None, dtype='float32',
                  padding='post', truncating='post', value=0.)
        embs_padded = torch.from_numpy(embs_padded).float()
        
        mask = torch.zeros(embs_padded.shape[:2]).bool()
        for i, emb in enumerate(embs):
            mask[i, :len(emb)] = True
            
        embs_padded = embs_padded.to(self.device)
        mask = mask.to(self.device)
        
        return embs_padded, mask
    
    def emb_tokens(self, tokens):
    
        tokens = tuple(tokens)
        
        if tokens not in self.emb_dict:
            raise Exception(f'{tokens} not pre-emb')
            
        return self.emb_dict[tokens]
    

class BERTEmbedding(nn.Module):
    
    def __init__(self, ckpt_name='bert-base-uncased'):
        super().__init__()
        
#         self.config = config
#         self.device = config.device
        self.ckpt_name = ckpt_name
        self.model = BertModel.from_pretrained(ckpt_name)
        self.tokenizer = BertTokenizer.from_pretrained(ckpt_name)
        
    def preprocess_sentences(self, sentences):
        
        if len(sentences) > 1 and isinstance(sentences[1], torch.Tensor):
            # sentences is Tensor or is a list/tuple of Tensor
            return sentences #[(x.to(self.device) if isinstance(x, torch.Tensor) else x) for x in sentences]
        
        if 'uncased' in self.ckpt_name:
            sentences = [[w.lower() for w in s] for s in sentences]
        
        idxs = [self.tokenizer.convert_tokens_to_ids(s) for s in sentences]
        idxs = pad_sequences(
            idxs, maxlen=None, dtype='int64',
            padding='post', truncating='post', value=0.)
        
        idxs = torch.from_numpy(idxs)
        
        return [sentences, idxs]
        
    def forward(self, sentences):
        ret = self.model(sentences)
#         print(ret[0].shape, ret[1].shape)
        return ret[0]
    
    
class LMAllEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.lm_embedding = BERTEmbedding(ckpt_name='bert-large-uncased')
        self.masking = Masking()
        
    def load_pretrained(self, path, freeze=True):
        pass
    
    def preprocess_sentences(self, sentences):
        return self.lm_embedding.preprocess_sentences(sentences)
    
    def forward(self, sentences):
        sentences, t_indexs = self.preprocess_sentences(sentences)
        t_indexs = t_indexs.to(self.config.device)
        masks = self.masking(t_indexs, mask_val=0)
        return self.lm_embedding(t_indexs), masks
        
        
        