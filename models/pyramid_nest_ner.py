import os, sys, pickle

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from layers import *
from functions import *
from data import *

from .base import *
from .basic_taggers import *

import copy


class PyramidNestNER(LSTMTagger):
    
    def set_embedding_layer(self):
        
        if self.config.tag_form.lower() == 'iob2':
            self.one_entity_n_tags = 2
        elif self.config.tag_form.lower() == 'iobes':
            self.one_entity_n_tags = 4
        else:
            raise Exception('no such tag form.')
        self.pyramid_tag_indexing = PyramidNestIndexing(self.config)
        
        self.token_embedding = AllEmbedding(self.config)
        self.token_indexing = self.token_embedding.preprocess_sentences
    
    def set_encoding_layer(self):
        
        emb_dim = self.config.token_emb_dim + self.config.char_emb_dim
        
        self.sentence_encoding = LSTMEncoding(self.config, emb_dim)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
        self.combine_layer = nn.Linear(self.config.hidden_dim*2, self.config.hidden_dim)
        init_linear(self.combine_layer)
#         self.combine_layer = NGramEncoding(self.config, ngram=2)
        
        self.reuse_decoding = LSTMEncoding(self.config)
        self.max_depth = self.config.max_depth
        
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        
        self.reduce_dim = nn.Linear(self.config.hidden_dim+self.config.lm_emb_dim, self.config.hidden_dim)
        init_linear(self.reduce_dim)
        
    def set_loss_layer(self):
        
        self.loss_layer = nn.CrossEntropyLoss(reduction='none')
        
    def check_attrs(self):
        # indexing
        assert hasattr(self, 'pyramid_tag_indexing')
        assert hasattr(self, 'token_indexing')
        
    def get_default_trainer_class(self):
        return PyramidNestNERTrainer
        
    def forward(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor], #(token idx, char idx)
            'tags': List(List(str)),
            '_tags'(*): Tensor,
        }
        outputs: +{
            'loss': Tensor,
        }
        '''

        rets = self.forward_step(inputs)
        logits_list = rets['logits_list']
        mask_list = rets['mask_list']
        
        if '_labels' in rets:
            labels = [x.to(self.device) for x in rets['_labels']]
        else:
            labels = [x.to(self.device) for x in self.pyramid_tag_indexing(rets['labels'])]
            
        loss = 0
        for i, (logits, tags, mask) in enumerate(zip(logits_list, labels, mask_list)):
            loss_tensor = self.loss_layer(logits.permute(0, -1, 1), tags) # (B, T)
            loss += (loss_tensor * mask.float()).sum()# * (1/(i+1.) + 0.5)

        rets['_labels'] = labels
        rets['loss'] = loss
            
        return rets
    
    def forward_step(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor],
        }
        outputs: +{
            'logits_list': List(Tensor),
            'mask_list': List(Tensor),
        }
        '''
        
        if '_tokens' in inputs:
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']
            
        embeddings_list, masks = self.token_embedding(sents, return_list=True)
        if self.config.lm_emb_dim > 0:
            embeddings = torch.cat(embeddings_list[:-1], dim=-1)
        else:
            embeddings = torch.cat(embeddings_list, dim=-1)
        embeddings = self.dropout_layer(embeddings)
        embeddings = self.sentence_encoding(embeddings)
        if self.config.lm_emb_dim > 0:
            # add lm embedding to the output of lstm encoder
            embeddings = torch.cat([embeddings, embeddings_list[-1]], dim=-1)
            embeddings = self.dropout_layer(embeddings)
            embeddings = self.reduce_dim(embeddings)
        
        B, T, H = embeddings.shape
        
        logits_list = []
        mask_list = []
        
        max_depth = self.max_depth if self.max_depth is not None else embeddings.shape[1]-1
        for i in range(max_depth + 1):
            
            if i == 0:
                mask = masks
                mask_list.append(mask)
            else:
                if embeddings.shape[1] == 1:
                    break
                    
                embeddings = torch.cat([embeddings[:, :-1], embeddings[:, 1:]], dim=-1) # (B, T, 2*H)
                embeddings = self.combine_layer(embeddings) # (B, T, H)
                
                mask = masks[:, i:]
                mask_list.append(mask)
            
            embeddings = self.norm(embeddings)
            embeddings = self.dropout_layer(embeddings)
            
            embeddings = self.reuse_decoding(embeddings)

            embeddings = self.dropout_layer(embeddings)
            logits_hat = self.logits_layer(embeddings)
    
            logits_list.append(logits_hat)
        
        rets = inputs
        rets['logits_list'] = logits_list
        rets['mask_list'] = mask_list
        
        return rets
    
    
    def predict_step(self, inputs):
        
        rets = self.forward_step(inputs)
        logits_list = rets['logits_list']
        mask_list = rets['mask_list']
        
        preds_list = [logits.argmax(dim=-1) for logits in logits_list if logits.shape[1]!=0]
        preds_list = [p * m.long() for p, m in zip(preds_list, mask_list)]
        preds_list = [self.pyramid_tag_indexing.inv(preds.cpu().detach().numpy()) for preds in preds_list]
        
        rets['preds_list'] = preds_list
        
        pred_set = [set() for _ in range(logits_list[0].shape[0])] 
        for depth, preds_batch in enumerate(preds_list):
            for b, preds in enumerate(preds_batch):
                for (start, end), _type in zip(*tag2span(preds, True)):
                    for _type_splited in _type.split('|'):
                        pred_set[b].add((_type_splited, start, end+depth))
                
        rets['pred_set'] = pred_set
            
        return rets
    
    
    def save_ckpt(self, path):
        torch.save(self.state_dict(), path+'.pt')
        with open(path+'.vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.token_indexing.vocab, f)
        with open(path+'.char_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.char_indexing.vocab, f)
        with open(path+'.tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.pyramid_tag_indexing.tag_indexing.vocab, f)
            
    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path+'.pt'))
        with open(path+'.vocab.pkl', 'rb') as f:
            self.token_embedding.token_indexing.vocab = pickle.load(f)
            self.token_embedding.token_indexing.update_inv_vocab()
        with open(path+'.char_vocab.pkl', 'rb') as f:
            self.token_embedding.char_indexing.vocab = pickle.load(f)
            self.token_embedding.char_indexing.update_inv_vocab()
        with open(path+'.tag_vocab.pkl', 'rb') as f:
            self.pyramid_tag_indexing.tag_indexing.vocab = pickle.load(f)
            self.pyramid_tag_indexing.tag_indexing.update_inv_vocab()
    
    
class BiPyramidNestNER(PyramidNestNER):
        
    def before_init(self):
        
        self.split_layer = NGramEncoding(self.config, self.config.hidden_dim*2, ngram=2, padding=1)
        
    def set_logits_layer(self):
        
        self.logits_layer = nn.Linear(self.config.hidden_dim*2, self.config.tag_vocab_size)
        init_linear(self.logits_layer)
        
    def forward_step(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor],
        }
        outputs: +{
            'logits_list': List(Tensor),
            'mask_list': List(Tensor),
        }
        '''
        
        if '_tokens' in inputs:
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']
            
        embeddings_list, masks = self.token_embedding(sents, return_list=True)
        if self.config.lm_emb_dim > 0:
            embeddings = torch.cat(embeddings_list[:-1], dim=-1)
        else:
            embeddings = torch.cat(embeddings_list, dim=-1)
        embeddings = self.dropout_layer(embeddings)
        embeddings = self.sentence_encoding(embeddings)
        if self.config.lm_emb_dim > 0:
            # add lm embedding to the output of lstm encoder
            embeddings = torch.cat([embeddings, embeddings_list[-1]], dim=-1)
            embeddings = self.dropout_layer(embeddings)
            embeddings = self.reduce_dim(embeddings)
        
        B, T, H = embeddings.shape
        
        embeddings_list = []
        embeddings_list_inv = []
        mask_list = []
        
        max_depth = self.max_depth if self.max_depth is not None else embeddings.shape[1]-1
        for i in range(max_depth + 1):
            
            if i == 0:
                mask = masks
                mask_list.append(mask)
            else:
                if embeddings.shape[1] == 1:
                    max_depth = i - 1 # reduce the max_depth if the sentence is too short
                    break
                    
                embeddings = torch.cat([embeddings[:, :-1], embeddings[:, 1:]], dim=-1) # (B, T, 2*H)
                embeddings = self.combine_layer(embeddings) # (B, T, H)
                
                mask = masks[:, i:]
                mask_list.append(mask)
            
            embeddings = self.norm(embeddings)
            embeddings = self.dropout_layer(embeddings)
            
            embeddings = self.reuse_decoding(embeddings)

            embeddings = self.dropout_layer(embeddings)
    
            embeddings_list.append(embeddings)
        
        for i in range(max_depth, -1, -1):
            
            if i == max_depth:
                
                embeddings = torch.zeros_like(embeddings)
                embeddings_list_inv.append(embeddings)
                
                continue
                
            else:
                embeddings = self.split_layer(torch.cat([
                    embeddings, embeddings_list[i+1]
                ], dim=-1))
                
            embeddings = self.norm(embeddings)
            embeddings = self.dropout_layer(embeddings)
                
            embeddings = self.reuse_decoding(embeddings)
            
            embeddings = self.dropout_layer(embeddings)
            
            embeddings_list_inv.append(embeddings)
            
        logits_list = []
        
        for i in range(max_depth + 1):
            
            logits_list.append(
                self.logits_layer(
                    torch.cat([embeddings_list[i], embeddings_list_inv[-i-1]], dim=-1)
                )
            )
            
        rets = inputs
        rets['logits_list'] = logits_list
        rets['mask_list'] = mask_list
        
        return rets
    
