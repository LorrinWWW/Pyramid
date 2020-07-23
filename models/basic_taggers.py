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

import copy

class LSTMTagger(Tagger):
    
    def set_embedding_layer(self):
        
        if self.config.tag_form.lower() == 'iob2':
            self.one_entity_n_tags = 2
        elif self.config.tag_form.lower() == 'iobes':
            self.one_entity_n_tags = 4
        else:
            raise Exception('no such tag form.')
        self.tag_indexing = get_tag_indexing(self.config)
        
        self.token_embedding = AllEmbedding(self.config)
        self.token_indexing = self.token_embedding.preprocess_sentences
        
    def set_encoding_layer(self):
        
        emb_dim = self.config.token_emb_dim + self.config.char_emb_dim
        self.sentence_encoding = LSTMEncoding(self.config, emb_dim)
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
    def set_logits_layer(self):
        
        self.logits_layer = nn.Linear(self.config.hidden_dim, self.config.tag_vocab_size)
        init_linear(self.logits_layer)
        
    def set_loss_layer(self):
        
        if self.config.crf:
            self.crf_layer = eval(self.config.crf)(self.config)
        else:
            self.loss_layer = nn.CrossEntropyLoss(reduction=self.config.loss_reduction)
            
    def get_default_trainer_class(self):
        return SlotFillingTrainer
            
    ####
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
        logits = rets['logits']
        mask = rets['masks']
        if hasattr(rets, '_tags'):
            tags = rets['_tags'].to(self.device)
        else:
            tags = self.tag_indexing(rets['tags']).to(self.device)
        
        if self.config.crf == 'CRF' or self.config.crf == 'DTCRF':
            loss = - self.crf_layer(logits, tags, mask=mask, reduction=self.config.loss_reduction)
        elif not self.config.crf:
            loss = self.loss_layer(logits.permute(0, 2, 1), tags)
        else:
            raise Exception('not a compatible loss')

        rets['_tags'] = tags
        rets['loss'] = loss
            
        return rets
    
    def forward_step(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor],
        }
        outputs: +{
            'logits': Tensor,
            'masks': Tensor
        }
        '''
        
        if hasattr(inputs, '_tokens'):
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']
            
        embeddings, masks = self.token_embedding(sents)
        embeddings = self.dropout_layer(embeddings)
        embeddings = self.sentence_encoding(embeddings)
        embeddings = self.dropout_layer(embeddings)
        logits = self.logits_layer(embeddings)
        
        rets = inputs
        rets['logits'] = logits
        rets['masks'] = masks
        
        return rets
    
    def predict_step(self, inputs):
        
        rets = self.forward_step(inputs)
        logits = rets['logits']
        mask = rets['masks']
        
        if self.config.crf == 'CRF' or self.config.crf == 'DTCRF':
            preds = self.crf_layer.decode(logits, mask=mask)
        elif not self.config.crf:
            preds = logits.argmax(dim=-1).cpu().detach().numpy()
        else:
            raise Exception('not a compatible decode')
            
        preds = np.array(preds)
        preds = self.tag_indexing.inv(preds)
        
        rets['preds'] = preds
            
        return rets
    
    def train_step(self, inputs):
        
        rets = self(inputs)
        loss = rets['loss']
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        
        return rets
        
    def save_ckpt(self, path):
        torch.save(self.state_dict(), path+'.pt')
        with open(path+'.vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.token_indexing.vocab, f)
        with open(path+'.char_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.char_indexing.vocab, f)
        with open(path+'.tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.tag_indexing.vocab, f)
            
    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path+'.pt'))
        with open(path+'.vocab.pkl', 'rb') as f:
            self.token_embedding.token_indexing.vocab = pickle.load(f)
            self.token_embedding.token_indexing.update_inv_vocab()
        with open(path+'.char_vocab.pkl', 'rb') as f:
            self.token_embedding.char_indexing.vocab = pickle.load(f)
            self.token_embedding.char_indexing.update_inv_vocab()
        with open(path+'.tag_vocab.pkl', 'rb') as f:
            self.tag_indexing.vocab = pickle.load(f)
            self.tag_indexing.update_inv_vocab()
        
        
        