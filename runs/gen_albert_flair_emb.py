import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import json
import pickle
from tqdm import tqdm

from transformers import AlbertModel, AlbertTokenizer
from flair.embeddings import BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Token, Sentence

class AlbertEmbeddings(BertEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "albert-base-v2",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
    ):
        super().__init__()
        self.tokenizer = AlbertTokenizer.from_pretrained(bert_model_or_path)
        self.model = AlbertModel.from_pretrained(
            pretrained_model_name_or_path=bert_model_or_path,
            output_hidden_states=True,
        )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        s.add_token(Token(w))
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)


parser = argparse.ArgumentParser(description='Arguments for training.')

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)

parser.add_argument('--model_name',
                    default='albert-xxlarge-v2',
                    action='store',)

parser.add_argument('--flair_name',
                    default='news',
                    action='store',)

parser.add_argument('--lm_emb_save_path',
                    default='../wv/lm.emb.pkl',
                    action='store',)

args = parser.parse_args()


bert_embedding = AlbertEmbeddings(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
flair_embedding = StackedEmbeddings([
    FlairEmbeddings(f'{args.flair_name}-forward'),
    FlairEmbeddings(f'{args.flair_name}-backward'),
])


flag = args.dataset
dataset = []
with open(f'./datasets/unified/train.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/valid.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/test.{flag}.json') as f:
    dataset += json.load(f)
    
    
bert_emb_dict = {}
for item in tqdm(dataset):
    tokens = tuple(item['tokens'])
    s = form_sentence(tokens)
    
    s.clear_embeddings()
    bert_embedding.embed(s)
    emb = get_embs(s) # (T, 4*H)
        
    s.clear_embeddings()
    flair_embedding.embed(s)
    emb = np.concatenate([emb, get_embs(s)], axis=-1)
    
    bert_emb_dict[tokens] = emb.astype('float16')
    
    
with open(args.lm_emb_save_path, 'wb') as f:
    pickle.dump(bert_emb_dict, f)