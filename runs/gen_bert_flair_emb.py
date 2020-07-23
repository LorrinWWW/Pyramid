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

from flair.embeddings import BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Token, Sentence

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
                    default='bert-base-multilingual-cased',
                    action='store',)

parser.add_argument('--flair_name',
                    default='news',
                    action='store',)

parser.add_argument('--lm_emb_save_path',
                    default='../wv/lm.emb.pkl',
                    action='store',)

args = parser.parse_args()


bert_embedding = BertEmbeddings(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
flair_embedding = StackedEmbeddings([
    FlairEmbeddings(f'{args.flair_name}-forward'),
    FlairEmbeddings(f'{args.flair_name}-backward'),
])

if 'pubmed' in args.model_name.lower():
    bert_embedding.tokenizer.basic_tokenizer.do_lower_case = False


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