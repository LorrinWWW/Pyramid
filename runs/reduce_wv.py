import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm

import json
import argparse

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--dataset',
                    action='store',)

parser.add_argument('--read_wv_path',
                    action='store',)

parser.add_argument('--save_wv_path',
                    action='store',)

parser.add_argument('--cased', default='0',
                    action='store',)

args = parser.parse_args()


flag = args.dataset
with open(f'./datasets/unified/train.{flag}.json') as f:
    train = json.load(f)
    
with open(f'./datasets/unified/valid.{flag}.json') as f:
    valid = json.load(f)
    
with open(f'./datasets/unified/test.{flag}.json') as f:
    test = json.load(f)
    
    
    
dataset = train + valid + test

vocab = set()
for item in dataset:
    tokens = item['tokens']
    if int(args.cased) == 0:
        tokens = [t.lower() for t in tokens]
    vocab.update(tokens)
    
with open(args.read_wv_path) as fin, \
    open(args.save_wv_path, 'w') as fout:
    
    for line in tqdm(fin):
        w = line.split(' ')[0]
        if w in vocab:
            fout.write(line)