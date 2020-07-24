import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils import *
from data import *
from models import *


import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--model_class',
                    default='SeqProto',
                    action='store',)

parser.add_argument('--model_read_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--model_write_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--pretrained_wv',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)

parser.add_argument('--max_depth',
                    default=None, type=none_or_int,
                    action='store',)

parser.add_argument('--batch_size',
                    default=32, type=int,
                    action='store',)

parser.add_argument('--evaluate_interval',
                    default=1000, type=int,
                    action='store',)

parser.add_argument('--max_steps',
                    default=int(1e9), type=int,
                    action='store')

parser.add_argument('--max_epoches',
                    default=100, type=int,
                    action='store')

parser.add_argument('--decay_rate',
                    default=0.05, type=float,
                    action='store')

#### Model Config
parser.add_argument('--token_emb_dim',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--char_encoder',
                    default='lstm',
                    action='store',)

parser.add_argument('--char_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--cased',
                    default=False, type=int,
                    action='store',)

parser.add_argument('--hidden_dim',
                    default=200, type=int,
                    action='store',)

parser.add_argument('--loss_reduction',
                    default='sum',
                    action='store',)

parser.add_argument('--maxlen',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--dropout',
                    default=0.3, type=float,
                    action='store',)

parser.add_argument('--optimizer',
                    default='sgd',
                    action='store',)

parser.add_argument('--lr',
                    default=0.01, type=float,
                    action='store',)

parser.add_argument('--vocab_size',
                    default=20000, type=int,
                    action='store',)

parser.add_argument('--vocab_file',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--tag_vocab_size',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--tag_form',
                    default='iob2',
                    action='store',)

parser.add_argument('--freeze_wv',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--lm_emb_path',
                    default=None,
                    action='store',)

parser.add_argument('--lm_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--device',
                    default=None, type=none_or_str,
                    action='store',)


args = parser.parse_args()


if args.device is not None:
    torch.cuda.set_device(args.device)
else:
    gpu_idx, gpu_mem = set_max_available_gpu()
    args.device = f"cuda:{gpu_idx}"

# Model
config = Config(**args.__dict__)
ModelClass = eval(args.model_class)
model = ModelClass(config)

# load weight
if args.model_read_ckpt:
    print(f"reading params from {args.model_read_ckpt}")
    model.load(args.model_read_ckpt)
    model.token_embedding.token_indexing.update_vocab = False
elif args.pretrained_wv:
    print(f"reading pretrained wv from {args.pretrained_wv}")
    model.token_embedding.load_pretrained(args.pretrained_wv, freeze=args.freeze_wv)
    model.token_embedding.token_indexing.update_vocab = False
    
# dataset
print("reading data..")
Trainer = model.get_default_trainer_class()
flag = args.dataset
trainer = Trainer(
    model=model,
    train_path=f'./datasets/unified/train.{flag}.json',
    test_path=f'./datasets/unified/test.{flag}.json',
    valid_path=f'./datasets/unified/valid.{flag}.json',
    batch_size=int(args.batch_size),
    tag_form=args.tag_form, num_workers=1,
    max_depth=args.max_depth,
)


print("=== start training ===")
trainer.train_model(args=args)




