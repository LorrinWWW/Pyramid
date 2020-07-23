

import os, sys
import numpy as np
import torch
import six
import json
import random
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import *
from itertools import combinations 

from .basics import *
from .base import *


class PyramidNestNERDataLoader(DataLoader):
    
    def __init__(self, json_path, 
                 model=None, num_workers=0, tag_form='iob2', 
                 skip_empty=False, max_depth=None, *args, **kargs):
        self.model = model
        self.num_workers = num_workers
        self.max_depth = max_depth
        self.dataset = SimpleJsonDataset(json_path)
        super().__init__(dataset=self.dataset, collate_fn=self._collect_fn, num_workers=num_workers, *args, **kargs)
        
        if self.num_workers == 0:
            pass # does not need warm indexing
        elif self.model is not None:
            print("warm indexing...")
            tmp = self.num_workers
            self.num_workers = 0
            for batch in self:
                pass
            self.num_workers = tmp
        else:
            print("warn: model is not set, skip warming.")
            print("note that if num_worker>0, vocab will be reset after each batch step,")
            print("thus a warming of indexing is required!")
            
    def add_entities_to_tags(self, tags, entities, depth):
        span2entities = defaultdict(set)
        for entity in entities:
            start, end = entity['span'][0], entity['span'][1] - depth
            etype = entity['entity_type']
            span2entities[(start, end)].add(etype)
        
        for (start, end), etypes in span2entities.items():
            etype = '|'.join(sorted(list(etypes)))
            for i in range(start+1, end):
                tags[i] = f'I-{etype}'
            tags[start] = f'B-{etype}'
        return tags
            
    def _normalize_nested_labels(self, entities, length, max_depth):
        tmp = defaultdict(list)
        for entity in entities:
            if entity['span'][1] <= length:
                tmp[entity['span'][1] - entity['span'][0] - 1].append(entity)
            else:
                print(f'entity exceeds the given length: {entity}')
            
        ret = [['O']*(length-depth) for depth in range(max_depth+1)]
        for depth in range(max([max_depth, *tmp.keys()])+1):
            ents = tmp.get(depth, [])
            depth = min(depth, max_depth)
            ret[depth] = self.add_entities_to_tags(ret[depth], ents, depth)
        
        return ret
        
    def _collect_fn(self, batch):
        tokens, labels, entities = [], [], []
        max_depth = self.max_depth if self.max_depth is not None else max(len(item['tokens']) for item in batch)-1
        for item in batch:
            _tokens = item['tokens'] if self.model is None else item['tokens']#[:150] # TODO: temporally limit maxlen
            tokens.append(_tokens)
            labels.append(self._normalize_nested_labels(
                item['entities'], length=len(_tokens), max_depth=max_depth))
            entities.append(item['entities'])
        
        rets = {
            'tokens': tokens,
            'labels': labels,
            'original_entities': entities,
        }
        
        if self.model is not None:
            tokens = self.model.token_indexing(tokens) # (B, T)
            labels = self.model.pyramid_tag_indexing(labels)
        
            rets['_tokens'] = tokens
            rets['_labels'] = labels
        
        return rets
    
    
class PyramidNestNERTrainer(Trainer):
    def __init__(self, train_path, test_path, valid_path,
                 batch_size=128, shuffle=True, model=None, num_workers=0, tag_form='iobes', 
                 max_depth=None,
                 *args, **kargs):
        self.batch_size = batch_size
        self.model = model
        self.train = PyramidNestNERDataLoader(
            train_path, model=model, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, tag_form=tag_form, max_depth=max_depth)
        self.test = PyramidNestNERDataLoader(
            test_path, model=model, batch_size=batch_size, num_workers=num_workers, 
            tag_form=tag_form, max_depth=max_depth)
        self.valid = PyramidNestNERDataLoader(
            valid_path, model=model, batch_size=batch_size, num_workers=num_workers, 
            tag_form=tag_form, max_depth=max_depth)
        
    def get_metrics(self, sents, pred_set_list, entities, verbose=0):
        
        assert len(pred_set_list) == len(entities)
        n_recall = n_pred = n_correct = 0
        for b in range(len(entities)):
            _entities = entities[b]

            _preds_set = pred_set_list[b]

            _labels_set = {
                (e['entity_type'], *e['span']) for e in _entities
            }

            n_recall += len(_labels_set)
            n_pred += len(_preds_set)
            n_correct += len(_labels_set & _preds_set)
            
        rec = n_correct / (n_recall + 1e-8)
        prec = n_correct / (n_pred + 1e-8)
        f1 = 2 / (1/(rec+1e-8) + 1/(prec+1e-8))
        return {  
            'precision' : prec,
            'recall' : rec,
            'f1' : f1,
            'confusion_dict' : None,
            'sents': sents,
            'pred_set_list': pred_set_list,
            'entities': entities,
        }
        
    
    def evaluate_model(self, model=None, verbose=0, test_type='valid'):
        
        if model is None:
            model = self.model
        
        if test_type == 'valid':
            g = self.valid
        elif test_type == 'test':
            g = self.test
        else:
            g = []
            
        sents = []
        pred_set_list = []
        entities = []
        for i, inputs in enumerate(g):
            rets = model.predict_step(inputs)
            pred_set_list += list(rets['pred_set'])
            entities += inputs['original_entities']
            sents += inputs['tokens']
        
        return self.get_metrics(sents=sents, pred_set_list=pred_set_list, entities=entities, verbose=verbose)
    
    
    def _evaluate_during_train(self, model=None, trainer_target=None, args=None):
        
        if not hasattr(self, 'max_f1'):
            self.max_f1 = 0.0
        
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='test')
        precision, recall, f1, confusion_dict = rets['precision'], rets['recall'], rets['f1'], rets['confusion_dict']
        print(f">> test prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='valid')
        precision, recall, f1, confusion_dict = rets['precision'], rets['recall'], rets['f1'], rets['confusion_dict']
        print(f">> valid prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        if f1 > self.max_f1:
            self.max_f1 = f1
            print('new max f1 on valid!')
            if args.model_write_ckpt:
                model.save(args.model_write_ckpt)
                
