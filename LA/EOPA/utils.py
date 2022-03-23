from torch.optim import lr_scheduler
import torch
import numpy as np
import logging
import os
from datetime import datetime
import sys

import itertools
import os

import numpy as np
import pandas as pd
import random
import pickle
from collections import Counter


def top75_labels(train_sessions, test_sessions, dataset_dir):
    try:
        with open(dataset_dir / f'top75_labels.pickle', 'rb') as f:
            top_labels = pickle.load(f)
    except:
        train_sesses = train_sessions.tolist()
        test_sesses = test_sessions.tolist()
        sesses = train_sesses + test_sesses
        labels = [sess[-1] for sess in sesses]

        target_cnt_dict = Counter(labels)
        target_dict_sorted = sorted(target_cnt_dict.items(), reverse=True, key=lambda item: item[1])
        target_dict_keys = [item[0] for item in target_dict_sorted]
        target_dict_values = [item[1] for item in target_dict_sorted]

        occr_cum_sum = np.cumsum(np.array(target_dict_values))
        split_point = int(round(len(labels) * 0.75))
        split_idx = np.sum(occr_cum_sum < split_point)

        top_labels = target_dict_keys[:split_idx]

        with open(dataset_dir / f'top75_labels.pickle', 'wb') as f:
            pickle.dump(top_labels, f, pickle.HIGHEST_PROTOCOL)

    return top_labels

def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype = np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype = np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header = None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())

    return train_sessions, test_sessions, num_items


class Dataset:
    def __init__(self, sessions, sort_by_length=True):
        self.sessions = sessions
        index = create_index(sessions)
        if sort_by_length:
            # sort by label Index in descending order (label means length)
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        return seq, label

    def __len__(self):
        return len(self.index)


def get_metric_scores(scores, targets, k, eval):
    sub_scores = scores.topk(k)[1]
    sub_scores = sub_scores.cpu().detach().numpy()
    targets = targets.numpy()

    cur_hits, cur_mrrs = [], []
    for score, target in zip(sub_scores, targets):
        # hit@K
        hit_temp = np.isin(target, score)
        cur_hits.append(hit_temp)
        
        # MRR@K
        if len(np.where(score == target)[0]) == 0:
            mrr_temp = 0
        else:
            mrr_temp = 1 / (np.where(score == target)[0][0]+1)
        cur_mrrs.append(mrr_temp)
    
    

    eval[0] += cur_hits
    eval[1] += cur_mrrs
    
    # Coverage@K
    eval[2] += np.unique(sub_scores).tolist()

    return eval


def metric_print(eval10, eval20, n_items, time):
    for evals in [eval10, eval20]:
        # hit, mrr, cov
        evals[0] = np.mean(evals[0]) * 100
        evals[1] = np.mean(evals[1]) * 100
        evals[2] = len(np.unique(evals[2])) / n_items * 100
    
    print('Metric\t\tHR@10\tMRR@10\tCov@10\tHR@20\tMRR@20\tCov@20')
    print(f'Value\t\t'+'\t'.join(format(eval, ".2f") for eval in eval10+eval20))

    print(f"Time elapse : {time}")
    return [eval10, eval20]


def get_best_results(results, epoch, best_results, best_epochs):
    # results: eval10, eval20
    # eval: HR, MRR, cov

    for result, best_result, best_epoch in zip(results, best_results, best_epochs):
        flag = 0
        for i in range(3):           
            if result[i] > best_result[i]:
                best_result[i] = result[i]
                best_epoch[i] = epoch
                flag += 1

    print("-"*100)
    print('Best Result\tHR@10\tMRR@10\tCov@10\tHR@20\tMRR@20\tCov@20\tEpochs')
    print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in best_results[0]+best_results[1])
          + '\t' + ', '.join(str(epoch) for epoch in best_epochs[0]+best_epochs[1]))

    return flag


def fix_weight_decay(model):
    # ignore weight decay for parameters in bias, batch norm and activation
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params

class WarmupCosineLrScheduler(lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
            #ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio