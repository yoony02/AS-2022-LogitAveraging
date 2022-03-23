#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
Modified on January, 2022
@author: Tangrizzly
#modifier: heeyooon
"""

import networkx as nx
import numpy as np
import random
import pdb
import torch


def get_metric_scores(scores, targets, k, eval):
    # eval : hit, mrr, cov
    sub_scores = scores.topk(k)[1]
    sub_scores = sub_scores.cpu().detach().numpy()
    
    targets = targets.cpu().detach().numpy()
    
    cur_hits, cur_mrrs = [], []
    for score, target in zip(sub_scores, targets):

        # HR@K
        hit_temp = np.isin(target, score)
        cur_hits.append(hit_temp)

        # MRR@K    
        if len(np.where(score == target)[0]) == 0:
            mrr_temp = 0
        else:
            mrr_temp = 1 / (np.where(score == target)[0][0] + 1)
        cur_mrrs.append(mrr_temp)

    eval[0] += cur_hits
    eval[1] += cur_mrrs
    
    # Coverage@K
    eval[2] += np.unique(sub_scores).tolist()

    return eval


def metric_print(eval10, eval20, n_node, time):

    for evals in [eval10, eval20]:
        # hit, mrr, cov
        evals[0] = np.mean(evals[0]) * 100
        evals[1] = np.mean(evals[1]) * 100
        evals[2] = len(np.unique(evals[2])) / n_node * 100


    print('Metric\t\tHR@10\tMRR@10\tCov@10\tHR@20\tMRR@20\tCov@20')
    print(f'Value\t\t'+'\t'.join(format(eval, ".2f") for eval in eval10+eval20))

    print(f"Time elapse : {time}")
    return [eval10, eval20]


def get_best_result(results, epoch, best_results, best_epochs):
    # results: eval10, eval20
    # eval: HR, MRR, cov

    for result, best_result, best_epoch in zip(results, best_results, best_epochs):
        flag = 0
        for i in range(3):           
            if result[i] > best_result[i]:
                best_result[i] = result[i]
                best_epoch[i] = epoch
                flag = 1

    print("-"*100)
    print('Best Result\tHR@10\tMRR@10\tCov@10\tHR@20\tMRR@20\tCov@20\tEpochs')
    print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in best_results[0]+best_results[1])
          + '\t' + ', '.join(str(epoch) for epoch in best_epochs[0]+best_epochs[1]))

    return flag


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False):
        self.inputs = np.asarray(data[0], dtype=object)
        self.targets = np.asarray(data[1])
        self.length = len(self.inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, targets = self.inputs[i], self.targets[i]
        inputs_len = np.array([len(input) for input in inputs])

        sorted_idxs = np.argsort(inputs_len)[::-1]
        inputs = inputs[sorted_idxs]
        targets = targets[sorted_idxs]
        inputs_len = inputs_len[sorted_idxs]

        padded_sesss = torch.zeros(len(inputs), max(inputs_len)).long()
        for i, input in enumerate(inputs):
            padded_sesss[i,:inputs_len[i]] = torch.LongTensor(input)
            
        padded_sesss = padded_sesss.transpose(0, 1)
        return padded_sesss, torch.LongTensor(targets), inputs_len
        
        
