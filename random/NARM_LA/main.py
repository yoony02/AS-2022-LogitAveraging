#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019
Modified on January, 2022
@author: wangshuo
@modifier: heeyooon
"""

import os
import time
import argparse
import pickle

import torch

from utils import get_best_result, top75_labels, Data
from narm import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='diginetica/tmall/nowplaying')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--n_layers', type=int, default=1,)
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', action='store_true', help='test')
# parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--gpu_num', type=int, default=0, help='cuda number')
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--input_aug_type', type=str, default=None, help='insertion/deletion')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu_num)
if opt.save_model:
    os.makedirs(f'ckpt/{opt.dataset}', exist_ok=True)

def main():
    train_data = pickle.load(open(f'../../Dataset/{opt.dataset}/train.txt', 'rb'))
    test_data = pickle.load(open(f'../../Dataset/{opt.dataset}/test.txt', 'rb'))
    # n_items = pickle.load(open(f'../../Dataset/{opt.dataset}/num_items.txt', 'rb'))
    if 'yoochoose' in opt.dataset:
        n_items = 37484
    elif 'tmall' in opt.dataset:
        n_items = 40728
    elif 'diginetica' in opt.dataset:
        n_items = 43098
    elif 'retailrocket' in opt.dataset:
        n_items = 27413
    else:
        print("there's no dataset information")
        
    top_labels = top75_labels(train_data, test_data, opt.dataset)

    train_data = Data(train_data, opt.input_aug_type, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    model = trans_to_cuda(NARM(n_items, opt))

    start = time.time()
    best_results = [[0 for i in range(3)] for j in range(2)]
    best_epochs = [[0 for i in range(3)] for j in range(2)]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-' * 100)
        print('Epoch: ', epoch)
        loss, results = train_test(model, train_data, test_data, n_items, top_labels)

        flag = get_best_result(results, epoch, best_results, best_epochs)

        
        if flag > 0 :
            if opt.save_model == True:
                model_save_path = f'ckpt/{opt.dataset}/{epoch}.pt'
                torch.save(model.state_dict(), model_save_path)
                print("Model Saving Done")
        else:
            bad_counter += 1
            if bad_counter >= opt.patience:
                break

    print('-' * 100)
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
