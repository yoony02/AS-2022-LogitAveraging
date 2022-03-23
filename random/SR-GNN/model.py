#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

from utils import get_metric_scores, metric_print





class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def session_encoding(self, hidden, alias_inputs, mask):
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        return a

    def compute_scores(self, a):
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def find_mixup_srcs(tail_idxs, overlap_A, batch_size):
    with torch.no_grad():
    # construct tail mask
        tail_masks = torch.zeros(batch_size)
        tail_masks[tail_idxs] = 1

        # eliminate self loop
        A_hat2 = torch.FloatTensor(overlap_A) - torch.eye(batch_size)
        tails_overlap = tail_masks.view(-1, 1) * A_hat2

        # select nonzero max weight session
        mixup_sess_idxs = torch.max(tails_overlap, axis=1)[1]
        srcs = torch.nonzero(mixup_sess_idxs)
        mixup_sess_srcs1 = torch.cat([srcs, mixup_sess_idxs[srcs]], axis=1)

        # sample random sessions for others
        rand_srcs = torch.LongTensor(np.setdiff1d(tail_idxs, srcs.tolist()))
        mixup_sess_idxs = torch.randint(high=batch_size, size=(len(rand_srcs),))
        mixup_sess_srcs2 = torch.cat([rand_srcs.view(-1, 1), mixup_sess_idxs.view(-1, 1)], axis=1)

        # concat two kinds of mixup idxs
        srcs_mixup_sess = torch.vstack([mixup_sess_srcs1, mixup_sess_srcs2])
        
    return srcs_mixup_sess


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)


def forward(model, i, data,  input_aug_type,lam=0.6, train=True, mixup=False):
    alias_inputs, A, items, mask, targets, num_augs = data.get_slice(i,  input_aug_type)
    if mixup:
        overlap_A, _ = data.get_overlap(items)
    alias_inputs = trans_to_cuda(torch.Tensor(np.array(alias_inputs)).long())
    items = trans_to_cuda(torch.Tensor(np.array(items)).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model(items, A)
    feats = model.session_encoding(hidden, alias_inputs, mask)

    if not train:
        scores = model.compute_scores(feats)
        return targets, scores, num_augs
    else: # train mode
        # if mixup:
        #     # print("mixuping")
        #     head_feats = feats[head_idxs]
        #     head_targets = targets[head_idxs]
        #     # all_idxs = np.concatenate((head_idxs, tail_idxs), axis = 0)
        #     mixup_sess_srcs = find_mixup_srcs(tail_idxs, overlap_A, batch_size=A.shape[0])
        #     tail_mixed_feats = lam * feats[trans_to_cuda(mixup_sess_srcs[:, 0]), :] + (1 - lam) * feats[trans_to_cuda(mixup_sess_srcs[:, 1]), :]
        #     y_as, y_bs = targets[mixup_sess_srcs[:, 0]], targets[mixup_sess_srcs[:, 1]]
        #
        #     tail_mixed_logits = model.compute_scores(tail_mixed_feats)
        #     head_logits = model.compute_scores(head_feats)
        #     return head_targets, y_as, y_bs, head_logits, tail_mixed_logits, tail_idxs, num_augs
            # print("NOT mixup-ing")
        scores= model.compute_scores(feats)
        return targets, scores,  num_augs


def train_test(model, train_data, test_data, mixup, input_aug_type, n_node,  lam, Ks=[10, 20]):
    epoch_start_train = time.time()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_num_augs = 0
    slices = train_data.generate_batch(model.batch_size)

    tail_emb, head_emb = [], []
    for i, j in zip(slices, np.arange(len(slices))):
        if mixup:
            head_targets, targets_a, targets_b, head_logits, tail_logits, _, num_augs = forward(model, i,
                                                                                                train_data,
                                                                                                lam,
                                                                                                train=True,
                                                                                                mixup=True)
            if targets_a.shape == torch.Size([]):
                targets_a = targets_a.view(-1)
                targets_b = targets_b.view(-1)

            targets_a = trans_to_cuda(torch.Tensor(targets_a).long())
            targets_b = trans_to_cuda(torch.Tensor(targets_b).long())
            head_targets = trans_to_cuda(torch.Tensor(head_targets).long())
            h_loss = model.loss_function(head_logits, head_targets - 1)

            t_loss = mixup_criterion(model.loss_function, tail_logits, targets_a - 1, targets_b - 1, lam)
            loss = h_loss + t_loss
        else:
            targets, scores, num_augs = forward(model, i, train_data,  input_aug_type, lam=None, train=True, mixup=False)

            # tail_emb.append(tail_b.cpu().detach().numpy())
            # head_emb.append(head_b.cpu().detach().numpy())

            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        total_loss += loss.item()
        total_num_augs +=num_augs

        if j % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d]\tLoss: %.3f  Time: %.2f' % (j, len(slices), loss.item(), t))
            epoch_start_train = time.time()

    print('\t\tTotal Loss: %.3f \tTotal # Augs : %d' % (total_loss, total_num_augs))

    # np.save(path + f'tail_emb_{epoch}', np.array(tail_emb))
    # np.save(path + f'head_emb_{epoch}', np.array(head_emb))

    print('start predicting: ', datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(3)], [[] for i in range(3)]
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, num_augs  = forward(model, i, test_data, input_aug_type, lam=None, train=False,
                                                                 mixup=False)


        eval10 = get_metric_scores(scores, targets, Ks[0],  eval10)
        eval20 = get_metric_scores(scores, targets, Ks[1], eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

#    checkpoint = {
#        'epoch': epoch ,
#        'loss_min': loss,
#        'state_dict': model.state_dict(),

#    }

#    save_ckp(checkpoint, path)

    return loss, results



