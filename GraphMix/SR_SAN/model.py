import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
import time
from utils import *

class SelfAttentionNetwork(Module):
    def __init__(self, opt, n_node):
        super(SelfAttentionNetwork, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=opt.nhead,dim_feedforward=self.hidden_size * opt.feedforward)
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, opt.layer)

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
        # if not self.nonhybrid:
        #     a = self.linear_transform(torch.cat([a, ht], 1))

        return a

    def compute_scores(self, a):
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = hidden.transpose(0,1).contiguous()
        hidden = self.transformerEncoder(hidden)
        hidden = hidden.transpose(0,1).contiguous()
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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    a = criterion(pred, y_a)
    b = criterion(pred, y_b)

    return lam * a + (1-lam) * b


def forward(model, i, data, lam=None, train=True):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    feats = model.session_encoding(hidden, alias_inputs, mask)

    if not train:
        scores = model.compute_scores(feats)
        return targets, scores
    else:  
        mixup_sess_srcs = torch.randint(high=A.shape[0], size=(A.shape[0], ))
        mixed_feats = lam * feats + (1-lam) * feats[trans_to_cuda(mixup_sess_srcs), :]
        y_as, y_bs = targets, targets[mixup_sess_srcs]

        mixed_logits = model.compute_scores(mixed_feats)
        logits = model.compute_scores(feats)
        return targets, y_as, y_bs, logits, mixed_logits

def train_test(model, train_data, test_data, n_node, lam=0.6, Ks = [10, 20]):
    epoch_start_train = time.time()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        targets, targets_a, targets_b, scores, mixed_scores = forward(model, i, train_data, lam)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        targets_a = trans_to_cuda(torch.Tensor(targets_a).long())
        targets_b = trans_to_cuda(torch.Tensor(targets_b).long())
        loss_o = model.loss_function(scores, targets-1)
        loss_m = mixup_criterion(model.loss_function, mixed_scores, targets_a-1, targets_b-1, lam)
        loss = loss_o + loss_m

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        total_loss += loss.item()
        if (j + 1) % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d]\tLoss: %.3f  Time: %.2f' % (j + 1, len(slices), loss.item(), t))

    print('\t\tTotal Loss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(3)], [[] for i in range(3)]
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, logits = forward(model, i, test_data, train=False)
        
        eval10 = get_metric_scores(logits, targets, Ks[0], eval10)
        eval20 = get_metric_scores(logits, targets, Ks[1], eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

    return loss, results