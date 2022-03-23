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
        self.scale = opt.scale
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

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)


        b = self.embedding.weight[1:]  # n_nodes x latent_size

        scores = torch.matmul(a, b.transpose(1,0))
        if self.scale:
            scores = 16 * scores

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

def forward(model, i, data, top_labels):
    alias_inputs, A, items, mask, targets, top_labels_sidx = data.get_slice(i,  top_labels)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    return targets, top_labels_sidx, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, n_node, top_labels, lam=1, Ks = [10, 20]):
    epoch_start_train = time.time()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        targets,top_labels_sidx, scores_o= forward(model, i, train_data, top_labels)
        targets_cuda = trans_to_cuda(torch.Tensor(targets).long())
        loss_o = model.loss_function(scores_o, targets_cuda - 1)

        probs = scores_o.clone()
        with torch.no_grad():
            for i, sidx in enumerate(top_labels_sidx):
                if len(sidx) == 0:
                    pass
                else:
                    gathered_logits = torch.mean(scores_o[sidx], dim = 0)
                    probs.index_copy_(0, trans_to_cuda(torch.tensor(sidx)),
                    gathered_logits.view(1, -1).repeat(len(sidx), 1))

        probs = trans_to_cuda(probs)
        scores_p = torch.softmax(probs, dim=1)

        loss_p = nn.functional.cross_entropy(scores_p, targets_cuda -1 )
        loss = loss_o + (lam* loss_p)

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
        targets,_, scores = forward(model, i, test_data, top_labels)
        #  scores, targets, test_data, k, pop_dict, ht_dict, test_label_dict, hit_label, mrr_label, eval

        eval10 = get_metric_scores(scores, targets, Ks[0],   eval10)
        eval20 = get_metric_scores(scores, targets, Ks[1],   eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

    return loss, results