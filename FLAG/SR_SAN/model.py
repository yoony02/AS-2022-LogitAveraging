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

def flag(model_forward, feats, targets, step_size, m=3):
    model, forward = model_forward
    model.train()
    model.optimizer.zero_grad()
    perturb = trans_to_cuda(torch.FloatTensor(feats.shape[0], feats.shape[1]).uniform_(-step_size, step_size))
    targets = trans_to_cuda(torch.Tensor(targets).long())
    perturb.requires_grad_()
    out = forward(perturb)
    loss = model.loss_function(out, targets - 1)
    loss /=m
    for _ in range(m-1):
        loss.backward(retain_graph = True)
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0
        out = forward(perturb)
        loss = model.loss_function(out, targets - 1)
        loss/=m
    loss.backward()
    model.optimizer.step()
    return loss, out



def forward(model, i, data,step_size, train):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)

    feats = model.session_encoding(hidden, alias_inputs, mask)
    if not train:
        targets = trans_to_cuda(torch.Tensor(targets).long())
        scores = model.compute_scores(feats)
        loss = 0
        return targets, loss, scores
    else:
        def forward(perturb):
            return model.compute_scores(feats + perturb)
        model_forward = (model, forward)
        loss, scores = flag(model_forward, feats, targets, step_size)
        return targets, loss, scores


def train_test(model, train_data, test_data, n_node, step_size = 8e-3,  lam=1, Ks = [10, 20]):
    epoch_start_train = time.time()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        targets, loss, scores = forward(model, i, train_data, step_size, train = True)
        total_loss+=loss.item()

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
        targets,_, scores = forward(model, i, test_data, step_size, train=False)
        #  scores, targets, test_data, k, pop_dict, ht_dict, test_label_dict, hit_label, mrr_label, eval

        eval10 = get_metric_scores(scores, targets, Ks[0],   eval10)
        eval20 = get_metric_scores(scores, targets, Ks[1],   eval20)
        # eval10 = get_metric_scores(logits, targets, test_data, Ks[0], eval10)
        # eval20 = get_metric_scores(logits, targets, test_data, Ks[1], eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

    return loss, results