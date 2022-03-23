import datetime
import math

import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import time
from utils import get_metric_scores, metric_print
from agc import AGC

class Attention_GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(Attention_GNN, self).__init__()
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

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]],
                                self.linear_edge_in(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

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


class Attention_SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(Attention_SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=2, dropout=0.1)

        self.linear_one = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_two = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.agc_optimizer = AGC(self.parameters(), self.optimizer, model=self)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def session_encoding(self, hidden, alias_inputs, mask):
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(
            mask, 1) - 1]  # batch_size x latent_size
        # batch_size x 1 x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
        # batch_size x seq_length x 1
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)  # batch_size x seq_length x 1
        # batch_size x latent_size
        a = torch.sum(alpha * seq_hidden *
                      mask.view(mask.shape[0], -1, 1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        # batch_size x seq_length x latent_size
        hidden = seq_hidden * mask.view(mask.shape[0], -1, 1).float()
        qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
        # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1, 2), -1)
        target = beta @ hidden  # batch_size x n_nodes x latent_size
        a = a.view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        a = a + target  # batch_size x n_nodes x latent_size
        return a, b

    def compute_scores(self, a, b):
        scores = torch.sum(a * b, -1)  # batch_size x n_nodes
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.tagnn(A, hidden)
        hidden = hidden.permute(1, 0, 2)

        skip = self.layer_norm1(hidden)
        hidden, attn_w = self.attn(
            hidden, hidden, hidden, attn_mask=get_mask(hidden.shape[0]))
        hidden = hidden+skip
        hidden = hidden.permute(1, 0, 2)
        return hidden

def get_mask(seq_len):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to('cuda')


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

def forward(model, i, data, top_labels, lam=None, train=True):
    alias_inputs, A, items, mask, targets, top_labels_sidx = data.get_slice(i, top_labels)
    alias_inputs = trans_to_cuda(torch.Tensor(np.array(alias_inputs)).long())
    items = trans_to_cuda(torch.Tensor(np.array(items)).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    feats, b = model.session_encoding(hidden, alias_inputs,mask)

    if not train:
        scores = model.compute_scores(feats, b)
        return targets, scores
    else: 
        mixup_sess_srcs = torch.randint(high=A.shape[0], size=(A.shape[0], ))
        mixed_feats = lam * feats + (1-lam) * feats[trans_to_cuda(mixup_sess_srcs), :]
        y_as, y_bs = targets, targets[mixup_sess_srcs]

        mixed_logits = model.compute_scores(mixed_feats, b)
        logits = model.compute_scores(feats, b)
        return targets, y_as, y_bs, logits, mixed_logits, top_labels_sidx


def train_test(model, train_data, test_data, n_node, top_labels, lam=0.6, Ks=[10, 20]):
    epoch_start_train = time.time()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        targets, targets_a, targets_b, logits_o, mixed_logits, top_labels_sidx = forward(model, i, train_data, top_labels, lam)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        targets_a = trans_to_cuda(torch.Tensor(targets_a).long())
        targets_b = trans_to_cuda(torch.Tensor(targets_b).long())
        
        probs = logits_o.clone()
        with torch.no_grad():
            for i, sidx in enumerate(top_labels_sidx):
                if len(sidx) == 0:
                    pass
                else:
                    gathered_logits = torch.mean(logits_o[sidx], dim=0)
                    probs.index_copy_(0, trans_to_cuda(torch.tensor(sidx)),
                    gathered_logits.view(1, -1).repeat(len(sidx), 1))
        
        probs = trans_to_cuda(probs)
        logits_p = torch.softmax(probs, dim=1)
        
        loss_o = model.loss_function(logits_o, targets-1)
        loss_m = mixup_criterion(model.loss_function, mixed_logits, targets_a-1, targets_b-1, lam)
        loss_p = model.loss_function(logits_p, targets-1)
        loss = loss_o + loss_m + loss_p
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        total_loss += loss.item()
        if (j + 1) % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d] Loss_o: %.4f   Loss_m: %.4f   Loss_p: %.4f  Time: %.2f' % (j, len(slices), loss_o.item(), loss_m.item(), loss_p.item(), t))
            epoch_start_train = time.time()
    
    print('\t\tTotal Loss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(3)], [[] for i in range(3)]
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, logits = forward(model, i, test_data, top_labels, train=False)
    
        eval10 = get_metric_scores(logits, targets, Ks[0], eval10)
        eval20 = get_metric_scores(logits, targets, Ks[1], eval20)
        
    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

    return loss, results