import datetime
import math
import pdb

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
        return a,b

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
        try:
            temp = lam * criterion(pred, y_a)+ (1 - lam) * criterion(pred, y_b)
        except:
            print( 'pred' , pred.shape )
            print('y_a', y_a.shape)
            print('y_b', y_b.shape)
            import pdb
            pdb.set_trace()

        return temp

def forward(model, i, data, lam=0.6, train=True, mixup=False):
    alias_inputs, A, items, mask, targets = data.get_slice(i, mixup=True)
    if mixup:
        overlap_A, _ = data.get_overlap(items)
    alias_inputs = trans_to_cuda(torch.Tensor(np.array(alias_inputs)).long())
    items = trans_to_cuda(torch.Tensor(np.array(items)).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model(items, A)
    feats, b = model.session_encoding(hidden, alias_inputs,mask)

    if not train:
        scores = model.compute_scores(feats, b)
        return targets, scores
    else: # train mode
        # if mixup:
        #     # print("mixuping")
        #     head_feats = feats[head_idxs]
        #     head_targets = targets[head_idxs]
        #     mixup_sess_srcs = find_mixup_srcs( overlap_A, batch_size=A.shape[0])
        #     tail_mixed_feats = lam * feats[trans_to_cuda(mixup_sess_srcs[:, 0]), :] + (1 - lam) * feats[trans_to_cuda(mixup_sess_srcs[:, 1]), :]
        #     y_as, y_bs = targets[mixup_sess_srcs[:, 0]], targets[mixup_sess_srcs[:, 1]]
        #
        #     tail_mixed_logits = model.compute_scores(tail_mixed_feats, b)
        #     head_logits = model.compute_scores(head_feats, b)
        #     return head_targets, torch.tensor(y_as).long(), torch.tensor(y_bs).long(), head_logits, tail_mixed_logits, tail_idxs
        # else:
        # print("NOT mixup-ing")
        scores = model.compute_scores(feats, b)
        return targets, scores



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



def train_test(model, train_data, test_data, mixup, n_node, lam, Ks = [10, 20]):
    epoch_start_train = time.time()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        if mixup:

            head_targets, targets_a, targets_b, head_logits, tail_logits, _= forward(model, i, train_data, lam, train=True, mixup=True)

            if targets_a.shape == torch.Size([]):
                targets_a = targets_a.view(-1)
                targets_b = targets_b.view(-1)
            #     print("무사히 넘어갔음")
            # try:
            #
            # except:
            #     print(tail_logits.shape)
            #     print(targets_a.shape)
            #     print(targets_b.shape)
            #     print("Error 남")
            #     pdb.set_trace()

            targets_a = trans_to_cuda(targets_a)
            targets_b = trans_to_cuda(targets_b)
            head_targets = trans_to_cuda(torch.Tensor(head_targets).long())
            h_loss = model.loss_function(head_logits, head_targets-1)
            t_loss = mixup_criterion(model.loss_function, tail_logits, targets_a-1, targets_b-1, lam)
            loss = h_loss + t_loss
        else:
            targets, scores = forward(model, i, train_data,  lam=None, train=True, mixup=False)
            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
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
        targets, logits = forward(model, i, test_data, lam=None, train=False, mixup=False)
        #  scores, targets, test_data, k, pop_dict, ht_dict, test_label_dict, hit_label, mrr_label, eval

        eval10 = get_metric_scores(logits, targets, Ks[0], eval10)
        eval20 = get_metric_scores(logits, targets, Ks[1], eval20)
        # eval10 = get_metric_scores(logits, targets, test_data, Ks[0], eval10)
        # eval20 = get_metric_scores(logits, targets, test_data, Ks[1], eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_node, t)

    return loss, results