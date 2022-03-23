import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import time
import datetime
import numpy as np
from utils import get_metric_scores, metric_print
import pdb
# import metric


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, opt):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hiddenSize
        self.batch_size = opt.batchSize
        self.n_layers = opt.n_layers
        self.embedding_dim = opt.embed_dim

        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        #self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    def forward(self, seq, lengths):
        # pdb.set_trace()
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths.tolist())
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        return c_t
    
    def compute_scores(self, c_t):
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)
        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)


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

def logit_avg(score, targets, top_labels_sidx):
    probs = score.clone()

    with torch.no_grad():
        for i, sidx in enumerate(top_labels_sidx):
            if len(sidx) == 0:
                pass
            else:
                gathered_logits = torch.mean(score[sidx], dim=0)
                probs.index_copy_(0, trans_to_cuda(torch.tensor(sidx)),
                                  gathered_logits.view(1, -1).repeat(len(sidx), 1))

    probs = trans_to_cuda(probs)
    scores_p = torch.softmax(probs, dim=1)
    loss_a = nn.functional.cross_entropy(scores_p, targets - 1)
    return loss_a

def flag(model_forward, feats, targets, step_size, top_label_sidx, m=3):
    model, forward = model_forward
    model.train()
    model.optimizer.zero_grad()
    perturb = trans_to_cuda(torch.FloatTensor(feats.shape[0], feats.shape[1]).uniform_(-step_size, step_size))
    targets = trans_to_cuda(torch.Tensor(targets).long())
    perturb.requires_grad_()
    out = forward(perturb)
    loss_p = model.loss_function(out, targets - 1)
    loss_a = logit_avg(model.compute_scores(feats), targets, top_label_sidx)
    loss = loss_p + loss_a
    loss /=m
    for _ in range(m-1):
        loss.backward(retain_graph = True)
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0
        out = forward(perturb)
        loss_p = model.loss_function(out, targets - 1)
        loss_a = logit_avg(model.compute_scores(feats), targets , top_label_sidx)
        loss = loss_p + loss_a
        loss/=m
    loss.backward()
    model.optimizer.step()
    return loss


def forward(model, i, data, top_labels, step_size, train):
    inputs, targets, inputs_len, top_labels_sidx = data.get_slice(i, top_labels)
    inputs = trans_to_cuda(inputs)
    feats = model(inputs, inputs_len)

    if train:
        forward = lambda perturb: model.compute_scores(feats + perturb)
        model_forward = (model, forward)
        loss = flag(model_forward, feats, targets, step_size, top_labels_sidx)
        return top_labels_sidx, loss
    
    else:
        scores = model.compute_scores(feats)
        targets = torch.Tensor(targets).long()
        targets = trans_to_cuda(targets)
        return targets, top_labels_sidx, scores

def train_test(model, train_data, test_data, n_items, top_labels, step_size=8e-3, Ks=[10, 20]):
    epoch_start_train = time.time()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())

    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        top_labels_sidx, loss = forward(model, i, train_data, top_labels, step_size, train = True)
        total_loss += loss
        if j % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d]\tLoss: %.3f  Time: %.2f' % (j, len(slices), loss.item(), t))
            epoch_start_train = time.time()

    print('\t\tTotal Loss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(3)], [[] for i in range(3)]
    slices = test_data.generate_batch(model.batch_size)    
    with torch.no_grad():
        for i in slices:
            targets,_, scores = forward(model, i, test_data, top_labels, step_size, train=False)
            logits = F.softmax(scores, dim=1)
            eval10 = get_metric_scores(logits, targets, Ks[0], eval10)
            eval20 = get_metric_scores(logits, targets, Ks[1], eval20)

    t = time.time() - epoch_start_eval

    results = metric_print(eval10, eval20, n_items, t)                                                            

    return loss, results
