import time
import datetime
import numpy as np

import dgl
import torch
from torch import nn
import torch.nn.functional as F

from utils import AverageMeter, WarmupCosineLrScheduler, fix_weight_decay, get_metric_scores, metric_print

class EOPA(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']
        _, hn = self.gru(m)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)

            mg.ndata['ft'] = self.feat_drop(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(dgl.function.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)

            if self.activation is not None:
                rst = self.activation(rst)
            return rst


# graph level representation : sessions' global embedding
class AttnReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (nn.Linear(input_dim, output_dim, bias=False)
                       if output_dim != input_dim else None)
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)

        # Equation (13)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v) # dgl.broadcast ?!

        # Equation (12)
        e = self.fc_e(torch.sigmoid(feat_u + feat_v))
        beta = dgl.ops.segment.segment_softmax(g.batch_num_nodes(), e)

        # Equation (11)
        feat_norm = feat * beta
        rst = dgl.ops.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')

        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst



class LESSR_part(nn.Module):
    def __init__(self, num_items, embedding_dim, num_layers, device, batch_norm=True, feat_drop=0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(torch.arange(num_items, dtype=torch.long), requires_grad=False)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.device = device

        input_dim = embedding_dim

        for i in range(num_layers):
            layer = EOPA(input_dim, embedding_dim, batch_norm=batch_norm,
                         feat_drop=feat_drop, activation=nn.PReLU(embedding_dim))

            input_dim += embedding_dim
            self.layers.append(layer)

        self.readout = AttnReadout(input_dim, embedding_dim, embedding_dim, batch_norm=batch_norm,
                                   feat_drop=feat_drop, activation=nn.PReLU(embedding_dim))

        input_dim += embedding_dim
        self.feat_drop = nn.Dropout(feat_drop)
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg):
        iid = mg.ndata['iid']
        feat = self.embedding(iid)

        for i, layer in enumerate(self.layers):
            out = layer(mg, feat)
            feat = torch.cat([out, feat], dim=1)

        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]

        # Equation (14) + concat cluster representation
        sr = torch.cat([sr_l, sr_g], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))

        # Equation (15)
        logits = sr @ self.embedding(self.indices).t()

        return sr, logits


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


def train_test(model, Ks, train_loader, test_loader, n_iters_all, n_items, device, 
               lr=1e-3, weight_decay=1e-4):
    if weight_decay > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()

    optim = torch.optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    print('start training: ', datetime.datetime.now())
    model.train()
    loss_meter = AverageMeter()
    epoch_start_time = time.time()
    i = 0

    for batch in train_loader:
        inputs, labels = batch
        inputs = [x.to(device) for x in inputs]
        # inputs = trans_to_cuda(inputs)
        labels = trans_to_cuda(labels)
        _, logits = model(*inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        
        loss.backward()
        optim.step()
        lr_schdlr.step()
        optim.zero_grad()
        loss_meter.update(loss.item())

        if (i + 1) % 1000 == 0:
            t = time.time() - epoch_start_time
            print("[%d/%d] Loss: %.4f   Time: %.2f" % (i+1, 0, loss_meter.avg, t))
            epoch_start_time = time.time()
        i += 1
    print('\t Total Loss:\t%.3f' % loss_meter.avg)

    # eval
    print("start predicting: ", datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(3)], [[] for i in range(3)]
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            _, logits = model(*inputs)

            eval10 = get_metric_scores(logits, labels, Ks[0], eval10)
            eval20 = get_metric_scores(logits, labels, Ks[1], eval20)
            
        t = time.time() - epoch_start_eval
        results = metric_print(eval10, eval20, n_items, t)  

    return loss, results  
