from collections import Counter
import numpy as np
import torch
import dgl

def label_last(g, last_nid):
    is_last = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]

    else:
        src = torch.LongTensor([])
        dst = torch.LongTensor([])

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = torch.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])
    return g


def collate_fn_factory(seq_to_graph, top_labels):

    def collate_fn(samples):
        seqs, labels = zip(*samples)

        inputs = []
        # make session graph for each sessions
        graphs = list(map(seq_to_graph, seqs))
        bg = dgl.batch(graphs)
        inputs.append(bg)

        # top label sidxs
        top_labels_sidx = []
        for label in top_labels:
            try:
                sidx = np.where(labels == label)[0]
                top_labels_sidx.append(sidx.tolist())
            except:
                top_labels_sidx.append([])  

        labels = torch.LongTensor(labels)
        return inputs, top_labels_sidx, labels

    return collate_fn