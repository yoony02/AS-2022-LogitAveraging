from collections import Counter
import numpy as np
import torch
import dgl
import random
import itertools


def random_deletion(sessions, targets):
    candidate_sess_idx = [i for i, sess in enumerate(sessions) if len(sess) > 1]
    aug_sess_sidx = random.sample(candidate_sess_idx, int(len(candidate_sess_idx)*0.8))

    aug_sess = []
    aug_tars = []
    for i in aug_sess_sidx:
        cur_sess = sessions[i].copy()
        remove_item = random.choice(cur_sess)
        cur_sess.remove(remove_item)
        aug_sess.append(cur_sess)
        aug_tars.append(targets[i])

    return aug_sess, aug_tars

def random_insertion(sessions, targets, sample_num):
    aug_sess_sidx = random.sample(range(len(sessions)), sample_num)
    candidate_item = list(set(itertools.chain.from_iterable(sessions)))
    
    aug_sess = []
    aug_tars = []
    for i in aug_sess_sidx:
        cur_sess = sessions[i].copy()
        insert_index = random.randint(0, len(cur_sess)-1)
        cur_sess.insert(insert_index, random.choice(candidate_item))
        aug_sess.append(cur_sess)
        aug_tars.append(targets[i])

    return aug_sess, aug_tars


def create_aug_sessions(batch_seqs, targets, input_aug_type, sample_num=50):

    if input_aug_type == 'deletion':
        aug_sess, aug_tars = random_deletion(batch_seqs, targets)
    elif input_aug_type == 'insertion':
        aug_sess, aug_tars = random_insertion(batch_seqs, targets, int(len(batch_seqs)*0.5))
    else:
        print("please select type deletion or insertion")
    return aug_sess, aug_tars

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


def collate_fn_factory(seq_to_graph, top_labels, input_aug_type=None):

    def collate_fn(samples):
        seqs, labels = zip(*samples)

        if input_aug_type is not None:
            aug_inputs, aug_targets = create_aug_sessions(seqs, labels, input_aug_type)
            seqs = list(seqs) + aug_inputs
            labels = list(labels) + aug_targets
            seqs = tuple(seqs)
            labels = tuple(labels)

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