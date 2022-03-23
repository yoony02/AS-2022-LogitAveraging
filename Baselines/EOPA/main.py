# vanilla EOPA

import os
import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from utils import read_dataset, Dataset, get_best_results
from collate import seq_to_eop_multigraph, collate_fn_factory
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', type=str, help='dataset directory')
parser.add_argument('--embed_dim', type=int, default=32, help='the embedding size')
parser.add_argument('--n-layers', type=int, default=3, help='the number of layers')
parser.add_argument('--feat-drop', type=float, default=0.2, help='the dropout ratio for features')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for training')
parser.add_argument('--epoch', type=int, default=30, help='the number of training epochs')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='the parameter for L2 regularization', )
parser.add_argument('--patience', type=int, default=10,
                        help='the number of epochs that the performance does not improves after which the training stops', )
parser.add_argument('--num-workers', type=int, default=4, help='the number of processes to load the input graphs', )
parser.add_argument('--valid-split', type=float, default=None, help='the fraction for the validation set')
parser.add_argument('--save_model', default=False)
parser.add_argument('--seed', type=int, default=220, help='seed for random behaviors, no seed if negtive')
parser.add_argument('--gpu_num', type=int, default=0)
opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu_num)
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
if opt.save_model:
    os.makedirs(f'ckpt/{opt.dataset}', exist_ok=True)

def main():
    dataset_name = opt.dataset
    dataset_dir = Path(f'../../Dataset_eopa/{opt.dataset}')
    Ks = [10, 20]

    print("reading Dataset")
    train_sessions, test_sessions, num_items = read_dataset(dataset_dir)

    train_set = Dataset(train_sessions)
    test_set = Dataset(test_sessions)

    print("loading Dataset")
    collate_fn = collate_fn_factory(seq_to_eop_multigraph)

    train_loader = DataLoader(train_set,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            collate_fn=collate_fn,
                            )

    test_loader = DataLoader(test_set,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            collate_fn=collate_fn
                            )

    # set seed
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    n_iters_per_epoch = len(train_set) // opt.batch_size 
    n_iters_all = n_iters_per_epoch * opt.epoch
    
    model = trans_to_cuda(LESSR_part(num_items, opt.embed_dim, opt.n_layers, opt.feat_drop))

    start = time.time()
    best_results = [[0 for i in range(3)] for j in range(2)]
    best_epochs = [[0 for i in range(3)] for j in range(2)]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-' * 100)
        print('Epoch: ', epoch)
        loss, results = train_test(model, Ks, train_loader, test_loader, n_iters_all, num_items, device)

        flag = get_best_results(results, epoch, best_results, best_epochs)

        if flag > 0:
            if opt.save_model == True:
                model_save_path = f'ckpt/{opt.dataset}/{epoch}.pt'
                torch.save(model.save_dict(), model_save_path)
                print("Model Saving Done")
        else:
            bad_counter += 1
            if bad_counter >= opt.patience:
                break

    print('-' * 100)
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()