import argparse
import pickle
import time
from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=96, help='hidden state size')
parser.add_argument('--nhead', type=int, default=2, help='the number of heads of multi-head attention')
parser.add_argument('--layer', type=int, default=1, help='number of SAN layers')
parser.add_argument('--feedforward', type=int, default=4, help='the multipler of hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--gpu_num', type = int, default = 0, help = 'cuda number')
parser.add_argument('--mixup', type=bool, default=False, help='tail mixup')
parser.add_argument('--lam', type=float, default=0.6, help='mixup ratio')
parser.add_argument('--batch_aug', type=bool, default=False, help='batch graph augmentation')
parser.add_argument('--save_model', type=bool, default=True)
opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu_num)
if opt.save_model:
    os.makedirs(f'ckpt/{opt.dataset}', exist_ok=True)

def main():
    train_data = pickle.load(open(f'../../Dataset/{opt.dataset}/train.txt', 'rb'))
    test_data = pickle.load(open(f'../../Dataset/{opt.dataset}/test.txt', 'rb'))


    #ht_dict = pickle.load(open(f'../../Dataset/{opt.dataset}/ht_dict.pickle', 'rb'))

    train_data = Data(train_data, opt.batch_aug, opt.mixup, shuffle=True)
    test_data = Data(test_data, batch_aug=False, mixup=False, shuffle=False)

    if 'retailrocket' in opt.dataset:
        n_node = 27413
    elif 'digi' in opt.dataset:
        n_node = 43098
    elif 'tmall' in opt.dataset:
        n_node = 40728
    elif 'yoochoose' in opt.dataset:
        n_node = 37484
    else:
        print("no dataset")


    model = trans_to_cuda(SelfAttentionNetwork(opt, n_node))

    start = time.time()
    best_results = [[0 for i in range(8)] for j in range(2)]
    best_epochs = [[0 for i in range(8)] for j in range(2)]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        loss, results = train_test(model, train_data, test_data, opt.mixup, n_node, opt.lam)
        flag = get_best_result(results, epoch, best_results, best_epochs)

        if flag > 0 :
            if opt.save_model == True:
                model_save_path = f'ckpt/{opt.dataset}/{epoch}.pt'
                torch.save(model.state_dict(), model_save_path)
                print("Model Saving Done")
        else:
            bad_counter += 1
            if bad_counter >= opt.patience:
                break

    print('-' * 100)
    end = time.time()
    print("Run time: %f s" % (end - start))
    print('-------------------------------------------------------')



if __name__ == '__main__':
    main()