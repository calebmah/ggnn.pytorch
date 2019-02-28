import argparse
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from model import GGNN
from model import Graph_OurConvNet
from utils.train import train
from utils.test import test
from utils.data.dataset import bAbIDataset
from utils.data.dataloader import bAbIDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--processed_path', default='processed_1', help='path to processed data')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--train_size', type=int, default=50, help='number of training data')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--vocab', type=int, default=3, help='size of vocabulary')
parser.add_argument('--nb_clusters_target', type=int, default=1, help='number of target clusters')
parser.add_argument('--D', type=int, default=50, help='Dimensions')
parser.add_argument('--H', type=int, default=50, help='Hidden layer size')
parser.add_argument('--L', type=int, default=10, help='Number of layers')

parser.add_argument('--debug', action='store_true', help='print debug')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = 'babi_data/%s/train/%d_graphs.txt' % (opt.processed_path, opt.task_id)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True, opt.train_size)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2)

    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False, opt.train_size)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    #net = GGNN(opt)
    net = Graph_OurConvNet(opt)
#    net.double()
    print(net)

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    
    start_time = time.time()

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        test(test_dataloader, net, criterion, optimizer, opt)
        
    print("--- Run time: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main(opt)
