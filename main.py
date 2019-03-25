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
from utils.save_results import save_results

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--processed_path', default='processed', help='path to processed data')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--train_size', type=int, default=50, help='number of training data')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size') # H
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN') # L
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--vocab', type=int, default=3, help='size of vocabulary')
parser.add_argument('--nb_clusters_target', type=int, default=1, help='number of target clusters')
parser.add_argument('--D', type=int, default=50, help='Dimensions')
parser.add_argument('--H', type=int, default=50, help='Hidden layer size')
#parser.add_argument('--L', type=int, default=10, help='Number of layers')
parser.add_argument('--self_loop', type=bool, default=True, help='Self loop nodes with edges')
parser.add_argument('--net', type=str, default='RGGC', choices=['GGNN','RGGC'], help='GGNN or RGGC')

parser.add_argument('--debug', action='store_true', help='print debug')
parser.add_argument('--save_all', action='store_true', help='print debug')
parser.add_argument('--grid', action='store_true', help='grid search')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
#print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = 'babi_data/%s/train/%d_graphs.txt' % (opt.processed_path, opt.task_id)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)

def run(opt):
    start_time = time.time()
    opt.dataroot = 'babi_data/%s/train/%d_graphs.txt' % (opt.processed_path, opt.task_id)
    print(opt)

    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True, opt.train_size)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2)

    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False, opt.train_size)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    if opt.net == 'GGNN':
        net = GGNN(opt)
        net.double()
    else:
        net = Graph_OurConvNet(opt)
        net.double()
    print(net)

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for epoch in range(0, opt.niter):
        train_loss = train(epoch, train_dataloader, net, criterion, optimizer, opt)
        test_loss, numerator, denominator = test(test_dataloader, net, criterion, optimizer, opt)

    return train_loss, test_loss, numerator, denominator, time.time() - start_time

def main(opt):

    if opt.save_all:
        TASK_IDS = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 18]
        if opt.net == "GGNN":
            N_STEPS = [5,5,10,5,10,5,5,10,5,10,5]
            STATE_DIMS = [8,4,10,8,8,8,8,4,8,4,8]
            LRS =  [0.005,0.005,0.05,0.005,0.005,0.01,0.005,0.05,0.01,0.005,0.005]
        else:
            N_STEPS = [5,10,10,5,10,10,5,10,5,5,10]
            STATE_DIMS = [10,8,8,8,10,4,4,10,4,4,10]
            LRS =  [0.05,0.05,0.01,0.005,0.01,0.005,0.005,0.05,0.01,0.01,0.005]
        
        results = []
        for i in range(len(TASK_IDS)):
            opt.task_id = TASK_IDS[i]
            opt.n_steps = N_STEPS[i]
            opt.state_dim = STATE_DIMS[i]
            opt.lr = LRS[i]
            results.append(run(opt))
        save_results(opt, results)
        
    else:
        start_time = time.time()
        train_loss, test_loss, numerator, denominator, clock = run(opt)
        print("--- Run time: %s seconds ---" % (time.time() - start_time))
    
def grid(opt):
    lrs = [0.005, 0.01, 0.05]
    state_dims = [4,8,10] 
    n_steps = [5,10] # L

    for lr in lrs:
        opt.lr = lr
        for state_dim in state_dims:
            opt.state_dim = state_dim
            for n_step in n_steps:
                opt.n_steps = n_step
                main(opt)
                    
if __name__ == "__main__":
    if opt.grid:
        grid(opt)
    else:
        main(opt)
