import argparse
import random
import time
from datetime import datetime
import xlsxwriter

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
parser.add_argument('--self_loop', type=bool, default=True, help='Self loop nodes with edges')
parser.add_argument('--net', type=str, default='RGGC', choices=['GGNN','RGGC'], help='GGNN or RGGC')

parser.add_argument('--debug', action='store_true', help='print debug')
parser.add_argument('--save_all', action='store_true', help='print debug')

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
#    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True, opt.train_size)
#    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
#                                      shuffle=True, num_workers=2)
#
#    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False, opt.train_size)
#    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
#                                     shuffle=False, num_workers=2)
#
#    opt.annotation_dim = 1  # for bAbI
#    opt.n_edge_types = train_dataset.n_edge_types
#    opt.n_node = train_dataset.n_node
#
#    if opt.net == 'GGNN':
#        net = GGNN(opt)
#    else:
#        net = Graph_OurConvNet(opt)
##    net.double()
#    print(net)
#
#    criterion = nn.CrossEntropyLoss()
#
#    if opt.cuda:
#        net.cuda()
#        criterion.cuda()
#
#    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
#    
#    start_time = time.time()
#
#    for epoch in range(0, opt.niter):
#        train(epoch, train_dataloader, net, criterion, optimizer, opt)
#        test(test_dataloader, net, criterion, optimizer, opt)

    if opt.save_all:
        TASK_IDS = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 18]
        results = []
        for i in TASK_IDS:
            opt.task_id = i
            results.append(run(opt))
            
        # Create a workbook and add a worksheet.
        name = datetime.now().strftime("%d-%m-%Y %H.%M.%S")
        workbook = xlsxwriter.Workbook('{} {} {} {} {}.xlsx'.format(name,opt.net, opt.train_size, opt.niter, opt.state_dim))
        worksheet = workbook.add_worksheet()
        
        # Add a bold format to use to highlight cells.
        bold = workbook.add_format({'bold': True})
        
        # Iterate over the data and write it out row by row.
        worksheet.write(0, 0, name)
        worksheet.write(1, 0, "Parameters", bold)
        worksheet.write(1, 3, "Task", bold)
        worksheet.write(1, 4, "Average Loss", bold)
        worksheet.write(1, 6, "Accuracy", bold)
        worksheet.write(1, 7, "Time", bold)
        for row, item in enumerate(["net","cuda","train_size", "niter", "n_steps","state_dim","lr"],2):
            worksheet.write(row, 0, item)
            worksheet.write(row, 1, vars(opt)[item])

        row = 2
        col = 3 
        
        for i, (train_loss, test_loss, numerator, denominator, clock) in enumerate(results):
            worksheet.write(row, col,     TASK_IDS[i])
            worksheet.write(row, col + 1, train_loss)
            worksheet.write(row, col + 2, "{}/{}".format(numerator,denominator))
            worksheet.write(row, col + 3, numerator.item()/denominator, workbook.add_format({'num_format': '0.00%'}))
            worksheet.write(row, col + 4, clock)
            row += 1
        
        workbook.close()
        print('{}.xlsx saved'.format(name))
    else:
        start_time = time.time()
        train_loss, test_loss, numerator, denominator, clock = run(opt)
        print("--- Run time: %s seconds ---" % (time.time() - start_time))
        

if __name__ == "__main__":
    main(opt)
