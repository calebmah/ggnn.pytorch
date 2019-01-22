import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb
import time
import numpy as np
import pickle

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    #torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    #torch.manual_seed(1)

# import files in folder util
import sys
sys.path.insert(0, 'utils/')
import block
import graph_generator as g

from sklearn.metrics import confusion_matrix
from model import Graph_OurConvNet

#################
# select task and task parameters
#################

# subgraph matching
if 1==1:
    task_parameters = {}
    task_parameters['flag_task'] = 'matching'
    task_parameters['nb_communities'] = 10
    task_parameters['nb_clusters_target'] = 2
    task_parameters['Voc'] = 3
    task_parameters['size_min'] = 15
    task_parameters['size_max'] = 25
    task_parameters['size_subgraph'] = 20
    task_parameters['p'] = 0.5
    task_parameters['q'] = 0.1
    task_parameters['W0'] = block.random_graph(task_parameters['size_subgraph'],task_parameters['p'])
    task_parameters['u0'] = np.random.randint(task_parameters['Voc'],size=task_parameters['size_subgraph'])
    file_name = 'data/set_100_subgraphs_p05_size20_Voc3_2017-10-31_10-23-00_.txt'
    with open(file_name, 'rb') as fp:
        all_trainx = pickle.load(fp)
    task_parameters['all_trainx'] = all_trainx[:100]

# semi-supervised clustering
if 2==1:
    task_parameters = {}
    task_parameters['flag_task'] = 'clustering'
    task_parameters['nb_communities'] = 10
    task_parameters['nb_clusters_target'] = task_parameters['nb_communities']
    task_parameters['Voc'] = task_parameters['nb_communities'] + 1
    task_parameters['size_min'] = 5
    task_parameters['size_max'] = 25
    task_parameters['p'] = 0.5
    task_parameters['q'] = 0.1
    file_name = 'data/set_100_clustering_maps_p05_q01_size5_25_2017-10-31_10-25-00_.txt'
    with open(file_name, 'rb') as fp:
        all_trainx = pickle.load(fp)
    task_parameters['all_trainx'] = all_trainx[:100]

#print(task_parameters)

#################
# network and optimization parameters
#################

# network parameters
net_parameters = {}
net_parameters['Voc'] = task_parameters['Voc']
net_parameters['D'] = 50
net_parameters['nb_clusters_target'] = task_parameters['nb_clusters_target']
net_parameters['H'] = 50
net_parameters['L'] = 10
#print(net_parameters)

# optimization parameters
opt_parameters = {}
opt_parameters['learning_rate'] = 0.00075   # ADAM
opt_parameters['max_iters'] = 5000
opt_parameters['batch_iters'] = 100
if 2==1: # fast debugging
    opt_parameters['max_iters'] = 101
    opt_parameters['batch_iters'] = 10
opt_parameters['decay_rate'] = 1.25
#print(opt_parameters)


#########################
# Graph convnet function
#########################
def our_graph_convnets(task_parameters,net_parameters,opt_parameters):

    # Delete existing network if exists
    try:
        del net
        print('Delete existing network\n')
    except NameError:
        print('No existing network to delete\n')

    # instantiate
    net = Graph_OurConvNet(net_parameters)
    if torch.cuda.is_available():
        net.cuda()
    print(net)

    # number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('nb_param=',nb_param,' L=',net_parameters['L'])

    # task parameters
    flag_task = task_parameters['flag_task']
    # network parameters
    Voc = net_parameters['Voc']
    D = net_parameters['D']
    nb_clusters_target = net_parameters['nb_clusters_target']
    H = net_parameters['H']
    L = net_parameters['L']
    # optimization parameters
    learning_rate = opt_parameters['learning_rate']
    max_iters = opt_parameters['max_iters']
    batch_iters = opt_parameters['batch_iters']
    decay_rate = opt_parameters['decay_rate']

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    lr = learning_rate
    optimizer = net.update(lr)

    #############
    # loop over epochs
    #############
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    tab_results = []
    for iteration in range(1*max_iters):  # loop over the dataset multiple times

        # generate one train graph
        if flag_task=='matching': # subgraph matching
            train_x = g.variable_size_graph(task_parameters)
        elif flag_task=='clustering': # semi supervised clustering
            train_x = g.graph_semi_super_clu(task_parameters)
        train_y = train_x.target
        train_y = Variable( torch.LongTensor(train_y).type(dtypeLong) , requires_grad=False)

        # forward, loss
        y = net.forward(train_x)
        # compute loss weigth
        labels = train_y.data.cpu().numpy()
        V = labels.shape[0]
        nb_classes = len(np.unique(labels))
        cluster_sizes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(labels==r)[0]
            cluster_sizes[r] = len(cluster)
        weight = torch.zeros(nb_classes)
        for r in range(nb_classes):
            sumj = 0
            for j in range(nb_classes):
                if j!=r:
                    sumj += cluster_sizes[j]
            weight[r] = sumj/ V
        loss = net.loss(y,train_y,weight)
        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        S = train_y.data.cpu().numpy()
        C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
        CM = confusion_matrix(S,C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = train_y.data.cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y==r)[0]
            CM[r,:] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ nb_classes

        # backward, update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # learning rate, print results
        if not iteration%batch_iters:

            # time
            t_stop = time.time() - t_start
            t_start = time.time()

            # confusion matrix
            average_conf_mat = running_conf_mat/ running_total
            running_conf_mat = 0

            # accuracy
            average_accuracy = running_accuracy/ running_total
            running_accuracy = 0

            # update learning rate
            average_loss = running_loss/ running_total
            if average_loss > 0.99* average_loss_old:
                lr /= decay_rate
            average_loss_old = average_loss
            optimizer = net.update_learning_rate(optimizer, lr)
            running_loss = 0.0
            running_total = 0

            # save intermediate results
            tab_results.append([iteration,average_loss,100* average_accuracy,time.time()-t_start_total])

            # print results
            if 1==1:
                print('\niteration= %d, loss(%diter)= %.3f, lr= %.8f, time(%diter)= %.2f' %
                      (iteration, batch_iters, average_loss, lr, batch_iters, t_stop))
                #print('Confusion matrix= \n', 100* average_conf_mat)
                print('accuracy= %.3f' % (100* average_accuracy))

    ############
    # Evaluation on 100 pre-saved data
    ############
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    for iteration in range(100):

        # generate one data
        if flag_task == 'matching':
            train_x = g.variable_size_graph(task_parameters)
        if flag_task == 'clustering':
            train_x = task_parameters['all_trainx'][iteration][1]
        train_y = train_x.target
        train_y = Variable( torch.LongTensor(train_y).type(dtypeLong) , requires_grad=False)

        # forward, loss
        y = net.forward(train_x)
        # compute loss weigth
        labels = train_y.data.cpu().numpy()
        V = labels.shape[0]
        nb_classes = len(np.unique(labels))
        cluster_sizes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(labels==r)[0]
            cluster_sizes[r] = len(cluster)
        weight = torch.zeros(nb_classes)
        for r in range(nb_classes):
            sumj = 0
            for j in range(nb_classes):
                if j!=r:
                    sumj += cluster_sizes[j]
            weight[r] = sumj/ V
        loss = net.loss(y,train_y,weight)
        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        S = train_y.data.cpu().numpy()
        C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
        CM = confusion_matrix(S,C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = train_y.data.cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y==r)[0]
            CM[r,:] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ nb_classes

        # confusion matrix
        average_conf_mat = running_conf_mat/ running_total
        average_accuracy = running_accuracy/ running_total
        average_loss = running_loss/ running_total

    # print results
    print('\nloss(100 pre-saved data)= %.3f, accuracy(100 pre-saved data)= %.3f' % (average_loss,100* average_accuracy))

    #############
    # output
    #############
    result = {}
    result['final_loss'] = average_loss
    result['final_acc'] = 100* average_accuracy
    result['final_CM'] = 100* average_conf_mat
    result['final_batch_time'] = t_stop
    result['nb_param_nn'] = nb_param
    result['plot_all_epochs'] = tab_results
    #print(result)

    return result

#run it
result = our_graph_convnets(task_parameters,net_parameters,opt_parameters)
