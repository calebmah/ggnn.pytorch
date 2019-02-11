import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb 
import time
import numpy as np
import pickle

dtypeFloat = torch.FloatTensor
dtypeLong = torch.LongTensor

##############################
# Class cell definition
##############################
class OurConvNetcell(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(OurConvNetcell, self).__init__()

        # conv1
        self.Ui1 = nn.Linear(dim_in, dim_out, bias=False) 
        self.Uj1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Vi1 = nn.Linear(dim_in, dim_out, bias=False) 
        self.Vj1 = nn.Linear(dim_in, dim_out, bias=False)  
        self.bu1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        
        # conv2
        self.Ui2 = nn.Linear(dim_out, dim_out, bias=False) 
        self.Uj2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Vi2 = nn.Linear(dim_out, dim_out, bias=False) 
        self.Vj2 = nn.Linear(dim_out, dim_out, bias=False)  
        self.bu2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        
        # bn1, bn2
        self.bn1 = torch.nn.BatchNorm1d(dim_out)
        self.bn2 = torch.nn.BatchNorm1d(dim_out)
        
        # resnet
        self.R = nn.Linear(dim_in, dim_out, bias=False) 
            
        # init
        self.init_weights_OurConvNetcell(dim_in, dim_out, 1)
        
         
    def init_weights_OurConvNetcell(self, dim_in, dim_out, gain):
        
        # conv1
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.Ui1.weight.data.uniform_(-scale, scale) 
        self.Uj1.weight.data.uniform_(-scale, scale) 
        self.Vi1.weight.data.uniform_(-scale, scale) 
        self.Vj1.weight.data.uniform_(-scale, scale) 
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.bu1.data.fill_(0)
        self.bv1.data.fill_(0)
        
        # conv2
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.Ui2.weight.data.uniform_(-scale, scale) 
        self.Uj2.weight.data.uniform_(-scale, scale) 
        self.Vi2.weight.data.uniform_(-scale, scale) 
        self.Vj2.weight.data.uniform_(-scale, scale) 
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.bu2.data.fill_(0)
        self.bv2.data.fill_(0)
        
        # RN
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.R.weight.data.uniform_(-scale, scale)  
            
            
    def forward(self, x, E_start, E_end):
        
        # E_start, E_end : E x V

        xin = x
        # conv1
        Vix = self.Vi1(x)  #  V x H_out
        Vjx = self.Vj1(x)  #  V x H_out
        x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv1  # E x H_out
        x1 = F.sigmoid(x1)
        Ujx = self.Uj1(x)  #  V x H_out
        x2 = torch.mm(E_start, Ujx)  #  V x H_out
        Uix = self.Ui1(x)  #  V x H_out
        x = Uix + torch.mm(E_end.t(), x1*x2) + self.bu1 #  V x H_out
        # bn1
        x = self.bn1(x)
        # relu1
        x = F.relu(x)
        # conv2
        Vix = self.Vi2(x)  #  V x H_out
        Vjx = self.Vj2(x)  #  V x H_out
        x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv2  # E x H_out
        x1 = F.sigmoid(x1)
        Ujx = self.Uj2(x)  #  V x H_out
        x2 = torch.mm(E_start, Ujx)  #  V x H_out
        Uix = self.Ui2(x)  #  V x H_out
        x = Uix + torch.mm(E_end.t(), x1*x2) + self.bu2 #  V x H_out
        # bn2
        x = self.bn2(x)
        # addition
        x = x + self.R(xin)
        # relu2
        x = F.relu(x)
            
        return x
        
        
        
        
        
##############################
# Class NN definition
##############################  
class Graph_OurConvNet(nn.Module):
    
    def __init__(self, net_parameters):
        
        super(Graph_OurConvNet, self).__init__()
        
        # parameters
        flag_task = "matching"
        Voc = net_parameters['Voc']
        D = net_parameters['D']
        nb_clusters_target = net_parameters['nb_clusters_target']
        H = net_parameters['H']
        L = net_parameters['L']
        
        # vector of hidden dimensions
        net_layers = []
        for layer in range(L):
            net_layers.append(H)

        # embedding
        self.encoder = nn.Embedding(Voc, D)      
        
        # CL cells
        # NOTE: Each graph convnet cell uses *TWO* convolutional operations
        net_layers_extended = [D] + net_layers # include embedding dim
        L = len(net_layers)
        list_of_gnn_cells = [] # list of NN cells
        for layer in range(L//2):
            Hin, Hout = net_layers_extended[2*layer], net_layers_extended[2*layer+2]
            list_of_gnn_cells.append(OurConvNetcell(Hin,Hout))
            
        # register the cells for pytorch
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)
              
        # fc
        Hfinal = net_layers_extended[-1]
        self.fc = nn.Linear(Hfinal,nb_clusters_target) 
        
        # init
        self.init_weights_Graph_OurConvNet(Voc,D,Hfinal,nb_clusters_target,1)
        
        # print
        print('\nnb of hidden layers=',L)
        print('dim of layers (w/ embed dim)=',net_layers_extended)      
        print('\n')
        
        # class variables
        self.L = L
        self.net_layers_extended = net_layers_extended      
        self.flag_task = flag_task
        
        
    def init_weights_Graph_OurConvNet(self, Fin_enc, Fout_enc, Fin_fc, Fout_fc, gain):

        scale = gain* np.sqrt( 2.0/ Fin_enc )
        self.encoder.weight.data.uniform_(-scale, scale)  
        scale = gain* np.sqrt( 2.0/ Fin_fc )
        self.fc.weight.data.uniform_(-scale, scale)  
        self.fc.bias.data.fill_(0)  
    
            
    def forward(self, G):
        
        # signal
        x = G.signal  # V-dim
        x = Variable( torch.LongTensor(x).type(dtypeLong) , requires_grad=False)
           
        # encoder
        x_emb = self.encoder(x) # V x D
        
        # graph operators
        # Edge = start vertex to end vertex
        # E_start = E x V mapping matrix from edge index to corresponding start vertex
        # E_end = E x V mapping matrix from edge index to corresponding end vertex
        E_start = G.edge_to_starting_vertex
        E_end   = G.edge_to_ending_vertex 
        E_start = torch.from_numpy(E_start.toarray()).type(dtypeFloat)
        E_end = torch.from_numpy(E_end.toarray()).type(dtypeFloat) 
        E_start = Variable( E_start , requires_grad=False) 
        E_end = Variable( E_end , requires_grad=False) 
        
        # convnet cells  
        x = x_emb
        for layer in range(self.L//2):
            gnn_layer = self.gnn_cells[layer]            
            x = gnn_layer(x,E_start,E_end) # V x Hfinal
            
        # FC
        x = self.fc(x)   
        
        return x
        
        
    def loss(self, y, y_target, weight):
        
        loss = nn.CrossEntropyLoss(weight=weight.type(dtypeFloat))(y,y_target)
        
        return loss
       
        
    def update(self, lr):
                
        update = torch.optim.Adam( self.parameters(), lr=lr )
        
        return update
    
    
    def update_learning_rate(self, optimizer, lr):
   
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer
    
    
    def nb_param(self):

        return self.nb_param
    
    
    