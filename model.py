import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.sparse as sp

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
#        prop_state = prop_state.to(torch.float32)
#        annotation = annotation.to(torch.float32)
#        A = A.to(torch.float32)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)

        return output

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
        x1 = torch.sigmoid(x1)
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
        x1 = torch.sigmoid(x1)
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

    def __init__(self, opt):

        super(Graph_OurConvNet, self).__init__()

        # parameters
        #flag_task = task_parameters['flag_task']
        Voc = opt.vocab
        D = opt.D
        nb_clusters_target = opt.nb_clusters_target
        H = opt.H
        L = opt.n_steps
        if opt.self_loop:
            self.self_loop = True
        else:
            self.self_loop = False
        if opt.cuda:
            #print('cuda available')
            self.dtypeFloat = torch.cuda.DoubleTensor
            self.dtypeLong = torch.cuda.LongTensor
            #torch.cuda.manual_seed(1)
        else:
            #print('cuda not available')
            self.dtypeFloat = torch.DoubleTensor
            self.dtypeLong = torch.LongTensor
            #torch.manual_seed(1)

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
        #self.flag_task = flag_task


    def init_weights_Graph_OurConvNet(self, Fin_enc, Fout_enc, Fin_fc, Fout_fc, gain):

        scale = gain* np.sqrt( 2.0/ Fin_enc )
        self.encoder.weight.data.uniform_(-scale, scale)
        scale = gain* np.sqrt( 2.0/ Fin_fc )
        self.fc.weight.data.uniform_(-scale, scale)
        self.fc.bias.data.fill_(0)


    def forward(self, prop_state, annotation, A):
        
        n_nodes = len(annotation[0])

        # signal
        x = annotation[0].reshape(n_nodes)  # V-dim
        x = x.to(torch.long)
        x = Variable( self.dtypeLong(x).type(self.dtypeLong) , requires_grad=False)


        # encoder
        x_emb = self.encoder(x) # V x D
        
        # adj_matrix
        A = A[0].cpu().numpy()
        n_nodes = A.shape[0]
        n_col = A.shape[1]
        A_left = A[:,:int(n_col/2)]
        A_right = A[:,int(-n_col/2):]
        A_new = np.where(A_left != 1, A_right, A_left)

            
#        edge_types = torch.tensor([[x//A_new.shape[0] + 1 for x in range(A_new.shape[1])]] * A_new.shape[0], device=A_new.device, dtype=torch.float64)
#        A_new = torch.where(A_new == 1, edge_types, A_new)
#
#        
#        W_coo=sp.coo_matrix(A_new)
#        nb_edges=W_coo.nnz
#        nb_vertices=A_new.shape[0]
#        edge_to_starting_vertex=sp.coo_matrix( ( W_coo.data ,(np.arange(nb_edges), W_coo.row) ),
#                                               shape=(nb_edges, nb_vertices) )
#        new_col = np.where(W_coo.col >= nb_vertices, W_coo.col % nb_vertices, W_coo.col)
#        edge_to_ending_vertex=sp.coo_matrix( ( W_coo.data ,(np.arange(nb_edges), new_col) ),
#                                               shape=(nb_edges, nb_vertices) )
            
        edge_types = np.array([[x//A_new.shape[0] + 1 for x in range(A_new.shape[1])]] * A_new.shape[0])
        A_new = np.where(A_new == 1, edge_types, A_new)
        
        # self loop
        if self.self_loop:
            for i in range(A_new.shape[1]):
                A_new[i%A_new.shape[0],i]=i//A_new.shape[0]+1

        
        W_coo=sp.coo_matrix(A_new)
        nb_edges=W_coo.nnz
        nb_vertices=A_new.shape[0]
        edge_to_starting_vertex=sp.coo_matrix( ( W_coo.data ,(np.arange(nb_edges), W_coo.row) ),
                                               shape=(nb_edges, nb_vertices) )
        new_col = np.where(W_coo.col >= nb_vertices, W_coo.col % nb_vertices, W_coo.col)
        edge_to_ending_vertex=sp.coo_matrix( ( W_coo.data ,(np.arange(nb_edges), new_col) ),
                                               shape=(nb_edges, nb_vertices) )

        # graph operators
        # Edge = start vertex to end vertex
        # E_start = E x V mapping matrix from edge index to corresponding start vertex
        # E_end = E x V mapping matrix from edge index to corresponding end vertex
        E_start = edge_to_starting_vertex
        E_end   = edge_to_ending_vertex
        E_start = torch.from_numpy(E_start.toarray()).type(self.dtypeFloat)
        E_end = torch.from_numpy(E_end.toarray()).type(self.dtypeFloat)
        E_start = Variable( E_start , requires_grad=False)
        E_end = Variable( E_end , requires_grad=False)

        # convnet cells
        x = x_emb
        for layer in range(self.L//2):
            gnn_layer = self.gnn_cells[layer]
            x = gnn_layer(x,E_start,E_end) # V x Hfinal

        # FC
        x = self.fc(x)
        x = x.view(-1, nb_vertices)

        return x


    def loss(self, y, y_target, weight):

        loss = nn.CrossEntropyLoss(weight=weight.type(self.dtypeFloat))(y,y_target)

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
