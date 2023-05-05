import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T

from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, MLP, GINConv
from models import SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_sparse import SparseTensor

Layers = {
    "gcn": GCNConv,
    "graphsage": SAGEConv,
    "gin": GINConv
}

class CustomizedGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_self = Linear(in_channels, out_channels, bias=False)
        self.lin_nbr = Linear(out_channels, out_channels, bias=False)

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_nbr.reset_parameters()

    def forward(self, x, edge_index, node_features):
        out = self.propagate(edge_index, x=x)
        out = torch.tanh(out)
        out = F.relu(self.lin_self(node_features) + self.lin_nbr(out))
        return out

    def message(self, x_j):
        return torch.tanh(x_j)

class MPGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MPGNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.conv = CustomizedGraphConv(in_channels, hidden_channels)

        self.dropout = dropout
        ''' Graph classification'''
        self.pred_head = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.pred_head.reset_parameters()
    
    def forward(self, x, edge_index, batch, use_bn=False):
        # for i, conv in enumerate(self.convs):
        #     x = conv(x, edge_index)
        #     # if use_bn:
        #     #     x = self.bns[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        node_features = x
        hidden_features = torch.zeros(x.size(0), self.hidden_channels).to(x.device)
        for i in range(self.num_layers-1):
            hidden_features = self.conv(hidden_features, edge_index, node_features)
            hidden_features = F.relu(hidden_features)
            hidden_features = F.dropout(hidden_features, p=self.dropout, training=self.training)
        
        # Apply a readout pooling
        assert batch is not None
        hidden_features = global_mean_pool(hidden_features, batch)
        
        # Apply a linear classifier
        hidden_features = F.dropout(hidden_features, p=self.dropout, training=self.training)
        hidden_features = self.pred_head(hidden_features)
        return hidden_features.log_softmax(dim=-1) 


class GNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                use_bn = True, JK="last", graph_pooling = "mean", gnn_type = "gin", aggr="add"):
        super(GNN, self).__init__()
        
        self.dropout = dropout
        self.use_bn = use_bn
        self.JK = JK
        self.graph_pooling = graph_pooling
        self.gnn_type = gnn_type

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if gnn_type in ["gcn", "graphsage"]:
            self.convs.append(Layers[gnn_type](in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers-1):
                self.convs.append(Layers[gnn_type](hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        elif gnn_type == "gin":
            mlp = MLP([in_channels, hidden_channels, hidden_channels], batch_norm=True)
            self.convs.append(GINConv(nn=mlp, train_eps=False, aggr=aggr))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers-1):
                mlp = MLP([hidden_channels, hidden_channels, hidden_channels], batch_norm=True)
                self.convs.append(GINConv(nn=mlp, train_eps=False, aggr=aggr))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        else:
            print("Nonvalid GNN type")
        
        if self.JK == "concat":
            self.pred_head = Linear(hidden_channels*(num_layers)+in_channels, out_channels)
        else:
            self.pred_head = Linear(hidden_channels, out_channels)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        # elif graph_pooling == "attention":
        #     if self.JK == "concat":
        #         self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
        #     else:
        #         self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        # elif graph_pooling[:-1] == "set2set":
        #     set2set_iter = int(graph_pooling[-1])
        #     if self.JK == "concat":
        #         self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
        #     else:
        #         self.pool = Set2Set(emb_dim, set2set_iter)
        # else:
        #     raise ValueError("Invalid graph pooling type.")

        # #For graph-level binary classification
        # if graph_pooling[:-1] == "set2set":
        #     self.mult = 2
        # else:
        #     self.mult = 1
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.pred_head.reset_parameters()

    def forward(self, x, adj_t, batch):
        h_list = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h_list.append(x)

        # Different implementations of JK-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        # elif self.JK == "max":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        # elif self.JK == "sum":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        x = node_representation

        # Apply a readout pooling
        assert batch is not None
        x = self.pool(x, batch)
        
        # Apply a linear classifier
        x = self.pred_head(x)
        return x.log_softmax(dim=-1) 