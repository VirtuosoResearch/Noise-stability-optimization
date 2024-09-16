import random
import torch
import networkx as nx
import numpy as np
from networkx.generators import barabasi_albert_graph, stochastic_block_model
from torch_geometric.data import Data

def generate_sbm_graph(node_num, alpha, beta, feature_dim=16):
    ''' Generate a two block model graph '''
    sizes_list = [int(node_num*0.5), int(node_num*0.5)] 
    network = stochastic_block_model(
        sizes = sizes_list,  p = [[alpha, beta], [beta, alpha]], seed=42,
    )

    node_list = np.arange(0, sum(sizes_list))

    edge_list = np.array(list(network.edges)); inds = [1, 0]
    reverse_edge_list = edge_list[:, inds]
    edge_list = np.concatenate([edge_list, reverse_edge_list], axis=0)
    edge_list = np.transpose(edge_list)

    ''' generate node features based on node degrees '''
    dim = feature_dim
    node_features = np.eye(node_num)
    # node_features = np.zeros((node_num, dim))

    # mean_1 = np.zeros(dim); mean_1[:int(dim/2)] = 1
    # mean_2 = np.zeros(dim); mean_2[int(dim/2):] = 1
    # cov = np.eye(dim)*1e-3
    # node_features[node_list[:sizes_list[0]], :] = np.random.multivariate_normal(mean=mean_1, cov=cov, size=sizes_list[0])
    # node_features[node_list[sizes_list[0]:], :] = np.random.multivariate_normal(mean=mean_2, cov=cov, size=sizes_list[1])

    edge_index = torch.tensor(edge_list, dtype=torch.long)
    x = torch.Tensor(node_features)

    label = torch.LongTensor([1 if alpha > beta else 0])
    data = Data(x=x, edge_index=edge_index, y=label)
    return data

def generate_ba_graph(node_num, edge_density, feature_dim=16, class_ratio=0.6, train_ratio=0.1):

    network = barabasi_albert_graph(n = node_num, m = edge_density)

    edge_list = np.array(list(network.edges)); inds = [1, 0]
    reverse_edge_list = edge_list[:, inds]
    edge_list = np.concatenate([edge_list, reverse_edge_list], axis=0)
    edge_list = np.transpose(edge_list)

    node_degrees = list(network.degree)
    node_degrees.sort(key=lambda x: x[-1])
    node_idxes = np.array([pair[0] for pair in node_degrees])

    ''' generate node features based on node degrees '''
    dim = feature_dim

    node_features = np.zeros((node_num, dim))
    class_len_1 = int(class_ratio*node_num)
    class_len_2 = node_num - class_len_1

    mean_1 = np.zeros(dim); mean_1[:int(dim/2)] = 1
    mean_2 = np.zeros(dim); mean_2[int(dim/2):] = 1
    cov = np.eye(dim)*1e-2
    node_features[node_idxes[:class_len_1], :] = np.random.multivariate_normal(mean=mean_1, cov=cov, size=class_len_1)
    node_features[node_idxes[class_len_1:], :] = np.random.multivariate_normal(mean=mean_2, cov=cov, size=class_len_2)

    node_labels = np.zeros(node_num)
    node_labels[node_idxes[class_len_1:]] = 1


    edge_index = torch.tensor(edge_list, dtype=torch.long)
    x = torch.Tensor(node_features)
    y = torch.LongTensor(node_labels)
    indexes = np.random.permutation(edge_index.size()[1])

    shuffle = np.random.permutation(node_num)
    train_len, val_len, test_len = int(node_num*train_ratio), int(node_num*0.1), int(node_num*(1-train_ratio-0.1))
    train_mask = shuffle[:train_len]
    val_mask = shuffle[train_len:train_len+val_len]
    test_mask = shuffle[train_len+val_len:train_len+val_len+test_len]

    data = Data(x=x, edge_index=edge_index[:, indexes], y = y,  
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data