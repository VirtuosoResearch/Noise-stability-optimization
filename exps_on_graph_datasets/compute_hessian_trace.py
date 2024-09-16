import argparse

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant, OneHotDegree
import torch_geometric.transforms as T
from collections import OrderedDict
from torch_geometric.nn.dense.linear import Linear

from utils.splitter import cv_random_split
from utils.norms import frob_norm, spectral_norm

from model import GNN

max_degrees = {
    "COLLAB": 491,
    "IMDB-BINARY": 135,
    "IMDB-MULTI": 88,
    "REDDIT-BINARY": 3062, 
    "REDDIT-MULTI-5K": 2011,
    "MUTAG": 4, 
    "PTC_MR": 4, 
    "NCI1": 4,
    "PROTEINS": 25,
}

def get_weights(model):
    layers = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear or type(module) == Linear:
            # print(name)
            layers.append(module.weight)
    return layers

def compute_hessians_trace(model, loss, maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer
    weights = get_weights(model)
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    layer_traces = []
    trace_vhv = []
    trace = 0.

    # Start Iterations
    for _ in range(maxIter):
        vs = [torch.randint_like(weight, high=2) for weight in weights]
            
        # generate Rademacher random variables
        for v in vs:
            v[v == 0] = -1

        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    return np.mean(np.array(layer_traces), axis=0)


def main(args):
    dataset_name = args.dataset
    if "REDDIT" in dataset_name:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True, transform=T.Compose([Constant(value=1.0)]))
    elif dataset_name in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True, transform=T.Compose([OneHotDegree(max_degree=max_degrees[dataset_name])]) )
    else:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)

    ''' Split dataset '''
    train_dataset, test_dataset = cv_random_split(dataset, fold_idx = args.fold_idx, seed = args.seed)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ''' Initialize model '''
    in_features = dataset.num_node_features if dataset.num_node_features != 0 else 1
    model = GNN(
        in_channels=in_features,
        hidden_channels=args.hidden, 
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        JK=args.jk_type, 
        graph_pooling=args.graph_pooling,
        aggr=args.aggr,
        gnn_type=args.model)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    model.load_state_dict(
            torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_fold_{args.fold_idx}_run_1" + ".pth")
        )
    model.to(device)

    if args.model == "gcn":
        max_layer_trace = np.zeros(shape=(args.num_layers+1, )) 
    elif args.model == "gin":
        max_layer_trace = np.zeros(shape=(args.num_layers*2+1, )) 
    
    model.eval()
    traces = []
    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        loss = F.nll_loss(out, data.y)

        layer_traces = compute_hessians_trace(model, loss)
        traces.append(np.sum(layer_traces))
        max_layer_trace = np.maximum(max_layer_trace, layer_traces)
        print(max_layer_trace)
    print("sum of hessian traces: {}".format(np.mean(traces)))

    # state_dict = torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_fold_{args.fold_idx}_run_{args.run}" + ".pth", map_location="cpu")
    # keys = [f'convs.{i}.lin.weight' for i in range(args.num_layers)]
    # keys.append('pred_head.weight')

    # frob_norms = []
    # spectral_norms = []

    # for weight_name in keys:
    #     weight = state_dict[weight_name]

    #     W_i_F = frob_norm(weight)
    #     W_i_2 = spectral_norm(weight, n_power_iterations=100)

    #     frob_norms.append(pow(W_i_F, 2))
    #     spectral_norms.append(pow(W_i_2.item(), 2))
    
    # train_size = len(train_dataset)
    # bound = 0
    # for i in range(args.num_layers+1):
    #     bound += np.math.sqrt(max_layer_trace[i]*frob_norms[i]/train_size)
    # print(f"Hessian-based bound: {bound}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--fold_idx', type=int, default=0, help = "Seed for splitting the dataset.")
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--batch_size', type=int, default=4)

    ''' Model '''
    parser.add_argument('--model', type=str, default="gcn")
    parser.add_argument('--jk_type', type=str, default="last")
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--aggr', type=str, default="add")
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    main(args)