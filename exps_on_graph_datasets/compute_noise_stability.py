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

def compute_hessian_traces(model, loss, maxIter=100, tol=1e-3):
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

def compute_loss(model, loader, device, batch_num=250):
    model.eval()
    model.to(device)

    correct = 0; total_loss = 0; size = 0; batch_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)  
        loss = F.nll_loss(out, data.y)
        total_loss += loss * data.y.shape[0]
        size += data.y.shape[0]

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        batch_count += 1
        if batch_count > batch_num:
            break
    return correct / len(loader.dataset), total_loss.cpu().item()/size  # Derive ratio of correct predictions.

def perturbe_model_weights(state_dict, eps=0.001, use_neg = False, perturbation = {}):
    if not use_neg:
        perturbation = {} 
    for key, value in state_dict.items():
        if ("weight" in key and 'bn' not in key and 'norm' not in key) or ('att' in key):
            if use_neg:
                state_dict[key] -= perturbation[key]
            else:
                tmp_perturb = torch.randn_like(value)*eps
                state_dict[key] += tmp_perturb
                perturbation[key] = tmp_perturb
    return state_dict, perturbation

def deep_copy(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict.update({k:v.clone().detach()})
    return new_state_dict

def compute_stability(model, data_loader, device, eps = 1e-2, runs = 20, batch_num=250):
    _, loss_before = compute_loss(model, data_loader, device, batch_num=batch_num)
    state_dict_before = deep_copy(model.state_dict())
    print(loss_before)

    '''
    Calculate the perturbed loss
    '''
    differences = []
    for i in range(runs):
        differece = 0
        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, perturbations = perturbe_model_weights(state_dict_after, eps = eps)
        model.load_state_dict(state_dict_after)
        
        _, loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        # differences.append(differece.cpu().item())

        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, _ = perturbe_model_weights(state_dict_after, eps = eps, use_neg=True, perturbation = perturbations)
        model.load_state_dict(state_dict_after)
        
        _, loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        differences.append(differece/2)
    return differences

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
    
    data_loader = train_loader
    model.load_state_dict(
                torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{args.fold_idx}_run_{args.run}_epoch_{args.epoch}" + ".pth")
            )
    model = model.to(device)

    _, train_loss = compute_loss(model, train_loader, device, batch_num=10000)
    _, test_loss = compute_loss(model, test_loader, device, batch_num=10000)
    print("Training loss: {}".format(train_loss))
    print("Test loss: {}".format(test_loss))
    if args.compute_hessian_trace:
        traces = []
        sample_count = 0
        max_traces = np.zeros(3); max_loss = 0
        model.eval()
        for data in data_loader:
            model.load_state_dict(
                torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{args.fold_idx}_run_{args.run}_epoch_{args.epoch}" + ".pth")
            )
            model = model.to(device)
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            loss = F.nll_loss(out, data.y)

            layer_traces = compute_hessian_traces(model, loss)
            
            max_traces = np.maximum(max_traces, layer_traces)
            max_loss = np.maximum(max_loss, loss.cpu().item())

            traces.append(np.sum(layer_traces))
            # print(layer_traces)
            print("Current layer traces: {}".format(np.sum(layer_traces)))
            print("Traces mean: {}".format(np.mean(traces)))
            print("Max traces: {}".format(max_traces))
            print("Max loss: {}".format(max_loss))

            sample_count += 1
            if sample_count > args.sample_size:
                break
        
        start_state_dict = torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{args.fold_idx}_run_{args.run}_epoch_0" + ".pth", 
                                      map_location=device)
        state_dict = torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{args.fold_idx}_run_{args.run}_epoch_{args.epoch}" + ".pth",
                                map_location=device)

        weights_1 = []; weights_2 = []
        for key in state_dict.keys():
            if "weight" in key and "bn" not in key:
                weights_1.append(state_dict[key] - start_state_dict[key])
                weights_2.append(state_dict[key])

        norms_1 = (np.array([torch.norm(w).cpu().item() for w in weights_1]))**2
        norms_2 = (np.array([torch.norm(w).cpu().item() for w in weights_2]))**2
        print("Norms: {}".format(norms_1))
        print("Norms: {}".format(norms_2))
        norms = np.minimum(norms_1, norms_2)

        train_num = len(train_loader.dataset)
        bound = max_loss*np.math.sqrt((max_traces.sum()*norms.sum())/train_num)

        print("Empirical gap: {}".format(test_loss-train_loss))
        print("Bound: {}".format(bound))
    else:
        ''' Measure noise stability '''
        perturbs = np.arange(0.04, 0.061, 0.001)
        for perturb in perturbs:
            model.load_state_dict(
                torch.load(f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{args.fold_idx}_run_{args.run}" + ".pth")
            )
            model.to(device)

            diff_losses = compute_stability(model, data_loader, device, eps=perturb, batch_num=args.sample_size)
            print("Noise stability of {}: {:.4f} +/- {:.4f}".format(
                perturb, np.mean(diff_losses), np.std(diff_losses)
            ))  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--fold_idx', type=int, default=0, help = "Seed for splitting the dataset.")
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--sample_size', type=int, default=250)
    parser.add_argument('--epoch', type=int, default=0)

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


    parser.add_argument('--compute_hessian_trace', action='store_true')
    args = parser.parse_args()

    main(args)