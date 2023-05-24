import argparse

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant, OneHotDegree
import torch_geometric.transforms as T

from utils.splitter import cv_random_split
from torch_geometric.utils import add_self_loops
from dataset import SBMDataset
from utils.sam import SAM
from utils.nsm import NSM

from model import MPGNN, GNN

datasets = [
    "COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K",
    "MUTAG", "PTC_MR", "NCI1", "PROTEINS"
]

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

train_sizes = {
    "COLLAB": 4500,
    "IMDB-BINARY": 900,
    "IMDB-MULTI": 1350,
    "REDDIT-BINARY": 1800, 
    "REDDIT-MULTI-5K": 4500,
    "MUTAG": 170, 
    "PTC_MR": 310, 
    "NCI1": 3699,
    "PROTEINS": 1002,
}

batch_sizes = {
    "COLLAB": 128,
    "IMDB-BINARY": 128,
    "IMDB-MULTI": 128,
    "REDDIT-BINARY": 128, 
    "REDDIT-MULTI-5K": 128,
    "MUTAG": 128, 
    "PTC_MR": 128, 
    "NCI1": 128,
    "PROTEINS": 128,
    "SBM": 128
}

def train(model, train_loader, optimizer, lr_scheduler, device):
    model.train()
    model.to(device)

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)

        # transform = T.ToSparseTensor(remove_edge_index=False)
        # data = transform(data)
        # data.adj_t = data.adj_t.to_symmetric()

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = F.nll_loss(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        lr_scheduler.step()

def test(model, loader, device):
    model.eval()
    model.to(device)

    correct = 0; total_loss = 0; size = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)

        # transform = T.ToSparseTensor(remove_edge_index=False)
        # data = transform(data)
        # data.adj_t = data.adj_t.to_symmetric()

        out = model(data.x, data.edge_index, data.batch)  
        loss = F.nll_loss(out, data.y)
        total_loss += loss * data.y.shape[0]
        size += data.y.shape[0]

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), total_loss.cpu().item()/size  # Derive ratio of correct predictions.

def main(args):
    dataset_name = args.dataset
    if dataset_name == "SBM":
        dataset = SBMDataset('data/SBM', num_graphs=200, num_nodes=100)
    elif "REDDIT" in dataset_name:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True, transform=T.Compose([Constant(value=1.0)]))
    elif dataset_name in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True, transform=T.Compose([OneHotDegree(max_degree=max_degrees[dataset_name])]) )
    else:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)

    if args.add_self_loop:
        for graph in dataset:
            graph.edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

    train_losses = []; test_losses = []
    train_accs = []; test_accs = []
    for fold_idx in range(args.folds):
        if fold_idx not in args.fold_idx:
            continue

        ''' Split dataset '''
        # permutations = np.random.permutation(200)
        # train_dataset, test_dataset = dataset[permutations[:50]], dataset[permutations[50:]]
        train_dataset, test_dataset = cv_random_split(dataset, fold_idx = fold_idx, seed = args.seed)

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes[dataset_name], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes[dataset_name], shuffle=False)

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
            gnn_type=args.model,
            use_bn=not args.no_bn)
        device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")

        ''' Start training for runs '''
        train_losses_run = []; test_losses_run = []
        train_accs_run = []; test_accs_run = []
        for run in range(1, args.runs+1):
            model.reset_parameters()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*int(args.epochs/4), gamma=0.5)
        
            tmp_train_losses = []; tmp_test_losses = []
            tmp_train_acces = []; tmp_test_acces = []

            checkpoint_dir = f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{fold_idx}_run_{run}_epoch_0" 
            torch.save(model.state_dict(), 
                checkpoint_dir + ".pth")
            for epoch in range(1, args.epochs+1):
                train(model, train_loader, optimizer, lr_scheduler, device)
                
                train_acc, train_loss = test(model, train_loader, device)
                test_acc, test_loss = test(model, test_loader, device)
                
                print(
                    f'Fold index: {fold_idx}, '
                    f'Run: {run}, '
                    f'Epoch: {epoch:03d}, '
                    f'Train loss: {train_loss:.4f}, '
                    f'Train acc: {train_acc:.4f}, '
                    f'Test loss: {test_loss:.4f}, '
                    f'Test acc: {test_acc:.4f}')
                
                if epoch % args.log_interval == 0:
                    tmp_train_losses.append(train_loss); tmp_test_losses.append(test_loss)
                    tmp_train_acces.append(train_acc); tmp_test_acces.append(test_acc)
                
                ''' Save checkpoint '''
                if epoch % 5 == 0:
                    checkpoint_dir = f"./saved/{args.dataset}_{args.model}_layer_{args.num_layers}_aggr_{args.aggr}_fold_{fold_idx}_run_{run}_epoch_{epoch}" 
                    torch.save(model.state_dict(), 
                        checkpoint_dir + ".pth")

            train_losses_run.append(np.array(tmp_train_losses)); test_losses_run.append(np.array(tmp_test_losses))
            train_accs_run.append(np.array(tmp_train_acces)); test_accs_run.append(np.array(tmp_test_acces))
        
        train_losses_run = np.array(train_losses_run).mean(axis=0); test_losses_run = np.array(test_losses_run).mean(axis=0)
        train_accs_run = np.array(train_accs_run).mean(axis=0); test_accs_run = np.array(test_accs_run).mean(axis=0)
        train_losses.append(train_losses_run); test_losses.append(test_losses_run)
        train_accs.append(train_accs_run); test_accs.append(test_accs_run)
    train_losses = np.array(train_losses); test_losses = np.array(test_losses)
    train_accs = np.array(train_accs); test_accs = np.array(test_accs)
    ''' Log training results '''
    for i in range(train_losses.shape[1]):
        log_epoch = (i+1)*args.log_interval
        print(
            f"Performance at epoch {log_epoch}: " +
            f"train loss {np.mean(train_losses[:, i]):.4f}+/-{np.std(train_losses[:, i]):.4f} " + 
            f"test loss {np.mean(test_losses[:, i]):.4f}+/-{np.std(test_losses[:, i]):.4f} " + 
            f"generalization error {np.mean(test_losses[:, i])-np.mean(train_losses[:, i]):.4f} " + 
            f"train accuracy {np.mean(train_accs[:, i]):.4f}+/-{np.std(train_accs[:, i]):.4f} " + 
            f"test accuracy {np.mean(test_accs[:, i]):.4f}+/-{np.std(test_accs[:, i]):.4f} " 
            )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--folds', type=int, default=10, help = "Seed for splitting the dataset.")
    parser.add_argument('--fold_idx', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--add_self_loop', action="store_true")

    ''' Training '''
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    ''' Model '''
    parser.add_argument('--model', type=str, default="gcn")
    parser.add_argument('--jk_type', type=str, default="last")
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--aggr', type=str, default="add")
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_bn', action="store_true")

    args = parser.parse_args()

    main(args)