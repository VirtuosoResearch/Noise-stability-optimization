import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from utils.random_graphs import generate_sbm_graph

class SBMDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
        num_graphs=200, num_nodes=100, feature_dim=32):

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return 2

    @property
    def num_node_features(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        np.random.seed(42); random.seed(42)
        for i in range(self.num_graphs):
            alpha = np.random.uniform(0, 1)
            data = generate_sbm_graph(self.num_nodes, alpha=alpha*20/self.num_nodes, beta=(1-alpha)*20/self.num_nodes, feature_dim=self.feature_dim)
            data_list.append(data)
            print(data.y)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        