import pickle
from random import shuffle

import torch
from torch_geometric.utils import negative_sampling

from clean_datasets import clean_data_path, Graph_
from features_extraction import FeaturesExtraction
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit


clean_data_path = f'../{clean_data_path}'
device = 'cpu'
class Dataset:

    def __init__(self):
        self.day = 0

        self.graph = Data(x = self.get_node_attributes(),
                          edge_index = self.get_day_graph_edge_index(self.day))


    def increment_day(self):
        self.day += 1

    def get_node_attributes(self):
        x = pickle.load(open(f'{clean_data_path}node_attributes', 'rb'))
        x = FeaturesExtraction(x, turn_to_numeric=True).attributes   # dataframe
        x = x.values.tolist()
        x = torch.tensor(x, device=device, dtype=torch.float32)
        return x

    def get_day_graph_edge_index(self, day):
        # get undirected edges of graph_day as tensor
        edges = pickle.load(open(f'{clean_data_path}{Graph_}{day}', 'rb')).edges()
        edges = torch.tensor(list(edges), device=device, dtype=torch.long).T
        edges = self.turn_to_undirected(edges)
        return edges

    def turn_to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)


    def split_edges(self, msg_pass_ratio=.5):
        transform = RandomLinkSplit(is_undirected=True, num_val= 1 - msg_pass_ratio, num_test=0,
                                    add_negative_train_samples=False, neg_sampling_ratio=0)
        msg_pass, supervision, _ = (split.edge_label_index for split in transform(self.graph))
        # print(self.graph.edge_index)
        # print(msg_pass, '\n', supervision)
        self.negative_sampling(self.graph.edge_index, msg_pass, supervision)

    def negative_sampling(self, init, msg_pass, supervision):
        nodes = list(set(v.item() for v in init.view(-1)))
        print(nodes)
        for v in supervision[0]:
            shuffle(nodes)
            neg = None
            for u in nodes:
                if not ()






Dataset().split_edges()