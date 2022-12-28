import pickle

import torch
from clean_datasets import clean_data_path, Graph_
from features_extraction import FeaturesExtraction
from torch_geometric.data import Data

clean_data_path = f'../{clean_data_path}'
device = 'cpu'
class Dataset:

    def __init__(self):
        self.day = 0
        x = self.get_node_attributes()
        edge_index = self.get_day_graph_edge_index(self.day)
        graph = Data(x, edge_index)
        print(graph)

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
        edges = torch.cat([edges, edges[[1, 0]]], dim=1)
        return edges



Dataset()