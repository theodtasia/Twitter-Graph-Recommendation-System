import glob

import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from preprocessing.clean_datasets import *
from preprocessing.clean_datasets import clean_data_path, Graph_
from preprocessing.features_extraction import FeaturesExtraction
from preprocessing.find_negative_edges import FindNegativeEdges


class Dataset:

    def __init__(self, device):
        self.device = device

        self.day = 0
        self.numOfGraphs = len(glob.glob(f'{clean_data_path}{Graph_}*'))

        # load first day graph
        self.graph = Data(x = self._get_node_attributes(),
                          edge_index = self._get_day_graph_edge_index(self.day))
        self.graph.to(device=self.device)
        self.max_node = torch.max(self.graph.edge_index).item()

        self.attr_dim = self.graph.x.shape[1]


    def has_next(self):
        return self.day + 1 < self.numOfGraphs


    def _increment_day(self):
        self.day += 1
        self.max_node = torch.max(self.graph.edge_index).item()
        self.next_day_edges = self._get_day_graph_edge_index(self.day)
        self.graph.edge_index = torch.cat([self.graph.edge_index,
                                           self.next_day_edges], dim=1)


    def _get_node_attributes(self):
        x = CleanData.readNodeAttributes()
        x = FeaturesExtraction(x, extract_attr=True, turn_to_numeric=True, scale=True).attributes   # dataframe
        x = x.values.tolist()
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x

    def _get_day_graph_edge_index(self, day):
        # get directed edges of graph_day as tensor
        edges = pickle.load(open(f'{clean_data_path}{Graph_}{day}', 'rb')).edges()
        edges = torch.tensor(list(edges), dtype=torch.long, device=self.device).T
        return edges


    def _to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)


    def get_dataset(self):

        train_edges = self._to_undirected(self.graph.edge_index)
        test_edges = FindNegativeEdges().retrieveGraphNegatives(self.day + 1)
        self._increment_day()

        return train_edges, test_edges

    def negative_sampling(self):
        neg_edge_index = negative_sampling(
            edge_index=self.graph.edge_index,               # positive edges
            num_nodes=self.max_node,                        # max node index in graph
            num_neg_samples=self.graph.edge_index.size(1))  # num of negatives = num of positives
        return self._to_undirected(neg_edge_index)



"""
    def get_dataset(self):
        
        edge_sets = self._split_edges()

        g_train = dotdict({'msg_pass' : edge_sets[0],
                            'supervision' : edge_sets[1],
                            'negative' : edge_sets[2]})
        self.next_day_edges = self._get_day_graph_edge_index(self.day + 1)
        self._increment_day()

        return g_train, self._to_undirected(self.next_day_edges)
        
        
        def _split_edges(self, msg_pass_ratio=.6):
        transform = RandomLinkSplit(is_undirected=False, num_val= 1 - msg_pass_ratio, num_test=0,
                                    add_negative_train_samples=False, neg_sampling_ratio=0)
        msg_pass, supervision, _ = (self._to_undirected(split.edge_label_index)
                                    for split in transform(self.graph))
        neg = self.negative_sampling_method2()
        return msg_pass, supervision, neg


    def _negative_sampling(self, supervision):
        nodes = list(set(v.item() for v in self.graph.edge_index.reshape(-1)))
        positive_edges = set((v.item(),u.item()) for v, u in zip(self.graph.edge_index[0],
                                                                 self.graph.edge_index[1]))
        negative_samples = set()
        print(supervision.shape)
        for v in supervision[0]:
            v = v.item()
            random.shuffle(nodes)
            for u in nodes:
                if u != v and (u,v) not in negative_samples \
                and (v,u) not in positive_edges and (u,v) not in positive_edges:
                    negative_samples.add((v, u))
                    break
        return self._to_undirected(
               self._to_directed_edge_index(negative_samples))

"""