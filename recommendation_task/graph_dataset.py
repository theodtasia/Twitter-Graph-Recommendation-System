import pickle
from random import shuffle
from time import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from preprocessing.clean_datasets import *
from preprocessing.features_extraction import FeaturesExtraction
from utils import dotdict, device


class Dataset:

    def __init__(self):
        self.device = device()

        self.day = 1
        self.next_day_edges = None

        self.graph = Data(x = self._get_node_attributes(),
                          edge_index = self._get_day_graph_edge_index(self.day))
        self.graph.to(device=self.device)
        self.attr_dim = self.graph.x.shape[1]


    def has_next(self):
        return self.day <= numOfGraphs

    def get_dataset(self):
        """
        Split training edges to training msg passing and supervision
        and perform negative sampling.
        :return: the training graph (Data obj), the test (next day) edges (2xE tensor)
        """
        edge_sets = self._split_edges()

        g_train = dotdict({'msg_pass' : edge_sets[0],
                            'supervision' : edge_sets[1],
                            'negative' : edge_sets[2]})
        self.next_day_edges = self._get_day_graph_edge_index(self.day + 1)
        self._increment_day()

        return g_train, self._to_undirected(self.next_day_edges)


    def _increment_day(self):
        self.day += 1

        self.graph.edge_index = torch.cat([self.graph.edge_index,
                                          self.next_day_edges], dim=1)



    def _get_node_attributes(self):
        x = pickle.load(open(f'{clean_data_path}node_attributes', 'rb'))
        x = FeaturesExtraction(x, extract_attr=True, turn_to_numeric=True, scale=True).attributes   # dataframe
        x = x.values.tolist()
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x

    def _get_day_graph_edge_index(self, day):
        # get directed edges of graph_day as tensor
        edges = pickle.load(open(f'{clean_data_path}{Graph_}{day}', 'rb')).edges()
        return self._to_directed_edge_index(edges)

    def _to_directed_edge_index(self, edges):
        return torch.tensor(list(edges), dtype=torch.long, device=self.device).T

    def _to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)


    def _split_edges(self, msg_pass_ratio=.6):
        transform = RandomLinkSplit(is_undirected=False, num_val= 1 - msg_pass_ratio, num_test=0,
                                    add_negative_train_samples=False, neg_sampling_ratio=0)
        msg_pass, supervision, _ = (self._to_undirected(split.edge_label_index)
                                    for split in transform(self.graph))
        neg = self._negative_sampling(supervision)
        return msg_pass, supervision, neg


    def _negative_sampling(self, supervision):
        nodes = list(set(v.item() for v in self.graph.edge_index.reshape(-1)))
        positive_edges = set((v.item(),u.item()) for v, u in zip(self.graph.edge_index[0],
                                                                 self.graph.edge_index[1]))
        negative_samples = set()
        print(supervision.shape)
        for v in supervision[0]:
            v = v.item()
            shuffle(nodes)
            for u in nodes:
                if u != v and (u,v) not in negative_samples \
                and (v,u) not in positive_edges and (u,v) not in positive_edges:
                    negative_samples.add((v, u))
                    break
        return self._to_undirected(
               self._to_directed_edge_index(negative_samples))


    def get_dataset_method2(self):

        negative = dotdict({'training' : self._to_undirected(self.graph.edge_index),
                                'negative' : self.negative_sampling_method2()})

        self.next_day_edges = self._get_day_graph_edge_index(self.day + 1)
        self._increment_day()

        return negative, self._to_undirected(self.next_day_edges)

    def negative_sampling_method2(self):
        neg_edge_index = negative_sampling(
            edge_index=self.graph.edge_index,  # positive edges
            num_nodes=self.graph.num_nodes,  # number of nodes
            num_neg_samples=self.graph.edge_index.size(1))  # number of neg_sample equal to number of pos_edges
        return self._to_undirected(neg_edge_index)
