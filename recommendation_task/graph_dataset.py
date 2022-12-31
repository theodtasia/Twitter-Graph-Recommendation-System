import glob
import random
from random import Random, choice, seed

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from preprocessing.clean_datasets import *
from preprocessing.clean_datasets import clean_data_path, Graph_
from preprocessing.features_extraction import FeaturesExtraction
from preprocessing.find_negative_edges import FindNegativeEdges
from utils import dotdict, device, set_seed


class Dataset:

    def __init__(self):
        self.device = device()
        set_seed()

        self.day = 0
        self.numOfGraphs = len(glob.glob(f'{clean_data_path}{Graph_}*'))
        self.next_day_edges = None

        # load first day graph
        self.graph = Data(x = self._get_node_attributes(),
                          edge_index = self._get_day_graph_edge_index(self.day))
        self.graph.to(device=self.device)
        self.calledNegativePicker = 0
        self.attr_dim = self.graph.x.shape[1]


    def has_next(self):
        return self.day < self.numOfGraphs


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
        neg = self.negative_sampling_method2()
        print(neg)
        return msg_pass, supervision, neg


    def _negative_sampling(self, supervision):
        self.calledNegativePicker += 1
        seed(self.calledNegativePicker)
        negative_samples = set()
        print(supervision.shape)

        for v in supervision[0]:
            v = v.item()
            negativesV = self.negativesPerNode[v]

            if len(negativesV) > 0:
                already_selected = True
                i = 0
                while already_selected and i <= len(negativesV):
                    i += 1
                    u = negativesV[random.randint(0, len(negativesV) - 1)]
                    already_selected = (u,v) in negative_samples or (v,u) in negative_samples

                    if not already_selected:
                        negative_samples.add((v, u))

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
            num_nodes=torch.max(self.graph.edge_index).item(),  # number of nodes
            num_neg_samples=self.graph.edge_index.size(1))  # number of neg_sample equal to number of pos_edges
        return self._to_undirected(neg_edge_index)

ds = Dataset()
while ds.has_next():
    ds.get_dataset()
    print(ds.day, ds.graph)

