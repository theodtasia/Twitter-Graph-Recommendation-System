import glob

import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from preprocessing.clean_datasets import *
from other.FILE_PATHS import CLEAN_DATA_PATH, Graph_, DAY_NODE_ATTRS_PATH, EDGE_ATTRIBUTES_PATH
from preprocessing.features_extraction import FeaturesExtraction
from preprocessing.edge_handler import EdgeHandler
from other.utils import numOfGraphs, dotdict

INIT_DAY = 0

class Dataset:

    def __init__(self, args):
        self.device = args.device

        # will be incremented to 0 when the first graph (Graph_{INIT_DAY}) is loaded
        self.day = INIT_DAY - 1
        self.numOfGraphs = numOfGraphs()
        self.last_day = self._last_day()

        self.featureExtractor = FeaturesExtraction(extract_topological_attr=EXTRACT_TOPOL_ATTRS,
                                                   load_from_file= SAME_ATTRS_CONFIG)
        self.edgeHandler = EdgeHandler()

        self.attr_dim = self.featureExtractor.attr_dim

        self.graph = Data(edge_index=torch.empty((2,0), dtype=torch.long, device=device))

    def has_next(self):
        # checked before loading graph
        # when day = -1, Graph_0 and Graph_1 will be loaded (therefore 2)
        return self.day + 2 < self.last_day

    def _last_day(self):

        last_day_with_centrality_attrs = len(glob.glob(f'{DAY_NODE_ATTRS_PATH}nodeAttrsG_*')) \
            if EXTRACT_TOPOL_ATTRS and exists(DAY_NODE_ATTRS_PATH[:-1]) else self.numOfGraphs+1

        last_day_with_edge_attrs = len(glob.glob(f'{EDGE_ATTRIBUTES_PATH}edge_attrsG_*')) \
            if EXTRACT_EDGE_ATTRIBUTES and exists(EDGE_ATTRIBUTES_PATH[:-1]) else self.numOfGraphs+1

        return min(self.numOfGraphs, last_day_with_centrality_attrs, last_day_with_edge_attrs)

    def get_dataset(self):
        """
        :return:
            train_edges : positive (existing) edges for message passing and scoring (undirected tensor (2,M))
            test_edges : next day's edges (positive and negative) for testing.
                dotdict {edges : tensor (2, N),
                        attributes : (N, edge attributes dim)
                        targets, indexes : tensors (N, 1)}
        """
        self._set_day()
        train_edges = self._to_undirected(self.graph.edge_index)
        train_edges = dotdict({'edges':train_edges,
                              'attributes': self.edgeHandler.lookup_edge_attributes(self.edge_attributes,
                                                                                    train_edges)})
        test_edges = self.edgeHandler.loadTestEdges(self.day + 1)
        return train_edges, test_edges


    def _set_day(self):
        self.day += 1
        day_edges = self._load_day_graph_edges(self.day)
        self.graph.edge_index = torch.cat([self.graph.edge_index,
                                           day_edges], dim=1)
        self.max_node = torch.max(self.graph.edge_index).item()
        self._update_node_attributes()


    def _update_node_attributes(self):
        # load feature vector x for the first day, or update daily if centrality based feats are included
        # if not, do nothing (x stays as it is every day)
        if self.day == INIT_DAY or EXTRACT_TOPOL_ATTRS:
            # dataframe
            x = self.featureExtractor.loadDayAttributesDataframe(self.day)
            x = x.values.tolist()
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            self.graph.x = x


    def _load_day_graph_edges(self, day):
        """
        :param day: to load Graph_{day}
        :return: edges : directed edge_index tensor
        + load current day edge attributes         """
        edges = pickle.load(open(f'{CLEAN_DATA_PATH}{Graph_}{day}', 'rb')).edges()
        edges = torch.tensor(list(edges), dtype=torch.long, device=self.device).T
        self.edge_attributes = self.edgeHandler.loadEdgeAttributes(self.day)
        return edges


    def _to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)



    def negative_sampling(self):
        neg_edge_index = negative_sampling(
            edge_index=self.graph.edge_index,               # positive edges
            num_nodes=self.max_node,                        # max node index in graph
            num_neg_samples=self.graph.edge_index.size(1))  # num of negatives = num of positives
        neg_edge_index = self._to_undirected(neg_edge_index)
        return dotdict({
            'edges' : neg_edge_index,
            'attributes' : EdgeHandler.lookup_edge_attributes(self.edge_attributes, neg_edge_index)
        })

