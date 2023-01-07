import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from preprocessing.clean_datasets import *
from preprocessing.clean_datasets import clean_data_path, Graph_
from preprocessing.features_extraction import FeaturesExtraction
from preprocessing.find_negative_edges import FindNegativeEdges
from recommendation_task.utils import numOfGraphs


INIT_DAY = 0
# change to True to include centrality based feats
EXTRACT_TOPOL_ATTRS = True
# feature extractor configuration (for ex, scale arg value) haven't changed
# since last run
SAME_ATTRS_CONFIG = True

class Dataset:

    def __init__(self, device):
        self.device = device

        # will be incremented to 0 when thw first graph (Graph_{INIT_DAY}) is loaded
        self.day = INIT_DAY - 1
        self.numOfGraphs = numOfGraphs()

        self.featureExtractor = FeaturesExtraction(extract_topological_attr=EXTRACT_TOPOL_ATTRS,
                                                   load_from_file= SAME_ATTRS_CONFIG)
        self.attr_dim = self.featureExtractor.attr_dim

        self.graph = Data(edge_index=torch.empty((2,0), dtype=torch.long, device=device))

    def has_next(self):
        # checked before loading graph
        # when day = -1, Graph_0 and Graph_1 will be loaded (therefore 2)
        return self.day + 2 < self.numOfGraphs


    def get_dataset(self):
        """
        :return:
            train_edges : positive (existing) edges for message passing and scoring (undirected tensor (2,M))
            test_edges : next day's edges (positive and negative) for testing.
                dotdict {test_edges : tensor (2, N), targets, indexes : tensors (N, 1)}
        """
        self._set_day()
        train_edges = self._to_undirected(self.graph.edge_index)
        test_edges = FindNegativeEdges().loadTestEdges(self.day + 1)
        return train_edges, test_edges


    def _set_day(self):
        self.day += 1
        day_edges = self._load_day_graph(self.day)
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


    def _load_day_graph(self, day):
        """
        :param day: to load Graph_{day}
        :return: edges : directed edge_index tensor
        """
        edges = pickle.load(open(f'{clean_data_path}{Graph_}{day}', 'rb')).edges()
        edges = torch.tensor(list(edges), dtype=torch.long, device=self.device).T
        return edges


    def _to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)



    def negative_sampling(self):
        neg_edge_index = negative_sampling(
            edge_index=self.graph.edge_index,               # positive edges
            num_nodes=self.max_node,                        # max node index in graph
            num_neg_samples=self.graph.edge_index.size(1))  # num of negatives = num of positives
        return self._to_undirected(neg_edge_index)



