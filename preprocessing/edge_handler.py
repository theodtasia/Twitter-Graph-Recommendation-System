import pickle
from os import mkdir
from os.path import exists

import networkx as nx
import torch

from preprocessing.clean_datasets import CleanData
from other.FILE_PATHS import TEST_EDGES_PATH, EDGE_ATTRIBUTES_PATH, EDGE_ATTRIBUTES_DIM
from other.utils import dotdict


class EdgeHandler:

    def __init__(self, extract_edge_attrs = False):

        find_test_edges = not exists(TEST_EDGES_PATH[:-1])
        self.find_edge_attrs = not exists(EDGE_ATTRIBUTES_PATH[:-1]) and extract_edge_attrs

        if find_test_edges:
            mkdir(TEST_EDGES_PATH[:-1])
        if self.find_edge_attrs:
            mkdir(EDGE_ATTRIBUTES_PATH[:-1])

        if find_test_edges or self.find_edge_attrs:
            print(find_test_edges, self.find_edge_attrs)
            print("preprocessing")
            self._preproccessing()


    def loadTestEdges(self, day):
        test_edges = dotdict(pickle.load(open(EdgeHandler._negativeEdgesFile(day), 'rb')))
        edge_attributes = self.loadEdgeAttributes(day)
        test_edges.attributes = self.lookup_edge_attributes(edge_attributes, test_edges.edges)

        return test_edges

    def loadEdgeAttributes(self, day):
        try:
            return pickle.load(open(EdgeHandler._edgeAttributesFile(day), 'rb'))
        except Exception:
            return None

    def _preproccessing(self):
        self.graphs = CleanData.loadDayGraphs()
        self.merged = nx.Graph()

        for day, graph in enumerate(self.graphs):
            print(day)
            self.merged = nx.compose(self.merged, graph)
            if not exists(self._negativeEdgesFile(day)):
                self._save_negativeGi(
                    self._dayGraph_negativeEdges(graph), day
                )
            if not exists(self._edgeAttributesFile(day)) and self.find_edge_attrs:
                self._save_edge_attributes(day)

    def _dayGraph_negativeEdges(self, graph):

        test_edges = torch.tensor([sorted(edge) for edge in graph.edges()], dtype=torch.long).T
        targets = [1] * test_edges.shape[1]
        indexes = test_edges[0].tolist()

        negative_tests = []
        for v in graph.nodes():
            negatives = [(v, u) for u in graph.nodes()
                         if v < u and not self.merged.has_edge(v, u) and not self.merged.has_edge(u, v)]
            negative_tests.extend(negatives)
            targets.extend([0] * len(negatives))
            indexes.extend([v] * len(negatives))

        test_edges = {
            'edges': torch.cat([test_edges, torch.tensor(negative_tests, dtype=torch.long).T],
                                    dim=1),
            'targets': torch.tensor(targets, dtype=torch.bool),
            'indexes': torch.tensor(indexes, dtype=torch.long)
        }
        return test_edges


    def _save_negativeGi(self, test_edges, day):
        pickle.dump(
            test_edges,
            open(EdgeHandler._negativeEdgesFile(day), 'wb'))

    @staticmethod
    def _negativeEdgesFile(day):
        return f'{TEST_EDGES_PATH}negativeG_{day}'


    def _save_edge_attributes(self, day):

        attributes_functs = [nx.jaccard_coefficient, nx.resource_allocation_index, nx.preferential_attachment]
        attributes = [
            list(nx_funct(self.merged)) for nx_funct in attributes_functs
        ]
        attributes = {
            EdgeHandler.edge_key(v,u): [jaccard, res_alloc, attach]
            for (v, u, jaccard), (_, _, res_alloc), (_, _, attach) in zip(*attributes)
        }
        pickle.dump(
            attributes,
            open(EdgeHandler._edgeAttributesFile(day), 'wb'))

    @staticmethod
    def edge_key(v, u):
        return min(v, u), max(v, u)

    @staticmethod
    def lookup_edge_attributes(attributes, edge_index):
        if attributes is None:
            return None
        attributes = [
            attributes.get(EdgeHandler.edge_key(v, u),
                           [0] * EDGE_ATTRIBUTES_DIM)
            for (v, u) in edge_index.T
        ]
        return torch.tensor(attributes, dtype=torch.float32)

    @staticmethod
    def _edgeAttributesFile(day):
        return f'{EDGE_ATTRIBUTES_PATH}edge_attrsG_{day}'
