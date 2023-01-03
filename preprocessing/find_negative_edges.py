import glob
import pickle
from os import mkdir
from os.path import exists

import networkx as nx
import torch

from preprocessing.clean_datasets import CleanData, clean_data_path, Graph_
from recommendation_task.utils import dotdict

negativeEdges = f'{clean_data_path}negative_edges/'
class FindNegativeEdges:

    def __init__(self):
        numOfGraphs = len(glob.glob(f'{clean_data_path}{Graph_}*'))
        if not exists(negativeEdges) or not exists(FindNegativeEdges._negativeEdgesFile(numOfGraphs - 1)):
            print("preprocessing")
            self._preproccessing()


    def retrieveGraphNegatives(self, day):
        return dotdict(pickle.load(open(FindNegativeEdges._negativeEdgesFile(day), 'rb')))

    def _preproccessing(self):
        self.graphs = CleanData.loadDayGraphs()
        self.merged = nx.Graph()

        for day, graph in enumerate(self.graphs):
            print(day)
            self._save_negativeGi(
                self._dayGraph_negativeEdges(graph), day
            )

    def _dayGraph_negativeEdges(self, graph):

        test_edges = torch.tensor(list(graph.edges()), dtype=torch.long).T
        test_edges = torch.cat([test_edges, test_edges[[1, 0]]], dim=1)

        targets = [1] * test_edges.shape[1]
        indexes = test_edges[0].tolist()

        self.merged = nx.compose(self.merged, graph)
        negative_tests = []
        for v in graph.nodes():
            negatives = [(v, u) for u in graph.nodes()
                         if not self.merged.has_edge(v, u) and not self.merged.has_edge(u, v)]
            negative_tests.extend(negatives)
            targets.extend([0] * len(negatives))
            indexes.extend([v] * len(negatives))

        test_edges = {
            'test_edges': torch.cat([test_edges, torch.tensor(negative_tests, dtype=torch.long).T],
                                    dim=1),
            'targets': torch.tensor(targets, dtype=torch.bool),
            'indexes': torch.tensor(indexes, dtype=torch.long)
        }
        return test_edges

    def _save_negativeGi(self, test_edges, day):
        if not exists(negativeEdges[:-1]):
            mkdir(negativeEdges[:-1])
        pickle.dump(
            test_edges,
            open(FindNegativeEdges._negativeEdgesFile(day), 'wb'))

    @staticmethod
    def _negativeEdgesFile(day):
        return f'{negativeEdges}negativeG_{day}'

