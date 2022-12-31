import glob
import pickle
import sys
from os import mkdir
from os.path import exists


from preprocessing.clean_datasets import CleanData, clean_data_path, Graph_

negativeEdges = f'{clean_data_path}negative_edges/'
class FindNegativeEdges:

    def __init__(self):
        numOfGraphs = len(glob.glob(f'{clean_data_path}{Graph_}*'))
        if not exists(negativeEdges) or not exists(FindNegativeEdges._negativeEdgesFile(numOfGraphs - 1)):
            print("preprocessing")
            self._preproccessing()

    @staticmethod
    def retrieveGraphNegatives(day):
        return pickle.load(open(FindNegativeEdges._negativeEdgesFile(day), 'rb'))

    def _preproccessing(self):
        self.graphs = CleanData.loadDayGraphs()
        self.negativesPerNode = {node : [] for node in CleanData.readNodeAttributes()}

        for day, graph in enumerate(self.graphs):
            print(day)
            self._dayGraph_negativeEdges(graph, day)
            self._save_negativeGi(day)

    def _dayGraph_negativeEdges(self, graph, day):

        newNodes = set(graph.nodes()).difference(set(self.graphs[day-1].nodes())) if day > 0 \
                    else graph.nodes()

        for node in graph.nodes():
            self.negativesPerNode[node].extend(
                [new for new in newNodes
                if node < new and not (graph.has_edge(node, new) or graph.has_edge(new, node))]
            )

    def _save_negativeGi(self, day):
        if not exists(negativeEdges[:-1]):
            mkdir(negativeEdges[:-1])
        pickle.dump(
            self.negativesPerNode,
            open(FindNegativeEdges._negativeEdgesFile(day), 'wb'))

    @staticmethod
    def _negativeEdgesFile(day):
        return f'{negativeEdges}negativeG_{day}'

