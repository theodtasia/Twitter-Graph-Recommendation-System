import pickle

import networkx as nx
from networkx import relabel_nodes, write_gpickle

original_path = 'data/'
clean_data_path = 'data/clean_data/'
numOfGraphs = 62
nodeFeaturesDim = 7
Graph_ = 'day_graphs/Graph_'

class CleanData:
    def __init__(self):
        self.node_attributes = self.readPickleFile('node_attributes')
        self.graphs = self.loadDayGraphs()
        self.dropEdgesInvolvingFeaturelessNodes()
        self.correctIndex()
        self.saveToPickle()

    def readPickleFile(self, file_path):
        return pickle.load(open(original_path + file_path, 'rb'))

    def saveToPickle(self):
        with open(clean_data_path + 'node_attributes', 'wb') as f:
            pickle.dump(self.node_attributes, f)
        for i, graph in enumerate(self.graphs):
            write_gpickle(graph, f'{clean_data_path}{Graph_}{i}')

    def nodesWithAvailableFeatures(self):
        return self.node_attributes.keys()

    def checkForNullNodeFeatures(self):
        # no null values found
        for node, attributes in self.node_attributes.items():
            if len(attributes) < nodeFeaturesDim:
                print(node, attributes)
            for attr, value in attributes.items():
                if not isinstance(value, (int, str)):
                    print(node, attr, value)

    def loadDayGraphs(self):
        graphs = []
        for day in range(numOfGraphs + 1):
            day_graph = self.readPickleFile(f'{Graph_}{day}')
            graphs.append(day_graph)
        return graphs

    def dropEdgesInvolvingFeaturelessNodes(self):
        nodesToKeep = self.nodesWithAvailableFeatures()

        for day_graph in self.graphs:
            for v, u in day_graph.edges():
                if v not in nodesToKeep or u not in nodesToKeep:
                    day_graph.remove_edge(v, u)
            day_graph.remove_nodes_from(list(nx.isolates(day_graph)))

    def correctIndex(self):
        newIndexes = {
            old_index : new_index
            for new_index, old_index in enumerate(self.nodesWithAvailableFeatures())
        }
        self.node_attributes = {
            newIndexes[node] : attributes
            for node, attributes in self.node_attributes.items()
        }
        for graph in self.graphs:
            relabel_nodes(graph, newIndexes, copy=False)


