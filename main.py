import pickle
import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib

from features_extraction import FeaturesExtraction


def loadGraphs():
    #read node attributes
    node_attributes = pickle.load(open('data/node_attributes', 'rb'))

    #read mainGraph to reduce time
    mainGraph = pickle.load(open('data/mergedGraph', 'rb'))

    # assign directory
    directory = 'data/day_graphs'

    #for one merged graph
    mergedGraph = nx.Graph()

    #for all graphs in a list
    all_graphs = np.array([])

    # iterate over files in that directory
    for filename in os.scandir(directory):
        if filename.is_file():
            graph = pickle.load(open(filename, 'rb'))
            # print(graph.nodes())
            all_graphs = np.append(all_graphs, graph)

            # create merged Graph
            # mergedGraph = nx.compose(mergedGraph, graph)

    # nx.write_gpickle(mergedGraph, "data/mergedGraph")
    # nx.draw(mainGraph, with_labels=True, pos=nx.spring_layout(mainGraph), edge_color='red')

    node_features = FeaturesExtraction(node_attributes)
    # print(node_features.attributes)
    # print(node_features.attributes['followers_to_following'])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loadGraphs()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
