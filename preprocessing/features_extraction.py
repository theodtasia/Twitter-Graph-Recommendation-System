import pickle
from os import mkdir
from os.path import exists

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.clean_datasets import CleanData, clean_data_path, Graph_
from recommendation_task.utils import numOfGraphs

DAY_NODE_ATTRS_PATH = f'{clean_data_path}node_attrs_per_day/'

# TODO temporary solution to init gnn in_channels parameter
NUMBER_OF_TOPOLOGICAL_ATTRIBUTES = 5

class FeaturesExtraction:

    def __init__(self, extract_topological_attr,
                       load_from_file = True,
                       extract_stats_based_attr=True,
                       turn_to_numeric=True,
                       scale=False):

        self.load_from_file = load_from_file

        # if doesn't want to load saved node attributes (or save files not found)
        # or don't extract topological attributes (therefore saving not required)
        if not load_from_file or not exists(DAY_NODE_ATTRS_PATH[:-1]) \
            or not extract_topological_attr \
            or not exists(FeaturesExtraction._nodeAttributesFile(numOfGraphs())):

            self.__run_feature_extraction(extract_stats_based_attr, turn_to_numeric, scale)
            if extract_topological_attr:
                self.attr_dim += NUMBER_OF_TOPOLOGICAL_ATTRIBUTES  # TODO temporary solution
                # save node attributes per day graph as a pd csv file
                self._save_attributes_per_day()

        else:
            self.attr_dim = len(self.loadDayAttributesDataframe(0).columns)


    def loadDayAttributesDataframe(self, day):
        if self.load_from_file:
            # files are already created by the constructor, therefore simply read and return
            # day graph node attributes
            return pd.read_csv(FeaturesExtraction._nodeAttributesFile(day))
        else:
            return self.attributes

    def __run_feature_extraction(self, extract_stats_based_attr, turn_to_numeric, scale):
        # read from node_attributes file (dictionary)
        self.attributes = CleanData.readNodeAttributes()
        self.attributes = self.turn_to_dataframe(self.attributes)

        if extract_stats_based_attr:
            self.__extract_stats_based_attributes()
        if turn_to_numeric:
            self.__turn_to_numeric()
        if scale:
            self.__scale()
        self.current_graph = nx.Graph()

        # feature vector dimension
        self.attr_dim = len(self.attributes.columns)


    def _save_attributes_per_day(self):
        if not exists(DAY_NODE_ATTRS_PATH[:-1]):
            mkdir(DAY_NODE_ATTRS_PATH[:-1])

        graphs = CleanData.loadDayGraphs()
        for day, graph in enumerate(graphs):
            print(day)
            file = FeaturesExtraction._nodeAttributesFile(day)

            self.updated_topological_attrs(graph) \
                .to_csv(file, sep=',', encoding='utf-8', index=False)

    @staticmethod
    def _nodeAttributesFile(day):
        return f'{DAY_NODE_ATTRS_PATH}nodeAttrsG_{day}.csv'

    def updated_topological_attrs(self, nxDayGraph):
        # TODO should scale ?
        # nxDayGraph a nx.Graph object made up of next (only) day's edges
        # Gt doesn't overlap with G0...t-1 => merge current with next day's graph to
        # have the full graph and calc centrality measures
        if len(self.current_graph.nodes()) > 0 :
            self.current_graph = nx.compose(self.current_graph, nxDayGraph)
        else:
            self.current_graph = nxDayGraph
        print("Update topological feats according to ", self.current_graph)
        self.__extract_topological_attributes()
        return self.attributes


    def turn_to_dataframe(self, attributes):
        return pd.DataFrame(attributes).transpose()

    def __extract_stats_based_attributes(self):
        """Preprocessing and extraction of new features from node attributes."""
        self.attributes["verified"] = self.attributes["verified"].apply(lambda item: 1 if item else 0)
        self.attributes["party"] = self.attributes["party"].astype('category')

        self.attributes["followers_to_following"] = self.attributes.apply(
            lambda x: x['followers'] / x['following'] if x['following'] != 0 else 0, axis=1)

        self.attributes["average_tweets_per_day"] = self.attributes["total_tweets"] / self.attributes["twitter_age"]

        self.attributes["average_list_per_day"] = self.attributes["lists"] / self.attributes["twitter_age"]

        self.attributes["total_tweets"] = self.attributes.apply(
            lambda x: x['total_tweets'] / x['lists'] if x['lists'] != 0 else 0, axis=1)

        self.attributes["followers_per_day"] = self.attributes["followers"] / self.attributes["twitter_age"]

        self.attributes["total_tweets"] = self.attributes["following"] / self.attributes["twitter_age"]

        self.attributes["followers_per_age"] = self.attributes["following"] / self.attributes["twitter_age"]

        self.attributes["following_per_age"] = self.attributes["following"] / self.attributes["twitter_age"]

        self.attributes['tweets_per_party_number'] = self.attributes.groupby(['party'])['total_tweets'].transform('sum')

        self.attributes['lists_per_party_number'] = self.attributes.groupby(['party'])['total_tweets'].transform('sum')

        self.attributes["percentage_tweets_of_party"] = self.attributes["total_tweets"] / self.attributes['tweets_per_party_number']

        self.attributes["percentage_list_of_party"] = self.attributes["total_tweets"] / self.attributes['lists_per_party_number']

        self.attributes.drop(['tweets_per_party_number', 'lists_per_party_number'], axis=1, inplace=True)


    def __extract_topological_attributes(self):
        self.attributes['clustering'] = pd.Series(nx.clustering(self.current_graph))
        self.attributes['degree_centrality'] = pd.Series(nx.degree_centrality(self.current_graph))
        self.attributes['closeness'] = pd.Series(nx.closeness_centrality(self.current_graph))
        self.attributes['betweeness'] = pd.Series(nx.betweenness_centrality(self.current_graph))
        # TODO  raises networkx.exception.PowerIterationFailedConvergence: (PowerIterationFailedConvergence(...),
        # TODO 'power iteration failed to converge within 1000 iterations')
        # self.attributes['katz_centrality'] = pd.Series(nx.katz_centrality(self.current_graph))
        self.attributes['pr'] = pd.Series(nx.pagerank(self.current_graph))
        self.attributes.replace(np.nan, 0, inplace=True)

    def __turn_to_numeric(self):
        self.attributes['verified'] = self.attributes['verified'].astype(int)
        self.attributes = pd.get_dummies(self.attributes, prefix=['party'], columns=['party'])

    def __scale(self):
        scaler = StandardScaler()
        self.attributes[self.attributes.columns] = \
            scaler.fit_transform(self.attributes[self.attributes.columns])


