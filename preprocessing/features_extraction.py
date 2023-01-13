from os import mkdir
from os.path import exists

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.clean_datasets import CleanData
from other.handle_files import DAY_NODE_ATTRS_PATH, numOfGraphs


class FeaturesExtraction:

    def __init__(self, args):

        self.__run_feature_extraction(args.use_stats_based_attrs)

        self.use_topological_node_attrs = args.use_topological_node_attrs

        if args.use_topological_node_attrs:
            self.attr_dim += args.topological_attrs_dim

        if args.rerun_topological_node_attrs:
            self._save_attributes_per_day(args.rerun_topological_node_attrs_day_limit)



    def loadDayAttributesDataframe(self, day):
        if self.use_topological_node_attrs:
            # files are already created by the constructor, therefore simply read and return
            # day graph node attributes
            self.attributes = pd.read_csv(FeaturesExtraction._nodeAttributesFile(day))
        self.__scale()
        return self.attributes


    def __run_feature_extraction(self, extract_stats_based_attr):
        # read from node_attributes file (dictionary)
        self.attributes = CleanData.readNodeAttributes()
        self.attributes = self.turn_to_dataframe(self.attributes)

        if extract_stats_based_attr:
            self.__extract_stats_based_attributes()
        self.__turn_to_numeric()

        # feature vector dimension
        self.attr_dim = len(self.attributes.columns)


    def _save_attributes_per_day(self, day_limit):

        graphs = CleanData.loadDayGraphs()
        merged_graph = nx.Graph()

        for day, graph in enumerate(graphs):
            if day_limit < day:
                return

            print(day)
            merged_graph = nx.compose(merged_graph, graph)
            print("Update topological feats according to ", merged_graph)

            file = FeaturesExtraction._nodeAttributesFile(day)
            if not exists(file):
                self.__extract_topological_attributes(merged_graph)
                self.attributes.to_csv(file, sep=',', encoding='utf-8', index=False)



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


    def __extract_topological_attributes(self, merged_graph):
        self.attributes['clustering'] = pd.Series(nx.clustering(merged_graph))
        self.attributes['degree_centrality'] = pd.Series(nx.degree_centrality(merged_graph))
        self.attributes['closeness'] = pd.Series(nx.closeness_centrality(merged_graph))
        self.attributes['betweeness'] = pd.Series(nx.betweenness_centrality(merged_graph))
        # TODO  raises networkx.exception.PowerIterationFailedConvergence: (PowerIterationFailedConvergence(...),
        # TODO 'power iteration failed to converge within 1000 iterations')
        # self.attributes['katz_centrality'] = pd.Series(nx.katz_centrality(self.current_graph))
        self.attributes['pr'] = pd.Series(nx.pagerank(merged_graph))
        self.attributes.replace(np.nan, 0, inplace=True)


    def __turn_to_numeric(self):
        self.attributes['verified'] = self.attributes['verified'].astype(int)
        self.attributes = pd.get_dummies(self.attributes, prefix=['party'], columns=['party'])


    def __scale(self):
        # don't scale the following features:
        columns = ['clustering', 'degree_centrality', 'closeness', 'betweeness', 'pr', 'verified'] \
                  + [f'party_{p}' for p in ['left', 'right', 'neutral', 'middle']]
        columns = self.attributes.columns.difference(columns)
        scaler = StandardScaler()
        self.attributes[columns] = \
            scaler.fit_transform(self.attributes[columns])



    @staticmethod
    def _nodeAttributesFile(day):
        return f'{DAY_NODE_ATTRS_PATH}nodeAttrsG_{day}.csv'