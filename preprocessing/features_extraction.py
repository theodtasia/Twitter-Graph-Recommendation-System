import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturesExtraction:

    def __init__(self, attributes, extract_attr=False, turn_to_numeric=False, scale=False):
        """
        :param attributes: as read from node_attributes file (dictionary)
        :param extract_attr:
        :param turn_to_numeric:
        """
        self.attributes = self.turn_to_dataframe(attributes)
        if extract_attr:
            self.__extract_features()
        if turn_to_numeric:
            self.__turn_to_numeric()
        if scale:
            self.__scale()

    def turn_to_dataframe(self, attributes):
        return pd.DataFrame(attributes).transpose()

    def __extract_features(self):
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


    def __turn_to_numeric(self):
        self.attributes['verified'] = self.attributes['verified'].astype(int)
        self.attributes = pd.get_dummies(self.attributes, prefix=['party'], columns=['party'], drop_first=True)

    def __scale(self):
        scaler = StandardScaler()
        self.attributes[self.attributes.columns] = \
            scaler.fit_transform(self.attributes[self.attributes.columns])