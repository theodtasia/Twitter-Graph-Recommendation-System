import pickle

from preprocessing.features_extraction import FeaturesExtraction


def loadGraphs():
    #read node attributes
    node_attributes = pickle.load(open('data/clean_data/node_attributes', 'rb'))

    # assign directory
    directory = 'data/clean_data/day_graphs'

    #extract features
    node_features = FeaturesExtraction(node_attributes)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loadGraphs()

