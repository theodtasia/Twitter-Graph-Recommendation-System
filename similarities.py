import numpy as np
from scipy.spatial import distance

"""Calculate different similarity methods"""

class Similarity:

    def __init__(self):
        pass

    @staticmethod
    def jaccard_similarity(a, b):
        return len(a.intersection(b)) / len(a.union(b))

    @staticmethod
    def euclidean_distance_similarity(a, b):
        return distance.euclidean(a, b)

    @staticmethod
    def manhattan_distance_similarity(a, b):
        return distance.cityblock(a, b)

    @staticmethod
    def cosine(a, b):
        return np.sqrt(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))