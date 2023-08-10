import numpy as np

from tqdm import tqdm
from time import time

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from .index import HierarchicalRetrievalIndex


class HierarchicalRetrievalIndexMatching(object):

    _metrics = ['cosine', 'euclidean']

    def __init__(self, metric, threshold):
        
        self.threshold = threshold
        self.matching_fn = self._get_matching_fn(metric)

        self.reset()


    @classmethod
    def available_metrics(cls):
        return cls._metrics


    def reset(self):
        """Discard previously processed feature vectors and create new index.
        """
        self.index = HierarchicalRetrievalIndex(self.matching_fn, self.threshold)
        self.labels = []
        self.times = []


    def match(self, features):
        """Group feature vectors into clusters

        Args:
            features (ndarray):     List of feature vectors as numpy arrays
        
        Returns:
            labels (ndarray):       Labels assigned to feature vectors
            times (ndarray):        Processing time for each feature vector
        """
        for feature in tqdm(features):
            s = time()
            sibling, pairdist = self._query(feature)
            label = self._update(feature, sibling, pairdist)
            self.labels.append(label)
            self.times.append(time() - s)

        return np.array(self.labels), np.array(self.times)


    def _query(self, feature):

        sibling, pairdist = self.index.search(feature)

        return sibling, pairdist


    def _update(self, feature, sibling, pairdist):

        label = self.index.insert(feature, sibling, pairdist)

        return label


    def _get_matching_fn(self, metric):

        if metric == 'cosine':
            return cosine_distances
        elif metric == 'euclidean':
            return euclidean_distances
        else:
            msg = 'Metric \'{}\' is not available. '.format(metric)
            msg = msg + 'Use HierarchicalIndexMatcher.available_metrics() to get a list of available metrics.'
            raise Exception(msg)
