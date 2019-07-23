import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class Clusterer:
    __slots__ = [
            "_features",
            "_labels",
            "_hdbscan",
            "_kmeans"
        ]
    def __init__(self, features:np.ndarray, labels:np.ndarray,):
        self._features = features
        self._labels = labels

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def hdbscan(self) -> hdbscan.HDBSCAN:
        return self._hdbscan

    def apply_hdbscan(self, params:dict={}) -> tuple:
        features:np.ndarray = self._features
        hdbs = hdbscan.HDBSCAN(**params) # min_cluster_size
        self._hdbscan = hdbs
        self._hdbscan.fit(features)
        return self.silhouette_score(features, list(self._hdbscan.labels_))

    @property
    def kmeans(self) -> KMeans:
        return self._kmeans

    def apply_kmeans(self, params:dict={}, subset:str=None):
        features:np.ndarray = self._features

        kmeans:KMeans = KMeans(**params)
        self._kmeans = kmeans
        labels:np.ndarray = self._kmeans.fit_predict(features)
        return self.silhouette_score(features, labels)

    @staticmethod
    def silhouette_score(features, labels) -> tuple:
        return silhouette_score(features, labels)
