"""
Specification of validity indices used for clustering evaluation.

Defines abstract and concrete classes encapsulating various internal validity metrics.
"""


from abc import ABC, abstractmethod

import numpy as np

from ts_clustering.validation.index_impl import (calinski_harabasz_index,
                                                 davies_bouldin_index,
                                                 dunn_index, hartigan_index,
                                                 silhouette_index,
                                                 stability_index)


class ValidityIndex(ABC):
    name: str
    lower_better: bool

    @abstractmethod
    def compute(self, *, X: np.ndarray, labels: np.ndarray, distance_matrix: np.ndarray = None, **kwargs) -> float:
        pass


class SilhouetteIndex(ValidityIndex):
    name = 'silhouette'
    lower_better = False

    def compute(self, *, X, labels, distance_matrix=None, **kwargs) -> float:
        return silhouette_index(X=X, labels=labels, distance_matrix=distance_matrix)


class DunnIndex(ValidityIndex):
    name = 'dunn'
    lower_better = False

    def compute(self, *, labels, distance_matrix=None, **kwargs) -> float:
        return dunn_index(labels=labels, distance_matrix=distance_matrix)


class DaviesBouldinIndex(ValidityIndex):
    name = 'davies-bouldin'
    lower_better = True

    def compute(self, *, X, labels, **kwargs) -> float:
        return davies_bouldin_index(X=X, labels=labels, **kwargs)


class ADIndex(ValidityIndex):
    name = 'ad'
    lower_better = True

    def __init__(self, perc_col_del):
        self.perc_col_del = perc_col_del

    def compute(self, *, X, labels, distance_matrix=None, **kwargs) -> float:
        return stability_index(X=X, labels=labels, distance_matrix=distance_matrix, labels_cut=kwargs['labels_cut'],
                               method=self.name)


class APNIndex(ValidityIndex):
    name = 'apn'
    lower_better = True

    def __init__(self, perc_col_del):
        self.perc_col_del = perc_col_del

    def compute(self, *, X, labels, distance_matrix=None, **kwargs) -> float:
        return stability_index(X=X, labels=labels, distance_matrix=distance_matrix, labels_cut=kwargs['labels_cut'],
                               method=self.name)


class CalinskiHarabaszIndex(ValidityIndex):
    name = 'calinski_harabasz'
    lower_better = False

    def compute(self, *, X, labels, distance_matrix=None, **kwargs) -> float:
        return calinski_harabasz_index(X=X, labels=labels, **kwargs)

class HartiganIndex(ValidityIndex):
    name = 'hartigan'
    lower_better = True

    def compute(self, *, X, labels, distance_matrix=None, **kwargs) -> float:
        return hartigan_index(X=X, labels_k=labels, **kwargs)
