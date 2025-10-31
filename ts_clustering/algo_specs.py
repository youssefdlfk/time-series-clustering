"""
Specification of clustering algorithms and associated metrics used in the project.

Defines structured representations (AlgorithmSpec) for clustering algorithms, enabling flexible selection
and configuration of clustering methods (e.g., KMeans Euclidean, KMeans DTW, KShape).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Type

from config import ClusteringConfig


class Metric(Enum):
    EUCLIDEAN = "euclidean"
    DTW = "dtw"
    CC = "cross-correlation"


@dataclass
class AlgorithmSpec:
    name: str
    metric: Metric
    model: Type
    config: ClusteringConfig
    metric_params: Dict[str, Any] = field(default_factory=dict)

    def build(self, n_clusters: int):
        if self.metric.value == 'cross-correlation':
            return self.model(
                n_clusters=n_clusters,
                random_state=self.config.random_seed,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
            )
        else:
            return self.model(
                n_clusters=n_clusters,
                metric=self.metric.value,
                metric_params=self.metric_params,
                random_state=self.config.random_seed,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                n_jobs=-1,
            )
