from dataclasses import field
from enum import Enum
from typing import Type, Dict, Any
from dataclasses import dataclass
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
        return self.model(
            n_clusters=n_clusters,
            metric=self.metric.value,
            metric_params=self.metric_params,
            random_state=self.config.random_seed,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            n_jobs=-1,
        )