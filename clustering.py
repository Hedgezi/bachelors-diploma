import dataclasses
import itertools
from abc import ABC
from collections.abc import Generator

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN, AffinityPropagation, KMeans


@dataclasses.dataclass
class ClusteringResult:
    algorithm: ClusterMixin
    parameters: dict[str, float | int | str]
    labels: np.ndarray


class AbstractClusteringAlgorithm(ABC):
    def __init__(self):
        self.algorithm = None
        self.parameters_list = None

    def modify_app_array(self, app_array: np.ndarray) -> np.ndarray:
        return app_array

    def fit_predict(self, preprocessed_array: np.ndarray, **kwargs) -> np.ndarray:
        return self.algorithm(**kwargs).fit_predict(preprocessed_array)

    def generate_multiple_clustering_results(self, preprocessed_array: np.ndarray) -> Generator[
        ClusteringResult | None]:
        for params in itertools.product(*self.parameters_list.values()):
            params_dict = dict(zip(self.parameters_list.keys(), params))

            yield ClusteringResult(
                self.algorithm,
                params_dict,
                self.fit_predict(preprocessed_array, **params_dict),
            )

        yield None

    def print_parameters(self, result: ClusteringResult):
        print(f"{self.algorithm.__name__}; Parameters: {result.parameters}")


class KMeansAlgorithm(AbstractClusteringAlgorithm):
    def __init__(self):
        self.algorithm = KMeans
        self.parameters_list = {
            "n_clusters": [3, 5, 7, 10, 15, 20, 25, 30, 35, 40],
            "metric": ["manhattan"],
        }


class DBSCANAlgorithm(AbstractClusteringAlgorithm):
    def __init__(self):
        self.algorithm = DBSCAN
        self.parameters_list = {
            "eps": [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
            "min_samples": [3, 5, 7, 10, 15],
            "metric": ["manhattan"],
        }

    def modify_app_array(self, app_array: np.ndarray) -> np.ndarray:
        # non_outliers_positions = label != -1
        # app_array_no_outliers = app_array[non_outliers_positions]
        # label_no_outliers = label[non_outliers_positions]
        return app_array


class AgglomerativeAlgorithm(AbstractClusteringAlgorithm):
    def __init__(self):
        self.algorithm = AgglomerativeClustering
        self.parameters_list = {
            "n_clusters": [3, 10, 20, 50, 100, 150],
            "linkage": ["ward", "complete", "average", "single"],
        }


class AffinityClusteringAlgorithm(AbstractClusteringAlgorithm):
    def __init__(self):
        self.algorithm = AffinityPropagation
        self.parameters_list = {
            "damping": [0.5, 0.6, 0.7, 0.8, 0.9],
            "max_iter": [200, 400, 600, 800, 1000],
        }