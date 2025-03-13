import dataclasses
import itertools
from abc import ABC
from collections.abc import Generator

import numpy as np
from sklearn.base import ClusterMixin

@dataclasses.dataclass
class ClusteringResult:
    algorithm = None
    parameters: dict[str, list[float | int | str]]
    labels: np.ndarray


class AbstractClustering(ABC):
    def modify_app_array(self, app_array: np.ndarray) -> np.ndarray:
        return app_array

    def fit_predict(self, preprocessed_array: np.ndarray, **kwargs) -> np.ndarray:
        return self.algorithm(**kwargs).fit_predict(preprocessed_array)

    def generate_multiple_clustering_results(self, preprocessed_array: np.ndarray,
                                             params_list: dict[str, list[float | int | str]]) -> Generator[np.ndarray]:
        for params in itertools.product(*params_list.values()):
            params_dict = dict(zip(params_list.keys(), params))

            yield self.fit_predict(preprocessed_array, **params_dict)

        yield None


def clustering_fit_predict(clustering_algo: ClusterMixin, preprocessed_array: np.ndarray, **kwargs) -> np.ndarray:
    return clustering_algo(**kwargs).fit_predict(preprocessed_array)


def clustering_params_list_fit_predict(clustering_algo: ClusterMixin, preprocessed_array: np.ndarray,
                                       params_list: dict[str, list[float | int | str]]) -> Generator[np.ndarray]:
    for params in itertools.product(*params_list.values()):
        params_dict = dict(zip(params_list.keys(), params))

        yield clustering_fit_predict(clustering_algo, preprocessed_array, **params_dict)
