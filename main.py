import numpy as np
import itertools
import matplotlib.pyplot as plt

from clustering import AgglomerativeAlgorithm, AffinityClusteringAlgorithm
from dataset import prepare_dataset
from feature_engineering import add_new_features
from preprocessing import preprocess_data
from scores import evaluate_clustering

FEATURES_TO_EXTRACT = ("PACKETS", "PACKETS_REV", "BYTES", "BYTES_REV", "DURATION", "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION")
APPS_TO_EXTRACT = ("alza-webapi", "drmax", "dns-doh", "flightradar24",
                   "bongacams", "cloudflare-cdnjs", "playradio", "gmail",
                   "google-recaptcha", "chrome-remotedesktop")


def main():
    features_array, app_array = preprocess_data(add_new_features(prepare_dataset(FEATURES_TO_EXTRACT, APPS_TO_EXTRACT)), FEATURES_TO_EXTRACT)
    algo = AffinityClusteringAlgorithm()
    app_array = algo.modify_app_array(app_array)
    for result in algo.generate_multiple_clustering_results(features_array):
        if result is None:
            break

        print(algo.print_parameters(result))
        print(f"Clusters count: {len(np.unique(result.labels))}")
        evaluate_clustering(app_array, result.labels, f"")
        print("")


if __name__ == '__main__':
    main()
