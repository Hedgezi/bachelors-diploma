from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy as np
import itertools
import matplotlib.pyplot as plt

from clustering import clustering_params_list_fit_predict
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
    params = {
        "n_clusters": [30, 60, 90, 100, 120],
        "linkage": ["ward", "complete", "average", "single"],
    }
    labels = clustering_params_list_fit_predict(AgglomerativeClustering, features_array, params)
    for label, param_variant in zip(labels, itertools.product(*params.values())):
        print(param_variant)
        evaluate_clustering(app_array, label, param_variant)
        print("")


if __name__ == '__main__':
    main()
