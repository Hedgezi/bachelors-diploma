import numpy as np
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score


def purity_score(y_true, y_pred) -> float:
    cm = contingency_matrix(y_true, y_pred)

    return float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))


def nmi_score(y_true, y_pred) -> float:
    return normalized_mutual_info_score(y_true, y_pred)


def evaluate_clustering(y_true, y_pred, metadata):
    print(f"Purity score {metadata}: {purity_score(y_true, y_pred)}")
    print(f"NMI score {metadata}: {nmi_score(y_true, y_pred)}")