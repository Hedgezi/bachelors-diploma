import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def convert_df_to_array(data_df: pd.DataFrame, features_to_extract: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features_array = data_df[[*features_to_extract]].to_numpy()
    app_array = data_df[["APP"]].to_numpy().reshape(-1)

    return features_array, app_array


def standardize_features(features_array: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features_array)


def preprocess_data(data_df: pd.DataFrame, features_to_extract: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features_array, app_array = convert_df_to_array(data_df, features_to_extract)
    features_array = standardize_features(features_array)

    return features_array, app_array