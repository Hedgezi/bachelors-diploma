import pandas as pd

from cesnet_datazoo.config import DatasetConfig
from cesnet_datazoo.datasets import CESNET_QUIC22


def load_dataset(features_to_extract: list[str]) -> pd.DataFrame:
    dataset = CESNET_QUIC22(data_root="data/CESNET_QUIC22/", size="XS", silent=True)

    dataset_config = DatasetConfig(
        dataset=dataset,
        train_period_name="W-2022-44",
        train_size=100_000,
        use_packet_histograms=True,
        return_other_fields=True,
    )
    dataset.set_dataset_config_and_initialize(dataset_config)

    data_df = dataset.get_train_df()
    data_df["APP_NAME"] = data_df["APP"].apply(lambda x: dataset._tables_app_enum.get(x, "Unknown"))

    return data_df[[*features_to_extract, "APP", "APP_NAME"]]


def limit_apps_in_df(data_df: pd.DataFrame, apps_to_extract: list[str]) -> pd.DataFrame:
    return data_df[data_df["APP_NAME"].isin(apps_to_extract)]


def prepare_dataset(
        features_to_extract: list[str],
        apps_to_extract: list[str] | None = None,
) -> pd.DataFrame:
    data_df = load_dataset(features_to_extract)

    if apps_to_extract:
        data_df = limit_apps_in_df(data_df, apps_to_extract)

    return data_df
