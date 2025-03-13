import pandas as pd

def add_new_features(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df["BYTES_PER_PACKET"] = data_df["BYTES"] / data_df["PACKETS"]
    data_df["BYTES_PER_PACKET_REV"] = data_df["BYTES_REV"] / data_df["PACKETS_REV"]
    data_df["BYTES_TOTAL"] = data_df["BYTES"] + data_df["BYTES_REV"]
    data_df["PACKETS_TOTAL"] = data_df["PACKETS"] + data_df["PACKETS_REV"]

    return data_df

def convert_app_column_to_app_name(data_df: pd.DataFrame, dataset) -> pd.DataFrame:
    data_df["APP_NAME"] = data_df["APP"].apply(lambda x: dataset._tables_app_enum.get(x, "Unknown"))

    return data_df