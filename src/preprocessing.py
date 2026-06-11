from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import add_features
from utils import DATA_PATH, PROCESSED_DIR


def load_data(path=DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["date", "product_id", "city"])
    dataframe = dataframe.sort_values(["date", "city", "product_id"]).reset_index(drop=True)
    return dataframe


def prepare_data(save_processed: bool = False) -> pd.DataFrame:
    dataframe = add_features(load_data())
    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(PROCESSED_DIR / "featured_quick_commerce_data.csv", index=False)
    return dataframe


def time_based_split(
    dataframe: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(dataframe["date"].drop_duplicates().sort_values())
    split_index = max(1, int(len(dates) * train_fraction))
    cutoff = dates.iloc[min(split_index, len(dates) - 1)]
    train = dataframe[dataframe["date"] < cutoff].copy()
    test = dataframe[dataframe["date"] >= cutoff].copy()
    return train, test


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
