from __future__ import annotations

import numpy as np
import pandas as pd

from utils import time_bucket


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_bucket"] = df["hour_of_day"].astype(int).map(time_bucket)

    df["is_hot"] = (df["temperature_c"] >= 34).astype(int)
    df["is_cold"] = (df["temperature_c"] <= 18).astype(int)
    df["is_rainy"] = (df["rainfall_mm"] > 0).astype(int)
    df["is_heavy_rain"] = (df["rainfall_mm"] > 10).astype(int)
    df["is_humid"] = ((df["humidity"] >= 80) & (df["rainfall_mm"] == 0)).astype(int)
    df["rain_intensity"] = pd.cut(
        df["rainfall_mm"],
        bins=[-0.01, 0, 5, 10, np.inf],
        labels=["none", "light", "moderate", "heavy"],
    ).astype(str)

    df["discount_percentage"] = df["discount"].fillna(0)
    df["price_after_discount"] = df["base_price"] * (
        1 - df["discount_percentage"] / 100
    )
    df["final_price"] = df["final_price"].fillna(df["price_after_discount"])

    riders = df["available_riders"].clip(lower=1)
    df["rider_demand_ratio"] = df["order_volume_city"] / riders
    df["delivery_pressure_score"] = df["order_volume_city"] / riders
    df["bad_weather_score"] = (
        df["rainfall_mm"] * 0.5
        + (df["wind_speed_kmph"] - 20).clip(lower=0) * 0.2
        + (df["humidity"] - 70).clip(lower=0) * 0.05
    )

    df["product_avg_sales"] = df.groupby("product_id")["units_sold"].transform("mean")
    df["category_avg_sales"] = df.groupby("category")["units_sold"].transform("mean")
    df["city_product_avg_sales"] = df.groupby(["city", "product_id"])[
        "units_sold"
    ].transform("mean")
    return df
