from __future__ import annotations

import numpy as np
import pandas as pd

from utils import OUTPUTS_DIR, classify_weather


WEATHER_CATEGORIES = {
    "hot": ["Cold Beverages", "Ice Cream and Frozen Desserts", "Summer Essentials"],
    "heavy_rain": ["Quick Meals", "Hot Beverages", "Snacks", "Rain Essentials"],
    "rain": ["Hot Beverages", "Snacks", "Quick Meals"],
    "cold": ["Hot Beverages", "Quick Meals"],
    "humid": ["Cold Beverages", "Summer Essentials"],
}


def relevant_categories(weather_type: str) -> list[str]:
    return WEATHER_CATEGORIES.get(weather_type, [])


def _normalize(series: pd.Series) -> pd.Series:
    minimum, maximum = series.min(), series.max()
    if pd.isna(minimum) or maximum == minimum:
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def recommend_products(
    candidates: pd.DataFrame,
    weather_type: str | None = None,
    top_n: int = 10,
    save: bool = False,
) -> pd.DataFrame:
    ranked = candidates.copy()
    if weather_type is None:
        first = ranked.iloc[0]
        weather_type = classify_weather(
            first["temperature_c"],
            first["humidity"],
            first["rainfall_mm"],
            first["wind_speed_kmph"],
        )

    categories = relevant_categories(weather_type)
    if categories:
        filtered = ranked[ranked["category"].isin(categories)].copy()
        if not filtered.empty:
            ranked = filtered

    if "historical_popularity" not in ranked:
        ranked["historical_popularity"] = ranked.get(
            "product_avg_sales", ranked["predicted_demand"]
        )
    ranked["recommendation_score"] = (
        0.45 * _normalize(ranked["predicted_demand"].clip(lower=0))
        + 0.25 * _normalize(ranked["inventory_level"])
        + 0.20 * _normalize(ranked["discount"])
        + 0.10 * _normalize(ranked["historical_popularity"])
    )
    columns = [
        "product_id",
        "product_name",
        "category",
        "predicted_demand",
        "inventory_level",
        "discount",
        "historical_popularity",
        "recommendation_score",
    ]
    result = ranked.sort_values(
        ["recommendation_score", "predicted_demand"], ascending=False
    ).head(top_n)
    result = result[columns].reset_index(drop=True)
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUTPUTS_DIR / "weather_product_recommendations.csv", index=False)
    return result
