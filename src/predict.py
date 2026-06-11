from __future__ import annotations

import pandas as pd

from feature_engineering import add_features
from preprocessing import prepare_data
from recommendation_engine import recommend_products
from train_demand_model import FEATURES as DEMAND_FEATURES
from train_pricing_model import FEATURES as PRICING_FEATURES
from utils import MODELS_DIR, OUTPUTS_DIR, load_model


def predict_scenario(
    city: str = "Mumbai",
    weather_type: str = "heavy_rain",
    temperature_c: float = 24,
    rainfall_mm: float = 20,
    humidity: float = 88,
    wind_speed_kmph: float = 28,
    available_riders: int = 60,
    order_volume_city: int = 800,
    delivery_distance_km: float = 3.5,
    hour_of_day: int = 19,
    top_n: int = 10,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    data = prepare_data()
    catalog = (
        data.sort_values("date")
        .groupby(["product_id", "product_name", "category"], as_index=False)
        .tail(1)
        .copy()
    )
    catalog["date"] = data["date"].max() + pd.Timedelta(days=1)
    catalog["city"] = city
    catalog["weather_type"] = weather_type
    catalog["temperature_c"] = temperature_c
    catalog["rainfall_mm"] = rainfall_mm
    catalog["humidity"] = humidity
    catalog["wind_speed_kmph"] = wind_speed_kmph
    catalog["available_riders"] = available_riders
    catalog["order_volume_city"] = order_volume_city
    catalog["delivery_distance_km"] = delivery_distance_km
    catalog["hour_of_day"] = hour_of_day
    catalog["delivery_time_minutes"] = (
        12
        + delivery_distance_km * 4.2
        + rainfall_mm * 0.45
        + max(0, wind_speed_kmph - 20) * 0.28
        + max(0, order_volume_city / max(available_riders, 1) - 6) * 1.5
    )
    catalog = add_features(catalog)

    demand_model = load_model(MODELS_DIR / "demand_model.pkl")
    pricing_model = load_model(MODELS_DIR / "delivery_fee_model.pkl")
    catalog["predicted_demand"] = demand_model.predict(catalog[DEMAND_FEATURES]).clip(0)
    delivery_fee = float(pricing_model.predict(catalog.iloc[[0]][PRICING_FEATURES])[0])
    delivery_fee = min(100, max(15, delivery_fee))
    catalog["predicted_delivery_fee"] = round(delivery_fee, 2)

    recommendations = recommend_products(
        catalog, weather_type=weather_type, top_n=top_n, save=True
    )
    prediction_columns = [
        "date",
        "city",
        "weather_type",
        "product_id",
        "product_name",
        "category",
        "predicted_demand",
        "predicted_delivery_fee",
    ]
    predictions = catalog[prediction_columns].sort_values(
        "predicted_demand", ascending=False
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    return predictions, delivery_fee, recommendations


if __name__ == "__main__":
    predictions, fee, recommendations = predict_scenario()
    print(f"Predicted delivery fee: INR {fee:.2f}")
    print(recommendations[["product_name", "category", "predicted_demand"]].to_string(index=False))
    print(f"Saved {len(predictions)} product predictions to outputs/predictions.csv")
