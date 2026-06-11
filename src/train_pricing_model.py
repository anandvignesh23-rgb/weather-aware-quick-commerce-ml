from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from evaluate import evaluate_regression
from preprocessing import build_preprocessor, prepare_data, time_based_split
from utils import MODELS_DIR, OUTPUTS_DIR, ensure_directories, save_json, save_model


NUMERIC_FEATURES = [
    "temperature_c",
    "humidity",
    "rainfall_mm",
    "wind_speed_kmph",
    "order_volume_city",
    "available_riders",
    "delivery_distance_km",
    "delivery_time_minutes",
    "is_weekend",
    "holiday_flag",
    "hour_of_day",
    "rider_demand_ratio",
    "delivery_pressure_score",
    "bad_weather_score",
    "day_of_week",
    "month",
]
CATEGORICAL_FEATURES = ["weather_type", "city", "time_bucket", "rain_intensity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train() -> dict:
    ensure_directories()
    dataframe = prepare_data()
    train_df, test_df = time_based_split(dataframe)
    x_train, y_train = train_df[FEATURES], train_df["delivery_fee"]
    x_test, y_test = test_df[FEATURES], test_df["delivery_fee"]

    estimators = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=140,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=42,
            n_jobs=-1,
        ),
    }
    results: dict[str, dict[str, float]] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        results[name] = evaluate_regression(y_test, pipeline.predict(x_test))
        pipelines[name] = pipeline

    best_name = min(results, key=lambda name: results[name]["RMSE"])
    payload = {
        "target": "delivery_fee",
        "split": "time_based_80_20",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_model": best_name,
        "models": results,
    }
    save_model(pipelines[best_name], MODELS_DIR / "delivery_fee_model.pkl")
    save_json(payload, OUTPUTS_DIR / "pricing_model_metrics.json")
    return payload


if __name__ == "__main__":
    metrics = train()
    print(metrics)
