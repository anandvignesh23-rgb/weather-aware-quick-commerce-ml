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
    "is_rainy",
    "is_hot",
    "is_cold",
    "is_weekend",
    "holiday_flag",
    "hour_of_day",
    "base_price",
    "discount",
    "final_price",
    "inventory_level",
    "day_of_week",
    "month",
    "is_heavy_rain",
    "is_humid",
    "discount_percentage",
    "price_after_discount",
]
CATEGORICAL_FEATURES = [
    "weather_type",
    "category",
    "city",
    "time_bucket",
    "rain_intensity",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train() -> dict:
    ensure_directories()
    dataframe = prepare_data(save_processed=True)
    train_df, test_df = time_based_split(dataframe)
    x_train, y_train = train_df[FEATURES], train_df["units_sold"]
    x_test, y_test = test_df[FEATURES], test_df["units_sold"]

    estimators = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=3,
            max_features=0.8,
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
        "target": "units_sold",
        "split": "time_based_80_20",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_model": best_name,
        "models": results,
    }
    save_model(pipelines[best_name], MODELS_DIR / "demand_model.pkl")
    save_json(payload, OUTPUTS_DIR / "demand_model_metrics.json")
    return payload


if __name__ == "__main__":
    metrics = train()
    print(metrics)
