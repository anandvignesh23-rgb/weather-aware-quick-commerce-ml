from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "synthetic_quick_commerce_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_directories() -> None:
    for directory in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def time_bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def classify_weather(
    temperature_c: float,
    humidity: float,
    rainfall_mm: float,
    wind_speed_kmph: float,
) -> str:
    if temperature_c >= 34:
        return "hot"
    if 0 < rainfall_mm <= 10:
        return "rain"
    if rainfall_mm > 10:
        return "heavy_rain"
    if temperature_c <= 18:
        return "cold"
    if humidity >= 80:
        return "humid"
    if wind_speed_kmph >= 30:
        return "windy"
    return "clear"


def delivery_difficulty(
    bad_weather_score: float,
    rider_demand_ratio: float,
    delivery_distance_km: float,
) -> str:
    score = bad_weather_score + max(0, rider_demand_ratio - 6) * 2
    score += max(0, delivery_distance_km - 3) * 2
    if score >= 35:
        return "Extreme"
    if score >= 20:
        return "High"
    if score >= 10:
        return "Medium"
    return "Low"
