from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import DATA_PATH, classify_weather, ensure_directories


RANDOM_SEED = 42


@dataclass(frozen=True)
class Product:
    product_id: str
    product_name: str
    category: str
    base_price: float
    base_demand: float


PRODUCTS = [
    Product("P001", "Coca Cola", "Cold Beverages", 45, 19),
    Product("P002", "Pepsi", "Cold Beverages", 45, 18),
    Product("P003", "Sprite", "Cold Beverages", 45, 16),
    Product("P004", "Cold Coffee", "Cold Beverages", 75, 14),
    Product("P005", "Mineral Water", "Cold Beverages", 20, 25),
    Product("P006", "Lemon Juice", "Cold Beverages", 60, 12),
    Product("P007", "Vanilla Ice Cream", "Ice Cream and Frozen Desserts", 120, 15),
    Product("P008", "Chocolate Ice Cream", "Ice Cream and Frozen Desserts", 130, 15),
    Product("P009", "Ice Cream Tub", "Ice Cream and Frozen Desserts", 260, 9),
    Product("P010", "Kulfi", "Ice Cream and Frozen Desserts", 50, 13),
    Product("P011", "Frozen Yogurt", "Ice Cream and Frozen Desserts", 150, 8),
    Product("P012", "Tea Powder", "Hot Beverages", 180, 17),
    Product("P013", "Coffee Powder", "Hot Beverages", 240, 13),
    Product("P014", "Hot Chocolate", "Hot Beverages", 220, 8),
    Product("P015", "Green Tea", "Hot Beverages", 190, 9),
    Product("P016", "Chips", "Snacks", 30, 25),
    Product("P017", "Biscuits", "Snacks", 40, 24),
    Product("P018", "Popcorn", "Snacks", 65, 12),
    Product("P019", "Namkeen", "Snacks", 80, 18),
    Product("P020", "Cookies", "Snacks", 90, 16),
    Product("P021", "Umbrella", "Rain Essentials", 450, 7),
    Product("P022", "Raincoat", "Rain Essentials", 700, 6),
    Product("P023", "Waterproof Phone Pouch", "Rain Essentials", 250, 3),
    Product("P024", "Instant Noodles", "Quick Meals", 55, 22),
    Product("P025", "Pasta", "Quick Meals", 110, 12),
    Product("P026", "Soup Packets", "Quick Meals", 75, 10),
    Product("P027", "Ready-to-Eat Meals", "Quick Meals", 160, 11),
    Product("P028", "Sunscreen", "Summer Essentials", 320, 7),
    Product("P029", "ORS", "Summer Essentials", 25, 12),
    Product("P030", "Electrolyte Drink", "Summer Essentials", 50, 13),
    Product("P031", "Coconut Water", "Summer Essentials", 55, 16),
    Product("P032", "Milk", "General Grocery", 65, 23),
    Product("P033", "Bread", "General Grocery", 50, 22),
    Product("P034", "Eggs", "General Grocery", 90, 21),
    Product("P035", "Rice", "General Grocery", 350, 11),
    Product("P036", "Atta", "General Grocery", 280, 12),
    Product("P037", "Curd", "General Grocery", 50, 16),
    Product("P038", "Butter", "General Grocery", 60, 10),
    Product("P039", "Paneer", "General Grocery", 110, 12),
    Product("P040", "Bananas", "Fresh Produce", 60, 18),
    Product("P041", "Apples", "Fresh Produce", 180, 13),
    Product("P042", "Tomatoes", "Fresh Produce", 45, 20),
    Product("P043", "Onions", "Fresh Produce", 55, 20),
    Product("P044", "Potatoes", "Fresh Produce", 50, 19),
]

CITIES = {
    "Mumbai": {"base_temp": 29, "humidity": 75, "rain_factor": 1.55, "demand": 1.25},
    "Delhi": {"base_temp": 27, "humidity": 55, "rain_factor": 0.65, "demand": 1.20},
    "Bengaluru": {"base_temp": 24, "humidity": 65, "rain_factor": 1.00, "demand": 1.10},
    "Chennai": {"base_temp": 30, "humidity": 76, "rain_factor": 1.15, "demand": 1.05},
    "Kolkata": {"base_temp": 28, "humidity": 72, "rain_factor": 1.20, "demand": 1.00},
}


def weather_for_day(
    date: pd.Timestamp, city: str, rng: np.random.Generator
) -> dict[str, float | str]:
    city_data = CITIES[city]
    seasonal_temp = 5.5 * np.sin(2 * np.pi * (date.dayofyear - 75) / 365)
    temperature = city_data["base_temp"] + seasonal_temp + rng.normal(0, 2.8)
    monsoon = date.month in (6, 7, 8, 9)
    rain_probability = (0.42 if monsoon else 0.10) * city_data["rain_factor"]
    rainfall = rng.gamma(2.1, 5.5) if rng.random() < rain_probability else 0.0
    rainfall = min(rainfall, 45)
    humidity = np.clip(
        city_data["humidity"] + rainfall * 0.7 + rng.normal(0, 8), 30, 98
    )
    wind = np.clip(10 + rainfall * 0.45 + rng.normal(0, 5), 2, 48)
    weather_type = classify_weather(temperature, humidity, rainfall, wind)
    return {
        "temperature_c": round(float(temperature), 1),
        "humidity": round(float(humidity), 1),
        "rainfall_mm": round(float(rainfall), 1),
        "wind_speed_kmph": round(float(wind), 1),
        "weather_type": weather_type,
    }


def demand_multiplier(category: str, weather_type: str) -> float:
    category_effects = {
        "hot": {
            "Cold Beverages": 1.8,
            "Ice Cream and Frozen Desserts": 1.9,
            "Summer Essentials": 1.8,
        },
        "heavy_rain": {
            "Quick Meals": 1.9,
            "Hot Beverages": 1.7,
            "Snacks": 1.6,
            "Rain Essentials": 6.5,
            "General Grocery": 0.9,
            "Fresh Produce": 0.82,
        },
        "rain": {
            "Quick Meals": 1.45,
            "Hot Beverages": 1.4,
            "Snacks": 1.35,
            "Rain Essentials": 2.8,
        },
        "cold": {"Hot Beverages": 1.75, "Quick Meals": 1.5},
        "humid": {"Cold Beverages": 1.4, "Summer Essentials": 1.3},
    }
    return category_effects.get(weather_type, {}).get(category, 1.0)


def generate_dataset(days: int = 210, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rows: list[dict[str, object]] = []

    for date in dates:
        for city, city_data in CITIES.items():
            weather = weather_for_day(date, city, rng)
            weather_type = str(weather["weather_type"])
            weekend = int(date.dayofweek >= 5)
            holiday = int(rng.random() < 0.035)
            festival = int(date.month in (10, 11) and rng.random() < 0.12)
            hour = int(rng.choice([8, 10, 13, 16, 19, 21], p=[0.1, 0.13, 0.17, 0.14, 0.3, 0.16]))
            city_orders = 480 * city_data["demand"]
            city_orders *= 1 + 0.18 * weekend + 0.14 * holiday + 0.16 * festival
            if weather_type in ("rain", "hot", "humid"):
                city_orders *= 1.08
            elif weather_type == "heavy_rain":
                city_orders *= 0.92
            order_volume = max(120, int(rng.normal(city_orders, 55)))
            rider_base = 92 * city_data["demand"]
            rider_weather_factor = {
                "clear": 1.0,
                "hot": 0.93,
                "rain": 0.78,
                "heavy_rain": 0.56,
                "cold": 0.92,
                "humid": 0.9,
                "windy": 0.72,
            }[weather_type]
            available_riders = max(20, int(rng.normal(rider_base * rider_weather_factor, 8)))

            for product in PRODUCTS:
                discount = float(rng.choice([0, 5, 10, 15, 20, 25], p=[0.34, 0.15, 0.2, 0.15, 0.11, 0.05]))
                final_price = product.base_price * (1 - discount / 100)
                inventory = int(rng.integers(35, 240))
                multiplier = demand_multiplier(product.category, weather_type)
                multiplier *= city_data["demand"] * (1 + 0.12 * weekend + 0.1 * holiday)
                multiplier *= 1 + discount / 160
                if hour in (19, 21):
                    multiplier *= 1.16
                expected_sales = product.base_demand * multiplier
                units_sold = int(min(inventory, max(0, rng.poisson(expected_sales))))
                stockout = int(units_sold >= inventory)
                distance = float(np.clip(rng.gamma(2.2, 1.15), 0.5, 9))
                pressure = order_volume / max(available_riders, 1)
                delivery_time = (
                    12
                    + distance * 4.2
                    + float(weather["rainfall_mm"]) * 0.45
                    + max(0, float(weather["wind_speed_kmph"]) - 20) * 0.28
                    + max(0, pressure - 6) * 1.5
                    + rng.normal(0, 3)
                )
                delivery_time = float(np.clip(delivery_time, 12, 90))
                delivery_fee = (
                    15
                    + float(weather["rainfall_mm"]) * 1.2
                    + max(0, float(weather["wind_speed_kmph"]) - 20) * 0.5
                    + max(0, order_volume - available_riders * 8) * 0.05
                    + distance * 4
                    + max(0, delivery_time - 30) * 0.18
                    + rng.normal(0, 3)
                )
                delivery_fee = float(np.clip(delivery_fee, 15, 100))
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "city": city,
                        "product_id": product.product_id,
                        "product_name": product.product_name,
                        "category": product.category,
                        **weather,
                        "is_rainy": int(float(weather["rainfall_mm"]) > 0),
                        "is_hot": int(float(weather["temperature_c"]) >= 34),
                        "is_cold": int(float(weather["temperature_c"]) <= 18),
                        "is_weekend": weekend,
                        "holiday_flag": holiday,
                        "hour_of_day": hour,
                        "base_price": product.base_price,
                        "discount": discount,
                        "final_price": round(final_price, 2),
                        "inventory_level": inventory,
                        "units_sold": units_sold,
                        "order_volume_city": order_volume,
                        "available_riders": available_riders,
                        "delivery_distance_km": round(distance, 2),
                        "delivery_time_minutes": round(delivery_time, 1),
                        "delivery_fee": round(delivery_fee, 2),
                        "customer_segment": rng.choice(["value", "regular", "premium"], p=[0.35, 0.5, 0.15]),
                        "competitor_price": round(product.base_price * rng.uniform(0.9, 1.12), 2),
                        "marketing_campaign_flag": int(rng.random() < 0.12),
                        "stockout_flag": stockout,
                        "festival_flag": festival,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic quick-commerce data.")
    parser.add_argument("--days", type=int, default=210)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()
    dataframe = generate_dataset(days=max(args.days, 180), seed=args.seed)
    dataframe.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(dataframe):,} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
