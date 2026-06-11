from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_scenario  # noqa: E402
from preprocessing import prepare_data  # noqa: E402
from utils import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    delivery_difficulty,
)


st.set_page_config(
    page_title="Weather-Aware Quick Commerce ML System",
    page_icon="☁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e6eaf0;
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Weather-Aware Quick Commerce ML System")
st.caption(
    "Demand forecasting, dynamic delivery pricing, and weather-relevant product discovery."
)


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    return prepare_data()


def load_metrics(filename: str) -> dict:
    with (OUTPUTS_DIR / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


required_files = [
    MODELS_DIR / "demand_model.pkl",
    MODELS_DIR / "delivery_fee_model.pkl",
    OUTPUTS_DIR / "demand_model_metrics.json",
    OUTPUTS_DIR / "pricing_model_metrics.json",
]
if not all(path.exists() for path in required_files):
    st.error(
        "Required model artifacts are missing. Run the data generation and both "
        "training commands from the README, then restart the dashboard."
    )
    st.stop()

data = load_dashboard_data()
cities = sorted(data["city"].unique())

st.header("1. Weather Input")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    city = st.selectbox("City", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0)
    weather_type = st.selectbox(
        "Weather type",
        ["clear", "hot", "rain", "heavy_rain", "cold", "humid", "windy"],
        index=3,
        format_func=lambda value: value.replace("_", " ").title(),
    )
with col2:
    temperature = st.number_input("Temperature (°C)", -5.0, 50.0, 24.0, 0.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 20.0, 1.0)
with col3:
    humidity = st.slider("Humidity (%)", 20, 100, 88)
    wind_speed = st.number_input("Wind speed (km/h)", 0.0, 80.0, 28.0, 1.0)
with col4:
    hour = st.slider("Hour of day", 0, 23, 19)
    riders = st.number_input("Available riders", 1, 500, 60, 5)
with col5:
    order_volume = st.number_input("Order volume", 1, 5000, 800, 25)
    distance = st.number_input("Delivery distance (km)", 0.1, 20.0, 3.5, 0.1)

predictions, delivery_fee, recommendations = predict_scenario(
    city=city,
    weather_type=weather_type,
    temperature_c=temperature,
    rainfall_mm=rainfall,
    humidity=humidity,
    wind_speed_kmph=wind_speed,
    available_riders=int(riders),
    order_volume_city=int(order_volume),
    delivery_distance_km=distance,
    hour_of_day=hour,
    top_n=10,
)

bad_weather_score = (
    rainfall * 0.5
    + max(0, wind_speed - 20) * 0.2
    + max(0, humidity - 70) * 0.05
)
rider_ratio = order_volume / max(riders, 1)
difficulty = delivery_difficulty(bad_weather_score, rider_ratio, distance)

st.header("2. Dynamic Delivery Pricing")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Delivery Fee", f"₹{delivery_fee:.0f}")
m2.metric("Delivery Difficulty", difficulty)
m3.metric("Bad Weather Score", f"{bad_weather_score:.1f}")
m4.metric("Rider Demand Ratio", f"{rider_ratio:.1f}")

st.header("3. Product Recommendations")
display_recommendations = recommendations.copy()
display_recommendations["predicted_demand"] = display_recommendations[
    "predicted_demand"
].round(1)
display_recommendations["recommendation_score"] = display_recommendations[
    "recommendation_score"
].round(3)
st.dataframe(
    display_recommendations[
        [
            "product_name",
            "category",
            "predicted_demand",
            "inventory_level",
            "discount",
            "recommendation_score",
        ]
    ],
    width="stretch",
    hide_index=True,
)

st.subheader("Top Products Likely to Sell")
top_products = predictions.head(10).set_index("product_name")["predicted_demand"]
st.bar_chart(top_products)

st.header("4. Demand Insights")
chart1, chart2 = st.columns(2)
with chart1:
    demand_weather = (
        data.groupby("weather_type", as_index=False)["units_sold"].mean().sort_values("units_sold")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(demand_weather["weather_type"], demand_weather["units_sold"], color="#3b82f6")
    ax.set_title("Average Demand by Weather Type")
    ax.set_xlabel("Average units sold")
    st.pyplot(fig)
    plt.close(fig)
with chart2:
    fee_weather = (
        data.groupby("weather_type", as_index=False)["delivery_fee"].mean().sort_values("delivery_fee")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fee_weather["weather_type"], fee_weather["delivery_fee"], color="#f97316")
    ax.set_title("Average Delivery Fee by Weather Type")
    ax.set_xlabel("Average fee (₹)")
    st.pyplot(fig)
    plt.close(fig)

chart3, chart4 = st.columns(2)
for container, title, weather_values, color in [
    (chart3, "Top Products During Hot Weather", ["hot"], "#ef4444"),
    (chart4, "Top Products During Rainy Weather", ["rain", "heavy_rain"], "#0ea5e9"),
]:
    with container:
        subset = data[data["weather_type"].isin(weather_values)]
        top = (
            subset.groupby("product_name")["units_sold"]
            .mean()
            .nlargest(8)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top.index, top.values, color=color)
        ax.set_title(title)
        ax.set_xlabel("Average units sold")
        st.pyplot(fig)
        plt.close(fig)

st.header("5. Model Metrics")
demand_metrics = load_metrics("demand_model_metrics.json")
pricing_metrics = load_metrics("pricing_model_metrics.json")
left, right = st.columns(2)
for container, title, metrics in [
    (left, "Demand Model", demand_metrics),
    (right, "Delivery Pricing Model", pricing_metrics),
]:
    with container:
        st.subheader(title)
        st.caption(f"Best model: {metrics['best_model']}")
        best = metrics["models"][metrics["best_model"]]
        a, b, c = st.columns(3)
        a.metric("MAE", f"{best['MAE']:.3f}")
        b.metric("RMSE", f"{best['RMSE']:.3f}")
        c.metric("R²", f"{best['R2']:.3f}")
        st.dataframe(pd.DataFrame(metrics["models"]).T, width="stretch")
