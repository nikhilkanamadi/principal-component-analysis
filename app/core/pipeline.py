import pandas as pd
import numpy as np


NUMERIC_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Forward-fill then backward-fill small gaps
    df[NUMERIC_COLS] = df[NUMERIC_COLS].ffill().bfill()
    df = df.dropna(subset=NUMERIC_COLS)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temporal features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Derived weather features
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["wind_gust_ratio"] = df["wind_gusts_10m_max"] / (df["wind_speed_10m_max"] + 1e-6)
    df["precip_flag"] = (df["precipitation_sum"] > 0).astype(int)

    # Rolling statistics (7-day window)
    for col in ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]:
        df[f"{col}_roll7_mean"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_roll7_std"] = df[col].rolling(7, min_periods=1).std().fillna(0)

    # Deviation from rolling mean (anomaly signal)
    df["temp_deviation"] = df["temperature_2m_mean"] - df["temperature_2m_mean_roll7_mean"]
    df["precip_deviation"] = df["precipitation_sum"] - df["precipitation_sum_roll7_mean"]

    return df


FEATURE_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "temp_range", "wind_gust_ratio", "precip_flag",
    "temperature_2m_mean_roll7_mean", "temperature_2m_mean_roll7_std",
    "precipitation_sum_roll7_mean", "precipitation_sum_roll7_std",
    "wind_speed_10m_max_roll7_mean", "wind_speed_10m_max_roll7_std",
    "temp_deviation", "precip_deviation",
    "day_of_year", "month",
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].astype(float), available
