import httpx
import pandas as pd
from datetime import date


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


def fetch_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"})
    df["latitude"] = latitude
    df["longitude"] = longitude
    return df
