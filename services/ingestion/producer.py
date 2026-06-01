"""
Kafka Producer — Weather Ingestion Service

Fetches daily weather from Open-Meteo and publishes each day
as a JSON message to the `raw-weather` Kafka topic.

Triggered by Airflow or invoked directly:
  python -m services.ingestion.producer \
      --location "London" --lat 51.5074 --lon -0.1278 \
      --start 2024-01-01 --end 2024-01-31
"""

import os
import json
import argparse
import logging
from datetime import date, timedelta

import httpx
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_RAW       = os.getenv("KAFKA_TOPIC_RAW", "raw-weather")
OPEN_METEO_URL  = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=3,
        linger_ms=10,
    )


def _fetch_chunk(lat: float, lon: float, start: date, end: date) -> list[dict]:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      ",".join(WEATHER_VARIABLES),
        "timezone":   "UTC",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
    daily = resp.json()["daily"]
    records = []
    for i, dt in enumerate(daily["time"]):
        row = {"date": dt}
        for var in WEATHER_VARIABLES:
            row[var] = daily.get(var, [None])[i]
        records.append(row)
    return records


def produce(
    location: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> int:
    try:
        producer = _build_producer()
    except Exception as e:
        log.warning("Kafka unavailable (%s) — messages not published.", e)
        return 0

    records = _fetch_chunk(latitude, longitude, start, end)
    published = 0
    for record in records:
        message = {
            "location":  location,
            "latitude":  latitude,
            "longitude": longitude,
            **record,
        }
        producer.send(TOPIC_RAW, key=f"{location}:{record['date']}", value=message)
        published += 1

    producer.flush()
    producer.close()
    log.info("Published %d records for %s to topic '%s'", published, location, TOPIC_RAW)
    return published


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="London")
    parser.add_argument("--lat",   type=float, default=51.5074)
    parser.add_argument("--lon",   type=float, default=-0.1278)
    parser.add_argument("--start", default=str(date.today() - timedelta(days=30)))
    parser.add_argument("--end",   default=str(date.today() - timedelta(days=1)))
    args = parser.parse_args()

    n = produce(
        args.location,
        args.lat,
        args.lon,
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
    )
    print(f"Published {n} records.")
