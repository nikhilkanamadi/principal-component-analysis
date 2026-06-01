"""
Kafka Consumer — Writes raw-weather messages to PostgreSQL.

Subscribes to `raw-weather`, batches messages, and bulk-upserts
into the raw_weather table via SQLAlchemy.

Run:
  python -m services.ingestion.consumer
"""

import os
import json
import logging
import signal
import sys
from datetime import datetime

import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_RAW       = os.getenv("KAFKA_TOPIC_RAW", "raw-weather")
CONSUMER_GROUP  = os.getenv("KAFKA_CONSUMER_GROUP", "weather-consumer")
BATCH_SIZE      = int(os.getenv("KAFKA_BATCH_SIZE", "50"))
POLL_TIMEOUT_MS = int(os.getenv("KAFKA_POLL_TIMEOUT_MS", "5000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Shutdown signal received.")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


def _flush_batch(batch: list[dict]) -> None:
    if not batch:
        return
    # Import here to avoid circular deps when running standalone
    from app.core.storage import save_raw
    df = pd.DataFrame(batch)
    df["date"] = pd.to_datetime(df["date"])
    locations = df["location"].unique()
    for loc in locations:
        subset = df[df["location"] == loc].copy()
        n = save_raw(subset, loc)
        log.info("Flushed %d records for location '%s'", n, loc)


def run() -> None:
    try:
        consumer = KafkaConsumer(
            TOPIC_RAW,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            consumer_timeout_ms=POLL_TIMEOUT_MS,
        )
    except NoBrokersAvailable:
        log.error("Kafka broker unavailable — consumer cannot start.")
        sys.exit(1)

    log.info("Consumer started. Listening on topic '%s'...", TOPIC_RAW)
    batch: list[dict] = []

    while _running:
        records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS, max_records=BATCH_SIZE)
        for tp, messages in records.items():
            for msg in messages:
                batch.append(msg.value)
        if len(batch) >= BATCH_SIZE:
            _flush_batch(batch)
            batch.clear()
            consumer.commit()

    # Final flush on shutdown
    _flush_batch(batch)
    consumer.commit()
    consumer.close()
    log.info("Consumer stopped cleanly.")


if __name__ == "__main__":
    run()
