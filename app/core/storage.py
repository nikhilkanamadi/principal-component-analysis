import os
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool, StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/weather.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

# ---------- DDL -----------------------------------------------------------

_SERIAL = "INTEGER" if _is_sqlite else "SERIAL"
_JSON   = "TEXT"    if _is_sqlite else "JSONB"
_TS     = "TEXT"    if _is_sqlite else "TIMESTAMP DEFAULT NOW()"
_UNIQUE_CONFLICT = "OR REPLACE" if _is_sqlite else ""


def init_db() -> None:
    with engine.begin() as con:
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS raw_weather (
                id                          {_SERIAL} PRIMARY KEY,
                location                    TEXT NOT NULL,
                date                        TEXT NOT NULL,
                temperature_2m_max          REAL,
                temperature_2m_min          REAL,
                temperature_2m_mean         REAL,
                precipitation_sum           REAL,
                wind_speed_10m_max          REAL,
                wind_gusts_10m_max          REAL,
                shortwave_radiation_sum     REAL,
                et0_fao_evapotranspiration  REAL,
                latitude                    REAL,
                longitude                   REAL,
                ingested_at                 TEXT,
                UNIQUE(location, date)
            )
        """))
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS processed_weather (
                id           {_SERIAL} PRIMARY KEY,
                location     TEXT NOT NULL,
                date         TEXT NOT NULL,
                features     {_JSON} NOT NULL,
                processed_at TEXT,
                UNIQUE(location, date)
            )
        """))


# ---------- Raw weather ---------------------------------------------------

_RAW_COLS = [
    "location", "date",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "latitude", "longitude", "ingested_at",
]


def save_raw(df: pd.DataFrame, location: str) -> int:
    df = df.copy()
    df["location"] = location
    df["date"] = df["date"].astype(str)
    df["ingested_at"] = datetime.utcnow().isoformat()
    present = [c for c in _RAW_COLS if c in df.columns]
    rows = df[present].to_dict(orient="records")

    upsert = (
        f"INSERT OR REPLACE INTO raw_weather ({', '.join(present)}) "
        f"VALUES ({', '.join(':' + c for c in present)})"
        if _is_sqlite else
        f"INSERT INTO raw_weather ({', '.join(present)}) "
        f"VALUES ({', '.join(':' + c for c in present)}) "
        f"ON CONFLICT (location, date) DO UPDATE SET "
        + ", ".join(f"{c}=EXCLUDED.{c}" for c in present if c not in ("location", "date"))
    )
    with engine.begin() as con:
        con.execute(text(upsert), rows)
    return len(rows)


def load_raw(location: str) -> pd.DataFrame:
    with engine.connect() as con:
        rows = con.execute(
            text("SELECT * FROM raw_weather WHERE location = :loc ORDER BY date"),
            {"loc": location},
        ).mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- Processed weather ---------------------------------------------

def save_processed(df: pd.DataFrame, location: str) -> int:
    df = df.copy()
    df["date"] = df["date"].astype(str)
    now = datetime.utcnow().isoformat()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    rows = [
        {
            "location": location,
            "date": str(row["date"]),
            "features": json.dumps({c: row[c] for c in numeric_cols if c in row}),
            "processed_at": now,
        }
        for _, row in df.iterrows()
    ]

    upsert = (
        "INSERT OR REPLACE INTO processed_weather (location, date, features, processed_at) "
        "VALUES (:location, :date, :features, :processed_at)"
        if _is_sqlite else
        "INSERT INTO processed_weather (location, date, features, processed_at) "
        "VALUES (:location, :date, :features::jsonb, :processed_at) "
        "ON CONFLICT (location, date) DO UPDATE SET "
        "features=EXCLUDED.features, processed_at=EXCLUDED.processed_at"
    )
    with engine.begin() as con:
        con.execute(text(upsert), rows)
    return len(rows)


def load_processed(location: str) -> pd.DataFrame:
    with engine.connect() as con:
        rows = con.execute(
            text("SELECT date, features FROM processed_weather WHERE location = :loc ORDER BY date"),
            {"loc": location},
        ).mappings().all()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        feat = r["features"] if isinstance(r["features"], dict) else json.loads(r["features"])
        records.append({"date": r["date"], **feat})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def count_records(location: str) -> dict:
    with engine.connect() as con:
        raw = con.execute(
            text("SELECT COUNT(*) FROM raw_weather WHERE location = :loc"), {"loc": location}
        ).scalar()
        processed = con.execute(
            text("SELECT COUNT(*) FROM processed_weather WHERE location = :loc"), {"loc": location}
        ).scalar()
    return {"raw": raw or 0, "processed": processed or 0}
