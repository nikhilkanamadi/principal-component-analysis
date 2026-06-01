import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager


DB_PATH = Path(__file__).parent.parent.parent / "data" / "weather.db"


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS raw_weather (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                location    TEXT NOT NULL,
                date        TEXT NOT NULL,
                temperature_2m_max          REAL,
                temperature_2m_min          REAL,
                temperature_2m_mean         REAL,
                precipitation_sum           REAL,
                wind_speed_10m_max          REAL,
                wind_gusts_10m_max          REAL,
                shortwave_radiation_sum     REAL,
                et0_fao_evapotranspiration  REAL,
                latitude    REAL,
                longitude   REAL,
                UNIQUE(location, date)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS processed_weather (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                location    TEXT NOT NULL,
                date        TEXT NOT NULL,
                features    TEXT NOT NULL,
                UNIQUE(location, date)
            )
        """)


def save_raw(df: pd.DataFrame, location: str) -> int:
    df = df.copy()
    df["location"] = location
    df["date"] = df["date"].astype(str)

    cols = [
        "location", "date",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        "latitude", "longitude",
    ]
    existing = [c for c in cols if c in df.columns]
    rows = df[existing].to_dict(orient="records")

    with _conn() as con:
        con.executemany(
            f"""INSERT OR REPLACE INTO raw_weather ({', '.join(existing)})
                VALUES ({', '.join(':' + c for c in existing)})""",
            rows,
        )
    return len(rows)


def load_raw(location: str) -> pd.DataFrame:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM raw_weather WHERE location = ? ORDER BY date", (location,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_processed(df: pd.DataFrame, location: str) -> int:
    import json
    df = df.copy()
    df["location"] = location
    df["date"] = df["date"].astype(str)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "location": location,
            "date": str(row["date"]),
            "features": json.dumps({c: row[c] for c in numeric_cols if c in row}),
        })

    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO processed_weather (location, date, features) VALUES (:location, :date, :features)",
            rows,
        )
    return len(rows)


def load_processed(location: str) -> pd.DataFrame:
    import json
    with _conn() as con:
        rows = con.execute(
            "SELECT date, features FROM processed_weather WHERE location = ? ORDER BY date", (location,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        record = {"date": r["date"]}
        record.update(json.loads(r["features"]))
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def count_records(location: str) -> dict:
    with _conn() as con:
        raw = con.execute(
            "SELECT COUNT(*) as n FROM raw_weather WHERE location = ?", (location,)
        ).fetchone()["n"]
        processed = con.execute(
            "SELECT COUNT(*) as n FROM processed_weather WHERE location = ?", (location,)
        ).fetchone()["n"]
    return {"raw": raw, "processed": processed}
