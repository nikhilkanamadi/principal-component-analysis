"""
Airflow DAG — Weather Anomaly Detection Pipeline

Schedule: daily at 06:00 UTC

Task graph:
  ingest → process → train → notify_success
              └─────── (on failure) ─── notify_failure

Each task calls the corresponding FastAPI endpoint so the
orchestration layer stays decoupled from the ML logic.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty  import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ---- Configuration -------------------------------------------------------

LOCATIONS = [
    {"name": "London",   "lat": 51.5074, "lon": -0.1278},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Tokyo",    "lat": 35.6762, "lon": 139.6503},
]

API_BASE_URL    = "http://api:8000"          # Docker Compose service name
LOOKBACK_DAYS   = 30
N_PCA_COMPONENTS = 3
CONTAMINATION    = 0.05

default_args = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
    "email_on_failure": False,
}


# ---- Task functions -------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict:
    import httpx
    url = f"{API_BASE_URL}{endpoint}"
    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def ingest_location(location: dict, **context) -> None:
    logical_date = context["logical_date"]
    end_date     = logical_date.date()
    start_date   = end_date - timedelta(days=LOOKBACK_DAYS)

    result = _post("/ingest", {
        "latitude":      location["lat"],
        "longitude":     location["lon"],
        "location_name": location["name"],
        "start_date":    str(start_date),
        "end_date":      str(end_date),
    })
    log.info("Ingest result: %s", result)
    context["ti"].xcom_push(key=f"ingest_{location['name']}", value=result)


def process_location(location: dict, **context) -> None:
    result = _post("/pipeline/process", {"location_name": location["name"]})
    log.info("Process result: %s", result)
    context["ti"].xcom_push(key=f"process_{location['name']}", value=result)


def train_location(location: dict, **context) -> None:
    result = _post("/train", {
        "location_name": location["name"],
        "n_components":  N_PCA_COMPONENTS,
        "contamination": CONTAMINATION,
    })
    log.info("Train result: %s", result)
    context["ti"].xcom_push(key=f"train_{location['name']}", value=result)


def summarise(**context) -> None:
    ti = context["ti"]
    summary = {}
    for loc in LOCATIONS:
        summary[loc["name"]] = {
            "ingest":  ti.xcom_pull(key=f"ingest_{loc['name']}"),
            "process": ti.xcom_pull(key=f"process_{loc['name']}"),
            "train":   ti.xcom_pull(key=f"train_{loc['name']}"),
        }
    log.info("Pipeline summary:\n%s", json.dumps(summary, indent=2))


# ---- DAG definition -------------------------------------------------------

with DAG(
    dag_id="weather_anomaly_pipeline",
    description="Daily weather ingestion → feature engineering → PCA anomaly detection",
    start_date=datetime(2024, 1, 1),
    schedule="0 6 * * *",
    default_args=default_args,
    catchup=False,
    tags=["weather", "anomaly-detection", "pca", "mlflow"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)
    done  = PythonOperator(task_id="summarise", python_callable=summarise)

    prev_train = None

    for location in LOCATIONS:
        loc_name = location["name"].replace(" ", "_").lower()

        t_ingest = PythonOperator(
            task_id=f"ingest_{loc_name}",
            python_callable=ingest_location,
            op_kwargs={"location": location},
        )
        t_process = PythonOperator(
            task_id=f"process_{loc_name}",
            python_callable=process_location,
            op_kwargs={"location": location},
        )
        t_train = PythonOperator(
            task_id=f"train_{loc_name}",
            python_callable=train_location,
            op_kwargs={"location": location},
        )

        start >> t_ingest >> t_process >> t_train >> done

    done >> end
