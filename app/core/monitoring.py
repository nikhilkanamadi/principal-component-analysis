import os
from app.core import storage, model

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

if _PROM_AVAILABLE:
    records_ingested   = Counter("weather_records_ingested_total",   "Records ingested",   ["location"])
    records_processed  = Counter("weather_records_processed_total",  "Records processed",  ["location"])
    anomalies_detected = Counter("weather_anomalies_detected_total", "Anomalies detected", ["location"])
    training_runs      = Counter("model_training_runs_total",        "Training runs",      ["location"])
    api_requests       = Counter("api_requests_total",               "API requests",       ["method", "endpoint", "status"])
    api_latency        = Histogram("api_request_duration_seconds",   "API latency (s)",    ["endpoint"])
    active_models      = Gauge("active_models_total",                "Models trained and saved")


def track_ingest(location: str, n: int) -> None:
    if _PROM_AVAILABLE:
        records_ingested.labels(location=location).inc(n)


def track_process(location: str, n: int) -> None:
    if _PROM_AVAILABLE:
        records_processed.labels(location=location).inc(n)


def track_anomalies(location: str, n: int) -> None:
    if _PROM_AVAILABLE:
        anomalies_detected.labels(location=location).inc(n)


def track_train(location: str) -> None:
    if _PROM_AVAILABLE:
        training_runs.labels(location=location).inc()
        active_models.inc()


def get_report(location: str) -> dict:
    counts   = storage.count_records(location)
    trained  = model.model_exists(location)
    metadata = model.load_metadata(location) if trained else None

    feature_stats = None
    if counts["processed"] > 0:
        df      = storage.load_processed(location)
        numeric = df.select_dtypes(include="number")
        feature_stats = {
            col: {
                "mean": round(float(numeric[col].mean()), 4),
                "std":  round(float(numeric[col].std()),  4),
                "min":  round(float(numeric[col].min()),  4),
                "max":  round(float(numeric[col].max()),  4),
            }
            for col in ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
            if col in numeric.columns
        }

    training_anomaly_rate = None
    if metadata:
        samples = metadata.get("training_samples") or 1
        training_anomaly_rate = round((metadata.get("anomaly_count") or 0) / samples, 4)

    mlflow_runs = model.list_mlflow_runs(location)

    return {
        "location":            location,
        "raw_records":         counts["raw"],
        "processed_records":   counts["processed"],
        "model_trained":       trained,
        "last_trained":        metadata["trained_at"] if metadata else None,
        "n_pca_components":    metadata["n_components"] if metadata else None,
        "explained_variance":  metadata["explained_variance"] if metadata else None,
        "training_anomaly_rate": training_anomaly_rate,
        "feature_stats":       feature_stats,
        "mlflow_runs":         mlflow_runs,
    }
