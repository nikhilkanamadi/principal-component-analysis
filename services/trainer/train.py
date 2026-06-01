"""
Standalone Trainer — MLflow Experiment Tracking

Reads processed features from PostgreSQL, trains PCA + Isolation Forest,
and logs the full experiment to MLflow (params, metrics, model artifact).

Run:
  python -m services.trainer.train \
      --location "London" --n-components 3 --contamination 0.05
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

DATABASE_URL        = os.getenv("DATABASE_URL", "sqlite:///./data/weather.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT", "weather-anomaly-detection")
MODEL_DIR           = Path(os.getenv("MODEL_DIR", "./data"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum",  "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "temp_range", "wind_gust_ratio", "precip_flag",
    "temperature_2m_mean_roll7_mean", "temperature_2m_mean_roll7_std",
    "precipitation_sum_roll7_mean",   "precipitation_sum_roll7_std",
    "wind_speed_10m_max_roll7_mean",  "wind_speed_10m_max_roll7_std",
    "temp_deviation", "precip_deviation",
    "day_of_year", "month",
]


def _load_data(location: str) -> pd.DataFrame:
    from app.core.storage import load_processed
    df = load_processed(location)
    if df.empty:
        raise RuntimeError(f"No processed data for '{location}'. Run the pipeline first.")
    return df


def run_training(
    location: str,
    n_components: int = 3,
    contamination: float = 0.05,
    n_estimators: int = 100,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df           = _load_data(location)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X            = df[feature_cols].dropna().values

    with mlflow.start_run(run_name=f"{location}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}") as run:
        # Log params
        mlflow.log_params({
            "location":        location,
            "n_components":    n_components,
            "contamination":   contamination,
            "n_estimators":    n_estimators,
            "n_features":      len(feature_cols),
            "training_samples": len(X),
            "feature_cols":    ",".join(feature_cols),
        })

        # Preprocessing
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca   = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        evr   = pca.explained_variance_ratio_.tolist()

        # Anomaly detection
        iso    = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        labels = iso.fit_predict(X_pca)
        scores = iso.score_samples(X_pca)

        anomaly_count = int((labels == -1).sum())
        anomaly_rate  = anomaly_count / len(X)

        # Log metrics
        mlflow.log_metrics({
            "anomaly_count":            anomaly_count,
            "anomaly_rate":             round(anomaly_rate, 4),
            "total_variance_explained": round(float(sum(evr)), 4),
            "mean_anomaly_score":       round(float(np.mean(scores)), 4),
            "min_anomaly_score":        round(float(np.min(scores)),  4),
            **{f"explained_variance_pc{i+1}": round(v, 4) for i, v in enumerate(evr)},
        })

        # Log model
        X_pca_sample  = X_pca[:5]
        signature     = infer_signature(X_pca_sample, iso.predict(X_pca_sample))
        mlflow.sklearn.log_model(
            sk_model=iso,
            artifact_path="isolation_forest",
            signature=signature,
            registered_model_name=f"weather-anomaly-{location.replace(' ', '-').lower()}",
        )

        # Save full artifact locally (scaler + pca + iso bundled)
        artifact = {
            "scaler": scaler, "pca": pca, "iso": iso,
            "feature_cols": feature_cols,
            "n_components": n_components,
            "explained_variance_ratio": evr,
            "contamination": contamination,
            "training_samples": len(X),
            "anomaly_count": anomaly_count,
            "trained_at": datetime.utcnow().isoformat(),
            "mlflow_run_id": run.info.run_id,
        }
        local_path = MODEL_DIR / f"{location.replace(' ', '_').lower()}_model.joblib"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, local_path)
        mlflow.log_artifact(str(local_path), artifact_path="artifacts")

        run_id = run.info.run_id

    log.info(
        "Training complete | run_id=%s | location=%s | anomalies=%d/%d (%.1f%%)",
        run_id, location, anomaly_count, len(X), anomaly_rate * 100
    )
    return {
        "run_id": run_id, "location": location,
        "anomaly_count": anomaly_count, "anomaly_rate": round(anomaly_rate, 4),
        "total_variance_explained": round(float(sum(evr)), 4),
        "training_samples": len(X),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location",      default="London")
    parser.add_argument("--n-components",  type=int,   default=3)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-estimators",  type=int,   default=100)
    args = parser.parse_args()

    result = run_training(
        args.location,
        n_components=args.n_components,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )
    print(result)
