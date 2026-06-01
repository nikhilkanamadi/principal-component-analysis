import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT", "weather-anomaly-detection")
MODEL_DIR           = Path(__file__).parent.parent.parent / "data"


def _setup_mlflow() -> bool:
    if not _MLFLOW_AVAILABLE:
        return False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception:
        return False


def _local_path(location: str) -> Path:
    safe = location.replace(" ", "_").lower()
    return MODEL_DIR / f"{safe}_model.joblib"


# ---------- Training ------------------------------------------------------

def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 3,
    contamination: float = 0.05,
    location: str = "default",
) -> dict:
    X = df[feature_cols].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca      = PCA(n_components=n_components)
    X_pca    = pca.fit_transform(X_scaled)
    iso      = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels   = iso.fit_predict(X_pca)
    iso.score_samples(X_pca)

    anomaly_count = int((labels == -1).sum())
    explained     = pca.explained_variance_ratio_.tolist()
    trained_at    = datetime.utcnow().isoformat()

    artifact = {
        "scaler": scaler, "pca": pca, "iso": iso,
        "feature_cols": feature_cols,
        "n_components": n_components,
        "explained_variance_ratio": explained,
        "contamination": contamination,
        "training_samples": len(X),
        "anomaly_count": anomaly_count,
        "trained_at": trained_at,
        "location": location,
    }

    # Always persist locally as fallback
    joblib.dump(artifact, _local_path(location))

    # Log to MLflow when available
    use_mlflow = _setup_mlflow()
    if use_mlflow:
        try:
            with mlflow.start_run(run_name=f"{location}-{trained_at[:10]}"):
                mlflow.log_params({
                    "location": location,
                    "n_components": n_components,
                    "contamination": contamination,
                    "n_features": len(feature_cols),
                    "training_samples": len(X),
                })
                mlflow.log_metrics({
                    "anomaly_count": anomaly_count,
                    "anomaly_rate": anomaly_count / len(X),
                    **{f"explained_variance_pc{i+1}": v for i, v in enumerate(explained)},
                    "total_variance_explained": float(sum(explained)),
                })
                mlflow.sklearn.log_model(
                    sk_model=iso,
                    artifact_path="isolation_forest",
                    registered_model_name=f"weather-anomaly-{location.replace(' ', '-').lower()}",
                )
        except Exception:
            pass  # MLflow unavailable; local artifact is sufficient

    return {
        "n_components": n_components,
        "explained_variance_ratio": explained,
        "total_variance_explained": float(sum(explained)),
        "training_samples": len(X),
        "anomalies_found": anomaly_count,
    }


# ---------- Inference ------------------------------------------------------

def predict(df: pd.DataFrame, location: str = "default") -> pd.DataFrame:
    artifact    = joblib.load(_local_path(location))
    feature_cols = artifact["feature_cols"]
    available   = [c for c in feature_cols if c in df.columns]
    X           = df[available].values
    X_scaled    = artifact["scaler"].transform(X)
    X_pca       = artifact["pca"].transform(X_scaled)
    labels      = artifact["iso"].predict(X_pca)
    scores      = artifact["iso"].score_samples(X_pca)

    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"]    = labels == -1
    return result


# ---------- Metadata -------------------------------------------------------

def model_exists(location: str) -> bool:
    return _local_path(location).exists()


def load_metadata(location: str) -> Optional[dict]:
    path = _local_path(location)
    if not path.exists():
        return None
    a = joblib.load(path)
    return {
        "trained_at":       a.get("trained_at"),
        "n_components":     a.get("n_components"),
        "explained_variance": float(sum(a.get("explained_variance_ratio", []))),
        "training_samples": a.get("training_samples"),
        "anomaly_count":    a.get("anomaly_count"),
        "contamination":    a.get("contamination"),
    }


def list_mlflow_runs(location: str) -> list[dict]:
    if not _setup_mlflow():
        return []
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=f"params.location = '{location}'",
            order_by=["start_time DESC"],
            max_results=5,
        )
        return runs[["run_id", "start_time", "metrics.anomaly_rate",
                      "metrics.total_variance_explained"]].to_dict(orient="records")
    except Exception:
        return []
