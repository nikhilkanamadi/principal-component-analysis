import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


MODEL_DIR = Path(__file__).parent.parent.parent / "data"


def _model_path(location: str) -> Path:
    safe = location.replace(" ", "_").lower()
    return MODEL_DIR / f"{safe}_model.joblib"


def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 3,
    contamination: float = 0.05,
    location: str = "default",
) -> dict:
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = iso.fit_predict(X_pca)       # -1 = anomaly, 1 = normal
    scores = iso.score_samples(X_pca)     # lower = more anomalous

    anomaly_count = int((labels == -1).sum())

    artifact = {
        "scaler": scaler,
        "pca": pca,
        "iso": iso,
        "feature_cols": feature_cols,
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "contamination": contamination,
        "training_samples": len(X),
        "anomaly_count": anomaly_count,
        "trained_at": datetime.utcnow().isoformat(),
    }
    joblib.dump(artifact, _model_path(location))

    return {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "training_samples": len(X),
        "anomalies_found": anomaly_count,
    }


def predict(df: pd.DataFrame, location: str = "default") -> pd.DataFrame:
    artifact = joblib.load(_model_path(location))

    feature_cols = artifact["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    X_scaled = artifact["scaler"].transform(X)
    X_pca = artifact["pca"].transform(X_scaled)
    labels = artifact["iso"].predict(X_pca)
    scores = artifact["iso"].score_samples(X_pca)

    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = labels == -1
    return result


def model_exists(location: str) -> bool:
    return _model_path(location).exists()


def load_metadata(location: str) -> Optional[dict]:
    path = _model_path(location)
    if not path.exists():
        return None
    artifact = joblib.load(path)
    return {
        "trained_at": artifact.get("trained_at"),
        "n_components": artifact.get("n_components"),
        "explained_variance": float(sum(artifact.get("explained_variance_ratio", []))),
        "training_samples": artifact.get("training_samples"),
        "anomaly_count": artifact.get("anomaly_count"),
        "contamination": artifact.get("contamination"),
    }
