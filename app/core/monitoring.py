import pandas as pd
from app.core import storage, model


def get_report(location: str) -> dict:
    counts = storage.count_records(location)
    trained = model.model_exists(location)
    metadata = model.load_metadata(location) if trained else None

    feature_stats = None
    if counts["processed"] > 0:
        df = storage.load_processed(location)
        numeric = df.select_dtypes(include="number")
        feature_stats = {
            col: {
                "mean": round(float(numeric[col].mean()), 4),
                "std": round(float(numeric[col].std()), 4),
                "min": round(float(numeric[col].min()), 4),
                "max": round(float(numeric[col].max()), 4),
            }
            for col in ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
            if col in numeric.columns
        }

    training_anomaly_rate = None
    if metadata:
        samples = metadata.get("training_samples") or 1
        anomaly_count = metadata.get("anomaly_count") or 0
        training_anomaly_rate = round(anomaly_count / samples, 4)

    return {
        "location": location,
        "raw_records": counts["raw"],
        "processed_records": counts["processed"],
        "model_trained": trained,
        "last_trained": metadata["trained_at"] if metadata else None,
        "n_pca_components": metadata["n_components"] if metadata else None,
        "explained_variance": metadata["explained_variance"] if metadata else None,
        "training_anomaly_rate": training_anomaly_rate,
        "feature_stats": feature_stats,
    }
