from fastapi import APIRouter, HTTPException
from app.schemas.models import PredictRequest, PredictResponse, AnomalyRecord
from app.core import data_fetcher, model
from app.core.pipeline import clean, engineer_features

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model.model_exists(req.location_name):
        raise HTTPException(status_code=404, detail=f"No trained model for '{req.location_name}'. Run /train first.")

    try:
        df = data_fetcher.fetch_weather(
            req.latitude, req.longitude, req.start_date, req.end_date
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo API error: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="No data returned for the given parameters.")

    df = clean(df)
    df = engineer_features(df)
    df = df.dropna()

    result_df = model.predict(df, location=req.location_name)

    anomalies = int(result_df["is_anomaly"].sum())
    total = len(result_df)

    records = []
    for _, row in result_df.iterrows():
        records.append(AnomalyRecord(
            date=str(row["date"].date()),
            anomaly_score=round(float(row["anomaly_score"]), 6),
            is_anomaly=bool(row["is_anomaly"]),
            temperature_2m_max=row.get("temperature_2m_max"),
            wind_speed_10m_max=row.get("wind_speed_10m_max"),
            precipitation_sum=row.get("precipitation_sum"),
        ))

    return PredictResponse(
        location=req.location_name,
        total_records=total,
        anomalies_detected=anomalies,
        anomaly_rate=round(anomalies / total, 4) if total else 0.0,
        records=records,
    )
