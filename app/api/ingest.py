from fastapi import APIRouter, HTTPException
from app.schemas.models import IngestRequest, IngestResponse
from app.core import data_fetcher, storage

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        df = data_fetcher.fetch_weather(
            req.latitude, req.longitude, req.start_date, req.end_date
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo API error: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="No data returned for the given parameters.")

    rows_stored = storage.save_raw(df, req.location_name)

    return IngestResponse(
        location=req.location_name,
        rows_stored=rows_stored,
        date_range=f"{req.start_date} → {req.end_date}",
        message=f"Successfully ingested {rows_stored} daily records for {req.location_name}.",
    )
