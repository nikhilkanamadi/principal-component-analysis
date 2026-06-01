from fastapi import APIRouter, HTTPException
from app.schemas.models import ProcessRequest, ProcessResponse
from app.core import storage
from app.core.pipeline import clean, engineer_features, FEATURE_COLS

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    df = storage.load_raw(req.location_name)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No raw data found for '{req.location_name}'. Run /ingest first.")

    df = clean(df)
    df = engineer_features(df)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    rows_saved = storage.save_processed(df, req.location_name)

    return ProcessResponse(
        location=req.location_name,
        rows_processed=rows_saved,
        features=available_features,
        message=f"Processed {rows_saved} records with {len(available_features)} engineered features.",
    )
