from fastapi import APIRouter, HTTPException
from app.schemas.models import TrainRequest, TrainResponse
from app.core import storage, model
from app.core.pipeline import get_feature_matrix

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("", response_model=TrainResponse)
def train(req: TrainRequest):
    df = storage.load_processed(req.location_name)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for '{req.location_name}'. Run /pipeline/process first.")

    if len(df) < req.n_components + 5:
        raise HTTPException(status_code=422, detail=f"Not enough data to train with {req.n_components} PCA components.")

    X_df, feature_cols = get_feature_matrix(df)
    X_df = X_df.dropna()

    result = model.train(
        X_df,
        feature_cols=feature_cols,
        n_components=req.n_components,
        contamination=req.contamination,
        location=req.location_name,
    )

    return TrainResponse(
        location=req.location_name,
        message=f"Model trained on {result['training_samples']} samples. "
                f"{result['anomalies_found']} anomalies flagged during training.",
        **result,
    )
