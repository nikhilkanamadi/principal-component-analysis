from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class IngestRequest(BaseModel):
    latitude: float = Field(default=40.7128, description="Location latitude")
    longitude: float = Field(default=-74.0060, description="Location longitude")
    start_date: date = Field(default=date(2024, 1, 1))
    end_date: date = Field(default=date(2024, 12, 31))
    location_name: str = Field(default="New York")


class IngestResponse(BaseModel):
    location: str
    rows_stored: int
    date_range: str
    message: str


class ProcessRequest(BaseModel):
    location_name: str = Field(default="New York")


class ProcessResponse(BaseModel):
    location: str
    rows_processed: int
    features: list[str]
    message: str


class TrainRequest(BaseModel):
    location_name: str = Field(default="New York")
    n_components: int = Field(default=3, ge=1, le=10, description="PCA components")
    contamination: float = Field(default=0.05, ge=0.01, le=0.5, description="Expected anomaly fraction")


class TrainResponse(BaseModel):
    location: str
    n_components: int
    explained_variance_ratio: list[float]
    total_variance_explained: float
    training_samples: int
    anomalies_found: int
    message: str


class PredictRequest(BaseModel):
    latitude: float = Field(default=40.7128)
    longitude: float = Field(default=-74.0060)
    location_name: str = Field(default="New York")
    start_date: date = Field(default=date(2025, 1, 1))
    end_date: date = Field(default=date(2025, 1, 31))


class AnomalyRecord(BaseModel):
    date: str
    anomaly_score: float
    is_anomaly: bool
    temperature_2m_max: Optional[float] = None
    wind_speed_10m_max: Optional[float] = None
    precipitation_sum: Optional[float] = None


class PredictResponse(BaseModel):
    location: str
    total_records: int
    anomalies_detected: int
    anomaly_rate: float
    records: list[AnomalyRecord]


class MonitorResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    location: str
    raw_records: int
    processed_records: int
    model_trained: bool
    last_trained: Optional[str]
    n_pca_components: Optional[int]
    explained_variance: Optional[float]
    training_anomaly_rate: Optional[float]
    feature_stats: Optional[dict]
