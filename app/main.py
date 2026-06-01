from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.storage import init_db
from app.api import ingest, pipeline, train, predict, monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Weather Anomaly Detection API",
    description=(
        "End-to-end ML pipeline: API Integration → Data Pipeline → "
        "Feature Engineering → PCA → Anomaly Detection → Monitoring"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(pipeline.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(monitor.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Weather Anomaly Detection",
        "workflow": [
            "POST /ingest          — fetch & store weather data from Open-Meteo",
            "POST /pipeline/process — clean & engineer features",
            "POST /train           — run PCA + train Isolation Forest",
            "POST /predict         — detect anomalies on new data",
            "GET  /monitor         — view model stats & data health",
        ],
        "docs": "/docs",
    }
