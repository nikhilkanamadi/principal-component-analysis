from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import time
from app.core.storage import init_db
from app.api import ingest, pipeline, train, predict, monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Weather Anomaly Detection API",
    description=(
        "Production ML pipeline: Open-Meteo → Kafka → PostgreSQL → "
        "PySpark → PCA → Isolation Forest → MLflow → FastAPI → Grafana"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------- Prometheus middleware ------------------------------------------

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from app.core.monitoring import api_requests, api_latency
    _PROM = True
except ImportError:
    _PROM = False


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    if _PROM:
        elapsed  = time.time() - start
        endpoint = request.url.path
        api_requests.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        api_latency.labels(endpoint=endpoint).observe(elapsed)
    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    if not _PROM:
        return Response("prometheus_client not installed", status_code=501)
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------- Routers --------------------------------------------------------

app.include_router(ingest.router)
app.include_router(pipeline.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(monitor.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Weather Anomaly Detection v2",
        "stack":   ["Open-Meteo", "Kafka", "PostgreSQL", "PySpark",
                    "PCA", "Isolation Forest", "MLflow", "FastAPI",
                    "Docker", "Kubernetes", "Prometheus", "Grafana"],
        "workflow": [
            "POST /ingest           — fetch & store weather (→ Kafka raw-weather topic)",
            "POST /pipeline/process — engineer 21 features  (PySpark job)",
            "POST /train            — PCA + Isolation Forest (logged to MLflow)",
            "POST /predict          — anomaly scores on new data",
            "GET  /monitor          — health: DB counts, MLflow runs, feature stats",
            "GET  /metrics          — Prometheus scrape endpoint",
        ],
        "docs": "/docs",
    }
