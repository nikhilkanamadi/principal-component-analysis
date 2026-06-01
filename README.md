# Weather Anomaly Detection — Production ML Platform

An end-to-end, production-grade ML platform that detects anomalous weather events using a full modern data-engineering stack.

```
Open-Meteo API → Kafka → PostgreSQL → PySpark → PCA → Isolation Forest → MLflow → FastAPI → Docker/K8s → Grafana
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION                                  │
│                      Apache Airflow (DAG)                               │
│            ingest → process → train  (daily at 06:00 UTC)              │
└──────────────┬──────────────┬──────────────┬──────────────────────────┘
               │              │              │
               ▼              ▼              ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│  INGESTION      │  │  PROCESSING       │  │  TRAINING                │
│                 │  │                   │  │                          │
│ Open-Meteo API  │  │ PySpark           │  │ StandardScaler           │
│     ↓           │  │ Window functions  │  │   → PCA (n components)   │
│ Kafka Producer  │  │ 21 features       │  │   → Isolation Forest     │
│     ↓           │  │   ↓               │  │        ↓                 │
│ raw-weather     │  │ PostgreSQL        │  │ MLflow Experiment        │
│   (topic)       │  │ processed_weather │  │ (params + metrics +      │
│     ↓           │  │                   │  │  model artifact)         │
│ Kafka Consumer  │  └──────────────────┘  └──────────────────────────┘
│     ↓           │
│ PostgreSQL      │   ┌──────────────────────────────────────────────┐
│ raw_weather     │   │  SERVING                                      │
└─────────────────┘   │                                               │
                      │  FastAPI                                      │
                      │  POST /ingest          POST /predict          │
                      │  POST /pipeline/process GET  /monitor         │
                      │  POST /train            GET  /metrics ←──┐   │
                      └──────────────────────────────────────────│───┘
                                                                  │
                      ┌───────────────────────────────────────────┘
                      │
               ┌──────▼──────┐    ┌──────────────┐
               │ Prometheus  │───▶│   Grafana     │
               │ (scrape)    │    │  (dashboards) │
               └─────────────┘    └──────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Integration** | [Open-Meteo](https://open-meteo.com) | Free weather archive API, no key required |
| **Streaming** | Apache Kafka 7.6 | Decoupled ingestion via `raw-weather` topic |
| **Storage** | PostgreSQL 16 | Raw + processed weather, MLflow metadata |
| **Processing** | Apache Spark 3.5 | Distributed feature engineering (window functions) |
| **ML — Reduction** | PCA (`sklearn`) | Dimensionality reduction, configurable components |
| **ML — Detection** | Isolation Forest (`sklearn`) | Unsupervised anomaly detection |
| **Experiment Tracking** | MLflow 2.15 | Params, metrics, model registry, artifact storage |
| **Serving** | FastAPI 0.115 + Uvicorn | REST API with Prometheus middleware |
| **Containerisation** | Docker / Docker Compose | Local full-stack in one command |
| **Orchestration** | Kubernetes | Production deployment with HPA |
| **Scheduling** | Apache Airflow 2.9 | Daily pipeline DAG with retries |
| **Monitoring** | Prometheus + Grafana 11 | Real-time pipeline and model metrics |

---

## Project Structure

```
principal-component-analysis/
├── app/                          # FastAPI application
│   ├── main.py                   # App factory, Prometheus middleware, /metrics
│   ├── api/
│   │   ├── ingest.py             # POST /ingest
│   │   ├── pipeline.py           # POST /pipeline/process
│   │   ├── train.py              # POST /train
│   │   ├── predict.py            # POST /predict
│   │   └── monitor.py            # GET  /monitor
│   ├── core/
│   │   ├── data_fetcher.py       # Open-Meteo HTTP client
│   │   ├── pipeline.py           # Feature engineering (21 features)
│   │   ├── model.py              # PCA + IsolationForest + MLflow integration
│   │   ├── storage.py            # SQLAlchemy (PostgreSQL / SQLite fallback)
│   │   └── monitoring.py         # Prometheus counters, health report
│   └── schemas/models.py         # Pydantic request/response models
│
├── services/
│   ├── ingestion/
│   │   ├── producer.py           # Kafka producer: Open-Meteo → raw-weather topic
│   │   └── consumer.py           # Kafka consumer: raw-weather → PostgreSQL
│   ├── processor/
│   │   └── spark_processor.py    # PySpark feature engineering job
│   └── trainer/
│       └── train.py              # Standalone MLflow training run
│
├── airflow/
│   └── dags/
│       └── weather_pipeline_dag.py  # Daily DAG: ingest→process→train ×3 cities
│
├── infra/
│   ├── docker/
│   │   ├── docker-compose.yml    # Full stack: 12 services
│   │   ├── .env.example          # Environment variable template
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.ingestion
│   │   ├── Dockerfile.processor
│   │   ├── Dockerfile.trainer
│   │   └── Dockerfile.airflow
│   ├── k8s/
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml        # Shared env + secrets
│   │   ├── postgres/             # StatefulSet + Service
│   │   ├── kafka/                # Zookeeper + Kafka StatefulSet
│   │   ├── api/                  # Deployment + Service + HPA
│   │   ├── mlflow/               # Deployment + PVC + Service
│   │   ├── spark/                # Master + Worker Deployments
│   │   ├── airflow/              # Webserver + Scheduler
│   │   ├── prometheus/           # Deployment + RBAC + ConfigMap
│   │   ├── grafana/              # Deployment + PVC + Service
│   │   └── ingress.yaml          # Nginx ingress for all UIs
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/prometheus.yml
│       │   └── dashboards/provider.yml
│       └── dashboards/pipeline.json  # 11-panel auto-provisioned dashboard
│
└── requirements/
    ├── api.txt
    ├── ingestion.txt
    ├── processor.txt
    ├── trainer.txt
    └── airflow.txt
```

---

## Quickstart — Local (SQLite, no Docker)

```bash
git clone https://github.com/nikhilkanamadi/principal-component-analysis.git
cd principal-component-analysis

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/api.txt

uvicorn app.main:app --reload
```

API at **http://localhost:8000** · Docs at **http://localhost:8000/docs**

---

## Quickstart — Full Stack (Docker Compose)

```bash
cd infra/docker
cp .env.example .env          # edit passwords if needed
docker compose up -d
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8081 (admin / admin123) |
| Spark UI | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin123) |

---

## API Workflow

### 1 — Ingest (Open-Meteo → Kafka → PostgreSQL)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"latitude":51.5074,"longitude":-0.1278,"start_date":"2023-01-01","end_date":"2023-12-31","location_name":"London"}'
```

```json
{ "location": "London", "rows_stored": 365, "date_range": "2023-01-01 → 2023-12-31" }
```

### 2 — Process (PySpark feature engineering)

```bash
curl -X POST http://localhost:8000/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{"location_name":"London"}'
```

**21 engineered features:** raw meteo variables, `temp_range`, `wind_gust_ratio`, 7-day rolling mean/std, deviation from rolling mean, `day_of_year`, `month`.

### 3 — Train (PCA + Isolation Forest → MLflow)

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"location_name":"London","n_components":3,"contamination":0.05}'
```

```json
{
  "n_components": 3,
  "explained_variance_ratio": [0.2825, 0.2033, 0.1048],
  "total_variance_explained": 0.5906,
  "training_samples": 365,
  "anomalies_found": 19
}
```

Logs params, metrics, and the `IsolationForest` artifact to MLflow. Model also registered in MLflow Model Registry.

### 4 — Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude":51.5074,"longitude":-0.1278,"location_name":"London","start_date":"2024-01-01","end_date":"2024-01-31"}'
```

Returns a per-day `anomaly_score` (lower = more anomalous) and `is_anomaly` flag.

### 5 — Monitor

```bash
curl "http://localhost:8000/monitor?location=London"
```

Returns record counts, model metadata, feature stats, and the last 5 MLflow run IDs.

### 6 — Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Exposes `weather_records_ingested_total`, `weather_anomalies_detected_total`, `api_requests_total`, `api_request_duration_seconds`, and more — scraped automatically by Prometheus.

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/postgres/
kubectl apply -f infra/k8s/kafka/
kubectl apply -f infra/k8s/mlflow/
kubectl apply -f infra/k8s/spark/
kubectl apply -f infra/k8s/api/
kubectl apply -f infra/k8s/prometheus/
kubectl apply -f infra/k8s/grafana/
kubectl apply -f infra/k8s/ingress.yaml

# Watch rollout
kubectl rollout status deployment/weather-api -n weather-anomaly

# Scale API horizontally
kubectl scale deployment weather-api --replicas=4 -n weather-anomaly
```

The `HorizontalPodAutoscaler` automatically scales the API between 2–8 replicas at 70% CPU.

---

## Airflow DAG

The `weather_anomaly_pipeline` DAG runs daily at 06:00 UTC and processes three cities in parallel:

```
start
  ├── ingest_london  → process_london  → train_london
  ├── ingest_new_york → process_new_york → train_new_york
  └── ingest_tokyo   → process_tokyo   → train_tokyo
         └─────────────── summarise ──────────────── end
```

Each task calls the corresponding FastAPI endpoint with a 30-day lookback window. Failed tasks retry twice with a 5-minute delay.

---

## Grafana Dashboard

The auto-provisioned **Weather Anomaly Detection Pipeline** dashboard includes:

| Panel | Metric |
|---|---|
| Records Ingested | `weather_records_ingested_total` |
| Records Processed | `weather_records_processed_total` |
| Anomalies Detected | `weather_anomalies_detected_total` |
| Training Runs | `model_training_runs_total` |
| API Request Rate | `rate(api_requests_total[5m])` by endpoint |
| API Latency P99 | `histogram_quantile(0.99, ...)` |
| Anomaly Rate by Location | `rate(weather_anomalies_detected_total[1h])` |
| HTTP Status Codes | Pie chart of 2xx / 4xx / 5xx |
| Latency Heatmap | `api_request_duration_seconds_bucket` |

---

## Standalone Services

### Run Kafka producer manually

```bash
python -m services.ingestion.producer \
  --location "Tokyo" --lat 35.6762 --lon 139.6503 \
  --start 2024-01-01 --end 2024-01-31
```

### Run Spark processor

```bash
# Local mode
python -m services.processor.spark_processor --location Tokyo

# On cluster
spark-submit --master spark://spark-master:7077 \
  services/processor/spark_processor.py --location Tokyo
```

### Run MLflow training

```bash
python -m services.trainer.train \
  --location Tokyo --n-components 3 --contamination 0.05
```

---

## Example Locations

| City | Latitude | Longitude |
|---|---|---|
| London | 51.5074 | -0.1278 |
| New York | 40.7128 | -74.0060 |
| Tokyo | 35.6762 | 139.6503 |
| Sydney | -33.8688 | 151.2093 |
| Mumbai | 19.0760 | 72.8777 |

---

## License

MIT
