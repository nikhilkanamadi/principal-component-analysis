# Weather Anomaly Detection — End-to-End ML Pipeline

A production-style FastAPI application that demonstrates a complete machine learning pipeline:
**API Integration → Data Pipeline → Feature Engineering → PCA → Anomaly Detection → Monitoring → Deployment**

Real daily weather data is fetched from the [Open-Meteo](https://open-meteo.com) archive API (free, no API key required), processed through a feature engineering pipeline, compressed with PCA, and scored by an Isolation Forest to surface anomalous weather events.

---

## Architecture

```
Open-Meteo API
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  POST       │     │  POST            │     │  POST            │
│  /ingest    │────▶│  /pipeline/      │────▶│  /train          │
│             │     │  process         │     │                  │
│  Fetch &    │     │  Clean + Engineer│     │  StandardScaler  │
│  store raw  │     │  21 features     │     │  → PCA (n comps) │
│  weather    │     │                  │     │  → IsolationForest│
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                      │
                    ┌──────────────────┐              │ saved model
                    │  GET             │              │ (.joblib)
                    │  /monitor        │◀─────────────┤
                    │                  │              │
                    │  Health report,  │     ┌────────▼─────────┐
                    │  feature stats,  │     │  POST            │
                    │  model metadata  │     │  /predict        │
                    └──────────────────┘     │                  │
                                             │  Fetch new data  │
                           SQLite            │  → anomaly score │
                        (weather.db)         │  per day         │
                                             └──────────────────┘
```

---

## Skills Demonstrated

| Layer | What it covers |
|---|---|
| **API Integration** | `httpx` client against Open-Meteo historical archive; parameterised by lat/lon, date range, and variable list |
| **Data Pipeline** | Missing-value handling (forward/back-fill), sort-by-date, schema normalisation via pandas |
| **Feature Engineering** | 21 features: raw meteorological variables, derived ratios (`wind_gust_ratio`, `temp_range`), 7-day rolling mean/std, deviation-from-rolling-mean signals, temporal features (`day_of_year`, `month`) |
| **PCA / Dimensionality Reduction** | `sklearn.decomposition.PCA` — configurable component count, explained-variance reported per component |
| **Storage** | SQLite via `sqlite3` — two tables (`raw_weather`, `processed_weather`), upsert on `(location, date)` |
| **Model Training** | `sklearn.ensemble.IsolationForest` trained on PCA-projected space; model artifact serialised with `joblib` |
| **Monitoring** | `/monitor` endpoint reports raw/processed record counts, training timestamp, explained variance, per-feature descriptive stats, and training anomaly rate |
| **Deployment** | FastAPI + Uvicorn; Pydantic v2 request/response validation; OpenAPI docs auto-generated at `/docs` |

---

## Project Structure

```
principal-component-analysis/
├── app/
│   ├── main.py                  # FastAPI app, lifespan (DB init)
│   ├── api/
│   │   ├── ingest.py            # POST /ingest
│   │   ├── pipeline.py          # POST /pipeline/process
│   │   ├── train.py             # POST /train
│   │   ├── predict.py           # POST /predict
│   │   └── monitor.py           # GET  /monitor
│   ├── core/
│   │   ├── data_fetcher.py      # Open-Meteo HTTP client
│   │   ├── pipeline.py          # Cleaning + feature engineering
│   │   ├── model.py             # PCA + Isolation Forest train/predict
│   │   ├── storage.py           # SQLite read/write helpers
│   │   └── monitoring.py        # Report builder
│   └── schemas/
│       └── models.py            # Pydantic request & response models
├── data/                        # Auto-created: weather.db + *.joblib models
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/nikhilkanamadi/principal-component-analysis.git
cd principal-component-analysis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

---

## API Walkthrough

Work through the endpoints in order — each step feeds the next.

### Step 1 — Ingest raw weather data

Fetch a full year of daily weather for any location on Earth.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 51.5074,
    "longitude": -0.1278,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "location_name": "London"
  }'
```

```json
{
  "location": "London",
  "rows_stored": 365,
  "date_range": "2023-01-01 → 2023-12-31",
  "message": "Successfully ingested 365 daily records for London."
}
```

**What happens internally:**
- Calls Open-Meteo `/archive` endpoint for 8 daily weather variables
- Stores raw rows in SQLite `raw_weather` table with upsert on `(location, date)`

---

### Step 2 — Process & engineer features

Clean the raw data and compute 21 engineered features.

```bash
curl -X POST http://localhost:8000/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{"location_name": "London"}'
```

```json
{
  "location": "London",
  "rows_processed": 365,
  "features": [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "temp_range", "wind_gust_ratio", "precip_flag",
    "temperature_2m_mean_roll7_mean", "temperature_2m_mean_roll7_std",
    "precipitation_sum_roll7_mean", "precipitation_sum_roll7_std",
    "wind_speed_10m_max_roll7_mean", "wind_speed_10m_max_roll7_std",
    "temp_deviation", "precip_deviation",
    "day_of_year", "month"
  ],
  "message": "Processed 365 records with 21 engineered features."
}
```

**Feature engineering highlights:**
| Feature | Description |
|---|---|
| `temp_range` | Daily max − min temperature |
| `wind_gust_ratio` | Gust speed ÷ sustained wind (turbulence proxy) |
| `precip_flag` | Binary: any precipitation that day |
| `*_roll7_mean/std` | 7-day rolling mean and standard deviation |
| `temp_deviation` | Temperature departure from 7-day rolling mean |
| `precip_deviation` | Precipitation departure from 7-day rolling mean |

---

### Step 3 — Train PCA + Isolation Forest

Fit a `StandardScaler → PCA → IsolationForest` pipeline and persist the artifact.

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "location_name": "London",
    "n_components": 3,
    "contamination": 0.05
  }'
```

```json
{
  "location": "London",
  "n_components": 3,
  "explained_variance_ratio": [0.2825, 0.2033, 0.1048],
  "total_variance_explained": 0.5906,
  "training_samples": 365,
  "anomalies_found": 19,
  "message": "Model trained on 365 samples. 19 anomalies flagged during training."
}
```

**Training parameters:**
| Parameter | Default | Description |
|---|---|---|
| `n_components` | `3` | Number of PCA components (1–10) |
| `contamination` | `0.05` | Expected fraction of anomalies (0.01–0.5) |

---

### Step 4 — Predict anomalies on new data

Fetch unseen data, run it through the saved pipeline, and get a per-day anomaly verdict.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 51.5074,
    "longitude": -0.1278,
    "location_name": "London",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }'
```

```json
{
  "location": "London",
  "total_records": 31,
  "anomalies_detected": 8,
  "anomaly_rate": 0.2581,
  "records": [
    {
      "date": "2024-01-02",
      "anomaly_score": -0.640344,
      "is_anomaly": true,
      "temperature_2m_max": 12.9,
      "wind_speed_10m_max": 52.3,
      "precipitation_sum": 7.4
    }
    ...
  ]
}
```

`anomaly_score` is the raw Isolation Forest score — more negative = more anomalous.

---

### Step 5 — Monitor

Get a health snapshot: record counts, model metadata, and feature statistics.

```bash
curl "http://localhost:8000/monitor?location=London"
```

```json
{
  "location": "London",
  "raw_records": 365,
  "processed_records": 365,
  "model_trained": true,
  "last_trained": "2026-06-01T22:53:23.948418",
  "n_pca_components": 3,
  "explained_variance": 0.5906,
  "training_anomaly_rate": 0.0521,
  "feature_stats": {
    "temperature_2m_mean": { "mean": 11.64, "std": 5.62, "min": -2.4, "max": 24.1 },
    "precipitation_sum":   { "mean": 2.14,  "std": 3.73, "min": 0.0,  "max": 27.4 },
    "wind_speed_10m_max":  { "mean": 22.41, "std": 7.97, "min": 5.4,  "max": 50.4 }
  }
}
```

---

## Data Source

All weather data is sourced from **[Open-Meteo](https://open-meteo.com)** — a free, open-source weather API with no authentication required.

**Variables fetched (daily):**

| Variable | Unit |
|---|---|
| `temperature_2m_max` / `min` / `mean` | °C |
| `precipitation_sum` | mm |
| `wind_speed_10m_max` | km/h |
| `wind_gusts_10m_max` | km/h |
| `shortwave_radiation_sum` | MJ/m² |
| `et0_fao_evapotranspiration` | mm |

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Web framework | [FastAPI](https://fastapi.tiangolo.com) 0.115 |
| ASGI server | [Uvicorn](https://www.uvicorn.org) 0.30 |
| HTTP client | [httpx](https://www.python-httpx.org) 0.27 |
| Data processing | [pandas](https://pandas.pydata.org) 2.2, [NumPy](https://numpy.org) 1.26 |
| ML | [scikit-learn](https://scikit-learn.org) 1.5 |
| Model serialisation | [joblib](https://joblib.readthedocs.io) 1.4 |
| Validation | [Pydantic](https://docs.pydantic.dev) v2 |
| Storage | SQLite (stdlib `sqlite3`) |

---

## Example Locations

| City | Latitude | Longitude |
|---|---|---|
| New York | 40.7128 | -74.0060 |
| London | 51.5074 | -0.1278 |
| Tokyo | 35.6762 | 139.6503 |
| Sydney | -33.8688 | 151.2093 |
| Mumbai | 19.0760 | 72.8777 |

---

## License

MIT
