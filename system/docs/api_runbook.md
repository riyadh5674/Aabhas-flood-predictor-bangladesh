# System API — Runbook

**Project:** Bangladesh Flood Predictor
**Component:** Inference API (Step 4 of system build)
**Location:** `system/api/`
**Status:** Implemented; the model artefact and feature store are present in this
workspace. The API can be run locally using the instructions below. Production
availability depends on deployment and service supervision.
**Verification snapshot:** 2026-10-30. Latest feature partition: 2026-10-29.

---

## Overview

The inference API serves the frozen D16 model (LightGBM for 24h/48h, XGBoost for 72h)
with Phase-29 cost-derived thresholds and risk bands. It loads features from the
partitioned parquet store produced by Step 3 (Feature Computation Layer).

## Components

| File | Purpose |
|------|---------|
| `config/model_config.json` | Model path, thresholds, risk bands, API settings |
| `config/model_config.py` | Typed constants loaded from JSON |
| `api/model_loader.py` | D16 model wrapper (joblib load, shim modules, predict + risk band) |
| `api/inference.py` | Feature store access, prediction functions (single/batch/all) |
| `api/app.py` | FastAPI application with 7 endpoints |
| `api/run_api.py` | Uvicorn entry point |
| `docs/api_runbook.md` | This document |

## Model Artefact

- **File:** `models/final_v2/selected_model_v2.pkl` (frozen at D16, MI-5, 2026-10-15)
- **Loader:** joblib
- **Per-horizon family:** 24h → lightgbm, 48h → lightgbm, 72h → xgboost
- **Features:** 110 FULL_V2 columns (exact order from `feature_sets_v2.json`)
- **Seed:** 20261008
- **Module shims:** `decision` (numpy.dtype), `baseline_model` (FreqEncoder)

The model artefact requires `baseline_model` (from `scripts/`) and a synthetic
`decision` module to unpickle. Both are shimmed automatically in `model_loader.py`.

## Thresholds & Risk Bands (Phase-29, cost-derived, C_FN=10)

| Horizon | Model | t_warn | Recall | Precision | FAR | Cost/1k |
|---------|-------|--------|--------|-----------|-----|---------|
| 24h | lightgbm | 0.0605 | 0.980 | 0.746 | 0.254 | 27.25 |
| 48h | lightgbm | 0.0868 | 0.967 | 0.727 | 0.273 | 41.16 |
| 72h | xgboost | 0.0963 | 0.954 | 0.684 | 0.316 | 59.45 |

**Risk bands (cost-derived):**
- `low` → routine monitoring
- `moderate` → watch
- `high` → warning
- `very_high` → high-risk warning / activate

Band boundaries vary per horizon (see `phase29_risk_bands.csv`).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health, model loaded status, feature-store coverage and latest feature date |
| GET | `/model/info` | D16 metadata (version, seed, features, frozen date) |
| GET | `/thresholds` | Phase-29 thresholds + risk bands per horizon |
| GET | `/dates` | Available dates with feature data |
| GET | `/predict` | Single station prediction (query params) |
| POST | `/predict/batch` | Batch predictions (JSON body) |
| GET | `/predict/all` | All stations for a date/horizon |

### GET /predict

Query parameters:
- `station_id` (int, required): FFWC station ID
- `date` (string, required): YYYY-MM-DD
- `horizon` (string, optional): 24h \| 48h \| 72h (default: 24h)

Response:
```json
{
  "station_id": 1,
  "date": "2024-07-15",
  "horizon": "24h",
  "probability": 0.127,
  "band": "moderate",
  "action": "watch",
  "threshold": 0.0605,
  "model_version": "D16",
  "computed_at": "2026-10-28T18:00:00+00:00"
}
```

### POST /predict/batch

Request body:
```json
{
  "station_ids": [1, 2, 4, 6],
  "date": "2024-07-15",
  "horizon": "24h"
}
```

Max 133 stations per request.

### GET /predict/all

Query parameters:
- `date` (string, required): YYYY-MM-DD
- `horizon` (string, optional): 24h \| 48h \| 72h (default: 24h)

Returns one prediction per station in the feature store.

## Usage

### Start the API
```powershell
python system/api/run_api.py
# or with custom port
python system/api/run_api.py --port 8080
# or with auto-reload
python system/api/run_api.py --reload
```

Or with uvicorn directly:
```powershell
uvicorn system.api.app:app --host 0.0.0.0 --port 8000
```

### Example calls

```powershell
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Thresholds
curl http://localhost:8000/thresholds

# Available dates
curl http://localhost:8000/dates

# Single prediction
curl "http://localhost:8000/predict?station_id=1&date=2024-07-15&horizon=24h"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch ^
  -H "Content-Type: application/json" ^
  -d '{"station_ids":[1,2,4],"date":"2024-07-15","horizon":"24h"}'

# All stations
curl "http://localhost:8000/predict/all?date=2024-07-15&horizon=24h"
```

### API documentation (auto-generated)

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Dependencies

- `fastapi`, `uvicorn`, `pydantic`
- `joblib`, `numpy`, `pandas`, `pyarrow`
- `scikit-learn`, `lightgbm`, `xgboost` (for model pipelines)
- `scipy` (for api_09/api_07 in features)

Install:
```powershell
pip install fastapi uvicorn pydantic joblib numpy pandas pyarrow scikit-learn lightgbm xgboost scipy
```

## Prerequisites Checklist

- [x] Step 2 (Ingestion) completed — raw data in `system/data/raw/`
- [x] Step 3 (Features) completed — parquet in `system/data/processed/model_ready/`
- [x] Parent repo model artefact present: `models/final_v2/selected_model_v2.pkl`
- [x] Parent repo reference data: `data/raw/ffwc/ffwc_stations.csv`

## Error Handling

| Situation | Response |
|-----------|----------|
| Model artefact missing | `/health` returns `degraded`, prediction endpoints return 500 |
| Feature store missing for date | 404 with detail message |
| Station not found for date | `probability: null`, `band: "unknown"`, `action: "no data"` |
| Invalid date format | 400 |
| Invalid horizon | 400 |
| Batch > 133 stations | 422 (pydantic validation) |

## Latency Targets

- Single prediction: P95 < 500 ms
- Batch (133 stations): P95 < 5 s
- Model preload at startup: < 5 s (artefact is ~5.5 MB)

## Data date semantics

`latest_feature_date` is the newest available feature-store partition date. The
`timestamp` field is the API server's current date; it is not the timestamp of
the newest source observation. Check ingestion manifests and raw source
timestamps to assess observation freshness.
