# System Architecture & Design — Bangladesh Flood Predictor

**Project:** Flood predictor (D:\Flood predictor)
**Status:** DRAFT — Step 1 of system build
**Champion model:** D16 (`models/final_v2/selected_model_v2.pkl`, per-horizon LightGBM 24h/48h + XGBoost 72h, FULL_V2 110 features)
**Governing workflow:** `final workflow.md` — Phases 0–34 complete; system work begins at Phase 35

---

## 1. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Model immutability** | D16 is frozen; system code never retrains or reweights. Model artefacts are versioned by SHA-256. |
| **Separation of concerns** | Ingestion, feature computation, inference, alerting, and UI are separate components with defined contracts. |
| **Fail-safe defaults** | If ingestion fails or features are missing, the system degrades gracefully (returns stale/NaN with a flag), never produces silent wrong predictions. |
| **Idempotent ingestion** | Re-running ingestion for the same date range must produce identical results. |
| **Auditability** | Every prediction, every data fetch, every threshold application is logged with provenance. |
| **Per-station isolation** | One station's bad data never contaminates another station's predictions (D8 — never pool). |

---

## 2. Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SOURCES                         │
│  FFWC API (water levels)  │  Open-Meteo (rainfall)  │  GloFAS   │
└──────────────┬────────────┴────────────┬────────────────────────┘
               │                         │
               ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│   INGESTION LAYER   │   │   INGESTION LAYER   │
│  FFWC water levels  │   │  Rainfall (PWR/CH)  │
│  (3-hourly → daily) │   │  (daily)            │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA STORE (immutable)                   │
│  data/raw/ffwc/waterlevel/{station_id}/{year}.csv              │
│  data/raw/rainfall/{source}/daily/{date}.csv                   │
│  Provenance manifest + SHA-256 per batch                        │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION LAYER                    │
│  (station_id, date) → 110 FULL_V2 feature columns              │
│  - Static (geo): elevation, slope, distances                   │
│  - Hydrology (lags, rolls, rise, pct_danger)                   │
│  - Rainfall (1d/3d/7d/14d/30d, max_7d, wet_days_7)             │
│  - Upstream (same-river lag1/lag2)                             │
│  - Neighbour (nb3/nb5 rain + wl_pct_danger)                    │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL-READY FEATURE STORE (versioned)              │
│  data/processed/model_ready/{date}/station_{id}.parquet         │
│  Schema: station_id, date, f1..f110, feature_schema_version     │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER (API)                        │
│  Load D16 model artefacts (frozen)                              │
│  predict_proba(station_id, date, horizon) → probability         │
│  Apply Phase-29 thresholds → risk band                          │
│  Emit: probability, risk_band, threshold_used, model_version    │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ALERT / EARLY-WARNING ENGINE                       │
│  Risk bands: watch (½×t_warn) / warning (t_warn) / high (≥0.90 precision)│
│  Per-station alert state, persistence, cooldown                 │
│  Output: alert events → notification queue                      │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GEOGRAPHIC / MAP LAYER                          │
│  Station → district → upazila → division mapping                │
│  Risk choropleth (Bangladesh map)                               │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    USER-FACING LAYER (UI)                       │
│  Citizen: location → risk → precaution                          │
│  Admin: map, affected areas, trend, historical comparison       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Contracts

### 3.1 Ingestion Layer

**Input:** External APIs (FFWC, Open-Meteo, GloFAS)
**Output:** Raw CSV files under `data/raw/` + provenance manifest

| Contract | Specification |
|----------|---------------|
| FFWC water level | Fetch 3-hourly readings per station per day; aggregate to daily max/mean/min/first/last; apply F3 (>100m drop) and F8 (spike filter) QC |
| Rainfall | Fetch daily precipitation per station; source: POWER (MERRA-2) or CHIRPS |
| GloFAS | Fetch river discharge at upstream anchors per basin |
| Sentinels | ≤ -1000 → missing; > 100 → impossible; drop both |
| Rate limits | Exponential backoff with jitter; max 5 retries; respect Open-Meteo 429 windows |
| Provenance | `harvest_manifest.csv` records every (station_id, query_date) API call |
| Idempotency | Re-fetching same (station, date) overwrites with identical content |
| Output format | `{source}/{granularity}/{station_id or date}/{year or date}.csv` |

### 3.2 Feature Computation Layer

**Input:** Raw data store + static geographic features
**Output:** Model-ready feature rows (110 FULL_V2 columns)

| Contract | Specification |
|----------|---------------|
| Feature set | Exactly the 110 columns in `models/final_v2/final_config_v2.json` |
| Feature order | Must match model artefact's `features` list exactly |
| NaN policy | Median imputation (train-derived); never zero-fill missing labels |
| Leakage rule | No feature computed from day ≥ t+1 (Phase-13 rule) |
| Upstream lag | Same-river upstream gauge day t-1 only |
| Neighbour pool | 15 nearest stations, take top-3 with reported data |
| Static join | From `data/processed/station_spatial_features.csv` |
| Output | Parquet per (date, station) with schema version |

### 3.3 Inference Layer (API)

**Input:** (station_id, date, horizon) → Output: probability + risk band

| Contract | Specification |
|----------|---------------|
| Model artefact | `models/final_v2/selected_model_v2.pkl` (frozen, SHA-256 verified at load) |
| Horizons | 24h, 48h, 72h (one pipeline per horizon) |
| Thresholds | Phase-29 frozen thresholds from `models/final_v2/final_config_v2.json` |
| Risk bands | Watch = ½ × t_warn; Warning = t_warn; High = precision ≥ 0.90 |
| Response | `{station_id, date, horizon, probability, risk_band, threshold, model_version, computed_at, feature_version}` |
| Error mode | If features missing → return `probability: null, risk_band: "unknown", error: "missing_features"` |
| Latency target | < 100 ms per single prediction; < 5 s for batch of 133 stations |

### 3.4 Alert / Early-Warning Engine

**Input:** Inference output → Output: alert events

| Contract | Specification |
|----------|---------------|
| Threshold source | Phase-29 cost-optimal (C_FN:C_FP = 10:1), period-specific (monsoon/non-monsoon) |
| Band definitions | Watch: p ≥ ½·t_warn; Warning: p ≥ t_warn; High: precision ≥ 0.90 |
| Station state | Persisted per station; cooldown period between repeated alerts |
| Alert event | `{station_id, date, horizon, risk_band, probability, previous_band, triggered_at}` |
| Notification | Queued for dispatch (email/SMS/dashboard) — stub interface for now |

### 3.5 Geographic / Map Layer

**Input:** Station risk bands + admin boundaries → Output: choropleth data

| Contract | Specification |
|----------|---------------|
| Station mapping | station_id → district → upazila → division (from `ffwc_stations.csv`) |
| GeoBoundaries | ADM1 (division), ADM2 (district) from `data/raw/geo/admin/` |
| Output | GeoJSON with risk band per polygon, timestamped |

### 3.6 User-Facing Layer

**Input:** Map data + alert events → Output: HTML/JSON views

| Contract | Specification |
|----------|---------------|
| Citizen view | Location selector (district → upazila → station), risk level, simple explanation, precaution |
| Admin view | Full map, affected areas, trend charts, historical comparison, downloadable CSV/PDF |
| Disclaimer | "Research prototype — not an official government warning system" (Phase 34 rule) |

---

## 4. Data Flow (End-to-End)

```
T-1 day:  Ingestion fetches FFWC water levels + rainfall for day T-1
          → raw CSV written + manifest updated

T day 00:00 UTC:  Feature computation runs for day T
                   → reads raw water levels (day T), rainfall (day T), upstream (day T-1)
                   → computes 110 FULL_V2 features
                   → writes model-ready parquet

T day 00:30 UTC:  Inference API pre-computes predictions for all 133 stations
                   horizons 24h/48h/72h
                   → applies Phase-29 thresholds
                   → writes prediction results

T day 01:00 UTC:  Alert engine evaluates risk bands against previous state
                   → emits alert events for band transitions
                   → queues notifications

On demand:  UI queries prediction results + map data
            → citizen/admin views rendered
```

---

## 5. Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      HOST (Windows 11 / Linux)                  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Ingestion   │  │  Feature   │  │  Inference   │          │
│  │  (scheduler) │  │  (batch)   │  │  API (FastAPI)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FILE SYSTEM                          │   │
│  │  data/raw/    data/processed/  models/  reports/        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Alerts     │  │   Map/UI     │  │   Tests      │          │
│  │  (worker)    │  │  (Streamlit/ │  │  (pytest)    │          │
│  └──────┬───────┘  │   FastAPI)   │  └──────────────┘          │
│         │          └──────┬───────┘                           │
│         ▼                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LOGS & MONITORING                      │   │
│  │  ingestion.log  feature.log  api.log  alert.log           │   │
│  │  drift_metrics.json  prediction_log.csv                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Execution model:** Scheduled tasks (Windows Task Scheduler or cron) trigger ingestion and feature batches; FastAPI serves inference on-demand; alert worker polls prediction results.

---

## 6. Configuration Management

All configuration in `system/config/`, never hardcoded:

| File | Contents |
|------|----------|
| `config/model_config.json` | D16 model path, SHA-256, feature list, thresholds per horizon/season |
| `config/ingestion_config.json` | API endpoints, rate limits, retry policy, station list |
| `config/feature_config.json` | 110 FULL_V2 column names, imputation values, leakage rules |
| `config/alert_config.json` | Risk band thresholds, cooldown periods, notification rules |
| `config/system_config.json` | Paths, log levels, SLA targets, environment |

---

## 7. Error Handling & Resilience

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| FFWC API down | HTTP error / timeout | Retry with backoff; log; use last known good water level |
| Open-Meteo 429 | Rate-limit header | Exponential backoff; queue for next window |
| Missing station data | NaN in feature row | Impute with train median; flag `data_quality: degraded` |
| Model load failure | SHA-256 mismatch | Refuse to serve; alert operator; fall back to D7 persistence |
| Feature schema mismatch | Column count/order mismatch | Refuse prediction; log error; alert operator |
| Inference timeout | > 5 s batch | Return partial results; log slow query |

---

## 8. Monitoring & Observability

| Metric | Where | Threshold |
|--------|-------|-----------|
| Ingestion success rate | ingestion.log | ≥ 99% stations per day |
| Feature null rate | feature.log | < 5% per column |
| Prediction latency | api.log | P95 < 500 ms |
| Drift detection | drift_metrics.json | Feature distribution shift > 2σ from train |
| Alert volume | alert.log | Sudden 2× spike triggers review |

---

## 9. Directory Layout (system/)

```
system/
├── architecture/          # This document + diagrams
├── config/                # All configuration (JSON/YAML)
├── ingestion/             # Data fetch scripts (FFWC, rainfall, GloFAS)
├── features/              # Feature computation pipeline
├── api/                   # FastAPI inference service
├── alerts/                # Early-warning engine
├── ui/                    # Citizen + admin interfaces
├── tests/                 # Unit/integration tests per component
└── docs/                  # Runbooks, API reference, deployment guide
```

---

## 10. Definition of Done — Step 1

- [x] Component map with clear boundaries
- [x] Interface contracts for all 6 layers
- [x] End-to-end data flow defined
- [x] Deployment topology specified
- [x] Configuration management plan
- [x] Error handling & resilience policy
- [x] Monitoring & observability plan
- [x] Directory layout created

**Next step (Step 2):** Build the Ingestion Layer — scheduled FFWC water-level fetch + rainfall fetch with QC gates and provenance manifest.

---

**Approval requested:** Proceed to Step 2 (Ingestion Layer implementation) after this architecture is accepted.