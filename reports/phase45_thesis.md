# Leakage-Aware Probabilistic Flood Forecasting for Bangladesh

### From a frozen research model to an operational decision-support prototype

**Phase 45 / stage 73 — research report (thesis draft)**
**Status:** COMPLETE · **Written:** 2026-11-06 · **Seed:** `20261008`
**Model:** D16 (`models/final_v2/selected_model_v2.pkl`) · **Feature set:** FULL_V2 (110)
**Frozen test policy:** closed — `test_rows_read = 0` in every analysis added after MI-6

> **Research prototype — not an official flood-warning service.**
> FFWC advisories and government instructions take precedence. The system is intended
> for decision support, evaluation, and further research.

All quantitative statements below are traceable to the artefacts listed in
`reports/FREEZE.md` and `reports/freeze_manifest.json`. The frozen test period was read
exactly once (MI-6); no result in this report re-opens it.

---

## Abstract

Flood early warning in Bangladesh must answer a probabilistic question: *given the
information available on issue day t, what is the chance that a station crosses its
danger level within the next 24, 48, or 72 hours?* This report documents a
leakage-controlled machine-learning pipeline built around that question, the frozen
D16 model, and the system that turns its output into station-level risk bands and
alert records.

D16 is a per-horizon gradient-boosting model (LightGBM at 24 h and 48 h, XGBoost at
72 h) over 110 hydrology, rainfall, soil, transboundary-discharge, network, and static
features. On the primary validation cell (`val_2022`, 84 non-holdout stations) it reaches
a horizon-mean PR-AUC of **0.957427** (0.970789 / 0.956485 / 0.945007). On the single
frozen test read it reaches **0.770758** (0.836227 / 0.768959 / 0.707087), compared with
**0.667549** for the persistence baseline, and beats persistence on **21/21** test
cell×horizon points. A fair deep-learning rematch did not beat the classical model
(best DL 0.933991 vs D16 0.957427 on `val_2022`), so the deep-learning queue was closed.

The engineering contribution is as important as the score. A hard-coded end date in
three source fetchers silently degraded 42 of 110 features for roughly three weeks in
October–November 2026. The incident motivated incremental source refreshers, a
mandatory feature-quality gate, per-source freshness reporting, an approval-gated alert
engine with hash-chained audit logging, and an 84-test contract suite. The result is a
reproducible research artefact and a deployable prototype whose failure modes are
observable rather than silent.

---

## 1. Introduction

### 1.1 Problem

Bangladesh sits at the confluence of the Ganges, Brahmaputra, and Meghna. Riverine
flooding is driven by monsoon rainfall, upstream discharge, soil saturation, and the
slow propagation of water through a dense network of channels. Operational forecasts
are typically deterministic and short-lead. A probabilistic outlook that states *where*
and *when* risk rises, with an explicit false-alarm/miss trade-off, is more useful for
pre-positioning and public communication than a single binary forecast.

The prediction unit is therefore **station × issue day × horizon**, and the target is
`P(any reading in (t, t+h] ≥ danger level)` for `h ∈ {24, 48, 72}` hours.

### 1.2 Objectives

1. Build a leakage-controlled, reproducible pipeline for station-level flood
   probability at 24/48/72 h.
2. Compare classical and deep models fairly on identical splits, labels, features,
   metrics, seeds, and tuning budgets.
3. Freeze one model and publish its decision layer (thresholds and risk bands).
4. Turn the model into a usable system: ingestion, features, quality gates, inference
   API, map/UI, alerts, monitoring, and audit.
5. Document the transition from research code to operations, including a real
   production incident and its remediation.

### 1.3 Contributions

- **Label and split contract.** A strict issue-time label, a 7-day embargo, temporal
  splits (train < 2022, val 2022, test ≥ 2023), and station-disjoint spatial folds.
- **Frozen D16 model.** Per-horizon gradient boosting over FULL_V2 (110 features),
  selected on validation evidence only, then read once on test.
- **Decision layer.** Cost-derived thresholds (`C_FN:C_FP = 10:1`) and four risk bands
  with an observed flood rate of ~90% in the top band.
- **Operational system.** Daily pipeline with source refresh, quality gate, static
  snapshot, FastAPI inference service, React UI, alert engine, and monitoring.
- **Production lesson.** A documented null-feature incident caused by hard-coded end
  dates, and the controls added to prevent recurrence.

### 1.4 Scope and limitations

This is a research prototype. It is not an official warning service, does not replace
FFWC, and has not been validated by a government agency. The frozen model was trained
on data through 2021 and validated on 2022; the one permitted test read covers
2023–2026. Post-2022 data are used for evaluation and live inference, not retraining.

---

## 2. Data and definitions

### 2.1 Study area and station network

- **133 FFWC stations** appear in the experimental table.
- **84 non-holdout stations** provide the P1 training and validation rows.
- **26 stations** form the fold-0 spatial holdout cohort (used for P3 spatial-CV).
- The remaining non-holdout stations appear only in the test period.
- Each station carries river, basin, division, district, upazila, latitude, longitude,
  and danger-level metadata.

### 2.2 Experimental table

| Property | Value |
| --- | --- |
| File | `data/processed/experimental_dataset.csv` |
| Rows × columns | **200,665 × 67** |
| Date range | 2017-01-06 … 2026-10-05 |
| SHA-256 | `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` |
| Seed | `20261008` |

### 2.3 Label definition

For issue date `t` and horizon `h`:

```
label_h(t) = 1  if any observed water-level reading in (t, t+h] is ≥ danger level
           = 0  otherwise
           = NaN if the horizon window is not fully observed
```

Missing labels stay `NaN`, never `0`. Columns describing the label window
(`obs_days_24h`, `obs_days_48h`, `obs_days_72h`) are banned as features. Labels are
verified by `tests/test_label_contract.py`.

Positive rates are **4.98% / 5.68% / 6.32%** at 24/48/72 h, reflecting the class
imbalance that makes accuracy an inappropriate headline metric.

### 2.4 Splits

| Split | Dates | Rows | Role |
| --- | --- | --- | --- |
| Train | 2017-01-06 … 2021-11-03 | 77,571 | Model fitting (non-holdout stations) |
| Validation | 2022-03-15 … 2022-10-30 | 21,920 | Selection, calibration, thresholds |
| Test | 2023-01-27 … 2026-10-05 | 101,174 | Frozen; read once in MI-6 |

A **7-day embargo** separates adjacent periods with zero rows inside the embargo
windows. The 2018 archive gap is never interpolated.

### 2.5 Sources

| Source | Variables | Native cadence | Typical lag | Role |
| --- | --- | --- | --- | --- |
| FFWC | Water level, danger level | 3-hourly / daily aggregate | hours | Target and hydrology |
| NASA POWER | Rainfall, soil wetness | Daily | 2–3 d | Rainfall, soil memory |
| Open-Meteo ERA5-Land | Precipitation, temperature, soil moisture | Daily | 5–7 d | Weather context |
| GloFAS | Upstream/nearby river discharge | Daily | 1–2 d | Transboundary signal |
| ESA WorldCover / SRTM / OSM | Landcover, elevation, rivers | Static | — | Static geography |
| geoBoundaries | ADM1/ADM2 boundaries | Static | — | Map |

Independent flood-extent observations are not available; this is documented as
limitation **R8**.

### 2.6 Feature set FULL_V2 (110 features)

| Group | Prefix / examples | Count | Quality-gate status |
| --- | --- | ---: | --- |
| Static / geography / station metadata | `lat`, `lon`, `elevation_m`, `dist_*`, encoded geography | 19 | informational |
| Hydrology | `wl_*` | 26 | watch |
| Rainfall | `precip_mm`, `rain_*` | 7 | watch |
| Temperature | `tmax_c`, `tmin_c` | 2 | **critical** |
| Seasonality | `doy_sin`, `doy_cos` | 2 | informational |
| Network | `nb3_*`, `nb5_*`, `up_*` | 12 | watch |
| Soil wetness | `ps_*` | 19 | **critical** |
| GloFAS discharge | `gf_*` | 18 | **critical** |
| Interactions | `ix_*` | 5 | **critical** |

The quality gate treats GloFAS, soil, interactions, and temperature as critical because
the frozen model uses them directly. An all-null column always fails the gate.

---

## 3. Methodology

### 3.1 Leakage controls

1. **Temporal separation.** Training data precede validation data, which precede test
   data. No future information enters a feature row.
2. **7-day embargo.** Zero rows fall inside the embargo windows between periods.
3. **Station-disjoint spatial folds.** Fold 0 (26 stations) is held out from spatial-CV
   fitting. Unseen-station and new-network test buckets are reported separately.
4. **Label-window ban.** `obs_days_*` columns are excluded from every feature set.
5. **Frozen test.** All selection, calibration, thresholds, ablations, and
   explainability use development cells only. The test read is logged once.

### 3.2 Model

D16 is selected by MI-5 and frozen under `models/final_v2/`:

| Horizon | Family | Key hyperparameters |
| --- | --- | --- |
| 24 h | LightGBM | 300 trees, lr 0.02, 63 leaves, min child 40, subsample 0.85, colsample 0.6, λ 3 |
| 48 h | LightGBM | same as 24 h |
| 72 h | XGBoost | 800 trees, lr 0.02, max depth 4, subsample 1.0, colsample 0.8, λ 3 |

Categorical geography is frequency-encoded with statistics fitted on the training fold;
numeric missing values are median-imputed inside the pipeline. Calibration and
thresholds are applied after the raw probability.

### 3.3 Evaluation protocol

- **Primary metric:** PR-AUC (average precision), because positives are rare.
- **Secondary:** ROC-AUC, Brier score, ECE, precision, recall, F1, false-alarm rate.
- **Granularity:** every result is reported per cell × horizon. Pooling across years or
  basins is explicitly avoided because base rates drift.
- **Uncertainty:** paired station bootstrap confidence intervals for model comparisons.
- **Decision evaluation:** cost `C_FN: C_FP = 10:1`, threshold sweep 0.01–0.60, and
  event-level recall/false alarms per station-year.

---

## 4. Candidate models and fair comparison

Twenty-four models were compared on identical dev cells:

- **L1 baselines:** climatology, persistence (D7), regularised logistic regression
  variants, shallow tree.
- **L2 classical:** random forest, histogram gradient boosting, XGBoost, LightGBM,
  XGBoost-SELECTED.
- **L3 temporal / fusion:** LSTM at several lookbacks, GRU, TCN, static-only control,
  fusion and hybrid architectures.
- **L4 spatiotemporal:** GCN, GAT, GNN-LSTM, temporal Transformer.

On `val_2022` the best classical model reaches **0.954634** (Phase-33 XGBoost on FULL)
and the best deep model reaches **0.933991** (MI-7 `fusion_v2_L3`). D16 reaches
**0.957427**. The MI-7 rematch used equal features and an equal tuning budget; paired
DL–ML confidence intervals excluded zero in ML's favour at all three horizons. The
deep-learning queue was therefore closed, and D16 remained champion after the MI-8
score push failed to win a paired confidence interval.

### 4.1 Robustness evidence

- Leaders beat persistence in **63/63** all-season cell×horizon points (minimum margin
  **+0.0143**).
- Leaders beat persistence in **126/126** seasonal points.
- The temporal monsoon/non-monsoon gap is ≤ **0.0435**.
- On unseen-station folds the seasonal gap reaches **0.2586**, but all models weaken
  there; it is not a comparative loss.
- The best deep model (`fusion_L3`) collapses to **0.7657** on spatial fold 1 at 24 h,
  versus a leader minimum fold of **0.9147**.

---

## 5. Results

### 5.1 Validation — `val_2022`

| Model | 24 h | 48 h | 72 h | Horizon mean |
| --- | ---: | ---: | ---: | ---: |
| **D16 (frozen)** | **0.970789** | **0.956485** | **0.945007** | **0.957427** |
| Best classical (Phase-33 XGBoost, FULL) | 0.9681 | 0.9549 | 0.9408 | 0.954634 |
| Best deep (MI-7 `fusion_v2_L3`) | — | — | — | 0.933991 |
| Persistence (D7) | 0.9203 | 0.8954 | 0.8683 | 0.8947 |

D16 improves on the previous frozen incumbent by **+0.0028** horizon-mean and on
persistence by **+0.0627**.

### 5.2 Frozen test read (MI-6, single read)

| Horizon | D16 PR-AUC | D7 PR-AUC | Margin |
| --- | ---: | ---: | ---: |
| 24 h | **0.836227** | 0.733961 | +0.102266 |
| 48 h | **0.768959** | 0.662517 | +0.106441 |
| 72 h | **0.707087** | 0.606167 | +0.100920 |
| **Mean** | **0.770758** | **0.667549** | **+0.103209** |

D16 wins **21/21** test cell×horizon points; the smallest margin is **+0.0424**.

Per test year (PR-AUC, D16 / D7):

| Cell | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: |
| `test_2023` | 0.7956 / 0.6276 | 0.6814 / 0.5523 | 0.6257 / 0.5108 |
| `test_2024` | 0.9177 / 0.8022 | 0.8574 / 0.7302 | 0.8043 / 0.6727 |
| `test_2025` | 0.8048 / 0.7328 | 0.7598 / 0.6670 | 0.7019 / 0.6041 |
| `test_2026` | 0.7935 / 0.7233 | 0.7467 / 0.6495 | 0.7024 / 0.5929 |

Unseen-station (`test_unseen`) and new-network (`test_new_network`) buckets remain above
persistence at every horizon. The weakest cell is unseen-station 72 h (0.6411), which is
the expected cost of spatial generalisation.

**Interpretation.** Test PR-AUC is lower than validation because 2023–2026 contains
harder years, different station composition, and monsoon/dry-season mix shifts. The
relative margin over persistence is stable across years and horizons.

### 5.3 Calibration

- 17 of 24 models clear the all-season ECE ≤ 0.02 bar; the median raw ECE is **0.0067**.
- D16's out-of-fold ECE is **0.0031**.
- Miscalibration is structural over-confidence, worst in monsoon and growing with lead
  time; it is concentrated in recurrent models and random forest.
- Platt/isotonic recalibration fitted on validation and scored out-of-fold brings every
  skill model within tolerance. D16 is used uncalibrated because it already clears the
  bar; the option remains available in the decision layer.

### 5.4 Thresholds and risk bands

Cost-optimal thresholds on `val_2022` (`C_FN:C_FP = 10:1`):

| Horizon | Leader | Threshold | Recall | Precision | FAR | Missed rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 24 h | LightGBM | 0.0605 | 0.980 | 0.746 | 0.254 | 18 |
| 48 h | XGBoost | 0.0524 | 0.982 | 0.667 | 0.333 | 19 |
| 72 h | XGBoost-SELECTED | 0.0614 | 0.966 | 0.630 | 0.370 | 38 |

The trade-off is deliberately recall-heavy: at 10:1, a missed flood is ten times more
costly than a false alarm.

Risk bands on the 24 h leader:

| Band | Probability range | Share of rows | Observed flood rate | Lift | Action |
| --- | --- | ---: | ---: | ---: | --- |
| Low | < 0.0303 | 92.7% | 0.07% | 0.01× | routine monitoring |
| Moderate | 0.0303–0.0605 | 0.6% | 6.4% | 1.26× | watch |
| High | 0.0605–0.434 | 1.6% | 24.1% | 4.73× | warning |
| Very high | ≥ 0.434 | 5.1% | **90.1%** | **17.69×** | high-risk warning / activate |

The top band is the operational signal: it is rare (≈5% of rows) and contains a ~90%
observed flood rate. The quartile-band default was rejected because it placed 50–75% of
rows in zero-flood bands.

Period-specific thresholds matter. Carrying a dry-season threshold into monsoon loses
up to **9 points** of recall for the leaders and **16.6 points** for the crippled
static-only control. The deployed configuration uses the all-season threshold and
exposes the per-period values in `/thresholds` for future policy tuning.

### 5.5 Error analysis

Ten named error patterns were quantified. The three most important:

- **Pattern A — rising but still below danger.** Missed floods occur while the river is
  below danger but rising (FN median `wl_dist_to_danger` +0.76 m). This is a
  response-lag limitation: the model sees the trend but the threshold crossing arrives
  faster than the lead time.
- **Pattern F — false alarms just below danger.** False alarms cluster just below
  danger (median `wl_pct_danger` 0.983; 80.9% below danger). This is the intended price
  of `FN ≫ FP`.
- **Pattern E — data-poor stations.** Stations with the fewest training floods are
  missed about **14×** more often (0.230 vs 0.016 miss rate). Targeted oversampling or
  pooling for low-prevalence stations is the obvious next step.

### 5.6 Event-level behaviour (development only)

Strict pre-onset event evaluation on development cells (no test read):

| Metric | D16 mean | D7 mean |
| --- | ---: | ---: |
| Event recall (24/48/72 h) | 0.509 / 0.724 / 0.881 | 0.452 / 0.649 / 0.767 |
| Event precision (24/48/72 h) | 0.567 / 0.543 / 0.385 | 0.300 / 0.348 / 0.347 |
| False alarms per station-year | 0.58 / 0.98 / 2.30 | 1.56 / 1.97 / 2.44 |
| Median lead time (days) | 1.00 / 1.86 / 2.86 | 1.00 / 1.86 / 2.71 |

D16 improves event recall at every horizon while roughly halving false alarms at 24 h
and 48 h. The 72 h false-alarm rate (2.30 per station-year) is the main operational
cost and should be monitored in any pilot.

### 5.7 Explainability

Exact TreeSHAP on D16 (`val_2022`, 17,502 rows at 24 h) shows a hydrology-first model:

- **G4 hydrology = 64.0%** of 24 h attribution.
- Then geography 8.4%, network 6.1%, weather 4.7%.
- Top features: `wl_dist_to_danger` (mean |SHAP| 1.005), `wl_pct_danger` (0.539),
  `wl_exceed_days_30` (0.134), `lon` (0.085), `wl_rise_1d` (0.074).
- No banned label-window column appears in any horizon's top-20.
- Deep-model integrated gradients agree on the dominant hydrology layer
  (74.7% Transformer, 70.6% LSTM), so the tabular and sequence models learn the same
  physical signal even though the sequence models do not score better.

---

## 6. System implementation

### 6.1 Architecture

```
Sources (FFWC, NASA POWER, ERA5-Land, GloFAS)
        │  incremental refresh + manifests
        ▼
Daily pipeline ──► validation / QC ──► feature build (FULL_V2) ──► quality gate
        │                                                           │
        │                                                           ▼
        │                                             feature store (partitioned)
        ▼                                                           │
static snapshot (precomputed forecasts)                             │
        │                                                           ▼
        │                                             FastAPI inference (D16)
        │                                                           │
        ├──────────────► React web UI / map ◄───────────────────────┤
        │                                                           │
        └──────────────► alert engine (bands → approval → audit) ◄──┘
```

### 6.2 Daily pipeline

`system/ops/daily_pipeline.py` runs:

0. **Source refresh** — GloFAS, NASA POWER soil, ERA5-Land, incrementally, with
   `--end` / `--lookback-days` and per-source freshness checks.
1. **FFWC ingestion** — resumable station water-level ingest.
2. **Rainfall merge** — NASA POWER and Open-Meteo.
3. **Feature build** — FULL_V2, full-history computation then slice to output dates.
4. **Quality gate** — critical-group null share ≤ 5%, all-null column check.
5. **Snapshot / publish** — static snapshot, feature store, and optional HF upload.

`system/ingestion/refresh_sources.py` writes
`system/data/processed/source_status/<date>.json`, which `/health` and `/monitoring`
consume. A stale critical source blocks the pipeline unless `--allow-degraded` is
explicitly passed.

At the time of writing, local freshness is: GloFAS through 2026-11-05, POWER soil
through 2026-11-02, ERA5-Land through 2026-11-03, feature store through 2026-11-05,
and the quality gate passes with zero critical null share.

### 6.3 API

Public endpoints: `/health`, `/monitoring`, `/model/info`, `/thresholds`, `/dates`,
`/predict`, `/predict/batch`, `/predict/all`, `/predict/trend`, `/alerts/feed`.
Admin endpoints (`/admin/status`, `/admin/pipeline/trigger`, `/admin/model/reload`,
`/alerts/evaluate`, `/alerts/{id}/approve`) require `X-Admin-API-Key`.

Operational behaviour includes per-IP sliding-window rate limiting, security headers,
a `/api` prefix alias for same-origin proxies, request IDs, structured access logs,
gzip, a bounded per-date prediction cache, and a preview-mode basic-auth gate.

### 6.4 Web interface

The React/Vite client supports location selection (district → upazila → station), a
risk map, station detail with trend, horizon switching, bilingual labels, a
freshness banner, an explicit research-prototype disclaimer, and a PWA/offline shell.
It can read a precomputed static snapshot or the live API.

### 6.5 Alert engine

`system/alerts/alert_engine.py` maps bands to alert levels:

| Band | Alert level | Default handling |
| --- | --- | --- |
| Low | none | no alert |
| Moderate | watch | draft; auto-approval optional |
| High | warning | draft; duty-officer approval |
| Very high | high | draft; duty-officer approval |

Alerts use a 24 h per-station/horizon cooldown. Escalation to a worse band bypasses the
cooldown when configured; otherwise it waits and remains pending rather than being
downgraded to a routine re-fire. Delivery is via email, webhook, or dashboard; delivery
is **disabled by default** in `system/config/alert_config.json` so the prototype cannot
send real warnings without an operator enabling it. Every evaluation, draft, approval,
and dispatch is written to a hash-chained JSONL audit log.

### 6.6 Deployment status

- The local stack (pipeline → feature store → API → snapshot → UI) is verified working.
- The public static demo is a precomputed snapshot and must be rebuilt/redeployed to
  show new dates. The last public deployment predates the November data refresh.
- Full cloud hosting (paid tier, managed scheduler, SMTP/webhook credentials) is
  intentionally out of scope for this phase. The repository already contains Docker
  and Render configuration for a Docker-capable host.

---

## 7. Production incident: silent null features (October–November 2026)

### 7.1 Detection

A routine health check showed `gf_*` and `ps_*` features at or near 100% null. The
feature rows still built, so the pipeline did not fail; it produced degraded forecasts
silently.

### 7.2 Root cause

Three source fetchers contained a hard-coded end date (`2026-10-12`). After that date
they kept re-downloading the same tail instead of extending it. The model consumed
stale GloFAS and soil values while appearing healthy.

### 7.3 Impact

- 37 of 110 features were null (33.6%).
- Five interaction features derived from them were also compromised.
- 42 of 110 features (38%) were unusable for the affected dates.
- Duration: roughly 2026-10-13 to 2026-11-03.

### 7.4 Remediation

1. Rewrote the three fetchers as incremental refreshers with dynamic defaults,
   `--end`, and `--lookback-days`.
2. Added `system/ingestion/refresh_sources.py` as the single source-refresh entrypoint.
3. Added the feature-quality gate and made it mandatory; degradation requires an
   explicit `--allow-degraded`.
4. Backfilled the missing dates and rebuilt the affected feature partitions.
5. Added per-source freshness to `/health` and `/monitoring`.
6. Added regression tests for freshness classification, quality-gate failure, and
   alert cooldown/approval behaviour.

### 7.5 Lessons

- Never hard-code a data end date; derive it and make it overridable.
- A pipeline that cannot fail is not a quality gate.
- Freshness must be visible at the same level as model health.
- Single-laptop automation is a single point of failure; scheduling belongs on a
  managed runner.
- Every incident needs a runbook and a regression test, not just a fix.

---

## 8. Limitations

| ID | Limitation | Consequence |
| --- | --- | --- |
| R1 | Research prototype, not an official warning service | Requires government validation before public use |
| R2 | Frozen model trained through 2021 | Climate and station-network non-stationarity after 2022 |
| R3 | 72 h skill is materially weaker | Wider uncertainty; higher false-alarm rate |
| R4 | Daily, not sub-daily, updates | Cannot capture rapid within-day escalation |
| R5 | Spatial holdout and low-prevalence stations | Lower recall on data-poor stations (Pattern E) |
| R6 | No independent flood-extent labels | Label ceiling bounded at ~0.96–0.97 |
| R7 | Calibration but no full uncertainty quantification | No conformal intervals or quantile forecasts |
| R8 | Single frozen model family | No ensemble diversity or online adaptation |
| R9 | Public deployment lags local data | Static demo must be rebuilt; live scheduling not provisioned |
| R10 | Alert delivery disabled by default | Real notification requires credentials and an operator |

Additional known gaps: the alert engine is a prototype with no paging integration; the
UI has not been accessibility-audited; the system runs as a single instance; and there
is no formal SLA or on-call rotation.

---

## 9. Reproducibility and governance

### 9.1 Freeze

- `reports/FREEZE.md` and `reports/freeze_manifest.json` lock the headline numbers,
  environment, and 26 evidence files.
- `scripts/freeze_results.py --check` reports **13/13 checks, 0 changed files**.
- The manifest records an explicit revision history. Revision 2 (2026-11-06) is a
  documentation-only refresh of two narrative reports; no dataset, model, prediction,
  or headline number changed, and the test remained closed.

### 9.2 Verification commands

```powershell
# Python suite (84 tests: contract, splits, sequences, features, ingestion,
# quality gate, alert engine, API contract, pipeline smoke)
.venv\Scripts\python.exe tests\run_tests.py

# Evidence lock
.venv\Scripts\python.exe scripts\freeze_results.py --check

# Frontend
cd system/web
npm run typecheck
npm run test
npm run build

# API (local)
.venv\Scripts\python.exe -m uvicorn system.api.app:app --port 8000
```

### 9.3 Governance

- The frozen test is closed. New analyses use development cells or live inference.
- D16 and its thresholds are versioned under `models/final_v2/`.
- Alert approval roles, legal disclaimers, and duty-officer procedures are documented
  in the governance files referenced by `DEPLOYMENT.md`.
- The research-prototype label is present in the UI, API text, and alert email body.

---

## 10. Conclusions

The project delivers a leakage-aware, probabilistic, station-level 24/48/72-hour flood
outlook for Bangladesh. D16 achieves a validation horizon-mean PR-AUC of **0.957427**
and a single frozen test horizon-mean of **0.770758**, beating persistence on all 21
test points. The decision layer converts those probabilities into cost-derived
thresholds and risk bands with a ~90% observed flood rate in the top band. The system
around the model — incremental ingestion, quality gates, API, map/UI, alerts,
monitoring, audit, and a freeze manifest — is implemented and tested.

The most transferable finding is not the score. It is that a silent data-staleness
failure degraded 38% of the feature vector for three weeks while every health check
stayed green. The controls added after that incident (dynamic refresh, mandatory
quality gate, source-freshness telemetry, approval-gated alerts, and regression tests)
are what make the prototype safe to evaluate. Full operational deployment remains
future work and depends on funding, managed scheduling, credentials, and formal
validation with FFWC.

---

## Appendix A — Headline numbers (from `reports/FREEZE.md`)

| Quantity | Value |
| --- | ---: |
| D16 `val_2022` 24/48/72 h | 0.970789 / 0.956485 / 0.945007 |
| D16 `val_2022` horizon mean | **0.957427** |
| D16 frozen test 24/48/72 h | 0.836227 / 0.768959 / 0.707087 |
| D16 frozen test horizon mean | **0.770758** |
| D7 frozen test horizon mean | 0.667549 |
| Best classical / best deep, `val_2022` | 0.954634 / 0.933991 |
| Experimental dataset | 200,665 × 67, SHA `a67485…` |
| Test reads | 1 (MI-6); `test_rows_read = 0` afterwards |

## Appendix B — Key artefacts

| Artefact | Purpose |
| --- | --- |
| `reports/FREEZE.md` | Frozen headline numbers and hashes |
| `reports/phase43_final_evaluation.md` | Five-dimension validity assessment |
| `reports/mi6_summary.json` | Frozen test metrics |
| `reports/phase27_summary.json` | 24-model dev comparison |
| `reports/phase28_calibration.md` | Calibration evidence |
| `reports/phase29_thresholds.md` | Cost-derived thresholds and bands |
| `reports/phase30_errors.md` | Error patterns |
| `reports/phase31_robustness.md` | Temporal/spatial robustness |
| `reports/phase34_xai.md` | SHAP and DL attribution |
| `reports/event_level/summary.json` | Event-level development metrics |
| `reports/p2p3_rescore/summary.json` | P2/P3 correction gates |
| `models/final_v2/final_config_v2.json` | Frozen model configuration |
| `tests/run_tests.py` | 84-test verification suite |

## Appendix C — Data and code availability

The experimental table, model artefact, and reports are versioned in the repository.
The feature store is partitioned by date and can be published to a HF Dataset. The
static snapshot is regenerated with:

```powershell
.venv\Scripts\python.exe system/deploy/build_static_snapshot.py --incremental --trend-days 90
```

The snapshot health file now reports both the number of stations forecast on the latest
date and the wider trend-window station union, so an operator can distinguish "fewer
stations reported today" from "the system lost coverage".
