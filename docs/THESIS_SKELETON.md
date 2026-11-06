# Aabhas Flood Predictor — Thesis / Research Report Skeleton (Stage 73)

**Version:** 1.0  
**Date:** 2026-11-04  
**Status:** SUPERSEDED — the written report is [`reports/phase45_thesis.md`](../reports/phase45_thesis.md) (2026-11-06). This skeleton is retained as the outline.

---

## Title
**Leakage-Aware Flood Forecasting for Bangladesh: From Research Prototype to Operational Decision Support**

---

## Abstract
*[~300 words]*
- Problem: Flood forecasting in Bangladesh, need for decision support
- Approach: Leakage-controlled ML pipeline, D16 frozen model
- System: End-to-end automation (ingestion → features → inference → alerts)
- Key finding: Null feature incident (Oct–Nov 2026) — production lesson
- Status: Research prototype, not official warning

---

## 1. Introduction
### 1.1 Background
- Bangladesh flood context (GBM basin, FFWC, historical impacts)
- Current forecasting: deterministic, limited lead time
- ML opportunity: probabilistic, multi-source, longer lead

### 1.2 Objectives
- Build leakage-free ML pipeline for flood probability
- Achieve operational automation (ingestion to alert)
- Document production lessons for research-to-operations transition

### 1.3 Contributions
1. Leakage controls (spatial/temporal splits, label contract)
2. Frozen D16 model (LightGBM/XGBoost ensemble)
3. End-to-end automated pipeline with quality gates
4. Alert engine with approval workflow
5. Production incident analysis (null features)

### 1.4 Scope and Limitations
- Research prototype only
- 24h/48h/72h horizons
- 127 stations, 2017–2026 data

---

## 2. Data and Definitions
### 2.1 Station Network
- FFWC 153 stations, 127 with sufficient history
- Danger levels, basin/rivers, spatial metadata

### 2.2 Label Definition
- **Issue date** t: forecast issued morning of t
- **Horizons:** 24h (t, t+1], 48h (t, t+2], 72h (t, t+3]
- **Label:** Any FFWC reading ≥ danger level in horizon window
- **Class imbalance:** ~5% positive rate

### 2.3 Data Sources
| Source | Variables | Frequency | Lag | Coverage |
|--------|-----------|-----------|-----|----------|
| FFWC | Water levels | 3-hourly | ~6h | 2017– |
| NASA POWER | Rainfall, soil wetness | Daily | 2–3d | 2013– |
| Open-Meteo ERA5-Land | Precip, temp, soil moisture | Daily | 5–7d | 2013– |
| GloFAS | River discharge (anchors) | Daily | 1–2d | 2013– |
| Static | Elevation, slope, distances | Once | — | All stations |

### 2.4 Feature Groups (FULL_V2 = 110 features)
| Group | Prefix | Count | Critical |
|-------|--------|-------|----------|
| Weather | tmax, tmin | 2 | No |
| Rainfall | rain_* | 18 | No |
| Geography | elev, slope, dist_* | 7 | No |
| Hydrology | wl_* | 24 | No |
| Network | nb3, nb5, up_* | 12 | No |
| Soil | ps_* | 19 | **YES** |
| GloFAS | gf_* | 18 | **YES** |
| Interactions | ix_* | 5 | **YES** |
| Temporal | doy, month, etc. | 5 | No |

---

## 3. Methodology: Leakage Controls
### 3.1 Spatial Splits
- Basin-stratified: train/val/test by basin
- Upstream stations in train, downstream in test
- No spatial leakage across splits

### 3.2 Temporal Splits
- Year-based: train ≤ 2020, val 2021–2022, test 2023–2025
- No future data in training features (lags only)

### 3.3 Label Contract
- Strict: label at t uses readings in (t, t+h]
- No look-ahead: features at t use only data ≤ t
- Verification: automated tests in `tests/`

### 3.4 Feature Computation
- Full-history computation, then slice to output dates
- Lag/rolling features computed on complete series
- No per-partition computation (avoids edge nulls)

---

## 4. Baselines
### 4.1 Persistence
- Tomorrow = today's water level
### 4.2 Climatology
- Historical exceedance rate per station/month
### 4.3 FFWC Deterministic
- Official forecast where available

---

## 5. Classical ML (Phases 12–20)
### 5.1 Feature Selection
- MI screening, permutation importance
- Stability selection across splits

### 5.2 Models
- Logistic Regression (calibrated)
- Random Forest
- Gradient Boosting (LightGBM)
- XGBoost

### 5.3 Results
| Model | 24h AUC | 48h AUC | 72h AUC |
|-------|---------|---------|---------|
| Logistic | 0.78 | 0.74 | 0.70 |
| RF | 0.85 | 0.81 | 0.77 |
| LightGBM | **0.89** | **0.86** | **0.82** |
| XGBoost | 0.88 | 0.85 | **0.83** |

---

## 6. Deep Learning (Phases 21–28)
### 6.1 Sequence Models
- LSTM, GRU, Transformer
- Input: 30-day sequences per station

### 6.2 Fusion Architectures
- Static + sequential branches
- Attention gates

### 6.3 Results
- DL models did not significantly outperform LightGBM/XGBoost
- Added complexity not justified for operational use
- Frozen D16 uses classical ML only

---

## 7. Fair Comparison and Ablation (Phases 29–33)
### 7.1 Evaluation Protocol
- Frozen splits, identical features, identical labels
- Metrics: AUC, Brier, Reliability (ECE), Cost-loss

### 7.2 Ablation Studies
| Experiment | Change | Δ AUC (24h) |
|------------|--------|-------------|
| E0 (full) | — | 0.89 |
| E1 | No soil | -0.03 |
| E2 | No GloFAS | -0.04 |
| E3 | No interactions | -0.02 |
| E4 | No upstream | -0.01 |
| E5 | Linear only | -0.07 |

### 7.3 Threshold Selection (Phase 29)
- Cost-loss framework: C/L ratio from stakeholder input
- Per-horizon thresholds, risk bands

---

## 8. Model Selection and Calibration (Phases 34–38)
### 8.1 Final Model (D16)
- 24h/48h: LightGBM (calibrated)
- 72h: XGBoost (calibrated)
- Frozen: weights, features, thresholds, code version

### 8.2 Calibration
- Isotonic regression per horizon
- Reliability diagrams (ECE < 0.05)

### 8.3 Explainability (Phase 34)
- SHAP values per prediction
- Global feature importance

---

## 9. System Implementation (Stages 62–72)
### 9.1 Architecture
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Sources    │──▶│  Pipeline   │──▶│  Feature    │
│  (5 APIs)   │   │ (daily)     │   │  Store      │
└─────────────┘   └─────────────┘   └──────┬──────┘
                                           │
┌─────────────┐   ┌─────────────┐   ┌──────▼──────┐
│  Alerts     │◀──│  API/Infer  │◀──│  Model D16  │
│  (engine)   │   │  (FastAPI)  │   │  (frozen)   │
└─────────────┘   └─────────────┘   └─────────────┘
```

### 9.2 Pipeline Automation
- Step 0: Source refresh (GloFAS, NASA POWER, ERA5-Land)
- Step 1: FFWC ingestion + QC
- Step 2: Rainfall merge (NASA + Open-Meteo)
- Step 3: Feature build (110 FULL_V2)
- Step 4: Quality gate (null share ≤ 5% critical groups)
- Step 5: Snapshot publish (static site + HF dataset)

### 9.3 Quality Gates
- Source freshness SLAs (FFWC ≤24h, GloFAS ≤48h, Soil ≤72h, ERA5 ≤7d)
- Feature null share per group
- Build fails on degradation (--allow-degraded opt-in only)

### 9.4 Alert Engine
- Band → level mapping (watch/warning/high)
- 24h cooldown, escalation bypass
- Approval workflow (draft → approve → dispatch)
- Channels: email, webhook, dashboard
- Immutable audit log (hash-chained JSONL)

### 9.5 Deployment
- API: Cloud Run (asia-south1), HTTPS, IAM
- Pipeline: Cloud Run Job + Cloud Scheduler
- Static site: HF Static Space (fallback)
- Monitoring: /health, /monitoring, Prometheus endpoint

---

## 10. Production Incident: Null Features (Oct–Nov 2026)
### 10.1 Detection
- Routine health check showed `gf_*`/`ps_*` 100% null
- Root cause: Hard-coded END dates in fetchers (2026-10-12)
- Duration: ~3 weeks (2026-10-13 to 2026-11-03)

### 10.2 Impact
- 37/110 features null (33.6%)
- 5 interaction features derived from null → 42/110 compromised (38%)
- Model running degraded, predictions unreliable

### 10.3 Resolution
1. Rewrote 3 fetchers as incremental refreshers
2. Added `refresh_sources.py` entrypoint with QC
3. Enforced quality gate (no --allow-degraded default)
4. Backfilled 2026-10-13 → 2026-11-03
5. Rebuilt features, snapshot, re-ran evaluation

### 10.4 Lessons Learned
- **Never hard-code end dates** — use dynamic defaults
- **Quality gate must be mandatory** — opt-in degradation only
- **Source freshness monitoring** — /health now shows per-source status
- **Automation must be always-on** — single laptop is SPOF
- **Incident response runbook** — document for future

---

## 11. Limitations and Future Work
### 11.1 Known Limitations (R1–R8 from workflow)
- R1: Research prototype, not official warning
- R2: Training data ends 2025 (climate non-stationarity)
- R3: 72h horizon significantly weaker
- R4: No sub-daily updates
- R5: 26 stations excluded (insufficient history)
- R6: Null feature incident not yet re-evaluated
- R7: No uncertainty quantification beyond calibration
- R8: Single-model (no ensemble diversity)

### 11.2 Future Work
- Retrain with 2026 data (including fixed features)
- Ensemble methods (quantile regression, conformal prediction)
- Sub-daily inference (3-hourly FFWC updates)
- Transfer learning to ungauged locations
- Official validation study with FFWC

---

## 12. Conclusions
- Leakage-controlled ML pipeline achieves ROC-AUC 0.82–0.89
- End-to-end automation with quality gates operational
- Production incident revealed critical gaps in source management
- Framework ready for government evaluation as decision support
- **Not an official warning system** — requires formal validation

---

## References
*[Bibliography from references.bib]*

---

## Appendices
### A. Feature Dictionary (FULL_V2)
### B. Hyperparameters (D16 Frozen)
### C. Threshold Derivation Details
### D. Alert Configuration
### E. Deployment Runbooks
### F. Governance Documents (Threshold Sign-Off, Duty Officer, Legal Disclaimer)

---

## Revision History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-11-04 | | Skeleton created |
| 1.0 | | | First complete draft |

---

*This skeleton follows the structure outlined in `final workflow.md` Phase 45 / Stage 73.*