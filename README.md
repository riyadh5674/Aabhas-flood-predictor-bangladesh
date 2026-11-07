# Aabhas — Flood Outlook for Bangladesh

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/status-research--complete-brightgreen.svg)](reports/phase43_final_evaluation.md)
[![Model](https://img.shields.io/badge/model-D16%20frozen-orange.svg)](models/final_v2/)
[![Deploy](https://img.shields.io/badge/deploy-Render-purple.svg)](https://aabhas-flood-predictor.onrender.com)

**Station-level 24/48/72-hour probabilistic flood outlook for Bangladesh**, built on a leakage-aware ML pipeline with rigorous temporal/spatial validation and a frozen production model (D16).

**Live Demo:**

**Dashboard:** https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/  
**API Health:** https://aabhas-flood-predictor-xckw.onrender.com/health  
**API Docs (Swagger):** https://aabhas-flood-predictor-xckw.onrender.com/docs  
**Monitoring:** https://aabhas-flood-predictor-xckw.onrender.com/monitoring  
**Static Snapshot (fallback):** https://riyadhuggy-aabhas-flood-outlook.static.hf.space  
**Deployment runbook:** [DEPLOYMENT.md](DEPLOYMENT.md)

> ⚠️ Aabhas is a research prototype, **not an official flood warning service**. The live app serves model output from the latest published feature date; the in-app freshness banner and `/health` show exactly how old that is. Always follow FFWC advisories.

### Current verified state (2026-11-06)

| Check | Result |
|---|---|
| Frozen model | D16 (`models/final_v2/`), test split closed — `test_rows_read = 0` after MI-6 |
| Feature store | Current through **2026-11-05**; latest partition 112 stations |
| Source freshness | GloFAS 2026-11-05 · POWER soil 2026-11-02 · ERA5-Land 2026-11-03 — all within SLA; shown in `/health` and the freshness banner |
| Quality gate | `quality/2026-11-05.json`: critical null share 0.0, `passed: true` |
| Python tests | **84 / 84** pass (`tests/run_tests.py`) |
| Frontend | 21 / 21 Vitest tests; `typecheck` and `build` pass |
| Frozen evidence | `freeze_results.py --check` → **13/13**, 0 changed files |
| Alert engine | Implemented with approval + audit; external delivery disabled by config |
| Thesis | `reports/phase45_thesis.md` (Phase 45 / stage 73) |
| Deployment | Local/system complete; public live redeploy still pending funding/credentials |

**Scope note:** everything in `final workflow.md` is complete at the research/prototype level. The only remaining items are operational — paid always-on hosting, formal threshold sign-off plus SMTP/webhook credentials for alert delivery, and republishing the public static snapshot. See [FLAWS.md](FLAWS.md) open items O-01/O-02.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | This file — project overview & quick start |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Architecture, runbook, CI/CD, daily refresh, troubleshooting |
| [FLAWS.md](FLAWS.md) | Current hardening status, resolved issues, remaining risks |
| [REPRODUCE.md](REPRODUCE.md) | Paper reproduction levels A–E, freeze verification |
| [final workflow.md](final workflow.md) | Complete 45-phase research workflow (all 73 stages DONE) |
| [Phase 43 — Final Evaluation](reports/phase43_final_evaluation.md) | Scientific validity assessment (data, model, prediction, explainability, system) |
| [Phase 45 — Thesis](reports/phase45_thesis.md) | Research report / thesis draft (stage 73) |
| [FREEZE.md](reports/FREEZE.md) | Frozen evidence lock (13/13 checks pass) |
| [Duty Officer Process](docs/DUTY_OFFICER_PROCESS.md) | Alert review, approval and escalation runbook |

---

## 🏗 Repository Layout

```
flood-predictor/
├── src/                      # Research library code (ingest, features, models, evaluate, api)
├── notebooks/                # Executed research notebooks (Phase 0–34, MI-v2)
├── reports/                  # One Markdown per phase + metric CSVs + frozen artefacts
├── figures/                  # All paper figures (PNG)
├── scripts/                  # Phase builders, auditors, trainers, run logs
├── models/
│   └── final_v2/             # Frozen D16 production model (selected_model_v2.pkl)
├── system/                   # Deployable system
│   ├── api/                  # FastAPI inference server
│   ├── web/                  # React + Vite frontend
│   ├── config/               # model_config.json, alert_config.json, ingestion_config.json
│   ├── deploy/               # Static snapshot builder + HF Space publisher
│   ├── ops/                  # Daily pipeline, refresh/publish, scheduler
│   ├── ingestion/            # FFWC, Open-Meteo, GloFAS, ERA5, ESA WorldCover ingest + source refresh
│   ├── features/             # Feature engineering pipeline + quality gate
│   ├── alerts/               # Early-warning engine, approval lifecycle, audit
│   └── map/                  # Choropleth risk map generator
├── tests/                    # 84 unittest tests (contracts, splits, features, pipeline, alerts, API)
├── data/                     # Raw & processed (gitignored — large/reproducible)
├── docs/                     # Operational and governance documents
├── requirements.txt          # Research environment (Python 3.13)
├── requirements-api.txt      # API deployment environment
├── DEPLOYMENT.md             # Full deployment guide
├── REPRODUCE.md              # Paper reproduction guide
├── FREEZE.md                 # Frozen evidence manifest
└── final workflow.md         # Complete 45-phase workflow
```

---

## 🚀 Quick Start

### Web app (live API mode — production behaviour)
```powershell
# Terminal 1: Start API (reads local feature partitions or HF_FEATURE_DATASET)
.\.venv\Scripts\python.exe -m system.api.run_api --port 8000

# Terminal 2: Start frontend
cd system\web
npm ci
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```

### Web app (static snapshot — no API needed)
```powershell
cd system\web
npm ci
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run dev
```

### Verify the checkout

```powershell
# Python suite — 84 tests
.\.venv\Scripts\python.exe tests\run_tests.py

# Frozen evidence — 13/13 checks, 0 changed files
.\.venv\Scripts\python.exe scripts\freeze_results.py --check

# Frontend
cd system\web
npm run typecheck
npm run test
npm run build
```

### Research environment
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# PyTorch CPU: .venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

---

## 🧪 Reproduce Paper Results

| Level | Command | Time | What it proves |
|---|---|---:|---|
| **A. Verify freeze** | `.venv\Scripts\python.exe scripts\freeze_results.py --check` | ~1 min | All 13/13 checks pass, 0 changed files |
| **A2. System tests** | `.venv\Scripts\python.exe tests\run_tests.py` | ~15 s | 84 Python tests (contracts, splits, features, pipeline, alerts, API) |
| **B. P2/P3 correction** | `.venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py` | ~35 min | Per-cell refits, corrected metrics |
| **C. Event analysis** | `.venv\Scripts\python.exe -u scripts\mi11_event_eval.py` | ~10 min | Event recall, lead time, FA rates |
| **D. Full model programme** | See [REPRODUCE.md](REPRODUCE.md) §5 | hours | Phases 16–34, MI-1…MI-11 |
| **E. Rebuild from raw** | External data sources | days | Not in repo — see [REPRODUCE.md](REPRODUCE.md) |

---

## 🏭 Deploying

### Production (Render — single container, live API)
`render.yaml` + `Dockerfile` deploy the FastAPI service and compiled SPA
together. See [DEPLOYMENT.md](DEPLOYMENT.md#-render-deployment) for the
blueprint, environment variables, and verification commands.

### Data refresh (automatic)

The daily pipeline refreshes lagging external sources first, then builds features
through a quality gate:

```powershell
# Source refresh only (GloFAS / POWER soil / ERA5-Land) + freshness report
.\.venv\Scripts\python.exe system\ingestion\refresh_sources.py --end 2026-11-05

# Full local/manual run: source refresh → ingestion → features → quality gate → publish
.\.venv\Scripts\python.exe system\ops\daily_pipeline.py --date 2026-11-05
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --catchup --push-raw

# Windows Task Scheduler (daily 06:00)
powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1
```

Freshness telemetry is written to `system/data/processed/source_status/<date>.json`
and the feature gate to `system/data/processed/quality/<date>.json`. A degraded
critical source fails the date unless `--allow-degraded` is passed explicitly.
`.github/workflows/refresh.yml` provides the same flow on a daily GitHub
Actions schedule; see the runbook for required secrets.

### Static snapshot (optional public demo)
```powershell
.\.venv\Scripts\python.exe system\deploy\build_static_snapshot.py
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run build
cd ..\..
.\.venv\Scripts\python.exe scripts\deploy_static_space.py
```

### Dynamic API on another host
See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose, Render, and other
Docker-capable hosts. Set `VITE_API_BASE` and build **without**
`VITE_AABHAS_STATIC_SNAPSHOT`.

---

## 🤖 Model & Data

| Aspect | Details |
|---|---|
| **Model** | D16: per-horizon LightGBM (24h/48h) + XGBoost (72h) on FULL_V2 (110 features) |
| **Frozen at** | `models/final_v2/selected_model_v2.pkl` + `final_config_v2.json` |
| **Thresholds** | Cost-derived (C_FN:C_FP = 10:1), period-specific, in `system/config/model_config.json` |
| **Risk bands** | 4 bands per horizon (low/moderate/high/very_high) — top band 90% observed flood rate |
| **Inputs** | FFWC water levels, Open-Meteo ERA5-Land (rain + soil moisture), GloFAS discharge, ESA WorldCover, NASA POWER soil |
| **Test policy** | Frozen test split (2023+) — read **once** in MI-6 (`test_rows_read = 0` thereafter) |
| **Val performance** | PR-AUC 0.9574 (horizon-mean), beats D7 persistence on 21/21 dev points |
| **Test performance** | PR-AUC 0.7708 (horizon-mean) vs D7 0.6675; missed floods −58% |
| **Feature store** | Current through `2026-11-05`; latest partition 112 stations, 110 FULL_V2 features |
| **Quality gate** | Critical-group null share 0.0 on the latest partition; build fails on a stale critical source |
| **Alerts** | Band→level engine with 24 h cooldown, escalation bypass, draft→approval→audit; external delivery disabled by config |
| **Tests** | 84 Python unittest tests + 21 frontend Vitest tests; freeze check 13/13 |

---

## 🔬 Research Highlights

- **Leakage-free** temporal/spatial splits with 7-day embargo, 5 station-disjoint folds
- **Fair ML vs DL comparison** (Phase 27, MI-7): gradient boosting leads DL by 0.02 PR-AUC on equal features/budget
- **Hydrology-first explanations**: TreeSHAP attributes 64% to water-level features, no banned features in top-20
- **Robustness**: beats persistence on 63/63 cell×horizon points; seasonal gap ≤0.044 on seen stations
- **Calibration**: OOF ECE 0.0031; all skill models recalibrated to tolerance

---

## 🧾 Reproduction & Freeze

```powershell
# Verify frozen artefacts
.venv\Scripts\python.exe scripts\freeze_results.py --check
# → checks 13/13; changed files 0
```

All frozen artefacts SHA-256 tracked in `reports/freeze_manifest.json` and `reports/FREEZE.md`.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- FFWC Bangladesh for station water-level data
- Open-Meteo for ERA5-Land & GloFAS APIs
- NASA POWER for soil moisture reanalysis
- ESA WorldCover for land cover data
- Hugging Face for free Static Space hosting

---

**Citation:** If you use this work, please cite the associated paper (in preparation).

**Contact:** For questions, open an issue or check the [deployment troubleshooting guide](DEPLOYMENT.md#-troubleshooting).