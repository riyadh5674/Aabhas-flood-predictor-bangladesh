<div align="center">

<img src="aabhas-logo.png" alt="Aabhas" width="320" />

# Aabhas — Know your river. Plan ahead.

**Station-level, probabilistic flood outlook for Bangladesh — 24, 48 and 72 hours ahead.**

[![Live demo](https://img.shields.io/badge/static_demo-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/)
[![Live API](https://img.shields.io/badge/live_API-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://aabhas-flood-predictor-xckw.onrender.com/health)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React_19-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)](system/web)
[![Tests](https://img.shields.io/badge/tests-86_python_%2B_21_frontend-success?style=for-the-badge)](tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

[**Live app**](https://aabhas-flood-predictor-xckw.onrender.com) ·
[**Static demo**](https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/) ·
[**Investor brief**](docs/PITCH.md) ·
[**5-minute demo script**](docs/DEMO.md) ·
[**Architecture**](docs/ARCHITECTURE.md) ·
[**Reproduce the results**](REPRODUCE.md)

</div>

---

## The problem

Bangladesh sits at the confluence of the Ganges, Brahmaputra and Meghna. Every monsoon,
millions of people make life-or-death decisions — evacuate or stay, move livestock or
protect the harvest, open a shelter or wait — with flood warnings that are often
**district-level, deterministic, and published too late to act on**.

The operational question a family, a union council, an NGO or a logistics team actually
asks is:

> *"What is the chance that **my station** crosses its danger level in the next
> **24 / 48 / 72 hours** — and how confident is that number?"*

Aabhas answers that question directly, at **station level**, with a probability, a risk
band, plain-language guidance and an auditable trail.

## What Aabhas does

| | |
|---|---|
| 🗺️ **Station-level risk map** | 153 FFWC stations mapped across Bangladesh; risk bands for the next 24/48/72 h. |
| 📈 **Probabilistic outlook** | A calibrated probability per station and horizon — not a single binary "flood / no flood". |
| 🚦 **Actionable bands** | Low / Moderate / High / Very high, derived from an explicit **10:1 cost of a missed flood vs. a false alarm**. |
| ⏱️ **Lead time** | 24, 48 and 72 hours, with horizon-specific thresholds and freshness checks. |
| 🧭 **Explanations** | TreeSHAP drivers per prediction, so an officer can see *why* the model raised a flag. |
| 🔔 **Approval-gated alerts** | Draft → human approval → audit log. No automatic public warning by design. |
| 🌐 **Bilingual, offline-ready** | English + Bengali UI, installable PWA, offline shell for low-connectivity areas. |
| 🧪 **Evidence-first** | Frozen test split, leakage audits, calibration, robustness and an 84-test contract suite. |

<div align="center">
<img src="docs/screenshots/overview.png" alt="Aabhas overview — national flood outlook" width="100%" />
<sub><b>Overview</b> — national situation, horizon switch (24/48/72 h), freshness banner and station-level risk bands.</sub>
</div>

<br/>

| Map | Station detail |
|---|---|
| <img src="docs/screenshots/map.png" alt="Risk map" width="100%" /> | <img src="docs/screenshots/station-detail.png" alt="Station detail" width="100%" /> |

| Model & evidence | Alerts |
|---|---|
| <img src="docs/screenshots/model.png" alt="Model page" width="100%" /> | <img src="docs/screenshots/alerts.png" alt="Alerts page" width="100%" /> |

> **Research prototype, not an official warning service.** FFWC advisories and government
> instructions always take precedence. The app labels its own data freshness and never
> presents a stale or failed request as a live forecast.

---

## Why this is credible

Aabhas was built as a **leakage-aware research programme first**, then as a product. The
model is frozen, the test split was opened exactly once, and every headline number is
reproducible from committed artefacts.

| Evidence | Result |
|---|---|
| Validation PR-AUC (`val_2022`, horizon mean) | **0.9574** (0.9708 / 0.9565 / 0.9450) |
| Frozen test PR-AUC (2023–2026) | **0.7708** vs **0.6675** for persistence |
| Beats the operational persistence baseline | **21 / 21** test cell × horizon points |
| Missed floods vs. persistence | **−58 %** |
| Out-of-fold calibration error (ECE) | **0.0031** |
| Leakage control | 7-day embargo, station-disjoint folds, banned-feature assertions |
| Robustness | Beats persistence on **63 / 63** spatial cell × horizon points |
| Automated checks | **86 Python + 21 frontend tests**, frozen-evidence check **13 / 13** |
| Data coverage | **2,617** feature dates (2017 → 2026), **112** active stations in the latest partition |

The full scientific case — validity, calibration, ablations, error analysis and the
incident that motivated the quality gate — is in
[`reports/phase43_final_evaluation.md`](reports/phase43_final_evaluation.md) and
[`reports/phase45_thesis.md`](reports/phase45_thesis.md).

### What makes it different

1. **Probabilistic, station-level, multi-horizon** — useful for pre-positioning, not just alerting.
2. **Cost-aware thresholds** — the false-alarm/miss trade-off is explicit policy, not a hidden default.
3. **Leakage discipline** — most flood-ML demos report inflated scores from label-window leakage. Aabhas documents and tests against it.
4. **Operational honesty** — source-freshness telemetry, a hard quality gate, a frozen model and a human-in-the-loop alert workflow.
5. **Deployable on free infrastructure** — static snapshot for the public, FastAPI for a live deployment, one Docker image for a paid host.

---

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        A1[FFWC water levels]
        A2[ERA5-Land rain + soil]
        A3[GloFAS discharge]
        A4[NASA POWER soil]
        A5[ESA WorldCover]
    end

    subgraph Pipeline
        B1[Incremental ingestion] --> B2[Feature engineering<br/>110 FULL_V2 features]
        B2 --> B3[Freshness + quality gate]
        B3 --> B4[Partitioned feature store]
    end

    subgraph Model
        C1[D16 frozen model<br/>LightGBM 24/48 h · XGBoost 72 h]
        C2[Cost-derived thresholds<br/>+ 4 risk bands]
    end

    subgraph Delivery
        D1[FastAPI inference]
        D2[React PWA + risk map]
        D3[Static snapshot<br/>GitHub Pages / HF Space]
        D4[Approval-gated alerts<br/>hash-chained audit]
    end

    A1 & A2 & A3 & A4 & A5 --> B1
    B4 --> C1 --> C2
    C2 --> D1 --> D2
    C2 --> D3
    D1 --> D4
```

Full component map, data contracts and deployment topology: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Try it in 60 seconds

### Option A — public demo (no install)

**Static demo:** https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/

**Live API + app:** https://aabhas-flood-predictor-xckw.onrender.com

The GitHub Pages demo ships a precomputed snapshot and runs entirely in the browser.
The Render deployment runs the real FastAPI service and the frozen D16 model against a
bundled 120-day feature-store slice, so `/health`, `/predict/all` and `/predict/trend`
return live model output without any external credentials. Render's free tier sleeps
after inactivity; the first request can take about a minute to wake the service.

Both are research prototypes. FFWC advisories remain the operational authority.

### Option B — run the full stack locally

```powershell
# 1. Python environment (once)
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Frontend dependencies (once)
cd system\web; npm ci; cd ..\..

# 3. Start API + web app together
cd system\web; npm run dev
```

Open **http://localhost:5173**. The launcher starts the FastAPI service on port 8000 if
it is not already running. For the static snapshot mode:

```powershell
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run dev
```

### Option C — Docker Compose

```bash
docker compose up --build
# web: http://localhost:5173   api: http://localhost:8000/health
```

The full 5-minute walkthrough with talking points is in [`docs/DEMO.md`](docs/DEMO.md).

---

## Repository layout

```
├── src/                     Research library (ingestion, features, models, evaluation)
├── notebooks/               Executed research notebooks (Phases 0–34, MI-v2)
├── reports/                 One report per phase, metric CSVs, frozen evidence
├── figures/                 Paper and analysis figures
├── models/final_v2/         Frozen production model (D16) + config
├── scripts/                 Trainers, auditors, freeze checker, deployment scripts
├── system/
│   ├── api/                 FastAPI inference server
│   ├── web/                 React 19 + Vite + TypeScript PWA
│   ├── config/              Model, alert and ingestion configuration
│   ├── ingestion/           FFWC, Open-Meteo, GloFAS, ERA5, POWER, WorldCover
│   ├── features/            Feature pipeline + mandatory quality gate
│   ├── alerts/              Early-warning engine, approval lifecycle, audit log
│   ├── deploy/              Static snapshot builder + host publishers
│   ├── ops/                 Daily pipeline, refresh/publish, scheduler
│   └── map/                 Choropleth risk-map generator
├── tests/                   86 Python contract/regression tests
├── docs/                    Pitch, demo, architecture, ops and governance docs
└── DEPLOYMENT.md            Runbook, CI/CD, troubleshooting
```

---

## Verification

```powershell
# Python suite — 86 tests
.\.venv\Scripts\python.exe tests\run_tests.py

# Frozen evidence — 13/13 checks, 0 changed files
.\.venv\Scripts\python.exe scripts\freeze_results.py --check

# Frontend — 21 unit tests, typecheck and production build
cd system\web
npm run typecheck
npm run test
npm run build
```

Current verified state: **86/86 Python tests**, **21/21 frontend tests**, **freeze 13/13**,
quality gate green through **2026-11-05**, source freshness within SLA.

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/PITCH.md](docs/PITCH.md) | Investor / partner brief: problem, product, evidence, market, roadmap, ask |
| [docs/DEMO.md](docs/DEMO.md) | 5-minute demo script and stakeholder talking points |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data contracts, deployment topology, security controls |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Full runbook: Render, Docker, CI/CD, daily refresh, troubleshooting |
| [REPRODUCE.md](REPRODUCE.md) | Paper reproduction levels A–E and freeze verification |
| [FLAWS.md](FLAWS.md) | Hardening status, resolved issues, remaining risks |
| [final workflow.md](final workflow.md) | Complete 45-phase research workflow (73 stages) |
| [reports/phase43_final_evaluation.md](reports/phase43_final_evaluation.md) | Scientific validity assessment |
| [reports/phase45_thesis.md](reports/phase45_thesis.md) | Thesis-style research report |
| [docs/DUTY_OFFICER_PROCESS.md](docs/DUTY_OFFICER_PROCESS.md) | Alert review, approval and escalation runbook |
| [docs/THRESHOLD_SIGN_OFF.md](docs/THRESHOLD_SIGN_OFF.md) | Threshold governance and sign-off |

---

## Roadmap

| Phase | Status | What it unlocks |
|---|---|---|
| Research + prototype | ✅ Complete | Frozen model, full evaluation, working product, tests |
| Public static demo | ✅ Live | Shareable link for stakeholders, zero hosting cost |
| Pilot with an agency / NGO | 🔜 Next | Signed MOU, named duty officer, threshold sign-off, real delivery channel |
| Always-on managed deployment | 🔜 Next | Cloud Run / Render + managed scheduler; needs modest funding |
| Multi-country transfer | 🔬 Planned | Domain adaptation to Brahmaputra basin and other delta systems |
| Impact evaluation | 🔬 Planned | Prospective evaluation of decisions and lead time saved |

The honest boundary: the **model and system are complete**; what remains is
**operational** — paid always-on hosting, formal threshold sign-off, and an
institutional partner to receive approved alerts. See [FLAWS.md](FLAWS.md) open items
O-01/O-02 and [docs/PITCH.md](docs/PITCH.md) for the funding ask.

---

## Data sources & acknowledgements

- **FFWC Bangladesh** — station water levels and danger levels
- **Open-Meteo** — ERA5-Land rainfall, temperature and soil moisture
- **Copernicus GloFAS** — transboundary river discharge
- **NASA POWER** — soil moisture reanalysis
- **ESA WorldCover** — land-cover features
- **geoBoundaries** — administrative boundaries

Thank you to the open-data teams who make this work possible.

---

## Citation

```bibtex
@software{aabhas2026,
  title  = {Aabhas: Leakage-Aware Probabilistic Flood Forecasting for Bangladesh},
  year   = {2026},
  url    = {https://github.com/riyadh5674/Aabhas-flood-predictor-bangladesh},
  note   = {Research prototype; not an official flood-warning service}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

## Contributing & security

Contributions are welcome — read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Please report vulnerabilities privately per
[SECURITY.md](SECURITY.md), never in a public issue.

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Aabhas** · *Know your river. Plan ahead.*

Built in Bangladesh, for Bangladesh — and for every delta facing the same question.

</div>
