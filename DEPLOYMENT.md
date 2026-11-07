# Aabhas Flood Predictor — Deployment Guide

**Updated:** 2026-11-07

## 🎯 Deployed architecture

The production app is a single FastAPI container that serves both the JSON API
and the compiled React SPA. It runs on Render's free tier with a tracked,
120-day feature-store slice bundled into the image. The full feature store can
optionally be read from a private Hugging Face Dataset.

| Component | Host | URL |
|-----------|------|-----|
| Web app + API | Render (Docker, free) | <https://aabhas-flood-predictor-xckw.onrender.com> |
| Static snapshot (demo) | GitHub Pages | <https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/> |
| Bundled demo features | This repository | `system/data/demo/model_ready/` (120 dates) |
| Full feature store (optional) | HF Dataset (private) | `HF_FEATURE_DATASET` (e.g. `riyadhuggy/aabhas-flood-features`) |
| Raw/reference inputs | HF Dataset (private, optional) | `HF_RAW_DATASET` (used by scheduled refresh) |
| Model + API code | GitHub | `riyadh5674/Aabhas-flood-predictor-bangladesh` |

```
                    ┌───────────────────────────────┐
   browser ───────▶ │ Render: FastAPI + React SPA   │
                    │  /health /dates /predict/*    │
                    └──────────────┬────────────────┘
                                   │ per-date download
                                   ▼
                    ┌───────────────────────────────┐
                    │ HF Dataset                    │
                    │ data/train/date=*/data.parquet│
                    └──────────────┬────────────────┘
                                   ▲ upload
                    ┌──────────────┴────────────────┐
                    │ daily_pipeline.py              │
                    │ ingestion → features → HF      │
                    └───────────────────────────────┘
```

**Important safety semantics**

- A date is an **issue date**. `label_24h/48h/72h` means "any FFWC reading in
  `(t, t+h]` is at or above that station's danger level".
- The app is **not an official warning service**. FFWC advisories always take
  precedence. The disclaimer is rendered in the UI and documented here.
- The service worker never caches `/api` responses. Data freshness is shown by
  the in-app banner and `/health` (`latest_feature_date`,
  `latest_feature_age_days`).

**Free-tier notes**

- Render free web services sleep after inactivity; the first request can take
  30–60 s. `HEALTHCHECK` and `/health` are excluded from rate limiting.
- HF Datasets: free storage is sufficient for the partitioned feature store and
  the raw-input mirror.
- HF Docker/Gradio Spaces on `cpu-basic` require PRO, which is why the dynamic
  deployment lives on Render and the HF Space is static-only.

---

## 🚀 Local development

### Option A: Docker Compose

```powershell
Copy-Item .env.example .env
# Fill HF_FEATURE_DATASET / HF_TOKEN (optional for local feature store)
docker compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:5173
```

### Option B: Manual

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt

# Optional: use the remote feature store; otherwise local partitions are used.
$env:HF_FEATURE_DATASET="username/aabhas-flood-features"
$env:HF_TOKEN="hf_xxx"

.\.venv\Scripts\python.exe -m system.api.run_api --port 8000

# Frontend (second terminal)
cd system\web
npm ci
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```

Checks:

```powershell
cd system\web
npm run typecheck
npm run test
npm run build
```

---

## 🔄 Data refresh

Two layers must stay current:

1. **Raw inputs** — FFWC water levels, rainfall, Open-Meteo/ERA5, GloFAS, NASA
   POWER soil, station spatial features.
2. **Model-ready features** — one parquet partition per date under
   `system/data/processed/model_ready/date=YYYY-MM-DD/data.parquet`.

`system/ops/refresh_and_publish.py` orchestrates the daily update:

0. **source refresh** — `system/ingestion/refresh_sources.py` incrementally updates
   GloFAS, NASA POWER soil and ERA5-Land, checks each manifest against its lag SLA,
   and writes `system/data/processed/source_status/<date>.json`; a stale **critical**
   source fails the run unless `--allow-stale` / `--skip-source-refresh` is used;
1. ingestion → features (local partitioned parquet store), with
   `system/features/validate.py` enforcing the quality gate and writing
   `system/data/processed/quality/<date>.json`;
2. **incremental** snapshot build — only dates missing from the published
   snapshot are scored; trend files are trimmed to a 90-day window;
3. mirror the changed snapshot files into `system/web/dist`;
4. upload **only changed files** to the Hugging Face Static Space.

A degraded feature date is never published silently: `daily_pipeline.py` fails the
date unless `--allow-degraded` is passed explicitly.

The website is therefore dynamic without a server: each run appends the new
date and republishes a few hundred small JSON files. Feature-dataset and
raw-input uploads are opt-in (`--push-features`, `--push-raw`).

### Manual run

```powershell
# Backfill gaps in the last 30 days and publish (what the scheduled task runs)
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --catchup

# Yesterday only
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py

# Publish only, no pipeline
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --skip-pipeline

# Rebuild every date and upload every file (bootstrap / code changes)
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --static-full

# Verify the plan without writing anything
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --dry-run

# Source refresh only, with a freshness report (no feature build)
.\.venv\Scripts\python.exe system\ingestion\refresh_sources.py --end 2026-11-05
.\.venv\Scripts\python.exe system\ingestion\refresh_sources.py --only glofas,power_soil --dry-run

# One date, full pipeline (source refresh → features → quality gate)
.\.venv\Scripts\python.exe system\ops\daily_pipeline.py --date 2026-11-05

# Inspect the latest gate + source telemetry
Get-Content system\data\processed\quality\latest.json
Get-Content system\data\processed\source_status\2026-11-05.json
```

Publishing uses the token stored by `hf auth login` (or `HF_TOKEN` if set).
Without a token the local pipeline still runs and publishing is skipped with a
clear log line.

### Windows Task Scheduler (single-machine deployment)

```powershell
# Per-user task, no admin required; runs daily at 06:00 local time
powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1

# Optional: different time
powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1 -Time 07:30

# Run and inspect
Start-ScheduledTask -TaskName "Aabhas Daily Data Pipeline"
Get-ScheduledTaskInfo -TaskName "Aabhas Daily Data Pipeline"
Get-Content system\logs\daily_pipeline.log -Tail 100
```

The task calls `system\ops\run_daily.bat`, which runs
`refresh_and_publish.py --catchup` and publishes to the static Space. It uses
the Hugging Face token saved by `hf auth login`; set `HF_SPACE` only when
targeting a different Space.

`--catchup` backfills days missed while the machine was off, so a missed 06:00
run recovers on the next boot. Keep only one scheduled task: legacy duplicates
must be removed from an elevated shell, e.g.
`schtasks /delete /tn "FloodGuard Daily Pipeline" /f`.

### GitHub Actions (optional; not enabled)

> The active automation is the Windows Task Scheduler task above. This workflow
> is kept for a future always-on runner; it is not configured on the current
> repository (missing secrets and the local-only raw/files layout).

`.github/workflows/refresh.yml` runs daily at **04:30 UTC (10:30 Asia/Dhaka)**
and on manual dispatch.

One-time setup:

1. Create a **private** HF Dataset for raw inputs, e.g.
   `username/aabhas-flood-raw`.
2. Bootstrap it once from a machine that has the raw files:

   ```powershell
   $env:HF_TOKEN="hf_xxx"
   $env:HF_RAW_DATASET="username/aabhas-flood-raw"
   .\.venv\Scripts\python.exe system\ops\refresh_and_publish.py `
       --skip-pipeline --skip-features-upload --push-raw
   ```

3. Add repository secrets (Settings → Secrets and variables → Actions):

   | Secret | Purpose |
   |--------|---------|
   | `HF_TOKEN` | Read/write the private feature and raw datasets |
   | `HF_FEATURE_DATASET` | Feature-store dataset (e.g. `username/aabhas-flood-features`) |
   | `HF_RAW_DATASET` | Raw-input mirror (e.g. `username/aabhas-flood-raw`) |
   | `RENDER_DEPLOY_HOOK` | Optional Render deploy hook URL |

4. Add repository variable `AABHAS_REFRESH_ENABLED=true`. Scheduled runs are
   skipped until this is set, so a fork without secrets stays green.
5. Optionally set variable `AABHAS_HEALTH_URL` (default:
   `https://aabhas-flood-predictor-xckw.onrender.com/health`).

What the workflow does:

- downloads the raw bundle to the repo-relative paths;
- runs `daily_pipeline.py --catchup` (or a single `date` from manual dispatch);
- uploads new feature partitions to `HF_FEATURE_DATASET`;
- mirrors changed raw files back to `HF_RAW_DATASET`;
- POSTs the Render deploy hook (if configured) and polls `/health`;
- opens or comments on a `Daily data refresh failed` issue on failure.

The Render API discovers new dates from the HF repo listing with a 15-minute
cache TTL, so a new partition becomes visible without a container rebuild.

### Static HF Space (public site)

The public site is a Static Space serving a precomputed snapshot. The daily
Task Scheduler run keeps it current automatically: the snapshot builder scores
only new dates, and `deploy_static_space.py --changed-only` uploads only the
files whose SHA-256 changed (tracked in `system/deploy/.deploy_state.json`).

First-time publish on a new Space:

```powershell
.\.venv\Scripts\python.exe system\deploy\build_static_snapshot.py
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run build
cd ..\..
.\.venv\Scripts\python.exe scripts\deploy_static_space.py --dry-run
.\.venv\Scripts\python.exe scripts\deploy_static_space.py
```

If the Space already matches an existing build but the local state file is
missing, seed it once instead of re-uploading 200+ MB:

```powershell
.\.venv\Scripts\python.exe scripts\deploy_static_space.py --seed-state
```

Daily incremental publish (wired into `run_daily.bat`):

```powershell
.\.venv\Scripts\python.exe system\ops\refresh_and_publish.py --catchup
```

Notes:

- Delete `system/deploy/.deploy_state.json` to force a full re-upload.
- `--static-full` rebuilds every date and uploads every file; use it after
  frontend code or asset changes.
- `index.html` is uploaded last, so the Space keeps serving the previous build
  until the new data is in place.
- Trend files keep the last 90 dates (`--trend-days`), enough for the UI's
  30-day chart and cheap to republish daily.
- `data/snapshot/health.json` reports `station_count` for the **latest feature
  date** (currently 112), `trend_station_count` for the 90-day union (127), and
  `station_count_scope`; freshness banners use the latest-date value.
- Source freshness and the feature-quality result are available locally under
  `system/data/processed/source_status/` and `system/data/processed/quality/`.

### GitHub Pages (free, always-on public demo)

The current public demo is a static build published to the `gh-pages` branch:

**https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/**

Build and publish manually:

```powershell
# 1. Snapshot (from the local feature store)
.\.venv\Scripts\python.exe system\deploy\build_static_snapshot.py

# 2. Build for the project sub-path, without source maps
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
$env:AABHAS_BASE="/Aabhas-flood-predictor-bangladesh/"
$env:AABHAS_SOURCEMAP="false"
npm run build

# 3. Publish dist/ to gh-pages (force-push an orphan deploy repo)
git -C ..\.. subtree push --prefix system\web\dist origin gh-pages
```

`AABHAS_BASE` makes asset URLs, the service worker scope, the manifest and all data
fetches sub-path aware. For an automated rebuild from the HF feature dataset, run the
**Publish static demo to GitHub Pages** workflow (`.github/workflows/pages.yml`,
manual trigger; requires `HF_FEATURE_DATASET` and optional `HF_TOKEN`).

Pages source is **Deploy from a branch → `gh-pages` → `/`**. The site is served over
HTTPS at no cost and needs no server process.

---

## ☁️ Render deployment

`render.yaml` defines the service:

- Docker build from `Dockerfile` (multi-stage: Node builds the SPA, Python
  serves API + static files);
- `healthCheckPath: /health`;
- `autoDeploy: true` on pushes to the connected branch;
- no pinned `PORT` (the container reads Render's `$PORT`).

The image is **self-contained**: `system/data/demo/model_ready/` ships a tracked
120-day slice of the model-ready feature store, and `models/final_v2/selected_model_v2.pkl`
is included via `.dockerignore`. No secrets are required for a working demo.

Environment variables:

| Variable | Required | Value |
|----------|----------|-------|
| `AABHAS_FEATURE_STORE` | recommended | `system/data/demo/model_ready` |
| `HF_FEATURE_DATASET` | optional | private full feature-store dataset ID |
| `HF_TOKEN` | optional | read token for that dataset |
| `HF_FEATURE_REVISION` | optional | `main` |
| `AABHAS_PREVIEW_MODE` | no | `false` |
| `AABHAS_ENABLE_DOCS` | no | `false` |
| `AABHAS_RATE_LIMIT_PER_MINUTE` | no | `180` |
| `AABHAS_RATE_LIMIT_PREDICT_PER_MINUTE` | no | `60` |
| `AABHAS_ALLOWED_ORIGINS` | no | only if the SPA is hosted separately |

When `AABHAS_FEATURE_STORE` / `HF_FEATURE_DATASET` are absent, the API falls back to
the bundled demo store automatically. Refresh that slice after new dates are published:

```powershell
python scripts\build_demo_store.py --days 120
git add system/data/demo && git commit -m "chore: refresh bundled demo features"
git push origin main
```

First deploy: Render Dashboard → New → Blueprint → select this repo. Render deploys
the branch configured on the service; this repository keeps `main` and `master` in
sync so either choice receives the same code.

Verify:

```powershell
curl https://aabhas-flood-predictor-xckw.onrender.com/health
curl https://aabhas-flood-predictor-xckw.onrender.com/dates
curl "https://aabhas-flood-predictor-xckw.onrender.com/predict/trend?station_id=1&dates=2026-11-04,2026-11-05"
curl "https://aabhas-flood-predictor-xckw.onrender.com/predict/all?date=2026-11-05&horizon=24h"
```

Health should report `model_loaded: true`, `model_version: "D16"`, and a recent
`latest_feature_date`.

---

## 🔌 API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status, model version, latest feature date/age, per-source freshness, `degraded`, cache stats |
| `/model/info` | GET | Frozen D16 metadata |
| `/thresholds` | GET | Published thresholds and risk bands |
| `/dates` | GET | Available feature dates |
| `/predict` | GET | Single station prediction |
| `/predict/batch` | POST | Multiple stations (bounded) |
| `/predict/all` | GET | All stations for a date/horizon |
| `/predict/trend` | GET | One station across up to 60 dates |
| `/alerts/feed` | GET | Draft/approved alert feed (`limit`, `offset`, `status`) |
| `/alerts/{draft_id}/approve` | POST | Duty-officer approval (admin key required) |
| `/docs` | GET | Only when `AABHAS_ENABLE_DOCS=true` |

All JSON endpoints are also reachable under `/api/...`; the middleware strips
the prefix before routing.

---

## 🚨 Early-warning engine

`system/alerts/alert_engine.py` evaluates D16 output per `(station, horizon)` and
maps risk bands to alert levels (`watch` / `warning` / `high`). It persists
per-station state, applies a 24 h cooldown, bypasses the cooldown on escalation,
creates **draft** records, requires duty-officer approval, and writes an immutable
audit trail.

```powershell
# Evaluate one date (all horizons by default)
.\.venv\Scripts\python.exe system\alerts\run_alerts.py --date 2026-11-05 --all-horizons
.\.venv\Scripts\python.exe system\alerts\run_alerts.py --date 2026-11-05 --horizon 24h
```

State and logs:

- per-station state — `system/data/processed/alert_state.json`
- alert log — `system/logs/alerts.csv`
- audit trail — `system/logs/alert_audit.jsonl`

**Delivery is intentionally disabled** (`system/config/alert_config.json` →
`notification.enabled = false`; SMTP and webhook are empty). The dashboard feed
and approval endpoint work. Email/webhook delivery is enabled only after the
thresholds are formally signed off (`docs/THRESHOLD_SIGN_OFF.md`) and credentials
are supplied. Follow `docs/DUTY_OFFICER_PROCESS.md` for review, approval and
escalation. Alerts remain decision support, **not official warnings**.

---

## 🔐 Security and operations

- **Rate limiting** — in-process sliding window. General API: 180/min/IP;
  `/predict/*`: 60/min/IP; `/health` is exempt. 429 responses include
  `Retry-After`. Disable only for local tests with
  `AABHAS_RATE_LIMIT_DISABLED=true`.
- **Security headers** — CSP (non-docs), `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`,
  `Cross-Origin-Opener-Policy`, HSTS when `X-Forwarded-Proto=https`.
- **CORS** — closed by default. Set `AABHAS_ALLOWED_ORIGINS` (comma-separated)
  only when the SPA is on a different origin.
- **Preview/staging auth** — `AABHAS_PREVIEW_MODE=true` requires
  `AABHAS_PREVIEW_USER` and `AABHAS_PREVIEW_PASSWORD` and enables HTTP Basic
  auth on every request.
- **Alert approval auth** — `/alerts/{draft_id}/approve` requires
  `X-Admin-API-Key` matching `AABHAS_ADMIN_API_KEYS` (comma-separated). Without
  configured keys the endpoint cannot approve alerts.
- **Docs** — disabled in production via `AABHAS_ENABLE_DOCS=false`.
- **Caching** — predictions are cached in-process (64 entries) and HTTP
  responses use `Cache-Control` + `ETag`/`stale-while-revalidate`. The HF date
  list is cached for 15 minutes.
- **Never commit** `.env`, `HF_TOKEN`, or deploy hooks. Use repository/Render
  secrets.

For horizontal scaling, move rate limiting and prediction caching to a shared
store (Redis). The current implementation is intentionally single-instance.

---

## 🐛 Troubleshooting

### Live app says "Live API unavailable" / no forecasts
- Check `https://aabhas-flood-predictor-xckw.onrender.com/health`.
- If `model_loaded=false`, inspect Render logs for a missing model file or
  Python dependency.
- If `feature_store_dates=0`, confirm `system/data/demo/model_ready/` is committed
  and present in the image, or set `AABHAS_FEATURE_STORE` / `HF_FEATURE_DATASET`.
- If `latest_feature_age_days` is large, run the refresh workflow/task and then
  `python scripts/build_demo_store.py --days 120`.
- The frontend never substitutes demo data in production; it shows an explicit
  error instead.

### Render cold start
Free services sleep. The first request may take 30–60 s; subsequent requests
are normal. `/health` is exempt from rate limiting so Render probes do not
consume the budget.

### Feature upload fails
- Confirm `HF_TOKEN` has write access and `HF_FEATURE_DATASET` exists.
- Run `scripts/upload_features_to_hf.py --dry-run` to verify the partition
  scan.
- The HF date list is cached for 15 minutes; wait or restart the service if a
  new date does not appear immediately.

### Scheduled refresh fails
- Open the workflow run log or the auto-created issue.
- Common causes: FFWC/Open-Meteo outage, expired HF token, missing
  `HF_RAW_DATASET`, or a gap older than the 30-day catch-up window.
- Check `system/data/processed/source_status/<date>.json` first: a `stale`
  **critical** source (GloFAS > 2 d, POWER soil / ERA5-Land > 7 d) fails the run
  by design.
- If the feature quality gate fails, inspect
  `system/data/processed/quality/<date>.json` for the failing group; rerun the
  source refresh or pass `--allow-degraded` only for a documented exception.
- For older gaps, run `daily_pipeline.py --date YYYY-MM-DD` for each missing
  date (or extend the catch-up window) and re-upload.

### Alerts are not sending
- `notification.enabled` is `false` by default; fill SMTP/webhook settings in
  `system/config/alert_config.json` (or inject secrets) and set it to `true`.
- Delivery also requires a draft to be approved; check `/alerts/feed` and the
  audit trail in `system/logs/alert_audit.jsonl`.
- `/alerts/{draft_id}/approve` returns 401/403 when `AABHAS_ADMIN_API_KEYS` is
  unset or the `X-Admin-API-Key` header is wrong.

### Static Space 404 on deep links
The static build uses hash routing (`/#/stations/1`). The API-served build uses
clean browser URLs and its SPA fallback.

---

## 📊 Monitoring and maintenance

```powershell
# Freshness and model status
curl https://aabhas-flood-predictor-xckw.onrender.com/health

# Deployment history
# Render Dashboard → Logs / Events
# HF Space → Files → Commits

# Local pipeline log
Get-Content system\logs\daily_pipeline.log -Tail 200

# Source freshness + feature quality (local)
Get-Content system\data\processed\source_status\2026-11-05.json
Get-Content system\data\processed\quality\latest.json

# Snapshot metadata (latest-date vs trend-window station counts)
Get-Content system\web\public\data\snapshot\health.json
```

Daily expectations:

- `latest_feature_age_days` ≤ 2 during normal FFWC publication;
- `/dates` ends at the most recent successful pipeline date;
- every **critical** source is `ok` in `source_status/<date>.json`
  (GloFAS ≤ 2 d, POWER soil ≤ 7 d, ERA5-Land ≤ 7 d);
- `quality/<date>.json` has `passed: true` and critical null share 0.0;
- `health.json` reports `station_count` for the latest date (112 on 2026-11-05)
  and `trend_station_count` for the 90-day union (127) — do not compare the two
  as if they were the same scope;
- the UI freshness banner is green (no warning) only when the feature date is
  current.

Alert operations: run `system/alerts/run_alerts.py --date <date> --all-horizons`
after each successful pipeline, review drafts via `/alerts/feed`, and approve
only through the duty-officer process.

**Model updates** remain deliberate: retrain locally, update
`models/final_v2/` and `system/config/model_config.json`, rebuild/verify the
snapshot, then deploy. Never let a scheduled data refresh overwrite the frozen
model.
