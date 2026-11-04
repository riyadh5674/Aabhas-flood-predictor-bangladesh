# Aabhas Flood Predictor — Deployment Guide

**Updated:** 2026-11-02

## 🎯 Deployed architecture

The production app is a single FastAPI container that serves both the JSON API
and the compiled React SPA. It runs on Render's free tier and reads feature
partitions from a private Hugging Face Dataset on demand.

| Component | Host | URL |
|-----------|------|-----|
| Web app + API | Render (Docker, free) | <https://aabhas-flood-predictor.onrender.com> |
| Feature store | HF Dataset (private) | `HF_FEATURE_DATASET` (e.g. `riyadhuggy/aabhas-flood-features`) |
| Raw/reference inputs | HF Dataset (private, optional) | `HF_RAW_DATASET` (used by scheduled refresh) |
| Static snapshot (demo) | HF Static Space | <https://riyadhuggy-aabhas-flood-outlook.static.hf.space/> |
| Model + API code | GitHub | `riyadh5674/flood-predictor` |

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

1. ingestion → features (local partitioned parquet store);
2. **incremental** snapshot build — only dates missing from the published
   snapshot are scored; trend files are trimmed to a 90-day window;
3. mirror the changed snapshot files into `system/web/dist`;
4. upload **only changed files** to the Hugging Face Static Space.

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
   `https://aabhas-flood-predictor.onrender.com/health`).

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

---

## ☁️ Render deployment

`render.yaml` defines the service:

- Docker build from `Dockerfile` (multi-stage: Node builds the SPA, Python
  serves API + static files);
- `healthCheckPath: /health`;
- `autoDeploy: true` on pushes to the connected branch;
- no pinned `PORT` (the container reads Render's `$PORT`).

Required Render environment variables:

| Variable | Value |
|----------|-------|
| `HF_FEATURE_DATASET` | private feature-store dataset ID |
| `HF_TOKEN` | read token for that dataset |
| `HF_FEATURE_REVISION` | `main` |
| `AABHAS_PREVIEW_MODE` | `false` |
| `AABHAS_ENABLE_DOCS` | `false` |
| `AABHAS_RATE_LIMIT_PER_MINUTE` | `180` |
| `AABHAS_RATE_LIMIT_PREDICT_PER_MINUTE` | `60` |
| `AABHAS_ALLOWED_ORIGINS` | only if the SPA is hosted separately |

First deploy: Render Dashboard → New → Blueprint → select this repo.
Verify:

```powershell
curl https://aabhas-flood-predictor.onrender.com/health
curl https://aabhas-flood-predictor.onrender.com/dates
curl "https://aabhas-flood-predictor.onrender.com/predict/trend?station_id=1&dates=2026-10-30,2026-10-31"
```

Health should report `model_loaded: true`, `model_version: "D16"`, and a recent
`latest_feature_date`.

---

## 🔌 API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status, model version, latest feature date/age, cache stats |
| `/model/info` | GET | Frozen D16 metadata |
| `/thresholds` | GET | Published thresholds and risk bands |
| `/dates` | GET | Available feature dates |
| `/predict` | GET | Single station prediction |
| `/predict/batch` | POST | Multiple stations (bounded) |
| `/predict/all` | GET | All stations for a date/horizon |
| `/predict/trend` | GET | One station across up to 60 dates |
| `/docs` | GET | Only when `AABHAS_ENABLE_DOCS=true` |

All JSON endpoints are also reachable under `/api/...`; the middleware strips
the prefix before routing.

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
- Check `https://aabhas-flood-predictor.onrender.com/health`.
- If `model_loaded=false`, inspect Render logs for a missing model file or
  Python dependency.
- If `latest_feature_age_days` is large, run the refresh workflow/task.
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
- For older gaps, run `daily_pipeline.py --date YYYY-MM-DD` for each missing
  date (or extend the catch-up window) and re-upload.

### Static Space 404 on deep links
The static build uses hash routing (`/#/stations/1`). The API-served build uses
clean browser URLs and its SPA fallback.

---

## 📊 Monitoring and maintenance

```powershell
# Freshness and model status
curl https://aabhas-flood-predictor.onrender.com/health

# Deployment history
# Render Dashboard → Logs / Events
# HF Space → Files → Commits

# Local pipeline log
Get-Content system\logs\daily_pipeline.log -Tail 200
```

Daily expectations:

- `latest_feature_age_days` ≤ 2 during normal FFWC publication;
- `/dates` ends at the most recent successful pipeline date;
- the UI freshness banner is green (no warning) only when the feature date is
  current.

**Model updates** remain deliberate: retrain locally, update
`models/final_v2/` and `system/config/model_config.json`, rebuild/verify the
snapshot, then deploy. Never let a scheduled data refresh overwrite the frozen
model.
