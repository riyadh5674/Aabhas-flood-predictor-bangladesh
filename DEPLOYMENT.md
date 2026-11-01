# Aabhas Flood Predictor — Deployment Guide

## 🎯 Deployed Architecture (free tier, current)

As of **2026-11-01** the app is deployed entirely on free Hugging Face
resources:

| Component | Host | URL |
|-----------|------|-----|
| Web app | HF **Static** Space | <https://riyadhuggy-aabhas-flood-outlook.static.hf.space/> |
| Feature store | HF Dataset (private) | <https://huggingface.co/datasets/riyadhuggy/aabhas-flood-features> |
| Model + API code | GitHub (private) | `riyadh5674/flood-predictor` |

**Verified 2026-11-01:** Space stage `RUNNING` (public); `/`, `/index.html`,
`/data/snapshot/health.json`, one date's predictions, and a station trend all
return HTTP 200; `dates.json` lists 2,611 dates from `2017-01-01` through
`2026-10-30`.

```
┌──────────────────────────────┐
│  HF Static Space             │
│  React build + precomputed   │
│  forecasts (2,611 dates)     │
└──────────────┬───────────────┘
               │ build-time
               ▼
┌──────────────────────────────┐     ┌──────────────────────────┐
│  HF Dataset                  │     │  Frozen D16 model        │
│  data/train/date=*/data.parquet│────▶│  24h/48h/72h prediction  │
└──────────────────────────────┘     └──────────────────────────┘
```

**Why static instead of a live API?** Hugging Face now returns
`402 Payment Required` when creating Docker or Gradio Spaces on the free
`cpu-basic` hardware: *"hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription."* Static Spaces remain free, so the web app
ships a precomputed, all-real forecast snapshot instead of calling an API at
runtime.

**Live API remains optional.** FastAPI still runs locally or in a container
(`Dockerfile` / `docker-compose.yml`), and will work on any host with Docker
support. Set `VITE_API_BASE` and build **without**
`VITE_AABHAS_STATIC_SNAPSHOT` to use it.

**Free-tier notes:**
- ✅ HF Static Space: free, public, ~1 GB repo guidance
- ✅ HF Dataset (private): free up to 100 GB
- ✅ GitHub: private repository, Actions available
- ❌ HF Docker/Gradio Space on `cpu-basic`: now requires PRO

---

## 🚀 Rebuild & Redeploy (static Space)

Run these from the repository root (`D:\Flood predictor`) after the feature
store or model changes.

```powershell
# 1. Rebuild the forecast snapshot used by the static app
.\.venv\Scripts\python.exe system\deploy\build_static_snapshot.py

# 2. Build the frontend against the snapshot
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run build
cd ..\..

# 3. Publish the build (chunked upload, README last) — dry run first
.\.venv\Scripts\python.exe scripts\deploy_static_space.py --dry-run
.\.venv\Scripts\python.exe scripts\deploy_static_space.py
```

`scripts\deploy_static_space.py` uploads `system\web\dist` in a fixed sequence
of commits. A single commit carrying ~3,000 build files is slow and flaky, so
predictions are split into parts and `index.html` is published only after the
data is complete — the Space keeps serving the previous build until then. It
reads `HF_TOKEN` and accepts `--repo`, `--parts`, and `--skip-readme`.

The Space README (YAML `sdk: static` header) is versioned at
`system\deploy\static_space_README.md`, so its metadata travels with the code
instead of living in a temp file.

The static build uses **hash routing** (`.../#/stations/1`) because HF static
Spaces return 404 for arbitrary deep-link paths. The API-served build keeps
clean browser URLs.

### One-time setup (already done)

```bash
# Feature store layout: data/train/date=YYYY-MM-DD/data.parquet
python scripts/upload_features_to_hf.py --dataset riyadhuggy/aabhas-flood-features

# Static Space (free)
hf repos create riyadhuggy/aabhas-flood-outlook --type space --sdk static --public
```

### Optional: Cloudflare Pages + live API

If Cloudflare credentials or a Docker-capable host become available:

1. Create a Cloudflare Pages project (build `npm run build`, output `dist`,
   env `VITE_AABHAS_STATIC_SNAPSHOT=true`), **or** deploy `dist` with
   `npx wrangler pages deploy dist`.
2. Run the API with Docker (`docker-compose up --build`) and set
   `VITE_API_BASE` to its public URL (build without the snapshot flag).

---

## 📦 Feature Store Upload (Run Once, Then After Each Pipeline)

```bash
# Set environment variables
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export HF_FEATURE_DATASET="username/aabhas-flood-features"

# Upload local feature store partitions to HF Datasets
python scripts/upload_features_to_hf.py

# Verify partition count and date range
python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('username/aabhas-flood-features', repo_type='dataset')
parts = [f for f in files if f.endswith('/data.parquet')]
print(len(parts), 'partitions')
"
```

**Feature-store upload is local only.** CI cannot do it: the partitions are
generated on the machine that runs the pipeline, and the job would have no
source data to upload. Run `scripts/upload_features_to_hf.py` after each
successful local pipeline run.

---

## 🏃 Local Development

### Option A: Docker Compose (Recommended)

```bash
# Copy env example and fill values
cp .env.example .env
# Edit .env with your HF credentials

# Start everything
docker-compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:5173
# Health: http://localhost:8000/health
```

### Option B: Manual (No Docker)

```bash
# Backend
cd "D:\Flood predictor"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.venv\Scripts\python.exe -m pip install datasets huggingface_hub

# Set HF feature store (optional - leave empty for local)
$env:HF_FEATURE_DATASET="username/aabhas-flood-features"
$env:HF_TOKEN="hf_xxx"

# Start API
.venv\Scripts\uvicorn system.api.app:app --reload --port 8000

# Frontend (separate terminal)
cd system/web
npm install
$env:VITE_API_BASE="http://localhost:8000"
npm run dev
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

`.github/workflows/deploy.yml` runs two jobs:

1. **API and web checks** — on every push/PR to `main` or `master`: API import
   smoke tests, `npm ci`, TypeScript typecheck and a production build.
2. **Rebuild snapshot and publish static Space** — manual only
   (Actions → "Aabhas build & deploy" → Run workflow).

The deploy job is manual because it needs the private feature-store dataset
and has to score every date. Repository secrets:

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | read the private feature dataset, write the Space |
| `HF_FEATURE_DATASET` | e.g. `riyadhuggy/aabhas-flood-features` |
| `HF_SPACE_ID` | e.g. `riyadhuggy/aabhas-flood-outlook` |

The job downloads the partitioned feature store from HF Datasets (never from
git), builds the snapshot locally in one pass, builds the frontend, and calls
`scripts/deploy_static_space.py`, which publishes `index.html` last so the
Space keeps serving the previous build until the new snapshot is complete.

**Deliberately not automated:** the FastAPI service. Docker Spaces on free
`cpu-basic` return `402 Payment Required`, so `Dockerfile` and
`docker-compose.yml` are for local or paid hosts only. The previous Cloudflare
Pages job is also gone — the app is served from the HF static Space.

Run it locally with the same commands in "Rebuild & Redeploy" above.

---

## 🔧 Configuration Details

### Model Configuration (`system/config/model_config.json`)

Key production settings:
```json
{
  "inference": {
    "hf_feature_dataset": null,      // Set via HF_FEATURE_DATASET env var
    "hf_feature_revision": "main",   // Dataset revision/branch
    "feature_store": "system/data/processed/model_ready"  // Local fallback
  },
  "api": {
    "port": 8000                     // Overridden by PORT env var (HF Spaces uses 7860)
  }
}
```

### Feature Store Format

The HF Dataset expects partitioned parquet files with schema:
- Layout: `data/train/date=YYYY-MM-DD/data.parquet`
- Columns: `station_id`, `date`, + 110 feature columns
- The API downloads only the partition for the requested date
  (`hf_hub_download`) and lists dates from the repo file tree, so a request
  never scans the full multi-year store.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + model status |
| `/model/info` | GET | D16 model metadata |
| `/thresholds` | GET | Phase-29 thresholds & risk bands |
| `/dates` | GET | Available feature dates |
| `/predict` | GET | Single station prediction |
| `/predict/batch` | POST | Multiple stations |
| `/predict/all` | GET | All stations for date/horizon |

---

## 🐛 Troubleshooting

### Optional API container won't start
```bash
# The live API is not deployed: free HF cpu-basic returns 402 for Docker Spaces.
# If you host it yourself, common issues:
# 1. Model file not found → models/final_v2/selected_model_v2.pkl must be present
# 2. HF_TOKEN missing → read secret for the private feature dataset
# 3. HF_FEATURE_DATASET not set → e.g. riyadhuggy/aabhas-flood-features
# 4. Memory limit → 16GB on free CPU; model + features must fit
```

### Frontend shows "Unavailable" / no data
- The static build reads `data/snapshot/*` from the Space, not an API. Confirm
  `system/web/dist/data/snapshot/dates.json` exists before deploying.
- Check `.../data/snapshot/health.json` on the live Space for the snapshot date.
- If you built **without** `VITE_AABHAS_STATIC_SNAPSHOT=true`, the app expects a
  live API at `VITE_API_BASE` instead.

### Feature store not found
- Run `scripts/upload_features_to_hf.py` to upload local features
- Verify dataset name matches `HF_FEATURE_DATASET`
- Check dataset has `date` column for partitioning

### GitHub Actions fails
- Check secrets are correctly named (case-sensitive): `HF_TOKEN`,
  `HF_FEATURE_DATASET`, `HF_SPACE_ID`
- Verify `HF_TOKEN` can read the private feature dataset and write the Space
- The deploy job only runs from `workflow_dispatch`, never on push

---

## 📊 Monitoring & Maintenance

### Health Checks
- Snapshot status: `GET https://riyadhuggy-aabhas-flood-outlook.static.hf.space/data/snapshot/health.json`
- Deployment history: Space → Files → Commits

### Feature Store Updates
Run after each daily pipeline completion (local only):
```bash
python scripts/upload_features_to_hf.py
```

### Model Updates
1. Retrain locally → new `selected_model_v2.pkl`
2. Update `model_config.json` (feature set, horizons, frozen timestamp)
3. Rebuild the snapshot, rebuild the frontend, then run
   `scripts/deploy_static_space.py` (or dispatch the deploy workflow)

### Cost Monitoring
All services used are **free tier**. Monitor:
- HF Spaces: storage/analytics for the static Space
- HF Datasets: storage for the private feature store
- GitHub Actions: minutes in Settings → Billing (the deploy job is manual)

---

## 🔐 Security Notes

- **Never commit `.env` or tokens** - Use GitHub Secrets and HF Space secrets
- **Preview mode** - Enable `AABHAS_PREVIEW_MODE=true` with strong credentials for staging
- **CORS** - API allows all origins; restrict in production if needed
- **Rate limiting** - Not implemented; add if exposing publicly
- **HTTPS** - Automatic on HF Spaces (and Cloudflare Pages, if used)

---

## 📞 Support

- **HF Spaces issues**: Check Space logs, HF Community forums
- **Cloudflare Pages**: Cloudflare Dashboard → Pages → Deployments
- **GitHub Actions**: Actions tab → Failed workflow → Logs
- **Model/API bugs**: Check `system/logs/daily_pipeline.log` and API `/health`