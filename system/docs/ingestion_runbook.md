# System Ingestion — Runbook

**Project:** Bangladesh Flood Predictor
**Component:** Ingestion Layer (Step 2 of system build)
**Location:** `system/ingestion/`
**Status:** Implemented; live ingestion artifacts are present in this workspace.
**Verification snapshot:** 2026-10-30. The FFWC manifest records successful
fetches through query date 2026-10-28 (fetched 2026-10-29 UTC); raw gauge files
contain observations through 2026-10-29 12:00 UTC. Rainfall data runs through
2026-10-29 and includes NASA POWER / MERRA-2 and Open-Meteo ERA5-Land rows.
These dates describe this workspace snapshot, not a guarantee of ongoing updates.

---

## Overview

The ingestion layer fetches raw environmental data from external sources, applies quality-control (QC) filters, and writes immutable raw files under `system/data/raw/`. Every fetch is recorded in a provenance manifest for resumability and auditability.

## Components

| File | Purpose |
|------|---------|
| `run_ingestion.py` | Orchestration entry point |
| `ingest_ffwc.py` | FFWC water-level (3-hourly) harvest + daily aggregation |
| `ingest_rainfall.py` | Rainfall fetch (NASA POWER primary / Open-Meteo fallback) |
| `qc_rules.py` | QC filters (F1, F3, F8) + daily aggregation |
| `provenance.py` | Manifest management (resumability + audit) |
| `config/ingestion_config.json` | All configuration (API endpoints, rate limits, QC thresholds) |

## QC Filters

| ID | Filter | Rule |
|----|--------|------|
| F1 | NASA POWER sentinel | `<= -999` → NaN (never 0) |
| F3 | Impossible water level | `> 100 m` → dropped |
| F8 | Sensor spike | `\|wl - local 12-h median\| > 10 m` → dropped |
| — | Missing-code sentinel | `<= -1000` → dropped |
| — | Min readings per day | `< 3` → label invalid (NaN) |

## Usage

### 1. Fetch station master (one-time)
```powershell
python system/ingestion/run_ingestion.py --stations-only
```
Writes `system/data/raw/ffwc/ffwc_stations.csv` (153 stations).

### 2. Harvest FFWC water levels (bulk, resumable)
```powershell
python system/ingestion/run_ingestion.py --ffwc --start 2017-01-01 --end 2026-10-06 --workers 8
```
- One API call per station per stride day (default stride 35 days)
- Each call returns ~40.5-day trailing window; overlapping rows deduplicated on `(st_id, wl_date)`
- Writes `system/data/raw/ffwc/waterlevel/{station_id}/{year}.csv`
- Manifest at `system/data/raw/ffwc/harvest_manifest.csv` tracks completed calls
- Re-running skips already-fetched (station, date) pairs

### 3. Fetch rainfall
```powershell
python system/ingestion/run_ingestion.py --rainfall --source nasa_power --start 2016-12-01 --end 2026-10-06
```
- NASA POWER (MERRA-2 reanalysis): one call per station, ~0.6s sleep between stations
- Writes `system/data/raw/rainfall/weather_station_daily.csv`
- Partial checkpoints every 10 stations for resumability

Open-Meteo fallback (batched, ~20 stations per request):
```powershell
python system/ingestion/run_ingestion.py --rainfall --source open_meteo --start 2016-12-01 --end 2026-10-06
```

### 4. Run everything
```powershell
python system/ingestion/run_ingestion.py --all --start 2017-01-01 --end 2026-10-06 --workers 8
```

## Resumability

- **Manifest**: Every `(station_id, query_date, source)` call is recorded. Re-running skips completed entries.
- **Station-year CSVs**: Overlapping FFWC data is deduplicated on `(st_id, wl_date)` when written.
- **NASA POWER partial**: Checkpointed every 10 stations; stations with complete date ranges are skipped on resume.
- **Open-Meteo**: Per-station CSV files under `system/data/raw/openmeteo/era5land/`; concatenated at the end.

## Rate Limits & Retries

| Source | Retries | Backoff | Notes |
|--------|---------|---------|-------|
| FFWC | 4 | 1.5× base | Single-day queries; 8 parallel workers |
| NASA POWER | 5 | 2× base | 0.6s sleep between stations; 429/5xx triggers backoff |
| Open-Meteo | 6 | 45× base on 429 | Batched (20 stations/request); 429 triggers long wait |

MI-1a in the parent repo hit an Open-Meteo 429 window that took ~17 minutes to clear. The system ingestion waits up to 30 minutes for API recovery.

## Output Layout

```
system/data/raw/
├── ffc/
│   ├── ffc_stations.csv
│   ├── harvest_manifest.csv
│   └── waterlevel/
│       ├── 1/
│       │   └── 2017.csv
│       ├── 1/
│       │   └── 2018.csv
│       └── ...
├── rainfall/
│   ├── weather_station_daily.csv
│   ├── weather_station_daily.partial.csv
│   └── rainfall_manifest.csv
└── openmeteo/
    ├── era5land/
    │   ├── 1.csv
    │   └── ...
    └── era5land_station_daily.csv
```

## Verification

After ingestion, run:
```powershell
python system/ingestion/ingest_ffwc.py summary
```
to list all water-level files with byte counts.

Check source and fetch times in the manifests, and inspect `wl_date` in raw
station-year CSVs. A feature partition date is not the timestamp of every source
observation; gauge and rainfall sources have different publication latencies.

Check manifest:
```powershell
python -c "import json; from system.ingestion.provenance import ProvenanceManifest; m=ProvenanceManifest('system/data/raw/ffwc/harvest_manifest.csv'); print(m.summary())"
```

## Known Constraints

- FFWC API requires an internal key + browser Origin headers (already configured in `ingestion_config.json`)
- NASA POWER is reanalysis (MERRA-2, ~50 km grid), not observed gauge data — stated caveat R5
- Open-Meteo ERA5-Land has ~5-day latency; same-day data may not be available
- Full harvest (133 stations × 10 years) takes ~10-20 minutes depending on API responsiveness

## Scheduled operation

`system/ops/run_daily.bat` is a wrapper intended to run the daily pipeline and
append output to `system/logs/daily_pipeline.log`. The repository does not define
or install a Windows Task Scheduler task. On the development workstation checked
2026-10-30, `\Aabhas Daily Data Pipeline` is registered and enabled for 06:00
local time daily. It runs as the current user in interactive-only mode, and
Windows is configured to stop it on battery. It has not run yet; the first
scheduled run is expected at 06:00 on 2026-10-30. It will not run as an
always-on unattended production service. Recreate and verify equivalent
scheduling, monitoring, and power settings in the deployment environment.
