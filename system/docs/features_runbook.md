# System Features — Runbook

**Project:** Bangladesh Flood Predictor
**Component:** Feature Computation Layer (Step 3 of system build)
**Location:** `system/features/`
**Status:** Implemented; model-ready partitions are present in this workspace.
**Verification snapshot:** 2026-10-30. The latest available partition is
`date=2026-10-29` and contains 127 stations. Its date identifies the feature row
date; it does not mean all inputs were observed at that exact time or that the
feature store is updating continuously.

---

## Overview

The feature computation layer takes raw QC'd water levels + rainfall (ingested by Step 2),
joins static/reference data from the parent repo (read-only), and computes all 110
FULL_V2 features per `(station_id, date)`. Output is model-ready parquet partitioned by
date for fast inference lookup.

## Components

| File | Purpose |
|------|---------|
| `config.py` | 110 FULL_V2 column definitions, feature groups, paths, QC thresholds |
| `hydrology.py` | Water-level features (lags, rolls, rise, pct_danger, exceed_days) |
| `rainfall.py` | Rainfall features (rolling sums, max, wet days, temporal) |
| `static.py` | Static spatial join, soil moisture, GloFAS discharge |
| `spatial.py` | Neighbour (nb3/nb5) + upstream features |
| `derived.py` | V2_NEW derived (wl_last long lags, api_09/07) + interactions (ix_*) |
| `build.py` | Main orchestration pipeline |
| `run_build.py` | CLI entry point |

## Feature Groups (110 FULL_V2 columns)

| Group | Columns | Source |
|-------|---------|--------|
| static | 15 | station master + station_spatial_features.csv |
| hydrology_base | 6 | FFWC water level daily aggregates |
| hydrology_lags | 11 | wl_last/wl_max lags (1-7, 10, 14, 21, 30) |
| hydrology_derived | 9 | rise, dist_to_danger, pct_danger, rolls, exceed_days, api_09/07 |
| rainfall | 10 | NASA POWER precip/temp + rolling features |
| temporal | 2 | doy_sin, doy_cos |
| neighbour | 8 | nb3/nb5 rain_7d/14d, wl_pct_danger, n_wl_danger |
| upstream | 4 | up_rain7d_lag1/2, up_wl_pct_danger_lag1/2 |
| soil_moisture | 20 | NASA POWER soil wetness + derived |
| glofas | 19 | GloFAS transboundary discharge ratios |
| interactions | 5 | ix_wet_top_x_slope, ix_wet_top_x_updis, ix_updis_x_wlrise, ix_api09_x_wet_top, ix_updis14_x_api |

## QC Filters (same as ingestion layer)

| ID | Filter | Rule |
|----|--------|------|
| F1 | NASA POWER sentinel | `<= -999` → NaN (never 0) |
| F3 | Impossible water level | `> 100 m` → dropped |
| F8 | Sensor spike | `\|wl - local 12-h median\| > 10 m` → dropped |
| — | Missing-code sentinel | `<= -1000` → dropped |

## Leakage Rules

- All shifts are backward-looking (day t uses data up to day t-1)
- No forward shifts anywhere
- Climatologies built from <=2021 rows only
- Neighbour features use same-day values (known by issue instant)
- Upstream features use lag1/lag2 (strictly before label window)

## Usage

### Build full feature matrix (all dates)
```powershell
python -m system.features.run_build --start 2017-01-01 --end 2026-10-06
```

### Build specific dates only
```powershell
python -m system.features.run_build --dates 2024-07-01 2024-07-02 2024-07-03
```

### Write single parquet (not partitioned)
```powershell
python -m system.features.run_build --start 2017-01-01 --end 2026-10-06 --no-partition
```

## Output Layout

### Partitioned (default)
```
system/data/processed/model_ready/
├── date=2017-01-01/
│   └── data.parquet      (columns: station_id, f1..f110)
├── date=2017-01-02/
│   └── data.parquet
└── ...
```

### Single parquet (`--no-partition`)
```
system/data/processed/features.parquet
```

## Dependencies

- Step 2 (Ingestion) must have populated `system/data/raw/ffwc/` and `system/data/raw/rainfall/`
- Parent repo data (read-only): `data/raw/ffwc/ffwc_stations.csv`, `data/processed/station_spatial_features.csv`, `data/raw/openmeteo/era5land_station_daily.csv`, `data/raw/glofas/`, `data/raw/power_soil/`

## Verification

After build, check:
```powershell
python -c "
import pandas as pd
df = pd.read_parquet('system/data/processed/features.parquet')
print(df.shape)
print(df.columns.tolist())
print(df.isna().mean().sort_values(ascending=False).head(10))
"
```

The API health endpoint reports `latest_feature_date` from the partition list.
Use raw source timestamps and ingestion manifests when assessing observation
freshness; do not treat a partition date as a live sensor timestamp.

## Prerequisites Checklist

- [x] Step 2 ingestion completed (FFWC water levels + rainfall raw data present)
- [x] Parent repo reference data accessible (read-only)
- [x] `scipy` installed (for api_09/api_07 recursive filter)
- [x] `pyarrow` installed (for parquet output)
