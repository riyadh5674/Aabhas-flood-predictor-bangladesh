# MI-2 — v2 feature table: build record and Phase-13 leakage audit

| | |
| --- | --- |
| **Date** | 2026-10-16 |
| **Stage** | MI-2 (improvement plan) — build the v2 predictor table |
| **Module** | `scripts/build_mi_features.py` |
| **Inputs** | `data/processed/experimental_dataset.csv` (key) · `data/raw/power_soil/power_soil_daily.csv` · `data/raw/glofas/discharge_anchors_daily.csv` · `data/raw/openmeteo/era5land_station_daily.csv` (gated) |
| **Outputs** | `data/processed/mi_v2_features.csv` · `data/processed/feature_sets_v2.json` · log `scripts/mi2_run.log` |
| **Rows / keys** | 200,665 × (2 keys + 48 new columns), 133 stations |
| **FULL_V2 size** | **110** = FULL 62 (v1) + **48** new |

---

## 1. What was built

The immutable `experimental_dataset.csv` (200,665 × 67) is the key. Three source
blocks are joined on `(station_id, date)` with `validate="one_to_one"`, then five
interaction columns are added:

| Block | Source | New cols | Prefix |
| --- | --- | ---: | --- |
| POWER soil wetness | NASA POWER (133/133 stations) | 19 | `ps_` |
| GloFAS transboundary discharge | 11 river anchors → basin / nearest-anchor | 18 | `gf_` |
| Derived (water-level memory + antecedent rain) | existing WL + rain columns | 6 | `wl_last_lag{10,14,21,30}` · `api_` |
| Interactions | products of the above | 5 | `ix_` |
| **ERA5-Land** | Open-Meteo (only if ≥130 stations) | **0 — SKIPPED** | `e5_` |
| | | **48** | |

### 1.1 POWER soil wetness (`ps_`, 19)
Surface / root-zone / profile wetness (`ps_wet_top`, `ps_wet_root`, `ps_wet_prof`),
each with 1- and 7-day backward lags; 1-day and 7-day changes; 7/14/30-day means;
per-station day-of-year climatology and anomaly (`_clim`, `_anom`).

### 1.2 GloFAS discharge (`gf_`, 18)
Anchors are grouped into Ganges / Brahmaputra / Meghna (fallback = the three main
boundary gauges). Per basin-day we take the mean of the anchor `dis_ratio`
(discharge ÷ its ≤2021 climatology) and `rise3d`; then backward lags 1–14, rolling
3/7/14-day means, acceleration, and the **nearest anchor** absolute discharge with
its 1- and 3-day lags (`gf_near_dis*`).

### 1.3 Derived (`wl_last_lag{10,14,21,30}`, `api_09`, `api_07`, 6)
Long water-level memory (backward lags 10/14/21/30) and two Antecedent Precipitation
Index series (causal exponential decay, factors 0.9 and 0.7) built with
`scipy.signal.lfilter` — a strictly backward recurrence.

### 1.4 Interactions (`ix_`, 5)
`ix_wet_top_x_slope`, `ix_wet_top_x_updis`, `ix_updis_x_wlrise`,
`ix_api09_x_wet_top`, `ix_updis14_x_api`. (Two more, `ix_rain7_x_sm0_7` and
`ix_sm0_7_x_slope`, are defined but **not** created because the ERA5 block is
skipped.)

---

## 2. Phase-13 leakage audit of the new columns

**Rule (unchanged):** every stored value must be a function of days **≤ t**, issue
instant 24:00 UTC of day `t`. Each new column was checked against that rule by
reading the builder.

| Group | Time dependency | Verdict |
| --- | --- | --- |
| `ps_wet_*` (levels) | day **t** observation (daily mean) | ADMIT |
| `ps_wet_*_lag{1,7}` | `shift(+1)`, `shift(+7)` — backward | ADMIT |
| `ps_wet_*_chg1d/chg7d` | difference of `t` and a backward lag | ADMIT |
| `ps_wet_*_{7,14,30}d` | backward rolling mean, `min_periods=1` | ADMIT |
| `ps_wet_*_clim` | per-(station, doy) mean over rows **`date < 2022-01-01` only** | ADMIT (fixed past climatology) |
| `ps_wet_*_anom` | `t` value − the ≤2021 climatology | ADMIT |
| `gf_up_*` (ratio, rise3d) | derived from anchor discharge on day **t** | ADMIT |
| `gf_up_ratio_lag{1..14}` | `shift(+k)`, backward | ADMIT |
| `gf_up_ratio_{3,7,14}d` | backward rolling mean | ADMIT |
| `gf_up_rise3d_lag{1,3}`, `gf_up_accel` | backward lags / difference of lags | ADMIT |
| `gf_near_dis`, `gf_near_dis_lag{1,3}` | anchor day **t** and backward lags | ADMIT |
| `wl_last_lag{10,14,21,30}` | `shift(+10/+14/+21/+30)` | ADMIT |
| `api_09`, `api_07` | causal `lfilter` recurrence (uses `t` and earlier rain only) | ADMIT |
| `ix_*` | products of already-admitted columns | ADMIT |

**Climatology boundary.** Both climatologies (`ps_*_clim` and the GloFAS `clim`
used to form `dis_ratio`) are computed from rows whose date is **strictly before
2022-01-01** — i.e. the training window — so no test-period value informs them.

**No forward shifts.** The only `.shift(-` occurrences in the project live inside
`add_labels` (Phase 13, check L1); the MI-2 builder contains none. All shifts are
positive (backward).

**Verdict:** the 48 new columns are **ADMIT**; `FULL_V2 = 62 + 48 = 110`.

---

## 3. Missingness of the new columns

Worst null rates (all computed on the 200,665 key rows):

| column | null rate |
| --- | ---: |
| `wl_last_lag30` | 0.0199 |
| `wl_last_lag21` | 0.0139 |
| `wl_last_lag14` | 0.0093 |
| `wl_last_lag10` | 0.0066 |
| `ps_wet_top`, `ps_wet_root` | 0.0000 |

The ladder `lag10 < lag14 < lag21 < lag30` is the expected edge effect of longer
memory at the start of each station's record. POWER soil is complete (0 % null).
Tree models (XGBoost/LightGBM) treat NaN natively, so no imputation is applied.

---

## 4. ERA5-Land — deferred, not dropped

| | |
| --- | --- |
| Coverage on disk | **113 / 133** station files (`data/raw/openmeteo/era5land/`) |
| Combined file | `era5land_station_daily.csv` — 568,729 rows, 113 stations |
| Builder gate | `era5_block()` returns `None` when `nunique < 130` → `e5_*` skipped |
| Why | Open-Meteo `archive-api` rate-limited the remaining 20 stations (HTTP 429), then rejected large multi-location batches (HTTP 400) |
| Decision | **Skip ERA5 for now** (user, 2026-10-16); proceed to MI-3 on the POWER+GloFAS v2 set. ERA5 is a *later add-on*: once the last 20 stations land, re-running this builder will add **23 `e5_*` columns + 2 `ix_*`** → `FULL_V2 ≈ 135`. |

The 113 files already on disk are kept; nothing was discarded.

---

## 5. Reproduce

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\build_mi_features.py       # -> mi_v2_features.csv, feature_sets_v2.json
```

Deterministic: no sampling, no randomness (fixed grouping/rolling operations).
Seed is irrelevant to this stage.

---

## 6. Consumed by

* `scripts/mi3_compare.py` — scores FULL (62) vs FULL_V2 (110) on the seven
  development cells → `reports/mi3_featureset_compare.csv` / `_summary.json`.
* `scripts/mi3_search.py` — MI-3b HPO; loads `mi_v2_features.csv` via
  `feature_sets_v2.json` and optimises on `FULL_V2`.
