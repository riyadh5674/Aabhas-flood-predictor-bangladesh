# Phase 10–12 — Feature Engineering & Feature Selection
### Project study notes · Bangladesh flood-probability forecasting

> This continues the study series. Two formal sources back it: `reports/phase11_spatial_features.md`
> (Phases 10–11 — building the predictor columns) and `reports/phase12_feature_selection.md`
> (Phase 12 — choosing which of them to use). The machine-readable record they produced is
> `data/processed/feature_sets.json`, which is quoted directly below. Same rules as always:
> plain English, real numbers, real files, what/why/how, problems + solutions.
>
> *Where this sits:* the previous file (Phase 6–9) analysed the data; this one **builds the
> things the models will actually consume**, then **decides which to keep**.

---

## 0. The 60-second version

Phases 10–12 turn a raw per-day table into a **modelling table with chosen inputs**.

- **Phase 10–11 — Feature engineering.** We build the predictors the model reasons with:
  *where it is* (lat/lon, terrain, distance to coast/river), *how high the water is right now*
  (levels, lags, daily rises, distance to the danger level, 30-day exceedance counts),
  *how much it has rained* (1/3/7/14/30-day windows, 7-day maximum, wet days), *what the
  calendar says* (sin/cos of day-of-year), *what the neighbours are doing* (nearest-3 and
  nearest-5 gauges, resolved **per day**), and *what upstream did* (the same-river gauge at
  higher elevation, **lagged 1–2 days**). The last two groups are the **19 spatial columns**
  added in Phase 11, taking the table from **54 → 73 columns**.
- **Phase 12 — Feature selection.** From the 73 columns: 10 are banned (identifiers, labels,
  and the `obs_days_*` label-window columns), 1 is constant → **62 usable features**. We then
  ran **four** selection lenses and froze three sets: **FULL 62** (trees), **LINEAR 35**
  (logistic regression — VIF-pruned), **SELECTED 10** (a compact model).

The headline finding is uncomfortable and important: **one feature dominates**.
`wl_dist_to_danger` ("how far the current level is below its danger level") carries **77 % of
the tree's split gain** and a **0.53** mean permutation drop. The second feature is **0.027**.
And the whole 62-feature LightGBM only beats the *single-feature* persistence baseline by
**×1.06–1.10**. So most of the "signal" is one number — and the model's real contribution is a
modest, honest margin on top of it.

---

## 1. Where Phase 10–12 sit

| Phase | What | Plain words |
|---|---|---|
| 6–9 | Relationship / redundancy / temporal / spatial analysis | Understood the data (previous file) |
| **➜ 10–11** | **Feature engineering** (temporal, rain, lags, rolling) **+ spatial (19 cols)** | **Build the predictor columns** |
| **➜ 12** | **Feature selection** (filter → MI → model → recursive) | **Choose which columns to use** |
| 13 | Leakage audit | Prove nothing uses the future |
| 14–16 | Experimental table → splits → baselines | Freeze the table and set the bar |

Per the master workflow ledger (`final workflow.md` lines 98–100): Phases 10–11 =
"Feature engineering (temporal, rain, lags, rolling) + spatial features (19 columns)"; Phase 12 =
"Feature selection — FULL 62 / LINEAR 35 / SELECTED 10, MI step 19". Nothing under `data/raw/`
was modified — the raw inputs are read-only and everything is derived into `data/processed/`.

---

## 2. Words you'll meet

| Word | Meaning |
|---|---|
| **Feature (column)** | An input the model may use. |
| **Lag** | Yesterday's value (`lag1` = t−1), the day before (`lag2`), etc. Only the **past**. |
| **Rolling / window** | A summary over the last N days (max, sum, count). |
| **`wl_`** | Water-level features. **`rain_`/`precip_`** rainfall. **`tmax/tmin`** temperature. **`doy_`** day-of-year (season). |
| **Danger level** | The officially published level at which a station is considered flooded. |
| **`nb3`/`nb5`** | The **nearest 3 / nearest 5 gauges** (not "3 km"!) — "neighbourhood" context. |
| **`up_`** | The **upstream** gauge on the same river (higher elevation). |
| **Banned column** | A column that must never be a feature (ID, name, date, label, or label-window count). |
| **Constant** | A column with the same value everywhere — carries no information. |
| **VIF** | Variance Inflation Factor — how much a feature is already explained by the others (>10 = redundant for linear models). |
| **MI (mutual information)** | A non-linear "how informative is this column" score. |
| **Permutation importance** | Shuffle one column, measure the score drop → how much the *model* relies on it. |
| **Filter / model / recursive** | Three families of selection: statistics-only, use-a-model, or add/remove one at a time. |
| **D10** | Decision: the workflow's body-phase order is binding (Phase 12 before 13, etc.). |

---

## 3. Phase 10–11 — Feature engineering

**Goal:** give the model a fair, leakage-safe picture of one station on one day. The result is
the model-ready table **200,665 × 73**, 133 stations, 2017-01-06 … 2026-10-05.

### 3.1 The 62 features the model ends up with, by family

| Family | n | Columns (from `feature_sets.json` FULL) |
|---|---:|---|
| Text / place names | 5 | `river, basin, division, district, upazilla` |
| Location & danger | 3 | `lat, lon, danger_level` |
| Observation | 1 | `n_readings` |
| Water level — today's levels | 5 | `wl_mean, wl_max, wl_min, wl_first, wl_last` |
| **Static geography (Group A)** | **7** | `elevation_m, slope_deg, dist_to_coast_km, dist_river_km, dist_major_river_km, river_density_10km_km_per_km2, n_river_lines_10km` |
| Water level — lags | 10 | `wl_last_lag1…7`, `wl_max_lag1…3` |
| Water level — rises | 2 | `wl_rise_1d, wl_rise_3d` |
| Water level — danger-relative | 2 | `wl_dist_to_danger, wl_pct_danger` |
| Water level — rolling | 3 | `wl_max_roll7, wl_max_roll30, wl_exceed_days_30` |
| Rainfall windows | 7 | `precip_mm, rain_1d, rain_3d, rain_7d, rain_14d, rain_30d, rain_max_7d` |
| Rainfall — wetness | 1 | `wet_days_7` |
| Temperature | 2 | `tmax_c, tmin_c` |
| Calendar / season | 2 | `doy_sin, doy_cos` |
| **Neighbourhood (Group B)** | **8** | `nb3_rain_7d, nb3_rain_14d, nb3_wl_pct_danger, nb5_rain_7d, nb5_rain_14d, nb5_wl_pct_danger, nb3_n_wl_danger, nb5_n_wl_danger` |
| **Upstream (Group C)** | **4** | `up_rain7d_lag1, up_rain7d_lag2, up_wl_pct_danger_lag1, up_wl_pct_danger_lag2` |

Total = **62** (5 text + 57 numeric).

*Why the families:* a flood is *local level* (where the water is now) + *recent rain*
(what is feeding it) + *upstream* (what is coming) + *calendar* (monsoon) + *terrain*
(will it drain). Each family encodes one of those stories.

### 3.2 The water-level family (the engine of the whole model)

Today's stage readings (`wl_mean/max/min/first/last`) are summarised from the day's 3-hourly
observations. From them we build the three ideas that actually predict flooding:

- **Level vs danger:** `wl_dist_to_danger` (= `danger_level − wl_last`, negative means already
  over) and `wl_pct_danger` (= `wl_last / danger_level`). These are the top-2 features.
- **Movement:** `wl_rise_1d`, `wl_rise_3d` — a rising river matters even below danger.
- **Persistence over a month:** `wl_max_roll7`, `wl_max_roll30`, and `wl_exceed_days_30`
  (how many of the last 30 days were at/above danger). The 0-day vs 11-day split between
  non-flood and flood days (Phase 8 table) is exactly this column.
- **Lags:** `wl_last_lag1…7` and `wl_max_lag1…3` give the model the recent trajectory.

### 3.3 The rainfall family

`precip_mm` (today) plus accumulating windows `rain_1d/3d/7d/14d/30d`, the burst measure
`rain_max_7d`, and `wet_days_7` (how many of the last 7 days were wet). Rainfall is the
*second* layer of signal (Phase 8: `rain_30d` univariate AUC 0.702, below water level but
clearly real). The windows exist because *accumulation* (a soaked catchment) matters more
than a single day.

### 3.4 Calendar

`doy_sin` / `doy_cos` encode the day of year as a smooth cycle, so the model knows it is
monsoon without a hard June/September cliff. (Phase 6: monsoon floods 5.4× more often.)

### 3.5 Group A — static geography (7 columns, Phase 11)

Built once by `scripts/build_spatial_features.py`, joined per station. **0 NaN, 133/133
stations covered.**

| Column | mean | min | max | Source |
|---|---:|---:|---:|---|
| `elevation_m` | 15.45 | −10.0 | 70.0 | SRTM 1″ nearest sample |
| `slope_deg` | 3.07 | 0.00 | 13.20 | SRTM 3×3 finite difference |
| `dist_to_coast_km` | 8.26 | 0.05 | 100.87 | SRTM land/ocean boundary (~1 km) |
| `dist_river_km` | 0.41 | 0.001 | 7.82 | nearest OSM river line |
| `dist_major_river_km` | 1.48 | 0.003 | 24.32 | nearest line ≥ 20 km (trunk proxy) |
| `river_density_10km_km_per_km2` | 0.245 | 0.015 | 0.633 | river length in a 10 km disc ÷ area |
| `n_river_lines_10km` | 17.96 | 1 | 90 | count of river lines in the disc |

*Honest caveats:* the coast distance is a ~1 km approximation; "major river" is a *length*
proxy (OSM names are Bengali, so no name join); `elevation_m = −10` is a real coastal char
depression in the delta, not an error.

### 3.6 Group B — neighbourhood context (8 columns, Phase 11)

`add_neighbourhood()` — the **nearest-k gauges' state on day t**.

| Design choice | Value | Why it matters |
|---|---|---|
| Candidate pool | nearest **15** gauges by haversine (self excluded) | keeps the search local |
| `k` | **3 and 5** | two spatial scales, cheap |
| Membership | **resolved per day** — the nearest *k* that actually reported that day | the network grows 85 → 133 (risk R7); a fixed list would silently use stations that did not exist yet |
| Rain signals | `nb{3,5}_rain_7d`, `nb{3,5}_rain_14d` | from the weather record, so a WL-quiet day still carries the areal rain signal |
| Water-level signals | `nb{3,5}_wl_pct_danger` | mean wl/danger of the nearest-k reporting gauges |
| Count signal | `nb{3,5}_n_wl_danger` | how many of the nearest-k are **already at/above** danger on t |

Coverage: neighbour rain **0 % NaN**; `nb*_wl_pct_danger` **0.35 % NaN** (no pool gauge
reported); counts **0 % NaN** (an empty set means 0, which is correct, not missing).

### 3.7 Group C — upstream propagation (4 columns, Phase 11)

`add_upstream()` — the nearest same-river gauge at clearly **higher** elevation.

- *Why not latitude?* The lower Ganges/Padma flows **west**; a latitude rule would point the
  wrong way. Rule instead: same `river` string, `elevation_m` higher by ≥ 0.5 m, nearest by
  haversine, within **250 km**.
- Result: **56 downstream stations** mapped (50 appear in the model-ready table). The other 85
  have **no eligible upstream partner** → `up_*` is NaN **by design** (67–70 % of rows), never
  imputed at build time.
- Features are lagged: `up_*_lag1` and `up_*_lag2` — upstream's state on **t−1 and t−2**, i.e.
  strictly *before* the label window opens. This is the physically correct "flood wave coming"
  signal.

### 3.8 Leakage position (the argument)

| Group | Information used | Available at issue instant (24:00 UTC of t)? |
|---|---|---|
| Static (A) | coordinates + terrain | yes — time-invariant |
| Neighbourhood rain (B) | neighbour rain on days ≤ t | yes |
| Neighbourhood WL (B) | neighbour stage **on day t** | yes — same instant as the station's own `wl_*` |
| Upstream (C) | upstream state on **t−1, t−2** | yes — strictly past |

Nothing in A/B/C touches `obs_days_*`, the label window, or any day > t. This is only the
*argument*; the formal ruling belongs to **Phase 13** (leakage audit), which must add all 19
columns to the whitelist and rule on each — especially the observation-process signals
(`nb*_n_wl_danger`, `n_readings`).

### 3.9 A defect found and fixed in Phase 11 (worth remembering)

| | |
|---|---|
| Symptom | `audit_dataset.py` crashed: `KeyError: 'nb3_wl_pct_danger'`; model-ready had **71** columns instead of 73 |
| Root cause | `NEIGHB_SIGS` *declared* the column and the data dictionary documented it, but `add_neighbourhood()` only ever computed the rain means and the count — the wl-ratio line was never written |
| Why it hid | the `keep` filter was `df[[c for c in keep if c in df.columns]]` — a **declared-but-missing column was silently dropped**, so the build "succeeded" |
| Fix | (a) compute `nb{3,5}_wl_pct_danger` via the same `resolve()` path as the rain neighbours; (b) the build now **raises** if any declared column was not produced; (c) removed a dead line and an unused dict |
| Proof of no side-effects | labels **byte-identical** to the pre-fix build (`label_24h` 200,559/9,982 etc.) |

**Lesson for the whole project:** silent column-dropping is more dangerous than a crash. The
build now fails loudly instead.

---

## 4. Phase 12 — Feature selection

**Goal:** from 73 columns, decide exactly which the models may see, using **four independent
lenses**, and freeze the answer in a file. Test years (2023+) were **never read** by any
selection decision.

### 4.1 Admissible pool — arithmetic, not a re-invented whitelist

> 73 columns − **10 banned** (`station_id, name, date, source`; the three `label_*`; the three
> `obs_days_*`) = **63 approved** → `day_observed` is **constant** (finding G3) → **62 usable
> features** (5 text, 57 numeric).

The `obs_days_*` columns are banned because they count observations **inside the label
window** — using them would leak the answer.

### 4.2 The provisional split (needed by the selection steps)

| part | dates | rows | role |
|---|---|---:|---|
| train | 2017-01-06 → 2021-12-31 (2018 absent, FFWC gap) | 77,571 | fit encoder/imputer, MI, models |
| val | 2022-01-01 → 2022-12-31 | 21,920 | **all** importance/selection decisions |
| test | ≥ 2023-01-01 | 101,174 | **untouched** |

Base rates show the drift already: `label_24h` **9.60 %** (train) → **5.08 %** (val) →
**1.41 %** (test). Preprocessing (frequency encoding for the 5 text columns — deliberately
**not** target encoding, which would leak; median imputation) is fitted on **train only**.

### 4.3 Step 19 — Mutual information (the "MI step")

`mutual_info_classif`, k=5, train rows only, ranked by mean across the three horizons:

| rank | feature | MI 24h | MI 48h | MI 72h | mi_mean |
|---:|---|---:|---:|---:|---:|
| 1 | `wl_dist_to_danger` | 0.2560 | 0.2565 | 0.2533 | 0.2553 |
| 2 | `wl_pct_danger` | 0.2553 | 0.2539 | 0.2518 | 0.2537 |
| 3 | `wl_exceed_days_30` | 0.1541 | 0.1538 | 0.1533 | 0.1537 |
| 4 | `rain_max_7d` | 0.1204 | 0.1296 | 0.1368 | 0.1290 |
| 5 | `nb5_n_wl_danger` | 0.1219 | 0.1222 | 0.1228 | 0.1223 |
| 6 | `nb3_wl_pct_danger` | 0.1212 | 0.1229 | 0.1224 | 0.1221 |
| 7 | `nb3_n_wl_danger` | 0.1196 | 0.1206 | 0.1189 | 0.1197 |
| 8 | `nb5_wl_pct_danger` | 0.1133 | 0.1145 | 0.1147 | 0.1142 |
| 9 | `lon` | 0.0711 | 0.0752 | 0.0794 | 0.0752 |
| 10 | `river_density_10km_km_per_km2` | 0.0696 | 0.0750 | 0.0802 | 0.0749 |

*Read it:* water-level *state* first, then rainfall bursts, then **neighbourhood danger
counts** — the spatial groups earn their place here.

### 4.4 Step 18b — Redundancy collapse + VIF → the LINEAR set

- **Redundancy:** Spearman |ρ| ≥ 0.95 on train, walked **in MI order** so the survivor of each
  near-duplicate group is the higher-MI feature → **23 of 62 collapsed**. Examples:
  `nb5_wl_pct_danger` → `nb3_wl_pct_danger` (0.978); `rain_1d` → `precip_mm` (**1.000**);
  `wl_max/wl_last/wl_first/wl_mean` → `wl_min` (0.997–0.999).
- **VIF:** iterative removal of the worst offender while VIF > 10 → **4 removed**
  (`wl_min`, `up_rain7d_lag1`, `danger_level`, `rain_7d`).
- Result: **LINEAR = 35 features** — the set the logistic-regression baseline must use.
  (Phase 9 showed unpruned logreg is numerically unstable.) **Tree models keep all 62.**

### 4.5 Step 20 — Model-based importance vs the persistence bar (D7)

LightGBM (400 trees, lr 0.05, 64 leaves) on the 62 features, scored on **2022 only**:

| horizon | persistence (`wl_pct_danger` alone) | LightGBM 62 feats | ratio |
|---|---:|---:|---:|
| `label_24h` | 0.9064 | 0.9591 | 1.058 |
| `label_48h` | 0.8761 | 0.9442 | 1.078 |
| `label_72h` | 0.8446 | 0.9258 | 1.096 |

Permutation importance (5 repeats, mean PR-AUC drop), top 5:

| feature | perm 24h | perm 48h | perm 72h | mean |
|---|---:|---:|---:|---:|
| `wl_dist_to_danger` | 0.6555 | 0.5377 | 0.3847 | **0.5259** |
| `wl_pct_danger` | 0.0174 | 0.0262 | 0.0372 | 0.0269 |
| `wl_rise_1d` | 0.0162 | 0.0169 | 0.0183 | 0.0172 |
| `precip_mm` | 0.0043 | 0.0086 | 0.0089 | 0.0073 |
| `wl_exceed_days_30` | 0.0027 | 0.0029 | 0.0015 | 0.0024 |

Tree split-gain agrees directionally (`wl_dist_to_danger` = **0.77** of total gain). Rank
agreement across methods (Spearman on the 62 ranks) is only moderate: **perm vs gain 0.541,
MI vs gain 0.309, perm vs MI 0.087** — expected, because MI is a *filter* criterion and
perm/gain are *model* criteria. This disagreement is precisely why the workflow demands
multiple lenses rather than one.

### 4.6 Step 21 — Recursive/sequential selection → SELECTED 10

Top-k by permutation importance, refit on train, scored on val 2022, mean over horizons:

| k | 24h | 48h | 72h | mean |
|---:|---:|---:|---:|---:|
| 5 | 0.9426 | 0.9233 | 0.9130 | 0.9263 |
| **10** | 0.9543 | 0.9389 | 0.9283 | **0.9405** |
| 15 | 0.9566 | 0.9427 | 0.9304 | 0.9432 |
| 20 | 0.9589 | 0.9439 | 0.9273 | 0.9434 |
| 30 | 0.9598 | 0.9438 | 0.9320 | **0.9452 (best)** |
| 62 | 0.9591 | 0.9426 | 0.9258 | 0.9425 |

**Documented rule (not eyeballed):** smallest k within 0.01 of the best mean → best is 0.9452
at k=30, threshold 0.9352 → **k\* = 10**. k=10 also *beats* the full 62-feature model at 48h/72h
— the flat tail is noise, not signal.

**SELECTED (10):** `wl_dist_to_danger, wl_pct_danger, wl_rise_1d, precip_mm,
wl_exceed_days_30, nb3_wl_pct_danger, wl_last, danger_level, wl_rise_3d, lon`

### 4.7 The frozen outputs

`data/processed/feature_sets.json` records everything (banned list, thresholds, split, rule,
counts, val PR-AUC):

| set | n | who uses it |
|---|---:|---|
| **FULL** | 62 | tree models (LightGBM / XGBoost / RF) |
| **LINEAR** | 35 | logistic-regression baseline |
| **SELECTED** | 10 | compact/interpretable model + cross-check |

A late note in that file records Phase 13's ruling: **all 62 FULL columns ADMITTED**
(one `ADMIT-WATCH`: `n_readings`) → the sets were **not** changed and the notebook needed no re-run.

---

## 5. Handover to Phase 13

1. The leakage audit must rule explicitly on **all 62 FULL columns**, especially
   `wl_dist_to_danger`, `wl_pct_danger`, `wl_last`, `wl_exceed_days_30` (all use day-t stage vs
   `danger_level`) and the neighbourhood observation-process signals `nb*_n_wl_danger`, plus
   `n_readings` (Phase 6 finding E7).
2. `obs_days_*` and the labels stay banned (already enforced by `assert`).
3. If Phase 13 demotes/promotes any column, **re-run the selection notebook** so
   `feature_sets.json` stays authoritative.

**Next in this series:** Phase 13–16 — leakage audit, the immutable experimental table, the
split design, and the baselines (the six-model bar).

---

## 6. Problems & solutions

| Problem | Solution |
|---|---|
| Columns that would leak the answer | `obs_days_*` and the labels **banned by rule**; 10 columns removed up front. |
| Text place names can leak via target encoding | Frequency encoding fitted on **train only** — never target encoding. |
| The network grows (85 → 133) → fixed neighbour lists would use future stations | Neighbour membership **resolved per day** from the reporting pool. |
| Neighbour rainfall might vanish on WL-quiet days | Rain neighbours built from the **weather record**, not the WL matrix. |
| "Upstream" by latitude points the wrong way (Ganges flows west) | Upstream = same river, higher elevation, within 250 km, and **lagged 1–2 days**. |
| 70 % NaN on `up_*` could look like a bug | Structural and by design; trees handle NaN; the linear model needs an imputer **plus a "no upstream gauge" indicator**. |
| A declared column (`nb3_wl_pct_danger`) was silently dropped | Build now **raises** if a declared column is missing; root cause fixed; labels proven byte-identical. |
| One importance lens could mislead | Four lenses; disagreements (perm↔MI = 0.087) documented rather than hidden. |
| Pick k by eye | Documented rule: smallest k within 0.01 of best → **k\* = 10**. |
| 23/62 near-duplicate pairs break linear models | Redundancy collapse + VIF pruning → **LINEAR 35** for logreg; trees keep all 62. |
| "ML beats persistence" overstated | The margin is stated plainly: **×1.06–1.10**, driven by rainfall-rise + spatial terms. |

---

## 7. Files referenced

| File | Purpose |
|---|---|
| `reports/phase11_spatial_features.md` | Phases 10–11 — the 19 spatial columns and the defect fix |
| `reports/phase12_feature_selection.md` | Phase 12 — the four selection lenses and outputs |
| `data/processed/feature_sets.json` | Authoritative FULL / LINEAR / SELECTED lists + thresholds |
| `data/processed/feature_selection_mi.csv` | Full MI table (step 19) |
| `data/processed/feature_selection_permimportance.csv` | Full permutation table (step 20) |
| `data/processed/station_spatial_features.csv` | Group A per-station values (153 stations) |
| `figures/phase12_mi.png`, `phase12_perm_importance.png`, `phase12_selection_curve.png` | Visual evidence |
| `data/processed/flood_dataset_model_ready.csv` | The 200,665 × 73 table |
| `scripts/build_spatial_features.py`, `src/flood/features/build_dataset.py` | The builders |

---

## 8. Verdict

- **Phase 10–11:** the table grew **54 → 73** columns with temporal/rain/level features plus
  **19 spatial** columns (static terrain, per-day nearest-k neighbours, lagged upstream). All
  are leakage-safe by construction (past or same-instant only).
- **Phase 12:** from 73 columns we froze **FULL 62 / LINEAR 35 / SELECTED 10**, using MI as a
  filter, redundancy + VIF for the linear set, permutation importance and a backward-elimination
  curve for the compact set.
- **The honest headline:** one feature (`wl_dist_to_danger`) is ~77 % of the model, and the full
  model beats the one-feature persistence bar by only **×1.06–1.10**. Everything after this
  point is a fight for a small, real margin.

**Bottom line:** the modelling table and its input lists are now locked, documented, and
leakage-reasoned — and the project has stated plainly that its edge over a trivial baseline is
modest, which sets honest expectations for every later phase.
