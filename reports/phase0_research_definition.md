# Phase 0 — Research problem definition

**Status:** COMPLETE · **Written:** 2026-10-07 · **Applies to:** `data/processed/flood_dataset_model_ready.csv`
**Governing docs:** `Proposal.docx` (supervisor requirements), `flood prediction workflow.md` (62-stage master workflow)

Phase 0 fixes *what we are predicting, at what resolution, over what horizon, and under what
information rules*, before any further data work or modelling. Everything downstream inherits
this document. Anything that contradicts it is a bug.

---

## 1. Research question

Verbatim from `Proposal.docx`:

> **What is the probability of flooding at a particular location within the next 24/48/72 hours?**

Honest scope statement, required by the proposal's warning against relabelling a
classification problem as forecasting:

* This is a **probabilistic lead-time forecast**, not a static susceptibility map and not a
  contemporaneous flood/no-flood classifier.
* Each prediction is a **calibrated probability** `p` of exceeding the FFWC danger level
  within a stated future window, later mapped to risk bands (Phase 23).
* The forecast is produced from **antecedent hydrology + observed rainfall only**. There is
  no NWP (weather-model) input, so the correct claim is:

  > *empirical lead-time forecast from antecedent hydrology and observed rainfall*

  not "weather-model-driven prediction". If NWP is added later (Phase 29/optional), the claim
  may be upgraded and must then be reported separately.

## 2. Prediction unit

| Item | Definition |
| --- | --- |
| One observation = | one FFWC station × one issue day |
| Issue grid | daily, UTC |
| Rows in model-ready table | 200,665 station-days, 133 stations (200,671 originally; −2 F3, −4 F8 → D4/D9) |
| Full calendar matrix | 357,250 station-days, 141 stations (all days, observed or not) |
| Location resolution | station point (+ its district/upazila/river/basin attributes) |
| Not used | grid cell, district-day, upazila-day aggregates |

Station is the finest location for which **observed** hydrology exists, so it is the only
honest prediction unit. District/upazila are attributes of a station, used later for the
location-selection UI — a prediction is always *scored at a station*.

## 3. Prediction horizon and issue-time convention

| Lead `h` | Label column | Forecast window |
| --- | --- | --- |
| 24 h | `label_24h` | days `t+1` |
| 48 h | `label_48h` | days `t+1 … t+2` |
| 72 h | `label_72h` | days `t+1 … t+3` |

**Issue instant = 24:00 UTC of day `t`.** All features on row `t` use only information from
day `t` and earlier (end-of-day aggregates), so the forecast is issued *after* day `t` closes
and covers exactly the next `h` calendar days → effective lead time **0–24 h / 0–48 h /
0–72 h**.

Two consequences, stated openly:

1. Horizon is **calendar-day aligned**, not a rolling 24 h from an arbitrary wall-clock time.
2. A stricter variant exists (features through `t-1`, window `t … t+h`, i.e. lead 24–96 h).
   Not adopted now; may be tested as a robustness variant in Phase 26.

## 4. Label contract — NON-NEGOTIABLE

Implemented in `src/flood/features/build_dataset.py`, independently re-verified by
`scripts/audit_dataset.py` (check A5) and `scripts/verify_model_ready.py`.

```
y_h = 1   if ANY observed reading in (t, t+h] reaches the station's official FFWC danger level
y_h = 0   if EVERY day in that window has >= 3 readings AND a known danger level AND none reaches it
y_h = NaN otherwise   (window not fully observed, or station has no official danger level)
```

Rules:

1. **Missing is never converted to 0.** An unknown outcome stays unknown (the fatal flaw of
   the discarded flood-v3 dataset: 16,051 of 20,544 labels were assumed).
2. Danger level comes **only** from the official station master (`ffwc_stations.csv`);
   per-reading `dangerlevel` fields in raw CSVs are unreliable and are overwritten.
3. A positive is never vetoed by an unobserved day elsewhere in its window.
4. Stations without an official danger level (8 of 141) can never receive a label — their
   rows carry NaN labels permanently.
5. `obs_days_24/48/72h` describe the label window and are **label-window information →
   banned as features** (Phase 13 leakage audit).
6. "Observed day" = ≥ 3 readings (3-hourly source), a fixed contract constant
   (`MIN_READINGS_PER_DAY = 3`).

Current label statistics (model-ready, 2017‑01‑06 → 2026‑10‑05):

| Label | n | positives | base rate | NaN | imbalance (0:1) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_24h` | 200,559 | 9,982 | 4.98 % | 106 | 19.1 : 1 |
| `label_48h` | 197,748 | 11,241 | 5.68 % | 2,917 | 16.6 : 1 |
| `label_72h` | 195,174 | 12,340 | 6.32 % | 5,491 | 14.8 : 1 |

Rare-but-not-degenerate ⇒ **PR-AUC is the primary metric** (Phase 20), accuracy is not.

## 5. Information boundary — what may be known at issue time `t`

**LEGAL (features):** everything computable at 24:00 UTC of day `t`

| Group | Columns (63 approved) |
| --- | --- |
| Station identity/geo | `river basin division district upazilla lat lon danger_level` |
| Observation metadata | `n_readings day_observed` |
| Hydrology today | `wl_mean wl_max wl_min wl_first wl_last` |
| Hydrology lags | `wl_last_lag1..7 wl_max_lag1..3` |
| Hydrology derived | `wl_rise_1d wl_rise_3d wl_dist_to_danger wl_pct_danger wl_max_roll7 wl_max_roll30 wl_exceed_days_30` |
| Rain/temperature | `precip_mm rain_1d rain_3d rain_7d rain_14d rain_30d rain_max_7d wet_days_7 tmax_c tmin_c` |
| Season (narrow) | `doy_sin doy_cos` |
| Static geography (Phase 11) | `elevation_m slope_deg dist_to_coast_km dist_river_km dist_major_river_km river_density_10km_km_per_km2 n_river_lines_10km` |
| Neighbourhood (Phase 11) | `nb3_rain_7d nb5_rain_7d nb3_rain_14d nb5_rain_14d nb3_wl_pct_danger nb5_wl_pct_danger nb3_n_wl_danger nb5_n_wl_danger` |
| Upstream (Phase 11) | `up_rain7d_lag1 up_rain7d_lag2 up_wl_pct_danger_lag1 up_wl_pct_danger_lag2` |

Phase-11 groups were **pre-audit admitted** on the argument in
`reports/phase11_spatial_features.md` §4 (all computed at or before `t`); the formal
whitelist ruling is stage 19 and was issued on 2026-10-08 — **all 19 columns ADMIT**,
`nb*_n_wl_danger` evidenced separately (`reports/phase13_leakage_audit.md` §5,
`data/processed/feature_whitelist.csv`).

**ILLEGAL (must never appear in a feature matrix):**

| Category | Columns / patterns | Reason |
| --- | --- | --- |
| Targets | `label_24h label_48h label_72h` | the answer |
| Label-window info | `obs_days_24h obs_days_48h obs_days_72h` | describes the future window |
| Identity/date keys | `station_id name date source` | station memorisation + temporal leakage |
| Future-derived | anything using days `> t` | future information |
| Calendar leakage | year index, `bad_years_boost`, month-as-integer (as in flood-v3) | model reads the calendar, not the hydrology |
| Full-matrix-only rows | rows with `day_observed = False` | no observation ⇒ nothing to forecast from |

`doy_sin/doy_cos` are deliberately the *only* calendar signal, kept narrow on purpose.

## 6. What the current data supports (factual base)

* **Source:** FFWC API (`api.ffwc.gov.bd`), 3-hourly observed water level, harvested
  2014‑01‑01 → 2026‑10‑06 (stride 35 d, 20,655 calls) → 1,052,044 readings,
  992 station-year CSVs, 141 stations with data, 153 in the master.
* **Rain/temperature:** NASA POWER / MERRA‑2 point series at each station (reanalysis,
  *modelled* precipitation — a documented caveat, not observed rain).
* **Archive gap:** 2018 returns empty from FFWC for every station — verified as a genuine
  gap in their own archive (31,390 empty calendar rows in the matrix, 0 readings).
  Evaluation must never bridge 2017 → 2019 across that hole.
* **Time coverage of labelled rows:** 2017 (85 stations) → 2026 (133 stations).
  Station network grows over time; early years are not spatially comparable to late years.
* **Monsoon (JJAS):** 55.5 % of rows; `label_72h` base rate **9.79 % in monsoon vs 1.83 %
  outside** — the phenomenon is overwhelmingly monsoon, as the proposal expects.

## 7. Known risks identified in Phase 0 (must be handled downstream)

| # | Risk | Evidence | Where handled |
| --- | --- | --- | --- |
| R1 | **Base-rate drift over time** | `label_72h` positives per year: 2,831 (2017) → 276 (2023) → 500 (2025); rate 16.2 % → 1.2 % → 1.8 % | temporal splits report per-year metrics; never pool years blindly (Phases 15, 20, 26) |
| R2 | **Tidal/estuarine stations dominate** | Mongla 83.8 %, Khulna 52.3 %, Daulatkhan 40.4 % base rate | per-station metrics + tidal subgroup; do not treat as labelling errors. **Re-measured Phase 16 §10: 73.2 / 39.5 / 26.8 % — W-R2, the 83.8 % figure is not reproducible from the Phase-14 table; pending phase0 review** |
| R3 | **13 stations have 0 positives** (≥200 labelled rows) | Boalmari, Sonapur, Lakshmipur, Haripur … | keep (legitimate); report macro vs micro averages. **Re-measured Phase 16 §10: 14 stations (Dhaka, Faridpur, Bogura … added) — W-R3, pending phase0 review** |
| R4 | **Class imbalance** | 15–19 : 1 | Phase 19; PR-AUC primary; oversampling only inside the training fold. **Phase 16 §10 confirmed the whole-table range and found the split-specific ratios: 9.4:1 train → 18.6:1 val → 70:1 test (24 h)** |
| R5 | **Rainfall is reanalysis, not observed** | NASA POWER / MERRA‑2 | stated in every report; ablation with/without rain (Phase 38) |
| R6 | **2018 gap** | 0 readings in the whole year | split around it; never interpolate across it |
| R7 | **Growing station network** | 85 → 133 stations | station-out validation must not let network growth leak into "unseen location" claims |
| R8 | **Geospatial data partially sourced 2026-10-08** | SRTM 1" (40 tiles), OSM river network (6,792 centrelines), geoBoundaries ADM1/ADM2 → `data/raw/geo/` (manifest `geo/README.md`); soil / land cover / independent flood extents still absent | Phase 11 **engineering complete** (19 spatial columns in the dataset); the soil/landcover *feature group* stays behind R8 |

## 8. Mapped workflow — all 62 stages

Status key: **DONE** · **PARTIAL** (started, needs redo/finish) · **TODO** · **BLOCKED**
(needs data we do not have) · **N/A**

*The `#` column is the **62-stage checklist** numbering (a labelling convenience for this
table). Execution order follows the **body Phase sequence** per **D10** — use the crosswalk
below to translate; when the two disagree on order, D10 wins.*

> **Binding structure (decision D10, 2026-10-08):** the workflow file numbers things twice —
> body `Phase N` headings with their own `## NN.` step numbers, *and* a 62-stage "clean
> master checklist" at the end. **The body Phase order is law: Phase 0 → … → Phase 40, in
> numeric order.** Every reference in this project cites **both**, so nothing can drift:
>
> | Body (binding) | Steps | Checklist # | Topic |
> | --- | --- | --- | --- |
> | Phase 9 | 14 | 18 | multicollinearity / redundancy |
> | Phase 10–11 | 15–17 | 21–23 | feature engineering (temporal, rain, lag, spatial) |
> | **Phase 12** | **18–21** | **24–25** | feature selection: statistical filter → **MI (step 19)** → model-based → RFE |
> | **Phase 13** | **22–24** | **19–20** | critical leakage audit |
> | Phase 14 | — | 26 | experimental (model-ready) table |
> | Phase 15 | 25–27 | 27–28 | temporal / spatial / monsoon splits |
> | Phase 16 | 28 | 29–30 | baselines + class-imbalance analysis |
>
> The checklist's own "defensible order" (leakage before selection) is **superseded** by D10:
> Phase 12 runs before Phase 13. Nothing is lost — Phase 13 still lands before Phase 14/15,
> i.e. before any split or model, which is what the workflow actually demands.

### Understand & validate (stages 1–12)
| # | Stage | Status | Note |
| --- | --- | --- | --- |
| 01 | Define research question | DONE | §1 |
| 02 | Define prediction target | DONE | §4 |
| 03 | Define prediction horizon | DONE | §3 |
| 04 | Identify prediction unit | DONE | §2 |
| 05 | Collect and document datasets | DONE | §6 inventory; geo acquired (SRTM/OSM/geoBoundaries, 2026-10-08); soil via MI-1c POWER soil wetness 133/133 + MI-1a ERA5-Land soil moisture 133/133; landcover via ESA WorldCover (`data/processed/esa_worldcover_features.csv`); independent flood extents remains R8 (documented limitation, not a blocker) |
| 06 | Load raw data | DONE | `build_dataset.py` + notebook 02 (S06a–d) |
| 07 | Understand dataset structure | DONE | notebook 02 — 54 columns in 10 groups, identities recomputed (S07a–h) |
| 08 | Validate timestamps and locations | DONE | audit A8/A10 |
| 09 | Investigate flood labels | DONE | §4 + audit A5/A7 + `verify_model_ready.py` + notebook 02 (S09a–d) |
| 10 | Missing-value analysis | DONE | notebook 02 (Q10a–d) — F1 regression guard + every NaN cause named |
| 11 | Duplicate analysis | DONE | audit A3 + notebook 02 (Q11a–d) — 0 dups, keys unique, monotone |
| 12 | Outlier / data-quality analysis | DONE | notebook 02 (Q12a–l, Q14–Q15) — **43/43 checks**; findings F2–F8, G1–G4 → `reports/phase2_3_data_quality.md`; **F3 and F8 fixed same-day (D4/D9)** |

### Explore (stages 13–18)
| # | Stage | Status | Note |
| --- | --- | --- | --- |
| 13 | EDA | DONE | `notebooks/03_eda.ipynb` — 20/20 checks, `reports/phase5_eda.md` |
| 14 | Temporal analysis | DONE | monthly/yearly/JJAS frequencies, episodes; **D5** drift decomposition, **D6** missingness test |
| 15 | Spatial analysis | DONE | 133-station rates + division rates; choropleth now unblocked (ADM1/ADM2 in `geo/admin`) → Phase 31 |
| 16 | Target–feature analysis | DONE | 38 features × 3 horizons (AUC/KS/medians/MI) + **D7** persistence baseline per year |
| 17 | Correlation analysis | DONE | Pearson + Spearman matrices, label correlations, heatmap |
| 18 | Multicollinearity / redundancy | DONE | 160 pairs \|ρ\|>0.8, VIF pruning → 17-feature keep list, constants dropped |

### Leakage & experiment definition (stages 19–26)
| # | Stage | Status | Note |
| --- | --- | --- | --- |
| 19 | Data-leakage audit | DONE | `notebooks/05_leakage_audit.ipynb` — 11/11 checks (L1–L11), `reports/phase13_leakage_audit.md`; formal whitelist restored → `data/processed/feature_whitelist.csv` (61 ADMIT · 1 ADMIT-WATCH · 1 ADMIT-CONSTANT · 10 BAN) |
| 20 | Define prediction-time info boundary | DONE | §5 — issue instant 24:00 UTC of `t`; `latest_day_used` column of the whitelist is the machine-readable form |
| 21 | Feature engineering | DONE | core features in `build_dataset.py`; Phase 11 added static + neighbourhood + upstream groups → `reports/phase11_spatial_features.md` |
| 22 | Temporal / lag / rolling features | DONE | lags 1–7, roll 7/30, rain windows |
| 23 | Spatial / geographical features | DONE | static 7 (`station_spatial_features.csv`) + neighbourhood 8 + upstream 4 = 19 columns, built 2026-10-08; soil/landcover group added via MI-1/MI-2 (POWER soil + ERA5-Land soil moisture + ESA WorldCover, `FULL_V2` 110 features) |
| 24 | Feature selection | DONE | Phase 12 — `notebooks/04_feature_selection.ipynb` (13/13 cells), `reports/phase12_feature_selection.md`; sets FULL 62 / LINEAR 35 / SELECTED 10 |
| 25 | Mutual information | DONE | Phase 12 step 19 — MI for 62 features × 3 horizons on train only; used with redundancy collapse, rule-based k*=10 |
| 26 | Model-ready dataset | DONE | 200,665 × 73 source (post-F3 + F8 fixes, D4/D9; +19 spatial columns at Phase 11); **Phase 14 experimental table = 200,665 × 67** (`data/processed/experimental_dataset.csv`, dictionary alongside) |

### Split, model, evaluate (stages 27–46)
| # | Stage | Status | Note |
| --- | --- | --- | --- |
| 27 | Temporal train/val/test split | DONE | Phase 15 — `notebooks/07_split_design.ipynb` (S1–S10 green); train 77,571 / val 21,920 / test 101,174 (frozen), cuts inside 98–131-day dry-season gaps, 7-day embargo flags 0 rows (R1 + R6 respected) |
| 28 | Spatial validation design | DONE | Phase 15 — 5 cohort-balanced station folds (R7), fold 0 = 26 stations frozen out of every training set, 4-fold spatial CV on train years, test reported as trained / new-network / unseen buckets |
| 29 | Class-imbalance analysis | **DONE** | Phase 16 §10 (workflow Phase 19) — `notebooks/08_baseline_models.ipynb` I1–I5, **59/59 checks green**; ratios 9.4:1 train → 18.6:1 val → 70:1 test (24 h), per-year 4.9–83:1, top-10 stations = 38.9 % of positives, accuracy gain ≤ 0.045, PR-AUC primary; evidence `data/processed/class_imbalance.json`, `figures/phase16_class_imbalance.png`; **W-R2/W-R3 re-measured, see §7** |
| 30 | Baseline models | **DONE** | Phase 16 — `scripts/baseline_model.py` rewritten (early-stopping-on-test flaw gone), `scripts/phase16_run.log` exit 0 / 312 s, `reports/phase16_baseline.md` §1–§12; notebook S1–S7 verify 59/59 checks; logreg_linear beats the D7 persistence bar by only +2.1…+4.3 % on val 2022 |
| 31 | Candidate ML models | TODO | logreg / LightGBM / XGBoost justified in plan §6 |
| 32 | Hyperparameter tuning | TODO | |
| 33 | Model comparison | TODO | |
| 34 | ROC-AUC / PR-AUC | TODO | PR-AUC primary |
| 35 | Precision / Recall / F1 | TODO | |
| 36 | Confusion matrix | TODO | |
| 37 | False-negative analysis | TODO | proposal requirement |
| 38 | Probability calibration | TODO | Brier + ECE + reliability diagram |
| 39 | Threshold optimization | TODO | cost of FN ≫ FP |
| 40 | Temporal evaluation | TODO | |
| 41 | Spatial evaluation | TODO | |
| 42 | Monsoon evaluation | TODO | JJAS per §6 |
| 43 | Robustness analysis | TODO | |
| 44 | Error analysis | TODO | |
| 45 | Ablation study | PARTIAL | weather-only vs +hydrology vs +geography possible (static spatial features now exist); +soil/landcover group still blocked by R8 |
| 46 | Final model selection | TODO | |

### Explain, forecast, ship (stages 47–62)
| # | Stage | Status | Note |
| --- | --- | --- | --- |
| 47 | Explainable AI / SHAP | TODO | needs `shap` install |
| 48–50 | 24 / 48 / 72-hour prediction | TODO | three horizons, same protocol |
| 51 | Risk-level definition | TODO | must be derived from validation, not arbitrary quartiles |
| 52 | Location-based prediction | **BLOCKED** | boundaries available now; still needs a served reverse-geocode/API layer |
| 53 | Bangladesh risk map | **BLOCKED** | ADM1/ADM2 in `geo/admin`; needs the Phase 31 map module |
| 54 | Automatic data acquisition | PARTIAL | FFWC ingest exists and is resumable; weather has no auto-refresh |
| 55 | Real-time prediction pipeline | TODO | |
| 56 | Early-warning engine | TODO | prototype label mandatory |
| 57 | Citizen interface | TODO | |
| 58 | Administrator dashboard | TODO | |
| 59 | System testing | TODO | no `tests/` yet |
| 60 | Final scientific evaluation | TODO | §9 checklist of the workflow |
| 61 | Documentation / reproducibility | DONE | phase 0/1/2–3 reports + 4 git commits; F1 fix reproducible from history |
| 62 | Research report / thesis | TODO | |

**Totals:** 28 DONE · 4 PARTIAL · 28 TODO · 2 BLOCKED (60 table rows = 62 stages; stages 48–50 share one row, so 28 stages are TODO).
*Updated 2026-10-08 after Phase 15 (stages 19, 24, 25, 26, 27, 28 closed: Phase 13 leakage
audit 11/11 + whitelist; Phase 14 experimental table 200,665 × 67; Phase 15 split strategy
S1–S10 → `data/processed/split_assignments.csv` + `splits.json`).* 

## 9. Decisions logged (user-confirmed, 2026-10-07)

| # | Decision | Value |
| --- | --- | --- |
| D1 | Monsoon window for all seasonal evaluation (Phase 42) | **JJAS — June, July, August, September** |
| D2 | Headline horizon | **All three horizons (24/48/72 h) reported equally**, no privileged lead |
| D3 | Version control | **git initialised at Phase 0** — code/docs tracked, bulk data ignored |
| D4 | Defect handling (after Phase 2–3 review) | **Fix F3 immediately, not at Phase 15** — clip readings `> 100 m` to NaN in the builder, rebuild, re-audit. *Executed 2026-10-07: 5 readings dropped, 27 false-positive labels removed, audits re-passed.* |
| D5 | EDA must explain the label drift (R1) | **Decompose drift into within-station change vs network composition** (fixed panel of stations vs all stations) before any split design |
| D6 | Missing days must be treated as a potential bias | EDA must compare the **seasonal distribution of observed vs unobserved days** — if gaps cluster in monsoon, the observation process is informative |
| D7 | Mandatory reference model from Phase 16 on | Every model result is reported **next to a one-feature persistence baseline (`wl_pct_danger` alone)**; no "our model works" claim without beating it |
| D8 | Evaluation protocol | **Rolling-origin across years with bootstrap CIs, macro + micro averages**, per-year reporting — never a single global split, never pooled metrics |
| D9 | Standing water-level QC rule (after EDA review, finding F8) | **Spike filter on read**: a reading >10 m from its local 12-h median (gap-aware) is dropped as a sensor artefact — same raw-immutability pattern as F3; mirrored in builder, audit (A2c) and the independent verifier |
| D10 | **Binding workflow structure** (user ruling, 2026-10-08) | Follow the workflow file's **body `Phase N` sequence in numeric order**; the 62-stage checklist is a cross-check, not the driver. Every report reference cites *both* (crosswalk in §8). Consequence: **Phase 12 feature selection (steps 18–21, incl. MI step 19) runs before Phase 13 leakage audit (steps 22–24)**; both still close before Phase 14/15 (experimental table + splits) |

## 10. Definition of "done" for Phase 0

- [x] Research question stated and honestly scoped (§1)
- [x] Prediction unit fixed (§2)
- [x] Horizon + issue-time convention fixed (§3)
- [x] Label contract written as law (§4)
- [x] Information boundary written (§5)
- [x] Dataset facts verified against the actual files (§6)
- [x] Phase-0 risks recorded with evidence (§7)
- [x] All 62 workflow stages mapped (§8)

**Next stage:** Phase 16 closed (checklist 29–30: baseline models + class-imbalance
analysis; `notebooks/08_baseline_models.ipynb` 59/59 green, `reports/phase16_baseline.md`,
D7 persistence bar established, early-stopping flaw fixed). Next: **Phase 17 — candidate ML
models (body step 29 / stage 31)** — logreg / LightGBM / XGBoost justified in plan §6,
every candidate judged against the **D7 persistence bar** under the three Phase-16
protocols and per-season cells, under the stage-29 metric policy (PR-AUC primary, no
accuracy headline, resampling inside the training fold only). Order per **D10**.