# Research Progress Report

## An Intelligent Flood Risk Prediction and Early Warning System for Bangladesh Using Machine Learning and Spatiotemporal Data

| | |
| --- | --- |
| **Student** | Md Riyad |
| **Degree / academic program** | CSE (Computer Science and Engineering) |
| **Department / institution** | Department of CSE, State University of Bangladesh |
| **Supervisor** | Prof. Dr. Sharmin Parveen |
| **Report type** | Research progress report (condensed technical review) |
| **Reporting date** | 20 October 2026 |
| **Evidence base** | 1,052,044 FFWC readings; 200,665 × 67 frozen dataset; 24-model comparison; governed model D16 |

**How this report was produced.** The repository was inspected read-only and the key artefacts were verified (row counts, hashes, model loading): governing documents, phase reports, metric files, model artefacts, notebooks, scripts and logs were read and cross-checked. No model was retrained and no data were modified. This is the condensed version of the project's technical review: it keeps the full evidence trail, the headline numbers and the disclosed caveats, while omitting the per-phase narration and the completion roadmap. Every number is traceable to the phase report cited.

---

## 1. Executive summary

The project forecasts, for each river-gauge station in Bangladesh, the probability that the station reaches its FFWC-defined danger level within the next 24, 48 or 72 hours. One prediction is made per station per day (one row = one station × one issue day). The target is not a water-level regression and not a same-day flood classifier: it is a probabilistic lead-time forecast, issued from information available at the end of day *t*, of an exceedance in the window *(t, t+h]*. The definition has been fixed since Phase 0 and has not been weakened since.

**Completed work.** The project has completed a disciplined methodological programme: data acquisition and QC; a leakage audit; a temporal/spatial split design with a dry-season embargo; a model-ready dataset frozen by SHA-256; six baseline variants against a mandated one-feature persistence bar (D7); five classical ML families; deep-learning families (LSTM/GRU/TCN, fusion, gated GNN, gated Transformer); a single-path fair ML-vs-DL comparison over 24 models; probability calibration; cost-based thresholds and risk bands; error analysis; robustness analysis; feature-group ablation; model selection; and TreeSHAP-based explainability. A post-selection "Model Improvement v2" sprint (MI-1 … MI-8) added new data sources, expanded the feature set from 62 to 110 columns, re-selected a model and ran a fair deep-learning rematch. All 27 notebooks execute with zero errors, and an experiment registry tracks 24 registered experiments.

**Governed model.** The currently governed model is **D16**, `hpo_perhorizon_FULL_V2`: per-horizon LightGBM (24 h, 48 h) and XGBoost (72 h) on the 110-column FULL_V2 set, seed `20261008`, frozen at `models/final_v2/selected_model_v2.pkl` on 15 October 2026. It was selected over the Phase-33 XGBoost (62 features) on the pre-registered primary cell `val_2022` (horizon-mean PR-AUC **0.957427** vs 0.954634) and confirmed on the frozen test (**0.77076** vs 0.75601, mean of 7 test cells; persistence 0.66755). The later MI-7/MI-8 sprints did not change the champion.

**Strongest verified findings.**

- **Hydrology dominates, and the model is scientifically sensible.** `wl_dist_to_danger` and `wl_pct_danger` are the leading drivers at every horizon; the hydrology group carries ≈64 % of global attribution at 24 h, followed by geography (8.4 %), neighbouring-station stage (6.1 %) and weather (4.7 %). The ranking is stable across model generations (Spearman ρ = 0.76–0.88).
- **The learned model beats persistence out-of-sample.** It clears D7 in **63/63** Phase-31 all-season robustness points (minimum margin +0.0143), **21/21** Phase-32 points and **21/21** frozen-test points (test mean 0.77076 vs 0.66755; minimum test margin +0.0424).
- **The ML-vs-DL question has a fair, negative answer.** Under equal features and equal tuning budgets, the best deep model reached 0.93559 (Phase 27) and 0.93399 (MI-7 rematch) vs 0.95463 for the ML reference and 0.95743 for D16; the paired intervals exclude zero in the ML direction at 24 h and 48 h (72 h straddles by 0.0007).
- **Probabilities are usable for risk bands.** 17 of 24 models already satisfy ECE ≤ 0.02 without post-hoc calibration; D16's raw ECE is 0.0023, and its test ECE is 0.0028–0.0085. At a 10:1 missed-flood:false-alarm cost stance, the validation "very high" band is ≈90 % flood-pure (13–18× lift).
- **The signal's ceiling is understood.** The label audit found 4–10 % of positives hinge on a ≤ 2 cm threshold crossing and estimated the empirical ceiling at ≈0.96–0.97 on `val_2022`; a 0.99 PR-AUC target is not attainable, and differences of a few thousandths are noise.

**Most important limitations (all evidenced in this report).**

- **The frozen test was read twice after the freeze — three times in total.** `reports/test_read_ledger.md` records the Phase-16 baseline scoring (pre-freeze), the Phase-33 selection read and the MI-6 D16 read (18 hours later), plus an MI-4 descriptive label pass. MI-v2 was launched after the Phase-33 test result was seen, so D16's number is the maximum of the two post-freeze reads: a programme-level test-informed optimism that must be stated, not hidden.
- **The MI-sprint 7-cell summaries are in-sample for six of the seven cells.** Four sprint scripts fit once on training rows (≤ 2021) and then scored `val_2020`, `val_2021` and the spatial folds, which are training-row subsets. This invalidates the "21/21 dev cells", 7-cell means and SEDI champion means — but **not** the D16 selection (made on `val_2022`) and **not** the MI-6 test read (test rows are outside every fit). The fix is a scripted re-score under the project's own P2/P3 protocol.
- **Absolute test performance is modest.** The best test PR-AUC is 0.836 at 24 h and 0.707 at 72 h, far below validation (0.971/0.956/0.945). Part of the drop is base-rate physics and part is genuine difficulty; claims must be phrased relative to D7, never as high accuracy.
- **The 72 h product is the weak link.** Its test mean (0.707) is essentially tied with the previous model, and at the frozen threshold it misses 602 positive rows (precision 0.351). Thresholds in general do not transfer cleanly across years (test precision 0.549/0.388/0.351 at validation-derived thresholds).
- **No system prototype exists yet** (`app/` is absent); reproducibility has gaps (unpinned dependencies, no automated tests, unversioned data, several stale documents). None of these gaps affects the validity of the research results, but each must be closed or disclosed.

**Disposition.** The research is defensible as it stands, and the validity repairs are bounded: they require no new data, no new model and no further test read. Sections 6–9 present the evidence; Section 12 states the limitations and the concrete improvement priorities.

## 2. Research question and objectives

### 2.1 The problem and the research question

Bangladesh is dominated by the Ganges–Brahmaputra–Meghna delta. The annual monsoon routinely puts large parts of the river network above danger levels, and flood impacts are concentrated among communities with low protection and short warning times. The operational warning authority is the Flood Forecasting and Warning Centre (FFWC) of the Bangladesh Water Development Board; its station network and danger levels are the reference for this project. A station-level probabilistic lead-time forecast — "probability that station X crosses its danger level within the next 24/48/72 hours" — is directly usable for graded warnings and is the formulation the project brief requires.

> **"What is the probability of flooding at a particular location within the next 24/48/72 hours?"**

### 2.2 What exactly is predicted

- **Prediction unit:** one FFWC station × one issue day (daily resolution, UTC).
- **Issue time:** the forecast is issued at the end of day *t* from information observed up to and including day *t*.
- **Target:** a binary label per horizon. `label_h = 1` if any *observed* reading in the window *(t, t+h]* reaches or exceeds the station's danger level; `label_h = 0` only if every day in the window has ≥ 3 readings and a known danger level and none reaches it. **Missing data never counts as 0** — the label is `NaN` and the row is excluded from fitting and scoring.
- **Output:** a calibrated probability, converted into four risk bands (low / moderate / high / very high) by cost-derived thresholds.
- **Interpretation:** this is a probabilistic lead-time forecast, not a static susceptibility map and not a contemporaneous classifier. All three horizons are trained and reported separately; the project does not assume that one horizon says anything about another.

**Data availability constraint that shapes everything.** Features must be computable at issue time from data available then. The information boundary bans contemporaneous target information, and all rolling statistics, lags and aggregates are strictly backward-looking. Water-level persistence (`wl_pct_danger`) is by construction the strongest legitimate predictor — and therefore also the mandatory benchmark (decision D7).

### 2.3 Scope and intended application

The final modelling table covers **133 stations** (of 153 in the FFWC master with coordinates; 141 have any observation; 133 have a numeric danger level and at least one usable label). Coverage runs from 2017-01-06 to 2026-10-05; the year 2018 is missing from the FFWC archive and is never interpolated. The station network grows over time (85 stations observed by 2017; 133 by the end of the record), a fact kept visible in the validation design rather than smoothed over.

The proposal's application vision has five layers: a location-based prediction interface (district → upazila → location); an interactive flood-risk map with contributing factors; automatically refreshed data; an early-warning mechanism explicitly labelled a research prototype (distinct from any official government warning); and explainable predictions. The current project supports the model and explanation layers; the interface, map, real-time and alert layers are not built (Section 11.4).

### 2.4 Objectives and demonstrated contribution

The six binding objectives recorded in `reports/phase0_research_definition.md` are:

- Fix a correct flood-label definition (done; never weakened).
- Design a leakage-free temporal/spatial experimental design (done).
- Engineer meaningful, flood-science-based predictors (done; audited).
- Evaluate on recall, PR-AUC, calibration and false negatives (done; accuracy explicitly banned as a headline).
- Produce explainable, location-aware risk prediction (done for tabular models; location layer pending).
- Only then build the map, automatic data acquisition and early-warning application (not started except partial FFWC ingest).

The strongest defensible contribution is methodological rather than predictive novelty: a **fair, leakage-audited comparison of classical ML against multiple deep-learning families on a common probabilistic flood-forecasting task**, with one metric code path, identical splits and equal tuning budgets. Secondary contributions are the label-contract work, the cost-based risk-band derivation and the explainability analysis. The thesis should not claim a new forecasting breakthrough; the project's own label audit bounds what any model can achieve on this label.

**Demonstrated:** the pipeline exists; D16 is frozen and loadable; it beats the persistence bar out-of-sample on every proper protocol and on the frozen test; the ML/DL comparison is fair and DL did not win; explanations are hydrology-first and stable; probabilities calibrate well; the label's noise ceiling is quantified. **Not demonstrated:** operational usefulness for real warnings; generalisation beyond the covered years; behaviour under genuine distribution shift; stakeholder-calibrated costs; a working prototype.

## 3. Data sources and dataset construction

### 3.1 Data sources used

All sources were verified as present with documented provenance. Sources only probed are marked.

| Data group | Source | Role in forecasting | Coverage | Status / limitations |
|---|---|---|---|---|
| FFWC station master | FFWC API → `data/raw/ffwc/ffwc_stations.csv` | Station identity; **danger level = target threshold** | 153 stations | Numeric danger level on 133/153 (20 non-numeric) |
| FFWC observed water level | FFWC API, 3-hourly → `data/raw/ffwc/waterlevel/` | **Primary predictor** + target construction | 1,052,044 readings; 141/153 stations; 2017-01-01 → 2026-10-06 | 2018 entirely absent; sentinel values ≤ −1000 dropped |
| NASA POWER / MERRA-2 | `data/raw/rainfall/weather_station_daily.csv` | Daily rainfall + temperature | 550,341 rows (153 stations × 3,597 days) | Reanalysis ~50 km — **not** rain gauges; 459 `-999` sentinels |
| Geospatial | SRTM 1-arc-sec; OSM rivers; geoBoundaries ADM1/2 | Elevation, slope, river distance/density, administrative membership | 40 SRTM tiles; 6,792 river centrelines | Static features; licences documented |
| Open-Meteo GloFAS (MI-1b) | `flood-api.open-meteo.com` → `data/raw/glofas/` | Transboundary / upstream river discharge | 55,363 rows; 11 anchors; 2013 → 2026 | **Adopted** into FULL_V2 (`gf_*`) |
| POWER soil wetness (MI-1c) | POWER API → `data/raw/power_soil/` | Top/root/profile soil wetness | 669,389 rows; 133 stations | **Adopted** into FULL_V2 (`ps_*`) |
| Open-Meteo ERA5-Land (MI-1a) | `archive-api.open-meteo.com` → `data/raw/openmeteo/` | Higher-resolution rainfall/soil candidates | 669,389 rows; 133 stations; 8 variables | **Evaluated and not adopted** (wash: −0.00025) |
| CHIRPS observed rainfall | Probed only | Candidate replacement for reanalysis rain | — | Reachable but heavy; **not used** |

### 3.2 From raw observations to the model-ready table

The build path is implemented in `src/flood/features/build_dataset.py` and audited by `scripts/audit_dataset.py` and `scripts/verify_model_ready.py`:

- **Water levels** are aggregated from 3-hourly readings to daily per station (max/mean/min/last observed, reading count), using the master danger level as the reference.
- A **full calendar matrix** is materialised first — every station × every calendar day (`flood_dataset.csv`, 357,250 × 73) — so missing days are explicit rather than silently absent.
- **Quality-control rules** (decisions D4/D9) remove invalid values and sentinels; the complete 2018 archive gap is never interpolated.
- The **model-ready table** keeps only observed days with at least one knowable label: `flood_dataset_model_ready.csv`, **200,665 × 73**, 133 stations, 2017-01-06 → 2026-10-05.
- **Phase-11 spatial features** add 19 columns (terrain, river distances/density, administrative attributes, neighbourhood aggregates).
- The **Phase-14 experimental dataset** selects the conceptual layout Location / Timestamp / Predictors / Target: `data/processed/experimental_dataset.csv`, **200,665 × 67**, 119.9 MB, 67-row data dictionary. It is declared immutable; its SHA-256 is `a6748529083fcdcebb8b8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e`, re-checked by every later phase.

### 3.3 Label contract and base rates

For each station and issue day *t*: `label_h = 1` if any observed reading in *(t, t+h]* reaches or exceeds danger; `label_h = 0` only if every day in the window has ≥ 3 readings and a known danger level and none reaches it; otherwise `NaN`. Positives are not vetoed by missingness, and the closure property `label_24h = 1 ⇒ label_48h = 1 ⇒ label_72h = 1` is used as a consistency check.

| Quantity | 24 h | 48 h | 72 h |
|---|---:|---:|---:|
| Labelled rows (full table) | 200,559 | 197,748 | 195,174 |
| Positives | 9,982 | 11,241 | 12,340 |
| Base rate (full table) | 4.98 % | 5.68 % | 6.32 % |
| Train base rate (non-holdout) | 9.59 % | 10.60 % | 11.51 % |
| `val_2022` base rate | 5.09 % | 5.94 % | 6.58 % |
| Frozen-test base rate | 1.41 % | 1.77 % | 2.12 % |

Base rates drift strongly over time: the 72 h rate is 16.81 % in 2017, 6.64 % in 2022, **1.19 % in 2023** and 1.90 % in 2026. Monsoon (JJAS) holds 55.5 % of rows and a 5.4× higher 72 h base rate than the rest of the year (9.79 % vs 1.83 %). Positives are spatially concentrated: the top-10 stations carry 38.9 % of all 24 h positives. These three facts are why the project bans pooled metrics and requires per-year, per-season, per-cell reporting.

### 3.4 Features and feature sets

| Set | Columns | Composition | Used by |
|---|---:|---|---|
| FULL | 62 | weather 5 + rain history 7 + geography 14 + hydrology 24 + network 12 | Phase 20–33 models |
| LINEAR | 35 | redundancy-collapsed / VIF-pruned | logistic baselines |
| SELECTED | 10 | compact cross-check | logistic/tree checks |
| FULL_V2 | 110 | FULL + soil 19 + GloFAS 18 + water-level memory/API 6 + interactions 5 | **D16 (governed)** |
| FULL_V2_era5 | 136 | FULL_V2 + 26 ERA5-Land columns | probed, **not adopted** |
| FULL_V3 | 165 | FULL_V2 + 55 MI-8 columns | score push, **not adopted** |

The Phase-12 selection produced the three frozen base sets; its evidence includes mutual information, permutation importance and a backward-elimination curve, and the disagreement between filter and model criteria is reported rather than hidden. The 48 MI-v2 columns cleared a Phase-13-style leakage audit (backward shifts only; climatologies computed strictly before 2022-01-01).

### 3.5 Dataset limitations

- **2018 is absent** from the water-level archive; no interpolation was attempted.
- **Rainfall is reanalysis, not observed.** NASA POWER/MERRA-2 (~50 km) is the rainfall backbone; ERA5-Land was evaluated and did not help; CHIRPS was reachable but unused.
- **Danger level is missing for 20 of 153 stations**, and 8 of 141 observed stations have no usable danger level at all; the model covers 133 stations.
- **Network growth is non-random.** 23 stations first appear after 2022 and have a test base rate ≈3× the trained bucket's; they are reported as a separate bucket, never folded into "seen".
- **Labels are noisy at the margin.** 4–10 % of positives depend on a ≤ 2 cm crossing of the threshold, bounding achievable PR-AUC on `val_2022` at ≈0.96–0.97.
- **Positives are concentrated** in a minority of stations, so pooled metrics are dominated by them and macro per-station numbers must always accompany them.

## 4. Experimental design and evaluation protocol

### 4.1 Temporal split

A random split is explicitly forbidden. The design (`reports/phase15_split_design.md`; decision D8) is:

| Part | Rows | Dates | Stations | 24 h | 48 h | 72 h |
|---|---:|---|---:|---:|---:|---:|
| train | 77,571 | 2017-01-06 → 2021-11-03 | 105 | 9.60 % | 10.72 % | 11.73 % |
| val (`val_2022`) | 21,920 | 2022-03-15 → 2022-10-30 | 105 | 5.08 % | 5.96 % | 6.64 % |
| test (frozen) | 101,174 | 2023-01-27 → 2026-10-05 | 133 | 1.41 % | 1.77 % | 2.12 % |

The cuts are 2022-01-01 and 2023-01-01. Each cut sits inside an **88–131-day natural dry-season gap with zero model-ready rows**, so no label window can straddle a cut; an explicit 7-day embargo remains in code as a guard and flags zero rows. Validation model fitting actually uses the non-holdout subset: **61,991 train rows** and **17,538 val rows**. The test set holds **101,174 rows**, of which **19,763 (26 stations)** are the frozen unseen-station holdout.

### 4.2 Spatial design and test buckets

Five station-disjoint folds, balanced on label count within early/late cohorts (first observed ≤ 2017 vs ≥ 2018). **Fold 0 (26 stations, 19,763 test rows) is the frozen unseen-station holdout**; folds 1–4 (26–27 stations each) provide **P3 development-time** unseen-location transfer within the training years. The distinction matters: a model trained on train years has seen the P3 held-out stations' time series in other folds, so P3 measures transfer to unseen stations within the development period, while `test_unseen` (fold 0, 2023+) is the final unseen-location claim.

The test period is split into three buckets: `test_trained` (84 trained stations), `test_new_network` (23 stations first observed after 2022) and `test_unseen` (the 26 holdout stations). The new-network bucket exists precisely because those stations would otherwise masquerade as "seen".

### 4.3 Evaluation protocols

| Protocol | Fits on | Scores on | Purpose |
|---|---|---|---|
| **P1** primary | train ≤ 2021, no holdout | `val_2022` (17,538 rows) | model selection |
| **P2** rolling origin | `year ≤ Y−1` (refit per year) | `val_2020`, `val_2021`, `val_2022` | temporal generalisation, per year |
| **P3** spatial CV | train years, 4 of 5 folds (refit per fold) | left-out fold, same years | unseen-location transfer in development |
| **P4** frozen test | — | fold-0 stations, 2023+ | final unseen-location claim |
| **P5** season | — | every metric split JJAS vs rest | monsoon vs non-monsoon |

The distinction between P2/P3 (refit per cell) and a single global fit scored on all cells is central. The project's own baselines and evaluation phases implement P2/P3 correctly (`scripts/baseline_model.py` `build_tasks()`), and the MI-7 rematch also refits per cell. Four MI-sprint scripts deviate from this (Section 4.7).

### 4.4 Leakage prevention

The Phase-13 audit (11/11 checks green, no feature demoted) verified:

- **Banned columns never enter any model:** `station_id` (as a feature), `date`, `name`, `source`, all three `label_*` and all three `obs_days_*`.
- **No forward-looking features:** every lag/rolling/aggregate is computed strictly from data at or before day *t*; the label-window scan found no feature computed over the target window.
- **Preprocessing is train-only:** median imputation, frequency encoding (never target encoding) and standardisation are fitted on each fit fold only.
- **Thresholds are train-derived** for classification metrics, with the explicit exception of the Phase-29 decision layer, which derives operating thresholds on `val_2022` under a stated cost and freezes them before test use.
- **Holdout discipline:** every development mask excludes `station_holdout` rows; sequence windows inherit split/fold/holdout/season of their anchor day and never cross station identity (keep-all + missingness mask; 0 samples dropped).

The audit verifies the information boundary of the engineered features; it cannot prove generalisation, does not cover label noise (handled in MI-4), and does not test a hidden distribution shift such as a changed FFWC reporting convention after 2022.

### 4.5 Metrics and aggregation rules

- **PR-AUC (average precision) is the primary metric**; ROC-AUC is secondary context; **accuracy is banned as a headline** — the trivial always-negative classifier already scores 94.91 / 94.06 / 93.42 % (24/48/72 h) on validation.
- All metrics are computed **per cell** — year, season, bucket, horizon — and never pooled. Station-bootstrap 95 % CIs (100 resamples) accompany headline cells; overlapping CIs are reported as *not separable*.
- Probability quality: Brier score and expected calibration error (ECE, 10 equal-width bins) through a shared metric code path.
- Operational: missed-flood rows, false-alarm rate, false-positive rate, event-level recall and warning lead time.
- **SEDI/EDI** were added as a prevalence-robust secondary score in a development-only analysis; the model-zoo section is clean, but the champion section inherits the in-sample issue in Section 4.7 and must not be quoted.

### 4.6 Test governance and the actual read history

This is reported in detail because the difference between the documented claim and the actual history is material. The project-level record is `reports/test_read_ledger.md` (created 2026-10-16 by the MI-7 credibility-fix track), with a machine-readable companion:

| # | Date (UTC) | Phase / script | Kind | Window opened? |
|---|---|---|---|---|
| 1 | ≈2026-10-09 | Phase 16 `baseline_model.py` | baseline scoring (planned) | **Yes — full read, before any freeze** |
| 2 | 2026-10-15T05:31Z | Phase 33 `phase33_selection.py` | pre-registered selection read | **Yes — full read** |
| 3 | 2026-10-15T23:34Z | MI-6 `mi6_test_read.py` | second selection read | **Yes — full read, 18 h later** |
| 4 | ≈2026-10-16 | MI-4 `mi4_label_audit.py` | label statistics only | Descriptive; predictions not scored |

The ledger verdict: two full model reads of the frozen test; four window exposures including the baseline and descriptive passes; the production headline is the maximum over the two test reads; the "single read" claim is false at programme level and true only within each script's own guard file.

The MI-6 report's own "one read" statement is correct for MI-6's guard file, but `reports/mi_status.md` §6 ("the test is read exactly once, in MI-6") and `final workflow.md` line 14 ("read once") are contradicted by the ledger. The MI-7 and MI-8 sprints kept the test sealed (`test_rows_read = 0`). The operating rule is that any future read must be pre-registered, added to the ledger and reported as an additional read.

**Interpretation.** The Phase-16 read is an intended baseline that predates the freeze discipline, but it means the test was not pristine at the Phase-33 freeze. The decision to launch MI-v2 was informed by the Phase-33 test result. Therefore D16's test score is the maximum of two full reads and must be presented as such. Each individual selection step used validation only, but the *programme* is test-informed.

### 4.7 The MI-sprint protocol inconsistency

The audit discovered a methodological inconsistency that affects a set of published summary numbers but not the governed model's selection or its test read.

**What the scripts do.** `scripts/mi3_compare.py` (line 65), `mi3_search.py` (≈237), `mi5_selection.py` (≈198) and `mi8_select.py` (≈237) each fit a single model on `split == "train" & ~holdout` (≤ 2021) and then score all seven "dev cells": `val_2020`, `val_2021`, `val_2022` and `spatial_fold_1..4`. Six of those cells are subsets of the very training set — `val_2020` and `val_2021` are 2020/2021 rows and the spatial folds are train-year rows partitioned by station. Only `val_2022` is out-of-sample under this protocol.

**Evidence.** `reports/mi8_scorecard.csv` shows `train_rows = 61,977` for every cell and near-perfect scores on those six cells (PR-AUC 0.99998–1.00000), while `val_2022` behaves normally (0.9707). The same signature appears in `mi3_featureset_summary.json` and `mi5_scorecard.csv`, and it underlies the "unresolved leakage-shaped pattern" reported in `mi8_selection.md` §6.

**What this invalidates.** The 7-dev-cell means and "21/21 cells" claims in the MI reports, the D7 floors, the MI-8 "+0.00135 7-cell fixed-config gain" and the champion section of `sedi_edi_dev.md` (D16 7-cell SEDI means 0.9903/0.9845/0.9743). These are in-sample for six cells and overstate performance and robustness.

**What it does not invalidate.** D16's selection (MI-5 ranked candidates on `val_2022`, and MI-7 reproduced D16's `val_2022` number with proper per-cell refits); the MI-6 test read (test rows lie outside every fit mask); the Phase 16–27, 31 and 32 grids (correct P2/P3 refits throughout); the MI-7 rematch (explicit per-cell refits); and the `val_2022` SEDI values (0.9647/0.9516/0.9532) plus the model-zoo table.

**Fix (executed 2026-10-22).** The affected blocks were re-scored with the established per-cell protocol (`baseline_model.build_tasks()`); the design was pre-registered in `reports/p2p3_correction_protocol.md` before the run, and the results are in `reports/p2p3_correction.md`. The run reproduces the `val_2022` gates exactly, so **D16 remains the champion**. The seven-cell MI-8 claim is corrected or withdrawn according to the pre-registered rule, and the corrected D16 D7 margins replace the archived floors. This is a development-only operation and required no new test read.

### 4.8 What the design supports

- **Supports:** temporal generalisation across 2020–2022 (P2), unseen-station transfer during development (P3), performance vs D7 under one metric code path (Phase 27), calibration and threshold behaviour on validation, robustness slices, and a single unseen-station claim on the frozen test (subject to the double-read caveat).
- **Does not support:** statistical separation among the top models (CIs overlap); a "leakage-free" banner (specific risks were tested; reporting-convention shift is untested); future-period claims beyond the covered years; operational claims.

## 5. Experiments and model-selection history

The programme follows the workflow's "one family per phase, every phase evaluated on the same development cells, nothing pooled" pattern. The table below is the verified experiment sequence; the key numbers are developed in Section 6.

| Phase | Work | Verified outcome |
|---|---|---|
| 0–5 | Research definition; data inventory; QC/EDA; target study | Definition and label rules frozen; 2018 gap documented |
| 6–13 | Temporal/spatial/redundancy analysis; feature engineering/selection; leakage audit | FULL 62 / LINEAR 35 / SELECTED 10; 11/11 leakage checks green |
| 14–16 | Experimental dataset; split design; baselines | 200,665 × 67 frozen by SHA-256; D7 persistence 0.89465; logistic 0.92317 |
| 17–19 | Experiment framework and registry; sequence datasets | 24 registered experiments; 4 sequence tensors; keep-all + mask policy |
| 20 | Classical ML (five families) | Best XGBoost 0.95463; all five beat D7 |
| 21–25 | Deep learning: LSTM/GRU/TCN, fusion, gated GNN, gated Transformer | Best deep 0.93559 (fusion); graphs/Transformer 0.928–0.932; none beats classical |
| 26 | Multi-horizon matrix | Classical wins at every horizon; margin narrows at 72 h |
| 27 | Fair ML-vs-DL comparison (24 models, one scoring path) | Classical 0.95463 vs deep 0.93559; fresh re-score Δ = 0.00e+00 |
| 28 | Calibration | 17/24 pass ECE ≤ 0.02 raw; median 0.0067 → 0.0025 after calibration |
| 29 | Thresholds and risk bands (10:1) | 0.0605 / 0.0524 / 0.0614; validation recall 0.980 / 0.982 / 0.966 |
| 30 | Error analysis | 10 named patterns; misses mostly below danger but rising (68.7 %) |
| 31 | Robustness under every split protocol | 63/63 points above D7, minimum margin +0.0143 |
| 32 | Feature-group ablation and comparison matrix | Hydrology +0.4702, geography +0.3131; 21/21 above D7 |
| 33 | Final selection and first governed test read | XGBoost L2-03; test 0.8026 / 0.7498 / 0.7107; frozen `models/final/` |
| 34 | Explainability | Hydrology 64 % of attribution; cross-model ρ = 0.76–0.88 |
| MI-v2 | New data, features, HPO, re-selection, test read, DL rematch | D16 governed; test 0.77076 vs 0.75601; no further gain in MI-8 |

**Baselines (Phase 16).** Six variants were fitted on all development cells: climatology (0.0587 horizon mean), persistence (D7, 0.8946), logistic regression in three feature variants (0.9232 linear, 0.9232 without upstream features, 0.9203 selected) and a shallow tree (0.8870). A static-only reference (0.5325) confirms that geography alone is close to useless: the signal must come from dynamic hydro-meteorological features. D7 persistence is strong (0.92029 at 24 h), which is why it is the bar every model must clear. A first test-period exposure also occurred in Phase 16: across the seven test cells the horizon means were logistic 0.7511/0.6771/0.6227, persistence 0.7340/0.6625/0.6062 and shallow tree 0.7274/0.6799/0.6214 — the honest floor against which later models should be read, and evidence that the test period is much harder than validation for every model.

**Classical models (Phase 20).** Five tree families were tuned under a fixed budget on the same development grid. On the P1 cell the horizon means were LightGBM 0.9534, XGBoost 0.9546, XGBoost-selected 0.9539, random forest 0.9523 and gradient boosting 0.9471. Per-horizon winners differ (LightGBM 0.96816 at 24 h, XGBoost 0.95495 at 48 h, XGBoost-selected 0.94216 at 72 h) — the direct reason the final model is per-horizon rather than one model with a horizon input.

**Deep learning (Phases 21–25).** Sequence datasets (L3/L7/L14/L30) were built with explicit missingness masks. The best recurrent model was `lstm_L3` (0.9268) and the best TCN 0.9212 — both well above D7 but below the classical trees by 0.027–0.033. Fusion (`fusion_L3`, concatenating sequence representation with the 62 engineered features) reached 0.9356, the closest any deep family came before MI-7. Gated graph models reached 0.928–0.931 with narrow, horizon-dependent gains on unseen folds (GCN +0.018 / GAT +0.023 at 72 h), and the gated Transformer reached 0.9317 (+0.005 over the best LSTM). None approached the tabular leaders.

**Fair comparison (Phase 27).** All 24 models (6 baselines + 5 classical + 13 deep/graph/sequence) were re-scored through one scoring path on all seven dev cells × 3 horizons × 3 seasons, with the same seeds, splits and metric code; P2/P3 fold models were refit per task. Fresh P1 re-scores matched the phase grids bit-for-bit, a strong reproducibility check. The primary result — best classical 0.95463 vs best deep 0.93559 — is developed in Section 6.

**Calibration, decision layer and errors (Phases 28–30).** Phase 28 found 17 of 24 models already inside ECE ≤ 0.02 without post-hoc calibration; post-hoc calibration reduces median ECE from 0.0067 to 0.0025. Phase 29 chose thresholds on `val_2022` under a 10:1 missed-flood:false-alarm cost (with 5:1 and 20:1 sensitivity), and defined four risk bands. Phase 30 established ten named error patterns on the leaders' decisions (83 FN, 208 FP, 1,044 TP, 15,920 TN; recall 0.926, FAR 0.166): misses occur while the river is still below danger (median 0.76 m) but rising (68.7 % of FN rising over three days), grow with horizon (16/19/48) and concentrate at stations with fewer training floods. One documentation defect exists: pattern A's `detail` string in `phase30_patterns.csv` is inverted relative to its own numeric medians; the substantive finding is correct and the label needs fixing.

**Robustness and ablation (Phases 31–32).** Phase 31 re-fitted the surviving leaders under every split protocol and tested years, stations, seasons, extreme rainfall, low-data windows and a stricter issue-time variant; its headline is 63/63 cell × horizon points above D7 with two honest caveats (Section 8). Phase 32 ran the feature-group ablation: paired bootstrap increments are +rainfall history 0.0101, +geography 0.3131, +hydrology 0.4702, +station network 0.0009 and −0.0001 for dropping the upstream features — geography and hydrology are the decisive extensions.

**Selection and explainability (Phases 33–34).** A 24-model, multi-criteria scorecard selected XGBoost L2-03 on the 62-feature FULL set as the Phase-33 champion; its test read gave horizon means 0.8026/0.7498/0.7107 over the four per-year cells, 28,826 alarms, 1,044 missed floods, 27,782 false alarms at the 10:1 stance and a mean lead of ≈1.1 days. Phase 34 applied exact TreeSHAP to the frozen boosters; the hydrology dominance and its stability are reported in Section 9.

**The MI-v2 programme (15–17 October 2026).** The sprint was a response to the Phase-33 test result, not a new research question. Each stage has its own report and artefacts:

| Stage | Work | Result / verdict | In-sample issue? |
|---|---|---|---|
| MI-1 | ERA5-Land, GloFAS discharge, soil wetness landed | GloFAS and soil adopted; ERA5 probed | No (data only) |
| MI-2 | 48 new engineered features + leakage audit | FULL_V2 = 110; audit green on backward-only shifts | No |
| MI-3 / 3a | Feature-set comparison + HPO; ERA5 probe | FULL_V2 best on `val_2022`; ERA5 wash (−0.00025) | Partly (7-cell aggregates) |
| MI-4 | Label audit and ceiling | 4–10 % near-threshold positives; ceiling ≈0.96–0.97 | No (descriptive) |
| MI-5 | Re-selection under pre-registered rule | **D16 selected** (0.95743); frozen `models/final_v2/` | Partly (gates on six cells) |
| MI-6 | Guarded frozen-test read (D16 vs D7) | **0.77076 vs 0.66755**; beats D7 21/21, Phase-33 6/7 | No (test out-of-sample) |
| MI-7 | Fair DL rematch (equal features, equal budget) | DL 0.93399 vs ML 0.95463; DL loses at every horizon | No |
| MI-8 | Feature score push (FULL_V3, 165 columns) | No improvement (−0.00004); CIs straddle zero | Yes (7-cell gain) |

**Governance note.** Decision D15 froze the Phase-33 model and said "no re-selection after Phase 33". MI-5 re-selected under a disclosed, pre-registered rule (D16), and the workflow's status now names D16 as governed. This is a real deviation from the letter of D15 even though it was disclosed and methodologically pre-registered; it should be reconciled in the thesis, with the Phase-33 model retained as the reference.

## 6. Results and comparisons

### 6.1 Headline numbers and their context

The same models appear under several aggregation rules. The table fixes the context of each headline number; none of the rows may be combined into a leaderboard.

| Number | What it is | Model | Context |
|---:|---|---|---|
| **0.95743** | horizon-mean PR-AUC | D16 | P1 `val_2022`, all season |
| 0.95463 | horizon-mean PR-AUC | Phase-33 XGBoost | same cell |
| 0.93559 | horizon-mean PR-AUC | best deep (`fusion_L3`) | same cell |
| 0.89465 | horizon-mean PR-AUC | D7 persistence | same cell |
| **0.77076** | mean of 7 test-cell horizon means | D16 | frozen test, 101,174 distinct rows |
| 0.75601 | mean of 7 test-cell horizon means | Phase-33 XGBoost | same frozen test |
| 0.66755 | mean of 7 test-cell horizon means | D7 persistence | same frozen test |
| ≈0.98–0.99 | "7-dev-cell mean" | MI-v2 candidates | **in-sample for 6 of 7 cells — not a valid estimate** |

**Rule used throughout:** every figure is labelled with metric, protocol/cell, season and aggregation. A validation figure is never presented as a test figure; a horizon mean is never presented as a per-horizon result; a mean of cell means is never presented as a row-level metric.

### 6.2 Development evidence (P1 `val_2022`, proper out-of-sample)

The Phase-27 comparison is the cleanest single table in the project: 24 models, one scoring path, the same split, all three horizons, season = all. The top rows and the D7 reference (recomputed from `reports/phase27_metrics.csv`; PR-AUC):

| Model | Family | 24 h | 48 h | 72 h | Horizon mean |
|---|---|---|---:|---:|---:|---:|
| XGBoost (`L2-03`) | classical | 0.96812 | 0.95495 | 0.94084 | **0.95463** |
| XGBoost-selected | classical | 0.96631 | 0.95333 | 0.94216 | 0.95393 |
| LightGBM | classical | 0.96816 | 0.95379 | 0.93838 | 0.95344 |
| Random forest | classical | 0.96650 | 0.95155 | 0.93885 | 0.95230 |
| Gradient boosting | classical | 0.96184 | 0.94569 | 0.93383 | 0.94712 |
| Fusion (`fusion_L3`) | deep (best) | 0.94443 | 0.93854 | 0.92381 | 0.93559 |
| Transformer (gated) | deep | 0.94371 | 0.93304 | 0.91830 | 0.93168 |
| LSTM (`lstm_L3`, best) | deep | — | — | — | 0.92677 |
| **D7 persistence** | **bar** | **0.92029** | **0.89541** | **0.86826** | **0.89465** |
| Static-only | sanity | — | — | — | 0.53249 |
| Climatology | sanity | — | — | — | 0.05868 |

Three facts stand out. (1) **Every learned family beats D7**; even the weakest learned model clears it by +0.013 on the horizon mean and the weakest tabular model by +0.052. (2) **The best deep model is 0.019 below the best classical model** on the same inputs, and the gap is larger on P3. (3) **No single model wins all horizons** — LightGBM is marginally best at 24 h, XGBoost at 48 h, XGBoost-selected at 72 h — which is why the final system is per-horizon. The top four classical models are within 0.0023 of each other, well inside sampling noise, so the choice among them rests on secondary criteria (calibration, missed floods, robustness).

### 6.3 Proper per-cell evidence (P2/P3)

The same Phase-27 table contains the rolling-origin and spatial-fold refits. Horizon-mean PR-AUC by cell family:

| Model | P2 worst year | P2 mean (3 years) | P3 worst fold | P3 mean (4 folds) |
|---|---:|---:|---:|---:|
| XGBoost | 0.95463 (`val_2022`) | 0.96645 | 0.94455 | 0.96225 |
| LightGBM | 0.95344 | 0.96501 | 0.94011 | 0.96199 |
| D7 persistence | 0.89465 | 0.92427 | 0.88260 | 0.91771 |
| Fusion (`fusion_L3`) | 0.93559 | 0.94645 | **0.82792** | 0.91757 |

The tabular models are stable across years and folds (worst fold 0.940–0.945). The deep fusion model degrades sharply on one spatial fold (0.828), the clearest evidence that its proximity at 24 h does not survive an unseen-station test. These are the numbers to quote for "the model generalises across time and stations during development"; Phases 31–32 extend them to seasons, rainfall extremes and missingness (Section 8).

### 6.4 The governed model D16

MI-5 ranked five eligible v2/v1 candidates by the pre-registered rule (gate 1: beat D7 on all dev cells; gate 2: ECE ≤ 0.02; then horizon-mean `val_2022` PR-AUC, missed floods, robustness, inference cost). The ranking on the primary cell:

| Rank | Model | 24 h | 48 h | 72 h | Mean | Worst ECE |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | **`hpo_perhorizon_FULL_V2` (D16)** | 0.97079 | 0.95648 | 0.94501 | **0.95743** | 0.0023 |
| 2 | `lgbm7_FULL_V2` | 0.97079 | 0.95648 | 0.94287 | 0.95671 | 0.0025 |
| 3 | `xgboost_FULL_V2` | 0.96954 | 0.95492 | 0.94321 | 0.95589 | 0.0027 |
| 4 | `xgb3_FULL_V2` | 0.96752 | 0.95368 | 0.94501 | 0.95540 | 0.0019 |
| 5 | `xgboost_FULL` (Phase-33) | 0.96812 | 0.95494 | 0.94084 | 0.95463 | 0.0025 |

The gap between rank 1 and rank 2 is **+0.00072**, and the station-bootstrap 95 % CI for D16 at 24 h is [0.9372, 0.9895] — an interval ≈0.05 wide. The top candidates are therefore **not statistically separable**; D16 was chosen by a pre-registered rule, not by a significant margin. The improvement over the Phase-33 model is +0.0028 on the selection cell, smaller than the uncertainty. The MI-8 fixed-config controls confirm the picture: FULL_V2 control 0.95778 (+0.00036 vs D16, CI straddling zero), FULL_V3 0.95739 (−0.00004), and the pre-registered MI-8 gain does not survive a paired bootstrap. **D16 is defensible as the governed model; the claim "v2 is significantly better than v1" is not.**

### 6.5 Frozen-test evidence

The two post-freeze reads are reported together because they were scored on the same frozen cells. PR-AUC, all seasons:

| Test cell (definition) | D16 24 h | D16 48 h | D16 72 h | D7 24 h | D7 48 h | D7 72 h |
|---|---:|---:|---:|---:|---:|---:|
| `test_2023` | 0.7956 | 0.6814 | 0.6257 | 0.6276 | 0.5523 | 0.5108 |
| `test_2024` | 0.9177 | 0.8573 | 0.8043 | 0.8022 | 0.7302 | 0.6727 |
| `test_2025` | 0.8048 | 0.7598 | 0.7019 | 0.7328 | 0.6670 | 0.6041 |
| `test_2026` | 0.7935 | 0.7467 | 0.7023 | 0.7233 | 0.6495 | 0.5929 |
| `test_trained` (84 seen stations) | 0.8686 | 0.8091 | 0.7614 | 0.7468 | 0.6762 | 0.6183 |
| `test_new_network` (23 post-2022 stations) | 0.8434 | 0.7868 | 0.7129 | 0.7356 | 0.6930 | 0.6457 |
| `test_unseen` (26 holdout stations) | 0.8299 | 0.7416 | 0.6410 | 0.7696 | 0.6694 | 0.5986 |
| **Mean of 7 cells** | **0.8362** | **0.7690** | **0.7071** | **0.7340** | **0.6625** | **0.6062** |

- **D16 beats D7 on all 21 test points** (7 cells × 3 horizons). The smallest margin is **+0.0424** (`test_unseen`, 72 h); typical margins are +0.06 to +0.17, so the result is not a knife-edge.
- **D16 beats the Phase-33 model on 6 of 7 cells** (horizon-mean comparison), with the single loss at `test_2025` (−0.0037). At 72 h the two are a wash (0.70709 vs 0.70923; the Phase-33 model is marginally ahead).
- **The unseen-station claim is the most important one for the proposal**, and it is positive but smaller than on trained stations: +0.060 / +0.072 / +0.042 at 24/48/72 h over D7 (vs +0.122 on trained stations at 24 h). At 24 h the unseen PR-AUC (0.830) is close to the trained-station value (0.869), so the model does transfer to new locations.
- The **test level is far below validation** (0.77 vs 0.96 horizon mean). This is not a collapse of the model; it is the combined effect of much lower base rates (2023 24 h ≈ 0.7 %), a harder year distribution and PR-AUC's prevalence dependence. The correct comparison is D7 within the same cell.

The Phase-33 read remains the reference (horizon means 0.8026/0.7498/0.7107 over the four per-year cells; 28,826 alarms, 1,044 missed floods, 27,782 false alarms at the 10:1 stance, mean lead ≈1.1 days). The Phase-16 baseline read is older and uses different models, so it is context rather than comparison. The MI-6 read is the second full read, and the D16 headline is the maximum of the two.

### 6.6 Horizon behaviour

The horizon ordering is stable everywhere: **24 h > 48 h > 72 h** — on validation (0.971 > 0.956 > 0.945 for D16), on P2/P3 and on the test (0.836 > 0.769 > 0.707). The degradation is largest between 48 and 72 h, where the physical predictability of rainfall-driven stage peaks out; the Phase-30 analysis shows the 72 h misses concentrate in the early monsoon and at stations with rapid response. The 72 h model is the weakest link; on the seven test cells the Phase-33 72 h mean (0.70923) is essentially tied with D16 (0.70709). Any operational claim should treat 72 h as a lower-confidence product.

### 6.7 What is statistically established

- **D16 > D7: not marginal.** It holds on all 21 test points and on all proper development protocols; the minimum margin is +0.014 on development and +0.042 on test, both far outside bootstrap noise for these cells.
- **ML > DL:** the Phase-27 gap (0.019) and the MI-7 gap (0.0206) are consistent; the MI-7 paired intervals exclude zero at all three horizons (24 h [-0.053, -0.007], 48 h [-0.045, -0.005], 72 h [-0.043, -0.002]). The claim should be "the tested DL families did not beat the tested ML families on this task", not "DL cannot work here".
- **D16 vs Phase-33: not established.** +0.0028 on validation and +0.0148 on the test mean, but the per-cell comparison loses one cell and the validation CIs overlap heavily. The honest statement is "D16 is the governed model by a pre-registered rule and is at least as good as the Phase-33 model; the margin is within noise".
- **Top-4 classical models vs each other: not established** (all within 0.0023 on validation).
- **Any "7-cell mean" or "21/21 dev cells" claim from the MI sprint: withdrawn**, except where explicitly re-scored per cell in `reports/p2p3_correction.md` (Section 4.7).
- **What the numbers do not show:** that the system would have issued useful warnings in an operational sense (no end-to-end warning simulation); that the model is robust to a change in FFWC reporting practice; that the 72 h product is decision-grade; or that calibration transfers to future years with base rates an order of magnitude lower.

## 7. Calibration and warning interpretation

### 7.1 What was checked and what was found

Calibration asks whether a predicted probability of 0.2 corresponds to a 20 % observed frequency. The project reports the Brier score and the expected calibration error (ECE, 10 equal-width bins) for every model × horizon × season, with reliability diagrams for the leaders; ECE ≤ 0.02 is the tolerance.

- **Most models are already well calibrated.** On `val_2022`, 17 of 24 models clear the 0.02 bar **without** post-hoc calibration. The exceptions are `static_only` (ECE 0.374), `lstm_L3` (0.072), `tcn_bestL` (0.057), `climatology` (0.049), `lstm_L7` (0.048), `gru_bestL` (0.041) and `random_forest` (0.035).
- **Post-hoc calibration helps where needed.** Five-fold out-of-fold isotonic/Platt calibration reduces the median ECE from 0.0067 to 0.0025. The governed family's **raw** ECE is 0.0023 (D16), 0.0025 (XGBoost) and 0.0033 (LightGBM); the fusion model is also well calibrated (0.0045).
- **Test-period calibration held for D16.** The MI-6 test metrics give all-season ECE **0.0028 / 0.0042 / 0.0085** and Brier **0.0056 / 0.0090 / 0.0131** at 24/48/72 h (means over the seven test cells). This is a genuinely good result: the probabilities remained calibrated even though the base rate fell by roughly a factor of 3–7 relative to validation.
- **The result is cell-specific.** Pooling years with very different base rates would produce an ECE that measures prevalence drift, not model quality; all quoted ECEs are per cell and per season.

### 7.2 Thresholds and risk bands

The model outputs a probability; the warning system needs a decision. Phase 29 defines it explicitly:

- **Cost stance:** missing a flood is 10× worse than a false alarm (`C_FN:C_FP = 10:1`, with 5:1 and 20:1 sensitivity runs). This is a research assumption, not a stakeholder-validated cost.
- **Threshold rule:** for each horizon and period (all/monsoon/non-monsoon), choose the threshold that minimises `10·FN + 1·FP` on `val_2022`; freeze it before any test use. All-season values: 0.0605 (24 h), 0.0524 (48 h), 0.0614 (72 h). For D16, the same method gives 0.103 / 0.053 / 0.082.
- **Risk bands:** the probability is cut into four bands. On `val_2022` the very-high band is near-perfect (≈90 % observed flood rate, 13–18× lift over the base rate). The band boundaries are validation artefacts and migrate with prevalence; the bands are a communication device, not a calibrated probability statement.

| Context | Horizon | Threshold | Recall | Precision | Missed positive rows |
|---|---:|---:|---:|---:|---:|
| Validation (`val_2022`, Phase-29 leaders) | 24 h | 0.0605 | 0.980 | 0.746 | 18 / 891 |
| Validation | 48 h | 0.0524 | 0.982 | 0.667 | 19 / 1,027 |
| Validation | 72 h | 0.0614 | 0.966 | 0.630 | 38 / 1,124 |
| Frozen test (D16) | 24 h | 0.103 | 0.918 | 0.549 | 234 |
| Frozen test (D16) | 48 h | 0.053 | 0.903 | 0.388 | 342 |
| Frozen test (D16) | 72 h | 0.082 | 0.856 | 0.351 | 602 |

The validation and test blocks are different models and periods and are not directly comparable row-for-row.

- **On validation**, the operating point deliberately buys recall with false alarms: the false-alarm rate is 0.25 / 0.33 / 0.37 — one in four to one in three alarms is false even on validation.
- **On the frozen test**, the 10:1 stance produces many false alarms and material missed floods at 72 h (602 positive rows). The 72 h threshold transfers poorly from validation, which is itself a finding: thresholds are cell-specific and must be re-derived per season and per year if the system is ever deployed.

### 7.3 Interpretation limits

- **The cost ratio is not validated with users.** A 10:1 assumption is reasonable for early warning, but a warning agency may weight credibility differently; the 5:1/20:1 sensitivity should be carried into the discussion.
- **Calibration is not decision quality.** Good ECE plus a low threshold can still yield low precision when the base rate is 1 %, and the test numbers show exactly that.
- **Threshold transfer is the weak link.** Re-deriving thresholds per season and per year, or using a fixed alarm budget (e.g., the top 5 % of days), are the two obvious alternatives.
- **The band labels are not comparable across cells** and must always be shown with the underlying rate and lift for the cell they were derived on.

## 8. Robustness and generalisation

### 8.1 The proper per-cell robustness grid (Phase 31)

Phase 31 refits the surviving leaders under every split protocol and evaluates years, stations, seasons, rainfall extremes, low-data windows and a stricter issue-time convention. Verified results:

- **All-season, all cells: 63/63 points above D7**, minimum margin **+0.0143** (`xgboost_selected`, `spatial_fold_2`, 24 h). This is the strongest robustness statement in the project, and it is clean (proper per-cell refits).
- **The stricter issue-time variant is not robust.** If every feature is replaced by its previous-calendar-day value (a within-station *t−1* shift, so the model cannot use same-day observations), the leaders lose **0.0125–0.0675** PR-AUC and **19/63** points fall below D7. The current headline therefore depends on same-day observations being available at issue time; the thesis must state the issue-time convention explicitly.
- **Spatial folds are the main source of variance.** The monsoon/non-monsoon gap is ≤ 0.0435 on the temporal cells but reaches **0.2586** on `spatial_fold_1` at 72 h (monsoon 0.9339 vs non-monsoon 0.7069), driven by a collapsing non-monsoon base rate. A minority of station folds is genuinely hard, and the aggregate hides it.
- **Extreme rainfall is the hardest regime.** Extreme-rain windows roughly double the base rate (24 h 0.154 vs 0.063) but **lower** leader PR-AUC by −0.0401 / −0.0464 / −0.0486 versus low-rain cells. Across the 63 extreme-bucket points the leaders beat D7 in only **28/63**, with losses up to −0.035. Recall is held (minimum 0.940 / 0.894 / 0.893) but the alarm rate rises (median FAR 0.33 → 0.37 → 0.43). When rain is already extreme, yesterday's stage is a strong carry-forward predictor and the learned edge narrows.
- **Low-data windows:** the median PR-AUC is flat across the none/low/high/very-high missingness buckets (0.9664–1.0000); the low extrema come from thin buckets (e.g., 4 positives) and are treated as noise. The imputation/masking strategy is not hiding a broad weakness, though individual thin cells are unstable.

### 8.2 Test buckets and the unseen-station claim

The test is evaluated in three pre-defined buckets, never pooled into a single "test score":

| Bucket | Stations | D16 24 h | D16 48 h | D16 72 h | D7 24 h | D7 48 h | D7 72 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| trained (seen in training) | 84 | 0.8686 | 0.8091 | 0.7614 | 0.7468 | 0.6762 | 0.6183 |
| new network (first observed ≥ 2023) | 23 | 0.8434 | 0.7868 | 0.7129 | 0.7356 | 0.6930 | 0.6457 |
| unseen (26-station holdout) | 26 | 0.8299 | 0.7416 | 0.6410 | 0.7696 | 0.6694 | 0.5986 |

- **New-network stations are handled well** (+0.108 / +0.094 / +0.067 over D7), which matters because the operational network is growing.
- **Unseen-station performance is positive but the margin is smallest at 72 h** (+0.042). At 24 h D16 beats D7 by +0.060 (0.830 vs 0.770), a meaningful result and an improvement over the Phase-33 model, which **tied** D7 on `test_unseen` 24 h (0.7675 vs 0.7696, −0.0020, CIs overlapping).
- **The unseen claim is not "the model works anywhere".** The holdout stations are still FFWC gauges in the same country and hydrological regime; they are unseen by the model, not out-of-distribution by design.

### 8.3 Concentration and station-level fairness

Positives are concentrated: the top-10 stations carry 38.9 % of all 24 h positives, and Phase 30 finds that stations with fewer training floods are missed more. Phase 31's unseen-station folds show the same effect (seasonal gap up to 0.2586 on `spatial_fold_1`). A station-level recall table and a map of per-station skill should accompany the thesis; a single national number hides the equity problem.

### 8.4 What has not been tested

- **Future years** beyond 2026-10 and a full seasonal cycle under a changed FFWC reporting regime.
- **A genuine distribution shift** (new dam/embankment, changed danger levels, a different rainfall product). The rainfall-source sensitivity is partially covered by the ERA5 probe, but not a redefinition of the label.
- **Operational latency and data outages.** The pipeline's behaviour under late or missing FFWC updates is not tested.
- **Event-level usefulness.** Event recall exists in the metric set, but no end-to-end warning simulation (issue time → alert → lead time → missed events) has been run.

## 9. Explainability and scientific interpretation

### 9.1 Method

Phase 34 applies exact **TreeSHAP** to each frozen per-horizon booster after preprocessing (median imputation + train-fitted frequency encoding), maps the encoded geography columns back to their group, aggregates to ten feature groups and cross-checks with permutation importance (3,000 validation rows, 3 repeats, PR-AUC scoring). A cross-model comparison against the Phase-33 XGBoost and an integrated-gradients comparison against the DL models test whether the explanation is model-specific. All explanations are on the P1 `val_2022` cell; **no test row is used**.

### 9.2 Findings

- **Hydrology dominates at 24 h: 64.0 % of attribution** (24 water-level features, led by `wl_dist_to_danger`, with `wl_pct_danger`, rise rates, danger level and the API terms). Geography contributes 8.4 % (elevation, river distance, administrative membership), neighbouring-station stage 6.1 %, weather 4.7 %; the new v2 groups each carry a few percent — GloFAS discharge 4.6 %, soil wetness 4.1 %, interactions 4.1 %, water-level memory/API 2.4 % — and rainfall history 1.5 %. This is the 24 h group table; the ranking is not asserted for the longer horizons.
- **The explanation is stable across model generations.** D16 vs the Phase-33 XGBoost gives Spearman **ρ = 0.76–0.88** on the 62 shared features (all vs monsoon), so the change from 62 to 110 features did not reshape the model's logic. The DL models' integrated gradients reproduce the same layer ranking (hydrology first), a useful convergence result: the ML/DL difference is in accuracy, not in what the models look at.
- **Permutation importance disagrees sharply (ρ = 0.021 at 24 h; 0.40 and 0.26 at 48/72 h), and the report explains why.** Under strongly correlated features (distance-to-danger, percentage-of-danger, raw level, lagged level), permuting one feature leaves its near-duplicates intact, so importance mass collapses. This is the textbook failure mode of permutation importance, not evidence against the SHAP result; the cross-model SHAP agreement is the trustworthy stability check.
- **The local case studies are consistent with the error analysis.** The four examples (a correct high-risk alarm, a correct low-risk call, a missed flood and a false alarm) show the model responding to the same drivers Phase 30 identified — antecedent rain and upstream stage for hits; rapid local response or a late-forming event for misses; saturated soils in the early monsoon for false alarms.

### 9.3 Scientific reading and limits

The dominant role of distance-to-danger is physically expected but also a warning: the model is largely an **autoregressive stage model**, and the heavy lifting is done by the current water level relative to the threshold rather than by rainfall. That is legitimate for 24 h, less so for 72 h, where rainfall memory should matter more; the report presents the group table for 24 h only, so a horizon-wise shift in attribution is a physical expectation, **not a measured result**. The practical implication is that **upstream discharge and rainfall data are most valuable for extending lead time** — exactly where the model is weakest.

Limits: SHAP values are associational, not causal; the groups are correlated (hydrology encodes past rainfall); the analysis covers one validation cell and the 24 h story is over-represented; and the eight observed stations with no usable danger level cannot be explained at all. None of this invalidates the explanation, but the thesis should not present group percentages as physical effect sizes.

## 10. Deep-learning comparison

### 10.1 What was compared and how

The proposal's methodological question is whether models that learn temporal/spatial representations directly from sequences beat conventional ML on engineered features. The project answered it three times, with increasing fairness:

- **Phases 21–25 (family-by-family):** LSTM/GRU/TCN, fusion, gated GNN and gated Transformer, each compared with the classical leader on the same development grid.
- **Phase 27 (single-path comparison):** all 24 models re-scored through one metric path, same splits and seeds, with P2/P3 fold models refit per cell.
- **MI-7 (equal-features, equal-budget rematch):** the best DL architecture (`fusion_v2_L3`) rebuilt on the 110-column FULL_V2 set and tuned with the same HPO budget as the ML reference. Only one DL family met the 20/20 completion criterion, so the rematch is a fair test of *that family*, not of every architecture — a scope limitation stated in `reports/mi7_report.md`.

### 10.2 Results

| Comparison | Best deep model | Best ML reference | Difference (deep − ML) | Statistical reading |
|---|---|---:|---:|---|
| Phase 27, P1 `val_2022` | fusion_L3 0.93559 | XGBoost 0.95463 | **−0.01904** | DL loses at all three horizons |
| Phase 27, P3 worst fold | fusion_L3 0.82792 | XGBoost 0.94455 | −0.11663 | DL collapses on one unseen-station fold |
| MI-7, P1 `val_2022` | MI7_fusion_v2_L3 0.93399 | XGBoost 0.95463 | **−0.02064** | paired CI excludes zero at 24/48 h; 72 h straddles by 0.0007 |
| MI-7 vs governed D16 | 0.93399 | D16 0.95743 | −0.02344 | same direction |

- **The DL models are not broken.** They beat D7 comfortably (e.g., fusion 0.9356 vs 0.8946) and their calibration is good (Phase-28 fusion raw ECE 0.0022 / 0.0024 / 0.0045 at 24/48/72 h; MI-7 `fusion_v2_L3` mean ECE 0.0075, inside the 0.02 tolerance). The failure is relative, not absolute.
- **The gap is largest exactly where generalisation matters.** On the P3 folds the fusion model's worst fold (0.828) is far below the tabular models' (0.940–0.945).
- **Stacking does not rescue it.** A learned stack of ML + DL + persistence adds ≈+0.0002 PR-AUC over the best single model and is not worth the complexity.
- **Cost.** The DL models need orders of magnitude more training time, more parameters and slower inference; on a CPU-only project the tabular models are also the pragmatic choice.

### 10.3 Interpretation

The most defensible explanation is **inductive bias plus data scale**, not "deep learning doesn't work for floods": at ≈62k labelled training rows from 84 contributing stations (the 26-station holdout is excluded from fitting) over ≈5 years of daily observations, with strong engineered persistence and rainfall-memory features, a tree ensemble's axis-aligned splits on the water-level variables are close to the Bayes-optimal representation; the sequence models have to learn the same persistence from raw windows with far fewer effective samples. The project's ablation supports this: the engineered hydrology/geography groups contribute most of the signal, and the DL models are strongest when given the same features (fusion) rather than sequences alone. A future study with sub-daily data, a denser station network or a water-level regression target might reverse the result; that is an honest scope statement, not a hedge.

**Verdict for the thesis:** the DL comparison is a **completed negative result with a fair protocol**, and it is one of the project's strongest scientific contributions. The translation "DL did not outperform ML here" must always carry the conditions (same features, same budget, this dataset, this label).

## 11. Reproducibility, technical quality and system status

### 11.1 What is strong

- **Immutable data with a hash.** `experimental_dataset.csv` is 200,665 × 67 with SHA-256 `a6748529083fcdcebb8b8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e`, verified in the audit and re-verified by every phase; the source table `flood_dataset_model_ready.csv` (200,665 × 73) is read-only and separately hashed.
- **One seed, one metric path.** Seed `20261008` is used throughout. Phase 27's single scoring path eliminates the "different script, different PR-AUC" failure mode; fresh P1 re-scores matched the original grids exactly (ΔPR-AUC = 0.00e+00), and the MI-6 D7 cross-check reproduced the Phase-16 read to 5.6×10⁻¹⁷.
- **Executed notebooks.** All 27 notebooks are executed with stored outputs and zero errors; phase reports carry explicit checks (e.g., 59/59, 63/63, 21/21, 13/13).
- **Run logs and registries.** Each phase has a run log with exit code and wall time; the experiment registry tracks 24 registered experiments and their policies; MI stages have their own logs and summaries.
- **Provenance discipline.** Every report names its input hash, seed, script and outputs; the data dictionary has 67 rows; feature whitelists and leakage-audit JSONs are on disk.

### 11.2 What is weak

| Gap | Evidence | Consequence |
|---|---|---|
| **Dependencies unpinned** | `requirements.txt` lists packages without versions; `shap` 0.53.0 is installed although the file marks SHAP as deferred; `torch 2.14.1+cpu` is recorded only in prose | A fresh environment may not reproduce the numbers |
| **No tests, no CI** | `tests/` absent; verification is via notebooks and asserts inside scripts | Regressions are caught only by re-running phases |
| **Dirty, large working tree** | ~147 entries; ~2.5 GB data unversioned; MI artefacts and `models/final_v2/` untracked | The audited state cannot be reconstructed from git |
| **No end-to-end rerun in the audit** | Verification was artefact loading, hashes, executed notebooks, logs and cross-checks | "Runnable" is evidenced; a clean-machine rebuild is not demonstrated |
| **Stale documentation** | `models/final_v2/README.md` ("test never read"), `final_config_v2.json` (`test_rows_read = 0`), `final workflow.md` line 14 ("read once"), `reports/mi_status.md` §6 | Contradicts the ledger; risks a false claim in the thesis or a demo |
| **Phase-30 label inversion** | `reports/phase30_patterns.csv` pattern A `detail` string | A quoted statistic would be wrong if copied verbatim |
| **No packaging or entry point** | Pipeline is a collection of scripts with relative assumptions | Hard for an external examiner to run |

### 11.3 Reproducibility verdict

The **numeric evidence is reproducible within this machine and this repository**: hashes, seeds, logs, one scoring path and bit-identical re-scores. The **project is not yet reproducible from a clean environment**, because dependencies are unpinned, data are not versioned and there is no automated test or one-command pipeline. Closing that gap requires packaging work only; it does not require new modelling.

### 11.4 Current system status

What exists: raw-data ingestion (FFWC water levels and station master; POWER weather; geospatial layers), the feature/target build and QC, training and evaluation code, the frozen model artefacts (Phase-33 reference and D16 governed), batch scoring of a frozen table, and the explanation artefacts. The governed model loads and predicts: 110 features, per-horizon `{24 h: lgbm, 48 h: lgbm, 72 h: xgb}`, seed `20261008`.

What does **not** exist: a live prediction service or API; the location → district → upazila interface; the risk map; automatic weather refresh; a real-time pipeline; a warning engine; user interfaces; architecture documentation; and system tests (`app/` and `tests/` are absent). The project is therefore a **research modelling pipeline with a frozen model, not a prototype system**. It can produce predictions and explanations for a prepared table; it cannot ingest a new day's data end-to-end, serve a request, draw a map or issue an alert. Showing a notebook and a pickle is not showing an early-warning system, and the report should say so plainly.

One concrete caveat: the `models/final_v2/README.md` snippet omits the merge of `data/processed/mi_v2_features.csv` (the source of the 48 MI-v2 columns). A user who follows the README with raw model-ready columns will not have the 110 features, and the call will fail or silently use the wrong frame.

## 12. Limitations and improvement priorities

### 12.1 Key limitations

The table condenses the full limitation register; each item states the evidence, why it matters and the concrete action that addresses it.

| Area | Limitation (evidence) | Consequence / required action |
|---|---|---|
| Data | **Rainfall is reanalysis** (~50 km; CHIRPS unused; ERA5-Land no gain) | Rainfall-driven and 72 h claims are limited by input error; test an observed-rainfall product |
| Data | **2018 entirely missing** | One full monsoon season lost; no interpolation (correct); disclose and do not impute |
| Data | **Danger level unusable for 20/153 stations; 8 observed stations excluded** | Coverage is 133 stations, not national; map coverage and state the scope |
| Data | **Network growth is non-random** (23 post-2022 stations, ≈3× base rate) | Pooled test numbers confound station mix with skill; always report buckets |
| Data | **Label noise** (4–10 % of positives within 2 cm; ceiling ≈0.96–0.97) | A 0.99 target is unattainable; differences below ≈0.01 are not meaningful |
| Data | **Base-rate drift** (72 h: 16.8 % in 2017 → 1.2 % in 2023) | PR-AUC is prevalence-dependent; per-cell reporting and D7-relative claims only |
| Data | **Positive concentration** (top-10 stations = 38.9 % of 24 h positives) | National averages hide station-level failure; add a per-station table and map |
| Data | **GloFAS is modelled discharge**, not gauged | Upstream signal carries unquantified error; note it where the feature matters |
| Method | **MI 7-cell summaries in-sample for 6/7 cells** | Invalidates "21/21 dev", 7-cell means and SEDI champion means; re-score under P2/P3 and withdraw until done |
| Method | **Frozen test read twice post-freeze (three times total)** | D16's test number is the maximum of two reads; disclose, treat as upper bound, no further reads |
| Method | **D15 superseded by D16; top models not separable** (rank gap +0.00072; top-4 within 0.0023) | Document the deviation; phrase selection as rule-based, not significant |
| Method | **Thresholds transfer poorly** (test precision 0.549/0.388/0.351) | Re-derive per season/year or use an alarm budget; revalidate the decision layer |
| Method | **HPO uses a single seed and a limited budget; event-level evaluation incomplete** | Tuning noise is not quantified; add the event-level table (event recall, lead time, missed events) |
| Evaluation | **Absolute test skill is modest** (0.836/0.769/0.707; 72 h weakest) | Claims must be D7-relative; present 72 h as experimental/low-confidence |
| Evaluation | **Unseen stations are same-network, same-country; no future-year or external validation** | Scope the generalisation claim precisely; test the reporting-convention shift |
| Evaluation | **Extreme rainfall is the hardest regime** (leaders beat D7 in only 28/63 extreme points) | Report the extreme-bucket table; do not claim uniform robustness |
| Documentation | **Stale "read once / never read" documents; Phase-30 label inversion; unpinned dependencies; no tests/CI; no prototype** | Reconcile to the ledger, fix the label, pin and lock dependencies, add smoke tests, and build the smallest honest inference path |

### 12.2 Improvement priorities

The priorities below address the existing results; none requires new data collection, a new model class or another test read.

**P0 — Validity (blocking).** Re-score the MI candidates and D16 under the project's own P2/P3 protocol (`baseline_model.build_tasks()` or MI-7's `build_tasks_index()`), producing all seven cells × three horizons; correct or withdraw every affected claim (21/21 dev, 7-cell means, MI-8 gain, SEDI champion means). Reconcile every document that says "read once", "never read" or `test_rows_read = 0` to the ledger, and fix the Phase-30 pattern-A label. These are bounded scripted and editorial tasks.

**P1 — Evaluation that strengthens the existing results.** Add event-level evaluation on the test (event recall, missed events, lead time, alarms per event, per year and bucket); a per-station skill table and map; calibration by band and horizon on the test; threshold alternatives (per-season/year re-derivation and a fixed alarm budget); and a paired-bootstrap claim audit for D16 vs D7, D16 vs Phase-33 and ML vs DL.

**P2 — Model skill, targeted.** The 72 h horizon is where effort can pay: stronger rainfall-memory and upstream-discharge features, and an evaluation of an observed-rainfall product (CHIRPS/IMERG) against the reanalysis input. Station-transfer diagnostics on the hardest spatial folds would make the unseen-station claim more informative. Further tuning on `val_2022` is not a priority: the label-noise ceiling (≈0.96–0.97) and the inseparable top models bound what tuning can demonstrate.

**P3 — Reproducibility and honest artefacts.** Pin dependencies and add a lockfile; add a data manifest (file, rows, SHA-256, source, retrieval date); add smoke tests (dataset hash, feature count, D16 load, one prediction, one metric); run the pipeline in a fresh environment and record the result; update the stale README/config; and keep the "research prototype, not an official warning" disclaimer on every interface.

**Points for discussion.** Three framing decisions remain: (1) how the two test reads are presented — both reported, with D16 foregrounded as the D7-relative and unseen-station result rather than an unbiased estimate; (2) whether the headline contribution is the fair ML-vs-DL negative result plus the label/leakage methodology rather than a predictive-performance claim; and (3) whether 72 h is presented as experimental rather than decision-grade.

## 13. Conclusion

The project has completed a substantial and methodologically serious body of research: a frozen, hash-verified dataset; a label contract that refuses to treat missing data as no-flood; a leakage audit; a temporal/spatial design with an embargo and a frozen unseen-station holdout; a persistence bar that every model must beat; a single-path comparison of 24 models; calibration, cost-based thresholds, error analysis, robustness, ablation and explainability; and a post-selection improvement sprint that produced the governed model D16 and closed the deep-learning question with a fair but scope-limited negative result.

The evidence supports a clear set of claims. D16 is a legitimate governed model: it loads, it is calibrated, it beats persistence on every proper development protocol and on all 21 frozen-test points, and its explanations are hydrology-first and stable across model generations. The fair comparison shows that the tested deep-learning families did not beat the tested classical models on this dataset and label, at equal features and budget. The label audit explains why a 0.99 target is not attainable and why differences of a few thousandths are noise.

The evidence also supports a clear set of caveats, and they must not be buried. The frozen test was read twice at programme level, and the D16 number is the higher of the two reads; the MI-sprint 7-cell means are in-sample for six cells and must be re-scored before they are quoted; the top models are not statistically separable; the 72 h product is weak; the thresholds do not transfer well; and the system prototype does not exist. None of these is fatal — each has a concrete, bounded fix — but a thesis that omits them would be less credible than the work deserves.

The research question — *what is the probability of flooding at a particular location within the next 24/48/72 hours?* — has been answered with a working probabilistic model and an honest account of its limits. That is a defensible result.

## Appendix A — Evidence pointers

| Artefact | What it establishes |
|---|---|
| `data/processed/experimental_dataset.csv` | Frozen model-ready dataset (200,665 × 67; SHA-256 `a674…c8e`; 67-row dictionary) |
| `models/final_v2/selected_model_v2.pkl` | Governed D16 artefact (110 features, seed `20261008`, loads and predicts) |
| `reports/phase27_metrics.csv` | 24-model fair comparison grid (7 cells × 3 horizons × 3 seasons) |
| `reports/phase31_robustness.md` | Proper per-cell robustness grid (63/63 above D7) |
| `reports/phase32_ablation.md` | Feature-group ablation and comparative matrix |
| `reports/phase34_xai.md` | SHAP attributions, group shares and local cases |
| `reports/phase29_thresholds.md` | Cost-based thresholds, risk bands and transfer analysis |
| `reports/mi5_selection.md` | D16 selection rule and ranking |
| `reports/mi6_test_read.md` | Frozen-test read (D16 vs D7; 7 cells) |
| `reports/test_read_ledger.md` | Complete test-exposure history (four exposures, two full reads) |
| `reports/mi8_scorecard.csv` | Evidence of the in-sample MI issue (`train_rows = 61,977` for all cells) |

All phase reports, metric files, model artefacts and scripts named in this document are part of the project repository.
