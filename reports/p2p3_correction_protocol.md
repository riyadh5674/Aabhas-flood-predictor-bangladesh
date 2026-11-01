# P2/P3 correction of the MI-sprint 7-cell claims — pre-registered protocol

**Status:** written **before** the correction run (2026-10-22).
**Owner decision:** supervisor instruction, 2026-10-22 — make the P2/P3 re-scoring
correction mandatory before the results are frozen.
**Seed:** `20261008` (`baseline_model.SEED`), the project seed for every reported result.
**Test policy:** no row with `split == "test"` is fitted, predicted or scored.
`test_rows_read = 0`. The frozen test is not involved in this correction.

---

## 1. The defect

Four MI-sprint scripts fit **one** model on `split == "train" & ~station_holdout`
(≤ 2021) and then score all seven "development cells":

| Script | Fit | Cells scored |
|---|---|---|
| `scripts/mi3_compare.py` | once, train ≤ 2021 | `val_2020`, `val_2021`, `val_2022`, `spatial_fold_1..4` |
| `scripts/mi3_search.py` | once, train ≤ 2021 | same |
| `scripts/mi5_selection.py` | once, train ≤ 2021 | same |
| `scripts/mi8_select.py` | once, train ≤ 2021 | same |
| `scripts/mi8_probe.py` | once, train ≤ 2021 | same |

Six of the seven cells are **subsets of the fit rows**:

* `val_2020` and `val_2021` are calendar years 2020/2021, which are `split == "train"` rows.
* `spatial_fold_1..4` are `split == "train"` rows partitioned by station.

Only `val_2022` is out-of-sample under a single ≤-2021 fit.

The signature is visible in `reports/mi8_scorecard.csv`: `train_rows = 61977` for every
cell, and PR-AUC ≈ 0.9999–1.0000 on the six training subsets. This is the issue recorded
in `reports/supervisor_progress_report_condensed.md` §4.7.

## 2. Claims in scope

| Claim (as published) | Source | Old value | Action |
|---|---|---|---|
| MI-8 feature block improves the 7-cell mean | `mi8_probe.json` | +0.00135 | re-score |
| MI-8 feature block wins **21/21** cells | `mi8_probe.json` | 21/21 | re-score |
| `FULL_V3` − `FULL` over 7 cells | `mi8_probe.json` | +0.00479 | re-score |
| MI-8 arm 7-cell means | `mi8_scorecard.csv` | ≈0.980–0.996 | re-score |
| SEDI champion 7-cell means | `sedi_edi_dev.md` | 0.9903 / 0.9845 / 0.9743 | re-score |
| D7 persistence floors on dev cells | `mi8_*` reports | 21/21 above D7 | re-score |
| MI-3 ERA5 7-cell gain | `mi3_featureset_summary.json` | +0.00075 | withdraw unless re-scored |
| MI-3c ensemble 7-cell means | `mi3_ensemble.csv` | quoted | withdraw unless re-scored |

**Not in scope / not affected:** D16 selection (made on `val_2022`), the MI-6 and MI-9
test reads, the Phase 16–27/31/32 grids, the MI-7 rematch, and the `val_2022` rows of
every MI table (a ≤-2021 fit is the correct P1 fit for that cell).

## 3. Correction design

### 3.1 Cells and fit masks

Each cell is fitted **only** on rows that precede it in time or exclude its stations,
using the project's own masks (`baseline_model.build_tasks()`):

| Cell | Protocol | Fit rows | Evaluation rows |
|---|---|---|---|
| `val_2020` | P2 rolling origin | `year ≤ 2019`, non-holdout | `year == 2020`, non-holdout |
| `val_2021` | P2 rolling origin | `year ≤ 2020`, non-holdout | `year == 2021`, non-holdout |
| `val_2022` | P1 / P2 (identical fold) | `split == "train"`, non-holdout (≤ 2021) | `split == "val"`, non-holdout |
| `spatial_fold_k` (k = 1..4) | P3 spatial CV | `split == "train"`, non-holdout, `fold ≠ k` | `split == "train"`, `fold == k` |

`val_2022` is retained as a **reproduction gate**: its fit is identical to the original
P1 fit, so the corrected run must reproduce the archived `val_2022` numbers exactly.

### 3.2 Models re-scored

**A. Fixed-config feature A/B** (`mi8_probe` correction). Identical XGBoost specification
for all three feature sets (`mi3_compare.XGB_PARAMS`; no tuning):

* `FULL` (62 columns)
* `FULL_V2` (110 columns)
* `FULL_V3` (165 columns = `FULL_V2` + 55 MI-8 columns)

**B. MI-8 selected arms vs D16** (`mi8_select` correction). The frozen per-horizon
configs, refit per cell:

* `d16_frozen` — `lgbm7` @ 24/48 h, `xgb3` @ 72 h, on `FULL_V2` (from `reports/mi3_hpo.csv`)
* `mi8_full_v3` — per-horizon winner of `reports/mi8_hpo_full_v3.csv`, on `FULL_V3`
* `mi8_full_v2` — equal-budget control, per-horizon winner of `reports/mi8_hpo_full_v2.csv`, on `FULL_V2`

**C. D7 persistence floor.** The project's D7 reference is a logistic link on
`wl_pct_danger` alone (`baseline_model.build_models`), refit on each cell's training rows.

### 3.3 Metric path and thresholds

* One metric path: `baseline_model.metrics()` — PR-AUC (primary), ROC-AUC, Brier,
  log-loss, ECE (10 bins), base rate, confusion counts, recall, precision, specificity, F1.
* Operating threshold for every fit: `baseline_model.best_f1_threshold()` computed on the
  **training rows of that same cell**, never on the evaluation rows.
* SEDI/EDI recomputed from the corrected confusion counts using the definitions in
  `scripts/sedi_edi_report.py` (Ferro & Stephenson 2011; zeros → `1e-9`).
* Cells are never pooled for a decision. The 7-cell mean is reported as a descriptive
  summary only, per D8.

### 3.4 Reproduction gates (must pass before any corrected number is used)

1. `val_2022` `FULL_V3` − `FULL_V2` must reproduce the archived `mi8_probe.csv` values
   to `< 1e-6` (same fit, same seed, same rows).
2. `val_2022` `d16_frozen` must reproduce `0.970789 / 0.956485 / 0.945007` (24/48/72 h)
   to `< 1e-5`.
3. `val_2022` `mi8_full_v3` and `mi8_full_v2` must reproduce the archived
   `mi8_scorecard.csv` values to `< 1e-5`.
4. The evaluation union must contain **zero** rows with `split == "test"`.

If any gate fails, the run is void and nothing is reported from it.

## 4. Pre-registered reporting rules

1. Every re-scored claim is replaced by its corrected value, even if smaller.
2. The "21/21 cells" phrase is replaced by the observed count. No rounding up, no
   "essentially all".
3. The MI-8 **fixed-config** 7-cell claim is withdrawn if the corrected
   `FULL_V3 − FULL_V2` mean is ≤ 0. The `val_2022` fixed-config difference is reported
   separately and is unaffected by this correction.
4. SEDI champion means are replaced by values recomputed from the corrected confusion
   tables, at per-cell training thresholds.
5. The D7 margin is reported per cell; the "beat D7 everywhere" phrasing is retained only
   if every corrected cell × horizon margin is positive.
6. **This analysis cannot change the champion.** D16 was selected on `val_2022`, which is
   correct under P1; the correction is a reporting fix, not a re-selection.
7. No test row is read, so the test-read ledger does not change.

## 5. Outputs

Written to `reports/p2p3_rescore/`:

* `probe_p2p3.csv` — 63 rows (3 feature sets × 7 cells × 3 horizons)
* `arms_p2p3.csv` — 63 rows (3 models × 7 cells × 3 horizons)
* `d7_p2p3.csv` — 21 rows (7 cells × 3 horizons)
* `summary.json` — corrected means, deltas, win counts, gates, archived-vs-corrected
* `run.log` — console log of the executed run

The narrative report is `reports/p2p3_correction.md`, written after the run.

## 6. Reproduce

```powershell
.venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py
```

Dev cells only. Deterministic. No test read.
