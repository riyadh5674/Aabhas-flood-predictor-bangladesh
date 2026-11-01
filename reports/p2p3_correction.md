# P2/P3 re-scoring correction — the MI-sprint 7-cell claims

| | |
| --- | --- |
| **Status** | executed and superseding — 2026-10-22T23:34:35 |
| **Script** | `scripts/mi10_p2p3_rescore.py` (seed `20261008`) |
| **Pre-registration** | `reports/p2p3_correction_protocol.md` (written before the run) |
| **Outputs** | `reports/p2p3_rescore/{probe,arms,d7,d7_margins}_p2p3.csv`, `summary.json`, `run.log` |
| **Test policy** | **`test_rows_read = 0`** — dev cells only; every evaluation mask asserted free of `split == "test"` |
| **Wall clock** | 2216 s |
| **Champion impact** | **none** — D16 was selected on `val_2022`, which this correction reproduces exactly |

## 1. What was wrong

The MI sprint reported "7-cell" means and "21/21 cells" from a **single fit on the
≤ 2021 training rows**, then scored `val_2020`, `val_2021`, `val_2022` and
`spatial_fold_1..4` with that one model. Six of those seven cells are subsets of the fit
rows:

* `val_2020` and `val_2021` are calendar years inside `split == "train"`.
* `spatial_fold_1..4` are `split == "train"` rows partitioned by station.

Only `val_2022` is out-of-sample under that fit. The diagnostic signature was
`train_rows = 61,977` for every cell together with PR-AUC ≈ 0.9999–1.0000 on the six
training subsets. The defect is documented in
`reports/supervisor_progress_report_condensed.md` §4.7; the affected scripts are
`mi3_compare.py`, `mi3_search.py`, `mi5_selection.py`, `mi8_select.py` and
`mi8_probe.py`.

The correction does **not** re-open model selection. D16's decision cell (`val_2022`) is
the one cell that was already out-of-sample, and it is reproduced to < 1e-5 below.

## 2. Correction design (pre-registered)

Every cell gets its own fit and its own training-derived operating threshold:

| cell | protocol | fit rows | evaluation rows |
| --- | --- | --- | --- |
| `val_2020` | P2 rolling origin | year ≤ 2019, non-holdout | year = 2020, non-holdout |
| `val_2021` | P2 rolling origin | year ≤ 2020, non-holdout | year = 2021, non-holdout |
| `val_2022` | P1 / P2 (same fold) | `split == "train"`, non-holdout (≤ 2021) | `split == "val"`, non-holdout |
| `spatial_fold_k` | P3 spatial CV | `split == "train"`, non-holdout, fold ≠ k | `split == "train"`, fold = k |

Two blocks were re-scored:

* **Feature A/B** — the fixed XGBoost specification (`mi3_compare.XGB_PARAMS`, no tuning)
  on `FULL` (62), `FULL_V2` (110) and `FULL_V3` (165).
* **Frozen arms** — D16 (`lgbm7` @ 24/48 h, `xgb3` @ 72 h, `FULL_V2`), the MI-8
  `full_v3` winner, and the equal-budget `full_v2` control, each on its frozen
  per-horizon configuration, plus the D7 persistence floor.

The operating threshold for each fit is the best-F1 threshold on **that cell's training
rows** (`baseline_model.best_f1_threshold`), never on the evaluation rows. PR-AUC is the
primary metric; SEDI/EDI are recomputed from the corrected confusion tables.

## 3. Reproduction gates

The `val_2022` fit is identical to the original P1 fit, so it must reproduce the
archived numbers exactly. All gates passed before any corrected number was accepted:

| gate | result | detail |
|---|---|---|
| probe_FULL_24h | PASS | new=0.968117985 archived=0.968117985 d=+0.00e+00 |
| probe_FULL_48h | PASS | new=0.954944820 archived=0.954944820 d=+1.11e-16 |
| probe_FULL_72h | PASS | new=0.940839376 archived=0.940839376 d=+0.00e+00 |
| probe_FULL_V2_24h | PASS | new=0.969538100 archived=0.969538100 d=+0.00e+00 |
| probe_FULL_V2_48h | PASS | new=0.954923223 archived=0.954923223 d=+0.00e+00 |
| probe_FULL_V2_72h | PASS | new=0.943209015 archived=0.943209015 d=+0.00e+00 |
| probe_FULL_V3_24h | PASS | new=0.969648419 archived=0.969648419 d=+0.00e+00 |
| probe_FULL_V3_48h | PASS | new=0.956144907 archived=0.956144907 d=+0.00e+00 |
| probe_FULL_V3_72h | PASS | new=0.943621043 archived=0.943621043 d=+0.00e+00 |
| d16_24h | PASS | new=0.970788816 frozen=0.970789000 d=-1.84e-07 |
| d16_48h | PASS | new=0.956484535 frozen=0.956485000 d=-4.65e-07 |
| d16_72h | PASS | new=0.945006724 frozen=0.945007000 d=-2.76e-07 |
| mi8_full_v3_24h | PASS | new=0.970662030 archived=0.970662030 d=+0.00e+00 |
| mi8_full_v3_48h | PASS | new=0.956765408 archived=0.956765408 d=+1.11e-16 |
| mi8_full_v3_72h | PASS | new=0.944737241 archived=0.944737241 d=+1.11e-16 |
| mi8_full_v2_24h | PASS | new=0.971982855 archived=0.971982855 d=+0.00e+00 |
| mi8_full_v2_48h | PASS | new=0.956365237 archived=0.956365237 d=+0.00e+00 |
| mi8_full_v2_72h | PASS | new=0.945006724 archived=0.945006724 d=+1.11e-16 |

## 4. Corrected results

### 4.1 Fixed-config feature A/B (PR-AUC, mean over 7 cells)

| feature set | corrected 7-cell mean | archived (in-sample) | inflation |
|---|---|---|---|
| `FULL` | 0.964051 | 0.983406 | -0.019355 |
| `FULL_V2` | 0.965930 | 0.986846 | -0.020917 |
| `FULL_V3` | 0.967796 | 0.988203 | -0.020406 |

`FULL_V3 − FULL_V2`: **+0.001867** corrected, versus **+0.001356**
archived; **20/21** cell × horizon points improve, versus
**21/21** archived.

| horizon | corrected `FULL_V3 − FULL_V2` |
| --- | ---: |
| 24 h | +0.001778 |
| 48 h | +0.002012 |
| 72 h | +0.001809 |

| cell | corrected `FULL_V3 − FULL_V2` |
| --- | ---: |
| `val_2020` | +0.000248 |
| `val_2021` | +0.001181 |
| `val_2022` | +0.000581 |
| `spatial_fold_1` | +0.007915 |
| `spatial_fold_2` | +0.001197 |
| `spatial_fold_3` | +0.000658 |
| `spatial_fold_4` | +0.001287 |

**Status of the "21/21 / +0.00135" claim:** **Corrected.** The signed mean is +0.001867; the 21/21 phrase becomes **20/21**.

### 4.2 Frozen MI-8 arms versus D16 (PR-AUC, mean over 7 cells)

| model | corrected 7-cell mean | archived (in-sample) | vs D16 (corrected) |
|---|---|---|---|
| `d16_frozen` | 0.966532 | 0.988520 | +0.000000 |
| `mi8_full_v3` | 0.968558 | 0.990160 | +0.002026 |
| `mi8_full_v2` | 0.966605 | 0.988922 | +0.000072 |

`mi8_full_v3 − d16_frozen` = **+0.002026**; `mi8_full_v2 − d16_frozen` =
**+0.000072**.

**Status of the MI-8 arm 7-cell claim:** **Corrected.** The corrected arm difference is +0.002026.

### 4.3 SEDI (mean over 7 cells, per-cell training thresholds)

| model | horizon | corrected SEDI | archived SEDI |
|---|---|---|---|
| `d16_frozen` | 24 h | 0.9749 | 0.9903 |
| `d16_frozen` | 48 h | 0.9645 | 0.9845 |
| `d16_frozen` | 72 h | 0.9551 | 0.9743 |
| `mi8_full_v3` | 24 h | 0.9674 | 0.9899 |
| `mi8_full_v3` | 48 h | 0.9632 | 0.9886 |
| `mi8_full_v3` | 72 h | 0.9572 | 0.9750 |
| `mi8_full_v2` | 24 h | 0.9629 | 0.9896 |
| `mi8_full_v2` | 48 h | 0.9675 | 0.9935 |
| `mi8_full_v2` | 72 h | 0.9551 | 0.9743 |

### 4.4 D7 persistence floor (corrected)

| model | points above D7 | minimum margin |
|---|---|---|
| `d16_frozen` | 21/21 | +0.015611 |
| `mi8_full_v3` | 21/21 | +0.016264 |
| `mi8_full_v2` | 21/21 | +0.015090 |

**Status of the "beats D7 on every dev cell" claim:** **Retained.** Every corrected cell x horizon point is above D7.

### 4.5 The decision cell is unchanged

`val_2022` reproduces the archived values exactly: D16 = **0.970789 / 0.956485 /
0.945007** (24/48/72 h; horizon-mean **0.957427**), and the fixed-config probe reproduces
`FULL` 0.968118, `FULL_V2` 0.969538, `FULL_V3` 0.969648 at 24 h (all horizons in the
gate table). This is why D16 remains the champion.

## 5. Claim ledger

| claim | old value | corrected value | status |
| --- | --- | --- | --- |
| MI-8 fixed-config 7-cell gain | +0.00135 | +0.001867 | corrected |
| MI-8 fixed-config wins | 21/21 | 20/21 | corrected |
| `FULL_V3 − FULL` 7-cell | +0.00479 | +0.003745 | corrected |
| MI-8 `full_v3` vs D16, 7-cell | +0.00164 | +0.002026 | corrected |
| D16 SEDI 24/48/72 h, 7-cell | 0.9903 / 0.9845 / 0.9743 | 0.9749 / 0.9645 / 0.9551 | corrected |
| D16 beats D7 on all dev points | 21/21 | 21/21 | retained |
| D16 `val_2022` horizon-mean | 0.957427 | 0.957427 (reproduced) | retained |
| D16 frozen-test mean vs D7 | 0.77076 vs 0.66755 | unchanged (test untouched) | retained |
| Phase 16–27 / 31 / 32 grids | per-cell protocol | unchanged | retained |
| MI-7 fair ML-vs-DL rematch | per-cell protocol | unchanged | retained |
| MI-3 ERA5 7-cell gain | +0.00075 | not re-scored | withdrawn |
| MI-3c ensemble 7-cell means | quoted | not re-scored | withdrawn |
| MI-5 candidate 7-cell columns | spatial mean / worst temporal / D7 floor | not re-scored for non-D16 candidates | withdrawn (selection on `val_2022` unaffected) |

## 6. What survives, and what the paper may say

* **Survives:** D16's selection on `val_2022`; the D16 test read; the Phase 27 fair
  ML-vs-DL comparison; the Phase 31/32 robustness grids; the MI-7 rematch. These were
  always per-cell or per-year and are not affected by this defect.
* **Corrected:** any statement about the MI-8 feature block across the seven dev cells,
  the MI-8 arm means, the D16 SEDI means, and the dev-cell D7 floors. Use the values in
  §4, not the archived MI tables.
* **Withdrawn:** the MI-3/MI-3c 7-cell aggregates and the MI-5 non-D16 7-cell columns.
  Their `val_2022` results stand.
* **Never to be used as a headline:** a single mean over the seven dev cells. Four of the
  seven are spatial folds with near-saturated PR-AUC, so the mean is dominated by
  cell composition, not by model quality (D8). The per-cell and per-year results are the
  defensible evidence.

## 7. Per-cell appendix (audit trail)

Fixed-config PR-AUC by cell (season = all):

| cell | FULL | FULL_V2 | FULL_V3 | V3 − V2 |
|---|---|---|---|---|
| val_2020 | 0.976802 | 0.977645 | 0.977893 | +0.000248 |
| val_2021 | 0.967905 | 0.969465 | 0.970646 | +0.001181 |
| val_2022 | 0.954634 | 0.955890 | 0.956471 | +0.000581 |
| spatial_fold_1 | 0.944546 | 0.943390 | 0.951304 | +0.007915 |
| spatial_fold_2 | 0.983079 | 0.984277 | 0.985474 | +0.001197 |
| spatial_fold_3 | 0.975349 | 0.978136 | 0.978794 | +0.000658 |
| spatial_fold_4 | 0.946043 | 0.952706 | 0.953993 | +0.001287 |

Frozen arms by cell and horizon:

| cell | horizon | D16 | MI-8 full_v3 | MI-8 full_v2 |
|---|---|---|---|---|
| val_2020 | 24 h | 0.987258 | 0.987644 | 0.987654 |
| val_2020 | 48 h | 0.978260 | 0.977751 | 0.977865 |
| val_2020 | 72 h | 0.966793 | 0.966691 | 0.966793 |
| val_2021 | 24 h | 0.982921 | 0.983208 | 0.982786 |
| val_2021 | 48 h | 0.971287 | 0.971195 | 0.970175 |
| val_2021 | 72 h | 0.957018 | 0.959763 | 0.957018 |
| val_2022 | 24 h | 0.970789 | 0.970662 | 0.971983 |
| val_2022 | 48 h | 0.956485 | 0.956765 | 0.956365 |
| val_2022 | 72 h | 0.945007 | 0.944737 | 0.945007 |
| spatial_fold_1 | 24 h | 0.965159 | 0.971542 | 0.966392 |
| spatial_fold_1 | 48 h | 0.944640 | 0.953056 | 0.943916 |
| spatial_fold_1 | 72 h | 0.924990 | 0.933707 | 0.924990 |
| spatial_fold_2 | 24 h | 0.992182 | 0.992835 | 0.991660 |
| spatial_fold_2 | 48 h | 0.985659 | 0.986949 | 0.985485 |
| spatial_fold_2 | 72 h | 0.976138 | 0.978577 | 0.976138 |
| spatial_fold_3 | 24 h | 0.987692 | 0.988955 | 0.988262 |
| spatial_fold_3 | 48 h | 0.979596 | 0.979682 | 0.978883 |
| spatial_fold_3 | 72 h | 0.968171 | 0.968939 | 0.968171 |
| spatial_fold_4 | 24 h | 0.967271 | 0.969891 | 0.969584 |
| spatial_fold_4 | 48 h | 0.954161 | 0.956745 | 0.953864 |
| spatial_fold_4 | 72 h | 0.935707 | 0.940426 | 0.935707 |

## 8. Reproduce

```powershell
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py   # ~2216 s
.venv\Scripts\python.exe scripts\mi10_report.py
```

Deterministic given seed `20261008`. Dev cells only. `test_rows_read = 0`.
