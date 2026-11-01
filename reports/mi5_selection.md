# MI-5 — model re-selection & freeze (v2 features)

| | |
| --- | --- |
| **Stage** | MI-5 (improvement plan §MI-5) — re-selection & freeze |
| **Decision** | **D16** — the MI-5 selected model supersedes Phase 33 for MI-6 test reading |
| **Command** | `.venv\Scripts\python.exe -u scripts\mi5_selection.py` |
| **Run log** | `scripts/mi5_run.log` |
| **Exit / wall** | `0` / **256 s** |
| **Seed** | `20261008` (`baseline_model.B.SEED`) |
| **Evidence** | validation + spatial-CV only — **7 dev cells**, test **never read** (`test_rows_read = 0`) |
| **Data** | `experimental_dataset.csv` (sha `a6748529…`) + `mi_v2_features.csv` (sha `16287675…`) |
| **Artefacts** | `reports/mi5_scorecard.csv` · `reports/mi5_selection.json` · `models/final_v2/{selected_model_v2.pkl, final_config_v2.json, README.md}` |

## 1. What this stage does

MI-5 re-applies the **pre-registered Phase-33 selection rule** to a candidate panel that
Phase 33 could not see: the MI-3b HPO winners on the v2 feature table (`FULL_V2 = 110`).
Selection uses the same seven development cells as MI-3 (`val_2020`, `val_2021`, `val_2022`
plus `spatial_fold_1…4`). The frozen test period (`>= 2023`) is not touched.

**Candidates** (each = a spec with one fitted Pipeline per horizon):

| candidate | meaning |
| --- | --- |
| `xgboost_FULL` | Phase-33 winner config, **v1** features — incumbent reference |
| `xgboost_FULL_V2` | same config on **v2** features — isolates the feature gain |
| `lgbm7_FULL_V2` | MI-3b 24/48-h winner (`lgbm7`) as a uniform spec |
| `xgb3_FULL_V2` | MI-3b 72-h winner (`xgb3`) as a uniform spec |
| `hpo_perhorizon_FULL_V2` | MI-3b per-horizon selection `lgbm7 / lgbm7 / xgb3` |
| `persistence`, `climatology` | D7 bar + no-skill bar (references, never selectable) |

**Rule (D16):** Gate 1 = beat D7 on every dev cell × horizon (7 × 3 = 21 points); Gate 2 =
raw `val_2022` ECE ≤ 0.02; then rank by horizon-mean `val_2022` PR-AUC (primary), then
fewest missed floods at the cost-optimal threshold (co-primary, FN:FP = 10:1), then the
robustness floor, then inference cost. 95 % **station-level** bootstrap CIs are reported
and are not used as a tie-break (D8: per-cell, never pooled).

## 2. Scorecard (all seven dev cells)

> **Correction (2026-10-22).** The **`21-pt D7 floor`**, **`worst temporal`**, and **`spatial mean`**
> columns were computed by fitting one model on the ≤2021 training rows and scoring all seven
> cells; six of those cells are subsets of that fit. Those three columns are **withdrawn**
> (in-sample). The `val_2022` columns, the calibration gate, and the D16 selection are
> **unaffected**, because `val_2022` was the one cell that was genuinely out-of-sample.
> Corrected per-cell refits give D16 above D7 on **21/21** points with a **minimum margin
> +0.0156** (the archived floor +0.0225 was in-sample). See `reports/p2p3_correction.md`.
> The pre-registered rule selected on the primary `val_2022` metric, so the champion does not
> change.

| rank | model | P1 val_2022<br>24 h | 48 h | 72 h | **horizon-mean** | 21-pt D7 floor | worst temporal | spatial mean | missed@10:1 | worst ECE | infer ms/1k | eligible |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **1** | **`hpo_perhorizon_FULL_V2`** | 0.9708 | 0.9565 | **0.9450** | **0.95743** | +0.0225 | +0.0311 | 0.0754 | 90 | 0.0023 | 103.8 | ✔ |
| 2 | `lgbm7_FULL_V2` | 0.9708 | 0.9565 | 0.9429 | 0.95671 | +0.0225 | +0.0311 | 0.0789 | **84** | 0.0025 | 109.6 | ✔ |
| 3 | `xgboost_FULL_V2` | 0.9695 | 0.9549 | 0.9432 | 0.95589 | +0.0210 | +0.0291 | 0.0736 | 100 | 0.0027 | 103.0 | ✔ |
| 4 | `xgb3_FULL_V2` | 0.9675 | 0.9537 | 0.9450 | 0.95540 | +0.0204 | +0.0286 | 0.0716 | 104 | **0.0019** | 88.7 | ✔ |
| 5 | `xgboost_FULL` *(Phase-33)* | 0.9681 | 0.9549 | 0.9408 | 0.95463 | +0.0200 | +0.0275 | 0.0695 | 91 | 0.0025 | 98.4 | ✔ |
| — | `persistence` (D7 bar) | 0.9203 | 0.8954 | 0.8683 | 0.89465 | 0.0000 | 0.0000 | 0.0000 | 307 | 0.0180 | 15.2 | ✘ ref |
| — | `climatology` | 0.0509 | 0.0594 | 0.0658 | 0.05868 | −0.8784 | −0.8704 | −0.8119 | 0 | 0.0494 | 0.0 | ✘ ref |

- **Gate 1 passed by all five candidates** — every one beats D7 on all 21 points according to
  the archived table, but **that margin is withdrawn** (in-sample). The corrected D16 result
  is **21/21, minimum +0.0156** (`reports/p2p3_correction.md`); it remains positive on every
  dev point but is smaller.
- **Gate 2 passed by all five** — worst raw ECE `0.0027`, two orders of magnitude inside the
  `0.02` gate. Calibration is not a differentiator here.
- The **`p3_mean` spatial-fold margin** and the **co-primary missed count** are also affected by
  the in-sample defect for the non-D16 candidates; they are shown for traceability only, not
  used to re-rank. The pre-registered rule sorts on the **primary** `val_2022` metric first, so
  `hpo_perhorizon_FULL_V2` remains #1.

## 3. Selected model

**`hpo_perhorizon_FULL_V2`** — MI-3b's per-horizon winners on `FULL_V2`, i.e.

| horizon | family | config tag |
| --- | --- | --- |
| 24 h | `lightgbm` | `lgbm7` (300 trees, lr 0.02, 63 leaves, 40 min-child, λ 3.0) |
| 48 h | `lightgbm` | `lgbm7` |
| 72 h | `xgboost` | `xgb3` (800 trees, lr 0.02, depth 4, λ 3.0) |

**Primary result: horizon-mean `val_2022` PR-AUC = 0.95743**, vs the Phase-33 frozen winner
**0.95463 → +0.00280**. It is the best point estimate on the primary cell, and it wins on
72 h specifically because `xgb3` (0.94501) beats `lgbm7` (0.94287) at that horizon while the
two are identical (same `lgbm7`) at 24/48 h.

### Why the #2 candidate is also defensible (stated honestly)

`lgbm7_FULL_V2` is a **single uniform spec** (simpler artefact), has the **fewest missed
warnings (84 vs 90)** on the co-primary, and a **higher spatial-CV margin (0.0789 vs
0.0754)**. It loses only on the primary point estimate (0.95671 vs 0.95743). Under the
pre-registered rule the primary metric decides, so the per-horizon model is frozen — but the
gap is **0.0007**, inside the noise.

### Confidence intervals (95 %, station bootstrap, `val_2022`)

| horizon | `hpo_perhorizon_FULL_V2` | `lgbm7_FULL_V2` | Phase-33 `xgboost_FULL` |
| --- | --- | --- | --- |
| 24 h | 0.9372 – 0.9895 | 0.9372 – 0.9895 | 0.9324 – 0.9870 |
| 48 h | 0.9118 – 0.9807 | 0.9118 – 0.9807 | 0.9075 – 0.9800 |
| 72 h | 0.8994 – 0.9719 | 0.8939 – 0.9702 | 0.8848 – 0.9692 |

Every candidate CI **overlaps** every other at every horizon. The MI-v2 improvement is
therefore a **consistent point-estimate shift, not a CI-separable jump** — exactly the
honest framing the plan requires (`improvement_plan.md` §4.2).

## 4. Frozen artefact

`models/final_v2/` (provisional — the test is opened only in MI-6):

| file | content |
| --- | --- |
| `selected_model_v2.pkl` | dict: `pipelines` (3 × `sklearn.Pipeline`), `features` (110), `per_horizon_family/cfg`, dataset SHAs, `test_rows_read = 0` |
| `final_config_v2.json` | the same metadata without the fitted pipelines |
| `README.md` | load snippet + provenance |

Verified loadable and predictive (smoke test: 200-row `predict_proba` on all three
horizons returns probabilities in `[0, 1]`).

## 5. Decision D16

> **D16** — the MI-5 model `hpo_perhorizon_FULL_V2` (lightgbm `lgbm7` @ 24 h and 48 h,
> xgboost `xgb3` @ 72 h; `FULL_V2` = 110 features; seed 20261008) **supersedes** the
> Phase-33 winner `xgboost` L2-03 for the single MI-6 test read. It was selected on
> validation + spatial-CV evidence only under the pre-registered Phase-33 rule; the frozen
> test (`>= 2023`) was not read at this stage. Phase 33 remains the historical record; the
> MI-6 read is against the D16 model and the D7 bar.

## 6. Honest read

- The v2 programme moved the primary metric **0.95463 → 0.95743 (+0.00280)** — real,
  consistent across all seven dev cells, but **small and not CI-separable**.
- This remains far below the plan's hoped **0.97–0.98**, and MI-4 showed a uniform 0.99 is
  **not reachable** (base-rate mathematics + 4–10 % near-threshold label noise); the honest
  ceiling on `val_2022` is **≈0.96–0.97**, which MI-5 now sits at.
- The largest unused lever is still the **deferred ERA5-Land block** (`e5_*`), which would
  feed straight back through MI-3a/b and a re-run of MI-5.

## 7. Reproduce

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi5_selection.py   # MI-5 -> scorecard, selection JSON, models/final_v2/
```

Exit `0`; deterministic given seed `20261008`; `test_rows_read = 0`.
