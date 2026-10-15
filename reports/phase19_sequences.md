# Phase 19 — Sequence construction & validation

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 19 — Sequence construction & validation** → checklist **stage 33** |
| **Notebook** | `notebooks/09_sequence_validation.ipynb` — **28/28 green (S1a…S7d), 0 warnings, 26.9 s** (built by `scripts/build_phase19_notebook.py`, executed with the `flood-predictor` kernel) |
| **Run log** | `scripts/phase19_run.log` (exit 0, seed 20261008) |
| **Outputs** | `sequence_mask_L{3,7,14,30}.npy` · `sequence_policy.json` (6,656 B) · `sequence_manifest.json` amended (`missing_timestep_policy`, original 5 hashes re-checked unchanged) |
| **Input (immutable)** | `experimental_dataset.csv` SHA-256 `a67485…` re-verified at the end of the run |
| **Binding decisions** | **D5** information boundary (issue at 24:00 of *t*) · **D12** fair L ∈ {3,7,14,30} comparison · **R6** the 2018 archive gap · policy **P19-M1** |

**Gate result: PASS.** No sequence model is trained until this notebook is green — it is
the deep-learning counterpart of the Phase-13 leakage audit.

## 1. S1 — chronological order

`sequence_index.csv`: 200,665 rows, `sample_idx` = 0…n-1 contiguous; inside every one of
the 133 stations rows are strictly chronological and unique (0 duplicates). Years
covered: 2017, 2019–2026 (**2018 absent — R6 confirmed**).

## 2. S2 — continuity / gap distribution

Calendar 2017-01-06 … 2026-10-05 = 3,560 days; 200,665 / 473,480 station-days present
(42.4 %). **An empty timestep is exactly an absent calendar day** — checked against an
independently rebuilt calendar grid, **0 mismatches** at every L:

| L | fully consecutive | ≥ 1 absent day | ≥ 3 absent | mean empty | max run |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 97.1 % | 2.9 % | 0.0 % | 0.04 | 2 |
| 7 | 92.5 % | 7.5 % | 3.4 % | 0.21 | 6 |
| 14 | 85.9 % | 14.1 % | 8.4 % | 0.68 | 13 |
| 30 | 73.3 % | 26.7 % | 18.2 % | 2.53 | 29 |

R6: 0 rows in 2018; windows whose history reaches back into 2018: 42 / 68 / 177 / 224
for L = 3/7/14/30 — all their 2018 timesteps are empty (no shift, no fabrication).

## 3. S3 — missing-timestep policy → rule P19-M1

### Candidate costs (samples dropped, per L)

| rule | L=3 | L=7 | L=14 | L=30 |
| --- | ---: | ---: | ---: | ---: |
| **R1 keep-all + mask** (chosen) | **0 (0 %)** | **0 (0 %)** | **0 (0 %)** | **0 (0 %)** |
| R2 reject any gap | 5,719 (2.9 %) | 14,974 (7.5 %) | 28,380 (14.1 %) | 53,562 (26.7 %) |
| R3 min observed (> L/4 empty → drop) | 5,719 (2.9 %) | 12,383 (6.2 %) | 13,602 (6.8 %) | 23,644 (11.8 %) |
| R4 ffill age ≤ 2 then reject | 5,630 (2.8 %) | 11,168 (5.6 %) | 19,750 (9.8 %) | 37,385 (18.6 %) |

### Bias probe on the strongest alternative (R2, L=30)

* 26.69 % of samples dropped, **and unevenly**: monsoon 12.9 % vs non-monsoon 43.9 %
  (**3.4× season asymmetry**); per split train 25.1 % / val 31.2 % / test 27.0 %;
* positive rate among dropped windows 3.18 % vs 5.63 % among kept — rejection would
  shift the label distribution, L-dependently (D12 fairness broken).

### Decision — **P19-M1 “keep-all + observation mask channel”**

> No timestep and no sample is dropped or imputed. An absent calendar day stays
> all-NaN in the tensor and is handed to the model as `mask=0`
> (`sequence_mask_L{L}.npy`, uint8, observed = 1). Variable-level NaNs on present days
> (`precip_mm`, `tmax_c`, `tmin_c` — 0.027 % of observed timesteps) are replaced by 0
> **after** the training fold's standardisation (fold mean → no information).
> Phase 21 input = `concat(12 variables, mask)` = **13 channels**; a plain NaN must
> never reach a layer.

Why: (i) identical sample sets across L keeps the D12 sweep comparable; (ii) nothing
fabricated — interpolation/age-2 filling invents driver variables; (iii) the cheapest
rejected rule still costs 2.9–26.7 % with season/split bias.

### Executed outputs

| file | shape | size | notes |
| --- | --- | ---: | --- |
| `sequence_mask_L3.npy` | (200665, 3) uint8 | 602,123 B | == observed-day indicator, all 601,995 timesteps |
| `sequence_mask_L7.npy` | (200665, 7) uint8 | 1,404,783 B | 1,404,655 timesteps ✓ |
| `sequence_mask_L14.npy` | (200665, 14) uint8 | 2,809,438 B | 2,809,310 timesteps ✓ |
| `sequence_mask_L30.npy` | (200665, 30) uint8 | 6,020,078 B | 6,019,950 timesteps ✓ |

`sequence_manifest.json` gained the `missing_timestep_policy` section (rule id, statement,
why, mask-file SHA-256s, `amended_by: phase19`) — the **5 original artifacts' hashes are
re-verified unchanged** (S3e). Full counterfactual costs live in
`sequence_policy.json`.

## 4. S4 — no station mixing

2,036 timesteps sampled (40 windows × 4 lengths) each compared against their **own**
`(station, day)` row in the table: **0 violations** — a window containing another
station's day would mismatch.

## 5. S5 — no forward reach

For all 200,665 windows × 4 lengths: the newest day in the window **is** the anchor day
*t* and no day is later than *t* (max day index 3,559 = anchor). The label window
t+1…t+3 is therefore never inside a tensor; `label_*` excluded (re-verified). History
may legally reach **back** across a split boundary: 1,310 L=30 windows start in an
earlier year (1,024 test / 286 train) — past days at issue time are legal (D5).

## 6. S6 — label alignment (independent recompute)

100 random windows × 3 horizons, labels rebuilt from the **raw 3-hourly readings**
(same cleaning/danger-level contract as `scripts/audit_dataset.py` A5: drop dup
`st_id,wl_date`, sentinels ≤ −1000 / > 100 m, F8 rolling-9 median spike > 10 m,
danger level only from `ffwc_stations.csv::dangerlevel`): **292 comparable labels,
0 mismatches, 0 one-sided NaNs**.

## 7. S7 — split / holdout inheritance

* all 200,665 windows inherit anchor-day split / spatial_fold / station_holdout /
  season **1:1** with Phase 15 (0 disagreements);
* every training cell contains **0 holdout rows and 0 frozen-test rows** —
  P1 train<2022: 61,991 · P2 ≤2019/2020/2021: 28,711 / 44,614 / 61,991 ·
  P3 excl_fold 1–4: 46,793 / 45,727 / 47,207 / 46,246;
* 26 holdout stations appear in **0** training-cell station sets;
  `test_unseen` = 19,763 rows, all holdout; holdout total 39,725 rows.

## 8. Reproducibility

```
python scripts/build_phase19_notebook.py
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=flood-predictor notebooks\09_sequence_validation.ipynb
```

seed 20261008 · `experimental_dataset.csv` SHA-256 `a67485…` byte-stable at run end ·
28/28 checks (S1a…S7d) · 0 warnings · 26.9 s.

**Next → Phase 20 / stage 34:** classical machine learning, L2 rows
(`random_forest`, `gradient_boosting`, `xgboost`, `lightgbm`, `xgboost_selected`,
budget 40 configs, parity grid P1/P2/P3).
