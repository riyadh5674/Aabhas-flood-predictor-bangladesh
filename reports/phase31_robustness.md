# Phase 31 — Robustness analysis (stage 53; phase0 §3 evidence)

**Status:** DONE · **Seed:** `20261008` · **Rule:** row cost `C_FN·FN + C_FP·FP`, primary
`C_FN : C_FP = 10 : 1`, exact cost argmin per slice.
**Dataset:** `data/processed/experimental_dataset.csv` (200,665 × 67 model columns),
SHA-256 `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` — unchanged.
**Test period (`>= 2023`) never read:** `test_rows_read = 0`.

---

## 0. What this phase asks

`final workflow.md` **Phase 31** asks how the *surviving* model(s) behave under changed
evaluation conditions — different years, unseen stations, monsoon vs non-monsoon, high vs
extreme rainfall, low-data (missingness-heavy) windows, all three horizons, and the
**stricter issue-time variant** of phase0 §3 (features through `t-1`, window `t…t+h`) — and
requires **every result per cell, never pooled (D8)**.

## 1. Method

* **Surviving panel** (`scripts/phase31_robustness.py`): the Phase-27/28/29/30 leaders
  `lightgbm` (24 h), `xgboost` (48 h), `xgboost_selected` (72 h), plus the references
  `logreg_linear`, `persistence` (D7) and `climatology`.
* **No new search, no new architecture.** Each model is the *frozen* Phase-16/20 winner
  re-fit on each Phase-15 dev protocol: `lightgbm_c00`, `xgboost_c01`,
  `xgboost_selected_c00` (configs read straight from `phase20_search.csv`); L1 models come
  from `baseline_model.build_models`. Class-imbalance handling is the winning per-family
  setting (unweighted for all three leaders).
* **Cells (7, per D8):** `val_2020`, `val_2021`, `val_2022` (P2 rolling origin) and the four
  unseen-station `spatial_fold_1…4` (P3). Each × {all, monsoon, non-monsoon} × {24 h, 48 h,
  72 h}.
* **Decision point:** for every slice the operating threshold is that slice's exact 10:1 cost
  argmin (`phase29_thresholds.best_exact`). A second column (`recall_frozen`/`far_frozen`)
  applies the **frozen Phase-29 P1 `val_2022` threshold** to every cell to separate ranking
  loss from threshold-transfer loss; for cells other than `val_2022` this is explicitly a
  *transfer* application.
* **Rainfall-intensity buckets:** cell-relative `rain_7d` quantiles — `low` ≤ p50,
  `high` (p50, p90], `extreme` > p90 — measured on each cell's eval rows (the boundaries are
  printed in `phase31_rainfall.csv` as `q50`/`q90`).
* **Missingness buckets:** per-row NaN fraction over the model's own feature set —
  `none` = 0, `low` (0, 0.05], `high` (0.05, 0.15], `very-high` > 0.15.
* **Stricter issue-time variant:** every feature column is replaced by its value on the
  previous **calendar day** within the same station (a within-station `t-1` shift, valid for
  98.5 % of rows); the model is re-fit on the shifted training rows and scored on the shifted
  eval rows. Labels are unchanged, so the comparison is "same target window, one day earlier
  information".

## 2. Cross-check: reuse of the Phase-27 generalisation table

Phase 27 already scored all 24 models on exactly these dev cells through one code path. The
fresh Phase-31 re-fit of the six-model panel reproduces the Phase-27
`phase27_generalisation.csv` numbers within floating-point noise
(`reports/phase31_reuse_check.csv`; max |Δ PR-AUC| reported in §7). The Phase-27 table is
therefore sound to *reuse as context* for the models Phase 31 does not re-fit (all DL models,
`random_forest`, …).

For reference, the **reused** Phase-27 PR-AUC (per cell, all-season) of the leaders and of the
best DL contrast `fusion_L3`:

| model | val_2020 | val_2021 | val_2022 | fold_1 | fold_2 | fold_3 | fold_4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm (24 h) | 0.9852 | 0.9802 | 0.9682 | 0.9628 | 0.9909 | 0.9858 | 0.9611 |
| xgboost (24 h) | 0.9862 | 0.9815 | 0.9681 | 0.9659 | 0.9910 | 0.9866 | 0.9592 |
| xgboost_selected (24 h) | 0.9854 | 0.9819 | 0.9663 | 0.9614 | 0.9909 | 0.9861 | 0.9577 |
| — 48 h — | | | | | | | |
| lightgbm | 0.9750 | 0.9673 | 0.9538 | 0.9353 | 0.9841 | 0.9755 | 0.9492 |
| xgboost | 0.9772 | 0.9688 | 0.9549 | 0.9409 | 0.9829 | 0.9758 | 0.9475 |
| xgboost_selected | 0.9757 | 0.9717 | 0.9533 | 0.9352 | 0.9823 | 0.9743 | 0.9437 |
| — 72 h — | | | | | | | |
| lightgbm | 0.9645 | 0.9526 | 0.9384 | 0.9222 | 0.9761 | 0.9651 | 0.9358 |
| xgboost | 0.9670 | 0.9534 | 0.9408 | 0.9269 | 0.9754 | 0.9637 | 0.9314 |
| xgboost_selected | 0.9644 | 0.9580 | 0.9422 | 0.9147 | 0.9710 | 0.9606 | 0.9292 |
| fusion_L3 (best DL, 24 h) | 0.9682 | 0.9652 | 0.9444 | **0.7657** | 0.9848 | 0.9787 | 0.9322 |
| persistence (D7, 24 h) | 0.9678 | 0.9407 | 0.9203 | 0.9089 | 0.9766 | 0.9620 | 0.9006 |
| logreg_linear (24 h) | 0.9679 | 0.9480 | 0.9401 | 0.8747 | 0.9704 | 0.9551 | 0.9107 |

Read-out: on the **temporal** cells the leaders track the year difficulty (val_2020 > val_2021
> val_2022 — the harder labels sit in 2022), and on the **unseen-station** folds they stay
above 0.95 at 24 h while the best DL model (`fusion_L3`) **collapses to 0.77 on fold 1** — a
robustness gap the pooled Phase-27 headline does not surface.

<!-- fresh numbers (reports/phase31_*.csv, scripts/phase31_run.log, exit 0) -->

## 3. Rainfall intensity: high vs extreme

`reports/phase31_rainfall.csv` — 378 rows (6 models × 7 cells × 3 horizons × 3 buckets). Buckets
are **cell-relative** (q50/q90 printed per cell × horizon), so they adapt to each regime rather
than fixing one national threshold (e.g. 24 h `val_2020` q50/q90 = 89.5/186.4 mm vs `val_2022`
59.1/152.7 mm).

| horizon | bucket | base-rate median | leader PR-AUC median | leader PR-AUC min | leader recall min |
| --- | --- | --- | --- | --- | --- |
| 24 h | low ≤ p50 | 0.0629 | 0.9885 | 0.9569 | 0.9483 |
| 24 h | high (p50, p90] | 0.1129 | 0.9836 | 0.9619 | 0.9545 |
| 24 h | **extreme > p90** | 0.1538 | **0.9484** | 0.8852 | 0.9399 |
| 48 h | low | 0.0682 | 0.9816 | 0.9272 | 0.9247 |
| 48 h | high | 0.1278 | 0.9744 | 0.9398 | 0.9332 |
| 48 h | **extreme** | 0.1694 | **0.9352** | 0.8862 | 0.8944 |
| 72 h | low | 0.0740 | 0.9705 | 0.9003 | 0.8942 |
| 72 h | high | 0.1409 | 0.9593 | 0.9191 | 0.9108 |
| 72 h | **extreme** | 0.1852 | **0.9219** | 0.8607 | 0.8927 |

Read-out. Extreme-rain windows roughly **double the base rate** (24 h 0.154 vs 0.063) yet
**lower** the leader PR-AUC — extreme − low = −0.0401 (24 h), −0.0464 (48 h), −0.0486 (72 h).
Because average precision normally *rises* with prevalence, a fall despite ~2.5× the positives
means the score separation genuinely degrades when rain is extreme: many non-flood rows also
carry high scores. The cost-optimal threshold holds recall (min 0.9399 / 0.8944 / 0.8927 at
24/48/72 h) but at a rising alarm rate — median `far` climbs 0.330 → 0.366 → 0.433 across the
three horizons. The worst cells are `val_2021` 72 h for all three leaders (PR-AUC 0.861–0.868,
`far` ≈ 0.50) and `spatial_fold_4` 72 h (0.873–0.876).

**The ML edge over D7 persistence is smallest here.** Across the 63 extreme-bucket
(model, cell, horizon) points the leaders beat persistence in only **28/63**, with losses up to
−0.035 (e.g. `val_2020` 48 h: leaders 0.920–0.923 vs persistence 0.9559). When rainfall is
already extreme, yesterday's observed water level is a strong carry-forward predictor, so the
learned models' advantage narrows sharply.

## 4. Missingness-heavy (low-data) windows

`reports/phase31_missingness.csv` — 336 rows (6 models × 7 cells × 3 horizons × {none, low, high,
very-high}, empty buckets omitted). The NaN fraction is taken over each model's *own* feature
set: `none` = 0, `low` (0, 0.05], `high` (0.05, 0.15], `very-high` > 0.15. Row totals (summed
over models/cells/horizons): `none` 508,053 · `high` 473,201 · `low` 10,022 · `very-high` 9,101.

| bucket | median PR-AUC | min PR-AUC | min recall | min `n` | min `n_pos` |
| --- | --- | --- | --- | --- | --- |
| none | 0.9664 | 0.9021 | 0.9203 | 1,598 | 231 |
| low | 0.9765 | 0.5889 | 0.6667 | 71 | 0 |
| high | 0.9667 | 0.5545 | 0.5000 | 144 | 3 |
| very-high | 1.0000 | 0.6426 | 0.6000 | 26 | 1 |

The **median** PR-AUC is flat across the four buckets (0.9664–1.0000): input missingness does not
systematically degrade the ranking. Every low minimum is a small-`n` instance: the global min
0.5545 is `xgboost_selected` / `spatial_fold_1` / 72 h / high-missingness with **4 positives**;
the other minima (0.5583, 0.5889, 0.6426) have ≤ 5 positives and sit in the hardest station fold.
These are noise, not a missingness-driven collapse. Consistent with that, the leaders still beat
persistence in 192/231 bucket points, and the 39 misses are the same thin instances. The
`very-high` bucket is not easier or harder — it is simply tiny (median `n_pos` = 4), so its
median PR-AUC of 1.0 is uninformative.

## 5. Stricter issue-time variant (features through `t-1`)

`reports/phase31_issuetime.csv` — 252 rows (6 models × 7 cells × 3 horizons × {standard,
stricter_t-1}). Every feature is replaced by its previous-calendar-day (within-station) value;
the model is re-fit on the shifted train rows and re-scored (labels unchanged). The shift is
defined for 197,673 / 200,665 rows (98.5 %).

| variant | leader PR-AUC min | median | max | recall min |
| --- | --- | --- | --- | --- |
| standard | 0.9155 | 0.9655 | 0.9912 | — |
| **stricter `t-1`** | **0.8480** | **0.9380** | 0.9787 | 0.9072 |

Loss (`stricter − standard`) is **always negative**: range −0.0675 … −0.0125, median −0.0269.
It grows with horizon (median −0.0251 / −0.0270 / −0.0296 at 24/48/72 h) and concentrates in the
hardest cells — `spatial_fold_1` median −0.0588, `val_2022` −0.0485, `spatial_fold_4` −0.0368,
against −0.0158 / −0.0171 in folds 2/3. Removing the same-day observation raises the re-tuned
cost-optimal threshold (median +0.0087) and costs a median 0.0149 of recall (max 0.031), though
recall stays ≥ 0.9072 everywhere.

**The D7 margin is not robust to this convention.** In 19/63 (model, cell, horizon) points the
stricter variant slips *below* persistence, worst −0.0152 (`xgboost_selected` / `spatial_fold_1`
/ 48 h: 0.8698 vs 0.8850); the 24 h/48 h folds lose most, where one day of information matters
most. The shortfall is bounded (≤ 0.0152) and on the easier cells the margin is still large
(best +0.0399), so the models are not overturned — but the Phase-29 "always beats D7" claim must
be qualified to "holds under the standard issue convention, and is marginal at 24–48 h on unseen
stations if the same-day observation is withheld".

## 6. Named findings and limitations

**F31-1 — reuse validated.** A fresh re-fit of the six-model panel reproduces
`phase27_generalisation.csv` to **max |Δ PR-AUC| = 1.1 × 10⁻¹⁶** over 126 shared rows, so the
24-model Phase-27 table is sound context for the models Phase 31 does not re-fit.

**F31-2 — leaders beat D7 everywhere (all-season).** 63/63 (cell × horizon) points, min margin
+0.0143 (`xgboost_selected` / `spatial_fold_2` / 24 h). The DL contrast confirms the robustness
gap: leader min fold PR-AUC 0.9147 vs `fusion_L3` fold-1 24 h 0.7657.

**F31-3 — year difficulty ordering.** val_2020 > val_2021 > val_2022 holds for all three leaders
at every horizon.

**F31-4 — season edge case.** On the temporal cells the monsoon/non-monsoon PR-AUC gap is
≤ 0.0435, but on the unseen-station folds it reaches **0.2586** (`spatial_fold_1`, 72 h: monsoon
0.9339 vs non-monsoon 0.7069), driven by a non-monsoon base rate that collapses to 0.0098–0.0155
(24–72 h, vs 0.148–0.179 in monsoon) while the training rows are monsoon-dominated. Leaders still
beat persistence in **126/126** seasonal points, so this is a slice where all models weaken, not a
comparative loss.

**F31-5 — extreme rainfall is the hardest ranking regime and narrows the D7 edge.** See §3:
PR-AUC falls ~0.04–0.05 below the low-rain bucket despite a higher base rate; leaders win only
28/63 extreme-bucket points.

**F31-6 — missingness does not degrade the median.** See §4: flat median PR-AUC across buckets;
all minima are ≤ 5-positive instances.

**F31-7 — stricter issue-time loses 0.0125–0.0675 PR-AUC and can slip below D7.** See §5:
bounded but real, and the D7 margin is not robust at 24–48 h on the unseen-station folds.

**F31-8 — per-cell reporting (D8).** Every row above is a per-cell/per-slice result; no cell or
season is pooled.

**Limitations.**
* The stricter variant is a within-station *calendar-day* shift (valid 98.5 % of rows), not a
  full re-issued-forecast pipeline; rows without a previous day are dropped from that slice.
* Rainfall and missingness buckets are cell-relative and computed on each cell's all-season eval
  rows; season-specific rainfall/missingness slices are out of scope here (reported separately in
  §4 of this phase only as `all`).
* Thin buckets (`very-high` missingness, non-monsoon unseen-station) yield noisy extrema; they are
  flagged as small-`n`, not interpreted as skill.
* The panel is fixed to the Phase-16/20 winners at their target horizons; Phase 31 performs no new
  model selection, so it measures robustness of the chosen models, not of the search.
* DL context is reused from Phase-27 (justified by F31-1) rather than re-fitted.

## 7. Provenance

* `scripts/phase31_robustness.py` — driver (exit 0; seed `20261008`; wall-clock 772 s and the
  dataset SHA in `scripts/phase31_run.log`).
* `reports/phase31_cells.csv` — per cell × season × horizon × model (378 rows).
* `reports/phase31_rainfall.csv`, `reports/phase31_missingness.csv`,
  `reports/phase31_issuetime.csv` — the three sensitivity slices (378 / 336 / 252 rows).
* `reports/phase31_reuse_check.csv` — Phase-31 fresh vs Phase-27 generalisation
  (126 rows; max |Δ PR-AUC| = 1.1 × 10⁻¹⁶).
* `reports/phase31_summary.json` — seed, hardware, SHA, `test_rows_read = 0`, headline.
* `figures/phase31_robustness.png`.
* `notebooks/31_robustness.ipynb` (built by `scripts/build_phase31_notebook.py`, executed via
  `jupyter nbconvert --execute`, exit 0) — independent verification, checks **N31-01…N31-10**,
  **10/10 PASS**.
