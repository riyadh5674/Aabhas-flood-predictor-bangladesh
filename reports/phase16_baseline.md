# Phase 16 — Establish baseline models

| | |
| --- | --- |
| **Workflow** | **Phase 16 — Establish baseline models** — *before sophisticated ML, establish simple baselines* → `28` baseline models |
| **Checklist** | **stages 29 + 30, both closed here** — stage 30 = baseline models (this report), stage 29 = class-imbalance analysis → §10 |
| **Script** | `scripts/baseline_model.py` (576 lines) — a full rewrite of the pre-Phase-15 version |
| **Run** | `python -u scripts\baseline_model.py` → `scripts/phase16_run.log`, **exit 0, 312 s**, seed 20261008 |
| **Notebook** | `notebooks/08_baseline_models.ipynb` (17 cells) — **59/59 checks green**, 2 warnings, 8.3 s; S1–S7 verify the script's outputs without refitting, I1–I5 = stage 29. Built by `scripts/build_phase16_notebook.py` |
| **Outputs** | `reports/phase16_baseline_metrics.csv` (432 rows) · `phase16_test_metrics.csv` (378 rows) · `phase16_test_per_station.csv` (3,612 rows) · `data/processed/class_imbalance.json` (stage 29) |
| **Figures** | `figures/phase16_pr_skill.png` · `figures/phase16_calibration_test2025.png` · `figures/phase16_class_imbalance.png` |
| **Untouched** | `experimental_dataset.csv` SHA-256 `a6748529083fcdce…` (identical to Phase 15 S9). `splits.json`, `feature_sets.json` read-only |

No model selection, tuning or early stopping happens here (D10: candidate ML is Phase 17+).
The frozen test period is scored **once**, into its own file, and no decision in this phase reads it.

---

## 1. What the rewrite fixed

The old `scripts/baseline_model.py` was stopped in Phase 15 with four recorded defects
(`phase15_split_design.md` §5, `phase0_research_definition.md`):

| # | defect | fix |
| --- | --- | --- |
| 1 | LightGBM **early-stopped on the test fold** | no gradient boosting in this phase at all — the flaw is structurally gone |
| 2 | trained on `flood_dataset_model_ready.csv` | Phase-14 experimental table (200,665 × 67) merged 1:1 with Phase-15 `split_assignments.csv` |
| 3 | invented its own shuffled station folds | all masks built from Phase-15 columns only (`split`, `spatial_fold`, `station_holdout`, `season`) |
| 4 | no persistence baseline | **D7** (`wl_pct_danger` alone) is fit and printed first, next to every other model |

Additional guards enforced at run time:

* every mask filters `station_holdout` (Phase 15 stores holdout rows inside
  `split == train/val` — 21 of 26 holdout stations have pre-2023 rows);
* the threshold for precision/recall/F1 comes from **train predictions only**
  (`best_f1_threshold`), never from the cell being scored;
* the D7 model is asserted to have a **positive logistic coefficient** after fit, so its
  ranking stays identical to the raw `wl_pct_danger` score;
* asserted at load: train ≤ 2021, val = 2022, test ≥ 2023, zero shared dates, and the
  P2 train ≤ 2021 mask is byte-identical to the P1 train mask.

## 2. The six baselines (step 28)

| model | features | kind | notes |
| --- | --- | --- | --- |
| `climatology` | — | constant = train base rate | the "majority-class" reference; its PR-AUC *must* equal the cell's prevalence |
| `persistence` | `wl_pct_danger` (1) | logistic link, 1 feature | **D7 bar**, monotone link → same ranking as the raw score, valid probability for Brier/ECE |
| `logreg_linear` | LINEAR (35) | `LogisticRegression(C=1)` | the set Phase 12 mandates for linear models |
| `logreg_linear_no_up` | LINEAR − `up_*` (29) | same | **F14-2** ablation: drop the 67–70 %-NaN upstream columns |
| `logreg_selected` | SELECTED (10) | same | compact cross-check |
| `tree_depth4` | FULL (62) | `DecisionTreeClassifier(max_depth=4, min_samples_leaf=200)` | shallow, interpretable non-linearity |

Preprocessing inside every pipeline: median imputation (numerics) + train-fitted
**frequency encoding** of the 5 text columns (never target encoding), `StandardScaler` for
the logistic models, `remainder="drop"` so no column can enter by accident. Banned columns
(`station_id`, `date`, `label_*`, split bookkeeping) asserted absent from FULL.

Training rows per horizon (non-holdout, labels non-NaN): **61,977 / 60,992 / 60,144**
for 24 h / 48 h / 72 h. Fit time per model: climatology 0 s, persistence 0.18 s, logistic
regressions 0.28–1.2 s, tree 1.9 s.

## 3. Step 28 — primary result, validation 2022 (P1, season = all)

PR-AUC (bold = bar beaten), with Brier and ECE for calibration context. Base rate:
**5.09 % / 5.94 % / 6.58 %** (24/48/72 h).

| model | PR-AUC 24 h | 48 h | 72 h | ×persistence 24/48/72 | recall 24 h | Brier 24 h | ECE 24 h |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| climatology | 0.0509 | 0.0594 | 0.0658 | 0.06 | 0.000 | 0.0503 | 0.0450 |
| **persistence (D7)** | **0.9203** | **0.8954** | **0.8683** | 1.000 | 0.878 | 0.0118 | 0.0116 |
| **logreg_linear** | **0.9401** | **0.9241** | **0.9053** | **1.021 / 1.032 / 1.043** | 0.847 | 0.0099 | 0.0038 |
| logreg_linear_no_up | 0.9403 | 0.9242 | 0.9052 | 1.021 / 1.032 / 1.043 | 0.847 | 0.0098 | 0.0040 |
| logreg_selected | 0.9398 | 0.9206 | 0.9004 | 1.021 / 1.028 / 1.037 | 0.824 | 0.0102 | 0.0052 |
| tree_depth4 | 0.8972 | 0.8902 | 0.8735 | 0.975 / 0.994 / 1.006 | 0.891 | 0.0092 | 0.0032 |

Sanity checks that must hold and do: climatology PR-AUC = base rate **exactly**
(0.0509 = 5.09 %); every model beats climatology by 15–18×; climatology recall = 0 at its
0.5 threshold because 0.5 > train base rate 9.6 %.

**Answer to the step-28 question — "does sophisticated ML actually improve over simple
methods?"** Not much: a fully regularised logistic regression on 35 features beats the
one-feature persistence bar by only **+2.1 % to +4.3 %** relative on the selection cell,
and the shallow tree does not beat it at all at 24 h. The D7 bar is high and well
calibrated; Phase 17 candidates must clear it, not the climatology.

## 4. Rolling origin (P2, per year — D8: never pooled)

PR-AUC, season = all. Base rate collapses 14.18 % (2020) → 5.09 % (2022).

| model | val 2020 24/48/72 | val 2021 | val 2022 |
| --- | --- | --- | --- |
| climatology | 0.1418 / 0.1532 / 0.1632 | 0.0703 / 0.0776 / 0.0840 | 0.0509 / 0.0594 / 0.0658 |
| persistence | 0.9678 / 0.9559 / 0.9428 | 0.9407 / 0.9239 / 0.9034 | 0.9203 / 0.8954 / 0.8683 |
| logreg_linear | 0.9679 / 0.9521 / 0.9364 | 0.9480 / 0.9331 / 0.9132 | 0.9401 / 0.9241 / 0.9053 |
| logreg_selected | 0.9718 / 0.9591 / 0.9441 | 0.9489 / 0.9329 / 0.9123 | 0.9398 / 0.9206 / 0.9004 |
| tree_depth4 | 0.9657 / 0.9528 / 0.9359 | 0.9400 / 0.9315 / 0.9151 | 0.8972 / 0.8902 / 0.8735 |

Skill vs persistence (logreg_linear): **1.000 / 0.996 / 0.993** in 2020 → **1.008 / 1.010 /
1.011** in 2021 → **1.021 / 1.032 / 1.043** in 2022. The advantage over the bar grows as the
base rate falls; at the 14 % base rate of 2020 a linear model has essentially nothing to
add over the current river level.

## 5. Spatial CV (P3, unseen stations, train-period rows)

Mean over the 4 station-disjoint folds, season = all, with the station-bootstrap 95 % CI
(resample stations, 100 reps) for `logreg_linear`:

| model | PR-AUC 24 h | 48 h | 72 h | macro per-station 24 h |
| --- | ---: | ---: | ---: | ---: |
| climatology | 0.0960 | 0.1061 | 0.1152 | 0.1222 |
| **persistence (D7)** | **0.9370** | **0.9182** | **0.8979** | 0.8796 |
| logreg_linear | 0.9277 [0.851, 0.970] | 0.9032 [0.810, 0.952] | 0.8803 [0.780, 0.936] | **0.8865** |
| logreg_linear_no_up | 0.9295 | 0.9044 | 0.8821 | 0.8869 |
| logreg_selected | 0.9336 | 0.9097 | 0.8854 | **0.8936** |
| **tree_depth4** | **0.9401** | **0.9230** | **0.9022** | 0.8451 |

Per-fold skill vs persistence (`logreg_linear`): **0.962 / 0.941 / 0.935** (excl_fold_1),
0.994 / 0.991 / 0.988 (fold 2), 0.993 / 0.991 / 0.988 (fold 3), **1.011 / 1.010 / 1.009**
(fold 4) — i.e. the linear model is **below the D7 bar in 3 of 4 unseen-station folds**.
The tree stays within ±1.5 % of the bar in every cell (0.989–1.018) and is above it in
3 of 4 folds on average (fold means 1.008 / 0.994 / 1.003 / 1.015). Pooled PR-AUC says
tree ≳ persistence > logreg; the macro per-station companion (D8) reverses the first two
(logreg 0.8865 > persistence 0.8796 > tree 0.8451): the tree's pooled score is carried by
a few dense stations. **Both numbers are reported; neither alone decides Phase 17.**

## 6. Monsoon vs non-monsoon (D1, JJAS = monsoon)

Validation 2022 PR-AUC — the ranking is stable across seasons, only the base rate moves:

| model (24 h) | all | monsoon | non-monsoon |
| --- | ---: | ---: | ---: |
| persistence | 0.9203 | 0.9245 | 0.9131 |
| logreg_linear | 0.9401 | 0.9411 | 0.9387 |
| base rate | 0.0509 | 0.0684 | 0.0277 |

Frozen test, `test_trained` bucket — non-monsoon is much harder in absolute terms:

| 24 h | PR-AUC all | monsoon | non-monsoon | recall (all) |
| --- | ---: | ---: | ---: | ---: |
| base rate | 0.0135 | 0.0226 | **0.0029** | — |
| persistence | 0.7468 | 0.7618 | 0.6267 | 0.767 |
| logreg_linear | 0.7771 | 0.7931 | 0.6496 | 0.742 |

Non-monsoon carries a 0.29 % positive rate and PR-AUC 0.65 at 24 h, 0.51 at 72 h —
worth ~0.15 PR-AUC below the monsoon cell. Every headline in later phases must stay split
by season.

## 7. Frozen test (2023+) — recorded once, read by no decision

`reports/phase16_test_metrics.csv`, season = all, 24 h horizon. Per-year PR-AUC:

| model | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: |
| climatology (= prevalence) | 0.0068 | 0.0254 | 0.0117 | 0.0127 |
| persistence (D7) | 0.6276 | 0.8022 | 0.7328 | 0.7233 |
| logreg_linear | 0.6313 | 0.8198 | 0.7642 | 0.7461 |
| logreg_selected | 0.6497 | 0.8241 | 0.7473 | 0.7392 |
| tree_depth4 | 0.6198 | 0.8133 | 0.7094 | 0.7185 |

Buckets (D8, never pooled with years):

| PR-AUC (24 h) | `test_trained` (85 st) | `test_unseen` (26 st) | `test_new_network` (22 st) |
| --- | ---: | ---: | ---: |
| persistence | 0.6805 | **0.6792** | **0.6914** |
| logreg_linear | **0.7164** | 0.6774 | 0.6805 |
| logreg_selected | 0.7053 | **0.6935** | 0.7113 |
| tree_depth4 | 0.7042 | 0.6600 | 0.7010 |

Skill vs persistence, per year (logreg_linear): 1.006 / 1.022 / 1.043 / 1.032 (24 h) —
**but 0.997 on `test_unseen` and 0.984 on `test_new_network`.** The linear model's gain is
confined to stations it has seen; on genuinely unseen stations the D7 bar still matches or
beats it, and only `logreg_selected` clears it (+2.1 % unseen, +4.8 % new-network at 72 h).

Station-bootstrap 95 % CIs are present on every test cell (100 reps over stations, e.g.
`test_2025` 24 h: persistence 0.7328 [0.5287, 0.8326], logreg_linear 0.7642 [0.5739,
0.8673] — overlapping, so the 2025 gain is not separable at station level yet).
Per-station confusion counts for the two reference models are in
`phase16_test_per_station.csv`.

Calibration is good where it matters: ECE on test 2025 (24 h) — logreg_linear **0.0025**,
logreg_selected 0.0016, persistence 0.0059, climatology 0.0842 (figure
`phase16_calibration_test2025.png`). Train-derived thresholds transfer: val 2022
`logreg_linear` threshold 0.483 → recall 0.847 / precision 0.894 / specificity 0.995.

## 8. F14-2 — LINEAR with and without `up_*`

| cell (24 h) | `logreg_linear` | `logreg_linear_no_up` | Δ |
| --- | ---: | ---: | ---: |
| val 2022 PR-AUC | 0.9401 | 0.9403 | +0.0002 |
| val 2022 Brier | 0.0099 | 0.0098 | −0.0001 |
| spatial CV mean | 0.9277 | 0.9295 | +0.0018 |
| test trained / unseen | 0.7164 / 0.6774 | 0.7157 / 0.6769 | −0.0007 / −0.0005 |

The 67–70 %-NaN upstream columns contribute **nothing measurable** at baseline level —
differences are within noise and never favour keeping them. Open question for Phase 17:
drop `up_*` from the modelling feature set (keeps LINEAR at 29 columns) or keep for
consistency with Phase 12's frozen set?

## 9. Figures

* `figures/phase16_pr_skill.png` — PR-AUC per test year, all six models, three horizons
  (the D7 bar as one of the lines).
* `figures/phase16_calibration_test2025.png` — calibration curves, one panel per horizon.
* `figures/phase16_class_imbalance.png` — stage 29: positive rate per year + cumulative
  share of positives by station.

## 10. Stage 29 — class-imbalance analysis (workflow Phase 19)

Ran in `notebooks/08_baseline_models.ipynb`: **59/59 checks pass**, 2 warnings recorded,
8.3 s, no model refit. Evidence: `data/processed/class_imbalance.json`.

**Class distribution and imbalance ratio** (Phase-15 masks, holdout excluded from train/val):

| split | rows | 24 h positives · rate · ratio | 48 h | 72 h |
| --- | ---: | --- | --- | --- |
| train (≤ 2021) | 61,991 | 5,943 · 9.59 % · **9.4:1** | 6,465 · 10.60 % · 8.4:1 | 6,924 · 11.51 % · 7.7:1 |
| val (2022) | 17,538 | 891 · 5.09 % · **18.6:1** | 1,027 · 5.94 % · 15.8:1 | 1,124 · 6.58 % · 14.2:1 |
| test (2023+) | 101,174 | 1,425 · 1.41 % · **70.0:1** | 1,768 · 1.77 % · 55.5:1 | 2,094 · 2.12 % · 46.1:1 |

The whole table reproduces phase0 §4 exactly (24 h 4.98 % / 19.1:1 … 72 h 6.32 % / 14.8:1),
and every split × horizon cell still carries ≥ 800 positives (worst: validation 24 h, 891),
so minority-class metrics stay estimable. Three facts drive the rest of the project:

* **The imbalance worsens into deployment** — 9.4:1 in training → 18.6:1 on validation →
  **70:1** on the frozen test at 24 h. A model fitted at a 9.6 % base rate meets 1.4 %.
* **It is never a single number** — per-year 72 h ratios span **4.9:1 (2017) to 83.0:1
  (2023)**; 2018 is absent (R6) and never interpolated; monsoon vs non-monsoon in the test
  period differs 7.3× (2.33 % vs 0.32 % at 24 h). Every split × season cell holds > 1 000
  labelled rows, so all three horizons remain reportable per season (D1).
* **The positives are concentrated** — 133 stations, 111 ever positive, 122 with ≥ 200
  labelled rows; the **top-10 stations hold 38.9 %** of all 24 h positives, top-20 57.6 %.
  Micro averages are therefore dominated by a few tidal stations: macro per-station numbers
  stay beside them (D8). 14 stations with ≥ 200 rows have **zero** positives (Dhaka,
  Faridpur, Bogura, Chuadanga …) — kept, never dropped.

Two re-measurement warnings (recorded in the notebook, phase0 not overwritten):

* **W-R3** — phase0 R3 says 13 zero-positive stations (≥ 200 labelled rows); the Phase-14
  table has **14**.
* **W-R2** — phase0 R2 quotes Mongla 83.8 % (Khulna 52.3 %, Daulatkhan 40.4 %); the table
  gives **73.2 % / 39.5 % / 26.8 %** — the earlier figure is not reproducible from the
  current table.

**Why accuracy and ROC-AUC cannot be the headline (R4):**

* *Accuracy* — on validation 2022 the trivial always-negative classifier already scores
  94.91 / 94.06 / 93.42 % (24/48/72 h); the best real model gains at most **0.045**
  (< 0.05) while its PR-AUC gain over prevalence is ~18×.
* *ROC-AUC* — PR-AUC < ROC-AUC in **all 675 real-model cells** (minimum gap 0.014 dev,
  0.123 test): the majority class inflates ROC-AUC, so PR-AUC stays primary.
* *Threshold transfer* — the train-derived threshold of `logreg_linear` gives recall 0.847
  on validation 2022 (base 5.09 %) but only **0.531** on test 2023 (base 0.68 %): a single
  threshold does not follow the base rate.

**Minority-class performance** (frozen test, per cell, never pooled — D8), `logreg_linear`
recall / precision / PR-AUC: 2023 0.476 / 0.603 / 0.562 · 2024 0.735 / 0.761 / 0.747 ·
2025 0.636 / 0.696 / 0.708 · 2026 0.671 / 0.665 / 0.694; buckets trained 0.680 / 0.730 /
0.716, unseen 0.670 / 0.611 / 0.677, new-network 0.586 / 0.719 / 0.680. All real-model
cells finite — no model collapses to recall 0.

**Policy carried into Phase 17+ (R4 / phase0):**

1. **PR-AUC is the primary metric** everywhere; ROC-AUC secondary only; accuracy is banned
   as a headline (stages 34–35 inherit this rule).
2. **Resampling / class weighting / SMOTE only inside the training process** (workflow
   Phase 19) — never before the split, never in validation or test. Phase 16 baselines are
   unweighted; the rule binds Phases 17+.
3. **Thresholds are train-derived (S6 verified) and period-specific** — the 0.847 → 0.531
   recall drop is the standing argument against carrying one threshold across a base-rate
   shift; probability-level metrics are the transferable comparison.
4. **Never pool** years, seasons or station buckets; report macro beside micro.
5. The FFWC cost asymmetry (FN ≫ FP) is a *decision* rule and is handled in stage 39
   (threshold optimization), not here.

## 11. Findings handed to Phase 17

1. **The bar is persistence, not climatology.** Climatology is exactly prevalence
   (sanitised); every real model beats it 10–18×. Against D7 the honest gain is only
   +2–4 % (val), −4…+1 % (spatial), +0…+9 % (test years).
2. **Spatial generalisation is the weak point.** `logreg_linear` is below D7 in 3 of 4
   spatial folds and below it on the `test_unseen`/`test_new_network` buckets; the
   depth-4 tree is roughly on par with the bar there (0.989–1.018). Phase 17 candidates
   should be judged first on unseen stations, not on the temporal cell.
3. **The compact SELECTED set generalises as well or better than LINEAR on unseen data**
   (spatial macro 0.8936 vs 0.8865; `test_unseen` 0.6935 vs 0.6774) at a fifth of the
   features.
4. **Calibration is not the problem** — ECE ≤ 0.007 for all linear models on val and test;
   persistence sits at 0.012–0.018. Probability work in Phase 21 is optional, not urgent.
5. **Non-monsoon cells are the hard ones** (PR-AUC 0.51–0.65, base rate 0.29–0.51 %) and
   must never be pooled away.
6. **Nothing here used the test set for a decision.** The first such use would be a
   protocol violation; test numbers above are a record, not an input.
7. **The deployment regime is the imbalanced one** — 70:1 at 24 h on test vs 9.4:1 in
   training, positives concentrated in 10 stations (38.9 %), threshold transfer costing
   0.32 of recall (§10). Every candidate must clear D7 under those conditions, per year,
   per season and per bucket.

## 12. Open items

* **phase0 R2 / R3 need re-measurement** — the notebook's W-R2 / W-R3 warnings (§10):
  zero-positive stations 13 → **14**, Mongla 83.8 % → **73.2 %** (Khulna, Daulatkhan
  likewise). phase0 was deliberately not overwritten; reconcile at the next phase0 review.
* `up_*` decision (§8) deferred to Phase 17.
* No per-station bootstrap CI on spatial-CV cells at station level across folds (CIs there
  are fold-wise row bootstraps); the station bootstrap proper is applied on test cells and
  is specified for Phase 20.

---

**Next: Phase 17 — candidate ML models (body step 29 / checklist stage 31)**, judging every
candidate against the D7 persistence bar under the same three protocols and per-season
cells, under the stage-29 metric policy of §10. Order per **D10**.
