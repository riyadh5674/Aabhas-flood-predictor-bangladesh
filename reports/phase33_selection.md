# Phase 33 — Final model selection (stage 57)

**Status:** DONE · **Seed:** `20261008` · **Wall-clock:** 21.4 s
**Dataset:** `data/processed/experimental_dataset.csv` (200,665 rows), SHA-256
`a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` — unchanged.
**Frozen test (`>= 2023`):** read **exactly once** (`n_reads = 1`) — for the **selected
model** and the **D7 persistence bar** only — after the choice was frozen; logged in
`reports/phase33_test_read_log.json`.

**Question (workflow Phase 33).** *Which single model goes forward?* A multi-criteria
**scorecard**, not "highest accuracy wins", answered on **validation + spatial-CV evidence
only**; then the frozen test is read once to confirm the choice and report the operational
number.

---

## 0. Method

### 0.1 The scorecard (weights straight from the workflow)

| criterion | role | how scored here |
| --- | --- | --- |
| PR-AUC vs D7 bar (per year, per season, per bucket) | **primary** | horizon-mean P1 `val_2022` PR-AUC; D7 margin over all 7 dev cells × 3 horizons |
| flood recall / missed-flood count at the chosen threshold | **co-primary** (FN ≫ FP) | missed rows at the Phase-29 cost-optimal `t_warn` (10:1), summed over horizons |
| calibration (Brier/ECE, post-Phase 28) | **gate** | `phase28_calibration.csv`: `cleared_all`, or OOF ECE ≤ 0.02 |
| temporal generalisation (P2, worst year) | **strong evidence** | worst of `val_2020/21/22` D7 margin |
| spatial generalisation (P3, unseen, `new_network`) | **strong evidence** | mean fold D7 margin on `spatial_fold_1..4` |
| robustness + ablation consistency | **evidence** | Phase-31/32 (leaders robust; hydrology dominates) |
| interpretability (SHAP availability, family transparency) | **tie-breaker** | tree/linear (exact SHAP) > deep (IG/attention) |
| train/inference cost, model size, hardware | **tie-breaker** | `phase27_cost.csv` |

Every number is taken from the **frozen** Phase-16/20-32 artefacts (no re-tuning, no
re-fitting): `phase27_metrics.csv`, `phase27_cost.csv`, `phase28_calibration.csv`,
`phase29_chosen_thresholds.csv`, `phase31_*`, `phase32_*`. Nothing in the scorecard touches
the test period.

### 0.2 Selection rule (pre-registered, applied before any test read)

1. **Gate 1 (skill):** beat the D7 `persistence` bar (PR-AUC) on **every** dev point —
   7 cells × 3 horizons = 21 points.
2. **Gate 2 (calibration):** usable for risk bands (`cleared_all`, or best OOF ECE ≤ 0.02).
3. References `climatology` / `persistence` are **bars**, never candidates.
4. Survivors ranked by **primary** = horizon-mean P1 `val_2022` PR-AUC, then **co-primary**
   = fewest missed floods (Phase-29 operating point), then robustness floor, then cost, then
   interpretability.
5. Overlapping 95 % station-bootstrap CIs => **not separable**; later tie-breakers decide.

### 0.3 The test read

Only after the choice is frozen, the script loads the frozen pipelines (fit on P1
`train_<2022`, holdout excluded; 61,977 rows @ 24 h) and the D7 pipeline, predicts the seven
test cells (`test_2023..2026`, `test_trained`, `test_new_network`, `test_unseen`) × three
seasons × three horizons, and scores them through the **one Phase-27 path**
`evaluate_all.score` at the **Phase-29 period-specific thresholds**. The read is guarded: a
completed read is cached and re-used (log records `n_reads`). As an independent check the D7
test PR-AUC/Brier reproduce the Phase-16 test read to **|Δ| ≤ 5.6e-17** over 63 shared rows.

---

## 1. Scorecard (all 24 models, dev evidence only)

P1 = `val_2022`; `D7 min` = worst D7 margin over the 7 dev cells; `P2 worst` = worst
temporal-year margin; `P3 mean` = mean spatial-fold margin; `missed` = missed-flood rows
summed over 24/48/72 h at the Phase-29 threshold.

| # | model | lvl | P1 24 h | P1 48 h | P1 72 h | **P1 mean** | D7 min | P2 worst | P3 mean | cal | ECE | missed | elig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | **xgboost** | L2 | 0.96812 | 0.95494 | 0.94084 | **0.95463** | +0.0144 | +0.0184 | +0.0445 | ✅ | 0.0031 | **87** | ✅ |
| 2 | xgboost_selected | L2 | 0.96631 | 0.95333 | 0.94216 | 0.95393 | +0.0143 | +0.0176 | +0.0412 | ✅ | 0.0035 | 88 | ✅ |
| 3 | lightgbm | L2 | 0.96816 | 0.95379 | 0.93838 | 0.95344 | +0.0144 | +0.0174 | +0.0443 | ✅ | 0.0036 | 112 | ✅ |
| 4 | random_forest | L2 | 0.96650 | 0.95155 | 0.93885 | 0.95230 | +0.0134 | +0.0168 | +0.0432 | ✅ | 0.0032 | 96 | ✅ |
| — | gradient_boosting | L2 | 0.96184 | 0.94569 | 0.93383 | 0.94712 | **−0.0192** | +0.0107 | +0.0190 | ✅ | 0.0045 | 101 | ❌ G1 |
| — | fusion_L3 | L3 | 0.94443 | 0.93854 | 0.92381 | 0.93559 | −0.1432 | +0.0004 | −0.0001 | ✅ | 0.0054 | 109 | ❌ G1 |
| — | transformer_bestL | L4 | 0.94371 | 0.93304 | 0.91830 | 0.93168 | −0.1000 | −0.0081 | −0.0052 | ✅ | 0.0039 | 112 | ❌ G1 |
| — | gcn | L4 | 0.94207 | 0.93534 | 0.91414 | 0.93052 | −0.1448 | −0.0016 | −0.0094 | ✅ | 0.0042 | 130 | ❌ G1 |
| — | gat | L4 | 0.94079 | 0.93145 | 0.91356 | 0.92860 | −0.1180 | −0.0182 | −0.0061 | ✅ | 0.0062 | 138 | ❌ G1 |
| — | gnn_lstm | L4 | 0.94237 | 0.92640 | 0.91529 | 0.92802 | −0.1374 | +0.0038 | −0.0127 | ✅ | 0.0046 | 118 | ❌ G1 |
| — | lstm_L3 | L3 | 0.94206 | 0.92534 | 0.91291 | 0.92677 | −0.1409 | −0.0073 | −0.0166 | ✅ | 0.0043 | 113 | ❌ G1 |
| — | gru_bestL | L3 | 0.93592 | 0.92632 | 0.90861 | 0.92362 | −0.1275 | −0.0067 | −0.0189 | ✅ | 0.0052 | 124 | ❌ G1 |
| — | logreg_linear_no_up | L1 | 0.94026 | 0.92416 | 0.90523 | 0.92321 | −0.0506 | −0.0056 | −0.0124 | ✅ | 0.0046 | 131 | ❌ G1 |
| — | logreg_linear | L1 | 0.94007 | 0.92414 | 0.90528 | 0.92316 | −0.0562 | −0.0064 | −0.0140 | ✅ | 0.0048 | 144 | ❌ G1 |
| — | tcn_bestL | L3 | 0.94052 | 0.92236 | 0.90083 | 0.92124 | −0.1238 | −0.0237 | −0.0203 | ✅ | 0.0040 | 120 | ❌ G1 |
| — | logreg_selected | L1 | 0.93979 | 0.92065 | 0.90042 | 0.92029 | −0.0400 | +0.0013 | −0.0082 | ✅ | 0.0082 | 161 | ❌ G1 |
| — | hybrid_L3 | L3 | 0.92747 | 0.92145 | 0.90715 | 0.91869 | −0.1434 | −0.0056 | −0.0200 | ✅ | 0.0049 | 117 | ❌ G1 |
| — | lstm_L30 | L3 | 0.93883 | 0.91827 | 0.89355 | 0.91688 | −0.1232 | −0.0339 | −0.0104 | ✅ | 0.0055 | 118 | ❌ G1 |
| — | lstm_L7 | L3 | 0.93091 | 0.91642 | 0.88707 | 0.91146 | −0.1289 | −0.0244 | −0.0136 | ✅ | 0.0065 | 132 | ❌ G1 |
| — | lstm_L14 | L3 | 0.93107 | 0.90983 | 0.88234 | 0.90775 | −0.1063 | −0.0426 | −0.0097 | ✅ | 0.0060 | 136 | ❌ G1 |
| — | **persistence (D7 bar)** | L1 | 0.92029 | 0.89540 | 0.86826 | 0.89465 | 0.0000 | 0.0000 | 0.0000 | ✅ | 0.0044 | 293 | bar |
| — | tree_depth4 | L1 | 0.89715 | 0.89024 | 0.87347 | 0.88695 | −0.0231 | −0.0231 | +0.0041 | ✅ | 0.0049 | 205 | ❌ G1 |
| — | static_only | L3 | 0.50676 | 0.53858 | 0.55212 | 0.53249 | −0.8378 | −0.7706 | −0.7793 | ❌ | 0.0333 | 905 | ❌ G1,G2 |
| — | **climatology** | L1 | 0.05091 | 0.05936 | 0.06577 | 0.05868 | −0.8784 | −0.8704 | −0.8119 | ✅ | 0.0003 | 3042 | bar |

**Four models clear both gates** — `xgboost`, `xgboost_selected`, `lightgbm`,
`random_forest`, all L2 tree ensembles. Everything else fails Gate 1 (the best deep model
`fusion_L3` sits *at* the D7 bar on average but collapses by −0.14 on the hardest fold), and
`static_only` also fails Gate 2.

---

## 2. The choice — `xgboost` (L2-03), frozen

| | |
| --- | --- |
| **Model** | `xgboost`, registry `L2-03`, config `xgboost_c01` |
| **Features** | FULL — 62 (Phase-12 set) |
| **Hyper-parameters** | `n_estimators=300, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=1.0, colsample_bytree=0.8` |
| **Training rows** | P1 `train_<2022` (split train & not holdout, ≤ 2021): 61,977 / 60,992 / 60,144 @ 24/48/72 h |
| **Seed** | `20261008` |
| **Params / cost** | 15,002 tree nodes; 1.15 s to fit all three horizons; **0.066 ms / 1,000 rows** inference |
| **Frozen** | `models/final/selected_model.pkl` (+ `final_config.json`, `README.md`) |

Why it wins, in the order the rule applies:

1. **Primary** — highest horizon-mean P1 `val_2022` PR-AUC (**0.95463**); it also leads 48 h
   outright (0.95494) and is a close 2nd at 24 h (0.96812, −0.00004 behind lightgbm) and 72 h
   (0.94084, −0.00132 behind `xgboost_selected`).
2. **Co-primary (FN ≫ FP)** — the **fewest missed floods** of any model: **87** vs 88 / 96 /
   112 for the other eligible models, at the Phase-29 cost-optimal thresholds (24 h recall
   0.982, 48 h 0.982, 72 h 0.954).
3. **Robustness** — beats D7 on **all 21** dev points, floor **+0.0144** (tied best), and
   generalises in time (+0.0184 worst year) and space (+0.0445 mean fold).
4. **Calibration** — cleared (raw), post-recalibration OOF ECE **0.0031** (best of the four).
5. **Tie-breakers** — exact TreeSHAP interpretability and cheap inference (0.066 ms/1k).

**Non-separability (stated, not hidden).** The four eligible models share a common 95 %
station-bootstrap CI **at every horizon**: 24 h `[0.9324, 0.9860]`, 48 h `[0.9075, 0.9780]`,
72 h `[0.8862, 0.9680]` (the intersection of all four CIs). Their PR-AUC differences are
therefore within sampling noise, and the choice rests on the **co-primary** (missed floods)
and tie-breakers — exactly as the scorecard intends. The per-horizon argmax differs
(lightgbm @ 24 h, `xgboost_selected` @ 72 h) but is **not** statistically distinguishable
from `xgboost`; keeping **one** frozen family is the honest, simpler call.

---

## 3. Rejected families and why

| family | verdict | reason (scorecard) |
| --- | --- | --- |
| L1 references — `climatology`, `persistence` (D7) | excluded | bars, not candidates; D7 is the benchmark every family must clear |
| L1 linear — `logreg_linear`, `logreg_linear_no_up`, `logreg_selected` | rejected | beat D7 on P1 but fall **below** D7 at some dev cell (floor −0.040 to −0.056); linear ceiling, ~3 pts behind L2 |
| L1 tree — `tree_depth4` | rejected | −0.0231 floor; underfits |
| L2 `gradient_boosting` (sklearn) | rejected | Gate 1: **−0.0192** below D7 on a dev cell (weakest of the boosting family) |
| L2 `random_forest` | not selected | eligible, but 4th on primary (0.95230) and 96 missed floods; costliest inference (0.120 ms/1k) |
| L2 `lightgbm`, `xgboost_selected` | not selected | eligible and statistically inseparable from the winner, but **more missed floods** (112 / 88 vs 87) — the FN≫FP co-primary decides |
| L3 deep temporal — LSTM/GRU/TCN (`lstm_L*`, `gru_bestL`, `tcn_bestL`) | rejected | Gate 1: worst D7 margin −0.107 to −0.141; recurrent nets also miscalibrated (Phase 28) |
| L3 fusion — `fusion_L3`, `hybrid_L3`, `static_only` | rejected | Gate 1: −0.143 floors (`static_only` also fails Gate 2); geo/static input adds only marginal skill (Phase 23) |
| L4 graph / transformer — `gcn`, `gat`, `gnn_lstm`, `transformer_bestL` | rejected | Gate 1: −0.100 to −0.145; graph/attention gains are horizon-specific and do not clear the D7 floor |

**The scientific verdict:** after 33 phases, the **gradient-boosting tabular model on the
62 engineered features** is the final model. Deep learning does **not** beat it on the
primary metric — consistent with Phases 26–32. The signal is hydrology (`wl_pct_danger`
family) plus geography, not sequence shape; the DL advantage appears only on `test_unseen`
at the **shortest** horizon, and even there only ties D7 (§4).

---

## 4. Frozen-test read (once) — the operational result

Test rows: **101,174**. `n_reads = 1`; D7 reproduced from Phase 16 to 5.6e-17.

### 4.1 PR-AUC on the frozen test (`xgboost` vs D7), all-season

| cell | 24 h sel / D7 | 48 h sel / D7 | 72 h sel / D7 |
| --- | --- | --- | --- |
| `test_2023` | 0.7473 / 0.6276 | 0.6575 / 0.5523 | 0.6254 / 0.5108 |
| `test_2024` | 0.9004 / 0.8022 | 0.8461 / 0.7302 | 0.7930 / 0.6727 |
| `test_2025` | 0.7967 / 0.7328 | 0.7633 / 0.6670 | 0.7176 / 0.6041 |
| `test_2026` | 0.7660 / 0.7233 | 0.7324 / 0.6495 | 0.7068 / 0.5929 |
| `test_trained` | 0.8480 / 0.7468 | 0.7990 / 0.6762 | 0.7567 / 0.6183 |
| `test_new_network` | 0.8177 / 0.7356 | 0.7695 / 0.6930 | 0.7290 / 0.6457 |
| `test_unseen` | **0.7675 / 0.7696** | 0.7001 / 0.6694 | 0.6360 / 0.5986 |

* **Selected ≥ D7 on 20/21 cell × horizon points.** Horizon-mean over the four test years:
  24 h **0.8026 vs 0.7215**, 48 h **0.7498 vs 0.6498**, 72 h **0.7107 vs 0.5951**.
* **The one exception is honest and specific:** `test_unseen` @ **24 h** — 0.7675 vs 0.7696
  (**−0.0020**), and the CIs overlap almost completely (`[0.555, 0.955]` vs `[0.600, 0.888]`).
  On the shortest lead time at **completely unseen stations**, the learned model is
  statistically **indistinguishable from D7 persistence**. At 48/72 h on the same stations
  it wins (+0.031 / +0.037), and on **new-network** stations it wins clearly (+0.082 / +0.077
  / +0.083).
* Monsoon (4 test years, mean PR-AUC): 24 h **0.8107 vs 0.7311**, 48 h **0.7642 vs 0.6635**,
  72 h **0.7276 vs 0.6135**.

### 4.2 Operational block (Phase-29 thresholds, FN ≫ FP)

| | missed floods | TP | FP | mean recall | mean FAR | event recall |
| --- | --- | --- | --- | --- | --- | --- |
| **xgboost** | **1,044** | 9,530 | 15,258 | **0.894** | 0.608 | **0.967** |
| D7 persistence | 2,478 | 8,096 | 14,370 | 0.769 | 0.614 | 0.901 |

The selected model **halves the missed floods** (2,478 → 1,044, −58 %), lifts mean row
recall **+12.5 pts**, and raises **event recall to 0.967**, at essentially the same
false-alarm rate as D7 — exactly the FN ≫ FP trade the operational design asks for.

---

## 5. Named findings

* **F33-1 — The final model is gradient boosting, not deep learning.** `xgboost` (L2-03,
  FULL 62) is the only family that both beats D7 on every dev point and is calibrated; the
  deep models miss the D7 floor by 0.10–0.14.
* **F33-2 — Four models are statistically inseparable.** xgboost / xgboost_selected /
  lightgbm / random_forest share a common CI at every horizon; the winner is decided by the
  co-primary (missed floods, 87) and tie-breakers, not by a noise-level PR-AUC gap.
* **F33-3 — The test confirms it.** 20/21 wins; horizon-mean PR-AUC +0.08/+0.10/+0.12 over
  D7 at 24/48/72 h; missed floods 1,044 vs 2,478.
* **F33-4 — The one limit is documented.** On `test_unseen` @ 24 h the model **ties** D7
  (−0.002, CI overlap): with no local training history and only 24 h of lead time, the
  persistence of the current level is as informative as the model.
* **F33-5 — Generalisation is positive.** New-network stations (+0.077 to +0.083) and unseen
  stations at 48/72 h (+0.031 / +0.037) both clear D7; the model is not overfit to trained
  stations.

---

## 6. Limitations & warnings

1. **Non-separability.** The selection among the four eligible L2 models is decided by
   secondary criteria; a different tie-break could pick `lightgbm` or `xgboost_selected`
   with no measurable loss. The frozen artefact records the chosen model, not a unique
   optimum.
2. **`test_unseen` @ 24 h** is a tie, not a win — reported, not smoothed over.
3. **Thresholds are frozen from Phase 29** (validation-derived, period-specific); they were
   **not** re-derived on test (that would be leakage). Recall/FAR therefore carry the
   Phase-29 operating-point assumptions (`C_FN:C_FP = 10:1`).
4. **Base rates on test are low** (0.7–6 %), so absolute PR-AUC is regime-dependent; the
   **D7-relative margin** is the stable comparison and is what the scorecard uses.
5. Only **two models** touched the test set (the selection + D7) — by construction; any other
   test number would be a new read and is not reported.

---

## 7. Provenance & definition of done

| artefact | contents |
| --- | --- |
| `scripts/phase33_selection.py` | scorecard → gate/rank → freeze → one guarded test read |
| `scripts/phase33_run.log` | run record (exit 0, wall-clock, seed) — deterministic re-run that reuses the logged single test read |
| `reports/phase33_scorecard.csv` | 24 models × criteria |
| `reports/phase33_selection.json` | the pre-registered rule + ranked choice + frozen thresholds |
| `models/final/selected_model.pkl` | the frozen pipelines (per horizon) + config + seed + dataset SHA |
| `models/final/final_config.json`, `models/final/README.md` | config (no pickles) + load recipe |
| `reports/phase33_test_metrics.csv` | the single test read (126 rows) |
| `reports/phase33_test_read_log.json` | who / when / which models / rows (=1 read) |
| `reports/phase33_summary.json` | headline + run record |
| `figures/phase33_scorecard.png` | scorecard + test vs D7 |
| `notebooks/33_selection.ipynb` | **12/12** checks, built by `scripts/build_phase33_notebook.py` |

**Definition of done.** ✅ code (`scripts/phase33_selection.py`) · ✅ run record
(`scripts/phase33_run.log`, exit 0, 21.4 s, seed `20261008`) · ✅ report (this file) ·
✅ checklist row 57 flipped to DONE and the sequencing updated. The selected model is
frozen under `models/final/` with config + seed; the frozen test was read **once** for the
selected model and the D7 bar and is logged. **Next: Phase 34 — Explainable AI (stage 58)**
(needs `pip install shap`).
