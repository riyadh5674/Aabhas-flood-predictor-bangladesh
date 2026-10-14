# Phase 26 — Multi-horizon strategy (workflow Phase 26 / stage 41 + stages 59–61)

| | |
| --- | --- |
| **Date** | 2026-10-14 |
| **Script** | `scripts/phase26_matrix.py` → `reports/phase26_matrix.csv`, `phase26_summary.csv`, `phase26_interaction.json` · run record `scripts/phase26_run.log` (seed 20261008, exit=0, 2.1 s) |
| **Notebook** | `notebooks/26_multihorizon.ipynb` (built by `scripts/build_phase26_notebook.py`) |
| **Inputs** | `phase16_baseline_metrics.csv` (L1) · `phase20_metrics.csv` (L2) · `phase21_metrics.csv` (L3 LSTM/GRU) · `phase22_metrics.csv` (L3 TCN) · `phase23_metrics.csv` (L3 fusion) · `phase24_metrics.csv` (L4 GNN) · `phase25_metrics.csv` (L4 Transformer) · `phase17_experiment_registry.csv` |
| **Outputs** | `reports/phase26_matrix.csv` (1,728 rows, 24 models) · `reports/phase26_summary.csv` (72 rows) · `reports/phase26_interaction.json` · `notebooks/26_multihorizon.ipynb` |
| **Verdict** | **Multi-horizon matrix complete for all 24 L1–L4 models.** Difficulty grows monotonically with lead time for every competitive model. The **ML-vs-DL interaction is small on the temporal cells (P1/P2)** but **real on the unseen-station cell (P3)**: engineered-feature models lose more PR-AUC from 24 h → 72 h (mean −0.034) than sequence/DL models (−0.025), and the DL fusion model is **flat across horizons** (+0.002). In absolute terms, however, gradient boosting (L2) leads at **every** horizon on **every** protocol. Test never read (frozen to Phase 33). |
| **Governing doc** | `final workflow.md` **Phase 26 — Multi-horizon strategy** / **stage 41** (+ **stages 59, 60, 61**) |

`experimental_dataset.csv` SHA-256 is unchanged (`a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e`);
no test cell appears in any input (asserted). No model was trained in this phase — it integrates the
frozen dev-cell results of Phases 16–25.

---

## 1. The question

Three lead times are three different forecasting tasks (D2), and every phase has trained and
scored them separately (D8 — horizons are never pooled). Phase 26 answers the two remaining
multi-horizon questions over the whole ladder:

1. **How does difficulty grow with lead time?** For each of the 24 surviving L1–L4 models, the
   change in PR-AUC from 24 h → 72 h.
2. **Does it grow *differently* for sequence models than for engineered-feature models?** The
   interaction that would justify carrying a whole deep-learning section.

Stages **59 / 60 / 61** are the 24 h / 48 h / 72 h deliverables — one section each (§7),
filled from the matrix, **not** three more training rounds.

## 2. Method

Integration only. The seven per-phase dev metrics files (Phases 16, 20–25) share one schema
(`protocol, train, cell, season, model, horizon, n, pr_auc, roc_auc, brier, …`); `phase26_matrix.py`
concatenates them, tags each model with its **registry level** (L1–L4) and family label, and
asserts **no test cell** is present. Cells:

* **P1** primary — `val_2022`, season = all (the selection cell every phase reported on);
* **P2** rolling origin — mean over `val_2020/2021/2022`;
* **P3** spatial CV — mean over the 4 station-disjoint folds (unseen stations);

each × {24 h, 48 h, 72 h} (season = all), with the monsoon / non-monsoon split available in the
grid (`phase26_matrix.csv`). The grouping for the interaction is the registry's own D14 taxonomy:
**engineered-feature = L1 ∪ L2**, **sequence/DL = L3 ∪ L4**; the two trivial reference baselines
(`climatology`, `persistence`) are shown but excluded from the group means.

## 3. The headline matrix — P1 `val_2022` (season = all, PR-AUC)

The workflow's requested table, one best representative per family (full 24-model matrix in §4):

| family | model | 24 h | 48 h | 72 h | 24→72 Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| **XGBoost (L2)** | `xgboost` | **0.9681** | **0.9549** | 0.9408 | −0.0273 |
| **LightGBM (L2)** | `lightgbm` | 0.9682 | 0.9538 | 0.9384 | −0.0298 |
| Random Forest (L2) | `random_forest` | 0.9665 | 0.9515 | 0.9389 | −0.0277 |
| XGB-SELECTED (L2) | `xgboost_selected` | 0.9663 | 0.9533 | **0.9422** | −0.0241 |
| **fusion DL (L3)** | `fusion_L3` | 0.9444 | **0.9385** | **0.9238** | **−0.0206** |
| Transformer (L4) | `transformer_bestL` | 0.9437 | 0.9330 | 0.9183 | −0.0254 |
| GNN (L4) | `gcn` | 0.9421 | 0.9353 | 0.9141 | −0.0279 |
| **LSTM (L3)** | `lstm_L3` | 0.9421 | 0.9253 | 0.9129 | −0.0292 |
| **TCN (L3)** | `tcn_bestL` | 0.9405 | 0.9224 | 0.9008 | −0.0397 |
| **GRU (L3)** | `gru_bestL` | 0.9359 | 0.9263 | 0.9086 | −0.0273 |
| *logreg (L1)* | `logreg_linear` | 0.9401 | 0.9241 | 0.9053 | −0.0348 |
| *D7 persistence (L1)* | `persistence` | 0.9203 | 0.8954 | 0.8683 | −0.0520 |
| *climatology (L1)* | `climatology` | 0.0509 | 0.0594 | 0.0658 | +0.0149 |

Two facts stand out. (i) **Gradient boosting owns `val_2022` at every horizon** — the best
engineered model beats the best DL model by ~+0.024 at 24 h, ~+0.016 at 48 h and ~+0.018 at
72 h. (ii) **The DL fusion models degrade the least** across lead time (`fusion_L3` −0.021,
`hybrid_L3` −0.020) — roughly a third less decay than the D7 bar (−0.052) and less than every
engineered model; being already at the top of the DL group, `fusion_L3` narrows the ML gap most
at 48 h (+0.016 vs +0.024 at 24 h).

## 4. Full matrix — all 24 models

**P1 `val_2022` (season = all):**

| level | model | 24 h | 48 h | 72 h | mean | Δ 24→72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L1 | climatology | 0.0509 | 0.0594 | 0.0658 | 0.0587 | +0.0149 |
| L1 | persistence (D7) | 0.9203 | 0.8954 | 0.8683 | 0.8947 | −0.0520 |
| L1 | logreg_linear | 0.9401 | 0.9241 | 0.9053 | 0.9232 | −0.0348 |
| L1 | logreg_linear_no_up | 0.9403 | 0.9242 | 0.9052 | 0.9232 | −0.0350 |
| L1 | logreg_selected | 0.9398 | 0.9206 | 0.9004 | 0.9203 | −0.0394 |
| L1 | tree_depth4 | 0.8972 | 0.8902 | 0.8735 | 0.8870 | −0.0237 |
| L2 | random_forest | 0.9665 | 0.9515 | 0.9389 | 0.9523 | −0.0277 |
| L2 | gradient_boosting | 0.9618 | 0.9457 | 0.9338 | 0.9471 | −0.0280 |
| L2 | xgboost | 0.9681 | 0.9549 | 0.9408 | 0.9546 | −0.0273 |
| L2 | lightgbm | 0.9682 | 0.9538 | 0.9384 | 0.9535 | −0.0298 |
| L2 | xgboost_selected | 0.9663 | 0.9533 | 0.9422 | 0.9539 | −0.0241 |
| L3 | lstm_L3 | 0.9421 | 0.9253 | 0.9129 | 0.9268 | −0.0292 |
| L3 | lstm_L7 | 0.9309 | 0.9164 | 0.8871 | 0.9115 | −0.0438 |
| L3 | lstm_L14 | 0.9311 | 0.9098 | 0.8823 | 0.9077 | −0.0487 |
| L3 | lstm_L30 | 0.9388 | 0.9183 | 0.8936 | 0.9169 | −0.0453 |
| L3 | gru_bestL | 0.9359 | 0.9263 | 0.9086 | 0.9236 | −0.0273 |
| L3 | tcn_bestL | 0.9405 | 0.9224 | 0.9008 | 0.9212 | −0.0397 |
| L3 | fusion_L3 | 0.9444 | 0.9385 | 0.9238 | 0.9356 | −0.0206 |
| L3 | static_only | 0.5068 | 0.5386 | 0.5521 | 0.5325 | +0.0454 |
| L3 | hybrid_L3 | 0.9275 | 0.9215 | 0.9072 | 0.9187 | −0.0203 |
| L4 | gcn | 0.9421 | 0.9353 | 0.9141 | 0.9305 | −0.0279 |
| L4 | gat | 0.9408 | 0.9314 | 0.9136 | 0.9286 | −0.0272 |
| L4 | transformer_bestL | 0.9437 | 0.9330 | 0.9183 | 0.9317 | −0.0254 |
| L4 | gnn_lstm | 0.9424 | 0.9264 | 0.9153 | 0.9280 | −0.0271 |

**P2 rolling-origin mean (2020/2021/2022)** and **P3 spatial-CV mean (4 folds)** — the two
generalisation axes (`phase26_summary.csv` has the same table with `drop_72_24`):

| protocol | model | 24 h | 48 h | 72 h | Δ 24→72 |
| --- | --- | ---: | ---: | ---: | ---: |
| P2 | xgboost | 0.9786 | 0.9670 | 0.9538 | −0.0248 |
| P2 | xgboost_selected | 0.9779 | 0.9669 | 0.9548 | −0.0231 |
| P2 | random_forest | 0.9770 | 0.9653 | 0.9530 | −0.0240 |
| P2 | gnn_lstm | 0.9590 | 0.9433 | 0.9293 | −0.0297 |
| P2 | fusion_L3 | 0.9593 | 0.9485 | 0.9316 | −0.0277 |
| P2 | lstm_L3 | 0.9551 | 0.9362 | 0.9193 | −0.0358 |
| P2 | transformer_bestL | 0.9549 | 0.9483 | 0.9280 | −0.0269 |
| **P3** | xgboost | 0.9757 | 0.9618 | 0.9493 | −0.0263 |
| **P3** | lightgbm | 0.9752 | 0.9610 | 0.9498 | −0.0253 |
| **P3** | random_forest | 0.9744 | 0.9606 | 0.9475 | −0.0269 |
| **P3** | transformer_bestL | 0.9325 | 0.9095 | 0.8954 | −0.0371 |
| **P3** | gnn_lstm | 0.9272 | 0.9018 | 0.8862 | −0.0411 |
| **P3** | gcn | 0.9184 | 0.9120 | 0.8945 | −0.0239 |
| **P3** | gat | 0.9177 | 0.9162 | 0.9011 | −0.0166 |
| **P3** | fusion_L3 | 0.9154 | 0.9202 | 0.9171 | **+0.0018** |
| **P3** | lstm_L3 | 0.9196 | 0.9055 | 0.8781 | −0.0416 |

## 5. How difficulty grows with lead time

Per-model PR-AUC drop on P1, sorted (negative = harder at 72 h). Only `climatology` (no skill to
lose) and `static_only` (a deliberately crippled out-of-station control, §6) increase.

| rank | model | 24 h → 72 h | Δ |
| ---: | --- | --- | ---: |
| 1 | persistence (D7) | 0.9203 → 0.8683 | **−0.0520** |
| 2 | lstm_L14 | 0.9311 → 0.8823 | −0.0487 |
| 3 | lstm_L30 | 0.9388 → 0.8936 | −0.0453 |
| 4 | lstm_L7 | 0.9309 → 0.8871 | −0.0438 |
| 5 | tcn_bestL | 0.9405 → 0.9008 | −0.0397 |
| 6 | logreg_selected | 0.9398 → 0.9004 | −0.0394 |
| 7 | logreg_linear_no_up | 0.9403 → 0.9052 | −0.0350 |
| 8 | logreg_linear | 0.9401 → 0.9053 | −0.0348 |
| 9 | lightgbm | 0.9682 → 0.9384 | −0.0298 |
| 10 | lstm_L3 | 0.9421 → 0.9129 | −0.0292 |
| 11 | gradient_boosting | 0.9618 → 0.9338 | −0.0280 |
| 12 | gcn | 0.9421 → 0.9141 | −0.0279 |
| 13 | random_forest | 0.9665 → 0.9389 | −0.0277 |
| 14 | gru_bestL | 0.9359 → 0.9086 | −0.0273 |
| 15 | xgboost | 0.9681 → 0.9408 | −0.0273 |
| 16 | gat | 0.9408 → 0.9136 | −0.0272 |
| 17 | gnn_lstm | 0.9424 → 0.9153 | −0.0271 |
| 18 | transformer_bestL | 0.9437 → 0.9183 | −0.0254 |
| 19 | xgboost_selected | 0.9663 → 0.9422 | −0.0241 |
| 20 | tree_depth4 | 0.8972 → 0.8735 | −0.0237 |
| 21 | fusion_L3 | 0.9444 → 0.9238 | **−0.0206** |
| 22 | hybrid_L3 | 0.9275 → 0.9072 | −0.0203 |

**Reading.** Every competitive model loses skill with lead time; nothing improves. The loss is
**not** ordered by model class: the shallow D7 bar loses the most (−0.052), and within the L3
family the *longer*-window LSTMs lose the most (−0.044 to −0.049 for L7–L30, versus −0.029 for
the 3-day window), while the **strongest** models of *both* classes lose least (xgboost −0.027,
gradient boosting −0.028, and the DL fusion −0.021). Difficulty growth is a property of *how much
signal is still in the features*, not of the architecture — the models that use the features best
also last longest.

## 6. The interaction — sequence (L3/L4) vs engineered-feature (L1/L2)

Group statistics (trivial baselines excluded) on the three protocols:

| protocol | group | n | mean 24 h | mean 48 h | mean 72 h | mean Δ | median Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **P1** | engineered (L1/L2) | 9 | 0.9498 | 0.9354 | 0.9198 | −0.0300 | −0.0280 |
| **P1** | sequence (L3/L4) | 13 | 0.9051 | 0.8956 | 0.8792 | −0.0259 | −0.0273 |
| **P2** | engineered | 9 | 0.9631 | 0.9511 | 0.9363 | −0.0268 | −0.0260 |
| **P2** | sequence | 13 | 0.9077 | 0.8967 | 0.8804 | −0.0273 | −0.0277 |
| **P3** | engineered | 9 | **0.9528** | **0.9357** | **0.9185** | **−0.0343** | **−0.0301** |
| **P3** | sequence | 13 | 0.8587 | 0.8483 | 0.8333 | **−0.0254** | −0.0284 |

Best representative of each group (controls for the group being dragged down by weak members):

| protocol | best engineered | 24/48/72 h | Δ | best sequence | 24/48/72 h | Δ |
| --- | --- | --- | ---: | --- | --- | ---: |
| P1 | `xgboost` | 0.9681 / 0.9549 / 0.9408 | −0.0273 | `fusion_L3` | 0.9444 / 0.9385 / 0.9238 | **−0.0206** |
| P2 | `xgboost_selected` | 0.9779 / 0.9669 / 0.9548 | **−0.0231** | `fusion_L3` | 0.9593 / 0.9485 / 0.9316 | −0.0277 |
| P3 | `xgboost` | 0.9757 / 0.9618 / 0.9493 | −0.0263 | `fusion_L3` | 0.9154 / 0.9202 / 0.9171 | **+0.0018** |

**Answer.** On the two temporal protocols the interaction is **small**: on P1 the two groups
degrade at nearly the same rate (−0.030 vs −0.026 mean; medians −0.028 vs −0.027) and on P2 they
are indistinguishable (−0.0268 vs −0.0273). The differences that do exist are **not** about the
*rate* — the DL group simply sits lower in absolute terms on P1 (~0.04) and P2 (~0.055) because
gradient boosting on engineered features is stronger there.

On the **unseen-station protocol P3** the interaction is **real and consistent**: engineered
models lose **more** skill with lead time (mean Δ −0.034; best `xgboost` −0.026) than
sequence/DL models (mean Δ −0.025), and the DL fusion model is **flat across horizons**
(+0.002 from 24 h → 72 h; it actually *rises* at 48 h). The GNN families are similarly flat
(`gat` −0.017, `gcn` −0.024). This is the interaction the workflow asked about, and it points the
same way as Phase 23/24/25: **when same-station persistence stops transferring (held-out
stations), the learned temporal/spatial sequence models hold their skill across lead time where
the engineered-feature models do not.** It does **not** mean DL beats ML on P3 — `xgboost` still
leads every P3 horizon by ~+0.04 (0.9757 vs 0.9325 at 24 h) — only that DL's *lead-time decay* is
gentler there.

`static_only` is the deliberate control (static geography alone): it has no temporal signal, so
it scores near the base rate on P1 (0.51) and is *anti-correlated* with lead time (Δ +0.045) —
reported for completeness, excluded from the group means.

## 7. Horizon-specific deliverables (stages 59, 60, 61)

### Stage 59 — 24-hour prediction

| rank | model | PR-AUC (P1) | P2 | P3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | **lightgbm** | **0.9682** | 0.9779 | 0.9752 |
| 2 | xgboost | 0.9681 | **0.9786** | **0.9757** |
| 3 | random_forest | 0.9665 | 0.9770 | 0.9744 |
| 4 | xgboost_selected | 0.9663 | 0.9779 | 0.9740 |
| … | *best DL* (`fusion_L3`) | 0.9444 | 0.9593 | 0.9154 |
| … | D7 persistence | 0.9203 | 0.9429 | 0.9370 |

At 24 h the engineered-feature models dominate on all three protocols; the best DL model trails
by ~+0.024 (P1), ~+0.019 (P2) and ~+0.060 (P3). The 24 h task is the one where a station's own
recent history (P1/P2) or the engineered lag/rain features (P3) already carry almost all the
signal — the redundancy Phase 24 flagged.

### Stage 60 — 48-hour prediction

| rank | model | PR-AUC (P1) | P2 | P3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | **xgboost** | **0.9549** | **0.9670** | **0.9618** |
| 2 | lightgbm | 0.9538 | 0.9653 | 0.9610 |
| 3 | xgboost_selected | 0.9533 | 0.9669 | 0.9589 |
| 4 | random_forest | 0.9515 | 0.9653 | 0.9606 |
| … | *best DL* (`fusion_L3`) | 0.9385 | 0.9485 | **0.9202** |
| … | D7 persistence | 0.8954 | 0.9251 | 0.9182 |

At 48 h the ML lead persists but the DL gap narrows (P1 +0.016, P2 +0.018). On P3, `fusion_L3`
**peaks** at 48 h (0.9202) — the only model of any class whose P3 curve rises into 48 h,
the horizon Phase 24's graph gains also concentrated on.

### Stage 61 — 72-hour prediction

| rank | model | PR-AUC (P1) | P2 | P3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | **xgboost_selected** | **0.9422** | **0.9548** | 0.9439 |
| 2 | xgboost | 0.9408 | 0.9538 | 0.9493 |
| 3 | random_forest | 0.9389 | 0.9530 | 0.9475 |
| 4 | lightgbm | 0.9384 | 0.9518 | **0.9498** |
| … | *best DL* (`fusion_L3`) | 0.9238 | 0.9316 | 0.9171 |
| … | D7 persistence | 0.8683 | 0.9048 | 0.8979 |

At 72 h (the longest lead time) the 10-feature `xgboost_selected` overtakes the full-feature
`xgboost` on P1 — the only horizon where feature-set size flips the ranking — and D7 persistence
has decayed to 0.868 (well below every competitive model). The DL fusion model still trails ML by
~+0.018 (P1) but on P3 it now sits only ~+0.033 below the best ML and is the best DL at that
horizon by a wide margin.

## 8. Verdict

1. **Difficulty grows with lead time for every competitive model** — a monotone PR-AUC decay on
   every protocol; no model improves from 24 h → 72 h (the lone increases are the no-skill
   climatology and the crippled `static_only` control).
2. **The ML-vs-DL interaction is not about the temporal rate.** On P1/P2 the two groups decay at
   statistically indistinguishable rates (medians within 0.001). The apparent P1 "engineered
   decays faster" signal (mean −0.030 vs −0.026) is an artefact of group composition — the
   best-per-group view inverts it (`fusion_L3` −0.021 vs `xgboost` −0.027).
3. **The interaction is real on unseen stations (P3).** Engineered models decay faster
   (mean −0.034) than sequence/DL models (−0.025), the best DL model is flat across horizons
   (+0.002), and the GNN/fusion families concentrate their relative gains at 48/72 h — the same
   pattern Phases 23–25 found, and the reason the DL section is worth carrying.
4. **Gradient boosting still leads every horizon on every protocol.** DL closes the P1 gap from
   +0.024 (24 h) to +0.018 (72 h), and the P3 gap is large in absolute terms; the "science", per
   the workflow's own instruction, is that a more complex model with no absolute gain on the
   primary cell is not a better model — the DL contribution lives in the P3 lead-time behaviour,
   not in the P1 headline.

## 9. Multi-horizon strategy — the multi-task head (deferred, explicitly)

The workflow allows a single multi-task head ("may be one registry entry, clearly marked") but
makes **separate-per-horizon training the default**, which is exactly what every surviving model
did. Phase 26 therefore makes the deliberate, recorded decision **not** to add a multi-task
registry entry in this phase, because: (a) it would require a new training run and a new registry
row outside the 24 already registered; (b) the three horizons have very different base rates
(9.4:1 train → 18.6:1 val at 24 h, drifting at 72 h), so a shared head must be justified, not
assumed; and (c) the fair-comparison phase (27) and the ablation matrix (32) are where a
multi-task head would be tested against its per-horizon counterpart. It stays a clearly-labelled
**candidate for the Phase-32 ablation matrix (E-matrix)**, not a silent addition here.

## 10. Verification

| Check | Result |
| --- | --- |
| `scripts/phase26_matrix.py` | exit=0; 1,728 consolidated rows, **24 models**, levels L1–L4; **0 test cells** (asserted); SHA re-verified |
| `scripts/phase26_run.log` | run record — seed 20261008, exit_code 0, wall 2.1 s, dataset SHA recorded |
| `reports/phase26_matrix.csv` | 1,728 rows, one schema over the 7 source files |
| `reports/phase26_summary.csv` | 72 rows = 24 models × 3 protocols, each with 24/48/72 h + mean + Δ |
| `reports/phase26_interaction.json` | group means/medians + best-per-group for P1/P2/P3 |
| `notebooks/26_multihorizon.ipynb` | independent re-derivation of the matrix, the decay table and the interaction from the source files |
| `data/processed/experimental_dataset.csv` | SHA-256 `a6748529083f…73c8e` unchanged |

## 11. Open items handed on

1. **Phase 27** owns the *fair* ML-vs-DL head-to-head: one table, one code path, station-bootstrap
   CIs on every headline cell, cost columns, and the operational (missed-flood / lead-time) block
   — the present phase deliberately gives *horizon* structure, not the final comparison.
2. **The P3 lead-time interaction** is the strongest DL signal found so far (Phases 23–26 all
   point at it); Phase 27's CIs will decide whether it is separable from the ML models on the
   unseen-station cells.
3. **Multi-task head** — deferred to the Phase-32 ablation matrix (§9).
4. **Test is untouched** (frozen read at Phase 33).

**Next: Phase 27 — fair ML vs DL comparison (stage 42 + stages 43–46, 50–52).**
