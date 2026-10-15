# Phase 33 — Final Model Selection & the Single Frozen Test Read
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase33_selection.md`, `scripts/phase33_selection.py`,
> `reports/phase33_scorecard.csv`, `reports/phase33_test_read_log.json`,
> `models/final/`. Previous study: Phase 31–32 (robustness & ablation). **This is the climax
> of the project.**

---

## 0. The 60-second version

Phase 33 answers one question: **which single model goes forward?** It does **not** simply pick
the highest number. It applies a **pre-registered multi-criteria scorecard** on **validation and
spatial-CV evidence only**, freezes the choice, and *then* reads the locked test set **exactly
once** to confirm it.

**The winner: `xgboost` (registry L2-03) on the 62 engineered features** — the plain gradient-
boosting table model, not deep learning.

Why it wins (in the order the rule applies):
1. **Primary** — highest horizon-mean `val_2022` PR-AUC (**0.95463**).
2. **Co-primary (FN ≫ FP)** — the **fewest missed floods** of any model: **87**.
3. **Robustness** — beats the D7 bar on **all 21** dev points (floor +0.0144).
4. **Calibration** — OOF ECE **0.0031**, the best of the four eligible models.
5. **Tie-breakers** — exact TreeSHAP interpretability and cheap inference.

**Only four models** clear both gates — and all four are **L2 tree ensembles**. Every deep model
**fails Gate 1** (the best, `fusion_L3`, collapses by −0.14 on the hardest spatial fold).

**On the frozen test (read once):** the selected model beats D7 on **20/21** cells, **halves the
missed floods** (2,478 → **1,044**, −58 %), lifts mean row recall **+12.5 pts**, and raises
**event recall to 0.967**. The single honest exception — `test_unseen` @ 24 h — is a **statistical
tie** with D7, reported as such.

---

## 1. Where Phase 33 sits

| Phase | What | Plain words |
|---|---|---|
| 31–32 | Robustness & ablation | Does it hold? What matters? |
| **➜ 33** | **Selection + frozen test** | **Pick the champion, open the test once** |
| 34 | Explainable AI | (next, after the freeze) |

Closes workflow **stage 57**. After this, the model is **frozen** in `models/final/`.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Scorecard** | Multi-criteria decision table (not "accuracy wins"). |
| **Pre-registered rule** | The selection criteria written down *before* seeing the test. |
| **Gate 1 (skill)** | Must beat D7 on **every** dev point (21 of them). |
| **Gate 2 (calibration)** | Must be usable for risk bands (ECE ≤ 0.02). |
| **Primary criterion** | Horizon-mean `val_2022` PR-AUC. |
| **Co-primary** | Missed-flood count at the 10:1 operating point. |
| **D7 floor** | The worst D7 margin across all dev cells (must stay positive). |
| **Non-separable** | Overlapping 95 % CIs → the difference is within noise. |
| **Tie-breaker** | A secondary criterion used only when models aren't separable. |
| **Frozen test read** | Opening the ≥ 2023 data **once**, after the choice is locked. |
| **n_reads** | Counter proving the test was read once (here = 1). |
| **Event recall** | Fraction of flood *events* (not rows) that got a warning. |
| **TreeSHAP** | Exact feature-attribution for tree models (interpretability). |
| **Frozen artefact** | `models/final/` — pipelines + config + seed + dataset SHA. |

---

## 3. The selection rule (locked before any test read)

1. **Gate 1 (skill):** beat D7 on **every** dev point — 7 cells × 3 horizons = **21 points**.
2. **Gate 2 (calibration):** `cleared_all`, or best OOF ECE ≤ 0.02.
3. `climatology` / `persistence` are **bars, never candidates**.
4. Survivors ranked by **primary** (horizon-mean `val_2022` PR-AUC) → **co-primary** (fewest
   missed floods) → robustness floor → cost → interpretability.
5. Overlapping 95 % CIs ⇒ **not separable**; later tie-breakers decide.

Every number comes from **frozen** Phase-16/20–32 artefacts — **no re-tuning, no re-fitting**,
and **nothing in the scorecard touches the test**.

---

## 4. The scorecard (dev evidence only)

`D7 min` = worst D7 margin over the 7 dev cells; `P3 mean` = mean spatial-fold margin;
`missed` = missed-flood rows summed over 24/48/72 h at the Phase-29 threshold.

| # | model | lvl | P1 mean | D7 min | P2 worst | P3 mean | ECE | missed | eligible |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **1** | **xgboost** | L2 | **0.95463** | +0.0144 | +0.0184 | +0.0445 | 0.0031 | **87** | ✅ |
| 2 | xgboost_selected | L2 | 0.95393 | +0.0143 | +0.0176 | +0.0412 | 0.0035 | 88 | ✅ |
| 3 | lightgbm | L2 | 0.95344 | +0.0144 | +0.0174 | +0.0443 | 0.0036 | 112 | ✅ |
| 4 | random_forest | L2 | 0.95230 | +0.0134 | +0.0168 | +0.0432 | 0.0032 | 96 | ✅ |
| — | gradient_boosting | L2 | 0.94712 | **−0.0192** | +0.0107 | +0.0190 | 0.0045 | 101 | ❌ G1 |
| — | fusion_L3 | L3 | 0.93559 | −0.1432 | +0.0004 | −0.0001 | 0.0054 | 109 | ❌ G1 |
| — | transformer_bestL | L4 | 0.93168 | −0.1000 | −0.0081 | −0.0052 | 0.0039 | 112 | ❌ G1 |
| — | lstm_L3 | L3 | 0.92677 | −0.1409 | −0.0073 | −0.0166 | 0.0043 | 113 | ❌ G1 |
| — | logreg_linear | L1 | 0.92316 | −0.0562 | −0.0064 | −0.0140 | 0.0048 | 144 | ❌ G1 |
| — | *persistence (D7 bar)* | L1 | *0.89465* | *0.0000* | *0.0000* | *0.0000* | *0.0044* | *293* | bar |
| — | static_only | L3 | 0.53249 | −0.8378 | −0.7706 | −0.7793 | 0.0333 | 905 | ❌ G1,G2 |
| — | *climatology* | L1 | *0.05868* | *−0.8784* | *−0.8704* | *−0.8119* | *0.0003* | *3042* | bar |

**Four models clear both gates** — `xgboost`, `xgboost_selected`, `lightgbm`, `random_forest` —
**all L2 tree ensembles**. Everything else fails Gate 1; the best deep model (`fusion_L3`) sits
*at* the D7 bar on average but **collapses by −0.14 on the hardest fold**.

---

## 5. The choice — `xgboost` (L2-03), frozen

| | |
|---|---|
| **Model** | `xgboost`, registry L2-03, config `xgboost_c01` |
| **Features** | FULL — 62 (Phase-12 set) |
| **Hyper-parameters** | `n_estimators=300, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=1.0, colsample_bytree=0.8` |
| **Training rows** | P1 `train_<2022` (≤ 2021), holdout excluded: 61,977 / 60,992 / 60,144 @ 24/48/72 h |
| **Seed** | `20261008` |
| **Params / cost** | 15,002 tree nodes; 1.15 s to fit all three horizons; **0.066 ms/1,000 rows** inference |
| **Frozen** | `models/final/selected_model.pkl` (+ `final_config.json`, `README.md`) |

**Non-separability (stated, not hidden):** the four eligible models share a common 95 % CI at
**every** horizon (24 h `[0.9324, 0.9860]`, 48 h `[0.9075, 0.9780]`, 72 h `[0.8862, 0.9680]`).
Their PR-AUC differences are within noise, so the choice rests on the **co-primary (missed
floods)** and the tie-breakers — exactly as the scorecard intends. Keeping **one** frozen family
is the honest, simpler call.

---

## 6. What was rejected, and why

| Family | Verdict | Reason |
|---|---|---|
| L1 `climatology`/`persistence` | excluded | bars, not candidates |
| L1 linear (`logreg_*`) | rejected | below D7 at some dev cell (floor −0.040 to −0.056) |
| L1 `tree_depth4` | rejected | −0.0231 floor; underfits |
| L2 `gradient_boosting` | rejected | Gate 1: −0.0192 below D7 |
| L2 `random_forest` | not selected | eligible but 4th on primary; costliest inference |
| L2 `lightgbm`, `xgboost_selected` | not selected | inseparable, but **more missed floods** (112 / 88 vs 87) |
| L3 LSTM/GRU/TCN | rejected | Gate 1: −0.107 to −0.141; recurrent nets also miscalibrated |
| L3 fusion (`fusion_L3`, …) | rejected | Gate 1: −0.143 floors |
| L4 graph/transformer | rejected | Gate 1: −0.100 to −0.145; gains horizon-specific |

**The scientific verdict:** after 33 phases, the final model is the **gradient-boosting tabular
model on 62 engineered features**. **Deep learning does not beat it** on the primary metric — the
signal is **hydrology + geography**, not sequence shape.

---

## 7. The frozen test read (once) — the operational result

Test rows: **101,174**. `n_reads = 1`; D7 reproduced from Phase 16 to |Δ| ≤ 5.6e−17.

### PR-AUC on the frozen test (`xgboost` vs D7), all-season

| cell | 24 h sel / D7 | 48 h sel / D7 | 72 h sel / D7 |
|---|---|---|---|
| `test_2023` | 0.7473 / 0.6276 | 0.6575 / 0.5523 | 0.6254 / 0.5108 |
| `test_2024` | 0.9004 / 0.8022 | 0.8461 / 0.7302 | 0.7930 / 0.6727 |
| `test_2025` | 0.7967 / 0.7328 | 0.7633 / 0.6670 | 0.7176 / 0.6041 |
| `test_2026` | 0.7660 / 0.7233 | 0.7324 / 0.6495 | 0.7068 / 0.5929 |
| `test_trained` | 0.8480 / 0.7468 | 0.7990 / 0.6762 | 0.7567 / 0.6183 |
| `test_new_network` | 0.8177 / 0.7356 | 0.7695 / 0.6930 | 0.7290 / 0.6457 |
| `test_unseen` | **0.7675 / 0.7696** | 0.7001 / 0.6694 | 0.6360 / 0.5986 |

- **Selected ≥ D7 on 20/21 cell × horizon points.** Horizon-mean over the four test years:
  24 h **0.8026 vs 0.7215**, 48 h **0.7498 vs 0.6498**, 72 h **0.7107 vs 0.5951**.
- **The one exception is honest and specific:** `test_unseen` @ **24 h** — 0.7675 vs 0.7696
  (**−0.0020**) with almost complete CI overlap. At the **shortest lead time at completely unseen
  stations**, the learned model is **indistinguishable from D7**. At 48/72 h on the same stations
  it wins (+0.031 / +0.037), and on **new-network** stations it wins clearly (+0.082/+0.077/+0.083).
- Monsoon (4 test years, mean): 24 h **0.8107 vs 0.7311**, 48 h **0.7642 vs 0.6635**,
  72 h **0.7276 vs 0.6135**.

### Operational block (Phase-29 thresholds, FN ≫ FP)

| | missed floods | TP | FP | mean recall | mean FAR | event recall |
|---|---:|---:|---:|---:|---:|---:|
| **xgboost** | **1,044** | 9,530 | 15,258 | **0.894** | 0.608 | **0.967** |
| D7 persistence | 2,478 | 8,096 | 14,370 | 0.769 | 0.614 | 0.901 |

The selected model **halves the missed floods** (2,478 → **1,044**, −58 %), lifts mean row recall
**+12.5 pts**, and raises **event recall to 0.967**, at essentially the same false-alarm rate as
D7 — exactly the FN ≫ FP trade the design asks for.

---

## 8. Named findings

- **F33-1** The final model is **gradient boosting, not deep learning**.
- **F33-2** **Four models are statistically inseparable**; the winner is decided by missed floods
  (87) and tie-breakers, not a noise-level PR-AUC gap.
- **F33-3** **The test confirms it:** 20/21 wins; horizon-mean PR-AUC +0.08/+0.10/+0.12 over D7;
  missed floods 1,044 vs 2,478.
- **F33-4** **The one limit is documented:** `test_unseen` @ 24 h **ties** D7 (−0.002, CI
  overlap) — with no local history and only 24 h lead time, persistence is as informative.
- **F33-5** **Generalisation is positive:** new-network (+0.077 to +0.083) and unseen stations at
  48/72 h (+0.031/+0.037) both clear D7.

---

## 9. Limitations & warnings (kept explicit)

1. **Non-separability** — a different tie-break could pick `lightgbm`/`xgboost_selected` with no
   measurable loss; the frozen artefact records the *chosen* model, not a unique optimum.
2. **`test_unseen` @ 24 h is a tie, not a win** — reported, not smoothed over.
3. **Thresholds frozen from Phase 29** — they were **not** re-derived on test (that would be
   leakage); results carry the `C_FN:C_FP = 10:1` assumption.
4. **Low test base rates** (0.7–6 %) make absolute PR-AUC regime-dependent; the **D7-relative
   margin** is the stable comparison.
5. **Only two models touched the test** (the selection + D7) — by construction.

---

## 10. Files referenced

| File | Purpose |
|---|---|
| `reports/phase33_selection.md` | Full Phase 33 report |
| `scripts/phase33_selection.py` | Scorecard → gate/rank → freeze → one guarded test read |
| `reports/phase33_scorecard.csv` | 24 models × criteria |
| `reports/phase33_selection.json` | The pre-registered rule + ranked choice |
| `models/final/selected_model.pkl` | The frozen pipelines (per horizon) |
| `models/final/final_config.json`, `README.md` | Config (no pickles) + load recipe |
| `reports/phase33_test_metrics.csv` | The single test read (126 rows) |
| `reports/phase33_test_read_log.json` | Who/when/which models/rows (= 1 read) |
| `figures/phase33_scorecard.png` | Scorecard + test vs D7 |
| `notebooks/33_selection.ipynb` | 12/12 checks |

---

## 11. Verdict (the project's answer)

- **The champion is `xgboost` (L2-03, FULL 62)** — a gradient-boosting table model, **frozen**
  with config, seed and dataset SHA.
- **Only four models passed both gates — all tree ensembles.** Every deep model **failed the
  skill gate**; the best collapsed on the hardest spatial fold.
- **The test, opened once and afterwards sealed, confirms the choice:** 20/21 wins, **58 % fewer
  missed floods**, **+12.5 pts row recall**, **event recall 0.967** at D7's false-alarm rate.
- **One honest limit:** at the **shortest lead time on completely unseen stations**, the model
  **ties** persistence — reported, not hidden.

**Bottom line:** After 33 phases of data work, leakage audits, fair comparisons, calibration and
thresholding, the honest answer is that a **well-built gradient-boosted model on hydrological and
geographic features** is the best operational system here — and it beats the strong persistence
baseline where it matters most, while openly admitting the one place it doesn't. **That
willingness to report the tie is the project's real credential.**

---

**This completes the 33-phase study series.** The remaining work (`Phase 34 — Explainable AI`)
needs `pip install shap`, and the separate **MI-v2 improvement sprint** (ERA5-Land soil, GloFAS
discharge, NASA POWER soil wetness, HPO, ensembling) continues outside the frozen contract,
producing `models/final_v2/` without ever re-reading the test.
