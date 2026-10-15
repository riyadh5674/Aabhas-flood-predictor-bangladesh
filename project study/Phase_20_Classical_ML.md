# Phase 20 — Classical Machine Learning Candidates
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase20_classical_ml.md`, `scripts/phase20_classical_ml.py`,
> `reports/phase20_metrics.csv`, `reports/phase20_search.csv`, `models/phase20_L2-0{1..5}.pkl`.
> Previous study: Phase 17 (framework), 18–19 (Dataset B). This phase produced the family that
> later wins the whole project (**xgboost, L2-03**).

---

## 0. The 60-second version

Phase 20 is where "real" machine learning enters: **classical tabular models** on Dataset A.
Five families were tuned and compared on the same grid as every other model:

- `random_forest` (L2-01) — many decision trees, averaged.
- `gradient_boosting` (L2-02) — sklearn's classic boosting.
- **`xgboost` (L2-03) — the tabular workhorse, and the eventual winner.**
- `lightgbm` (L2-04) — leaf-wise boosting, fast, handles NaN natively.
- `xgboost_selected` (L2-05) — XGBoost on just 10 features (compactness test).

Results on the selection cell (`val_2022`, all season):

| Model | 24 h | 48 h | 72 h |
|---|---:|---:|---:|
| **xgboost** | **0.9681** | **0.9549** | 0.9408 |
| lightgbm | 0.9682 | 0.9538 | 0.9384 |
| random_forest | 0.9665 | 0.9515 | 0.9389 |
| xgboost_selected | 0.9663 | 0.9533 | **0.9422** |
| *logreg_linear* (P16) | *0.9401* | *0.9241* | *0.9053* |
| *persistence* (D7, P16) | *0.9203* | *0.8954* | *0.8683* |

**Headline:** every candidate clears the D7 persistence bar by **3–7 PR-AUC points** — a big,
real jump over Phase 16's linear models. **XGBoost wins**, and notably `xgboost_selected` on
**10 of 62 features** is within ~0.2 points — the compact set is nearly sufficient.

Millions of small decisions, one rule throughout: **tune on validation only, never touch test.**

---

## 1. Where Phase 20 sits

| Phase | What | Plain words |
|---|---|---|
| 17 | Framework / registry | The rulebook |
| 18–19 | Dataset B + validation | DL input (not used here) |
| **➜ 20** | **Classical ML (L2)** | **Train & tune the workhorse models** |
| 21–22 | Temporal DL (L3) | LSTM/GRU/TCN on sequences |
| 23 | Fusion | Combine sequence + static + engineered |
| 26–33 | Calibrate → select → freeze | Pick the champion, read test once |

Closes checklist **stages 34 (candidates)** and **35 (tuning)**.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Random forest** | Many decision trees, each on random data/features; predictions averaged. |
| **Gradient boosting** | Trees built one at a time, each fixing the previous ones' errors. |
| **XGBoost / LightGBM** | Fast, strong boosted-tree libraries (the tabular workhorses). |
| **Hyperparameter** | A setting chosen *before* training (e.g. tree depth, learning rate). |
| **Random search** | Try random combinations of hyperparameters within a fixed space. |
| **Tuning budget** | Max configs allowed (here ≤ 40 per family, fixed in Phase 17). |
| **Selection cell** | `val_2022` — the one validation cell used to pick configs. |
| **Mean PR-AUC over horizons** | The tuning score = average of 24/48/72 h PR-AUC (D2). |
| **Early stopping** | Stopping training when a *validation* score stops improving. Here: **not used** (to avoid any test-fold contact). |
| **`class_weight=balanced`** | Telling the model to care more about rare positives. |
| **Station-bootstrap CI** | Resampling *stations* to get an honest uncertainty range. |
| **Macro per-station** | Average of per-station scores (fairer than pooling). |
| **Compactness** | Can a small feature set match a large one? |

---

## 3. The shared contract (fair comparison)

All five families used **Dataset A** with the **exact Phase-16 preprocessing**:
- median imputation (numerics),
- **train-fitted frequency encoding** of the 5 text columns (never target encoding),
- banned columns asserted absent.

This means every Phase-20 number is comparable **line-by-line** with Phase 16.

| Registry | Family | Feature set | # features |
|---|---|---|---:|
| L2-01 | `random_forest` | FULL | 62 |
| L2-02 | `gradient_boosting` | FULL | 62 |
| L2-03 | `xgboost` | FULL | 62 |
| L2-04 | `lightgbm` | FULL | 62 |
| L2-05 | `xgboost_selected` | SELECTED | 10 |

---

## 4. Tuning (stage 35) — validation-only, budget-respecting

Random search over a **fixed seeded space** per family; scored **only** on `val_2022`
(season = all) by **mean PR-AUC over 3 horizons**. **No early stopping anywhere** (fixed
`n_estimators`) — so no code path can touch the test fold.

| Family | Configs tried | Space size | Registry max | Best mean PR-AUC |
|---|---:|---:|---:|---:|
| random_forest | 12 | 72 | 40 | 0.95230 |
| gradient_boosting | 4 | 48 | 40 | 0.94712 |
| **xgboost** | 16 | 432 | 40 | **0.95463** |
| lightgbm | 16 | 216 | 40 | 0.95344 |
| xgboost_selected | 12 | 432 | 40 | 0.95393 |

### Selected configurations

| Family | Selected config | Weighting |
|---|---|---|
| random_forest | 200 trees, depth 24, leaf ≥ 20, max_features 0.5 | `class_weight=balanced` |
| gradient_boosting | 100 trees, lr 0.10, depth 3, subsample 0.8 | none |
| **xgboost** | **300 trees, lr 0.03, depth 6, min_child_weight 10, colsample 0.8** | **none** |
| lightgbm | 300 trees, lr 0.03, 31 leaves, min_child 50, feature_fraction 0.8 | none |
| xgboost_selected | 300 trees, lr 0.03, depth 6, colchild 1 | none |

**The imbalance decision was a *validation result*, not a guess:** weighting on/off was part
of every family's search space, and validation chose **unweighted** models for XGB/LGBM/GBM
and `balanced` only for RF. This overturns the registry's `balanced` default — exactly as the
workflow requires ("the choice is a validation result"). Weighting was **always inside the
training fold only**.

---

## 5. Selection-cell results (stage 34), with bootstrap CIs

`val_2022`, P1, season = all. PR-AUC with station-bootstrap 95 % CI:

| Model | 24 h | 48 h | 72 h |
|---|---|---|---|
| **xgboost** | **0.9681** [0.932, 0.987] | **0.9549** [0.908, 0.980] | 0.9408 [0.885, 0.969] |
| lightgbm | 0.9682 [0.930, 0.988] | 0.9538 [0.906, 0.979] | 0.9384 [0.881, 0.968] |
| random_forest | 0.9665 [0.927, 0.986] | 0.9515 [0.905, 0.978] | 0.9389 [0.884, 0.969] |
| xgboost_selected | 0.9663 [0.925, 0.987] | 0.9533 [0.902, 0.980] | **0.9422** [0.886, 0.972] |
| gradient_boosting | 0.9618 [0.923, 0.983] | 0.9457 [0.897, 0.977] | 0.9338 [0.877, 0.967] |

### Skill over the D7 persistence bar (PR-AUC Δ)

| Model | Δ24h | Δ48h | Δ72h |
|---|---:|---:|---:|
| xgboost | **+0.0478** | **+0.0595** | +0.0725 |
| lightgbm | +0.0479 | +0.0584 | +0.0701 |
| random_forest | +0.0462 | +0.0561 | +0.0706 |
| xgboost_selected | +0.0460 | +0.0579 | **+0.0739** |
| gradient_boosting | +0.0415 | +0.0503 | +0.0655 |
| logreg_linear (P16) | +0.0198 | +0.0287 | +0.0370 |

Every candidate beats persistence by **3–7 points** and the Phase-16 best linear model by
**2–4 points**. **This is the strong conventional baseline deep learning must now beat.**

---

## 6. Cross-cell robustness (does the win hold elsewhere?)

| Model | P3 spatial CV (4 folds) 24/48/72 h | P2 rolling val_2020/21/22 (24 h) |
|---|---|---|
| xgboost | **0.9757 / 0.9618 / 0.9493** | **0.9862 / 0.9815 / 0.9681** |
| lightgbm | 0.9752 / 0.9610 / 0.9498 | 0.9852 / 0.9802 / 0.9682 |
| random_forest | 0.9744 / 0.9606 / 0.9475 | 0.9847 / 0.9798 / 0.9665 |
| xgboost_selected | 0.9740 / 0.9589 / 0.9439 | 0.9854 / 0.9819 / 0.9663 |
| gradient_boosting | 0.9452 / 0.9390 / 0.9258 | 0.9812 / 0.9514 / 0.9618 |
| *persistence* (P16) | *0.9370 / 0.9182 / 0.8979* | — |

**The ranking is stable across all dev cells:** xgboost is the best classical candidate.
Season cells (monsoon vs non-monsoon) are near-identical for the boosters (≤ 1 point), so no
season-specific degradation was hidden by the `all` cells.

**Notable finding (compactness):** `xgboost_selected` on **10 of 62 features** is within ~0.2
points of full XGBoost. The Phase-12 SELECTED set is essentially sufficient for boosted trees.

---

## 7. Model artifacts & metric coverage

`models/phase20_L2-0{1..5}.pkl` — the **P1 `train_<2022` pipeline per horizon**
(`{horizon: Pipeline}`), i.e. the models Phase 33 will freeze:

| File | Size | Contents |
|---|---:|---|
| `phase20_L2-01.pkl` | 22.3 MB | random_forest × 3 horizons |
| `phase20_L2-02.pkl` | 0.4 MB | gradient_boosting × 3 |
| `phase20_L2-03.pkl` | 2.3 MB | **xgboost × 3** |
| `phase20_L2-04.pkl` | 3.2 MB | lightgbm × 3 |
| `phase20_L2-05.pkl` | 3.2 MB | xgboost_selected × 3 |

`phase20_metrics.csv`: **360 rows** = 5 families × 8 dev cells × 3 horizons × 3 seasons, with
station-bootstrap CIs on season=all cells and macro (per-station) PR-AUC.

---

## 8. Test protection (the hard rule)

`test_policy = frozen_read_at_phase33` for all L2 rows. This phase ran with all test eval
cells **structurally stripped** and asserted empty of test tokens. `phase20_metrics.csv`
contains **zero rows whose cell starts with `test`** — verified, not just promised.

**Plain point:** the test set is still sealed. The frozen read happens **once**, at Phase 33.

---

## 9. Reproducibility & cost

```powershell
python -u scripts\phase20_classical_ml.py random_forest gradient_boosting
python -u scripts\phase20_classical_ml.py xgboost lightgbm
python -u scripts\phase20_classical_ml.py xgboost_selected        # smoke family
```

Seed 20261008; total wall-clock ≈ **2.6 h** (tuning 5,556 s + grid 3,170 s + preprocessing),
resumable per family. One shell-timeout interruption was recovered by the resume logic.

---

## 10. Problems & solutions

| Problem | Solution |
|---|---|
| Risk of tuning on test | Tune on `val_2022` only; **no early stopping**; test cells stripped. |
| Tuning could bloat after good results | Budgets fixed in Phase 17 (≤ 40); recorded. |
| Imbalance handling could be guessed | Made part of the search space; decided **by validation**. |
| Weighting could leak | Applied **inside the training fold only**. |
| Pooled scores could mislead | Station-bootstrap CIs + macro per-station PR-AUC reported. |
| One horizon could hide weakness | Score = mean over all 3 horizons (D2). |
| Long training interrupted | Checkpoint per family; resume-safe. |

---

## 11. Files referenced

| File | Purpose |
|---|---|
| `reports/phase20_classical_ml.md` | Full Phase 20 report |
| `scripts/phase20_classical_ml.py` | Driver (validation-only, resume-safe) |
| `reports/phase20_metrics.csv` | 360 rows (dev grid) |
| `reports/phase20_search.csv` | 60 tuned configs |
| `models/phase20_L2-0{1..5}.pkl` | The five fitted pipelines |
| `reports/phase17_experiment_registry.csv` | L2-01…L2-05 definitions |
| `reports/phase16_baseline_metrics.csv` | Comparison floor (D7 bar) |
| `scripts/phase20_run.log` | Run log (exit 0) |

---

## 12. Verdict

- **Boosted trees clearly beat the linear baselines** (+2–4 points over logreg, +3–7 over D7).
- **XGBoost (L2-03) is the best classical candidate** — stable across P1/P2/P3 and seasons.
- **10 features nearly match 62** (`xgboost_selected`) — the compact set is powerful.
- **Imbalance decided by validation**, weighting only inside training.
- **Test untouched** (structurally stripped and asserted).
- Models saved for Phase 33 to freeze.

**Bottom line:** The workhorse models work — and a compact 10-feature XGBoost gets almost the
same result. That gives deep learning a **hard, honest target** to beat, and gives the project
a strong, simple fallback champion.

---

**Next in the series:** Phase 21–22 — Temporal deep learning (LSTM/GRU/TCN on Dataset B,
windows L = 3/7/14/30). Then Phase 23 (fusion), 24–25 (gated graph/transformer), 26–33
(calibration, cost-sensitive thresholds, selection, frozen test).
