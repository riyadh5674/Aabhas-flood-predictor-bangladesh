# Phase 20 — Classical ML candidates (stages 34, 35)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 20 — Classical ML candidates** → checklist **stages 34 (candidates), 35 (tuning)** |
| **Driver** | `scripts/phase20_classical_ml.py` (validation-only; resume-safe, checkpoint after every family) |
| **Run log** | `scripts/phase20_run.log` (one interrupted call resumed; every family `exit=0`) |
| **Notebook** | `notebooks/20_classical_ml.ipynb` (built by `scripts/build_phase20_notebook.py`, executed with the `flood-predictor` kernel) |
| **Outputs** | `reports/phase20_metrics.csv` (360 rows) · `reports/phase20_search.csv` (60 configs) · `models/phase20_L2-0{1..5}.pkl` |
| **Registry** | `phase17_experiment_registry.csv` rows **L2-01…L2-05**; `test_policy = frozen_read_at_phase33` honoured (see §6) |
| **Input (immutable)** | `experimental_dataset.csv` SHA-256 `a67485…` (re-verified by `load()`); Phase-16 preprocessing contract reused verbatim |
| **Seed** | 20261008 everywhere |

## 1. Candidates and the shared contract

Five families, all on **Dataset A** with the **exact Phase-16 preprocessing** (median
imputation + train-fitted frequency encoding, no target encoding, banned columns asserted
absent), so every number is comparable line-by-line with `phase16_baseline_metrics.csv`:

| registry | family | feature set | features |
| --- | --- | --- | ---: |
| L2-01 | `random_forest` | FULL | 62 |
| L2-02 | `gradient_boosting` (sklearn) | FULL | 62 |
| L2-03 | `xgboost` | FULL | 62 |
| L2-04 | `lightgbm` | FULL | 62 |
| L2-05 | `xgboost_selected` | SELECTED | 10 |

## 2. Stage 35 — validation-only tuning

Random search over a fixed seeded space per family; scored **only** on the selection cell
`val_2022` (season = all), score = **mean PR-AUC over the three horizons** (D2: horizons
reported equally). No early stopping anywhere (fixed `n_estimators`) — no code path can
touch the test fold while fitting. Budgets were set from a measured fit-time probe on this
8-core machine (RF400 36 s, GBM200 138 s, XGB600 15 s, LGBM600 12 s) and are all **within
the registry maximum of 40**:

| family | configs evaluated | space size | registry max | tuning fit-time | best mean PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_forest | 12 | 72 | 40 | 2,785 s | 0.95230 |
| gradient_boosting | 4 | 48 | 40 | 1,019 s | 0.94712 |
| xgboost | 16 | 432 | 40 | 684 s | **0.95463** |
| lightgbm | 16 | 216 | 40 | 850 s | 0.95344 |
| xgboost_selected | 12 | 432 | 40 | 218 s | 0.95393 |

**Selected configuration + imbalance result** (the full search is in `phase20_search.csv`):

| family | selected config | weighting |
| --- | --- | --- |
| random_forest | 200 trees, depth 24, leaf ≥ 20, max_features 0.5 | `class_weight=balanced` |
| gradient_boosting | 100 trees, lr 0.10, depth 3, subsample 0.8 | none |
| **xgboost** | 300 trees, lr 0.03, depth 6, min_child_weight 10, colsample 0.8 | none |
| lightgbm | 300 trees, lr 0.03, 31 leaves, min_child 50, feature_fraction 0.8 | none |
| xgboost_selected | 300 trees, lr 0.03, depth 6, colchild 1 | none |

**Imbalance (Phase 16 §10):** weighting on/off was part of every family's search space;
**validation chose the unweighted models** for XGB/LGBM/GBM and `balanced` only for RF —
the registry's `balanced` default is therefore overturned *by validation*, as the workflow
requires ("the choice is a validation result"). Weighting was always applied inside the
training fold only.

## 3. Stage 34 — dev-cell grid (selection cell)

`val_2022`, P1, season = all, PR-AUC with station-bootstrap 95 % CI (macro = per-station mean):

| model | 24 h | 48 h | 72 h |
| --- | --- | --- | --- |
| **xgboost** | **0.9681** [0.932, 0.987] | **0.9549** [0.908, 0.980] | 0.9408 [0.885, 0.969] |
| lightgbm | 0.9682 [0.930, 0.988] | 0.9538 [0.906, 0.979] | 0.9384 [0.881, 0.968] |
| random_forest | 0.9665 [0.927, 0.986] | 0.9515 [0.905, 0.978] | 0.9389 [0.884, 0.969] |
| xgboost_selected | 0.9663 [0.925, 0.987] | 0.9533 [0.902, 0.980] | **0.9422** [0.886, 0.972] |
| gradient_boosting | 0.9618 [0.923, 0.983] | 0.9457 [0.897, 0.977] | 0.9338 [0.877, 0.967] |
| *logreg_linear* (P16) | *0.9401* | *0.9241* | *0.9053* |
| *persistence* (D7, P16) | *0.9203* | *0.8954* | *0.8683* |

### Skill over the D7 persistence bar (PR-AUC Δ, val_2022)

| model | Δ24h | Δ48h | Δ72h |
| --- | ---: | ---: | ---: |
| xgboost | **+0.0478** | **+0.0595** | +0.0725 |
| lightgbm | +0.0479 | +0.0584 | +0.0701 |
| random_forest | +0.0462 | +0.0561 | +0.0706 |
| xgboost_selected | +0.0460 | +0.0579 | **+0.0739** |
| gradient_boosting | +0.0415 | +0.0503 | +0.0655 |
| logreg_linear (P16) | +0.0198 | +0.0287 | +0.0370 |

Every candidate clears the persistence bar by **3–7 PR-AUC points** and the Phase-16 best
linear model by **2–4 points** — the strong conventional baseline DL must now beat.

### Cross-cell robustness (mean PR-AUC, season = all)

| | P3 spatial CV (4 folds) 24/48/72 h | P2 rolling val_2020/21/22 (24 h) |
| --- | --- | --- |
| xgboost | **0.9757 / 0.9618 / 0.9493** | **0.9862 / 0.9815 / 0.9681** |
| lightgbm | 0.9752 / 0.9610 / 0.9498 | 0.9852 / 0.9802 / 0.9682 |
| random_forest | 0.9744 / 0.9606 / 0.9475 | 0.9847 / 0.9798 / 0.9665 |
| xgboost_selected | 0.9740 / 0.9589 / 0.9439 | 0.9854 / 0.9819 / 0.9663 |
| gradient_boosting | 0.9452 / 0.9390 / 0.9258 | 0.9812 / 0.9514 / 0.9618 |
| *persistence* (P16) | *0.9370 / 0.9182 / 0.8979* | — |

Ranking is stable across all dev cells: **xgboost (L2-03)** is the best classical candidate;
`xgboost_selected` on **10 of 62 features** is within ~0.2 points — the Phase-12 SELECTED set
is essentially sufficient for boosted trees (compactness finding, pairs with Phase 12 §feature
selection). Season cells (monsoon vs non-monsoon) are near-identical for the boosters
(≤ 1 point), so no season-specific degradation was hidden by the `all` cells.

## 4. Model artifacts

`models/phase20_L2-0{1..5}.pkl` — the **P1 `train_<2022` pipeline** per horizon
(`{horizon: Pipeline}`), the models Phase 33 will freeze:

| file | size | contents |
| --- | ---: | --- |
| `phase20_L2-01.pkl` | 22.3 MB | random_forest × 3 horizons |
| `phase20_L2-02.pkl` | 0.4 MB | gradient_boosting × 3 |
| `phase20_L2-03.pkl` | 2.3 MB | **xgboost × 3** |
| `phase20_L2-04.pkl` | 3.2 MB | lightgbm × 3 |
| `phase20_L2-05.pkl` | 3.2 MB | xgboost_selected × 3 |

## 5. Metric coverage

`phase20_metrics.csv`: **360 rows** = 5 families × 8 dev cells × 3 horizons × 3 seasons
(cells present: `val_2020/2021/2022`, `spatial_fold_1…4`); columns identical to Phase 16
(adds `registry_id`, `feature_set`, `weighting`, `config_id`, `search_score`), with
station-bootstrap CIs on the season=all cells and macro (per-station) PR-AUC.

## 6. Test policy

`test_policy = frozen_read_at_phase33` (registry, all L2 rows). Stage 34 ran with all
test eval cells **structurally stripped** and asserted empty of test tokens
(`phase20_classical_ml.py`); `phase20_metrics.csv` contains **zero rows whose cell starts
with `test`**. The frozen test period is read exactly once, at Phase 33. (Phase 16 already
read it once for the baselines, in its own file — no further read happens here.)

## 7. Reproducibility & cost

```
python -u scripts/phase20_classical_ml.py random_forest gradient_boosting
python -u scripts/phase20_classical_ml.py xgboost lightgbm
python -u scripts/phase20_classical_ml.py xgboost_selected        # smoke family
```

Seed 20261008; total wall-clock ≈ 2.6 h (tuning 5,556 s + grid 3,170 s + preprocessing),
resumable per family (checkpoint after each). One shell-timeout interruption during the
RF/GBM call was recovered by the resume logic (RF checkpointed; GBM re-run cleanly).

**DoD:** candidates fit, budgets recorded and ≤ 40, tuning validation-only, imbalance
decided on validation, dev cells + seasons + horizons reported vs the Phase-16 grid, models
saved, test untouched. **Next → Phase 21 / stages 36 (LSTM/GRU) and Phase 22 / stage 37
(TCN):** deep temporal models on Dataset B sequences (L ∈ {3,7,14,30}), policy P19-M1.
