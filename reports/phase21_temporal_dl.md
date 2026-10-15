# Phase 21 — Deep temporal learning I: LSTM & GRU (stage 36)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 21 — Deep temporal learning I: LSTM & GRU** → checklist **stage 36** |
| **Driver** | `scripts/train_sequence.py` (model = `lstm`\|`gru`\|`tcn`; resume-safe, checkpoint after every family) |
| **Run log** | `scripts/phase21_run.log` (per-family `exit=0`; final call `TOTAL 40625 s — exit=0`) |
| **Notebook** | `notebooks/21_temporal_dl.ipynb` (built by `scripts/build_phase21_notebook.py`, executed with the `flood-predictor` kernel) |
| **Outputs** | `reports/phase21_metrics.csv` (**360 rows**) · `reports/phase21_search.csv` (**46 configs**) · `models/phase21_L3-0{1..5}.pt` |
| **Registry** | `phase17_experiment_registry.csv` rows **L3-01…L3-05** (L3-06 `tcn_bestL` is Phase 22), `test_policy = frozen_read_at_phase33` honoured (§6) |
| **Input (immutable)** | `experimental_dataset.csv` SHA-256 `a6748529083fcdce…` re-verified by `load()`; Dataset B sequences (policy **P19-M1**) |
| **Policy** | `sequence_policy.json` P19-M1 keep-all + mask → **13 channels** = concat(12 raw vars, mask) |
| **Seed** | 20261008 everywhere |

## 1. Candidates and the shared contract

Five families on **Dataset B** raw sequences, all sharing the P19 policy and the Phase-20 dev
grid, so every number is comparable with `phase20_metrics.csv` / `phase16_baseline_metrics.csv`:

| registry | family | window L | model | kind |
| --- | --- | ---: | --- | --- |
| L3-01 | `lstm_L3` | 3 | LSTM | recurrence |
| L3-02 | `lstm_L7` | 7 | LSTM | recurrence |
| L3-03 | `lstm_L14` | 14 | LSTM | recurrence |
| L3-04 | `lstm_L30` | 30 | LSTM | recurrence |
| L3-05 | `gru_bestL` | **3** (= L\*) | GRU | recurrence |

**Training contract (Phase 21 §body).** BCE loss with an **unweighted baseline and a
`pos_weight = n_neg/n_pos` variant** in every search space; early stopping on **validation
PR-AUC** with a fixed epoch cap and recorded batch size; seed set; per-task (train-fold only)
standardisation with NaN→0 applied *after* standardisation, so a raw NaN never reaches a layer.

```text
input  X : (N, L, 13)   concat(12 raw variables, timestep mask)
        M : (N, L)      mask  (0 = calendar day absent, written all-NaN then masked)
head   : LSTM/GRU(L, hidden, layers) -> last hidden -> dropout -> Linear(hidden,1) -> sigmoid
```

## 2. Stage 36 — validation-only tuning

Fixed seeded space (96 cells/family) scored **only** on the selection cell `val_2022`
(season = all), score = **mean PR-AUC over the three horizons** (D2). No early stopping ever
touches the test fold. Budgets are within the registry maximum of 20:

| family | configs | space | budget max | tuning fit-time | best mean PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm_L3 | 12 | 96 | 12 | 1,126 s | **0.92677** |
| lstm_L7 | 10 | 96 | 10 | 1,466 s | 0.91146 |
| lstm_L14 | 8 | 96 | 8 | 2,914 s | 0.90775 |
| lstm_L30 | 6 | 96 | 6 | 11,958 s | 0.91688 |
| gru_bestL | 10 | 96 | 10 | 1,056 s | 0.92362 |

**Window selection (L\*):** mean val-2022 PR-AUC over the four LSTMs is **L3 0.9268 > L30
0.9169 > L7 0.9115 > L14 0.9078**, so **L\* = 3** (shortest window wins; long windows add
nothing here and cost more). `gru_bestL` is therefore tuned at L = 3.

**Selected configuration + imbalance** (full search in `phase21_search.csv`):

| family | selected config | params | weighting |
| --- | --- | ---: | --- |
| **lstm_L3** | lr 0.003, hidden 32, layers 2, dropout 0.0, batch 256 | 14,497 | `pos_weight=ratio` |
| lstm_L7 | lr 0.001, hidden 64, layers 2, dropout 0.3, batch 256 | 53,569 | `pos_weight=ratio` |
| lstm_L14 | lr 0.001, hidden 32, layers 2, dropout 0.3, batch 256 | 14,497 | none |
| lstm_L30 | lr 0.003, hidden 32, layers 2, dropout 0.0, batch 256 | 14,497 | none |
| **gru_bestL** | lr 0.001, hidden 64, layers 2, dropout 0.3, batch 256 | 40,193 | `pos_weight=ratio` |

Validation chose weighting **on** for the two 32/64-wide recurrent leaders (L3, GRU) and **off**
for L7/L14/L30 — the imbalance treatment is a validation result, decided inside the training fold.

## 3. Stage 36 — dev-cell grid (selection cell)

`val_2022`, P1, season = all, PR-AUC with station-bootstrap 95 % CI (macro = per-station mean):

| model | 24 h | 48 h | 72 h |
| --- | --- | --- | --- |
| **lstm_L3** | **0.9421** [0.865, 0.979] | 0.9253 [0.847, 0.967] | **0.9129** [0.838, 0.954] |
| gru_bestL | 0.9359 [0.842, 0.977] | **0.9263** [0.844, 0.970] | 0.9086 [0.821, 0.956] |
| lstm_L30 | 0.9388 [0.860, 0.977] | 0.9183 [0.833, 0.964] | 0.8936 [0.790, 0.949] |
| lstm_L7 | 0.9309 [0.836, 0.972] | 0.9164 [0.824, 0.962] | 0.8871 [0.793, 0.941] |
| lstm_L14 | 0.9311 [0.839, 0.971] | 0.9098 [0.808, 0.960] | 0.8823 [0.770, 0.945] |
| *xgboost* (P20) | *0.9681* | *0.9549* | *0.9408* |
| *logreg_linear* (P16) | *0.9401* | *0.9241* | *0.9053* |
| *persistence* (D7, P16) | *0.9203* | *0.8954* | *0.8683* |

Macro (per-station) PR-AUC is lower and spread by window, e.g. lstm_L3 0.8523/0.8387/0.7993,
gru_bestL 0.8704/0.8508/0.8125 — the GRU degrades most gently across horizons at the station level.

### Skill over the D7 persistence bar (PR-AUC Δ, val_2022)

| model | Δ24h | Δ48h | Δ72h |
| --- | ---: | ---: | ---: |
| lstm_L3 | **+0.0218** | **+0.0309** | **+0.0446** |
| gru_bestL | +0.0157 | +0.0309 | +0.0403 |
| lstm_L30 | +0.0185 | +0.0229 | +0.0253 |

Every recurrent family clears persistence at all three horizons, and all sit **≈0.02–0.03 below
the Phase-20 xgboost** bar (−0.0261 / −0.0286 / −0.0279 for the best DL model at
24/48/72 h). This is the expected outcome: on this tabular+sequence problem the boosted
tree on static features remains the strongest single model; the DL ladder's job is to close
that gap under fusion (Phase 23) and in the fair comparison (Phase 27).

### Cross-cell robustness (mean PR-AUC, season = all)

| | P3 spatial CV (4 folds) 24/48/72 h | P2 rolling val_2020/21/22 (24 h) |
| --- | --- | --- |
| lstm_L3 | 0.9196 / 0.9055 / 0.8781 | 0.9659 / 0.9575 / 0.9421 |
| lstm_L30 | **0.9245** / 0.9035 / **0.8940** | **0.9683** / **0.9668** / 0.9388 |
| lstm_L14 | 0.9244 / **0.9120** / 0.8877 | 0.9566 / 0.9545 / 0.9311 |
| gru_bestL | 0.9139 / 0.8972 / 0.8855 | 0.9655 / 0.9506 / 0.9359 |
| lstm_L7 | 0.9126 / 0.9117 / 0.8881 | 0.9520 / 0.9519 / 0.9309 |

Ranking shifts with the cell: **lstm_L3** leads on the primary selection cell (val_2022),
**lstm_L30** leads on P3/CV and the P2 rolling cells at 24 h — i.e. the window choice trades a
little in-distribution accuracy (L3) for out-of-sample stability (L30). No family collapses in
any cell; all are within ~1 point of each other, so the selection is genuinely cell-dependent
and is recorded as such rather than over-claimed.

## 4. Model artifacts

`models/phase21_L3-0{1..5}.pt` — the **P1 `train_<2022`** state per horizon
(`{"state": {horizon: state_dict}, "kind", "L", "params", "headline_metrics"}`), the models
Phase 33 will freeze:

| file | size | contents | n_params |
| --- | ---: | --- | ---: |
| `phase21_L3-01.pt` | 180 KB | **lstm_L3** × 3 horizons | 14,497 |
| `phase21_L3-02.pt` | 638 KB | lstm_L7 × 3 | 53,569 |
| `phase21_L3-03.pt` | 180 KB | lstm_L14 × 3 | 14,497 |
| `phase21_L3-04.pt` | 180 KB | lstm_L30 × 3 | 14,497 |
| `phase21_L3-05.pt` | 481 KB | **gru_bestL** × 3 | 40,193 |

Parameter counts are part of the comparison (Phase 27). The winning LSTM/GRU are ~14 K / 40 K
parameters versus xgboost's 300 trees — the DL models are *smaller* but, on this data, weaker
per-model.

## 5. Metric coverage

`phase21_metrics.csv`: **360 rows** = 5 families × **8 dev tasks** × 3 horizons × 3 seasons.
Cells present: `val_2020/2021/2022` (P2 rolling) + `spatial_fold_1…4` (P3) + the P1
`val_2022`; `test` rows = **0**. Columns match Phase 16 (plus `registry_id`, `feature_set`,
`weighting`, `config_id`, `search_score`, `n_params`), with station-bootstrap CIs and macro
PR-AUC on the `season=all` cells. Season cells (monsoon vs non-monsoon) move ≤ ~1 point, so
no season-specific degradation is hidden by the `all` cells.

## 6. Test policy

`test_policy = frozen_read_at_phase33` (registry, all L3 rows). The driver evaluates only the
8 dev tasks; `phase21_metrics.csv` contains **zero rows whose cell starts with `test`**. The
frozen test period is read exactly once, at Phase 33.

## 7. Determinism, reproducibility & cost

* **Determinism check (workflow-mandated):** `notebooks/21_temporal_dl.ipynb` refits a fixed
  config for 2 epochs twice on identical 4,000-row slices and asserts the parameter tensors
  are **bit-identical** and the val PR-AUCs agree to `<1e-12` (cell N21-08b — see notebook log).
* **Reproducibility:** seed 20261008; the run is resumable per family (checkpoint after each
  family; an earlier shell-timeout interruption was recovered by the resume logic).
* **Cost:** final `TOTAL 40625 s — exit=0`. Wall-clock was badly inflated by host sleep
  (single fits logged at 3.3 K–20.5 K s while CPU-active), so it is **not** a fair timing
  figure; the per-config `fit_s` column in `phase21_search.csv` (Σ = 1.1/1.5/2.9/12.0/1.1 K s
  per family) is the honest tuning cost, and GRU at L=3 is the cheapest family per epoch.

**DoD:** LSTM and GRU complete across 3 horizons × primary cells, dev grid + seasons reported,
budgets recorded and ≤ 20, tuning validation-only, imbalance decided on validation, models
saved, **determinism check green**, test untouched. **Next → Phase 22 / stage 37:** TCN on the
same data, splits, metrics and budget, for the recurrence-vs-convolution comparison.
