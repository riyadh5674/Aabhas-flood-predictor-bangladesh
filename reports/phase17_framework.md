# Phase 17 — Experimental modelling framework

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 17 — Experimental modelling framework** → body step 31 / checklist **stage 31** |
| **Script** | `scripts/phase17_framework.py` — builds and validates the experiment registry |
| **Notebook** | `notebooks/17_framework_registry.ipynb` — independent re-verification, **29/29 green, 0.9 s** (built by `scripts/build_phase17_notebook.py`) |
| **Run** | `python -u scripts\phase17_framework.py` → `scripts/phase17_run.log`, **exit 0, 0.4 s**, seed 20261008 |
| **Outputs** | `reports/phase17_experiment_registry.csv` (24 experiments × 22 columns; 21 at Phase 17 + 3 Phase-23 fusion rows, §2 amendment) |
| **Checks** | **20/20 green** (F1a–F1t) in the script + **29/29 green** (N17-01…N17-14) in the notebook, 0 warnings |
| **Untouched** | `experimental_dataset.csv` SHA-256 `a67485…` re-verified (F1a); no model trained, no test cell read |

Phase 17 fixes *what is being compared* before anything is trained (workflow Phase 17;
the anti-padding rule: **no algorithm is added just to say "we tested 10 models"**).

---

## 1. The three tasks (D2 — no privileged horizon)

Three separate forecasting tasks, three separate models each:

```
X_t → P(Y_{t+24} = 1)      X_t → P(Y_{t+48} = 1)      X_t → P(Y_{t+72} = 1)
```

All three horizons are trained, tuned and reported for every experiment; a result at 24 h
says nothing about 72 h (F1r). Issue instant, labels and information boundary are the frozen
Phase-0 contract — nothing in this phase may change them.

## 2. Four families, 21 registered experiments

Every row of the registry carries a written reason for existing (F1e: shortest reason
74 chars). Full detail in `phase17_experiment_registry.csv`.

### L1 — simple statistical (Phase 16, frozen, 6 runs)

| id | model | input | reason |
| --- | --- | --- | --- |
| L1-01 | `climatology` | train base rate | prevalence reference — PR-AUC of any model is judged against it |
| L1-02 | `persistence` | `wl_pct_danger` (1) | **D7 bar (D7)** — the benchmark every family must clear |
| L1-03 | `logreg_linear` | LINEAR (35) | honest linear ceiling for engineered features |
| L1-04 | `logreg_linear_no_up` | LINEAR − `up_*` | F14-2 ablation |
| L1-05 | `logreg_selected` | SELECTED (10) | does feature reduction hurt linear models |
| L1-06 | `tree_depth4` | FULL (62) | interpretable non-linearity without tuning |

All six are `status=done`; their single frozen-test read already happened in Phase 16
(`test_policy = read_once_phase16`, F1h).

### L2 — classical ML (Phase 20, 5 runs)

| id | model | input | reason | budget |
| --- | --- | --- | --- | ---: |
| L2-01 | `random_forest` | FULL | interactions, no scaling assumptions | 40 configs |
| L2-02 | `gradient_boosting` | FULL | classic boosting, no external dependency | 40 |
| L2-03 | `xgboost` | FULL | **the tabular workhorse** — strongest realistic classical competitor | 40 |
| L2-04 | `lightgbm` | FULL | leaf-wise boosting, native NaN handling, cheap capacity | 40 |
| L2-05 | `xgboost_selected` | SELECTED | compactness test: can 10 features match 62 | 40 |

xgboost and lightgbm are already installed in `.venv`; `gradient_boosting` (sklearn) covers
the case where either cannot be pinned reproducibly.

### L3 — deep temporal (Phases 21–22, Dataset B, 6 runs)

Window length **L ∈ {3, 7, 14, 30}** is an empirical question (D12), not a default:

| id | model | window | reason |
| --- | --- | --- | --- |
| L3-01 | `lstm_L3` | 3 d | shortest credible history — does recurrence help at all |
| L3-02 | `lstm_L7` | 7 d | like-for-like memory with the engineered antecedent features |
| L3-03 | `lstm_L14` | 14 d | longer memory than the engineered features (D12) |
| L3-04 | `lstm_L30` | 30 d | long-memory end of the sweep; where attention should matter |
| L3-05 | `gru_bestL` | best L | fewer parameters — cheaper recurrence, same skill? |
| L3-06 | `tcn_bestL` | best L | non-recurrent encoder — the LSTM/GRU/TCN three-way test |
| L3-07 | `fusion_L3` | 3 d + static | **Phase-23 amendment** — temporal encoder fused with the static geographic vector: does knowing *where* add skill over the sequence alone |
| L3-08 | `static_only` | — (static) | **Phase-23 amendment** — the static branch alone (feature-group baseline; no sequence) |
| L3-09 | `hybrid_L3` | 3 d + static + LINEAR | **Phase-23 amendment** — does the engineered LINEAR signal add to what the encoder learned |

Budget: 20 configs each (the three Phase-23 fusion rows declared a budget of 16, still
within the L3 cap); input = the 12 raw/near-raw variables per day (Dataset B, Phase 18),
never the engineered rolling features — that separation is what makes the **E13 experiment
(raw sequence → DL vs engineered features → XGBoost)** interpretable. (`hybrid_L3` adds the
engineered LINEAR set explicitly, as a distinct registered candidate.)

> **Registry amendment (Phase 23 / stage 38).** The registry is the contract, so the fusion
> variants were added here (rows `L3-07`, `L3-08`, `L3-09`, family *feature fusion*,
> `metrics_out=phase23_metrics.csv`) rather than run ad hoc; the framework was re-run and is
> **20/20 green** with **24** experiments (L1=6 / L2=5 / L3=9 / L4=4). `static_only` carries
> Dataset label **S** (static, non-sequential). Ledger references (workflow §1) were updated
> 21 → 24.

> **Registry amendment (Phases 24–25 / stages 39–40).** Both gates passed — Phase 24
> (`phase24_graph_gate`, 24.1) and Phase 25 (`phase25_transformer_gate`, 25.1) — so all four
> L4 rows were discharged **gated → planned** (L4-03 included, with its `model_out` corrected
> to `models/phase25_<id>.pt`); the framework was re-run and is **20/20 green**.

### L4 — advanced spatiotemporal (Phases 24–25, 4 runs, **gated → discharged — D13**)

| id | model | gate |
| --- | --- | --- |
| L4-01 | `gcn` | `phase24_graph_gate` |
| L4-02 | `gat` | `phase24_graph_gate` |
| L4-03 | `transformer_temporal` | `phase25_transformer_gate` |
| L4-04 | `gnn_lstm` | `phase24_graph_gate` |

`status=planned` for all four (F1m): both gates are **discharged**, so every L4 row is now
runnable. The D13 rule is unchanged — each row was `gated` until its gate report existed, and
a documented rejection is still a legitimate way to close a stage.

## 3. The shared experiment grid (D14 — fair comparison)

Identical for every planned run (F1i), so no ML-vs-DL claim can rest on a protocol
difference:

| axis | value |
| --- | --- |
| protocols | **P1** temporal primary · **P2** rolling origin (train ≤2020/21/21…, eval per year) · **P3** spatial CV (4 station-disjoint folds) |
| dev cells | `val_2022` (P1) · `val_2020`, `val_2021`, `val_2022` (P2) · `spatial_fold_1..4` (P3) — all confirmed constructible from `split_assignments.csv` (F1c) |
| seasons | all · monsoon · non-monsoon (D1) |
| horizons | 24 h · 48 h · 72 h |
| selection | **PR-AUC on `val_2022`** (primary), spatial evidence co-required for any unseen-station claim |
| seed | 20261008 |
| cells per model | 8 dev tasks × 3 seasons × 3 horizons = **72** (identical to Phase 16's 432-row file ÷ 6 models, F1q) |
| imbalance | `none` or `class_weight=balanced`, **inside training only** (F1p) |
| thresholds | train-derived (Phase 16 S6), period-specific (Phase 16 §10) |

## 4. Test protection

* Development grids contain **no test cell** and no test selection cell (F1j).
* `test_policy` column: L1 = `read_once_phase16` (already recorded); every planned
  experiment = `frozen_read_at_phase33` — the seven frozen cells
  (`test_2023…2026`, `trained`, `new_network`, `unseen`) exist (F1k) and are read
  **once, after Phase 33 freezes a model**, on top of the D7 bar.

## 5. Checks (20/20)

| id | assertion |
| --- | --- |
| F1a | experimental table SHA unchanged |
| F1b | split table 200,665 × 9; feature sets 62/35/10 |
| F1c | Phase-15 masks support all 8 dev tasks + 7 frozen test cells |
| F1d | registry = 24 experiments, L1=6 / L2=5 / L3=9 / L4=4 (L3-07/08/09 = Phase-23 fusion amendment) |
| F1e | every candidate has a stated reason (≥ 74 chars) |
| F1f | registry ids and model names unique |
| F1g | L1 model set == Phase-16 metric file's six models |
| F1h | L1 rows done + test policy `read_once_phase16` |
| F1i | parity (D14): same protocols, grid, seasons, horizons, selection, seed |
| F1j | no planned run sees a test cell before Phase 33 |
| F1k | the seven frozen test cells are constructible |
| F1l | tuning budgets recorded: L1 0, L2 40, L3 temporal 20 / L3 fusion 16, L4 20 |
| F1m | L4 gate reports named and both discharged (gated → planned); no row left gated |
| F1n | every Dataset-A feature set resolves to a Phase-12 set |
| F1o | every Dataset-B row names raw sequence variables + window/graph |
| F1p | imbalance handling inside-training only (R4) |
| F1q | 72 dev cells/model = Phase-16 grid |
| F1r | three separate horizons per experiment (D2) |
| F1s | D7 benchmark row present |
| F1t | registry CSV written (10,552 bytes, 24 rows × 22 cols) |

## 6. Consequences recorded here

1. **Phase 16's open `up_*` question is answered structurally**: the ablation lives on as
   registry row L1-04 (already measured — no effect) and as an E-matrix row; no candidate
   is built on `up_*`-only assumptions.
2. **Budgets are fixed before any result exists** (L2 40, L3/L4 20 configs) — tuning cannot
   silently expand after a model looks promising (D14).
3. **Gates are pre-committed**: if Phase 24's graph gate fails, rows L4-01/02/04 are
   *rejected with evidence*, not quietly dropped.

---

**Next: Phase 18 — ML/DL dataset preparation (stage 32)** — build Dataset B (raw sequence
windows for L = 3/7/14/30) beside the immutable Dataset A, with a manifest and hashes.
Then Phase 19 sequence validation before any LSTM/GRU/TCN is trained.
