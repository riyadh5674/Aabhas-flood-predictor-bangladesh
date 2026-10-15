# Phase 17 — The Experimental Modelling Framework (the "rulebook")
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase17_framework.md`, `scripts/phase17_framework.py`,
> `reports/phase17_experiment_registry.csv`, `notebooks/17_framework_registry.ipynb`.
> Previous study file: Phase 16 (baselines). Simple English, real numbers, what/why/how.

---

## 0. The 60-second version

Before training dozens of models, we write down **exactly what will be compared, how, and
with what budget** — *before* seeing any result. This is the project's "rulebook" so that no
later claim can be blamed on an unfair comparison or a model that quietly got more tuning.

Phase 17 produced an **experiment registry**: **24 experiments** in **4 families**:

| Family | Meaning | Count |
|---|---|---|
| **L1** | simple statistical (baselines, frozen from Phase 16) | 6 |
| **L2** | classical ML (RandomForest, GradientBoosting, XGBoost, LightGBM) | 5 |
| **L3** | deep temporal (LSTM/GRU/TCN + fusion variants) | 9 |
| **L4** | advanced spatiotemporal (GCN/GAT/Transformer/GNN-LSTM, **gated**) | 4 |

Every experiment has a **written reason** to exist (anti-padding rule: *never add a model just
to say "we tested 10 models"*), a **fixed tuning budget**, and the **same evaluation grid**.

Runs fast: **exit 0, 0.4 s**, **20/20 checks green** in the script, **29/29** in the notebook.

---

## 1. Where Phase 17 sits

| Phase | What | Plain words |
|---|---|---|
| 16 | Baselines + imbalance | "How good is simple?" |
| **➜ 17** | **Framework / registry** | **Write the rulebook before the race** |
| 18 | Dataset B (sequences) | Prepare data for deep learning |
| 19 | Sequence validation | Check the sequence data is correct |
| 20–25 | Run the candidate families | Actually train L2/L3/L4 |
| 26–33 | Calibrate, thresholds, select, freeze | Pick the champion |

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Registry** | A table listing every experiment: id, model, input, reason, budget. |
| **Family (L1–L4)** | A group of related models (simple → classical → deep → advanced). |
| **Anti-padding rule** | Don't register models you don't need. Every row needs a reason. |
| **Tuning budget** | Max number of configurations a model may try. Fixed *before* results. |
| **Protocol P1/P2/P3** | P1 temporal primary, P2 rolling origin, P3 spatial CV. |
| **Dev cell** | A validation cell used for development (never test). 8 per model. |
| **Cell** | One (split × season × horizon) combination, e.g. val_2022 × monsoon × 24h. |
| **Gate** | A pre-committed test: only run these models if the gate passes. |
| **Admission / discharge** | A gated row becomes runnable ("planned") once its gate passes. |
| **D2** | Decision: no horizon is privileged — always train/report 24, 48, 72 h. |
| **D10** | Ordering rule: build candidates in the planned order. |
| **D13** | Gate rule: advanced models only run if their gate is discharged. |
| **D14** | Parity rule: identical grid/protocol/budget for every planned run. |
| **F1a–F1t** | The 20 framework checks (assertions that must all pass). |
| **test_policy** | When a model may read the frozen test (`frozen_read_at_phase33`). |

---

## 3. The three tasks (D2 — no privileged horizon)

Three **separate** forecasting tasks, three **separate** models each:

```
X_t → P(flood at t+24h)      X_t → P(flood at t+48h)      X_t → P(flood at t+72h)
```

Why separate? Because a model that's good at 24 h is **not automatically** good at 72 h.
A 72 h forecast is genuinely harder (more time for conditions to change). So we never assume
one result carries to another horizon (check F1r).

---

## 4. The four families in detail

### L1 — Simple statistical (frozen from Phase 16, 6 runs)

| id | model | input | reason |
|---|---|---|---|
| L1-01 | `climatology` | train base rate | prevalence reference |
| L1-02 | `persistence` | `wl_pct_danger` (1) | **the D7 bar** every family must clear |
| L1-03 | `logreg_linear` | LINEAR (35) | honest linear ceiling |
| L1-04 | `logreg_linear_no_up` | LINEAR − `up_*` | F14-2 ablation |
| L1-05 | `logreg_selected` | SELECTED (10) | does feature reduction hurt? |
| L1-06 | `tree_depth4` | FULL (62) | interpretable non-linearity, no tuning |

All six are **done**; their single frozen-test read already happened in Phase 16
(`test_policy = read_once_phase16`).

### L2 — Classical ML (Phase 20, 5 runs, budget 40 configs each)

| id | model | input | reason |
|---|---|---|---|
| L2-01 | `random_forest` | FULL | interactions, no scaling assumptions |
| L2-02 | `gradient_boosting` | FULL | classic boosting, no external dependency |
| L2-03 | `xgboost` | FULL | **the tabular workhorse** — strongest realistic competitor |
| L2-04 | `lightgbm` | FULL | leaf-wise boosting, native NaN handling |
| L2-05 | `xgboost_selected` | SELECTED | can 10 features match 62? |

**Note:** `xgboost` (L2-03) is the model the frozen Phase-33 solution uses. Gradient boosting
covers the case where xgb/lightgbm can't be pinned reproducibly.

### L3 — Deep temporal (Phases 21–22, Dataset B, 9 runs, budget 20 configs)

Window length **L ∈ {3, 7, 14, 30} days** is treated as an **empirical question** (D12), not a
default — we sweep it:

| id | model | window | reason |
|---|---|---|---|
| L3-01 | `lstm_L3` | 3 d | shortest credible history |
| L3-02 | `lstm_L7` | 7 d | matches engineered antecedent features |
| L3-03 | `lstm_L14` | 14 d | longer memory than engineered |
| L3-04 | `lstm_L30` | 30 d | long-memory end; where attention should matter |
| L3-05 | `gru_bestL` | best L | fewer parameters — same skill? |
| L3-06 | `tcn_bestL` | best L | non-recurrent encoder (the three-way test) |
| L3-07 | `fusion_L3` | 3 d + static | does *where* add skill over sequence alone? |
| L3-08 | `static_only` | static only | static branch alone (no sequence) |
| L3-09 | `hybrid_L3` | 3 d + static + LINEAR | does engineered signal add to the encoder? |

**Key design:** L3 input = **12 raw/near-raw variables per day** (Dataset B), *never* the
engineered rolling features. That separation is what makes the **E13 experiment**
(*raw sequence → DL* vs *engineered features → XGBoost*) interpretable.

**Amendment (Phase 23):** rows L3-07/08/09 were added to the registry rather than run ad hoc —
because the registry **is the contract**. Re-run: still **20/20 green**, now 24 experiments.

### L4 — Advanced spatiotemporal (Phases 24–25, 4 runs, **gated**)

| id | model | gate |
|---|---|---|
| L4-01 | `gcn` | `phase24_graph_gate` |
| L4-02 | `gat` | `phase24_graph_gate` |
| L4-03 | `transformer_temporal` | `phase25_transformer_gate` |
| L4-04 | `gnn_lstm` | `phase24_graph_gate` |

**The gate idea (D13):** these are expensive/risky models. So each starts as `gated` and only
becomes `planned` (runnable) if its **pre-committed gate report** passes. Both gates *did*
pass (Phase 24 graph gate, Phase 25 transformer gate), so all four are now `planned`.

**Crucial:** a gate failing is a **legitimate way to close a stage** — you reject with
evidence, you don't quietly drop.

---

## 5. The shared grid (D14 — fairness)

Every planned run uses this **identical** grid, so no ML-vs-DL claim can rest on a protocol
difference:

| Axis | Value |
|---|---|
| Protocols | **P1** temporal · **P2** rolling origin (train ≤2020/21/21…; eval per year) · **P3** spatial CV (4 folds) |
| Dev cells | `val_2022` (P1) · `val_2020/2021/2022` (P2) · `spatial_fold_1..4` (P3) |
| Seasons | all · monsoon · non-monsoon (D1) |
| Horizons | 24 h · 48 h · 72 h |
| Selection | **PR-AUC on `val_2022`** (primary); spatial evidence co-required for unseen-station claims |
| Seed | **20261008** |
| Cells per model | 8 dev × 3 seasons × 3 horizons = **72** |
| Imbalance | `none` or `class_weight=balanced`, **inside training only** (R4) |
| Thresholds | train-derived, period-specific |

**Why 72 cells?** It matches Phase 16's grid exactly (432 rows ÷ 6 models) — check F1q.

---

## 6. Test protection (the most important rule)

- **No development grid contains a test cell.** The test set is invisible during development.
- `test_policy` column: L1 = `read_once_phase16` (already recorded). Every planned experiment
  = **`frozen_read_at_phase33`**: the **seven frozen cells** (`test_2023…2026`, `trained`,
  `new_network`, `unseen`) are read **once, after Phase 33 freezes a model**.

**Plain point:** the test set is opened **one time, at the very end**, against the D7 bar.
Anything else would be cheating.

---

## 7. The 20 checks (20/20 green)

A sample of what's asserted (`F1a`–`F1t`):

| id | assertion |
|---|---|
| F1a | experimental table SHA unchanged (no accidental edits) |
| F1c | Phase-15 masks support all 8 dev tasks + 7 frozen test cells |
| F1d | registry = 24 experiments (L1=6 / L2=5 / L3=9 / L4=4) |
| F1e | every candidate has a stated reason (≥ 74 chars) |
| F1i | parity (D14): same protocols, grid, seasons, horizons, seed |
| F1j | **no planned run sees a test cell before Phase 33** |
| F1l | tuning budgets recorded (L1 0, L2 40, L3 temp 20 / fusion 16, L4 20) |
| F1m | L4 gates named and discharged; no row left gated |
| F1p | imbalance handling inside-training only |
| F1q | 72 dev cells/model = Phase-16 grid |
| F1r | three separate horizons per experiment (D2) |
| F1s | D7 benchmark row present |

The notebook independently re-verifies with **29/29** checks (N17-01…N17-14).

---

## 8. Why this phase matters (the consequences)

1. **Phase 16's `up_*` question is answered structurally** — the ablation lives on as registry
   row L1-04 (already measured: no effect); no candidate is built on `up_*` assumptions.
2. **Budgets are fixed before any result** (L2 40, L3/L4 20 configs) — tuning can't silently
   expand after a model looks good (D14). This is *anti-p-hacking*.
3. **Gates are pre-committed** — if the graph gate had failed, L4-01/02/04 would be *rejected
   with evidence*, not quietly dropped.

**Plain point:** by writing the rulebook first, the project makes it very hard to fool itself.

---

## 9. Problems & solutions

| Problem | Solution |
|---|---|
| Risk of an unfair ML-vs-DL comparison | One shared grid + parity checks (D14, F1i). |
| Risk of "model shopping" / tuning until it looks good | Fixed budgets before results (D14, F1l). |
| Risk of adding models just for show | Anti-padding rule: every row needs a written reason (F1e). |
| Expensive/risky deep models might waste time | Gates (D13): run only if a pre-committed gate passes. |
| Test set creeping into decisions | No dev cell contains test; `frozen_read_at_phase33` (F1j). |
| Amendments could bypass the rulebook | Amendments (Phase 23 fusion) added **to the registry**, then re-verified. |
| One horizon assumed to imply others | Three separate tasks/horizons (D2, F1r). |

---

## 10. Files referenced

| File | Purpose |
|---|---|
| `reports/phase17_framework.md` | Full Phase 17 report |
| `scripts/phase17_framework.py` | Builds/validates the registry (exit 0, 0.4 s) |
| `reports/phase17_experiment_registry.csv` | 24 experiments × 22 columns |
| `notebooks/17_framework_registry.ipynb` | Independent verification (29/29) |
| `reports/phase15_split_design.md` | Dev/frozen cells come from here |
| `reports/phase16_baseline.md` | L1 results + D7 bar |
| `final workflow.md` | The phase plan (source of truth) |

---

## 11. Verdict

- **The rulebook is frozen:** 24 experiments, 4 families, each with a reason and a fixed budget.
- **Fair comparison is guaranteed** by one shared grid + parity checks (D14).
- **Advanced models are gated** (D13) — both gates passed, so L4 is runnable.
- **The test set is protected** — opened once, after Phase 33 (`frozen_read_at_phase33`).
- **All checks pass** (20/20 script, 29/29 notebook).

**Bottom line:** We didn't just start training models — we first agreed on *what fairness means*
and *what the rules are*, so that later "this model is best" claims can be trusted.

---

**Next in the series:** Phase 18 — ML/DL dataset preparation (build Dataset B: raw sequence
windows for L = 3/7/14/30 beside the immutable Dataset A). Then Phase 19 sequence validation,
then Phases 20–25 run the families.
