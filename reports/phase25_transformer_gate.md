# Phase 25 gate — is a temporal Transformer justified? (stage 40, D13)

| | |
| --- | --- |
| **Date** | 2026-10-13 |
| **Script** | `scripts/phase25_gate.py` → `reports/phase25_transformer_gate.log` (exit 0), `phase25_transformer_gate.json`, `figures/phase25_gate.png` |
| **Inputs** | `reports/phase17_experiment_registry.csv` · `phase21_search.csv` / `phase21_metrics.csv` (LSTM/GRU) · `phase22_search.csv` / `phase22_metrics.csv` (TCN) · `phase23_metrics.csv` (fusion) · `phase20_metrics.csv` (xgboost) · `phase16_baseline_metrics.csv` (D7) · `data/processed/phase21_lstar.json` |
| **Checks** | **G1–G5 PASS** (registry contract; baseline artefacts; budgets; window sweep; measured bars) |
| **Verdict** | **PASS (narrow, pre-registered).** The baselines are solid and real; the Transformer earns **one** registered attempt under pre-registered acceptance/kill criteria. The window-length evidence (best at L\* = 3) sets a **low prior** on a long-range attention gain — a tie-or-loss is a legitimate, reportable outcome. |

D13: the Transformer (Phase 25) runs only if this gate passes; a documented negative result
is acceptable. This report is 25.1; 25.2 (the model) runs only on a pass. No model was trained
here and the frozen test was not read.

---

## 1. The gate question

The workflow's Phase-25 gate is a **precondition + pre-registration** gate, not an empirical
one: *"run only after Phases 21–22 give a solid temporal baseline and Phase 23 shows whether
static context helps — a Transformer must be compared against something real."* The gate must
therefore confirm (a) the baselines exist and are solid, (b) the comparison is well-defined,
and (c) record the acceptance/kill rules **before** any Transformer is fit.

## 2. Preconditions (all met)

| Requirement | Evidence |
| --- | --- |
| Solid temporal baseline | Phase 21 LSTM/GRU (`phase21_metrics.csv`) and Phase 22 TCN (`phase22_metrics.csv`) complete, dev grid + budgets recorded |
| Static-context question answered | Phase 23 fusion (`phase23_metrics.csv`, L3-07) — static geography adds only marginal skill |
| GNN question answered | Phase 24 (`phase24_gnn.md`) — graph helps at 48/72 h on unseen stations, qualified positive |
| Registry contract | `L4-03 transformer_temporal`, gate `phase25_transformer_gate`, **budget 20 (D14)**, selection `val_2022`/`pr_auc`, test `frozen_read_at_phase33` |

## 3. Window-length sweep (the core evidence)

Phase 21 swept the LSTM window over L ∈ {3,7,14,30} — exactly the range where attention is
supposed to pay off ("30 days is where attention should matter"). Winner mean PR-AUC on
`val_2022` (season = all):

| L | 3 | 7 | 14 | 30 |
| --- | ---: | ---: | ---: | ---: |
| LSTM best mean PR-AUC | **0.92677** | 0.91146 | 0.90775 | 0.91688 |

**L\* = 3.** Performance is highest at the **shortest** window and does *not* increase with L
(L30 = 0.9169 ≤ L\* = 0.9268; L7/L14 are lower still). The Bangladeshi flood signal, at least
as captured here, is **short-range** relative to the anchor day. This does not forbid a
Transformer, but it means the mechanism it is sold on (long-range dependency) has little
headroom: the fair test is whether self-attention can match the short-window baseline more
efficiently, or win on unseen-station / rare-event behaviour.

## 4. The bars the Transformer must beat (`val_2022` and P3 spatial CV, season = all)

| model | val_2022 (mean PR-AUC) | P3 mean |
| --- | ---: | ---: |
| D7 persistence | 0.89465 | 0.91771 |
| tcn_bestL (L3-06) | 0.92124 | 0.89744 |
| lstm_L3 (L3-01) | 0.92677 | 0.90109 |
| gru_bestL (L3-05) | 0.92362 | 0.89885 |
| fusion_L3 (L3-07) | 0.93559 | 0.91757 |
| **xgboost (Phase 20)** | **0.95463** | **0.96225** |

`figures/phase25_gate.png` (left: the sweep; right: these bars). Note P3 is a *hard* cell: the
per-station temporal models (≈0.90) fall **below** persistence (0.9177) when stations are held
out — the regime Phase 24's graph attacked. xgboost is the standing ceiling on both.

## 5. Pre-registered acceptance / kill criteria

1. **Primary.** The temporal Transformer (L4-03), budget ≤ 20, tuned on `val_2022`
   (season = all, mean PR-AUC over 24/48/72 h), must beat the best temporal baseline
   (TCN L3-06 at 0.92124, and ideally the LSTM L3-01 at 0.92677) on `val_2022`.
2. **Secondary.** On **P3 spatial CV (unseen stations)** it must not lose to the temporal
   baselines; a **paired station-bootstrap CI** on the difference is reported on identical cells.
3. **Kill.** If there is no meaningful gain (CI spanning 0) over the temporal baselines, **stage
   40 closes as a documented rejection** and **no spatiotemporal Transformer is attempted**.
4. **Expectation.** The window sweep predicts the long-range advantage is unlikely to
   materialise; a tie-or-loss is an acceptable, reportable outcome (D13).

## 6. Registry & handoff

* Gate `phase25_transformer_gate` (attached to L4-03 in `phase17_experiment_registry.csv`) is
  **discharged by this report**. The framework still marks L4-03 `gated`; 25.2 will amend it
  `gated → planned` and re-run `scripts/phase17_framework.py` (its F1m check is updated then).
  L4-01/02/04 are already `planned` (Phase 24). No new data artifact is required: the
  Transformer reuses Dataset B sequences (`sequences_L{7,14,30}.npy`, built by Phase 18) and
  L\* = 3 from Phase 21.
* Windows: the workflow names {7,14,30}; L\* = 3 is retained as the parity point. The window is
  a tuning dimension inside the budget-20 search ("L = best").

**Decision: GATE PASSED (narrow) — proceed to 25.2 under the criteria above.**
