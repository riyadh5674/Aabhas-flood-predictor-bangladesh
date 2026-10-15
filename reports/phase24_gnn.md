# Phase 24 — Station graph & spatiotemporal GNN (workflow stage 39, D13)

| | |
| --- | --- |
| **Date** | 2026-10-13 |
| **Gate (24.1)** | `scripts/phase24_gate.py` → `reports/phase24_graph_gate.md` — **PASS (narrow)** |
| **Modules** | `scripts/train_gnn.py` (gcn / gat / gnn_lstm) · `scripts/phase24_ablation.py` (paired P3 kill-criterion) · `scripts/build_phase24_notebook.py` |
| **Inputs** | Dataset B `sequences_L3.npy` + mask (P19-M1) · `data/processed/station_graph.csv` + `station_graph_manifest.json` · `reports/phase21_metrics.csv` (L3-01 temporal-only) · `reports/phase20_metrics.csv`, `reports/phase16_baseline_metrics.csv` |
| **Outputs** | `reports/phase24_metrics.csv` (216) · `reports/phase24_search.csv` (24) · `reports/phase24_ablation.csv` · `reports/phase24_ablation_preds.npz` · `models/phase24_L4-0{1,2,4}.pt` · `figures/phase24_val2022.png` · `notebooks/24_gnn.ipynb` |
| **Verdict** | **Qualified positive (horizon-dependent)** — P3 unseen-station paired CI: no gain at 24 h (GAT Δ −0.013, CI < 0), **significant gain at 48 h (GAT +0.016) and 72 h (GCN +0.018, GAT +0.023)** · test never read (frozen to Phase 33) |
| **Governing doc** | `final workflow.md` Phase 24 / stage 39; D13 (GNN runs only on a gate pass; a documented rejection is acceptable) |

No file under `data/raw/` was modified. `experimental_dataset.csv` SHA-256 is unchanged
(`a6748529083f…73c8e`); the sequence-index labels were re-derived 1:1 against it.

---

## 1. The question

The gate (§7 of `phase24_graph_gate.md`) showed the neighbour signal is real but largely
**redundant with same-station persistence on P1**. The only protocol where a graph model has a
hypothesis the per-node model cannot have is **P3 spatial CV — held-out stations**, where
same-station persistence does not transfer. Phase 24 therefore asks exactly one registered
question: **does a defensible hydrology-first station graph let a spatiotemporal GNN beat the
temporal-only LSTM (L3-01) on unseen-station PR-AUC, with a paired bootstrap CI?** A documented
rejection is acceptable.

## 2. Design (all pre-registered in the gate)

**Graph.** `G = river_upstream (52 hydrological edges) ∪ knn5 (375 distance-fallback edges)`,
frozen as `data/processed/station_graph.csv` before any model saw it:
427 edges, connected (1 component, all 133 stations), mean degree 6.421,
SHA-256 `655ca5a4e048d880814d035e45122495a0858a9e716d7e3bc08c3d38311e8db7`. Distance is used
only to connect the 51 river-isolated gauges; the primary edges are hydrological.

**Models (node features = per-station sequence variables, temporal encoder as Phase 21, L\* = 3):**

| Registry | model | mechanism |
| --- | --- | --- |
| **L4-01** | `gcn` | graph convolution over the station graph — uniform message passing |
| **L4-02** | `gat` | graph attention — learns neighbour weights instead of assuming uniform |
| **L4-04** | `gnn_lstm` | GNN spatial encoder + LSTM temporal encoder (the full stack) |
| L3-01 | `lstm_L3` | temporal-only (Phase 21, reported, not re-trained) — the reference |

Same splits, masks, metric code, selection cell (`val_2022`), selection metric (PR-AUC), seed
(`20261008`), imbalance rule and budget (L4 = 20, D14) as every other family. Frozen 2023+ test
never read.

## 3. Results — fully-tuned grid (season = all, PR-AUC)

Winner configs (identical across families): `hidden` 32 · `enc_layers` 2 · `g_layers` 1 ·
`dropout` 0.3 · `lr` 3e-3 · `batch_dates` 8 · unweighted (`_weight` off). Budget 8 configs each
(≤ 20). Best epochs (24/48/72 h): gcn 14/14/14, gat 9/14/14, gnn_lstm 10/12/10.

**P1 `val_2022`** (the Phase-21/22/23 comparison cell):

| model | 24h | 48h | 72h | params |
| --- | ---: | ---: | ---: | ---: |
| **L4-01 gcn** | 0.9421 | 0.9353 | 0.9141 | 15,553 |
| **L4-02 gat** | 0.9408 | 0.9314 | 0.9136 | 15,617 |
| **L4-04 gnn_lstm** | **0.9424** | 0.9264 | **0.9153** | 18,433 |
| L3-01 lstm_L3 (temporal-only) | 0.9421 | 0.9253 | 0.9129 | 14,497 |
| *D7 persistence* | 0.9203 | 0.8954 | 0.8683 | — |
| *xgboost (Phase 20)* | 0.9681 | 0.9549 | 0.9408 | — |

**P2 rolling-origin mean (3 years) and P3 spatial-CV mean (4 folds):**

| protocol | model | 24h | 48h | 72h |
| --- | --- | ---: | ---: | ---: |
| P2 | gcn | 0.9565 | 0.9458 | 0.9292 |
| P2 | gat | 0.9436 | 0.9430 | 0.9276 |
| P2 | gnn_lstm | 0.9590 | 0.9433 | 0.9293 |
| P2 | L3-01 lstm_L3 | 0.9551 | 0.9362 | 0.9193 |
| **P3** | **gcn** | 0.9184 | 0.9120 | 0.8945 |
| **P3** | **gat** | 0.9177 | 0.9162 | 0.9011 |
| **P3** | **gnn_lstm** | 0.9272 | 0.9018 | 0.8862 |
| **P3** | **L3-01 lstm_L3** | 0.9196 | 0.9055 | 0.8781 |

Read from the grid: on P1 the graph families are **≈equal** to the temporal-only LSTM at 24 h
(0.942) and marginally ahead at 48/72 h; `gnn_lstm` is the best single 24/72 h number. On P3
(unseen stations) the picture is **mixed**: `gnn_lstm` leads at 24 h (+0.008 over L3-01),
`gat` leads clearly at 48 h (+0.011) and 72 h (+0.023), while `gcn` is ≈flat at 24/48 h and up
at 72 h (+0.016). All models remain well below the xgboost bar and above D7. The grid means
are point estimates; the paired CI in §4 is the registered decision.

## 4. Results — paired kill-criterion ablation (P3, GNN − LSTM L3-01)

Same P3 fold × horizon cells, identical batching/seed for the GNN winner and the L3-01 winner;
the LSTM is graph-independent so it is fit once per cell and reused. Difference PR-AUC per
sample under a **paired station-resampling bootstrap** (1,000 resamples of the held-out
stations). Full graph:

<!-- ABL_TABLE_START -->
Per fold × horizon (12 cells per family), paired station bootstrap, 1,000 resamples:

| family | horizon | GNN (mean) | LSTM L3-01 (mean) | Δ (mean) | cells with CI excluding 0 |
| --- | --- | ---: | ---: | ---: | --- |
| gcn | 24 h | 0.9184 | 0.9196 | **−0.0003** | 2/4 (both positive) |
| gcn | 48 h | 0.9120 | 0.9055 | **+0.0085** | 2/4 (both positive) |
| gcn | 72 h | 0.8945 | 0.8781 | **+0.0165** | 3/4 (all positive) |
| gat | 24 h | 0.9126 | 0.9196 | **−0.0069** | 2/4 (both negative) |
| gat | 48 h | 0.9169 | 0.9055 | **+0.0126** | 3/4 (all positive) |
| gat | 72 h | 0.8984 | 0.8781 | **+0.0191** | 3/4 (all positive) |

Pooled P3 CI per horizon (4 disjoint spatial folds concatenated — stations are held out exactly
once, so pooled station resampling is valid), 500 resamples:

| family | horizon | n | GNN | LSTM L3-01 | Δ | 95% CI | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gcn | 24 h | 61,977 | 0.9272 | 0.9297 | −0.0021 | [−0.0216, +0.0114] | no gain |
| gcn | 48 h | 60,992 | 0.9153 | 0.9068 | +0.0085 | [−0.0077, +0.0224] | no gain (direction +) |
| gcn | 72 h | 60,144 | 0.9010 | 0.8831 | **+0.0178** | [+0.0064, +0.0340] | **gain (CI > 0)** |
| gat | 24 h | 61,977 | 0.9166 | 0.9297 | **−0.0129** | [−0.0278, −0.0020] | **LSTM better (CI < 0)** |
| gat | 48 h | 60,992 | 0.9228 | 0.9068 | **+0.0157** | [+0.0081, +0.0247] | **gain (CI > 0)** |
| gat | 72 h | 60,144 | 0.9063 | 0.8831 | **+0.0228** | [+0.0026, +0.0563] | **gain (CI > 0)** |
<!-- ABL_TABLE_END -->

## 5. Verdict

**Qualified positive — horizon-dependent.** On P3 spatial CV (unseen stations), the graph
model does **not** help at 24 h (GCN ties; GAT is *significantly worse*, Δ −0.013, CI < 0), but
it **significantly beats the temporal-only LSTM at 48 h and 72 h**:

- **72 h** — the longest horizon, where hydrological propagation should dominate: GCN Δ **+0.018**
  (CI [+0.006, +0.034]) and GAT Δ **+0.023** (CI [+0.003, +0.056]), both excluding 0, with 3/4
  fold cells separable on each.
- **48 h** — GAT Δ **+0.016** (CI [+0.008, +0.025], significant); GCN Δ +0.008 (directionally
  positive, CI spans 0).

This is **not** the "documented rejection" branch of the gate — the CIs do exclude 0, in the
positive direction, at the horizons where the propagation hypothesis predicts them. The gain is
small in absolute terms (~+0.01–0.02 PR-AUC) and does not survive at 24 h, where a station's own
recent history already carries the signal — exactly the redundancy the gate flagged (§7 of
`phase24_graph_gate.md`). The two graph mechanisms behave consistently: attention (GAT) reaches
the **best** P3 numbers at 48/72 h but pays the **largest** 24 h penalty; the plain GCN is the
safer, flatter variant. Both remain below xgboost (P1) and above D7.

**Decision.** Stage 39 closes as a **qualified positive**: the hydrological station graph carries
transferable signal across *unseen stations* at 48–72 h. Phase 25 (Transformer, L4-03) may
therefore be run, but the evidence only supports a **temporal Transformer as the primary probe**;
a Transformer-on-graph is justified at 48/72 h but not at 24 h. The test set stays frozen to
Phase 33.

## 6. Registry amendment (why the phase touched a completed stage)

Phase 17's registry gates the L4 rows. The gate report §9 says 24.2 discharges
`phase24_graph_gate`: `build_registry()` in `scripts/phase17_framework.py` now marks
**L4-01/L4-02/L4-04 `planned`** (gate passed, run authorised) while **L4-03
`transformer_temporal` stays `gated`** by `phase25_transformer_gate`. The framework was
re-run: **20/20 PASS** (F1m updated to assert exactly this split), registry = 24 rows
(`done 6 · planned 17 · gated 1`). The amendment log is in `scripts/phase17_run.log`;
`build_phase17_notebook.py` N17-08 was updated to match.

## 7. Verification

| Check | Result |
| --- | --- |
| `scripts/train_gnn.py` (gcn, gat, gnn_lstm) | exit=0; dev grid 216 rows (72/family), 0 test cells |
| per-family `n_params` consistency | gcn 15,553 · gat 15,617 · gnn_lstm 18,433 — one value per family, == winner |
| `scripts/phase17_framework.py` | **20/20 PASS** (registry amended, gates/test-protection intact) |
| `scripts/phase24_ablation.py` | 24 rows (full-graph gcn/gat × 4 folds × 3 h); paired station bootstrap, **2591 s**; NPZ 96 arrays; pooled CIs in §4 |
| `notebooks/24_gnn.ipynb` | **16/16 PASS** (N24-01…N24-12, 667 s): artefacts, registry contract, test policy, graph, tuning, grid, headline, **paired-CI reproduction**, determinism, SHA, figure, summary |

## 8. Open items handed on

1. **River-only / identity diagnostics.** `--variants=river,id` (message-passing vs added-depth
   separation) were deferred to keep the registered full-graph comparison tractable on the
   CPU-starved host; the flag exists (`phase24_ablation.py --variants=river,id`).
2. **Phase 25 (Transformer).** L4-03 remains gated by `phase25_transformer_gate`; the Phase-24
   verdict informs whether the Transformer is run temporal-only or on the graph.
3. **Phase 27 head-to-head.** Graph vs xgboost — xgboost still leads on `val_2022`.
4. **Test is untouched** (frozen read at Phase 33).

**Next: Phase 25 — Transformer (stage 40, gated by `phase25_transformer_gate`).**
