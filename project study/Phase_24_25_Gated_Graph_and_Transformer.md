# Phase 24–25 — The Gated Advanced Models: Station-Graph GNN & Temporal Transformer
### Project study notes · Bangladesh flood-probability forecasting

> Sources: `reports/phase24_graph_gate.md`, `reports/phase24_gnn.md`,
> `reports/phase25_transformer_gate.md`, `reports/phase25_transformer.md`.
> Previous study: Phase 23 (fusion). These phases show how the project's **gate** mechanism
> (D13) works in practice — and both gates **passed**.

---

## 0. The 60-second version

Phases 24–25 run the two most advanced model families. Because they're expensive and easy to
over-claim, each is **gated** (D13): it may only run if a **pre-committed evidence test** passes
*first*. Both gates passed (narrowly), so both families ran.

**Phase 24 — Station-graph GNN.** Build a graph connecting hydrologically-related stations
(upstream/downstream) and let a graph neural network pass messages between them. The **only
question that matters** is: on **unseen stations (P3)**, does the graph beat the temporal-only
LSTM? Answer: **yes at 48 h and 72 h (significant), no at 24 h.**

**Phase 25 — Temporal Transformer.** Does attention over days beat recurrence? Answer:
**a modest yes** — the Transformer beats the LSTM/TCN on the primary cell and on unseen
stations, but the win is **small** and **does NOT come from long-range attention** (the best
window is L = 7, not 30). It stays **below `fusion_L3`** and **far below XGBoost**.

**The honest theme:** both advanced models produce **small, qualified, honestly-bounded** gains,
reported with paired confidence intervals rather than hype.

---

## 1. Where Phase 24–25 sit

| Phase | What | Plain words |
|---|---|---|
| 23 | Feature fusion | Geography barely helped |
| **➜ 24–25** | **Gated graph + transformer** | **Try advanced structure — only if the gate says so** |
| 26 | Multi-horizon strategy | Fill the full horizon matrix |
| 27 | Head-to-head | ML vs DL, fairly |
| 28–33 | Calibrate → select → freeze | Pick champion, read test once |

Closes workflow **stages 39 (GNN)** and **40 (Transformer)**.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Gate** | A pre-committed evidence test a model family must pass before it may run. |
| **D13** | The gate rule: advanced models run only on a gate pass; rejection is acceptable. |
| **Graph** | A set of nodes (stations) connected by edges (relationships). |
| **Edge** | A link between two stations (e.g. upstream → downstream). |
| **GNN** | Graph Neural Network — passes "messages" along edges. |
| **GCN** | Graph Convolutional Network — uniform message passing. |
| **GAT** | Graph Attention Network — learns how much to weight each neighbour. |
| **GNN-LSTM** | Graph encoder (spatial) + LSTM (temporal) = full stack. |
| **Transformer** | Attention-based model; each day can "attend" to other days. |
| **Self-attention** | Each token weighs all others to build its representation. |
| **Masked-mean pool** | Average over observed days only (missing days masked out). |
| **Positional embedding** | Tells the model the *order* of the days. |
| **Paired bootstrap CI** | Resample stations; get a CI on the **difference** in PR-AUC. |
| **Kill criterion** | The pre-registered result that closes a stage as "no gain". |
| **Head** | The final output layer (here: linear → sigmoid). |
| **Qualified positive** | A real but small/horizon-dependent gain — reported as such. |

---

## 3. Phase 24 gate (24.1) — is a station graph justified?

Before training any GNN, three numeric questions were asked:

### (a) Can a defensible edge set be built?
Yes — build the graph **hydrology-first**, distance only as a fallback.

| Graph | Definition | Edges | Connected? |
|---|---|---:|---|
| **river_upstream** (primary) | same `river`, neighbour ≥ 0.5 m higher, ≤ 250 km → directed upstream→downstream | 52 | no (51 isolated) |
| same_river_250km | all same-river pairs ≤ 250 km | 111 | no |
| basin_clique | all pairs sharing a basin | 2,178 | no |
| knn3 | nearest-3 gauges by distance | 260 | no |
| knn5 | nearest-5 gauges | 415 | **yes (all 133)** |

The hydrological graph is **sparse and fragmented** (81 components, largest 10). **51 of 133
stations (38 %) are isolated** → they need the distance fallback. **Approved graph:**
`river_upstream ∪ knn5` → **427 edges, connected, all 133 stations, mean degree 6.42**, frozen
as `station_graph.csv` with SHA-256 **before any model saw it**.

### (b) Does the edge mapping match what the project already trusts?
**Yes.** Re-running the Phase-11C rule yields 52 edges; **50 reproduce the stored
`up_wl_pct_danger_lag1` column exactly** (mean match 0.9997).

### (c) Do stations actually carry information about each other?
**Yes, strongly, with a propagation shape.** Correlation between upstream WL(t−lag) and
downstream WL(t) over river edges vs a random-pair null:

| Lag (days) | River edges (mean) | Null | Delta |
|---:|---:|---:|---:|
| 0 | 0.900 | 0.578 | **+0.322** |
| 1 | 0.892 | 0.574 | **+0.317** |
| 2 | 0.863 | 0.568 | +0.296 |
| 3 | 0.832 | 0.559 | +0.273 |

Every river edge has lag-1 r ≥ 0.5. The null is **not zero** (0.58) because all Bangladeshi
gauges share the monsoon cycle — the **river-specific excess (+0.27 to +0.32)** is the real
hydrological coupling.

### What the gate decided
**PASS (narrow).** But it added a crucial nuance: neighbour signal is **real but mostly
redundant with a station's own persistence on P1**. So the GNN gets **exactly one fair test**:
beat the temporal-only LSTM on **P3 unseen stations**, with a paired CI. Otherwise → documented
rejection.

**Plain point:** the gate did **not** promise the GNN would win. It authorised one honest test.

---

## 4. Phase 24 (24.2) — running the GNNs

Models (node features = per-station sequence, temporal encoder as Phase 21, L\* = 3):

| Registry | Model | Mechanism |
|---|---|---|
| **L4-01** | `gcn` | uniform message passing over the graph |
| **L4-02** | `gat` | attention — learns neighbour weights |
| **L4-04** | `gnn_lstm` | graph spatial encoder + LSTM temporal encoder |
| L3-01 | `lstm_L3` | temporal-only reference |

### P1 `val_2022`

| Model | 24 h | 48 h | 72 h |
|---|---:|---:|---:|
| L4-01 gcn | 0.9421 | 0.9353 | 0.9141 |
| L4-02 gat | 0.9408 | 0.9314 | 0.9136 |
| L4-04 gnn_lstm | **0.9424** | 0.9264 | **0.9153** |
| L3-01 lstm_L3 | 0.9421 | 0.9253 | 0.9129 |
| *xgboost (P20)* | *0.9681* | *0.9549* | *0.9408* |

On P1 the graph families are **≈equal** to the temporal-only LSTM at 24 h and marginally ahead
at 48/72 h — as the gate predicted (redundancy).

### P3 spatial CV (unseen stations) — the registered decision

Pooled P3 CI per horizon (4 disjoint folds concatenated), paired station bootstrap:

| Family | Horizon | GNN | LSTM L3-01 | Δ | 95% CI | Verdict |
|---|---:|---:|---:|---:|---|---|
| gcn | 24 h | 0.9272 | 0.9297 | −0.0021 | [−0.0216, +0.0114] | no gain |
| gcn | 48 h | 0.9153 | 0.9068 | +0.0085 | [−0.0077, +0.0224] | no gain (dir. +) |
| gcn | 72 h | 0.9010 | 0.8831 | **+0.0178** | [+0.0064, +0.0340] | **gain (CI > 0)** |
| gat | 24 h | 0.9166 | 0.9297 | **−0.0129** | [−0.0278, −0.0020] | **LSTM better** |
| gat | 48 h | 0.9228 | 0.9068 | **+0.0157** | [+0.0081, +0.0247] | **gain (CI > 0)** |
| gat | 72 h | 0.9063 | 0.8831 | **+0.0228** | [+0.0026, +0.0563] | **gain (CI > 0)** |

### Phase 24 verdict — qualified positive, horizon-dependent
- **72 h** (longest horizon, where hydrological propagation should dominate): **GCN +0.018,
  GAT +0.023**, both significant.
- **48 h**: GAT +0.016 significant; GCN directional.
- **24 h**: **no help** (GCN ties; GAT is significantly *worse*) — a station's own recent
  history already carries the signal. Exactly the redundancy the gate flagged.
- **GAT** reaches the best P3 numbers at 48/72 h but pays the largest 24 h penalty; **GCN** is
  the safer, flatter variant.

**Not a rejection** — the CIs exclude 0 in the positive direction where the hypothesis predicts.

---

## 5. Phase 25 gate (25.1) — is a Transformer justified?

The gate pre-registered: a temporal Transformer must (primary) beat the temporal baselines on
`val_2022`, and (secondary) **not lose** to them on P3 unseen stations, with a paired CI.

The gate also made a **prediction**: since the LSTM's best window was **L\* = 3**, the
"long-range attention" advantage is **unlikely to materialise** → **PASS (narrow, pre-registered)**.

**Plain point:** the gate locked the criteria *before* any Transformer was fit. That's the whole
idea.

---

## 6. Phase 25 (25.2) — running the Transformer

**Architecture:** per-day token → `Linear(13 → hidden)` + learned positional embedding →
`TransformerEncoder` (self-attention) → **masked-mean pool** (missing days masked, P19-M1) →
`Linear(hidden → 1)` → sigmoid. Early stopping on **validation PR-AUC only**.

### Window sweep (D12) — the gate's prediction confirmed

| L | 3 | **7** | 14 | 30 |
|---|---:|---:|---:|---:|
| best Transformer mean PR-AUC | 0.92800 | **0.93168** | 0.92748 | 0.92532 |

**The benefit does NOT grow with the window** — best at **L = 7**, and **L = 30 is the weakest**.
The Transformer wins as a **short-window attention model**, *not* via the 30-day long-range
dependency it's sold on. (Consistent with the LSTM sweep's L\* = 3 and the gate's "low prior".)

### Primary result — P1 `val_2022`

| Model | 24 h | 48 h | 72 h | Mean | Params |
|---|---:|---:|---:|---:|---:|
| **L4-03 transformer (L=7)** | 0.9437 | 0.9330 | 0.9183 | **0.93168** | 36,353 |
| L3-01 lstm_L3 | 0.9421 | 0.9253 | 0.9129 | 0.92677 | 14,497 |
| L3-06 tcn_bestL | 0.9405 | 0.9224 | 0.9008 | 0.92124 | — |
| L3-07 fusion_L3 | **0.9444** | **0.9385** | **0.9238** | **0.93559** | — |
| *xgboost (P20)* | *0.9681* | *0.9549* | *0.9408* | *0.95463* | — |

**Primary: PASS** — beats the LSTM (+0.005) and TCN (+0.010) on the mean.

### Secondary result — P3 unseen stations

The Transformer **leads the temporal-only ladder at all three horizons** (24 h +0.007, 48 h
+0.004, 72 h +0.014 vs LSTM), pooled mean **0.91247** > LSTM 0.90107 > TCN 0.89743.

### Paired kill-criterion ablation (the registered decision)

**5 separable cells, all in the Transformer's favour; 0 losses.** Every non-separable cell has a
**positive** point estimate. Key cells: vs LSTM at 72 h P3 **+0.0163 CI [+0.0042, +0.0289]**
(gain); vs TCN at 48 h and 72 h both **gain (CI > 0)**.

### Phase 25 verdict — qualified positive, modest
- **Primary PASS**, **Secondary PASS**, **kill not triggered**.
- But the gain is **small** (~+0.005–0.022 PR-AUC) and **not from long-range attention**.
- The Transformer stays **below `fusion_L3`** on P1 and **far below XGBoost**.

**Stated plainly:** a more complex model with a measurable but marginal gain is a *slightly*
better scientific model here — **not a new state of the art**.

---

## 7. Registry amendments (how gates change the rulebook)

Both phases **amended the registry** rather than running ad hoc:
- After the Phase-24 gate: L4-01/02/04 → `planned`.
- After the Phase-25 gate: L4-03 → `planned`. Now **all four L4 rows are `planned`**, no row
  `gated`.

The framework was re-run each time and stayed **20/20 PASS**. Ledger references updated.

---

## 8. Verification & warnings

| Check | Result |
|---|---|
| `scripts/train_gnn.py` | exit=0; dev grid 216 rows, **0 test cells** |
| `scripts/train_transformer.py` | exit=0; search 16 rows; dev grid 72 rows, 0 test cells |
| Phase 24 notebook | **16/16 PASS** (incl. paired-CI reproduction, determinism) |
| Phase 25 notebook | **12/12 PASS** (incl. paired-CI reproduction, determinism) |
| `experimental_dataset.csv` SHA | unchanged (`a6748529083f…73c8e`) |

**Recorded warnings (not hidden):**
1. **Wall clock ≠ compute.** Phase 25 logged `TOTAL 52217 s` (~14.5 h) but the host entered
   **standby** during one config; genuine compute ≈ 18,600 s (~5 h). No artefact affected.
2. **P1 per-horizon CIs span 0** even though the **mean** (the registered primary) passes — the
   per-horizon intervals are reported for completeness.
3. The Transformer's win is **not** just capacity: same-capacity `c07` and the small `c11`
   (9,985 params) are both below the winner.

---

## 9. Problems & solutions

| Problem | Solution |
|---|---|
| Advanced models could be run ad hoc and over-claimed | **Gates** with pre-registered criteria (D13). |
| Graph could be arbitrary | **Hydrology-first** edges; distance only as a documented fallback; frozen + SHA. |
| A "win" could just be capacity or noise | **Matched-capacity** runs + **paired station-bootstrap CIs**. |
| P1 gains could hide P3 weakness | Registered **P3 (unseen stations)** as the decisive protocol. |
| Long-range attention might be assumed to help | Window sweep proved **L = 7 best, L = 30 weakest** — reported honestly. |
| Gates could be quietly dropped | **Amendments logged**; framework re-run 20/20 each time. |
| Host sleep could corrupt timing claims | Warning recorded; compute estimate separated from wall clock. |

---

## 10. Files referenced

| File | Purpose |
|---|---|
| `reports/phase24_graph_gate.md` | The graph gate (24.1) |
| `reports/phase24_gnn.md` | GNN models & results (24.2) |
| `reports/phase25_transformer_gate.md` | The Transformer gate (25.1) |
| `reports/phase25_transformer.md` | Transformer results (25.2) |
| `data/processed/station_graph.csv` | Frozen graph (427 edges, SHA) |
| `models/phase24_L4-0{1,2,4}.pt`, `models/phase25_L4-03.pt` | Saved models |
| `reports/phase24_metrics.csv`, `phase25_metrics.csv` | Dev grids (0 test rows) |
| `notebooks/24_gnn.ipynb`, `notebooks/25_transformer.ipynb` | Verification notebooks |

---

## 11. Verdict

- **Phase 24 (GNN):** qualified positive, **horizon-dependent** — the hydrological graph
  significantly helps on **unseen stations at 48 h and 72 h** (GAT +0.016, +0.023; GCN +0.018
  at 72 h) but **not at 24 h** (redundant with own persistence). Consistent with the gate's
  prediction.
- **Phase 25 (Transformer):** qualified positive, **modest** — passes both the primary and
  secondary rules with **5 separable wins and 0 losses**, but the gain is **small** and comes
  from **short-window attention (L = 7)**, **not** long-range. Below `fusion_L3`, far below
  XGBoost.
- **Both gates passed**; registry amended; framework **20/20**; **test never read**.

**Bottom line:** The advanced families are worth exactly what the evidence says — small,
horizon-and-protocol-dependent gains, documented with confidence intervals. The **gate
mechanism worked**: it forced the claims to be pre-registered and honest, and let the project
say "this helped a little, here's the proof" instead of over-claiming a new state of the art.

---

**Next in the series:** Phase 26 — Multi-horizon strategy (fill the full 24/48/72 h matrix across
all surviving L1–L4 models). Then Phase 27 (ML vs DL head-to-head), 28–29 (calibration &
thresholds), 30–33 (error analysis, selection, frozen test).
