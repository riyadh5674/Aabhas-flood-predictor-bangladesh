# Phase 25 — Temporal Transformer (workflow stage 40, D13)

| | |
| --- | --- |
| **Date** | 2026-10-14 |
| **Gate (25.1)** | `scripts/phase25_gate.py` → `reports/phase25_transformer_gate.md` — **PASS (narrow, pre-registered)** |
| **Modules** | `scripts/train_transformer.py` (L4-03) · `scripts/phase25_ablation.py` (paired P1/P3 kill-criterion) · `scripts/build_phase25_notebook.py` |
| **Inputs** | Dataset B `sequences_L{3,7,14,30}.npy` + masks (Phase 18/19, P19-M1) · `reports/phase21_metrics.csv` (LSTM/GRU) · `phase22_metrics.csv` (TCN) · `phase23_metrics.csv` (fusion) · `phase20_metrics.csv` (xgboost) · `phase16_baseline_metrics.csv` (D7) |
| **Outputs** | `reports/phase25_metrics.csv` (72) · `reports/phase25_search.csv` (16) · `reports/phase25_ablation.csv` (12) · `reports/phase25_ablation_preds.npz` · `models/phase25_L4-03.pt` · `figures/phase25_val2022.png` · `notebooks/25_transformer.ipynb` · `scripts/phase25_run.log` |
| **Verdict** | **Qualified positive (modest, horizon- and protocol-dependent)** — **Primary PASS**: `val_2022` mean PR-AUC **0.93168** beats LSTM L3-01 (0.92677) and TCN L3-06 (0.92124). **Secondary PASS**: P3 unseen-station mean **0.91247** > LSTM 0.90107 / TCN 0.89743, with paired station-bootstrap **gain over TCN at 48 h and 72 h and over LSTM at 72 h (CI > 0)**, and **no loss at any horizon**. The gain is **not** long-range: the winning window is **L = 7**, not 30. Test never read (frozen to Phase 33). |
| **Governing doc** | `final workflow.md` **Phase 25 — Transformer models** / **stage 40**; D13 (gated advanced model; a documented negative result is acceptable) |

No file under `data/raw/` was modified. `experimental_dataset.csv` SHA-256 is unchanged
(`a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e`); the sequence-index
labels were re-derived 1:1 against it inside the run (0 mismatches).

---

## 1. The question

The gate (`reports/phase25_transformer_gate.md`) authorised **exactly one** fair test under
criteria frozen *before* any Transformer was fit: a temporal Transformer (registry **L4-03**)
tuned on `val_2022` must beat the temporal baselines (primary), and on **P3 spatial CV (unseen
stations)** it must **not lose** to them, with a **paired station-bootstrap CI** on the difference
(secondary). The gate's own window-sweep evidence (LSTM best at **L\* = 3**) predicted the
long-range-attention advantage was **unlikely to materialise**, and D13 makes a tie-or-loss a
legitimate, reportable outcome. This report states the verdict against those rules.

## 2. Design (all pre-registered in the gate)

**Model.** Per-day token → `Linear(13 → hidden)` embedding + learned positional embedding
(`max_len = 30`) → `TransformerEncoder` (self-attention, `nhead = heads`, feed-forward
`2·hidden`, GELU) → **masked-mean pool** (missing days are key-padding-masked, P19-M1) →
`Linear(hidden → 1)` → sigmoid. Channels = 12 raw sequence variables + the observed/missing
mask (13). BCE-with-logits; Adam; early stopping on validation PR-AUC (patience 4, cap 14).

**Fair-comparison parity (D14).** Same Dataset B, splits, masks, metric code, seed
(`20261008`), selection cell (`val_2022`) and selection metric (PR-AUC) as Phases 20–24; equal
class of tuning budget (tuning_budget ≤ 20). The window **L** and every architectural knob are
tuning dimensions inside one seeded random search (`np.random.default_rng(SEED+7)`).

**Tuning space.** `L ∈ {3, 7, 14, 30}` (the workflow's {7, 14, 30} + the L\* = 3 parity point) ·
`lr ∈ {3e-4, 1e-3, 3e-3}` · `hidden ∈ {32, 64}` · `heads ∈ {2, 4}` · `layers ∈ {1, 2}` ·
`dropout ∈ {0.0, 0.3}` · `batch_size ∈ {256, 512}` · `pos_weight ∈ {off, on}` → **768 configs**,
**16 sampled** (≤ registry 20). Score = **mean PR-AUC over 24/48/72 h on `val_2022` (season =
all)**; only the winner enters the 72-row dev grid. Frozen 2023+ test is never read.

## 3. Tuning result and the window sweep (D12)

16 of 768 configs evaluated; **winner `transformer_bestL_c05`, fitted mean PR-AUC 0.93168**.

| rank | config | L | weighting | 24 h | 48 h | 72 h | mean | n_params |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **1** | **c05** | **7** | none | 0.94371 | 0.93304 | 0.91830 | **0.93168** | 36,353 |
| 2 | c07 | 7 | none | 0.93625 | 0.93439 | 0.91358 | 0.92807 | 69,825 |
| 3 | c02 | 3 | pos_weight=ratio | 0.94441 | 0.92945 | 0.91015 | 0.92800 | 18,529 |
| 4 | c11 | 14 | none | 0.94442 | 0.92699 | 0.91101 | 0.92748 | 9,985 |
| 5 | c08 | 7 | pos_weight=ratio | 0.94053 | 0.93049 | 0.91130 | 0.92744 | 36,353 |
| 6 | c09 | 14 | none | 0.94000 | 0.92677 | 0.91065 | 0.92580 | 69,825 |

**Best config per window length** (the workflow's core evidence for the Transformer):

| L | 3 | **7** | 14 | 30 |
| --- | ---: | ---: | ---: | ---: |
| best Transformer mean PR-AUC | 0.92800 | **0.93168** | 0.92748 | 0.92532 |

As the gate predicted, the benefit does **not** grow with the window: the best config is at
**L = 7** and L = 30 is the **weakest** of the four. The Transformer wins here as a
short-window attention model, **not** via the 30-day long-range dependency it is sold on —
consistent with the LSTM sweep (L\* = 3) and with the gate's "low prior" note.

## 4. Results — fully-tuned dev grid (season = all, PR-AUC)

Winner `c05`: `hidden` 64 · `heads` 4 · `layers` 1 · `dropout` 0.0 · `lr` 1e-3 · `batch_size`
256 · unweighted · best epochs 24/48/72 h = 5/7/12 · **36,353 params**.

**P1 `val_2022`** (the Phase-21/22/23/24 comparison cell):

| model | 24 h | 48 h | 72 h | mean | params |
| --- | ---: | ---: | ---: | ---: | ---: |
| **L4-03 transformer (L=7)** | 0.9437 | 0.9330 | 0.9183 | **0.93168** | 36,353 |
| L3-01 lstm_L3 | 0.9421 | 0.9253 | 0.9129 | 0.92677 | 14,497 |
| L3-06 tcn_bestL | 0.9405 | 0.9224 | 0.9008 | 0.92124 | — |
| L3-07 fusion_L3 | **0.9444** | **0.9385** | **0.9238** | **0.93559** | — |
| *D7 persistence* | 0.9203 | 0.8954 | 0.8683 | 0.89465 | — |
| *xgboost (Phase 20)* | 0.9681 | 0.9549 | 0.9408 | 0.95463 | — |

Transformer **full metric rows** (`val_2022`, season = all):

| horizon | PR-AUC | CI (100-rep) | ROC-AUC | Brier | ECE | recall | precision | F1 | macro PR-AUC | n |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 h | 0.9437 | [0.855, 0.981] | 0.9965 | 0.0100 | 0.0024 | 0.825 | 0.896 | 0.859 | 0.861 | 17,502 |
| 48 h | 0.9330 | [0.863, 0.974] | 0.9936 | 0.0128 | 0.0040 | 0.790 | 0.882 | 0.833 | 0.842 | 17,300 |
| 72 h | 0.9183 | [0.843, 0.962] | 0.9906 | 0.0161 | 0.0057 | 0.802 | 0.847 | 0.824 | 0.815 | 17,089 |

**P2 rolling-origin mean (3 years) and P3 spatial-CV mean (4 folds):**

| protocol | model | 24 h | 48 h | 72 h |
| --- | --- | ---: | ---: | ---: |
| P2 | **transformer (L=7)** | 0.9549 | **0.9483** | **0.9280** |
| P2 | L3-01 lstm_L3 | 0.9551 | 0.9362 | 0.9193 |
| P2 | L3-06 tcn_bestL | 0.9478 | 0.9371 | 0.9063 |
| P2 | L3-07 fusion_L3 | 0.9593 | 0.9485 | 0.9316 |
| **P3** | **transformer (L=7)** | 0.9325 | **0.9095** | **0.8954** |
| **P3** | **L3-01 lstm_L3** | 0.9196 | 0.9055 | 0.8781 |
| **P3** | **L3-06 tcn_bestL** | 0.9258 | 0.8949 | 0.8716 |
| **P3** | **L3-07 fusion_L3** | 0.9154 | 0.9202 | 0.9171 |
| **P3** | **L4-02 gat** | 0.9177 | 0.9162 | 0.9011 |

Read from the grid: on P1 the Transformer is **≈tied with the temporal baselines at 24 h**
(0.9437 vs LSTM 0.9421) and **ahead at 48/72 h**; it remains **below fusion_L3** and **well
below xgboost**, and comfortably above D7. On P2 rolling origin the edge is the same shape —
tied at 24 h, clearly ahead at 48 h (+0.012) and 72 h (+0.009). On **P3 (unseen stations)** the
Transformer leads the temporal-only ladder at **all three horizons** (24 h +0.007, 48 h +0.004,
72 h +0.014 vs LSTM). Season cells: monsoon 0.9419/0.9344/0.9206, non-monsoon
0.9524/0.9302/0.9112 — no monsoon collapse.

## 5. Results — paired kill-criterion ablation (Transformer − baseline)

Each winner re-fit on the **same** cells with identical batching/seed; the per-sample
predictions differenced under a **paired station-resampling bootstrap** (1,000 resamples of the
held-out stations). P1 is `val_2022`; P3 pools the 4 disjoint spatial folds (each station held
out exactly once).

| protocol | comparison | horizon | n | Transformer | baseline | Δ (mean) | 95 % CI | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| P1 | vs LSTM L3-01 | 24 h | 17,502 | 0.94371 | 0.94206 | +0.0014 | [−0.0079, +0.0077] | no gain |
| P1 | vs LSTM L3-01 | 48 h | 17,300 | 0.93304 | 0.92534 | +0.0086 | [−0.0026, +0.0238] | no gain (dir. +) |
| P1 | vs LSTM L3-01 | 72 h | 17,089 | 0.91830 | 0.91291 | +0.0068 | [−0.0029, +0.0188] | no gain (dir. +) |
| P3 | vs LSTM L3-01 | 24 h | 61,977 | 0.93691 | 0.92972 | +0.0072 | [−0.0107, +0.0361] | no gain (dir. +) |
| P3 | vs LSTM L3-01 | 48 h | 60,992 | 0.91381 | 0.90679 | +0.0070 | [−0.0047, +0.0185] | no gain (dir. +) |
| P3 | vs LSTM L3-01 | 72 h | 60,144 | 0.89918 | 0.88312 | **+0.0163** | [+0.0042, +0.0289] | **gain (CI > 0)** |
| P1 | vs TCN L3-06 | 24 h | 17,502 | 0.94371 | 0.94052 | +0.0040 | [−0.0066, +0.0183] | no gain (dir. +) |
| P1 | vs TCN L3-06 | 48 h | 17,300 | 0.93304 | 0.92236 | **+0.0118** | [+0.0008, +0.0287] | **gain (CI > 0)** |
| P1 | vs TCN L3-06 | 72 h | 17,089 | 0.91830 | 0.90083 | **+0.0202** | [+0.0068, +0.0447] | **gain (CI > 0)** |
| P3 | vs TCN L3-06 | 24 h | 61,977 | 0.93691 | 0.93560 | +0.0020 | [−0.0086, +0.0119] | no gain (dir. +) |
| P3 | vs TCN L3-06 | 48 h | 60,992 | 0.91381 | 0.90158 | **+0.0127** | [+0.0057, +0.0201] | **gain (CI > 0)** |
| P3 | vs TCN L3-06 | 72 h | 60,144 | 0.89918 | 0.87772 | **+0.0221** | [+0.0028, +0.0402] | **gain (CI > 0)** |

**5 separable cells, all in the Transformer's favour; 0 losses.** Every non-separable cell has a
**positive** point estimate. The registered comparison is the **pooled** CI above; the
notebook (N25-08) re-derives every row from the saved per-sample predictions and reproduces the
stored CIs **bit-exactly**.

## 6. Verdict

**Qualified positive — modest, and horizon- and protocol-dependent.**

* **Primary: PASS.** On `val_2022` (season = all, mean PR-AUC over 24/48/72 h) the Transformer
  scores **0.93168**, beating the best temporal baseline — the LSTM at **0.92677** (+0.005) and
  the TCN at **0.92124** (+0.010). The gate's requirement ("beat TCN, and ideally the LSTM") is met.
* **Secondary: PASS.** On **P3 spatial CV (unseen stations)** the pooled mean is **0.91247**,
  above the LSTM (0.90107) and TCN (0.89743); the paired bootstrap shows **significant gains at
  72 h vs the LSTM and at 48/72 h vs the TCN**, and **no horizon loses** (all Δ ≥ 0).
* **Kill: not triggered.** The kill branch required "no meaningful gain (CI spanning 0)". CIs do
  exclude 0, in the positive direction, at the horizons where the gate expected any edge — so
  stage 40 closes as a **qualified positive**, not a documented rejection.

**Why "modest", and what it is not.** The gain is real but small (~+0.005–0.022 PR-AUC) and
**does not come from long-range attention**: the winning window is **L = 7** and L = 30 is the
weakest of the four — the workflow's hoped-for "30 days is where attention should matter" did
**not** materialise, exactly as the gate predicted. The edge is consistent with a **short-window
attention model that generalises slightly better than recurrent/convolutional encoders to
unseen stations and to longer lead times**. The Transformer stays **below `fusion_L3`
(0.93559) on P1** and **far below xgboost (0.95463)** on both bodies of evidence. Stated plainly:
a more complex model with a measurable but marginal gain is a *slightly* better scientific
model here — not a new state of the art.

**Consequence.** The gate's kill clause said a rejection would also forbid a *spatiotemporal*
Transformer. It was **not** triggered, so a Transformer-on-graph is **permitted** — but the
evidence (a ~+0.01–0.02 P3 gain that the Phase-24 GNN already achieves, on a temporal gain that
does not survive at 24 h) does **not** make it necessary. It is handed on as an optional item,
not a required one (§9).

## 7. Registry amendment (why the phase touched a completed stage)

The gate report §6 says 25.2 discharges `phase25_transformer_gate`. `build_registry()` in
`scripts/phase17_framework.py` now marks **L4-03 `transformer_temporal` `planned`** (gate passed,
run authorised) — so **all four L4 rows (L4-01/02/03/04) are `planned`**, with no row left
`gated`. The F1m framework check was updated to assert exactly this (both D13 gates discharged);
the framework was re-run **20/20 PASS**. The amendment log is in `scripts/phase17_run.log` and
the registry (`reports/phase17_experiment_registry.csv`, 24 rows) carries
`status = planned` for L4-03 with `model_out = models/phase25_L4-03.pt`;
`build_phase17_notebook.py` N17-08 was updated to match.

## 8. Verification

| Check | Result |
| --- | --- |
| `scripts/train_transformer.py` (L4-03) | exit=0; search 16 rows == budget 16 (space 768); dev grid **72 rows**, 0 test cells; `models/phase25_L4-03.pt` written |
| per-model `n_params` consistency | one value for the family (36,353) == winner |
| index-label parity | sequence-index labels re-derived 1:1 vs `experimental_dataset.csv` (200,665 rows, 133 stations) — 0 mismatches |
| `scripts/phase25_ablation.py` | 12 rows (P1 val_2022 + P3 pooled × 3 h × 2 baselines); paired station bootstrap **1,000** resamples; **2,270 s**; NPZ per-sample predictions for audit |
| `scripts/phase17_framework.py` | **20/20 PASS** (registry amended: L4-03 → `planned`; gates/test-protection intact) |
| `notebooks/25_transformer.ipynb` | **12/12 PASS** (N25-01…N25-12): artefacts, registry contract, test policy, gate, tuning + window sweep, grid, headline, **paired-CI reproduction**, determinism, SHA, figure |
| `data/processed/experimental_dataset.csv` | SHA-256 `a6748529083f…73c8e` unchanged |

**Warnings (recorded, not footnoted).**

1. **Wall clock ≠ compute.** `phase25_run.log` reports `TOTAL 52217 s` ≈ 14.5 h, but a single
   tuning config (`c11`, L = 14) logged **33,617 s** because the host **entered standby during
   it** (standby timeouts were found at AC 300 s / DC 180 s and disabled for the remainder of the
   phase). The genuine model compute is ≈ 18,600 s (~5 h); the host sleep is the outlier, not the
   model. No artefact was affected (the run is seeded and deterministic).
2. **P1 per-horizon CIs span 0** even though the **mean** (the registered primary) passes. The
   primary is a point-estimate rule on the mean; the per-horizon intervals are reported here for
   completeness and live in `phase25_ablation.csv`.
3. **`hidden`=64 / `heads`=4 raised `n_params`** to 36,353 (vs ~15 k for the Phase-21/24 nets);
   the win is **not** attributable to capacity alone — the same-capacity `c07` and the small
   `c11` (9,985 params) are both below the winner.

## 9. Open items handed on

1. **Spatiotemporal Transformer (optional).** Permitted (kill not triggered) but **not
   required**: a Transformer-on-graph would re-test at 48/72 h a P3 gain the Phase-24 GNN already
   delivers. Deferred to Phase 27/32 if the graph + attention interaction is worth one more
   registered slot.
2. **xgboost remains the ceiling** on P1 (0.95463) and P3 (0.96225). The Phase-27 head-to-head
   and Phase-32 ablation matrix decide whether any DL family closes it.
3. **Test is untouched** (frozen read at Phase 33).

**Next: Phase 26 — multi-horizon strategy (stage 41 + stages 59–61)**, filling the full 24/48/72 h
matrix across all surviving L1–L4 models with the Transformer added to the ladder.
