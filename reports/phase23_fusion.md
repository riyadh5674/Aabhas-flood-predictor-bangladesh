# Phase 23 — Multi-input / feature-fusion deep learning (workflow stage 38)

| | |
| --- | --- |
| **Date** | 2026-10-11 |
| **Modules** | `scripts/build_static.py` (static vector) · `scripts/train_fusion.py` (fused / static-only / hybrid) · `scripts/phase23_ablation.py` (matched-capacity gain + CI) · `scripts/build_phase23_notebook.py` |
| **Inputs** | Dataset B `sequences_L3.npy` + `sequence_mask_L3.npy` (P19-M1) · `data/processed/static_vectors.csv` + `static_manifest.json` · `reports/phase21_metrics.csv` (L3-01 temporal-only) · `reports/phase20_metrics.csv`, `reports/phase16_baseline_metrics.csv` |
| **Outputs** | `reports/phase23_metrics.csv` (216) · `reports/phase23_search.csv` (44) · `reports/phase23_ablation.csv` (144) · `reports/phase23_ablation_boot.csv` · `reports/phase23_ablation_preds.npz` · `models/phase23_L3-0{7,8,9}.pt` · `figures/phase23_val2022.png` · `notebooks/23_fusion.ipynb` |
| **Verdict** | `notebooks/23_fusion.ipynb` **14/14 PASS** · `build_static.py` **7/7 PASS** · `phase17_framework.py` **20/20 PASS** (registry amended to 24 rows) · `train_fusion` + `phase23_ablation` exit=0 · test never read |
| **Governing doc** | `final workflow.md` Phase 23 / stage 38; D14 parity; D13 (fusion does **not** open the gated GNN/Transformer stage) |

No file under `data/raw/` was modified. `experimental_dataset.csv` SHA-256 is unchanged
(`a6748529083f…73c8e`); every run re-derived the sequence-index labels 1:1 against it.

---

## 1. The question

Location-aware prediction is the project's point, so the temporal model must be allowed to
see **where** it is. Phase 23 asks: **does adding a static geographic vector to the Phase-21
temporal encoder add skill over the sequence alone — and by how much, with a
station-bootstrap CI on the gain, not just point estimates (DoD)?**

## 2. Design

**Branch 1 — temporal encoder.** Fixed to the Phase-21 winner: LSTM, **L\* = 3** (registry
L3-01). Fusion only *adds* the static branch, so any fully-tuned gain over L3-01 cannot be
attributed to using a different temporal model.

**Branch 2 — static vector** (`build_static.py` → `static_vectors.csv`, one row per station).
Every field is constant within a station (checked, S23-s3):

| Block | Columns | Encoding | Dim |
| --- | --- | --- | ---: |
| numeric | `lat`, `lon`, `danger_level`, `elevation_m`, `slope_deg`, `dist_to_coast_km`, `dist_river_km`, `dist_major_river_km`, `river_density_10km_km_per_km2`, `n_river_lines_10km` | standardised **on the training fold** | 10 |
| categorical | `river`, `basin`, `division`, `district`, `upazilla` | frequency-encoded **on the training fold** (unseen/NaN → 0; identical to the Phase-14/16 `FreqEncoder`) | 5 |

`static_dim = 15` over 133 stations. Cardinalities: river 80, basin 5, division 8, district
55, upazilla 115. The raw values are stored; all standardisers / frequency maps are fit inside
each training fold, so the file carries no cross-fold information and no label.

**Model.** `FusionNet` = (LSTM(L\*, 13ch) last-step hidden) ⊕ (static MLP on the 15-vector) →
concat → `Linear(·,1)` → sigmoid, `BCEWithLogitsLoss`, seeded (`20261008`). The static branch
is a stack of `layers` hidden layers of width `hidden`, so `hidden`/`layers`/`dropout` are
live knobs (a first cut that hard-coded the static width was corrected and re-run).

**Variants (all registered — the registry is the contract):**

| Registry | model | branch 1 | branch 2 | extra |
| --- | --- | --- | --- | --- |
| **L3-07** | `fusion_L3` | LSTM L\*=3 | static | — |
| **L3-08** | `static_only` | — | static | — |
| **L3-09** | `hybrid_L3` | LSTM L\*=3 | static | + Phase-12 LINEAR engineered set (35) |
| L3-01 | `lstm_L3` | LSTM L\*=3 | — | temporal-only (Phase 21, reported not re-trained) |

Same data, split masks, metric code, selection cell (`val_2022`), seed and budget class as
every other family (D14). Budgets 16 / 16 / 12 ≤ 16. Frozen 2023+ test never read.

## 3. Results — fully-tuned three-way (val_2022, season = all, PR-AUC)

Winner configs: `fusion_L3` hidden 64 / 2 layers / dropout 0.3 / lr 3e-3 / batch 512 (58,817
params, unweighted); `static_only` hidden 64 / 1 layer / lr 1e-3 (1,089 params,
pos_weight=ratio); `hybrid_L3` hidden 32 / 2 layers / lr 1e-3 / batch 512 (16,132 params).

| model | 24h | 48h | 72h | params |
| --- | ---: | ---: | ---: | ---: |
| **L3-07 fusion_L3** | **0.9444** | **0.9385** | **0.9238** | 58,817 |
| L3-09 hybrid_L3 | 0.9275 | 0.9215 | 0.9072 | 16,132 |
| L3-01 lstm_L3 (temporal-only) | 0.9421 | 0.9253 | 0.9129 | 14,497 |
| L3-08 static_only | 0.5068 | 0.5386 | 0.5521 | 1,089 |
| *D7 persistence* | 0.9203 | 0.8954 | 0.8683 | — |
| *xgboost (Phase 20)* | 0.9681 | 0.9549 | 0.9408 | — |

Read naively, fusion beats temporal-only by +0.002 / +0.013 / +0.011. **But the fusion winner
is a larger encoder** (58,817 vs 14,497 params), so this is confounded by capacity. The
hybrid (learned + engineered) is *worse* than pure fusion at every horizon — the engineered
LINEAR set adds nothing on top of what the encoder already learned. `static_only` is far
below both, and below D7.

## 4. Results — matched-capacity ablation (the honest answer, DoD)

Same encoder config and epoch counts for **fused** and **temporal-only** (only the static
branch toggled), scored on `val_2022`, with a **paired station bootstrap** (100 resamples of
the 84 reporting stations) on the difference PR-AUC(fused) − PR-AUC(temporal-only):

| horizon | fused | temporal-only | gain | 95% CI | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 24 h | 0.9444 | 0.9420 | **+0.0024** | [−0.0003, +0.0084] | not significant |
| 48 h | 0.9385 | 0.9332 | **+0.0053** | [−0.0010, +0.0123] | borderline, not significant |
| 72 h | 0.9238 | 0.9279 | **−0.0041** | [−0.0107, −0.0011] | static branch **hurts** (CI < 0) |

- The matched temporal-only already reaches 0.9420 / 0.9332 / 0.9279 — **better than the
  Phase-21 L3-01 winner** at 48 h/72 h. The fusion search sample simply found a stronger
  encoder config than the Phase-21 budget-12 sample, which is why the naive fusion−L3-01 gap
  is largely a *tuning* artefact, not a fusion effect.
- At fixed capacity the static vector buys a small, **statistically insignificant** gain at
  24 h and 48 h, and a small but **significant loss** at 72 h.

**Where the static branch does help** (matched): the rolling-origin P2 mean and the spatial
CV P3 mean — i.e. generalisation across years and *unseen stations*, not the pooled P1 number:

| protocol (mean) | fused 24/48/72 | temporal-only 24/48/72 |
| --- | --- | --- |
| P2 rolling | 0.9593 / 0.9485 / 0.9316 | 0.9536 / 0.9434 / 0.9274 |
| P3 spatial CV | 0.9154 / 0.9202 / 0.9171 | 0.9183 / 0.9183 / 0.9111 |

Fused wins all three P2 horizons and 48 h/72 h of P3 (the horizon where unseen-station
generalisation is hardest). `static_only` in P3 collapses to **0.12–0.15** PR-AUC — location
alone cannot identify a flooded station once its own dynamics are removed, which is exactly
the project's premise (water level, not latitude, drives the label).

## 5. Verdict

**Knowing *where* a station is, given its temporal sequence, adds little.** At matched
capacity the static branch moves `val_2022` PR-AUC by +0.002 / +0.005 / −0.004 across 24/48/72 h
— significant only at 72 h, and *negative*. Its clearest positive effect is a modest gain at
longer horizons when generalising to unseen stations (P3) and across rolling years (P2). The
headline fusion−L3-01 gap is mostly explained by the encoder config found during the fusion
search, not by the geographic input. The learned+engineered hybrid (L3-09) is dominated by
pure fusion. All deep variants remain below the Phase-20 xgboost bar and above D7.

This is a **reportable negative/marginal result** for the fusion input, consistent with the
D12 honouring of "the model is allowed to see location" without over-claiming its value.

## 6. Registry amendment (why the phase touched a completed stage)

Phase 17's registry is the contract ("anything not in the registry is an ad-hoc run"), and a
GNN/Transformer row is gated, but **no row existed for fusion**. The three variants are
therefore added to `build_registry()` in `scripts/phase17_framework.py` (L3-07/08/09, phase
"23", family "feature fusion", budget 16, `metrics_out=phase23_metrics.csv`) and the framework
re-run; it passes **20/20** and the registry now holds **24** experiments
(6/5/9/4). The amendment log is appended to `scripts/phase17_run.log`; the ledger references
(workflow lines 99, 663) were updated 21 → 24. `static_only` uses dataset label **"S"**
(static, non-sequential); F1o only constrains dataset "B" rows, which still all carry a
`raw_sequence` feature set.

## 7. Verification

| Check | Result |
| --- | --- |
| `scripts/build_static.py` | **7/7 PASS** — SHA unchanged; 133 stations; every field per-station constant; numeric block finite; covers all sequence-index stations; manifest hash matches |
| `scripts/phase17_framework.py` | **20/20 PASS** — registry 24 rows, parity/gates/test-protection intact |
| `scripts/train_fusion.py` (static_only, fusion_L3, hybrid_L3) | `TOTAL 5111s — exit=0`; dev grid 216 rows, 0 test cells; per-family n_params == winner |
| `scripts/phase23_ablation.py` | 144 rows; paired bootstrap CI written; `2381s` |
| `notebooks/23_fusion.ipynb` | **14/14 PASS** (N23-01…N23-14): artefacts, registry contract, test policy, mask identity, tuning, grid, headline, **paired-CI reproduction**, determinism (2 identical fits bit-identical), static-vector integrity, figures, logs, SHA, summary |

## 8. Open items handed on

1. **Phase 24 (station graph & GNN) — gated (D13).** The fusion result sharpens the gate: a
   purely *static* location vector barely helps, so the graph stage must show that
   *relational* structure (upstream/downstream, same-river neighbours) carries signal that
   per-station static geography does not. The evidence is here to test that claim.
2. **Phase 32 (ablation).** L3-07/08 are the registered feature-group ablation rows
   (temporal / +static / static-only), with the matched-capacity CI already computed.
3. **Phase 27 (head-to-head).** Fused deep vs xgboost — xgboost still leads by ~0.02–0.03
   PR-AUC at every horizon on val_2022.
4. **Upstream `up_*` engineered columns** (67–70 % NaN) were fed to the hybrid via the Phase-16
   median-imputer + linear-standardiser path; they did not help. If a future phase revisits the
   hybrid, a missingness-indicator variant is the obvious next probe.

**Next: Phase 24 — Station graph & spatiotemporal GNN (stage 39, GATED by D13).**
