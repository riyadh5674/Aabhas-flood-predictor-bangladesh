# Phase 23 — Multi-Input / Feature Fusion (does knowing *where* help?)
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase23_fusion.md`, `scripts/build_static.py`, `scripts/train_fusion.py`,
> `scripts/phase23_ablation.py`. Previous study: Phase 21–22 (temporal DL).

---

## 0. The 60-second version

The project's whole point is **location-aware** prediction. Phase 21's deep model only saw a
station's own recent days (a sequence). Phase 23 asks: **if we also give the model *where* the
station is (its geography), does it get better?**

Design: keep the Phase-21 temporal encoder, **add a static geography branch**, and measure the
gain **honestly** — with a *matched-capacity* ablation and a bootstrap confidence interval,
not just point estimates.

**Result (the honest answer): knowing *where* adds little.**
- Naively, fusion looks better (+0.002/+0.013/+0.011), **but** the fusion winner is a bigger
  network (58,817 vs 14,497 params) — so that gain is **confounded by capacity**.
- At **matched capacity**, the static branch moves `val_2022` PR-AUC by **+0.002 / +0.005 /
  −0.004** at 24/48/72 h — **not significant** at 24/48 h and a **small significant LOSS** at 72 h.
- Its one real positive: **modest gains on unseen stations (P3) and across years (P2)** — i.e.
  it helps *generalisation*, a bit.

**`static_only` collapses to ~0.51 PR-AUC** — location alone cannot identify a flooded station
once its own dynamics are removed. That confirms the premise: **water level, not latitude,
drives the label.**

---

## 1. Where Phase 23 sits

| Phase | What | Plain words |
|---|---|---|
| 21–22 | Temporal DL (LSTM/GRU/TCN) | Read the last days as a sequence |
| **➜ 23** | **Feature fusion** | **Add geography to the sequence — does it help?** |
| 24–25 | Gated graph/transformer | Does *relational* structure help? |
| 26–33 | Calibrate → select → freeze | Pick champion, read test once |

Closes workflow **stage 38**.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Fusion** | Combining two+ "branches" (here: temporal + static) into one model. |
| **Branch** | One sub-network handling one input type. |
| **Static vector** | A station's fixed geography: lat/lon, elevation, distances, river/basin. |
| **Frequency encoding** | Replacing a category with how often it appears (train-fitted). |
| **MLP** | Multi-layer perceptron (a small feed-forward network). |
| **Capacity** | Model size (number of parameters). Bigger can score better for free. |
| **Matched-capacity ablation** | Compare two models **with the same size** so the gain is real, not capacity. |
| **Paired bootstrap CI** | Resample stations, recompute the **difference**, get a confidence interval. |
| **Confounded** | The apparent effect is actually caused by something else (here, capacity). |
| **Ablation** | Remove one piece to see its contribution. |
| **`static_only`** | The static branch with no sequence — a "location-only" control. |

---

## 3. Design

**Branch 1 — temporal encoder.** **Fixed** to the Phase-21 winner: LSTM, **L\* = 3** (registry
L3-01). Fusion only *adds* the static branch, so any fully-tuned gain over L3-01 can't be
blamed on using a different temporal model.

**Branch 2 — static vector** (`build_static.py` → `static_vectors.csv`, one row per station).
Every field is constant within a station (verified):

| Block | Columns | Encoding | Dim |
|---|---|---|---|
| numeric | `lat`, `lon`, `danger_level`, `elevation_m`, `slope_deg`, `dist_to_coast_km`, `dist_river_km`, `dist_major_river_km`, `river_density_10km`, `n_river_lines_10km` | standardised **on the training fold** | 10 |
| categorical | `river`, `basin`, `division`, `district`, `upazilla` | frequency-encoded **on the training fold** | 5 |

`static_dim = 15` over 133 stations. All standardisers/frequency maps are fit **inside each
training fold**, so the file carries no cross-fold information and no label.

**Model.** `FusionNet` = (LSTM last-step hidden) ⊕ (static MLP on the 15-vector) → concat →
`Linear(·,1)` → sigmoid, `BCEWithLogitsLoss`, seeded **20261008**.

**Variants (all registered — the registry is the contract):**

| Registry | Model | Branch 1 | Branch 2 | Extra |
|---|---|---|---|---|
| **L3-07** | `fusion_L3` | LSTM L*=3 | static | — |
| **L3-08** | `static_only` | — | static | — |
| **L3-09** | `hybrid_L3` | LSTM L*=3 | static | + Phase-12 LINEAR engineered (35) |
| L3-01 | `lstm_L3` | LSTM L*=3 | — | temporal-only (reported, not re-trained) |

Budgets 16/16/12 ≤ 16 (D14 parity). Frozen test never read.

---

## 4. Results — fully-tuned three-way (`val_2022`, all season)

| Model | 24 h | 48 h | 72 h | Params |
|---|---:|---:|---:|---:|
| **L3-07 fusion_L3** | **0.9444** | **0.9385** | **0.9238** | 58,817 |
| L3-09 hybrid_L3 | 0.9275 | 0.9215 | 0.9072 | 16,132 |
| L3-01 lstm_L3 (temporal-only) | 0.9421 | 0.9253 | 0.9129 | 14,497 |
| L3-08 static_only | 0.5068 | 0.5386 | 0.5521 | 1,089 |
| *D7 persistence* | *0.9203* | *0.8954* | *0.8683* | — |
| *xgboost (Phase 20)* | *0.9681* | *0.9549* | *0.9408* | — |

**Read naively**, fusion beats temporal-only by +0.002 / +0.013 / +0.011. **But** the fusion
winner is a **larger encoder** (58,817 vs 14,497 params), so this is **confounded by capacity**.

**Two more observations:**
- The **hybrid** (learned + engineered) is **worse** than pure fusion at every horizon — the
  engineered LINEAR set adds nothing on top of what the encoder already learned.
- **`static_only` is far below both** (and below D7) — location alone is weak.

---

## 5. The honest answer — matched-capacity ablation

Same encoder config and epochs for **fused** and **temporal-only** (only the static branch
toggled), scored on `val_2022`, with a **paired station bootstrap** (100 resamples) on the
difference PR-AUC(fused) − PR-AUC(temporal-only):

| Horizon | Fused | Temporal-only | Gain | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| 24 h | 0.9444 | 0.9420 | **+0.0024** | [−0.0003, +0.0084] | **not significant** |
| 48 h | 0.9385 | 0.9332 | **+0.0053** | [−0.0010, +0.0123] | borderline, not significant |
| 72 h | 0.9238 | 0.9279 | **−0.0041** | [−0.0107, −0.0011] | static branch **hurts** (CI < 0) |

**Two crucial insights:**
1. The matched temporal-only already reaches **0.9420 / 0.9332 / 0.9279** — **better than the
   Phase-21 L3-01 winner** at 48/72 h. So the naive fusion−L3-01 gap is **mostly a tuning
   artefact**: the fusion search simply found a stronger encoder config. It is **not** a
   "fusion effect".
2. At **fixed capacity**, the static vector buys a small, **statistically insignificant** gain
   at 24/48 h, and a small but **significant loss** at 72 h.

### Where the static branch *does* help

| Protocol (mean) | Fused 24/48/72 | Temporal-only 24/48/72 |
|---|---|---|
| P2 rolling | 0.9593 / 0.9485 / 0.9316 | 0.9536 / 0.9434 / 0.9274 |
| P3 spatial CV | 0.9154 / 0.9202 / 0.9171 | 0.9183 / 0.9183 / 0.9111 |

Fused wins **all three P2 horizons** and **48 h/72 h of P3** (the hardest unseen-station cells).

**And `static_only` in P3 collapses to 0.12–0.15 PR-AUC** — with its own dynamics removed, a
station's location cannot tell you if it's flooding. This is exactly the project's premise.

---

## 6. Verdict

**Knowing *where* a station is, given its temporal sequence, adds little.**
- At matched capacity, the static branch changes `val_2022` PR-AUC by **+0.002 / +0.005 /
  −0.004** (significant only at 72 h, and **negative**).
- Its clearest positive effect is a **modest gain on unseen stations (P3) and across years (P2)**.
- The headline fusion−L3-01 gap is mostly a **tuning artefact**, not the geographic input.
- The learned+engineered hybrid is **dominated** by pure fusion.
- All deep variants remain **below the Phase-20 xgboost bar** and **above D7**.

This is a **reportable negative/marginal result** — and reporting it plainly is the point.

---

## 7. Why the phase touched the registry (a completed stage)

Phase 17's registry is the contract ("anything not in the registry is an ad-hoc run"), but
**no row existed for fusion**. So the three variants were **added to the registry**
(L3-07/08/09, phase "23", budget 16, `metrics_out=phase23_metrics.csv`) and the framework
re-run — it passes **20/20** and now holds **24 experiments** (6/5/9/4).

**Plain point:** even an amendment obeys the rulebook.

---

## 8. Verification

| Check | Result |
|---|---|
| `scripts/build_static.py` | **7/7 PASS** — SHA unchanged; 133 stations; every field per-station constant; manifest hash matches |
| `scripts/phase17_framework.py` | **20/20 PASS** — registry 24 rows, parity/gates/test-protection intact |
| `scripts/train_fusion.py` | `TOTAL 5111s — exit=0`; dev grid 216 rows, **0 test cells** |
| `scripts/phase23_ablation.py` | 144 rows; paired bootstrap CI written; 2381 s |
| `notebooks/23_fusion.ipynb` | **14/14 PASS** (incl. paired-CI reproduction, determinism bit-identical) |

Test set: **never read** (frozen to Phase 33).

---

## 9. Problems & solutions

| Problem | Solution |
|---|---|
| Fusion might look better merely because it's bigger | **Matched-capacity ablation** (toggle only the static branch). |
| Point estimates could mislead | **Paired station-bootstrap CI** on the difference. |
| The "gain" could be a tuning artefact | Matched temporal-only re-tuned — revealed the real story. |
| Adding an unregistered variant could bypass the rulebook | Variants added **to the registry**, framework re-verified. |
| Static features could leak across folds | Standardisers/frequency maps fit **inside each training fold**. |
| Location might be over-claimed | Reported as **negative/marginal**, with `static_only` collapse as evidence. |
| Engineered features might double-count | Hybrid tested — it was worse; no double-counting benefit. |

---

## 10. Files referenced

| File | Purpose |
|---|---|
| `reports/phase23_fusion.md` | Full Phase 23 report |
| `scripts/build_static.py` | Builds the static geography vector |
| `scripts/train_fusion.py` | Trains fusion / static-only / hybrid |
| `scripts/phase23_ablation.py` | Matched-capacity gain + paired CI |
| `data/processed/static_vectors.csv` | 133 stations × 15-dim static vector |
| `reports/phase23_metrics.csv` | 216 rows (dev grid) |
| `reports/phase23_ablation.csv` | 144 rows (ablation) |
| `models/phase23_L3-0{7,8,9}.pt` | Saved fusion models |
| `notebooks/23_fusion.ipynb` | 14/14 checks |

---

## 11. Verdict & what it unlocked

- **Location, added to a sequence, adds little** — a small positive on generalisation (P3/P2),
  a small **negative** at 72 h on the pooled cell.
- **`static_only` is very weak** — location alone can't identify flooding; **water level drives
  the label** (the project's premise, confirmed).
- The result **sharpens the Phase-24 gate**: a *purely static* location vector barely helps, so
  the graph stage must show **relational** structure (upstream/downstream neighbours) carries
  signal that per-station geography does not.

**Bottom line:** A clean, honest negative. We tried the intuitive idea ("tell it where it is"),
measured it carefully with matched capacity and confidence intervals, and found it mostly
doesn't help — which is a *result*, not a failure.

---

**Next in the series:** Phase 24–25 — the gated advanced models (station-graph GNN and temporal
Transformer): how the pre-committed gates worked, and the qualified-positive, horizon-dependent
gains they produced.
