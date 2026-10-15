# Phase 26–27 — Multi-Horizon Strategy & the Fair ML-vs-DL Comparison
### Project study notes · Bangladesh flood-probability forecasting

> Sources: `reports/phase26_multihorizon.md`, `reports/phase27_ml_vs_dl.md`,
> `scripts/phase26_matrix.py`, `scripts/evaluate_all.py`. Previous study: Phase 24–25 (gated
> models). These phases assemble every surviving model into one table and answer the project's
> **core scientific question**: does deep learning beat classical ML — fairly?

---

## 0. The 60-second version

**Phase 26** gathers all **24 models** (L1–L4) into a single **multi-horizon matrix** and asks:
*how does difficulty grow from a 1-day to a 3-day forecast, and does it grow differently for
deep models than for engineered-feature models?*

**Phase 27** is the **core scientific result**: a **fair head-to-head** between ML and DL — one
table, one code path, identical splits/labels/metrics/seeds/budgets.

**The headline (stated honestly):**
- **Gradient boosting wins outright on every horizon, every protocol.** Best engineered model
  beats best deep model by ~0.016–0.024 PR-AUC on the primary cell, and by ~0.028–0.033 on
  unseen stations.
- **BUT the ML>DL gap is not resolved at 95 %** — the confidence intervals overlap on the
  primary cell (only 84 stations).
- **A real DL advantage exists** only in *lead-time decay* on unseen stations: DL models hold
  their skill across horizons better than engineered models there (the `fusion_L3` model is
  even **flat** from 24 h → 72 h).
- **Trees are dramatically cheaper** — `gnn_lstm` costs ~**80× XGBoost's** training for no gain.

**The science:** a more complex model with **no absolute gain** on the primary cell is not a
better model. The DL contribution lives in **lead-time behaviour on unseen stations**, not in
the headline.

---

## 1. Where Phase 26–27 sit

| Phase | What | Plain words |
|---|---|---|
| 20–25 | Trained all candidate families | Built the field of models |
| **➜ 26** | **Multi-horizon strategy** | **Assemble the full horizon matrix** |
| **➜ 27** | **Fair ML vs DL comparison** | **One fair table: who really wins?** |
| 28–29 | Calibration & thresholds | Make probabilities/decisions right |
| 30–33 | Error analysis → select → freeze → test | Pick the champion, read test once |

Closes stages **41, 59–61** (Phase 26) and **42–46, 50–52** (Phase 27).

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Multi-horizon matrix** | A table of every model × every horizon × every protocol × season. |
| **Interaction** | Does the ML-vs-DL difference *change* with lead time? |
| **Engineered-feature group** | L1 ∪ L2 — models using hand-built features. |
| **Sequence/DL group** | L3 ∪ L4 — models reading raw day-sequences. |
| **Lead-time decay** | How much PR-AUC drops from 24 h → 72 h. |
| **Head-to-head** | A direct, fair comparison under identical conditions. |
| **Parity check** | Re-scoring the same cells two ways must give identical numbers. |
| **Bit-identical** | Exactly equal (Δ = 0.00), not "close". |
| **Event recall** | Fraction of flood *events* (runs of flood days) that got a warning. |
| **FAR / FNR / FPR** | False-alarm rate / false-negative rate / false-positive rate. |
| **Lead time** | Days of warning before the flood window starts. |
| **Cost block** | Parameters, training seconds, inference milliseconds. |
| **Negative control** | A deliberately crippled model (`static_only`) to prove the test is meaningful. |
| **Provenance tag** | A label saying where a row's number came from (`phase_grid` vs fresh re-score). |

---

## 3. Phase 26 — the multi-horizon matrix

**Integration only** — no new training. The seven per-phase dev metric files share one schema;
Phase 26 concatenates them, tags each model with its registry level (L1–L4), and asserts **no
test cell** is present.

**Cells:** P1 (`val_2022`) · P2 (rolling mean over 2020/2021/2022) · P3 (mean over 4 spatial
folds), each × {24, 48, 72 h}, season = all (monsoon/non-monsoon available in the grid).

### The headline matrix (P1 `val_2022`, PR-AUC)

| Family | Model | 24 h | 48 h | 72 h | Δ 24→72 |
|---|---|---:|---:|---:|---:|
| **XGBoost (L2)** | `xgboost` | **0.9681** | **0.9549** | 0.9408 | −0.0273 |
| LightGBM (L2) | `lightgbm` | 0.9682 | 0.9538 | 0.9384 | −0.0298 |
| Random Forest (L2) | `random_forest` | 0.9665 | 0.9515 | 0.9389 | −0.0277 |
| XGB-SELECTED (L2) | `xgboost_selected` | 0.9663 | 0.9533 | **0.9422** | −0.0241 |
| **fusion DL (L3)** | `fusion_L3` | 0.9444 | **0.9385** | **0.9238** | **−0.0206** |
| Transformer (L4) | `transformer_bestL` | 0.9437 | 0.9330 | 0.9183 | −0.0254 |
| GNN (L4) | `gcn` | 0.9421 | 0.9353 | 0.9141 | −0.0279 |
| **LSTM (L3)** | `lstm_L3` | 0.9421 | 0.9253 | 0.9129 | −0.0292 |
| TCN (L3) | `tcn_bestL` | 0.9405 | 0.9224 | 0.9008 | −0.0397 |
| *logreg (L1)* | `logreg_linear` | 0.9401 | 0.9241 | 0.9053 | −0.0348 |
| *D7 persistence (L1)* | `persistence` | 0.9203 | 0.8954 | 0.8683 | −0.0520 |

**Two facts stand out:**
1. **Gradient boosting owns `val_2022` at every horizon** (best engineered beats best DL by
   ~+0.024 / +0.016 / +0.018 at 24/48/72 h).
2. **The DL fusion models degrade the least** across lead time (`fusion_L3` −0.021 vs D7 −0.052),
   narrowing the ML gap most at 48 h.

### How difficulty grows with lead time

**Every competitive model loses skill with lead time; nothing improves.** The loss is **not**
ordered by model class:
- The shallow **D7 bar loses the most** (−0.052).
- Within L3, the **longer-window** LSTMs lose the most (−0.044 to −0.049), vs −0.029 for the
  3-day window.
- The **strongest** models of *both* classes lose least (xgboost −0.027, fusion −0.021).

**Insight:** difficulty growth is a property of **how much signal is still in the features**,
not of the architecture. The models that use the features best also last longest.

### The interaction — is it different for DL?

| Protocol | Group | Mean Δ (24→72) |
|---|---|---:|
| P1 | engineered (L1/L2) | −0.0300 |
| P1 | sequence/DL (L3/L4) | −0.0259 |
| P2 | engineered | −0.0268 |
| P2 | sequence/DL | −0.0273 |
| **P3** | engineered | **−0.0343** |
| **P3** | sequence/DL | **−0.0254** |

**Answer:**
- On the **temporal** protocols (P1/P2), the interaction is **small** — the two groups decay at
  nearly the same rate. The apparent P1 signal is an artefact of group composition (best-per-group
  inverts it: `fusion_L3` −0.021 vs `xgboost` −0.027).
- On **unseen stations (P3)** the interaction is **real and consistent**: engineered models lose
  **more** skill with lead time (−0.034) than sequence/DL (−0.025), and `fusion_L3` is **flat**
  across horizons (actually **+0.002**). GNN families are similarly flat (`gat` −0.017).

**But this does NOT mean DL beats ML on P3** — `xgboost` still leads every P3 horizon by ~+0.04.
It only means **DL's lead-time decay is gentler there.**

`static_only` (the crippled control) is **anti-correlated** with lead time (Δ +0.045) — reported
for completeness, excluded from group means.

### Horizon deliverables (stages 59/60/61)

| Horizon | Winner (P1) | Best DL | Persistence |
|---|---|---|---|
| 24 h | `lightgbm` 0.9682 | `fusion_L3` 0.9444 | 0.9203 |
| 48 h | `xgboost` 0.9549 | `fusion_L3` 0.9385 | 0.8954 |
| 72 h | `xgboost_selected` 0.9422 | `fusion_L3` 0.9238 | 0.8683 |

**Notable:** at **72 h**, the **10-feature** `xgboost_selected` overtakes the full-feature
`xgboost` on P1 — the only horizon where feature-set size flips the ranking.

---

## 4. Phase 27 — the fair ML-vs-DL comparison (core result)

**The contract (D14):** compare ML and DL only on identical splits, labels, features-as-available,
metric code, seeds and tuning budget. The comparison must be **earned, not asserted**.

### Two independent routes, one answer (fairness proof)

- Route 1: re-score the **P1 primary cell fresh** for all 24 models through a **single**
  load → predict → score path.
- Route 2: consolidate the per-phase grids into the master table.

**Result:**
```
max |fresh re-score PR-AUC − grid PR-AUC| = 0.00e+00   (bit-identical)
```

This proves the per-phase grids are **directly comparable** and the master table is genuinely
**one** table.

**Honest scope note:** only **P1-trained** checkpoints are persisted. P2/P3 rows are
**consolidated from the per-phase grids** (same metric code, same cells, provenance tagged
`source = phase_grid`). Refitting all DL families for P2/P3 would be infeasible *and* would be
leakage (a P1 model saw the P3 held-out stations).

### Headline — P1 `val_2022` PR-AUC (95 % CI, 100 station-bootstrap resamples)

| Level | Model | 24 h | 48 h | 72 h |
|---|---|---:|---:|---:|
| **L2** | **`lightgbm`** | **0.9682** | 0.9538 | 0.9384 |
| **L2** | **`xgboost`** | 0.9681 | **0.9549** | 0.9408 |
| L2 | `xgboost_selected` | 0.9663 | 0.9533 | **0.9422** |
| L3 | `fusion_L3` | 0.9444 | 0.9385 | 0.9238 |
| L4 | `transformer_bestL` | 0.9437 | 0.9330 | 0.9183 |
| L4 | `gnn_lstm` | 0.9424 | 0.9264 | 0.9153 |
| L4 | `gcn` | 0.9421 | 0.9353 | 0.9141 |
| L1 | `logreg_linear` | 0.9401 | 0.9241 | 0.9053 |
| L1 | `persistence` (D7) | 0.9203 | 0.8954 | 0.8683 |
| L3 | `static_only` (control) | 0.5068 | 0.5386 | 0.5521 |

**Leaders:** 24 h `lightgbm`; 48 h `xgboost`; 72 h `xgboost_selected`. The best deep model
(`fusion_L3`) is **0.016–0.024 behind** and is the **second** family at every horizon.

### Predictive metrics (best-F1 threshold, P1 24 h)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| `xgboost` | 0.917 | 0.888 | 0.902 | 0.9981 |
| `lightgbm` | 0.914 | 0.880 | 0.897 | 0.9981 |
| `fusion_L3` | 0.912 | 0.817 | 0.862 | 0.9960 |
| `lstm_L3` | 0.899 | 0.820 | 0.858 | 0.9961 |

**Pattern:** every competitive model is **specificity-dominated** (negatives ~95 %), so
**ROC-AUC is uniformly high (~0.995) and not discriminating**. The ML/DL difference shows up in
**PR-AUC and recall at fixed alarm budget.** L2 trees reach the highest recall at the highest
precision; fusion/GNN models trade ~5–7 points of recall for equal precision.

### Probability metrics (Brier, ECE)

| Model | Brier | ECE |
|---|---:|---:|
| `xgboost` | 0.0074 | 0.0018 |
| `lightgbm` | 0.0075 | 0.0022 |
| `gradient_boosting` | 0.0079 | 0.0016 |
| `fusion_L3` | 0.0100 | 0.0022 |
| `transformer_bestL` | 0.0100 | 0.0024 |
| `static_only` (control) | 0.2072 | 0.3722 |

**Probabilities already well calibrated** (ECE ≤ 0.026 for every competitive model). Phase 28
will deepen this.

### Operational block (the real-world view)

P1 24 h: **missed-flood rows**, **event recall** (fraction of flood *events* warned), lead time:

| Model | Missed rows | Event recall | FAR |
|---|---:|---:|---:|
| `gradient_boosting` | 99 | **0.881** | 0.096 |
| `tree_depth4` | 97 | 0.874 | 0.111 |
| `xgboost_selected` | 102 | 0.860 | 0.088 |
| `xgboost` | 100 | 0.846 | 0.083 |
| `gcn` | 124 | 0.825 | 0.115 |
| `persistence` (D7) | 109 | 0.825 | 0.115 |
| `fusion_L3` | 163 | 0.762 | 0.088 |
| `static_only` (control) | 209 | 0.657 | 0.761 |

**Read:** L2 trees give the best operating point — fewest missed rows at the lowest alarm cost,
highest event recall (up to 0.881). The gap to deep models is one of **recall**, not calibration:
fusion/transformer models miss ~55–65 more flood rows per season than `xgboost`.

### Generalisation blocks (stages 50–52)

**Temporal (per year):**

| Cell | engineered 24/48/72 | sequence/DL 24/48/72 | gap |
|---|---|---|---|
| `val_2020` | 0.9773 / 0.9656 / 0.9529 | 0.9627 / 0.9476 / 0.9325 | 0.015 / 0.018 / 0.020 |
| `val_2021` | 0.9621 / 0.9524 / 0.9361 | 0.9551 / 0.9411 / 0.9181 | 0.007 / 0.011 / 0.018 |
| `val_2022` | 0.9498 / 0.9354 / 0.9198 | 0.9383 / 0.9254 / 0.9065 | 0.012 / 0.010 / 0.013 |

**Spatial (P3, unseen locations):**

| Group | 24 h | 48 h | 72 h | within-model fold SD |
|---|---:|---:|---:|---:|
| engineered (L1∪L2) | 0.9528 | 0.9357 | 0.9185 | 0.029 / 0.034 / 0.038 |
| sequence/DL (L3∪L4) | 0.9199 | 0.9072 | 0.8903 | 0.087 / 0.083 / 0.084 |

Unseen-location transfer **widens** the gap to ~0.028–0.033 — the **largest of any protocol** —
and deep models are **~2.5–3× more sensitive to which locations are held out** (fold SD 0.08–0.09
vs 0.03).

**Monsoon vs non-monsoon (P1):** both groups do slightly *better* in monsoon despite the higher
base rate. The engineered advantage is larger in monsoon (~0.014–0.019) than dry season
(~0.005–0.011), where sequence models close to within ~0.005.

### Cost block

| Model | Params | P1 train, 3 horizons (s) | Inference (ms/1k) |
|---|---:|---:|---:|
| `lightgbm` | 18,300 (nodes) | 19.3 | 0.043 |
| `xgboost` | 15,002 (nodes) | 27.0 | 0.066 |
| `logreg_linear` | 36 (coeff) | 2.6 | 0.129 |
| `lstm_L3` | 14,497 (w) | 90.2 | 0.021 |
| `transformer_bestL` | 36,353 (w) | 197.1 | 0.140 |
| `gcn` | 15,553 (w) | 236.9 | 0.076 |
| **`gnn_lstm`** | 18,433 (w) | **2,191.7** | 0.049 |

**Read:** trees dominate the accuracy/cost frontier. `gnn_lstm` costs **~80× XGBoost** for no
accuracy gain. Total P1 training across all 24 models ≈ 5,029 s, of which `gnn_lstm` alone is 44 %.

### Uncertainty (reported as-is)

On P1 24 h the CIs are wide (only 84 stations):
- top L2: `lightgbm` [0.930, 0.988], `xgboost` [0.932, 0.987];
- best DL: `fusion_L3` [0.872, 0.978], `transformer_bestL` [0.855, 0.981].

The L2 and DL intervals **overlap**, so the ~0.024 point advantage of the trees is **not resolved
at 95 % on the primary cell** — but every competitive model's CI clears `climatology`
(0.027–0.085) by a wide margin. Reported as-is; no selection on significance.

---

## 5. Verdict (core scientific result)

1. **PR-AUC (primary):** the **L2 gradient-boosting family wins outright on every horizon** on the
   P1 cell. `fusion_L3` is 0.016–0.024 behind and is the **second** family at every horizon.
2. **ML > DL is real but small and not CI-resolved** — group means put engineered models ahead by
   0.010–0.013 (P1), 0.007–0.020 (P2 years) and **0.028–0.033 (P3 unseen locations)**; largest
   exactly where generalisation is hardest.
3. **The gap is about recall, not ranking** — ROC-AUC is ~0.995 for everyone.
4. **Trees are dramatically cheaper** — `gnn_lstm` ~80× `xgboost` for no gain.
5. **Difficulty grows with lead time for every model**; the gentlest decay is the DL fusion.
6. **Probability quality is already high** (ECE ≤ 0.026).
7. **Operationally**, the L2 trees give the best warning system at this threshold.
8. **Fairness is proven, not assumed** — bit-identical re-score; no test row read.

**Recommended default for downstream phases:** carry the **L2 gradient-boosting family**
(`xgboost`/`lightgbm`) as the operational baseline, plus `fusion_L3` and `transformer_bestL` as
the deep references into calibration (28), thresholds (29), error analysis (30), ablation (32) and
selection (33).

---

## 6. Problems & solutions

| Problem | Solution |
|---|---|
| ML-vs-DL claims could rest on protocol differences | One table, one code path, D14 parity; **bit-identical** re-score. |
| P2/P3 DL checkpoints weren't saved | Consolidated from per-phase grids with provenance tags; refitting would leak. |
| Point gaps could be noise | Station-bootstrap 95 % CIs on every headline cell; **overlaps reported**. |
| A complex model could be over-claimed | Report **no absolute gain** on the primary cell — the DL win is lead-time only. |
| Metric could be misleading (accuracy/ROC) | PR-AUC primary; ROC-AUC shown to be non-discriminating. |
| Cost could be ignored | Full cost block (params, train s, inference ms). |
| Training set could leak via saved models | Only P1 checkpoints persisted; re-scoring P3 with a P1 model would leak. |
| Test could creep in | Asserted **0 test cells**; frozen to Phase 33. |

---

## 7. Files referenced

| File | Purpose |
|---|---|
| `reports/phase26_multihorizon.md` | Multi-horizon matrix report |
| `reports/phase27_ml_vs_dl.md` | Fair ML-vs-DL comparison report |
| `reports/phase26_matrix.csv` (1,728 rows) | Consolidated dev grid (24 models) |
| `reports/phase27_metrics.csv` (1,728 rows) | Master comparison table |
| `reports/phase27_operational.csv` | Missed floods, event recall, lead time |
| `reports/phase27_cost.csv` | Params, training, inference, hardware |
| `reports/phase27_generalisation.csv` | Temporal / spatial / monsoon blocks |
| `figures/phase27_ml_vs_dl.png` | Headline PR-AUC with CIs |
| `scripts/evaluate_all.py` | The single scoring code path |

---

## 8. Verdict & what it unlocked

- **Gradient boosting is the operational champion** — it wins every horizon on every protocol at
  a fraction of the cost.
- **The ML>DL advantage is real but small and not CI-resolved** on the primary cell.
- **The DL value is specific:** gentler **lead-time decay on unseen stations** (`fusion_L3` flat
  24 h → 72 h) — the strongest DL signal found, and the reason the DL section is worth carrying.
- **Fairness is proven** — bit-identical re-score, tagged provenance, zero test rows.
- **Three model families carried forward:** L2 boosting (baseline), `fusion_L3` (best DL
  value-per-cost), `transformer_bestL` (deep reference).

**Bottom line:** The project's central question — "does deep learning beat classical ML?" — is
answered **honestly**: not on the headline, but yes on one specific axis (holding skill at longer
lead times on unseen stations), at 80× the cost. That nuance is the scientific product.

---

**Next in the series:** Phase 28–29 — Calibration & cost-sensitive thresholds (making the
probabilities trustworthy and the warning/decision rule match the FFWC cost asymmetry). Then
Phase 30 (error analysis), 31–32 (robustness & ablation), and 33 (selection + the single frozen
test read).
