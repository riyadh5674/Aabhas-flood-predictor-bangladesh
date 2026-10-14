# Phase 27 — Fair ML vs DL comparison

**Workflow:** `final workflow.md` → **Phase 27 — Fair ML vs DL comparison** · body **stage 42**
(+ stages **43–46, 50–52**).
**Produced by:** `scripts/evaluate_all.py` (seed `20261008`).
**Run:** exit `0` · wall-clock ≈ 13 s (cache re-run; first pass ≈ 9 min including the fresh
P1 re-score of all 24 models) · CPU · AMD64 · Windows · `torch 2.14.1+cpu` · 8 logical cores.
**Immutable input:** `data/processed/experimental_dataset.csv`
SHA-256 `a6748529083fcdcebb8b8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` (unchanged).
**Test period:** never read. `test_rows_read = 0`; every mask is built from `split != test`
columns only, and the frozen 2023+ split is untouched (held for Phase 33).

| Artefact | What it is |
| --- | --- |
| `reports/phase27_metrics.csv` | master comparison table — 1 728 rows = 24 models × 7 dev cells × 3 horizons × 3 seasons, with `val_2022` additionally listed under the **P2** protocol label (identical training fold, so its row duplicates P1) |
| `reports/phase27_p1_rescore.csv` | fresh P1 `val_2022` re-score — 216 rows = 24 models × 3 horizons × 3 seasons, one code path |
| `reports/phase27_operational.csv` | operational block (missed floods, false alarms, event recall, lead time) |
| `reports/phase27_cost.csv` | cost block (parameters, training, inference, hardware) |
| `reports/phase27_generalisation.csv` | generalisation blocks (temporal / spatial / monsoon) — 648 rows |
| `reports/phase27_summary.json` | parity + headline numbers |
| `figures/phase27_ml_vs_dl.png` | headline PR-AUC with station-bootstrap 95 % CI |

---

## 1. Contract and method (one table, one code path, one set of cells)

Phase 27 is the project's **core scientific result** (decision **D14**): ML and DL are compared
only on identical splits, labels, features-as-available, metric code, seeds and tuning budget.
The comparison therefore has to be *earned*, not asserted.

**One metric code path.** Every cell is scored by the single function
`baseline_model.metrics()` — ROC-AUC, PR-AUC, Brier, log-loss, ECE (10 bins), base rate,
best-F1 threshold, and the confusion counts `tp/fp/fn/tn` — plus the three ratios Phase 27 adds
(`fnr`, `far`, `fpr`) and `macro_pr_auc` (per-station mean PR-AUC, ≥ 30 rows/station).

**Two independent routes, one answer.** The seven per-phase dev grids (Phases 16, 20–25) already
scored every registered model on the same shared dev cells. Phase 27 (a) re-scores the **P1
primary cell fresh** for all 24 models through a *single* load → predict → score path, and
(b) consolidates the grids into the master table. The two routes must agree; the parity column
`delta_pr_auc` records the difference. The observed maximum over all 144 P1 `val_2022` rows
(24 models × 3 horizons × 2 protocol labels) is

```
max |fresh re-score PR-AUC − grid PR-AUC| = 0.00e+00   (bit-identical)
```

This is the evidence that the per-phase grids are directly comparable and that the master table
is genuinely *one* table.

**Prediction path.** L1 refits its scikit-learn pipelines on the P1 train fold; L2 loads the
persisted `phase20_L2-*.pkl`; L3/L4 load the persisted `phase2*_L*.pt` checkpoints
(`SeqNet`/`TCN`, `FusionNet`, `GNNNet`/`GNNLSTMNet`, `TransformerNet`) and recompute the
train-fold standardisation. Legacy Phase-21 LSTM/GRU checkpoints stored their recurrent block
under a `rnn.` key prefix that predates the `enc.` rename; `evaluate_all._remap_legacy` maps the
prefix back, leaving the architecture identical. The operating threshold is the **best-F1
threshold on the training fold** for every model (the constraint `climatology` keeps a fixed
0.5), so no model sees evaluation-time threshold information.

**Honest scope note.** Only **P1-trained** checkpoints are persisted — the P2/P3 fold models were
refit per task during their phases and were not saved. Refitting all DL families for P2/P3 would
cost ~5 000 s **per horizon** for P1 alone (see §8) and is infeasible here; re-scoring them would
also be leakage (the P1 model saw the P3 held-out stations). P1 rows therefore come from the
fresh re-score; **P2/P3 rows are consolidated from the per-phase grids** (same metric code, same
cells, provenance tagged `source = phase_grid` in the master table). The `unseen` / `new_network`
test buckets are **test-only** and deferred to Phase 33; the dev-time unseen-location protocol is
**P3** (`phase15_split_design.md`).

---

## 2. Cells

| Cell | Protocol | Definition | Rows | Stations |
| --- | --- | --- | --- | --- |
| `val_2022` (**primary**) | P1 | train `< 2022`, eval 2022, non-holdout stations | 17 538 | 84 |
| `val_2020`, `val_2021` | P2 | rolling-origin yearly refits | — | — |
| `spatial_fold_1..4` | P3 | unseen-location transfer (dev), 4 folds | — | — |

The P2 rolling-origin protocol runs three steps (`val_2020`, `val_2021`, `val_2022`); its
`val_2022` step trains on the identical fold as P1 (`train < 2022` ≡ `train ≤ 2021`), so the
master table carries `val_2022` under both the P1 and P2 labels with identical metrics. Distinct
cells are 7; the natural key is `(protocol, cell, season, model, horizon)`.

Seasonal slices: `all`, `monsoon`, `non-monsoon` (D1). Label base rates on P1 `val_2022`:
`label_24h` 5.09 % (891/17 502), `label_48h` 5.94 %, `label_72h` 6.58 %; monsoon is ~2.5× denser
(`label_24h` 6.84 %) than non-monsoon (2.77 %).

**D14 comparison groups.** Engineered-feature = **L1 ∪ L2**; sequence/DL = **L3 ∪ L4**. Trivial
baselines (`climatology`, `persistence`) and the crippled negative control `static_only` are
excluded from all group means.

---

## 3. Headline — P1 `val_2022` PR-AUC (all season)

Primary metric **PR-AUC** (average precision), 95 % CI from 100 station-bootstrap resamples.

| Level | Model | 24 h | 48 h | 72 h | 24 h CI |
| --- | --- | ---: | ---: | ---: | --- |
| L1 | `climatology` | 0.0509 | 0.0594 | 0.0658 | 0.0269 – 0.0851 |
| L1 | `persistence` (D7) | 0.9203 | 0.8954 | 0.8683 | 0.8163 – 0.9661 |
| L1 | `tree_depth4` | 0.8972 | 0.8902 | 0.8735 | 0.8183 – 0.9415 |
| L1 | `logreg_linear` | 0.9401 | 0.9241 | 0.9053 | 0.8549 – 0.9764 |
| L1 | `logreg_linear_no_up` | 0.9403 | 0.9242 | 0.9052 | 0.8546 – 0.9766 |
| L1 | `logreg_selected` | 0.9398 | 0.9206 | 0.9004 | 0.8562 – 0.9760 |
| **L2** | **`lightgbm`** | **0.9682** | 0.9538 | 0.9384 | 0.9298 – 0.9879 |
| **L2** | **`xgboost`** | **0.9681** | **0.9549** | 0.9408 | 0.9324 – 0.9870 |
| L2 | `random_forest` | 0.9665 | 0.9515 | 0.9389 | 0.9266 – 0.9860 |
| L2 | `xgboost_selected` | 0.9663 | 0.9533 | **0.9422** | 0.9251 – 0.9866 |
| L2 | `gradient_boosting` | 0.9618 | 0.9457 | 0.9338 | 0.9228 – 0.9833 |
| L3 | `lstm_L14` | 0.9311 | 0.9098 | 0.8823 | 0.8394 – 0.9715 |
| L3 | `lstm_L7` | 0.9309 | 0.9164 | 0.8871 | 0.8362 – 0.9720 |
| L3 | `lstm_L3` | 0.9421 | 0.9253 | 0.9129 | 0.8654 – 0.9787 |
| L3 | `lstm_L30` | 0.9388 | 0.9183 | 0.8936 | 0.8599 – 0.9767 |
| L3 | `gru_bestL` | 0.9359 | 0.9263 | 0.9086 | 0.8420 – 0.9772 |
| L3 | `tcn_bestL` | 0.9405 | 0.9224 | 0.9008 | 0.8537 – 0.9763 |
| L3 | `fusion_L3` | 0.9444 | 0.9385 | 0.9238 | 0.8720 – 0.9780 |
| L3 | `hybrid_L3` | 0.9275 | 0.9215 | 0.9072 | 0.8348 – 0.9706 |
| L3 | `static_only` (control) | 0.5068 | 0.5386 | 0.5521 | 0.1508 – 0.7680 |
| L4 | `gcn` | 0.9421 | 0.9353 | 0.9141 | 0.8749 – 0.9744 |
| L4 | `gat` | 0.9408 | 0.9314 | 0.9136 | 0.8530 – 0.9767 |
| L4 | `gnn_lstm` | 0.9424 | 0.9264 | 0.9153 | 0.8691 – 0.9766 |
| L4 | `transformer_bestL` | 0.9437 | 0.9330 | 0.9183 | 0.8552 – 0.9811 |

**Leaders.** 24 h `lightgbm` 0.9682 / `xgboost` 0.9681; 48 h `xgboost` 0.9549; 72 h
`xgboost_selected` 0.9422. The best deep model is `fusion_L3`, at 0.9444 / 0.9385 / 0.9238 —
i.e. **0.016–0.024 PR-AUC** behind the leader and *second* only to the L2 gradient-boosting
family at every horizon. Best non-fusion DL: `transformer_bestL` (0.9437/0.9330/0.9183),
`gnn_lstm` (0.9424/0.9264/0.9153) and `gcn` (0.9421/0.9353/0.9141).

The `static_only` control (no sequence branch) collapses to 0.51–0.55, and `climatology` sits at
the base rate — the two extremes bracket the comparison and confirm the metric behaves.

---

## 4. Predictive metrics

At the best-F1 operating threshold (P1 `val_2022`, 24 h, all season):

| Model | thr | precision | recall | F1 | specificity | FNR | FAR | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lightgbm` | 0.533 | 0.914 | 0.880 | 0.897 | 0.9955 | 0.120 | 0.086 | 0.9981 |
| `xgboost` | 0.500 | 0.917 | 0.888 | 0.902 | 0.9957 | 0.112 | 0.083 | 0.9981 |
| `random_forest` | 0.852 | 0.909 | 0.888 | 0.898 | 0.9952 | 0.112 | 0.091 | 0.9979 |
| `gradient_boosting` | 0.527 | 0.904 | 0.889 | 0.896 | 0.9949 | 0.111 | 0.096 | 0.9971 |
| `fusion_L3` | 0.619 | 0.912 | 0.817 | 0.862 | 0.9958 | 0.183 | 0.088 | 0.9960 |
| `transformer_bestL` | 0.499 | 0.896 | 0.825 | 0.859 | 0.9949 | 0.175 | 0.104 | 0.9965 |
| `gcn` | 0.573 | 0.885 | 0.861 | 0.873 | 0.9940 | 0.139 | 0.115 | 0.9956 |
| `lstm_L3` | 0.904 | 0.899 | 0.820 | 0.858 | 0.9951 | 0.180 | 0.101 | 0.9961 |
| `logreg_linear` | 0.483 | 0.894 | 0.847 | 0.870 | 0.9946 | 0.153 | 0.107 | 0.9954 |
| `tree_depth4` | 0.583 | 0.889 | 0.891 | 0.890 | 0.9940 | 0.109 | 0.111 | 0.9829 |

All 24 × 3 models' full precision/recall/F1/specificity/FNR live in `phase27_metrics.csv`
(and `phase27_p1_rescore.csv`). The pattern is stable: every competitive model is
**specificity-dominated** (the negative class is ~95 %), so ROC-AUC is uniformly high (~0.995)
and not discriminating; the ML/DL difference shows up in **PR-AUC and recall at fixed alarm
budget**. L2 trees reach the highest recall at the highest precision; fusion/GNN models trade
~5–7 points of recall for equal precision.

---

## 5. Probability metrics

Brier and ECE (10 bins) on P1 `val_2022`, 24 h, all season:

| Model | Brier | ECE |
| --- | ---: | ---: |
| `xgboost` | 0.0074 | 0.0018 |
| `lightgbm` | 0.0075 | 0.0022 |
| `gradient_boosting` | 0.0079 | 0.0016 |
| `random_forest` | 0.0125 | 0.0196 |
| `logreg_linear` | 0.0099 | 0.0038 |
| `fusion_L3` | 0.0100 | 0.0022 |
| `transformer_bestL` | 0.0100 | 0.0024 |
| `gcn` | 0.0100 | 0.0027 |
| `lstm_L3` | 0.0134 | 0.0177 |
| `tcn_bestL` | 0.0174 | 0.0262 |
| `static_only` (control) | 0.2072 | 0.3722 |

Probabilities are already well calibrated at this stage (ECE ≤ 0.026 for every competitive
model). **Phase 28 deepens** calibration (reliability diagrams, temperature/isotonic comparison,
ECE decomposition); Phase 27 only records the headline Brier/ECE so the calibration phase starts
from a measured baseline. Probability metrics for all cells are in `phase27_metrics.csv`.

---

## 6. Operational block (stage 46)

Row-level: **missed-flood rows** = FN; **false-alarm rate** (FAR) = FP/(FP+TP); **false-positive
rate** (FPR) = FP/(FP+TN).
Event-level (new this phase): a flood **event** is a maximal run of consecutive calendar days
(gap ≤ 1 d) with `label == 1` per station (143 events on P1 `val_2022` 24 h). An event is
**detected** if any warning day falls in `[start − horizon_days, end]`; **event recall** =
detected/total; **warning lead time** = first warning day minus event start, in days (bounded by
the horizon by construction).

P1 `val_2022`, 24 h, all season:

| Model | missed rows | FAR | FPR | events | detected | event recall | mean lead (d) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost` | 100 | 0.083 | 0.0043 | 143 | 121 | 0.846 | 0.149 |
| `lightgbm` | 107 | 0.086 | 0.0045 | 143 | 118 | 0.825 | 0.153 |
| `gradient_boosting` | 99 | 0.096 | 0.0051 | 143 | 126 | 0.881 | 0.135 |
| `tree_depth4` | 97 | 0.111 | 0.0060 | 143 | 125 | 0.874 | 0.160 |
| `xgboost_selected` | 102 | 0.088 | 0.0046 | 143 | 123 | 0.860 | 0.130 |
| `fusion_L3` | 163 | 0.088 | 0.0042 | 143 | 109 | 0.762 | 0.119 |
| `transformer_bestL` | 156 | 0.104 | 0.0051 | 143 | 111 | 0.776 | 0.099 |
| `gcn` | 124 | 0.115 | 0.0060 | 143 | 118 | 0.825 | 0.136 |
| `persistence` (D7) | 109 | 0.115 | 0.0061 | 143 | 118 | 0.825 | 0.119 |
| `static_only` (control) | 209 | 0.761 | 0.131 | 143 | 94 | 0.657 | 0.819 |

**Read.** The L2 trees give the best operating point: the fewest missed flood rows (single
digits-to-~100) at the lowest alarm cost, and the highest event recall (up to 0.881). The gap to
the deep models is one of **recall**, not calibration: at the same threshold the fusion and
transformer models miss ~55–65 more flood rows per season than `xgboost`. Lead times are
small (mean 0.10–0.18 d, median 0) because the label already encodes the lead window; the metric
is "days of warning before the label window starts", and most correct warnings land on the
window's first day. Per year and per season the full block is in `phase27_operational.csv`
(the monsoon slice carries ~2.5× the events and hence lower per-row FAR at equal recall).

---

## 7. Generalisation blocks (stages 50–52)

Group means (D14 groups, trivial + `static_only` excluded), PR-AUC.

**Stage 50 — temporal (per year).** `val_2020` (P2, 2020), `val_2021` (P2, 2021), `val_2022` (P1):

| Cell | engineered (L1∪L2) 24/48/72 h | sequence/DL (L3∪L4) 24/48/72 h | gap |
| --- | --- | --- | --- |
| `val_2020` | 0.9773 / 0.9656 / 0.9529 | 0.9627 / 0.9476 / 0.9325 | 0.015 / 0.018 / 0.020 |
| `val_2021` | 0.9621 / 0.9524 / 0.9361 | 0.9551 / 0.9411 / 0.9181 | 0.007 / 0.011 / 0.018 |
| `val_2022` | 0.9498 / 0.9354 / 0.9198 | 0.9383 / 0.9254 / 0.9065 | 0.012 / 0.010 / 0.013 |

Skill decays monotonically across the three years for **every** model (all models sit below their
2020 value by 2022); the engineered-vs-sequence gap is small (0.007–0.020) and *grows with lead
time* in 2020/2021, i.e. sequence models degrade faster across the horizon. Per-model yearly
means are in `phase27_generalisation.csv`.

**Stage 51 — spatial (P3, 4 folds; dev-time unseen-location transfer).**

| Group | 24 h | 48 h | 72 h | mean within-model fold SD |
| --- | ---: | ---: | ---: | ---: |
| engineered (L1∪L2) | 0.9528 | 0.9357 | 0.9185 | 0.029 / 0.034 / 0.038 |
| sequence/DL (L3∪L4) | 0.9199 | 0.9072 | 0.8903 | 0.087 / 0.083 / 0.084 |

Unseen-location transfer widens the engineered/M-L gap to ~0.028–0.033 — the **largest gap of any
protocol** — and deep models are ~2.5–3× more sensitive to *which* locations are held out (fold SD
0.08–0.09 vs 0.03). The frozen `test_unseen` / `test_new_network` buckets are deferred to Phase 33.

**Stage 52 — monsoon vs non-monsoon (D1), P1 `val_2022`.**

| Season | engineered 24/48/72 h | sequence/DL 24/48/72 h |
| --- | --- | --- |
| monsoon | 0.9514 / 0.9373 / 0.9224 | 0.9377 / 0.9267 / 0.9084 |
| non-monsoon | 0.9458 / 0.9303 / 0.9128 | 0.9412 / 0.9220 / 0.9014 |

Both groups do slightly *better* in monsoon despite the higher base rate — the wet-season signal
is stronger and denser. The engineered advantage is larger in monsoon (~0.014–0.019) than in the
dry season (~0.005–0.011), where the sequence models close to within ~0.005.

---

## 8. Cost block

| Model | params (kind) | P1 train, 3 horizons (s) | inference (ms / 1 k rows) |
| --- | ---: | ---: | ---: |
| `climatology` | 1 (coeff) | 0.0 | 0.000 |
| `logreg_linear` | 36 (coeff) | 2.6 | 0.129 |
| `tree_depth4` | 23 (nodes) | 5.8 | 0.273 |
| `random_forest` | 62 456 (nodes) | 242.3 | 0.120 |
| `gradient_boosting` | 1 420 (nodes) | 230.7 | 0.071 |
| `xgboost` | 15 002 (nodes) | 27.0 | 0.066 |
| `lightgbm` | 18 300 (nodes) | 19.3 | 0.043 |
| `xgboost_selected` | 23 650 (nodes) | 11.7 | 0.019 |
| `lstm_L3` | 14 497 (w) | 90.2 | 0.021 |
| `tcn_bestL` | 11 329 (w) | 124.3 | 0.030 |
| `fusion_L3` | 58 817 (w) | 170.0 | 0.051 |
| `hybrid_L3` | 16 132 (w) | 85.6 | 0.071 |
| `gcn` | 15 553 (w) | 236.9 | 0.076 |
| `gat` | 15 617 (w) | 224.8 | 0.080 |
| `gnn_lstm` | 18 433 (w) | **2 191.7** | 0.049 |
| `transformer_bestL` | 36 353 (w) | 197.1 | 0.140 |

Hardware identical for all models (CPU · AMD64 · Windows · `torch 2.14.1+cpu`, 8 logical cores).
`params` count is weights for neural nets, coefficients for linear models, and total tree nodes
for ensembles (a cross-family proxy — labelled by `params_kind`). `P1 train` sums the per-phase
`fit_s` for the three horizons of the P1 fold; the fresh re-score wall-clock (load+predict) is the
separate `p1_train_s` column in `phase27_cost.csv`.

**Read.** Trees dominate the accuracy/cost frontier: `xgboost`/`lightgbm` reach the top of the
table at 12–27 s of training and ≤ 0.066 ms/1 k rows inference. `gnn_lstm` costs ~2 192 s —
**~80× `xgboost`** — for no accuracy gain, and `transformer_bestL` 197 s. Total P1 training across
all 24 models is ~5 029 s, of which `gnn_lstm` alone is 44 %.

---

## 9. Uncertainty

Every headline cell carries a **station-level bootstrap 95 % CI** (100 resamples, ≥ 10 stations);
it is present on all 8-cell all-season headline rows in both the master table and the fresh
re-score. On P1 `val_2022` 24 h the CIs are wide relative to the point gaps because there are only
84 stations:

- top L2: `lightgbm` 0.930–0.988, `xgboost` 0.932–0.987, `random_forest` 0.927–0.986;
- best DL: `fusion_L3` 0.872–0.978, `gcn` 0.875–0.974, `transformer_bestL` 0.855–0.981.

The L2 and DL intervals **overlap**, so the ~0.024 PR-AUC point advantage of the trees is *not*
resolved at 95 % on the primary cell — but every competitive model's CI clears `climatology`
(0.027–0.085) by a wide margin. This is reported as-is (no selection on significance); the paired
station bootstrap and the decision-theoretic comparison are Phase 32/33 territory.

---

## 10. Findings (stage 42 verdict)

1. **PR-AUC (primary).** The **L2 gradient-boosting family wins outright on every horizon** on the
   P1 primary cell: `lightgbm` 24 h (0.9682), `xgboost` 48 h (0.9549), `xgboost_selected` 72 h
   (0.9422). The best deep model, `fusion_L3`, is 0.016–0.024 behind and is the *second* family at
   every horizon (best non-fusion DL: `transformer_bestL` / `gnn_lstm` / `gcn`).
2. **ML > DL is real but small and not CI-resolved.** Group means put engineered-feature models
   ahead of sequence/DL models by 0.010–0.013 (P1), 0.007–0.020 (P2 years) and **0.028–0.033 (P3
   unseen locations)**; the advantage is largest exactly where generalisation is hardest.
3. **The gap is about recall, not ranking.** ROC-AUC is ~0.995 for every competitive model; the
   difference appears in recall/FNR at the best-F1 operating point (trees miss ~100 flood rows,
   the fusion/transformer models ~155–165, the GNNs ~124–144) and in PR-AUC.
4. **Trees are dramatically cheaper.** `gnn_lstm` costs ~80× `xgboost`'s training for no gain;
   inference is comparable for all competitive models (≤ 0.14 ms/1 k).
5. **Difficulty grows with lead time for every model** (24 h → 72 h drop 0.020–0.052); the
   gentlest decay is the DL fusion (`hybrid_L3` 0.0203, `fusion_L3` 0.0206) and the steepest is
   `persistence` (0.0520) — consistent with Phase 26.
6. **Probability quality is already high** (ECE ≤ 0.026); Phase 28 remains the calibration phase.
7. **Operationally**, the L2 trees give the best warning system at this threshold (event recall up
   to 0.881, lowest alarm cost); `static_only` (precision 0.24, ECE 0.37) is a clear negative
   control and confirms the sequence branch carries the skill.
8. **Fairness is proven, not assumed.** The fresh single-path re-score reproduces all 144 P1 grid
   cells **bit-identically** (Δ = 0.00e+00); P2/P3 rows are consolidated from the same metric code
   with provenance tagged. No test row was read.

**Recommended default for downstream phases:** carry the **L2 gradient-boosting family**
(`xgboost` / `lightgbm`) as the operational baseline, and carry `fusion_L3` (best DL value-per-
cost) plus `transformer_bestL` as the deep references into calibration (28), thresholds/bands (29),
error analysis (30), ablation (32) and selection (33).

---

## 11. Provenance

- Code: `scripts/evaluate_all.py` (exit `0`), log `scripts/phase27_run.log`.
- Verifying notebook: `notebooks/27_ml_vs_dl.ipynb` (checks N27-01 … N27-12), built by
  `scripts/build_phase27_notebook.py`.
- Ledger / checklist: `final workflow.md` §1 (row 27), §3 (sequencing), Phase 27 heading,
  §4 (rows 42–46, 50–52).
- Immutable input SHA-256 unchanged; test untouched (frozen to Phase 33).
