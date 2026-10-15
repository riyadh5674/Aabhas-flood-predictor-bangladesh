# Phase 22 — Temporal convolutional network (stage 37)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 22 — Temporal convolutional network** → checklist **stage 37** |
| **Driver** | `scripts/train_sequence.py tcn_bestL` (model = `lstm`\|`gru`\|`tcn`; resume-safe; checkpoint after the family) |
| **Run log** | `scripts/phase22_run.log` (`TOTAL 2560 s — exit=0`) |
| **Notebook** | `notebooks/22_tcn.ipynb` (built by `scripts/build_phase22_notebook.py`, executed with the `flood-predictor` kernel, checks N22-01…N22-13) |
| **Outputs** | `reports/phase22_metrics.csv` (**72 rows**) · `reports/phase22_search.csv` (**12 configs**) · `models/phase22_L3-06.pt` |
| **Registry** | `phase17_experiment_registry.csv` row **L3-06** `tcn_bestL`, `test_policy = frozen_read_at_phase33` honoured (§6) |
| **Input (immutable)** | `experimental_dataset.csv` SHA-256 `a6748529083fcdce…`; Dataset B sequences (policy **P19-M1**) |
| **Policy** | P19-M1 keep-all + mask → **13 channels** = concat(12 raw vars, mask) |
| **Window** | **L\* = 3** (carried from Phase 21; the shortest window won the D12 length sweep) |
| **Seed** | 20261008 |

## 1. Candidate and the shared contract

One new family joins the L3 ladder, on **Dataset B** raw sequences at the Phase-21 winning
window, sharing the P19 policy and the Phase-20 dev grid so every number is comparable with
`phase21_metrics.csv` / `phase20_metrics.csv` / `phase16_baseline_metrics.csv`:

| registry | family | window L | model | kind |
| --- | --- | ---: | --- | --- |
| L3-06 | `tcn_bestL` | **3** (= L\*) | TCN | dilated causal convolution |

**Architecture (Phase 22 §body: "dilated causal convolutions → dense → sigmoid").** A stack of
`layers` residual blocks; block *i* applies two dilated **causal** `Conv1d` (kernel 3, dilation
2^i, left-only padding so step `t` never sees `t+1…`) with BatchNorm + ReLU + dropout, added to
a 1×1-conv skip when the channel width changes. The anchor-timestep representation feeds a
`Linear(hidden,1)` → sigmoid logit:

```text
input  X : (N, L, 13)   concat(12 raw variables, timestep mask)
        TCNBlock(x) = ReLU( x + Drop(BN(Conv1(ReLU(BN(Conv0(x)))))) ), dilation 1,2,...
head   : last timestep -> dropout -> Linear(hidden,1) -> sigmoid
```

Everything else is **identical to Phase 21** (D14 fair-comparison parity): same BCE loss with an
unweighted baseline **and** a `pos_weight = n_neg/n_pos` variant in the search space, same early
stopping on validation PR-AUC (patience 4, epoch cap 14), same per-task (train-fold-only)
standardisation with NaN→0 after standardisation, same dev cells, same metric code, same seed.

## 2. Stage 37 — validation-only tuning

Fixed seeded space (96 cells) scored **only** on the selection cell `val_2022`
(season = all), score = **mean PR-AUC over the three horizons** (D2). No early stopping ever
touches the test fold. Budget 12 ≤ registry maximum 20:

| family | configs | space | best mean PR-AUC | tuning fit-time |
| --- | ---: | ---: | ---: | ---: |
| **tcn_bestL** | 12 | 96 | **0.92124** | 1,796 s |

**Selected configuration + imbalance** (full search in `phase22_search.csv`):

| family | selected config | params | weighting |
| --- | --- | ---: | --- |
| **tcn_bestL** | lr 0.003, hidden 32, layers **2**, dropout 0.0, batch 256 | 11,329 | `pos_weight=ratio` |

The validation winner is a 2-level dilated stack (receptive field `1 + (k−1)·Σdilations =
1 + 2·3 = 7 ≥ L`) with class weighting on. Every per-config best-epoch count is recorded in
`phase22_search.csv`; the strongest result at 24 h (`0.94052`) is only ~0.5 pt above the
weakest (0.91893), so the TCN tuning landscape is flat — the family is not hypersensitive.

## 3. Stage 37 — the recurrence-vs-convolution table (24 / 48 / 72 h)

This is the phase's scientific product. `val_2022`, P1, season = all, PR-AUC with
station-bootstrap 95 % CI; LSTM/GRU rows come from `phase21_metrics.csv` (same L\*, same grid),
TCN from `phase22_metrics.csv`:

| model | 24 h | 48 h | 72 h | params | mean grid fit |
| --- | --- | --- | --- | ---: | ---: |
| **LSTM** (L3) | **0.9421** [0.865, 0.979] | **0.9253** [0.847, 0.968] | **0.9129** [0.838, 0.954] | 14,497 | 30 s |
| GRU (L3) | 0.9359 [0.842, 0.977] | 0.9263 [0.844, 0.970] | 0.9086 [0.821, 0.956] | **4,545** | 70 s |
| **TCN** (L3) | 0.9405 [0.854, 0.976] | 0.9224 [0.850, 0.962] | 0.9008 [0.818, 0.947] | 11,329 | 41 s |
| *xgboost* (P20) | *0.9681* | *0.9549* | *0.9408* | — | — |
| *logreg_linear* (P16) | *0.9401* | *0.9241* | *0.9053* | — | — |
| *persistence* (D7, P16) | *0.9203* | *0.8954* | *0.8683* | — | — |

**Answer: does convolution match or beat recurrence, cheaper?**

* **At 24 h it matches recurrence.** TCN 0.9405 vs LSTM 0.9421 vs GRU 0.9359 — the LSTM/TCN gap
  is **0.0016**, far inside overlapping CIs (`[0.854, 0.976]` vs `[0.865, 0.979]`) → *not
  separable*. TCN beats GRU at 24 h by an even smaller, non-separable margin.
* **At 72 h recurrence wins.** TCN is the weakest of the three (0.9008 vs LSTM 0.9129,
  −0.0121). The TCN's 72 h fit early-stopped at **epoch 2** (validation PR-AUC peaked
  immediately) — the dilated convolutions over a 3-day window overfit the hardest horizon
  fastest, so the architecture's edge is concentrated at short lead time.
* **Cheaper?** TCN (11,329 params, 41 s/fit) sits between LSTM (14,497, 30 s) and GRU (4,545,
  70 s). GRU is the genuinely cheap model here; TCN buys no cost advantage over the LSTM it
  ties. TCN's **one clear win is spatial generalisation** (§5).

### Skill over the D7 persistence bar (PR-AUC Δ, val_2022)

| model | Δ24h | Δ48h | Δ72h |
| --- | ---: | ---: | ---: |
| lstm_L3 | +0.0218 | +0.0299 | +0.0446 |
| gru_bestL | +0.0156 | +0.0309 | +0.0404 |
| **tcn_bestL** | +0.0202 | +0.0270 | +0.0326 |

The TCN clears persistence at all three horizons (like every L3 family) and sits ≈0.03 below the
Phase-20 xgboost bar (−0.0276 / −0.0326 / −0.0400 at 24/48/72 h). Same story as Phase 21: on this
tabular+sequence problem the boosted tree on engineered features remains the strongest single
model; the DL ladder's job is to close that gap under fusion (Phase 23) and in the fair
comparison (Phase 27).

## 4. Cross-cell robustness (mean PR-AUC, season = all)

| | P2 rolling origin 24/48/72 h | P3 spatial CV 24/48/72 h |
| --- | --- | --- |
| lstm_L3 | 0.9551 / 0.9362 / 0.9193 | 0.9196 / 0.9055 / 0.8781 |
| gru_bestL | 0.9507 / 0.9376 / 0.9231 | 0.9139 / 0.8972 / 0.8855 |
| **tcn_bestL** | 0.9478 / 0.9371 / 0.9063 | **0.9258** / 0.8949 / 0.8716 |

P2 rolling origin: 24 h **lstm_L3 0.9659 > gru 0.9655 > tcn 0.9617** (val_2020), tcn 0.9412
(val_2021), tcn 0.9405 (val_2022) — the TCN is mid-pack but closest to the LSTM on the primary
year. P3 spatial CV: **the TCN leads at 24 h (0.9258 vs LSTM 0.9196, GRU 0.9139)** and then
falls behind at 72 h — the same short-lead-time character seen on the selection cell. The season
split (val_2022, mean over horizons) barely moves across families (all / monsoon / non-monsoon:
lstm 0.9268/0.9276/0.9260, gru 0.9236/0.9219/0.9310, **tcn 0.9212/0.9230/0.9164**), so no
season-specific degradation is hidden by the `all` cells.

## 5. Model artifact

`models/phase22_L3-06.pt` — the **P1 `train_<2022`** state per horizon
(`{"state": {horizon: state_dict}, "kind":"tcn", "L":3, "params", "headline_metrics"}`), the
model Phase 33 may freeze:

| file | size | contents | n_params |
| --- | ---: | --- | ---: |
| `phase22_L3-06.pt` | 169 KB | TCN (L3) × 3 horizons | 11,329 |

## 6. Metric coverage

`phase22_metrics.csv`: **72 rows** = 1 family × **8 dev tasks** × 3 horizons × 3 seasons.
Cells present: `val_2020/2021/2022` (P2 rolling) + `spatial_fold_1…4` (P3) + the P1 `val_2022`;
`test` rows = **0**. Columns match Phase 21 exactly (same metric code path), with
station-bootstrap CIs and macro PR-AUC on the `season=all` cells.

## 7. Test policy

`test_policy = frozen_read_at_phase33` (registry L3-06). The driver evaluates only the 8 dev
tasks; `phase22_metrics.csv` contains **zero rows whose cell starts with `test`**. The frozen
test period is read exactly once, at Phase 33.

## 8. Determinism, reproducibility & cost

* **Determinism check (workflow-mandated):** `notebooks/22_tcn.ipynb` refits a fixed TCN config
  for 2 epochs twice on identical 4,000-row slices and asserts the parameter tensors are
  **bit-identical** and the val PR-AUCs agree to `<1e-12` (cell N22-08b).
* **Replay check:** all three saved state_dicts reload and reproduce the recorded val-2022
  PR-AUC exactly (cell N22-08a).
* **Reproducibility:** seed 20261008; the run is resumable per family (checkpoint after the
  family — `phase22_*.csv` + the `.pt`).
* **Cost:** `TOTAL 2560 s — exit=0` wall; the honest tuning cost is the per-config `fit_s`
  column in `phase22_search.csv` (Σ = 1,796 s for 12 configs × 3 horizons). Mean grid fit
  ≈ 41 s/cell. TCN is a mid-cost temporal encoder — cheaper per parameter than the LSTM it ties
  and more expensive than the small GRU.

**DoD:** the three-way LSTM/GRU/TCN table exists for all three horizons on the P1 selection cell
and the P2 rolling cells; the TCN is reported with train-derived thresholds, CIs and cost; test
untouched; determinism and replay checks green; budget 12 recorded ≤ 20.
**Next → Phase 23 / stage 38:** feature fusion (temporal encoder + static geographic vector) on
the same splits, so the location-aware gain can be measured against these temporal-only rows.
