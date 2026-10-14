# Phase 29 — Risk levels & threshold optimization

**Workflow:** `final workflow.md` → **Phase 29 — Risk levels & threshold optimization** · body
**stages 48, 49**.
**Produced by:** `scripts/phase29_thresholds.py` (seed `20261008`).
**Run:** exit `0` · wall-clock ≈ 4 s · CPU · AMD64 · Windows · `torch 2.14.1+cpu` · 8 logical cores.
**Immutable input:** `data/processed/experimental_dataset.csv`
SHA-256 `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` (unchanged).
**Test period:** never read. `test_rows_read = 0`; thresholds and bands are derived on the P1
primary validation cell `val_2022` only.

**Input.** The exact P1 `val_2022` probabilities Phase 28 produced
(`reports/phase28_predictions.npz`, 24 models × 3 horizons); **no retraining and no new model
path** — Phase 29 is a pure decision layer on Phase 27/28 evidence. Row counts: 17,502 (24 h),
17,447 (48 h), 17,089 (72 h).

| Artefact | What it is |
| --- | --- |
| `reports/phase29_threshold_sweep.csv` | full sweep — 14,040 rows = 24 models × 3 horizons × 3 periods × 65 thresholds |
| `reports/phase29_chosen_thresholds.csv` | cost-optimal + F1-optimal threshold per cell (216 rows) |
| `reports/phase29_transfer.csv` | period-transfer penalty (837 rows; raw + calibrated) |
| `reports/phase29_risk_bands.csv` | low/moderate/high/very-high — cost-derived vs quartile (72 rows) |
| `reports/phase29_pr_auc.csv` | independent PR-AUC per model/horizon (leader selection) |
| `reports/phase29_summary.json` | cost statement, frozen thresholds/bands, `test_rows_read = 0` |
| `figures/phase29_threshold_cost.png` | expected cost vs threshold, one panel per horizon |
| `figures/phase29_bands_transfer.png` | P–R curves, dry→wet recall loss, band flood rates |

---

## 1. Cost statement (explicit, stage 49)

For early warning, **a missed flood is far worse than a false alarm**. The operating objective
is stated as a row-level cost

```
cost = C_FN · FN + C_FP · FP ,   C_FP = 1  (primary C_FN = 10 ; sensitivity 5 and 20)
```

reported **per 1 000 rows**. `C_FN = 10` means one missed-flood row costs as much as ten
false-alarm rows. (This is a conservative row-level proxy: at the *event* level a miss is worse
still, because one missed event can be dozens of rows — Phase 27's event-recall block.)
Thresholds are swept over **0.01…0.60** (65 points; the spec's 0.05–0.60 range is extended
downward because the ~5 % base rate puts the optimum at 0.04–0.17), and the reported
"chosen" value is the **exact argmin over the observed scores** in that range.

A second, model-free anchor: with **perfectly calibrated** probabilities the cost-optimal
decision is a **fixed posterior value** `t* = C_FP/(C_FP+C_FN)` — `0.0476` (20:1), `0.0909`
(10:1), `0.1667` (5:1) — independent of the period. Section 4 tests how close the empirical
optima come to it.

---

## 2. Chosen thresholds — leaders and families

P1 `val_2022`, all-season, `C_FN = 10`. Base rate 5.09 % (24 h) / 5.94 % (48 h) / 6.58 % (72 h).

| horizon | leader | t (10:1) | recall | precision | FAR | alarm rate | missed rows | cost / 1k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 h | **lightgbm** | **0.0605** | 0.980 | 0.746 | 0.254 | 6.7 % | 18 | **27.25** |
| 48 h | **xgboost** | **0.0524** | 0.982 | 0.667 | 0.333 | 8.7 % | 19 | **40.06** |
| 72 h | **xgboost_selected** | **0.0614** | 0.966 | 0.630 | 0.370 | 10.1 % | 38 | **59.57** |

The leaders were selected independently from the cached probabilities by PR-AUC (lightgbm 0.9682
/ xgboost 0.9549 / xgboost_selected 0.9422) — the same ranking Phase 27 found. The cost-optimal
trade is **very high recall (96–98 %), modest precision (63–75 %), FAR 25–37 %**: at `FN:FP = 10:1`
the system is tuned to *catch floods*, accepting a false alarm about one time in three. That is
intended for early warning.

Representative cells (all-season, 10:1):

| model (level) | 24 h t / recall / missed | 48 h t / recall / missed | 72 h t / recall / missed |
| --- | --- | --- | --- |
| logreg_linear (L1) | 0.177 / 0.961 / 35 | 0.130 / 0.959 / 42 | 0.135 / 0.940 / 67 |
| random_forest (L2) | 0.449 / 0.983 / 15 | 0.423 / 0.966 / 35 | 0.353 / 0.959 / 46 |
| fusion_L3 (L3) | 0.072 / 0.978 / 20 | 0.083 / 0.967 / 34 | 0.076 / 0.951 / 55 |
| transformer_bestL (L4) | 0.079 / 0.969 / 28 | 0.064 / 0.977 / 24 | 0.062 / 0.947 / 60 |
| gnn_lstm (L4) | 0.033 / 0.978 / 20 | 0.077 / 0.949 / 52 | 0.038 / 0.959 / 46 |
| static_only (control) | 0.689 / 0.689 / **277** | 0.681 / 0.698 / **310** | 0.685 / 0.717 / **318** |

The crippled control `static_only` needs a threshold ~0.69 and still misses ~18 % of floods —
even with the operating point chosen for it, it cannot reach the field. Full 24-model tables are
in `phase29_chosen_thresholds.csv`.

**Cost-ratio sensitivity** (all-season thresholds):

| horizon | leader | t (5:1) | t (10:1) | t (20:1) | t (F1-optimal) | best F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 24 h | lightgbm | 0.1244 | 0.0605 | 0.0247 | 0.4349 | 0.904 |
| 48 h | xgboost | 0.1205 | 0.0524 | 0.0524 | 0.4557 | 0.883 |
| 72 h | xgboost_selected | 0.1904 | 0.0614 | 0.0495 | 0.4585 | 0.865 |

The threshold moves systematically with the cost assumption (×5 between 5:1 and 20:1), so **the
cost ratio is a policy input, not a free parameter** — it must be fixed by the stakeholder before
the bands are frozen. The F1-optimal ("balanced") threshold is ~7× higher and precision-oriented
(F1 ≈ 0.87–0.90); it is **not** the right operating point for FN ≫ FP early warning and is listed
only for context.

---

## 3. Period-specific thresholds — the Phase-16 lesson (stage 48)

The base rate is ~2.5× denser in monsoon, so the optimum moves:

| horizon | leader | monsoon base | t monsoon | non-monsoon base | t non-monsoon |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 h | lightgbm | 6.84 % | 0.0689 | 2.77 % | 0.0412 |
| 48 h | xgboost | 7.98 % | 0.0524 | 3.21 % | 0.0645 |
| 72 h | xgboost_selected | 8.81 % | 0.0697 | 3.58 % | **0.1657** |

**Carrying a dry-season threshold into the monsoon costs recall.** Deriving on non-monsoon and
applying to monsoon (`phase29_transfer.csv`, raw probabilities):

| model | horizon | t derived (dry) | monsoon-own t | carried recall | monsoon-own recall | **recall lost** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 24 h | 0.0412 | 0.0689 | 0.982 | 0.978 | −0.004 |
| xgboost | 48 h | 0.0645 | 0.0524 | 0.977 | 0.984 | +0.006 |
| xgboost_selected | 72 h | 0.1657 | 0.0697 | 0.928 | 0.961 | **+0.032** |
| gradient_boosting | 48 h | 0.111 | 0.028 | 0.935 | 0.981 | +0.046 |
| tree_depth4 | 72 h | 0.245 | 0.090 | 0.877 | 0.939 | +0.061 |
| lstm_L30 | 24 h | 0.253 | 0.040 | 0.912 | 0.982 | +0.070 |
| persistence | 48 h | 0.383 | 0.170 | 0.842 | 0.928 | +0.086 |
| static_only (control) | 48 h | 0.719 | 0.651 | 0.559 | 0.725 | **+0.166** |

The **direction is exactly the Phase-16 finding** (a threshold derived under a low base rate is
too strict when the base rate rises, and floods are missed): for the leaders the penalty is small
(≤ 3.2 pts) because their cost curves are flat near a very inclusive optimum, but for weaker and
plain-tree/recurrent models it reaches **4–9 pts**, and for the no-skill control **17 pts**. The
reverse direction (wet threshold → dry season) is safe (recall *gains*: e.g. 72 h `all→non-monsoon`
−0.031) — it merely over-alarms. **Conclusion: re-derive thresholds per period** (what this phase
freezes), because a single dry-season threshold is the dangerous artefact.

---

## 4. The "calibrated probability" alternative

Because a calibrated probability has a period-invariant meaning, the workflow allows carrying
**one** threshold instead of re-deriving per period. Two tests:

**(a) Fixed posterior-odds threshold.** With perfectly calibrated probabilities the optimum is
`t* = 0.0909` at 10:1 for *every* period. Applying that single value:

| horizon | period | per-period optimum cost/1k | at fixed t = 0.0909 | penalty |
| --- | --- | ---: | ---: | ---: |
| 24 h | monsoon | 35.92 | 41.14 | +15 % |
| 24 h | non-monsoon | 12.87 | 16.06 | +25 % |
| 48 h | monsoon | 51.67 | 54.51 | +5 % |
| 48 h | non-monsoon | 22.13 | 24.56 | +11 % |
| 72 h | monsoon | 77.27 | 80.64 | +4 % |
| 72 h | non-monsoon | 33.05 | 34.56 | +5 % |

The fixed threshold is a good **period-invariant fallback** at 48/72 h (≤ 11 % cost penalty) but
is beaten by re-derivation at 24 h, where the empirical optimum sits well below `t*`.

**(b) Cross-fitted isotonic** (deterministic 5-fold station-hash folds) for the models Phase 28
flagged as miscalibrated. Calibration changes the single-threshold transfer, but **not
uniformly**: `random_forest` 48 h `all→monsoon` improves (raw recall-loss 0.019 → **0.000**),
while `lstm_L3` 24 h `all→non-monsoon` worsens (0.024 → 0.053). A calibrated probability makes
the *number* interpretable; it does **not** by itself remove the period shift, because the
empirical cut still drifts with residual miscalibration and finite-sample noise.

**Conclusion:** use **period-specific thresholds** (the primary rule) and treat the calibrated
posterior line (`1/(1+C_FN/C_FP)`) as a documented fallback for periods with too little data to
re-derive.

---

## 5. Risk bands (stage 49)

Bands are derived from the cost-optimal thresholds (**not** the quartile default):
`watch = ½·t_warn`, `warning = t_warn` (the 10:1 optimum), `high =` the first score with validation
precision ≥ 0.90. Four levels, each with its validation frequency, observed flood rate and
operational meaning. P1 `val_2022`, all-season:

| horizon / leader | band | range | freq | observed flood rate | lift | action |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **24 h lightgbm** | low | < 0.030 | 92.7 % | 0.07 % | 0.01× | routine monitoring |
| | moderate | 0.030–0.061 | 0.6 % | 6.4 % | 1.3× | watch |
| | high | 0.061–0.434 | 1.6 % | 24.1 % | 4.7× | warning |
| | very high | ≥ 0.434 | 5.1 % | **90.1 %** | 17.7× | high-risk warning |
| **48 h xgboost** | low | < 0.026 | 90.2 % | 0.12 % | 0.02× | routine |
| | moderate | 0.026–0.052 | 1.1 % | 0.00 % | 0.0× | watch |
| | high | 0.052–0.530 | 3.1 % | 23.6 % | 4.0× | warning |
| | very high | ≥ 0.530 | 5.7 % | **90.0 %** | 15.2× | high-risk warning |
| **72 h xgboost_selected** | low | < 0.031 | 88.1 % | 0.16 % | 0.02× | routine |
| | moderate | 0.031–0.061 | 1.9 % | 4.4 % | 0.7× | watch |
| | high | 0.061–0.552 | 4.0 % | 22.6 % | 3.4× | warning |
| | very high | ≥ 0.552 | 6.0 % | **90.0 %** | 13.7× | high-risk warning |

The **very-high band is near-perfect (90 % observed flood rate, 13–18× lift)**, the high band is
~23 %, and ~90 % of days fall in the low band at < 0.2 % flood rate. The **quartile default is
rejected**: these probabilities are concentrated near 0, so the quartile cuts (0.0001–0.0006 at
24 h) put **50–75 % of rows in zero-flood bands** and dump essentially all floods into the
top quartile (~20–26 % rate) — exactly the workflow's warning that the quartile split is a starting
point, not a default. (The one 48 h moderate cell holds 1.1 % of rows with 0 floods — a
finite-sample noise band; the operationally essential separation, high & very-high ≫ low, holds
for all three leaders.)

---

## 6. Frozen decision (before any test read)

* **Operating cost:** `C_FN : C_FP = 10 : 1` (row-level). **Re-derive thresholds per period**;
  never carry a single dry-season threshold into the monsoon.
* **Provisional thresholds** (frozen for the Phase-33 selection; per period, 10:1):

| horizon | leader | monsoon | non-monsoon | all-season |
| --- | --- | ---: | ---: | ---: |
| 24 h | lightgbm | 0.0689 | 0.0412 | 0.0605 |
| 48 h | xgboost | 0.0524 | 0.0645 | 0.0524 |
| 72 h | xgboost_selected | 0.0697 | 0.1657 | 0.0614 |

* **Risk bands:** the cost-derived four-level scheme in §5 (low / moderate / high / very high
  = routine / watch / warning / high-risk). Bands and thresholds are **frozen here**; any test
  evaluation (Phase 33) applies them unchanged.
* **Miscalibration caveat:** for `random_forest` and the plain recurrent nets
  (`lstm_L3/L7`, `gru_bestL`, `tcn_bestL`) the decision should sit on the Phase-28
  validation-fitted calibrator, or on re-derived per-period thresholds — both are tabulated.
* Test untouched (frozen to Phase 33).

---

## 7. Provenance

- Code: `scripts/phase29_thresholds.py` (exit `0`), log `scripts/phase29_run.log`.
- Input: `reports/phase28_predictions.npz` (Phase 28) — single probability path, no retraining.
- Verifying notebook: `notebooks/29_thresholds.ipynb` (checks N29-01 … N29-12), built by
  `scripts/build_phase29_notebook.py`.
- Ledger / checklist: `final workflow.md` §1 (row 29), §3 (sequencing), Phase 29 heading,
  §4 (stages 48, 49).
- Immutable input SHA-256 unchanged; test untouched.
