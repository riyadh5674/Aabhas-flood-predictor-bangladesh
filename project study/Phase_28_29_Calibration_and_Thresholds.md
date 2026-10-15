# Phase 28–29 — Probability Calibration & Cost-Sensitive Thresholds
### Project study notes · Bangladesh flood-probability forecasting

> Sources: `reports/phase28_calibration.md`, `reports/phase29_thresholds.md`,
> `scripts/phase28_calibration.py`, `scripts/phase29_thresholds.py`,
> `reports/phase28_predictions.npz`. Previous study: Phase 26–27 (multi-horizon & ML-vs-DL).

---

## 0. The 60-second version

Two phases turn a *ranking* model into a *trustworthy warning system*.

**Phase 28 — Calibration.** A model that says "0.8" should be right ~80 % of the time. Most
models are already well calibrated; the plain recurrent nets and `random_forest` are
**over-confident**, especially in monsoon and at long lead time. Fixing this with an
**isotonic** recalibrator (fit on validation, scored **out-of-fold**) drops their error to
near-zero. Rule of thumb: **raw ECE ≤ 0.02 → may proceed uncalibrated (17 of 24 models).**

**Phase 29 — Thresholds & risk bands.** A probability becomes a decision only after a cut-off.
Because **missing a flood is 10× worse than a false alarm**, the chosen thresholds give **very
high recall (96–98 %)** and accept a false alarm about one time in three. Thresholds are
**re-derived per season** (a dry-season threshold carried into monsoon loses recall), and they
become four **risk bands** whose top band is right **90 % of the time**.

Both phases are **pure decision layers** on the Phase-27/28 probabilities — **no retraining, no
test read.**

---

## 1. Where Phase 28–29 sit

| Phase | What | Plain words |
|---|---|---|
| 26–27 | Multi-horizon + ML vs DL | Who wins, fairly |
| **➜ 28** | **Probability calibration** | **Make the probabilities honest** |
| **➜ 29** | **Thresholds & risk bands** | **Turn probabilities into warnings** |
| 30–33 | Error analysis → select → freeze → test | Pick champion, read test once |

Closes workflow **stages 47 (calibration), 48–49 (thresholds/bands)**.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Calibration** | Do predicted probabilities match observed frequencies? |
| **ECE** | Expected Calibration Error — average gap between predicted and observed (lower = better). |
| **Brier score** | Mean squared error of probabilities (lower = better). |
| **Over-confident** | Predicts probabilities too extreme (e.g. says 0.07 when truth is 0.04). |
| **Reliability diagram** | Plot of predicted vs observed frequency, bin by bin. |
| **Platt scaling** | Fit a logistic curve to recalibrate; needs few data points. |
| **Isotonic regression** | Fit a monotone step curve; flexible, needs more data. |
| **Out-of-fold (OOF)** | Calibrate on some folds, score on the held-out fold — honest. |
| **In-sample optimism** | Scoring a calibrator on the same rows it was fit on (too good). |
| **Base rate** | Overall flood frequency (~5 % validation, ~1.4 % test). |
| **Threshold** | Cut-off turning a probability into yes/no. |
| **Cost ratio C_FN:C_FP** | How many false alarms equal one missed flood (here 10:1). |
| **FAR** | False-alarm rate = FP / (FP+TN). |
| **Recall** | Of real floods, how many we warned (TP / (TP+FN)). |
| **Precision** | Of our warnings, how many were real (TP / (TP+FP)). |
| **Period transfer** | Using a threshold derived in one season in another. |
| **Risk band** | low / moderate / high / very high — operational action levels. |
| **Lift** | Band flood rate ÷ base rate (how much more likely than average). |
| **Posterior-odds line** | With perfect calibration, optimum = 1/(1+C_FN/C_FP). |

---

## 3. Phase 28 — calibration

### The method (and why it's honest)

Every model's `val_2022` probabilities come from the **same Phase-27 predictor** — the numbers
already scored in the head-to-head. Two recalibrators are tried:
- **Platt (sigmoid):** `σ(a·logit(p)+b)`.
- **Isotonic:** a monotone step function.

**Crucial honesty step:** fitting and scoring a calibrator on the *same* rows is optimistic —
isotonic drives in-sample ECE to **0.000 by construction**. So the headline number is
**cross-fitted (out-of-fold)**: the **84 `val_2022` stations** are split into **5 folds**, each
fold calibrated by a calibrator fit on the other four.

### Raw calibration (all-season, P1 `val_2022`) — worst horizon

| Level | Model | raw ECE (worst) | cleared ≤0.02? |
|---|---|---:|:---:|
| L1 | `logreg_linear` | 0.0115 | ✓ |
| L2 | **`xgboost`** | **0.0031** | ✓ |
| L2 | `lightgbm` | 0.0045 | ✓ |
| L2 | `random_forest` | 0.0428 | ✗ |
| L3 | `lstm_L3` | 0.1025 | ✗ |
| L3 | `gru_bestL` | 0.0556 | ✗ |
| L3 | `tcn_bestL` | 0.0789 | ✗ |
| L3 | `fusion_L3` | 0.0066 | ✓ |
| L4 | `transformer_bestL` | 0.0081 | ✓ |
| L4 | `gcn` | 0.0053 | ✓ |
| L4 | `gnn_lstm` | 0.0066 | ✓ |
| L3 | `static_only` (control) | 0.3992 | ✗ |

**Findings:**
1. **Most models are already well calibrated** — **17 / 24** clear the 0.02 bar; median raw ECE
   is **0.0067**. Gradient boosting, the transformer/GNNs and the fusion models are near-perfect
   (ECE ≤ 0.007).
2. **Miscalibration is concentrated and structural.** The failures are the crippled control
   `static_only` (0.37), the **plain recurrent nets** (0.04–0.10), **`random_forest`** (tree-vote
   over-confidence), and the no-skill baselines (base-rate gap).
3. **The failure grows with lead time** (`lstm_L3` 0.018 → 0.037 → 0.072) and is **worst in
   monsoon** (`lstm_L3` 0.1025 monsoon vs 0.0314 dry). Direction is **over-confidence**.

### Recalibration result (cross-fitted)

| Model | raw ECE | best OOF ECE | method | ΔBrier |
|---|---:|---:|:---:|---:|
| `static_only` | 0.3722 | 0.0190 | isotonic | +0.1614 |
| `tcn_bestL` (72 h) | 0.0574 | 0.0026 | isotonic | +0.0073 |
| `random_forest` (72 h) | 0.0353 | 0.0023 | isotonic | +0.0048 |
| `lstm_L3` (72 h) | 0.0722 | 0.0038 | isotonic | +0.0035 |

**Isotonic wins for every miscalibrated model.** For the 17 already-good models recalibration is
**optional** (ΔBrier within ±2×10⁻⁴). `static_only` stays at 0.033 even after recalibration — a
crippled model with no discrimination cannot be calibrated into usefulness.

### The season effect (D1)

- The recurrent over-confidence is a **monsoon** effect — the wet-season base rate is ~2.5× the
  dry-season one and plain RNNs don't track it.
- `climatology` **flips**: fine in monsoon (0.0275), bad in the dry season (0.0793) — the
  train↔2022 base-rate gap by season (the Phase-16 threshold-transfer lesson, restated for
  probabilities).
- Recalibration is **period-robust**: fit *within* each season, every skill model reaches OOF
  ECE ≤ 0.008.

### Decision (DoD)

- **17 / 24 may proceed uncalibrated** (raw ECE ≤ 0.02 at every horizon/season).
- **7 must be recalibrated** (isotonic) before bands: `random_forest`, `lstm_L3/L7`, `gru_bestL`,
  `tcn_bestL`, plus the control/no-skill rows.
- **Carry forward:** uncalibrated probabilities for `xgboost`/`lightgbm`/`logreg_linear`/
  `fusion_L3`/`transformer_bestL`/`gcn`/`gnn_lstm`; isotonic-recalibrated for the recurrent / RF
  family.

---

## 4. Phase 29 — thresholds & risk bands

### The cost statement

For early warning, **a missed flood is far worse than a false alarm**. The objective is a
row-level cost:

```
cost = C_FN · FN + C_FP · FP ,   C_FP = 1 ,  primary C_FN = 10  (sensitivity 5 and 20)
```

reported **per 1,000 rows**. At `10:1`, one missed-flood row costs as much as ten false alarms.
A model-free anchor: with perfect calibration the optimum is `t* = C_FP/(C_FP+C_FN)` =
**0.0909** at 10:1, independent of period.

### Chosen thresholds (P1 `val_2022`, all-season, 10:1)

| Horizon | Leader | t | Recall | Precision | FAR | missed rows | cost/1k |
|---|---|---:|---:|---:|---:|---:|---:|
| 24 h | **lightgbm** | 0.0605 | 0.980 | 0.746 | 0.254 | 18 | 27.25 |
| 48 h | **xgboost** | 0.0524 | 0.982 | 0.667 | 0.333 | 19 | 40.06 |
| 72 h | **xgboost_selected** | 0.0614 | 0.966 | 0.630 | 0.370 | 38 | 59.57 |

**Read:** the cost-optimal trade is **very high recall (96–98 %), modest precision (63–75 %),
FAR 25–37 %** — the system is tuned to **catch floods**, accepting a false alarm about one time
in three. That is *intended* for early warning.

The crippled control `static_only` needs a threshold ~0.69 and still misses ~18 % of floods.

The threshold moves **×5 between 5:1 and 20:1** → **the cost ratio is a policy input, not a free
parameter**; the stakeholder must fix it before bands are frozen. The **F1-optimal** threshold is
~7× higher and precision-oriented — **not** the right operating point when FN ≫ FP.

### Period-specific thresholds (the Phase-16 lesson)

| Horizon | Leader | monsoon base | t monsoon | dry base | t dry |
|---|---|---:|---:|---:|---:|
| 24 h | lightgbm | 6.84 % | 0.0689 | 2.77 % | 0.0412 |
| 48 h | xgboost | 7.98 % | 0.0524 | 3.21 % | 0.0645 |
| 72 h | xgboost_selected | 8.81 % | 0.0697 | 3.58 % | 0.1657 |

**Carrying a dry-season threshold into the monsoon costs recall** (leaders lose ≤ 3.2 pts, weaker
models 4–9 pts, the control **17 pts**). The reverse direction (wet→dry) is safe — it merely
alarms more. **Conclusion: re-derive thresholds per period.** The fixed posterior line (0.0909)
is a good fallback at 48/72 h (≤ 11 % cost penalty) but is beaten by re-derivation at 24 h.

### Risk bands (from the cost-optimal thresholds, **not** quartiles)

`watch = ½·t_warn`, `warning = t_warn`, `high =` first score with validation precision ≥ 0.90.

| 24 h lightgbm band | range | freq | observed flood rate | lift | action |
|---|---|---:|---:|---:|---|
| low | < 0.030 | 92.7 % | 0.07 % | 0.01× | routine monitoring |
| moderate | 0.030–0.061 | 0.6 % | 6.4 % | 1.3× | watch |
| high | 0.061–0.434 | 1.6 % | 24.1 % | 4.7× | warning |
| **very high** | ≥ 0.434 | 5.1 % | **90.1 %** | **17.7×** | high-risk warning |

**The very-high band is near-perfect (90 % observed flood rate, 13–18× lift).** ~90 % of days
sit in the low band (< 0.2 % flood rate). The **quartile default is rejected**: probabilities
cluster near 0, so quartile cuts put 50–75 % of rows in zero-flood bands and dump essentially
all floods into the top quartile.

### Frozen decision (before any test read)

| Horizon | Leader | monsoon | non-monsoon | all-season |
|---|---|---:|---:|---:|
| 24 h | lightgbm | 0.0689 | 0.0412 | 0.0605 |
| 48 h | xgboost | 0.0524 | 0.0645 | 0.0524 |
| 72 h | xgboost_selected | 0.0697 | 0.1657 | 0.0614 |

- **Operating cost fixed at 10:1**; re-derive per period; never carry dry→monsoon.
- **Bands frozen** (cost-derived four-level scheme); Phase 33 applies them unchanged.
- Miscalibrated models sit on their Phase-28 calibrator or per-period thresholds.
- **Test untouched** (frozen to Phase 33).

---

## 5. Problems & solutions

| Problem | Solution |
|---|---|
| A calibrator scored on its own fit looks perfect | **Cross-fitted (OOF)** on 84-station folds; in-sample only as an optimistic bound. |
| Some models over-confident, others fine | Per-model decision (17 clear the 0.02 bar; 7 recalibrate). |
| Threshold could be a free knob | **Cost ratio is a policy input**; sensitivity at 5:1/10:1/20:1 shown. |
| A single threshold could fail across seasons | **Re-derive per period**; quantify the dry→monsoon recall loss. |
| Quartile bands would be misleading | Cost-derived bands; quartile default explicitly rejected. |
| Calibration might be assumed to fix transfer | Tested — it helps some models, not uniformly. |
| Probabilities could be built on a new model path | Reuse the **same Phase-27 predictor**; no retraining. |
| Test could creep in | Thresholds/bands derived on `val_2022` only; `test_rows_read = 0`. |

---

## 6. Files referenced

| File | Purpose |
|---|---|
| `reports/phase28_calibration.md` | Calibration report |
| `reports/phase29_thresholds.md` | Thresholds & bands report |
| `reports/phase28_calibration.csv` | Calibrated-vs-raw table (216 rows) |
| `reports/phase28_predictions.npz` | Cached P1 `val_2022` probabilities (single path) |
| `reports/phase29_threshold_sweep.csv` | Full sweep (14,040 rows) |
| `reports/phase29_chosen_thresholds.csv` | Cost-optimal + F1-optimal per cell |
| `reports/phase29_risk_bands.csv` | Four-level bands |
| `reports/phase29_summary.json` | Frozen thresholds/bands, `test_rows_read = 0` |
| `figures/phase28_reliability.png`, `figures/phase29_threshold_cost.png` | Figures |

---

## 7. Verdict

- **Calibration:** most models were already honest (17/24); the over-confidence is concentrated
  in **plain recurrent nets, `random_forest`, and in monsoon at long lead time**, and **isotonic
  OOF recalibration fixes it** to ECE ≤ 0.008.
- **Thresholds:** tuned to **catch floods** (recall 96–98 %, FAR ~1-in-3) under a **10:1** cost;
  the ratio is a **policy input** that must be fixed by the stakeholder.
- **Period matters:** thresholds are **re-derived per season**; the dry→monsoon transfer is the
  dangerous direction.
- **Bands are near-perfect at the top:** the very-high band is right ~90 % of the time (13–18×
  lift).
- **No retraining, no test read** — a pure decision layer on the fair Phase-27/28 evidence.

**Bottom line:** The project now has **honest probabilities** and a **defensible warning rule**
whose top band is trustworthy — everything derived on validation, with the cost of a mistake
stated out loud.

---

**Next in the series:** Phase 30 — Error analysis (where and why the best models still fail),
31–32 (robustness & ablation), and Phase 33 — Selection + the **single frozen test read**.
