# Phase 28 — Probability calibration

**Workflow:** `final workflow.md` → **Phase 28 — Probability calibration** · body **stage 47**.
**Produced by:** `scripts/phase28_calibration.py` (seed `20261008`).
**Run:** exit `0` · wall-clock ≈ 29 s on the cached pass (the one-off prediction pass is ≈ 60 s;
per-model 0.8–5.8 s) · CPU · AMD64 · Windows · `torch 2.14.1+cpu` · 8 logical cores.
**Immutable input:** `data/processed/experimental_dataset.csv`
SHA-256 `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` (unchanged).
**Test period:** never read. `test_rows_read = 0`; calibration is fitted on `val_2022` only
(the P1 primary cell), and `val_2022` is itself carved out of `split != test`.

| Artefact | What it is |
| --- | --- |
| `reports/phase28_calibration.csv` | calibrated-vs-raw table — 216 rows = 24 models × 3 horizons × 3 seasons |
| `reports/phase28_summary.json` | tolerance, cleared/needs-recalibration lists, medians |
| `reports/phase28_predictions.npz` | cached P1 `val_2022` probabilities + labels/stations (5.4 MB) |
| `figures/phase28_reliability.png` | reliability diagrams — raw vs isotonic (OOF), per horizon |

---

## 1. Question and method

Phase 16 found ECE ≤ 0.007 for the linear baselines, and Phase 27 recorded headline Brier/ECE for
every model — but never asked whether the **deep** models are as trustworthy, whether
miscalibration is a **lead-time** or **season** effect, or whether recalibration is needed
**before a risk band can be defined** (Phase 29). Phase 28 answers exactly that.

**One predictor, reused.** Every model's `val_2022` probabilities come from the **same** Phase-27
predictor (`evaluate_all.predict_one`: refit L1 pipelines, load L2 `.pkl`, load L3/L4 `.pt`,
recompute train-fold standardisation), so the probability vectors are the ones the Phase-27
comparison already scored. Brier and ECE are computed with the **shared** metric code
(`baseline_model.ece`, 10 equal-width bins).

**Two recalibrators.**
* **Platt (sigmoid):** logistic regression on the logit of the model probability, `σ(a·logit(p)+b)`.
* **Isotonic:** monotone piecewise-constant regression, `out_of_bounds="clip"`, `y∈[0,1]`.

**Fitted on validation only, scored out-of-fold.** Fitting and scoring a calibrator on the same
rows is optimistic — isotonic in particular drives in-sample ECE to ~0 by construction. The
headline "calibrated" number is therefore **cross-fitted**: the **84 `val_2022` stations** are
split into **5 folds** and each fold is calibrated by a calibrator fitted on the other four
(**OOF**). In-sample numbers are reported only as the optimistic bound.

**Decision rule.** *A model may proceed uncalibrated* if its **raw ECE ≤ 0.02** — a "good
calibration" bar consistent with the risk-band use case. Two variants are reported:
per-horizon on all-season data (the headline bar) and the **stricter** bar across **all three
seasons *and* horizons**.

---

## 2. Raw calibration — all-season P1 `val_2022`

ECE (10 bins), raw probabilities. Bold = fails the 0.02 bar on the worst horizon.

| Level | Model | 24 h | 48 h | 72 h | worst (all seasons) | best OOF ECE | cleared |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| L1 | `climatology` | 0.0450 | 0.0466 | 0.0494 | 0.0793 | 0.0003 | ✗ (no skill) |
| L1 | `persistence` | 0.0116 | 0.0141 | 0.0180 | 0.0225 | 0.0044 | ✓* |
| L1 | `logreg_linear` | 0.0038 | 0.0051 | 0.0069 | 0.0115 | 0.0048 | ✓ |
| L1 | `logreg_linear_no_up` | 0.0040 | 0.0050 | 0.0070 | 0.0114 | 0.0046 | ✓ |
| L1 | `logreg_selected` | 0.0052 | 0.0070 | 0.0101 | 0.0142 | 0.0082 | ✓ |
| L1 | `tree_depth4` | 0.0032 | 0.0028 | 0.0044 | 0.0077 | 0.0049 | ✓ |
| L2 | `gradient_boosting` | 0.0016 | 0.0043 | 0.0046 | 0.0058 | 0.0045 | ✓ |
| L2 | `xgboost` | 0.0018 | 0.0024 | 0.0025 | 0.0031 | 0.0031 | ✓ |
| L2 | `lightgbm` | 0.0022 | 0.0030 | 0.0033 | 0.0045 | 0.0036 | ✓ |
| L2 | `xgboost_selected` | 0.0022 | 0.0024 | 0.0027 | 0.0037 | 0.0035 | ✓ |
| L2 | `random_forest` | **0.0196** | **0.0263** | **0.0353** | **0.0428** | 0.0032 | ✗ |
| L3 | `lstm_L3` | 0.0177 | **0.0369** | **0.0722** | **0.1025** | 0.0043 | ✗ |
| L3 | `lstm_L7` | **0.0201** | **0.0475** | **0.0450** | **0.0658** | 0.0065 | ✗ |
| L3 | `lstm_L14` | 0.0031 | 0.0038 | 0.0065 | 0.0096 | 0.0060 | ✓ |
| L3 | `lstm_L30` | 0.0035 | 0.0069 | 0.0087 | 0.0131 | 0.0055 | ✓ |
| L3 | `gru_bestL` | **0.0250** | **0.0412** | **0.0367** | **0.0556** | 0.0052 | ✗ |
| L3 | `tcn_bestL` | **0.0262** | **0.0396** | **0.0574** | **0.0789** | 0.0040 | ✗ |
| L3 | `fusion_L3` | 0.0022 | 0.0024 | 0.0045 | 0.0066 | 0.0054 | ✓ |
| L3 | `hybrid_L3` | 0.0022 | 0.0029 | 0.0035 | 0.0043 | 0.0049 | ✓ |
| L3 | `static_only` (control) | **0.3722** | **0.3742** | **0.3686** | **0.3992** | 0.0333 | ✗ (control) |
| L4 | `gcn` | 0.0027 | 0.0030 | 0.0035 | 0.0053 | 0.0042 | ✓ |
| L4 | `gat` | 0.0045 | 0.0042 | 0.0034 | 0.0063 | 0.0062 | ✓ |
| L4 | `gnn_lstm` | 0.0031 | 0.0034 | 0.0050 | 0.0066 | 0.0046 | ✓ |
| L4 | `transformer_bestL` | 0.0024 | 0.0040 | 0.0057 | 0.0081 | 0.0039 | ✓ |

`✓*` = clears the all-season bar but not the stricter all-seasons bar (`persistence` monsoon
ECE 0.0225). Family medians (all-season, worst horizon): **L1** 0.0069, **L2** 0.0027,
**L3** 0.0201, **L4** 0.0035.

**Findings.**
1. **Most models are already well calibrated.** 17 / 24 clear the 0.02 all-season bar; median raw
   ECE is **0.0067**. Gradient boosting (L2), the transformer/GNNs (L4) and the fusion models are
   essentially perfect (ECE ≤ 0.007).
2. **Miscalibration is concentrated and structural.** The seven that fail are the **crippled
   control** `static_only` (0.37), the **plain recurrent nets** `lstm_L3`/`lstm_L7`/`gru_bestL`/
   `tcn_bestL` (0.041–0.102), **`random_forest`** (0.043 — tree-vote over-confidence), and the
   **no-skill baselines** (`climatology`, `persistence`) whose ECE is just the train/eval base-rate
   gap. The miscalibration **grows with lead time** (e.g. `lstm_L3` 0.018 → 0.037 → 0.072) and is
   **worst in monsoon** (`lstm_L3` monsoon 0.1025 vs non-monsoon 0.0314).
3. **Direction is over-confidence.** For every failing model `mean_pred − base_rate > 0` at every
   horizon, e.g. `lstm_L3` +0.072 (72 h, all), +0.103 (monsoon 72 h); `random_forest` +0.035;
   `static_only` +0.37. The well-calibrated models sit within ±0.005 of the base rate.

---

## 3. Recalibration — Platt vs isotonic (cross-fitted)

For each model and horizon the out-of-fold ECE of both recalibrators is compared and the lower
-Brier method is reported. For the seven miscalibrated models recalibration is decisive:

| Model | horizon | raw ECE | best OOF ECE | σ method | raw Brier | best OOF Brier | ΔBrier |
| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| `static_only` | 24 h | 0.3722 | 0.0190 | isotonic | 0.2072 | 0.0458 | +0.1614 |
| `tcn_bestL` | 72 h | 0.0574 | 0.0026 | isotonic | 0.0290 | 0.0171 | +0.0073 |
| `random_forest` | 72 h | 0.0353 | 0.0023 | isotonic | 0.0207 | 0.0135 | +0.0048 |
| `gru_bestL` | 48 h | 0.0412 | 0.0052 | isotonic | 0.0232 | 0.0164 | +0.0044 |
| `lstm_L7` | 48 h | 0.0475 | 0.0030 | isotonic | 0.0323 | 0.0186 | +0.0040 |
| `lstm_L3` | 72 h | 0.0722 | 0.0038 | isotonic | 0.0330 | 0.0160 | +0.0035 |
| `persistence` | 72 h | 0.0180 | 0.0043 | isotonic | 0.0214 | 0.0188 | +0.0014 |

Isotonic wins on Brier for every miscalibrated model; Platt is competitive only when the
distortion is close to a monotone sigmoid. **For the 17 already-calibrated models recalibration is
approximately neutral** — the change in Brier is within ±2 × 10⁻⁴ and OOF ECE can move either way
by ~10⁻³ (extra cross-fitting variance), so it is *optional* for them.

The in-sample/out-of-fold contrast is the reason the OOF numbers are used: fitting isotonic and
scoring on the **same** rows drives ECE to **0.000** by construction, while the cross-fitted ECE is
0.002–0.019 — the honest estimate of what a deployed calibrator achieves.

---

## 4. Period shift (D1): monsoon vs non-monsoon

Per-season **worst-horizon** ECE, raw vs recalibrated (calibrator fitted within the season):

| Model | monsoon raw | monsoon OOF | non-monsoon raw | non-monsoon OOF |
| --- | ---: | ---: | ---: | ---: |
| `static_only` | 0.3561 | 0.0333 | 0.3992 | 0.0333 |
| `lstm_L3` | 0.1025 | 0.0043 | 0.0314 | 0.0034 |
| `climatology` | 0.0275 | 0.0002 | 0.0793 | 0.0005 |
| `tcn_bestL` | 0.0789 | 0.0040 | 0.0285 | 0.0025 |
| `lstm_L7` | 0.0658 | 0.0065 | 0.0229 | 0.0033 |
| `gru_bestL` | 0.0556 | 0.0042 | 0.0221 | 0.0040 |
| `random_forest` | 0.0428 | 0.0032 | 0.0252 | 0.0027 |
| `persistence` | 0.0225 | 0.0044 | 0.0135 | 0.0044 |
| `xgboost` | 0.0031 | 0.0031 | 0.0026 | 0.0026 |

Two period effects:
* **The recurrent-net over-confidence is a monsoon effect** (`lstm_L3` 0.1025 in monsoon vs 0.0314
  off-season) — the wet-season base rate is ~2.5× the dry-season one, and the plain RNNs do not
  track it.
* **`climatology` flips**: it is nearly calibrated in monsoon (0.0275) and badly off in the dry
  season (0.0793), purely because the train and 2022 base rates differ by season — the Phase-16
  "threshold/period transfer" lesson, restated for probabilities.

Recalibration is period-robust: fitted **within** each season, **all** models reach OOF ECE ≤ 0.008
except the no-skill control `static_only` (0.033).

---

## 5. Decision — which models may proceed uncalibrated (DoD)

**Tolerance:** raw ECE ≤ 0.02.

| Decision | Count | Models |
| --- | ---: | --- |
| **May proceed uncalibrated** (all-season, per horizon) | **17 / 24** | `logreg_linear`, `logreg_linear_no_up`, `logreg_selected`, `tree_depth4`, `gradient_boosting`, `xgboost`, `lightgbm`, `xgboost_selected`, `lstm_L14`, `lstm_L30`, `fusion_L3`, `hybrid_L3`, `gcn`, `gat`, `gnn_lstm`, `transformer_bestL`, `persistence`* |
| **Recalibrate before Phase 29** | **7 / 24** | `random_forest`, `lstm_L3`, `lstm_L7`, `gru_bestL`, `tcn_bestL`, `static_only` (control), `climatology` (no skill) |
| **Must recalibrate / not use for bands** (stricter, all seasons+horizons) | 8 / 24 | above 7 **plus `persistence`** (monsoon ECE 0.0225) |

**Statements.**
* The **L1 linear models, the L2 gradient-boosting family, the L4 transformer/GNNs and the fusion
  models may proceed to Phase 29 uncalibrated** — their raw probabilities are already within the
  0.02 bar at every horizon and season.
* **`random_forest`, the plain recurrent nets (`lstm_L3/L7`, `gru_bestL`, `tcn_bestL`) and the
  no-skill baselines must carry a validation-fitted calibrator** (isotonic preferred) before any
  risk band is defined; after recalibration every **skill** model reaches OOF ECE ≤ 0.02 (the
  crippled control `static_only` stays at 0.033 — it has no discrimination to calibrate).
* **`static_only`** is the crippled negative control and should not be deployed; it is retained
  only as a comparison row.
* Because the failure mode is **period-dependent** (monsoon), Phase 29 should use
  **period-specific calibration/thresholds** for the recurrent and ensemble-tree models, per the
  Phase-16 lesson — the recalibrated numbers in `phase28_calibration.csv` are already tabulated
  per season.

**Recommended downstream default:** carry the uncalibrated probabilities for
`xgboost`/`lightgbm`/`logreg_linear`/`fusion_L3`/`transformer_bestL`/`gcn`/`gnn_lstm`; carry
`isotonic`-recalibrated probabilities for `random_forest`/`lstm_*`/`gru_bestL`/`tcn_bestL`.

---

## 6. Figures

`figures/phase28_reliability.png` — reliability diagrams (predicted vs observed, 10 bins),
top row raw, bottom row isotonic(OOF), one column per horizon, for a representative set
(`logreg_linear`, `xgboost`, `fusion_L3`, `transformer_bestL`, `lstm_L3`, `static_only`) with the
diagonal of perfect calibration. The top row shows the recurrent/control curves swinging off the
diagonal at long lead time; the bottom row shows them snapped back onto it.

---

## 7. Provenance

- Code: `scripts/phase28_calibration.py` (exit `0`), log `scripts/phase28_run.log`.
- Reuses the Phase-27 predictor `evaluate_all.predict_one` (same one-code-path guarantee as
  Phase 27); Brier/ECE from `baseline_model`.
- Verifying notebook: `notebooks/28_calibration.ipynb` (checks N28-01 … N28-12),
  built by `scripts/build_phase28_notebook.py`.
- Ledger / checklist: `final workflow.md` §1 (row 28), §3 (sequencing), Phase 28 heading,
  §4 (row 47).
- Cached probabilities: `reports/phase28_predictions.npz` (delete to recompute, ≈ 60 s).
- Immutable input SHA-256 unchanged; test untouched (frozen to Phase 33).
