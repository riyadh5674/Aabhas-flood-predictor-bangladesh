# Phase 16 — Baseline Models (and the Class-Imbalance Reality Check)
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase16_baseline.md` (312 lines), `scripts/baseline_model.py`,
> `data/processed/class_imbalance.json`, `notebooks/08_baseline_models.ipynb`.
> Companion study phases: Phase 12–13 (features/leakage), 14–15 (dataset/splits).
> Simple English, real numbers, what/why/how + problems & solutions.

---

## 0. The 60-second version

Before trying clever machine learning, do the **sensible, boring thing first**: build the
simplest possible models and see how good they are. Why? Because if a simple model already
gets 0.92 and your fancy model gets 0.94, you have to be honest that the fancy model bought
you very little.

Phase 16 built **28 baseline models** across 3 horizons and 3 protocols:
- **climatology** (just guess the average flood rate) — the "do nothing" floor.
- **persistence (D7)** — the one-feature bar: *"today's water level as % of danger is the
  best single guess for tomorrow."* Everything must beat this, not climatology.
- **logistic regressions** on LINEAR (35) and SELECTED (10) features.
- **a shallow decision tree** (depth 4) for non-linearity.

Three big results:
1. **The bar is persistence (D7), and it's high** — logreg beats it by only **+2–4 %**.
2. **Spatial generalisation (new stations) is the weak point** — the linear model is
   *below* D7 in 3 of 4 unseen-station folds.
3. **The class imbalance gets much worse in deployment** — **9.4:1 in training → 70:1 on the
   frozen test** at 24 h. This is why **PR-AUC is the headline metric**, never accuracy.

---

## 1. Where Phase 16 sits

| Phase | What | Plain words |
|---|---|---|
| 12–13 | Feature selection + leakage audit | Picked features, proved no leaks |
| 14–15 | Experimental dataset + splits | Froze the table and train/val/test |
| **➜ 16** | **Baselines + class-imbalance analysis** | **"How good is simple?" + "how rare are floods, really?"** |
| 17+ | Candidate ML models | Now try to beat the bar |

Two workflow stages were closed here:
- **Stage 30** = baseline models (this phase).
- **Stage 29** = class-imbalance analysis (§10 of the report, run in the notebook).

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Baseline** | A simple reference model. Your fancy model must beat it to be worth anything. |
| **Climatology** | Always predict the training flood rate (~5%). The "do nothing" floor. |
| **Persistence / D7** | Predict from *today's* river state only (`wl_pct_danger`). Rivers change slowly, so this is strong. |
| **D7 bar** | The rule that every model must beat persistence, not climatology. |
| **PR-AUC** | Precision–Recall Area Under Curve. **Primary metric** for rare events. |
| **ROC-AUC** | Another area metric — but it's too kind when positives are rare. Secondary only. |
| **Brier score** | Average squared error of probabilities (lower = better). Calibration quality. |
| **ECE** | Expected Calibration Error — how far predicted probabilities are from reality (lower = better). |
| **Imbalance ratio** | Negatives : positives, e.g. 70:1 means 70 no-floods per flood. |
| **Threshold** | Probability cut-off that turns a number into yes/no. Must come from **train**, not the cell scored. |
| **Protocols P1/P2/P3** | P1 = temporal (main), P2 = rolling year, P3 = spatial (unseen stations). |
| **Bucket** | Test sub-group: `test_trained` (85 seen stations), `test_unseen` (26), `test_new_network` (22). |
| **Macro vs micro** | Micro = pool all rows (few big stations dominate). Macro = average per station (fairer). Report both (D8). |
| **F14-2** | The ablation that dropped the `up_*` (upstream) features to test their value. |

---

## 3. What the rewrite fixed (why a rewrite was needed)

The *old* baseline script had four defects recorded during Phase 15. They were serious
(they would have made results look better than reality):

| # | Defect | Why it's bad | Fix in the rewrite |
|---|---|---|---|
| 1 | LightGBM **early-stopped on the test fold** | That's tuning on test = leakage | No gradient boosting in this phase at all — flaw is structurally removed |
| 2 | Trained on the old `flood_dataset_model_ready.csv` | Not the frozen experimental table | Use Phase-14 table (200,665 × 67) joined to Phase-15 splits |
| 3 | Invented its own shuffled station folds | Inconsistent, leaky splits | All masks come from Phase-15 columns only |
| 4 | No persistence baseline | Can't judge if ML helps at all | **D7** (`wl_pct_danger`) is fit and printed first |

**Extra guards at run time:**
- Every mask filters out `station_holdout` rows.
- The precision/recall/F1 threshold comes from **train predictions only** — never the cell
  being scored.
- D7 is asserted to have a **positive** coefficient (so its ranking = raw water level).
- Asserts: train ≤ 2021, val = 2022, test ≥ 2023, no shared dates.

**Plain point:** the rewrite removed every way the test set could quietly influence results.

---

## 4. The baselines (validation 2022, all season)

PR-AUC. Base rate = **5.09 % / 5.94 % / 6.58 %** (24/48/72 h).

| Model | Features | PR-AUC 24 h | 48 h | 72 h | ×persistence 24 h |
|---|---|---:|---:|---:|---:|
| climatology | — | 0.0509 | 0.0594 | 0.0658 | 0.06 |
| **persistence (D7)** | `wl_pct_danger` (1) | **0.9203** | **0.8954** | **0.8683** | 1.000 |
| **logreg_linear** | LINEAR (35) | **0.9401** | **0.9241** | **0.9053** | **1.021 / 1.032 / 1.043** |
| logreg_linear_no_up | LINEAR − `up_*` (29) | 0.9403 | 0.9242 | 0.9052 | 1.021 |
| logreg_selected | SELECTED (10) | 0.9398 | 0.9206 | 0.9004 | 1.021 |
| tree_depth4 | FULL (62) | 0.8972 | 0.8902 | 0.8735 | 0.975 |

**How to read this:**
- Climatology's PR-AUC **equals the base rate exactly** (0.0509 = 5.09 %) — a good sanity check.
- Every real model beats climatology by **15–18×** — so climatology is a weak floor.
- But logreg beats **persistence** by only **+2.1 % to +4.3 %**. The shallow tree *doesn't
  beat* persistence at 24 h at all.

**Step-28 question — "does sophisticated ML actually help?"**
Honest answer: **not much, on this cell.** A one-feature persistence model is already 0.92.
That's why the D7 bar matters: fancy models must clear *that*, not climatology.

---

## 5. Rolling origin (P2) — the base rate moves a lot

PR-AUC by validation year. Watch the base rate collapse **14.18 % (2020) → 5.09 % (2022)**:

| Model | val 2020 (24h) | val 2021 | val 2022 |
|---|---:|---:|---:|
| climatology | 0.1418 | 0.0703 | 0.0509 |
| persistence | 0.9678 | 0.9407 | 0.9203 |
| logreg_linear | 0.9679 | 0.9480 | 0.9401 |
| tree_depth4 | 0.9657 | 0.9400 | 0.8972 |

Skill vs persistence (logreg): **1.000 (2020) → 1.008 (2021) → 1.021 (2022)**.

**Insight:** when the flood rate is high (2020, 14 %), the current river level already
explains everything — ML adds nothing. The lower the base rate, the *more* a model adds
over persistence. This is the same "PR-AUC tracks base rate" story from Phase 5/R1.

---

## 6. Spatial CV (P3) — the real weak spot

Mean over 4 station-disjoint folds (model tested on **stations it never saw**):

| Model | PR-AUC 24 h | 48 h | 72 h | macro per-station 24 h |
|---|---:|---:|---:|---:|
| climatology | 0.0960 | 0.1061 | 0.1152 | 0.1222 |
| **persistence (D7)** | **0.9370** | **0.9182** | **0.8979** | 0.8796 |
| logreg_linear | 0.9277 | 0.9032 | 0.8803 | **0.8865** |
| logreg_selected | 0.9336 | 0.9097 | 0.8854 | **0.8936** |
| **tree_depth4** | **0.9401** | **0.9230** | **0.9022** | 0.8451 |

**The awkward truth:** the linear model is **below D7 in 3 of 4** unseen-station folds.
And the pooled vs macro numbers disagree:
- Pooled says: tree ≳ persistence > logreg.
- Macro-per-station (D8) says: logreg (0.8865) > persistence (0.8796) > tree (0.8451).

Why the disagreement? The tree's good pooled score is **carried by a few dense tidal
stations**. Both numbers are reported; **neither alone decides** what comes next.

**Lesson:** Phase 17 candidates must be judged **first on unseen stations**, not on the
temporal cell — that's where generalisation is hard.

---

## 7. Monsoon vs non-monsoon (D1) — never pool seasons

Validation 2022 (24 h). The *ranking* is stable; the *difficulty* changes:

| Model | all | monsoon | non-monsoon |
|---|---:|---:|---:|
| persistence | 0.9203 | 0.9245 | 0.9131 |
| logreg_linear | 0.9401 | 0.9411 | 0.9387 |
| base rate | 0.0509 | 0.0684 | 0.0277 |

On the frozen test, non-monsoon is **much** harder:

| 24 h | PR-AUC all | monsoon | non-monsoon |
|---|---:|---:|---:|
| base rate | 0.0135 | 0.0226 | **0.0029** |
| persistence | 0.7468 | 0.7618 | 0.6267 |
| logreg_linear | 0.7771 | 0.7931 | 0.6496 |

**Plain point:** non-monsoon has a 0.29 % positive rate and PR-AUC ~0.65 — about 0.15 lower
than monsoon. Every headline must stay **split by season** forever.

---

## 8. Frozen test (2023+) — recorded once, read by no decision

Per-year PR-AUC (24 h):

| Model | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|
| climatology (= prevalence) | 0.0068 | 0.0254 | 0.0117 | 0.0127 |
| persistence (D7) | 0.6276 | 0.8022 | 0.7328 | 0.7233 |
| logreg_linear | 0.6313 | 0.8198 | 0.7642 | 0.7461 |

Buckets (never pooled with years — D8):

| PR-AUC (24 h) | `test_trained` (85 st) | `test_unseen` (26 st) | `test_new_network` (22 st) |
|---|---:|---:|---:|
| persistence | 0.6805 | **0.6792** | **0.6914** |
| logreg_linear | **0.7164** | 0.6774 | 0.6805 |
| logreg_selected | 0.7053 | **0.6935** | 0.7113 |

Skill vs persistence (logreg): per year **1.006 / 1.022 / 1.043 / 1.032** — but **0.997 on
`test_unseen`** and **0.984 on `test_new_network`**. Again: the linear model's gain is
confined to stations it has seen.

**Critical protocol note:** this test read was **a record, not an input**. No decision in
Phase 16 used it. Using it to choose anything would be a protocol violation.

Calibration is good: test 2025 (24 h) ECE — logreg_linear **0.0025**, persistence 0.0059,
climatology 0.0842.

---

## 9. F14-2 — do the `up_*` (upstream) features help?

| Cell (24 h) | logreg_linear | logreg_linear_no_up | Δ |
|---|---:|---:|---:|
| val 2022 PR-AUC | 0.9401 | 0.9403 | +0.0002 |
| spatial CV mean | 0.9277 | 0.9295 | +0.0018 |
| test trained / unseen | 0.7164 / 0.6774 | 0.7157 / 0.6769 | −0.0007 / −0.0005 |

**Verdict:** the `up_*` columns (67–70 % NaN!) contribute **nothing measurable**. All
differences are within noise and never favour keeping them. Decision deferred to Phase 17.

---

## 10. Class-imbalance analysis (stage 29) — the deployment reality

From `data/processed/class_imbalance.json`, **59/59 checks pass**.

| Split | Rows | 24 h positives · rate · ratio | 48 h | 72 h |
|---|---:|---|---|---|
| train (≤2021) | 61,991 | 5,943 · 9.59 % · **9.4:1** | 10.60 % · 8.4:1 | 11.51 % · 7.7:1 |
| val (2022) | 17,538 | 891 · 5.09 % · **18.6:1** | 5.94 % · 15.8:1 | 6.58 % · 14.2:1 |
| test (2023+) | 101,174 | 1,425 · 1.41 % · **70.0:1** | 1.77 % · 55.5:1 | 2.12 % · 46.1:1 |

**Three facts that drive the whole project:**
1. **Imbalance worsens into deployment** — 9.4:1 in training → **70:1 on test**. A model
   trained at a 9.6 % base rate meets 1.4 %. This is *the* core difficulty.
2. **It is never a single number** — per-year 72 h ratios span **4.9:1 (2017) to 83.0:1
   (2023)**. 2018 is absent (R6) and never interpolated. Monsoon vs non-monsoon differs 7.3×.
3. **Positives are concentrated** — the **top-10 stations hold 38.9 %** of all 24 h positives,
   top-20 hold 57.6 %. So micro-averages are dominated by a few tidal stations → keep macro
   per-station numbers beside them (D8). **14 stations have zero positives** (Dhaka, Faridpur,
   Bogura, Chuadanga …) — kept, never dropped.

**Two re-measurement warnings (recorded honestly):**
- **W-R3:** phase0 said 13 zero-positive stations; the actual table has **14**.
- **W-R2:** phase0 quoted Mongla 83.8 % / Khulna 52.3 %; the table gives **73.2 % / 39.5 %**.

Neither phase0 nor the table was overwritten — the discrepancy is flagged for a phase0 review.

### Why accuracy and ROC-AUC cannot be the headline (R4)

- **Accuracy:** the trivial *always-say-no* classifier already scores **94.9 %** on
  validation 24 h. The best real model gains < 0.05 accuracy while gaining ~18× on PR-AUC.
  Accuracy is useless here.
- **ROC-AUC:** PR-AUC < ROC-AUC in **all 675** real-model cells (minimum gap 0.014 dev,
  0.123 test). The majority class inflates ROC-AUC. So **PR-AUC stays primary**.
- **Threshold transfer:** a train-derived threshold gives recall 0.847 on val 2022 (base
  5.09 %) but only **0.531** on test 2023 (base 0.68 %). One threshold does **not** follow a
  base-rate shift.

### Policy carried into every later phase (R4)

1. **PR-AUC primary**; ROC-AUC secondary; **accuracy banned as a headline**.
2. **Resampling / class weighting / SMOTE only *inside* training** — never before the split,
   never in validation/test.
3. **Thresholds are train-derived and period-specific.**
4. **Never pool** years/seasons/buckets; report macro beside micro.
5. The FFWC cost asymmetry (missing a flood ≫ false alarm) is a **decision rule** handled
   later (threshold optimisation), not in modelling.

---

## 11. Problems & solutions

| Problem | Solution |
|---|---|
| Old script early-stopped on test (leakage) | Removed gradient boosting from this phase entirely. |
| Fancy models might look good vs a weak floor | Compare against **persistence (D7)**, the strong 1-feature bar. |
| Thresholds could be tuned on the scored cell | Thresholds come from **train predictions only**. |
| Pooled numbers hide station-level effects | Report **macro per-station** beside micro (D8). |
| Test could quietly influence decisions | Test scored once into its own file; **no decision reads it**. |
| Base rate drifts wildly by year/season | Never pool; report per year/season/bucket. |
| Accuracy looks impressive but is meaningless | **PR-AUC is primary**; accuracy banned as headline. |
| Upstream features are 67–70 % NaN and useless | F14-2 ablation showed no measurable effect → deferred decision. |
| phase0 risk numbers don't match the table | Recorded W-R2/W-R3 warnings; phase0 left untouched. |

---

## 12. Files referenced

| File | Purpose |
|---|---|
| `reports/phase16_baseline.md` | Full Phase 16 report (the source of this file) |
| `scripts/baseline_model.py` | The 576-line baseline script (exit 0, 312 s, seed 20261008) |
| `notebooks/08_baseline_models.ipynb` | Verification notebook (59/59 checks green) |
| `reports/phase16_baseline_metrics.csv` | 432 rows of dev metrics |
| `reports/phase16_test_metrics.csv` | 378 rows of frozen-test metrics |
| `reports/phase16_test_per_station.csv` | 3,612 rows per-station confusion counts |
| `data/processed/class_imbalance.json` | Stage-29 imbalance evidence |
| `reports/phase15_split_design.md` | The splits used here |
| `reports/phase0_research_definition.md` | R4 (metric policy), D7 (persistence bar) |

---

## 13. Verdict

- **The bar is persistence, not climatology.** Climatology is just the base rate; every real
  model beats it 10–18×, but logreg beats **D7** by only **+2–4 %**.
- **Spatial generalisation is the weak point** — the linear model is below D7 in 3 of 4
  unseen-station folds; Phase 17 candidates should be judged first on **unseen stations**.
- **The compact SELECTED set generalises as well or better than LINEAR** on unseen data
  (spatial macro 0.8936 vs 0.8865) at a fifth of the features.
- **Calibration is not the problem** (ECE ≤ 0.007 for linear models).
- **Non-monsoon cells are hard** (PR-AUC ~0.65, base rate 0.29 %) and must never be pooled away.
- **Deployment is the imbalanced regime** (70:1 at 24 h vs 9.4:1 in training), with positives
  concentrated in 10 stations and threshold transfer costing 0.32 of recall.
- **Nothing here used the test set for a decision.**

**Bottom line:** We now know exactly how good "simple" is, how hard the real deployment
problem is, and the rules (PR-AUC primary, never pool, weights only in training) that every
later phase must follow. Fancy models have a **high bar** to clear — and that's the point.

---

**Next in the series:** Phase 17 — Candidate ML models (XGBoost/LightGBM/etc. judged against
the D7 bar under P1/P2/P3 and per-season cells). Then Phase 18–25 (search/framework/DL), and
Phase 26–33 (calibration, cost-sensitive thresholds, selection, frozen test).
