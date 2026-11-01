# Advanced Experiments — Novel Methods for Flood Forecasting

**Status:** IN PROGRESS  
**Started:** 2026-10-21  
**Seed:** `20261008` (all experiments)  
**Governing rules:** Phase-13 leakage audit, D8 per-cell eval, PR-AUC primary, frozen test discipline

---

## **Experiment Overview**

| # | Experiment | Target | Data | Method | Status |
|---|------------|--------|------|--------|--------|
| 1 | **Domain Adaptation** | Fix distribution shift (2017-21 vs 2025-26) | Existing FULL_V4 + station metadata | GroupDRO / IRM / DANN | ✅ **COMPLETE (Negative, D16 stands)** |
| 2 | **SAR Inundation Prediction** | P(inundation > 0.1m | pixel) @ 100m | Sentinel-1 SAR + DEM + HAND | U-Net / SegFormer | ⏳ PLANNED |
| 3 | **NWP Ensemble Forecast** | P(flood \| ECMWF 50-member ensemble) | GFS GEFS / ECMWF ENS (if accessible) | Bayesian Model Averaging | ⏳ PLANNED |
| 4 | **Physics-Informed Neural Operator** | Learn PDE operator on river graph | River graph + DEM + gauges | FNO on graph + physics loss | ⏳ PLANNED |
| 5 | **Differentiable Hydrology** | Neural ODE rainfall-runoff | Daily P, ET, Q gauges | Neural ODE / LSTM assimilation | ⏳ PLANNED |
| 6 | **Monotonic Physical Priors** | Impose known feature→risk signs (physics-informed inductive bias) | FULL_V2 + expert sign table | LightGBM / XGBoost `monotone_constraints` | ⚠️ **COMPLETE (Parts A+B negative; audit positive)** |

---

## **Common Evaluation Framework**

All experiments use the **same rigorous evaluation** as the main pipeline:

| Aspect | Specification |
|--------|---------------|
| **Splits** | Train ≤2021 / Val 2022 / Test ≥2023 (7-day embargo) |
| **Spatial CV** | 5 station-disjoint folds (fold 0 held out) |
| **Test buckets** | `test_trained` / `test_new_network` / `test_unseen` |
| **Metrics** | PR-AUC primary, ROC-AUC, recall, precision, F1, Brier, ECE |
| **Per-cell** | 7 dev cells + 7 test cells, never pooled (D8) |
| **Bootstrap** | 400 station-level resamples, paired CIs |
| **Leakage** | Phase-13 audit on all new features |
| **Test discipline** | Frozen test read only after pre-registration |

---

## ⚠️ **Protocol correction (2026-10-21) — D17 is SUSPENDED**

While building Experiment 1, an audit of the MI-4 test read found that the
D17 comparison was **not valid**. Full detail: `reports/d17_comparability_audit.md`.

| Defect | Effect |
|--------|--------|
| `mi4_test_read.py` scored only **complete-case** rows (31.5 % of test) | retained subset has a 1.5–1.8× higher base rate in the year cells → PR-AUC inflated mechanically |
| `test_new_network` mask was `year ≥ 2023 & ~holdout` (26,292 rows) instead of the Phase-15 `first_year > 2023` (4,805 rows) | the claimed "+0.043 new-network" is not a comparison of the same cell |
| `mi4_select_v4.py` used a **made-up** "D16" config, not the frozen D16 hyper-parameters | the reported "D16 refit" was a different model |
| `mi4_hpo_v4.py` **was** correct (imputes, all rows) | only the HPO objective is trustworthy; and it showed Δ ≤ 0.001, not +0.005 |

**Consequence:** `models/final_v3/` is retained on disk but is **not** the
champion. The champion reverts to **D16** (`models/final_v2/`) pending the
corrected read `scripts/mi9_test_read_corrected.py` (test read #3).

### ✅ Resolution — test read #3 (MI-9): **D17 retracted, D16 champion (D18)**

Full report: `reports/mi9_test_read_decision.md`. The D16 sanity gate reproduced
MI-6 **exactly** (max |Δ PR-AUC| = 0 over 63 shared rows), so the comparison is
provably on the same protocol.

| Horizon | D16 | D17 hybrid | Δ |
| --- | ---: | ---: | ---: |
| 24 h | 0.836226 | 0.835923 | **−0.00030** |
| 48 h | 0.768958 | 0.768958 | 0.00000 |
| 72 h | 0.707088 | 0.707088 | 0.00000 |
| **horizon-mean** | **0.770757** | 0.770657 | **−0.00010** |

The claimed **+0.065 / +0.064 / +0.078** was entirely an artefact of scoring the
favourable 31.5 % complete-case subset. The only genuine V4 gains are
`test_2026` (+0.0175) and `test_unseen` (+0.0153); the 24 h bootstrap on
`test_2024` gives −0.0013 [−0.0061, +0.0031], P(Δ>0) = 0.283.

**V4 is closed as a null result.** This is important for Experiments 1–5: the
gap is *not* a feature-count problem, which is why the remaining experiments
target data sources and objectives rather than more columns.

### Baseline table — D16 (champion, D18)

| Horizon | Model | Feature Set | Val_2022 PR-AUC (HPO, all rows) | Test PR-AUC (MI-6, **valid**) |
|---------|-------|-------------|--------------------------------|------------------------------|
| 24h | lgbm7 | FULL_V2 (110) | 0.9708 | 0.8362 |
| 48h | lgbm7 | FULL_V2 (110) | 0.9565 | 0.7690 |
| 72h | xgb3 | FULL_V2 (110) | 0.9450 | 0.7071 |
| **Horizon-mean** | **D16** | — | **0.9574** | **0.7708** |

The 0.8061 / 0.7289 / 0.6662 values that appeared here earlier were the **invalid MI-4
complete-case re-score** of D16, not the MI-6 read. They are retracted with D17. The valid
MI-6 numbers are above and were reproduced bit-for-bit by MI-9.

### Baseline table — D17 hybrid (RETRACTED, traceability only)

| Horizon | Model | Feature Set | Val_2022 (HPO) | Test PR-AUC (MI-4, **invalid**) | Test PR-AUC (MI-9, valid) |
|---------|-------|-------------|----------------|---------------------------------|---------------------------|
| 24h | lgbm13 | FULL_V4 (187) | 0.9713 | 0.8572 | 0.8359 |
| 48h | lgbm7 | FULL_V2 (110) | — | 0.7932 | 0.7690 |
| 72h | xgb3 | FULL_V2 (110) | — | 0.7444 | 0.7071 |

The MI-9 column is the same protocol as MI-6; the hybrid is **−0.00010** behind D16 on the
horizon mean. **D17 is retracted; D18 keeps D16.**

**Key weaknesses to address:**
- Post-2023 base-rate collapse: 2017-21 wet regime (7–14 %) → 2025-26 dry regime (~1–2 %)
- `test_new_network` / `test_unseen` are the only large positive-transfer cells
- Basin-specific heterogeneity (SE Hill 24 h base rate 1.6 % vs Ganges 11.9 %)
- 14 zero-positive stations hurt macro metrics

### **Standing rule added to the checklist**

> Any evaluation that drops rows must report (a) the fraction retained and
> (b) the base rate of the retained subset, and must never be compared with a
> result computed on all rows. Prefer the pipeline imputer over complete-case
> filtering.

---

## **Experiment 1: Domain Adaptation (GroupDRO / IRM / DANN)**

### **1.1 Problem Statement**
- **Distribution shift:** `P_train(y|x) ≠ P_test(y|x)` — training (2017-21) wet, test (2025-26) dry
- **Station heterogeneity:** Tidal (Mongla 73%) vs non-tidal; basin differences; elevation effects
- **Manifestation:** Test_2025/2026 PR-AUC drops vs D16; test_unseen improves but test_trained degrades

### **1.2 Domain Definitions (Station Groups) — as implemented**

Station attributes only — all available at prediction time, no label information
(Phase-13 leakage satisfied by construction). Built by
`scripts/domain_adaptation.py::build_domains` → `reports/domain_adaptation/da_domains.csv`.

| Scheme | Groups | Source |
|--------|--------|--------|
| `basin` | 5 | FFWC `basin` (Brahmaputra 57, Meghna 36, Ganges 25, Unknown 18, SE Hill 17 stations) |
| `elev_q` | 4 | quartiles of DEM `elevation_m` |
| `coastal` | 2 | 40th percentile of `dist_to_coast_km` (61 coastal / 92 inland) |
| `cohort` | 2 | Phase-15 network cohort (`early` 85 / `late` 48) |
| `hydro` | 10 | `basin × coastal` cross |

**Tidal grouping was dropped.** The FFWC station table has no tidal flag, and
`dist_to_coast_km` cannot separate tidal from flashy rivers (median 2.3 km, max
101 km — nearly every station is "near" the coast). `basin` + `coastal` is the
defensible, reproducible substitute. This is a documented limitation, not an
oversight.

The domains are strongly unbalanced in label prevalence, which is what makes
worst-group optimisation meaningful:

| Domain | Train rows | 24 h base rate |
|--------|-----------|----------------|
| Ganges | 13,231 | 0.119 |
| Brahmaputra | 27,294 | 0.107 |
| Meghna | 15,543 | 0.087 |
| SE Hill | 5,923 | **0.016** |
| cohort `late` | 3,513 | **0.291** |
| cohort `early` | 58,478 | 0.084 |

### **1.3 Methods implemented**

| Arm | Learner | Objective |
|-----|---------|-----------|
| `erm_lgbm` | LightGBM (n=300, lr=0.02, 63 leaves) | mean BCE — **control** |
| `groupdro_lgbm` | LightGBM (identical) | Online GroupDRO: EG re-weighting of groups, 8 rounds, η=0.1, weights clipped to [1/G, 5/G] |
| `erm_mlp` | MLP 187→256→64, dropout 0.1 | mean BCE — neural control |
| `groupdro_mlp` | MLP (identical) | Online GroupDRO per mini-batch |
| `irm_mlp` | MLP (identical) | IRM v1 penalty ‖∇_w BCE‖², λ=100, annealed over 5 epochs |
| `dann_mlp` | MLP + domain head | DANN with gradient-reversal layer, λ=1, switched on after 3 epochs |

All arms share the **same features (FULL_V4, 187), the same training rows, the
same seed and the same capacity**; only the objective differs.

### **1.4 Evaluation protocol — strictly out-of-sample**

The MI-8 / MI-4 "7 dev cells" scorecard trained on the entire training split and
then scored cells that were *inside* it (`val_2020`, `val_2021`,
`spatial_fold_1..4`). Only `val_2022` was genuinely held out. That inflated the
absolute numbers.

Experiment 1 fixes the protocol; every scored row is unseen:

| Task | Training rows | Scored cell | Shift |
|------|---------------|-------------|-------|
| `t2020` | years ≤ 2019 | 2020 | temporal |
| `t2021` | years ≤ 2020 | 2021 | temporal |
| `t2022` | years ≤ 2021 (all train) | 2022 | temporal |
| `sf1..sf4` | train-split rows with `spatial_fold != k` | fold `k` | spatial, station-disjoint |

This mirrors `baseline_model.build_tasks` (`P2_rolling_origin`, `P3_spatial_cv`).

**Note on magnitude:** even with the corrected protocol the 24 h ERM cell values
are high (0.969–0.993), so the earlier "saturation is an artefact" reading was
too strong — the in-sample cells were inflated by roughly 0.01–0.03, but the
pre-2023 dev cells are genuinely easy. The real difficulty is confined to the
post-2023 test years, which this experiment cannot see without a pre-registered
read.

### **1.5 Pre-registered decision criteria**

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Decision cell `val_2022` horizon-mean PR-AUC | ≥ ERM control |
| 2 | Worst dev-cell PR-AUC | > ERM control |
| 3 | Paired station bootstrap on `val_2022` | CI must not favour the control |
| 4 | Per-cell safety | no cell may regress by more than 0.005 |

Only if all four hold does the method become eligible for a **pre-registered test
read**. Otherwise D16 stands and the result is recorded as negative.

---

### **1.6 Progress Log**

| Date | Step | Status | Notes |
|------|------|--------|-------|
| 2026-10-21 | Folder & report created | ✅ | `advanced_experiments/` + this report |
| 2026-10-21 | Notebook 01 created | ✅ | `advanced_experiments/01_domain_adaptation.ipynb` (12 cells, generated by `scripts/build_nb.py`) |
| 2026-10-21 | Domain builder | ✅ | `scripts/domain_adaptation.py domains` — 5 schemes, 153 stations, leakage-safe |
| 2026-10-21 | Data-source probe | ✅ | `scripts/probe_data_sources*.py` — Copernicus S1 catalogue reachable; CHIRPS/ECMWF unreachable; NASA POWER reachable |
| 2026-10-21 | **D17 comparability audit** | ❌→✅ | **D17 suspended** — MI-4 test read was not comparable (`reports/d17_comparability_audit.md`) |
| 2026-10-21 | Corrected test read (MI-9) | ✅ | `scripts/mi9_test_read_corrected.py` — MI-6 protocol, D16 sanity gate **PASS (exact)** |
| 2026-10-21 | **Decision D18** | ✅ | **D17 retracted; D16 is champion** (horizon-mean 0.77076 vs 0.77066). V4 closed as a null result |
| 2026-10-22 | **Experiment 1 Complete** | ✅ | Completed 3-horizon training across all 7 tasks for 7 methods (49 runs). Evaluated GroupDRO, IRM, DANN against ERM and D16. |

---

### **1.7 Experiment 1 Results & Decision**

#### Multi-Horizon Scorecard (Basin Domain Partition, Out-of-Sample Evaluation)

| Method | Val_2022 24h | Val_2022 48h | Val_2022 72h | Val_2022 Mean | Dev Mean (7 cells) | Worst Cell Mean |
|---|---:|---:|---:|---:|---:|---:|
| **`erm_d16` (Reference Champion)** | 0.970789 | 0.956485 | **0.945007** | **0.957427** | 0.966532 | 0.944930 |
| `groupdro_lgbm` | 0.970592 | **0.956881** | 0.942408 | 0.956627 | 0.968509 | 0.953267 |
| `erm_lgbm` (FULL_V4 Control) | **0.971282** | 0.955882 | 0.942587 | 0.956584 | **0.969203** | **0.954647** |
| `erm_mlp` (Neural Control) | 0.933912 | 0.928376 | 0.909903 | 0.924064 | 0.927736 | 0.818500 |
| `groupdro_mlp` | 0.933110 | 0.924104 | 0.897921 | 0.918378 | 0.925602 | 0.817516 |
| `irm_mlp` | 0.933232 | 0.928190 | 0.908622 | 0.923348 | 0.927331 | 0.816834 |
| `dann_mlp` | 0.930911 | 0.922919 | 0.908878 | 0.920903 | 0.927646 | 0.823299 |

#### Pre-Registered Decision Criteria Audit

1. **Criterion 1 (val_2022 $\ge$ ERM Control)**:
   * `groupdro_lgbm` achieves 0.956627 vs 0.956584 for `erm_lgbm` (+0.000043 marginal win), but is inferior to the reigning champion `erm_d16` (0.957427, $\Delta = -0.000800$).
   * All neural DA variants (`groupdro_mlp`, `irm_mlp`, `dann_mlp`) fail against the neural ERM control (0.924064).
2. **Criterion 2 (Worst dev-cell PR-AUC $>$ ERM Control)**:
   * `groupdro_lgbm` achieves 0.953267 vs 0.954647 for `erm_lgbm` — **FAILS**.
3. **Criterion 3 (Paired Bootstrap on val_2022 must not favor Control)**:
   * `groupdro_lgbm` vs `erm_d16`: 24h $\Delta = -0.00035$ ($P(\Delta>0) = 0.425$); 48h $\Delta = +0.00034$ ($P(\Delta>0) = 0.630$); 72h $\Delta = -0.00296$ ($P(\Delta>0) = 0.078$). At 72h, bootstrap strongly favors the control.
   * All neural DA models vs `erm_d16`: $P(\Delta>0) = 0.000000$ (completely dominated by gradient boosting).
4. **Criterion 4 (Safety $\Delta \ge -0.005$)**: Satisfied for GBDT, but criteria 1–3 failed.

#### Verdict
> **Experiment 1 Outcome: NEGATIVE RESULT. D16 STANDS AS CHAMPION.**  
> Pre-registered criteria are not met. In accordance with test discipline, **no test read is authorized**.  
> Objective re-weighting (GroupDRO / IRM / DANN) on tabular station-partitioned features cannot overcome the structural distribution shift. The gap is an informational and sensory ceiling of point gauges. We advance to **Experiment 2 (SAR Inundation Prediction)**.

---

## **Experiment 6: Monotonic Physical Priors** ⚠️ COMPLETE (Parts A+B negative)

Full report: `reports/mi12_monotonic.md` · Pre-registration:
`data/processed/monotonic_signs_v1.json` · Script: `scripts/mi12_monotonic.py` ·
Seed `20261008` · **`test_rows_read = 0`**.

### 6.1 Question

Every hydrological feature has a known *direction* (a higher stage or more rain can only
raise flood probability; a larger margin below danger can only lower it). Gradient
boosting does not know this and can learn an inverted response in a wet training regime
that fails to transfer to the dry evaluation regime. Does imposing the sign as a hard
monotone constraint improve out-of-sample robustness?

### 6.2 Design

Five arms, identical rows (`t2020/t2021/t2022` rolling origin + `sf1..sf4` station-
disjoint), identical `FULL_V2` features, identical D16 per-horizon hyper-parameters;
only the constraint vector changes.

| arm | constraints |
| --- | --- |
| `a0_unconstrained` | none — D16 control |
| `a1_strict` | 42 high-confidence hydrological signs |
| `a2_broad` | 90 signs (+ soil wetness, GloFAS, terrain, threshold) |
| `a3_placebo` | the 42 strict signs × −1 |
| `a4_random` | random ±1 on the same 42 features |

### 6.3 Result

| arm | val_2022 24 h | 48 h | 72 h | horizon mean | worst dev cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a0_unconstrained` (D16) | 0.970786 | 0.956485 | 0.945007 | **0.957426** | 0.924990 |
| `a1_strict` | 0.964248 | 0.951577 | 0.936357 | 0.950727 | **0.928731** |
| `a2_broad` | 0.963540 | 0.950956 | 0.936666 | 0.950387 | 0.920687 |
| `a3_placebo` | 0.642880 | 0.668522 | 0.670693 | 0.660698 | 0.431378 |
| `a4_random` | 0.958558 | 0.946690 | 0.938574 | 0.947941 | 0.899277 |

Pre-registered criteria: `a1` passes (2) worst-cell and (4) placebo, fails (1) mean and
(3) no-regression; `a2` passes only (4). Both are **NOT SUPPORTED**. Paired station
bootstrap confirms the loss is real (strict arm −0.0071/−0.0050/−0.0090 at 24/48/72 h,
all CIs exclude zero).

### 6.4 What did survive

1. **Physical-consistency audit of D16** (no retraining): 4/4, 5/5 and 9/9 *strong*
   features are correctly signed at 24/48/72 h; material sign violations are 0 at 24 h,
   6 at 48 h and 15 at 72 h, concentrated in lagged GloFAS discharge, 14/30-day rain,
   soil-moisture anomalies and short water-level lags. The champion is physically right
   where the signal is strong and wrong where it is weak.
2. **Sign information is real, not regularisation**: correct signs beat random signs on
   18/21 cell×horizon points (+0.0295 on the worst spatial cell, fold 1 at 72 h) and
   beat flipped signs by ~0.29. Random signs also widen the cross-cell spread
   (0.0714 vs 0.0465 at 72 h), while correct signs narrow it (0.0511 for D16).

### 6.5 Verdict

> **Experiment 6 outcome: NOT SUPPORTED. D16 stands; no test read authorised.**
> Global monotonicity is too strong a prior for a basin-heterogeneous panel: it costs
> 0.007 mean PR-AUC, buys 0.004 on the worst unseen-station cell, and helps only at 72 h
> where the violations are concentrated. The cheapest follow-up — constrain only the
> materially-violating features — was run as **Part B (§6.6)** and is also negative.

### 6.6 Part B — targeted priors (also not supported)

Full report: `reports/mi12b_targeted.md` · Pre-registration:
`data/processed/monotonic_signs_v1b_targeted.json` · **`test_rows_read = 0`**.

Target sets were selected from the **2020 and 2021 out-of-sample audits only**, never
from `val_2022`: a feature is constrained at horizon *h* iff its unconstrained PDP
response is wrong-signed with |ΔP| ≥ 0.001 in **both** years. The rule selects
**0 / 2 / 7** features at 24 / 48 / 72 h. Because the 24 h set is empty, `b1` is a
literal no-op there (its predictions are bit-identical to D16), so the four Part A
criteria were applied **per horizon**; this amendment was recorded before fitting.

| arm | 24 h | 48 h | 72 h | mean | worst cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a0_unconstrained` (D16) | 0.970786 | 0.956485 | 0.945007 | **0.957426** | 0.924990 |
| `b1_targeted` | 0.970786 | 0.956015 | 0.944361 | 0.957054 | 0.923770 |
| `b2_targeted_placebo` | 0.970786 | 0.956321 | 0.944702 | 0.957270 | 0.923741 |

`b1` passes only criterion 3 (no cell regression > 0.005): c1 and c2 fail at 48 h and
72 h, and c4 fails because the flipped-sign `b2` is at least as good. Paired station
bootstrap: `b1`−`a0` = −0.00055 [−0.00205, +0.00031] at 48 h and −0.00068
[−0.00230, +0.00049] at 72 h; `b1`−`b2` is also indistinguishable from zero. The
targeted prior is **safe but inert**.

Mechanism: every targeted feature has an unconstrained sibling with Spearman |ρ| =
0.60–0.998 (`wl_last_lag4`↔`wl_last_lag3` = 0.998; `ps_wet_root_anom`↔
`ps_wet_top_anom` = 0.929), so the trees route around the constraint. The features'
total PDP effects are ≤ 0.016 against 0.50 for `wl_dist_to_danger`. The targeted arm
also does **not** reproduce Part A's +0.0037 worst-cell gain (it is −0.0012 there), so
that gain came from constraining the broader strict set, not from fixing the audited
violations.

Part B does not contradict the Part A sign ablation; it localises it. Sign information
matters when the constrained set contains high-leverage features (flipping the 42 strict
signs destroys the model), but not when the set is restricted to redundant weak features
(flipping the targeted signs changes nothing measurable).

> **Experiment 6b outcome: NOT SUPPORTED. D16 stands; no test read authorised.**
> Targeted monotonicity costs ~1/10 of the 42-sign arm and buys nothing. It is the right
> way to impose sign consistency if it is required for non-accuracy reasons, but the
> audit — not the constraint — is the useful product.

---

## **Experiment 2: SAR Inundation Prediction** ⏳ PLANNED

*To be detailed after Experiment 1 completes.*

---

## **Experiment 3: NWP Ensemble Forecast** ⏳ PLANNED

*To be detailed after Experiment 2 completes.*

---

## **Experiment 4: Physics-Informed Neural Operator** ⏳ PLANNED

*To be detailed after Experiment 3 completes.*

---

## **Experiment 5: Differentiable Hydrology** ⏳ PLANNED

*To be detailed after Experiment 4 completes.*

---

## **Cross-Experiment Comparison Matrix** (To Be Filled)

| Experiment | Val_2022 PR-AUC | Test Horizon-Mean | Test Unseen | Test 2025/26 | Compute | Novelty |
|------------|-----------------|-------------------|-------------|--------------|---------|---------|
| D16 (champion, reinstated) | 0.9574* | **0.7708** | 0.7712 | — | CPU | — |
| D17 Hybrid (**RETRACTED**, D18) | 0.9708* | 0.7707 | 0.7712 | — | CPU | — |
| 1. Domain Adaptation (GroupDRO / IRM / DANN) | 0.9566 (GBDT) / 0.9241 (MLP) | — (No test read authorized) | — | — | CPU | High (Null result; D16 stands) |
| 2. SAR Inundation | — | — | — | — | GPU | High |
| 3. NWP Ensemble | — | — | — | — | HPC | High |
| 4. FNO on Graph | — | — | — | — | GPU | High |
| 5. Neural ODE | — | — | — | — | GPU | High |
| 6. Monotonic Physical Priors | **0.9507** (strict) / 0.9504 (broad) / **0.9571** (targeted) | — (No test read authorised) | — | — | CPU | Medium (Parts A+B negative; audit + cost characterisation positive) |

*HPO objective (all rows, imputed). Test columns only for reads that passed the
comparability gate.

---

## **Reproducibility Checklist** (Per Experiment)

- [ ] Code in `scripts/` + notebook in `advanced_experiments/`
- [ ] Run log with exit code, wall-clock, seed
- [ ] Report section updated with numbers, failures, warnings
- [ ] Phase-13 leakage audit passed
- [ ] Paired bootstrap vs D17 on all dev/test cells
- [ ] Test read guarded (only after pre-registration)