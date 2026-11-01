# Experiment 6 — Monotonic physical priors

| | |
| --- | --- |
| **Status** | **COMPLETE — pre-registered criteria NOT met; D16 stands** |
| **Date** | 2026-10-28 |
| **Script** | `scripts/mi12_monotonic.py` |
| **Pre-registration** | `data/processed/monotonic_signs_v1.json` — written before any constrained fit |
| **Part B follow-up** | `reports/mi12b_targeted.md` — targeted priors, also NOT SUPPORTED |
| **Run logs** | `scripts/mi12_audit.log`, `scripts/mi12_run.log`, `scripts/mi12_report.log` |
| **Seed** | `20261008` everywhere |
| **Feature set** | `FULL_V2` (110 features) — identical to D16 |
| **Learners** | D16 per-horizon family and configuration, never re-tuned: LightGBM 24/48 h, XGBoost 72 h |
| **Test policy** | **`test_rows_read = 0`** — no `split == "test"` row was scored |
| **Wall clock** | 1,230 s (105 fits) + ~60 s report |
| **Artefacts** | `reports/mi12_monotonic/{ledger,scorecard,bootstrap,deltas_vs_a0,pdp_audit}.csv`, `summary.json` |
| **Figures** | `figures/mi12_pr_auc_by_cell.png`, `mi12_delta_vs_d16.png`, `mi12_bootstrap.png` |

---

## 0. Verdict in one paragraph

Imposing physically known **signs** on the champion's features is **cheap, binding, and
informative — but it does not improve accuracy on this panel.** The strict arm loses
**−0.0067** mean `val_2022` PR-AUC (0.9574 → 0.9507, bootstrap CI excludes zero), the
broad arm loses **−0.0070**, and both fail the pre-registered criteria. The one genuine
positive is that the strict arm improves D16's **worst** unseen-station cell
(`spatial_fold_1`, 72 h) from 0.9250 to **0.9287** (+0.0037). The experiment also
produced a result that stands on its own: a **partial-dependence audit** showing that
the unconstrained champion is already physically consistent on every strong feature
(9/9 at 72 h) but inverts the sign on **15 secondary features at 72 h**. Those two
findings together are the useful output; the accuracy claim is not.

---

## 1. What was tested

The hypothesis was that a hard monotone shape prior would act as a regime-shift
regulariser: training is a wet regime (2017–21, 24 h base rate 7–14 %), evaluation is
dry (2025–26, ~1–2 %), and an unconstrained GBM is free to learn an inverted response
that does not transfer. If the constraint removes that freedom, the validation→test
collapse (0.957 → 0.771 for D16) might narrow.

Five arms, **identical rows, features, seed and hyper-parameters**; only the sign
vector changes:

| arm | constraints | role |
| --- | --- | --- |
| `a0_unconstrained` | none (110 zeros) | D16 control |
| `a1_strict` | 42 high-confidence hydrological signs | main treatment |
| `a2_broad` | 90 signs (+ soil wetness, GloFAS, terrain, threshold) | dose escalation |
| `a3_placebo` | the 42 strict signs × **−1** | anti-physics control |
| `a4_random` | random ±1 on the same 42 features | regularisation-only control |

The sign table is a committed artefact with a rationale per feature. Twenty features
are deliberately **unconstrained**: the five station-identity categoricals, `lat`/`lon`,
`n_readings`, `dist_to_coast_km`, `dist_river_km`, `dist_major_river_km`, `tmax_c`,
`tmin_c`, the two seasonal harmonics, and the five interaction terms. The interaction
terms matter: the sign of a product is undefined unless both factors are signed, so
constraining `ix_wet_top_x_slope` would be mathematically invalid.

Protocol is the **corrected** one from Experiment 1, not the flawed MI-sprint grid:

| task | training rows | scored cell |
| --- | --- | --- |
| `t2020` | years ≤ 2019 | 2020 |
| `t2021` | years ≤ 2020 | 2021 |
| `t2022` | train split (≤ 2021) | `val_2022` |
| `sf1..sf4` | train split with `spatial_fold != k` | fold `k` |

Every scored row is unseen by the model that scores it. `make_tasks()` asserts that no
evaluation mask contains a `split == "test"` row.

**A0 reproduces D16 to within 1 × 10⁻⁵ on all seven dev cells** (e.g. 0.987258 /
0.982921 / 0.970786 / 0.965159 / 0.992182 / 0.987692 / 0.967271 at 24 h), which is the
reproducibility gate for the whole comparison.

---

## 2. Pre-treatment audit — is the champion already physical?

Before constraining anything, the unconstrained D16 was fitted on `t2022` training rows
and probed with **partial dependence** on the 17,089–17,502 `val_2022` rows: each signed
feature was swept over nine quantiles, all other features held fixed, and the direction
of the averaged response recorded. A *material* violation is a wrong-signed total change
of at least 0.001 in probability; smaller flips are recorded separately because they are
numerically real but operationally irrelevant.

| horizon | strong features (\|ΔP\| ≥ 0.005) | correct | material violations | negligible flips |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 4 | **4 / 4** | **0** | 42 |
| 48 h | 5 | **5 / 5** | 6 | 33 |
| 72 h | 9 | **9 / 9** | **15** | 22 |

The strong features are the physics working exactly as expected:
`wl_dist_to_danger` (−0.41/−0.45/−0.50), `wl_pct_danger` (+0.058/+0.067/+0.036),
`wl_rise_1d` (+0.013/+0.015/+0.016), `precip_mm`, `wl_exceed_days_30`,
`wl_last_lag10`, `danger_level`, `ps_wet_root_clim`, `gf_up_ratio`.

The material violations at 72 h are concentrated in **lagged GloFAS discharge**
(`gf_up_ratio_lag10` −0.0047, `gf_near_dis_lag3` −0.0025), **long-window rain**
(`wet_days_7` −0.0037, `rain_30d` −0.0022, `rain_14d` −0.0014, `rain_max_7d` −0.0012),
**soil-moisture anomalies** (`ps_wet_root_anom` −0.0037, `ps_wet_top_anom` −0.0015) and
**short water-level lags** (`wl_last_lag1/2/3/4/6`, −0.0011 to −0.0020).

This is the key diagnostic result, and it is independent of whether constraints help:
**the champion is physically correct where the signal is strong, and wrong where the
signal is weak.** Inverting a strong, correctly-signed feature is not the failure mode;
the failure mode is a cluster of weak secondary features at the longest lead time.

---

## 3. Results

### 3.1 PR-AUC on the seven out-of-sample dev cells

| arm | 24 h | 48 h | 72 h | horizon mean | Δ vs D16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a0_unconstrained` (D16) | 0.970786 | 0.956485 | 0.945007 | **0.957426** | — |
| `a1_strict` | 0.964248 | 0.951577 | 0.936357 | 0.950727 | **−0.006699** |
| `a2_broad` | 0.963540 | 0.950956 | 0.936666 | 0.950387 | **−0.007039** |
| `a3_placebo` (flipped) | 0.642880 | 0.668522 | 0.670693 | 0.660698 | −0.296728 |
| `a4_random` | 0.958558 | 0.946690 | 0.938574 | 0.947941 | −0.009485 |

Two things follow immediately:

* **The constraints bind hard.** Flipping the signs destroys the model (0.66 mean); it is
  not a cosmetic change to the hypothesis class.
* **Correct signs beat random signs.** `a1_strict` is 0.0028 better than `a4_random` on
  the horizon mean and 0.29 better than the placebo; the prior carries real
  information, it is simply not enough to pay for the lost flexibility.

### 3.2 Paired station bootstrap against D16 (`val_2022`, 400 resamples)

| arm | 24 h Δ [95 % CI] | 48 h Δ [95 % CI] | 72 h Δ [95 % CI] |
| --- | --- | --- | --- |
| `a1_strict` | −0.00714 [−0.01734, −0.00128] | −0.00502 [−0.01049, −0.00069] | −0.00898 [−0.01736, −0.00299] |
| `a2_broad` | −0.00779 [−0.01840, −0.00166] | −0.00573 [−0.01209, −0.00096] | −0.00891 [−0.01912, −0.00187] |
| `a3_placebo` | −0.35771 [−0.64940, −0.14330] | −0.31698 [−0.59994, −0.12837] | −0.29558 [−0.60143, −0.11204] |
| `a4_random` | −0.01340 [−0.03086, −0.00297] | −0.01006 [−0.02019, −0.00195] | −0.00611 [−0.01378, −0.00037] |

Every treatment CI excludes zero, so the loss is **statistically real, not tuning noise**.
`P(Δ > 0)` is 0.000 / 0.0075 / 0.000 for the strict arm at 24/48/72 h.

### 3.3 Where the constraints do help: the worst unseen-station cell

| 72 h, spatial folds | fold 1 | fold 2 | fold 3 | fold 4 | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a0_unconstrained` | 0.924990 | 0.976138 | 0.968171 | 0.935707 | 0.924990 |
| `a1_strict` | **0.928731** | 0.975197 | 0.963660 | 0.932952 | **0.928731** |
| `a2_broad` | 0.920687 | 0.973735 | 0.962205 | 0.931240 | 0.920687 |
| `a4_random` | 0.899277 | 0.970674 | 0.964145 | 0.932886 | 0.899277 |

The strict arm is the only arm that beats D16 on the worst cell, and it does so at 72 h —
the horizon with the most headroom. It also **narrows the cross-cell spread** at 72 h
(max−min: 0.0511 → 0.0465), whereas random signs widen it to 0.0714. Correct physical
signs buy spatial stability; they do not buy average accuracy.

Against the random-sign control, `a1_strict` wins **18 of 21** cell×horizon points,
including +0.0295 on `spatial_fold_1`/72 h. That is the cleanest evidence in the whole
experiment that the sign prior is doing directional work rather than merely regularising.

### 3.4 Dose response

More constraints is monotonically worse: `a1` (−0.0067) > `a2` (−0.0070), and the broad
arm also loses the worst-cell gain (0.9207, below D16). The additional 48 "broad" signs —
terrain, threshold and soil/GlofAS proxies — are the ones most likely to be
basin-dependent, and they are the ones that cost the most.

---

## 4. Pre-registered decision

The rule was fixed before the run: an arm is supported only if it (1) matches or beats
D16 on the `val_2022` horizon mean, (2) beats D16 on the worst cell, (3) regresses no
cell by more than 0.005, and (4) beats both placebo arms.

| criterion | `a1_strict` | `a2_broad` |
| --- | --- | --- |
| 1. `val_2022` mean ≥ D16 | **FAIL** (−0.0067) | **FAIL** (−0.0070) |
| 2. worst cell > D16 | **PASS** (+0.0037) | **FAIL** (−0.0043) |
| 3. no cell regression > 0.005 | **FAIL** (−0.0087) | **FAIL** (−0.0089) |
| 4. beats both placebos | **PASS** | **PASS** |
| **verdict** | **NOT SUPPORTED** | **NOT SUPPORTED** |

**D16 remains the champion. No test read was performed and none is authorised.**

The worst-cell pass is a single cell at +0.0037; without a spatial-cell bootstrap it
should be treated as suggestive, not established.

---

## 5. Why it failed — and why that is informative

1. **Global monotonicity is the wrong granularity for a cross-sectional panel.** The
   features that carry most of the risk (`danger_level`, `elevation_m`, `slope_deg`,
   `dist_major_river_km`) are station attributes, not time-varying states. Their true
   partial effect can change sign between the Ganges and the SE Hill, so one global sign
   is wrong for part of the panel. The broad arm, which constrains exactly these,
   performs worst.
2. **The dominant features were already correct.** The audit shows 9/9 strong features
   correctly signed at 72 h. Constraining them removes flexibility without removing any
   error, so the cost is pure.
3. **The features that were wrong are the ones whose sign we are least sure about.**
   `rain_30d` and `wet_days_7` are not obviously monotone once a catchment is saturated;
   `ps_wet_*_anom` encodes anomaly, and a positive anomaly in a dry regime can mean a
   different season rather than a wetter catchment. Forcing a sign there is not obviously
   "physics" — it is a guess, and the model was right to resist it.
4. **Hard constraints are all-or-nothing.** A single wrong sign anywhere in a tree path
   propagates through every subsequent split. A soft penalty on negative partial
   derivatives would allow local exceptions at bounded cost.

---

## 6. What survives, and what to do next

**Survives as a result (usable in the paper now):**

* A **physical-consistency audit** of the champion: 9/9 strong features correct at 72 h,
  0 material violations at 24 h, 15 at 72 h. This is a new, honest XAI result and does
  not require any retraining. It says the model is trustworthy where it matters and
  over-fits weak lags at long lead time.
* A **sign-information ablation**: correct > random > flipped, on 18/21 points and by
  0.0295 on the worst spatial cell. This isolates *physical direction* from mere
  regularisation.

**Next steps, in cost order:**

1. **Constrain only the 72 h model, and only the 15 materially-violating features.** The
   24 h model has zero material violations, so there is nothing to fix there; 72 h is
   where both the violations and the worst-cell gain live. Expected cost is smaller than
   the 42-feature arm. → **Run as Part B** (`reports/mi12b_targeted.md`), using the
   2020/2021 audits rather than the `val_2022` list to avoid leakage. Also **NOT
   SUPPORTED**: the targeted arm is statistically indistinguishable from D16, and Part A's
   worst-cell gain does not survive.
2. **Per-basin sign tables** (5 basins × the same features). This directly addresses the
   heterogeneity objection; it is ~5× the sign-table work and still CPU-only.
3. **Soft monotonicity penalty** instead of a hard constraint — penalise negative
   finite-difference slopes on a subsample during training, with a tunable weight.
4. Only if one of those clears the dev-cell gate: a **single pre-registered test read**,
   logged as open #7 in `reports/test_read_ledger.md`.

Not recommended: constraining all 90 features, or constraining the cross-sectional
static features at all.

---

## 7. Reproducibility

```bat
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py signs    :: validate the table
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py audit    :: PDP audit of D16
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py run      :: 5 arms x 3 horizons x 7 tasks
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py report   :: scorecard, bootstrap, verdict
```

The run is resumable: every `(task, horizon, arm)` fit is appended to
`reports/mi12_monotonic/ledger.csv` and skipped on restart. Predictions are stored per
`(arm, horizon)` under `reports/mi12_monotonic/preds/`, so the bootstrap can be recomputed
without refitting.

**Guards enforced in code:** the sign table must cover exactly the 110 `FULL_V2` features;
tier `none` must have sign 0; every evaluation mask is asserted disjoint from
`split == "test"`; A0 must reproduce D16's `val_2022` numbers (observed max deviation
3 × 10⁻⁶, LightGBM thread-summation noise).

---

## 8. Part B follow-up — targeted priors

Full report: `reports/mi12b_targeted.md`.

Part B selected constraints from the **2020 and 2021** out-of-sample audits only (rule:
material wrong sign, |ΔP| ≥ 0.001, in both years), leaving `val_2022` as a clean decision
cell. The rule selected **0 / 2 / 7** features at 24 / 48 / 72 h. Results:

* `b1_targeted` changes `val_2022` PR-AUC by **−0.0005** (48 h) and **−0.0006** (72 h);
  both paired-bootstrap CIs include zero.
* The flipped-sign placebo `b2` is statistically indistinguishable from `b1`
  (−0.0003, CI includes zero), so the sign direction of the targeted features is not
  identifiable at this effect size.
* The targeted arm does **not** reproduce the strict arm's +0.0037 on the worst spatial
  cell (it is −0.0012 there).
* Diagnosis: every targeted feature is 0.60–0.998 Spearman-correlated with an
  unconstrained sibling, and the features' total PDP effects are ≤ 0.016 versus 0.50 for
  `wl_dist_to_danger`. The trees route around the constraints.

The targeted prior is **safe but inert**. It is the right way to impose monotonicity if
sign consistency is required for non-accuracy reasons — at roughly one-tenth the cost of
the 42-sign arm — but it does not improve the model. **Part B is NOT SUPPORTED; D16
stands; `test_rows_read = 0`.**
