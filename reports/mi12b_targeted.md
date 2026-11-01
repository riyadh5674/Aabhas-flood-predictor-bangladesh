# Experiment 6b — Targeted monotonic priors

| | |
| --- | --- |
| **Status** | **COMPLETE — pre-registered criteria NOT met; D16 stands** |
| **Date** | 2026-10-28 |
| **Parent** | Experiment 6, Part A — `reports/mi12_monotonic.md` |
| **Script** | `scripts/mi12_monotonic.py` — `audit`, `targeted`, `run --arms`, `report` |
| **Pre-registration** | `data/processed/monotonic_signs_v1b_targeted.json` — first written **20:49:46**, amended **20:50:42**, before the first Part B fit at **20:51:08** |
| **Selection data** | `reports/mi12_monotonic/pdp_audit_2020.csv`, `pdp_audit_2021.csv` — both out-of-sample, **no 2022 rows** |
| **Decision cell** | `val_2022` — never consulted for feature selection |
| **Seed** | `20261008` everywhere |
| **Feature set** | `FULL_V2` (110 features) — identical to D16 |
| **Learners** | D16 per-horizon family and configuration, never re-tuned: LightGBM 24/48 h, XGBoost 72 h |
| **Test policy** | **`test_rows_read = 0`** — no `split == "test"` row was scored |
| **Run logs** | `scripts/mi12_audit_2020.log`, `mi12_audit_2021.log`, `mi12_targeted.log`, `mi12b_run.log`, `mi12_report_final.log` |
| **Wall clock** | audits 1.5 + 1.9 min; 42 fits **612 s**; report ~60 s |
| **Artefacts** | `reports/mi12_monotonic/{pdp_audit_2020,pdp_audit_2021,scorecard,bootstrap,deltas_vs_a0}.csv`, `summary.json`, `preds/` |
| **Figures** | `figures/mi12_pr_auc_by_cell.png`, `mi12_delta_vs_d16.png`, `mi12_bootstrap.png` (now include the Part B arms) |

---

## 0. Verdict in one paragraph

Selecting the constraints from the champion's **own pre-2022 errors** removes Part A's
collateral damage but does not improve the model. The rule chooses **0 / 2 / 7** features
at 24 / 48 / 72 h. On `val_2022` the targeted arm changes PR-AUC by **−0.0005** at 48 h
and **−0.0006** at 72 h; both paired-bootstrap CIs include zero. Flipping the targeted
signs (the `b2` placebo) is statistically indistinguishable from keeping them
(+0.0003 in favour of the flip, CI includes zero). The targeted arm passes only the
"safe" criterion (no cell regresses by more than 0.005) and fails the other three at every
testable horizon. The mechanism is visible in the data: every targeted feature is
**0.60–0.998** Spearman-correlated with an unconstrained sibling, so the trees route
around the constraint, and the entire marginal effect of the targeted features is
≤ 0.016 in probability against 0.50 for `wl_dist_to_danger`. **Monotonicity is a good
audit, not a lever. D16 remains champion; no test read was performed or authorised.**

---

## 1. Why run a Part B at all

Part A imposed 42 (strict) or 90 (broad) hard signs and lost **−0.0067 / −0.0070** mean
`val_2022` PR-AUC. Two facts from that run motivated a narrower intervention:

1. The unconstrained champion was already correct on **all** strong features
   (4/4, 5/5, 9/9 at 24/48/72 h). The violations were confined to weak secondary
   features: lagged GloFAS discharge, 14/30-day rain, soil-moisture anomalies and short
   water-level lags.
2. The strict arm nevertheless bought **+0.0037** on the worst unseen-station cell
   (`spatial_fold_1`, 72 h) and narrowed the cross-cell spread. That suggested *some*
   part of the sign prior was useful, and the useful part might be separable from the
   expensive part.

The obvious follow-up was therefore: constrain **only the features that are actually
wrong**, and only at the horizon where they are wrong. Part A's violation list, however,
was measured on `val_2022` — the decision cell. Using it to choose constraints would
leak the decision cell into the design. Part B therefore rebuilds the list from the
**2020 and 2021 out-of-sample audits only** and keeps `val_2022` untouched until the
final scorecard.

The question: *does narrowing the prior recover the accuracy cost while keeping the
worst-cell benefit?*

---

## 2. Selection rule and pre-registration

### 2.1 Rule

For each horizon independently:

1. Fit unconstrained D16 on `t2020` training rows (years ≤ 2019) and run the same
   partial-dependence audit on the 2020 rows.
2. Fit unconstrained D16 on `t2021` training rows (years ≤ 2020) and audit the 2021
   rows.
3. Constrain feature *f* at horizon *h* **iff** its response has the wrong sign with
   **|ΔP| ≥ 0.001 in both years**.

The audit protocol is identical to Part A: 2,000 sampled evaluation rows, nine quantiles
from 1 % to 99 %, all other features held fixed, direction of the averaged response
recorded. The rule, threshold, seed and provenance (including both years' ΔP for every
selected feature) are stored in `monotonic_signs_v1b_targeted.json`. The file was written
before any Part B model was fitted.

### 2.2 What the rule selected

| horizon | material violations 2020 | material violations 2021 | in both | targeted |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 4 | 0 | 0 | **0** |
| 48 h | 7 | 2 | 2 | **2** |
| 72 h | 16 | 7 | 7 | **7** |

| horizon | feature | sign | ΔP 2020 | ΔP 2021 |
| --- | --- | ---: | ---: | ---: |
| 48 h | `gf_up_ratio_lag10` | +1 | −0.0023 | −0.0013 |
| 48 h | `gf_up_ratio_lag14` | +1 | −0.0028 | −0.0012 |
| 72 h | `gf_up_ratio_lag10` | +1 | −0.0027 | −0.0040 |
| 72 h | `nb5_n_wl_danger` | +1 | −0.0012 | −0.0017 |
| 72 h | `ps_wet_root_anom` | +1 | −0.0158 | −0.0043 |
| 72 h | `rain_30d` | +1 | −0.0078 | −0.0022 |
| 72 h | `wet_days_7` | +1 | −0.0030 | −0.0033 |
| 72 h | `wl_last_lag4` | +1 | −0.0023 | −0.0030 |
| 72 h | `wl_last_lag6` | +1 | −0.0017 | −0.0010 |

The 24 h set is empty: the four 2020 violations were all at the 0.001 screening
threshold (GloFAS lags 7/10/14 d and `wl_last_lag6`), and none survived into 2021. The
"both years" requirement is what makes the rule conservative; the 24 h model therefore
receives **no constraint at all**.

### 2.3 Pre-registration amendment (made before fitting)

Because the target set is horizon-specific, it can be empty. An empty set makes
`b1_targeted` *identical* to `a0_unconstrained` and to `b2_targeted_placebo` at that
horizon, so the pooled criterion "beats its own placebo" is then impossible to pass by
construction rather than by evidence. After the audits and **before any Part B fit**, the
JSON was amended to designate **per-horizon analysis as primary** — the same four
criteria applied inside each horizon — with the pooled mean reported descriptively only.
The file was first created at 20:49:46 and amended at 20:50:42, 26 seconds before the run
started at 20:51:08. No model outcome existed when the choice was made.

### 2.4 Does the rule actually find persistent violations?

As a post-hoc check only (not used for selection), 6 of the 7 targeted 72 h features and
both targeted 48 h features were **again** material violations in the `val_2022` audit.
The seventh (`nb5_n_wl_danger`) was correct-signed but negligible (+0.0004) in 2022.
The selection rule therefore generalises across years; the problem is not that it found
noise.

---

## 3. Results

### 3.1 Scorecard

| model | 24 h | 48 h | 72 h | horizon mean | worst dev cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a0_unconstrained` (D16) | 0.970786 | 0.956485 | 0.945007 | **0.957426** | 0.924990 |
| `b1_targeted` | 0.970786 | 0.956015 | 0.944361 | **0.957054** | 0.923770 |
| `b2_targeted_placebo` | 0.970786 | 0.956321 | 0.944702 | 0.957270 | 0.923741 |
| `a1_strict` (Part A reference) | 0.964248 | 0.951577 | 0.936357 | 0.950727 | **0.928731** |

`b1` is **−0.00037** below D16 on the horizon mean. The 24 h row is exactly identical to
D16 (correlation 1.000000, mean |Δp| = 0, max |Δp| = 0), which is the expected no-op and
also a useful integrity check on the pipeline.

The targeted arm does **not** reproduce Part A's worst-cell gain: at 72 h on
`spatial_fold_1`, `a1_strict` is 0.928731 (+0.0037 vs D16) while `b1_targeted` is
0.923770 (−0.0012). Whatever bought spatial robustness in Part A, it was not the
correction of the audited violations.

### 3.2 Pre-registered decision, per horizon (primary)

| horizon | targeted | Δ val_2022 | Δ worst cell | min cell Δ | c1 mean | c2 worst | c3 safe | c4 placebo | verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 24 h | 0 | 0.00000 | 0.00000 | 0.00000 | PASS | FAIL | PASS | FAIL | **no-op; not evidence** |
| 48 h | 2 | −0.00047 | −0.00123 | −0.00123 | FAIL | FAIL | PASS | FAIL | **NOT SUPPORTED** |
| 72 h | 7 | −0.00065 | −0.00122 | −0.00122 | FAIL | FAIL | PASS | FAIL | **NOT SUPPORTED** |

Criterion 4 compares `b1` against all three placebo arms (`a3`, `a4`, `b2`). At both
testable horizons the strongest placebo is `b2`, the targeted signs flipped.

Pooled descriptively, `b1` also fails c1 (−0.00037), c2 (−0.00122) and c4
(`b2` is 0.00022 higher), and passes c3. **No horizon and no pooling supports the
targeted prior.**

### 3.3 Paired station bootstrap (`val_2022`, 400 resamples)

| comparison | 48 h Δ [95 % CI] | 72 h Δ [95 % CI] |
| --- | --- | --- |
| `b1` − `a0` | −0.00055 [−0.00205, +0.00031], P(Δ>0)=0.145 | −0.00068 [−0.00230, +0.00049], P(Δ>0)=0.133 |
| `b2` − `a0` | −0.00019 [−0.00120, +0.00058], P=0.360 | −0.00038 [−0.00203, +0.00099], P=0.298 |
| `b1` − `b2` | −0.00036 [−0.00162, +0.00035], P=0.200 | −0.00031 [−0.00194, +0.00124], P=0.348 |

Every CI includes zero. The data cannot distinguish the targeted prior from no prior, nor
from the **reversed** targeted prior. That is the sharpest statement of the result: at
this effect size, the sign direction of these seven features is unidentifiable.

### 3.4 Did the constraints bind at all?

Yes — but an order of magnitude less than Part A. Prediction churn against D16 on
`val_2022`:

| arm | 48 h mean \|Δp\| | 48 h rows \|Δp\|>0.005 | 48 h top-1 % Jaccard | 72 h mean \|Δp\| | 72 h rows \|Δp\|>0.005 | 72 h top-1 % Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a1_strict` | 0.00558 | 12.2 % | 0.504 | 0.00529 | 13.0 % | 0.419 |
| `b1_targeted` | 0.00143 | 4.8 % | 0.891 | 0.00221 | 7.4 % | 0.829 |
| `b2_targeted_placebo` | 0.00120 | 4.7 % | 0.901 | 0.00183 | 6.9 % | 0.800 |

The targeted constraint changes a meaningful minority of scores (4.8–7.4 % of rows by
more than 0.005) and preserves most of the top-1 % ranking (Jaccard 0.83–0.89). The strict
arm reshuffles roughly half of the top 1 % (Jaccard 0.42–0.50) and loses ~10× more PR-AUC.
The ranking cost of a hard prior is therefore concentrated in the *dominant* features,
not in the audited weak ones.

### 3.5 Why the constraints are easy to route around

Spearman |ρ| of each targeted feature with its closest **unconstrained** sibling on
`val_2022` (the unconstrained pool is horizon-specific, because the target set is):

| targeted feature | horizon | max \|ρ\| with unconstrained feature | substitute available to the trees |
| --- | --- | ---: | --- |
| `gf_up_ratio_lag10` | 48 h | 0.925 | `gf_up_ratio_14d` |
| `gf_up_ratio_lag14` | 48 h | 0.808 | `gf_up_ratio_14d` |
| `gf_up_ratio_lag10` | 72 h | 0.925 | `gf_up_ratio_14d` |
| `nb5_n_wl_danger` | 72 h | 0.861 | `nb3_n_wl_danger` |
| `ps_wet_root_anom` | 72 h | 0.929 | `ps_wet_top_anom` |
| `rain_30d` | 72 h | 0.905 | `ix_api09_x_wet_top` (tier `none`) |
| `wet_days_7` | 72 h | 0.602 | `api_07` |
| `wl_last_lag4` | 72 h | 0.998 | `wl_last_lag3` |
| `wl_last_lag6` | 72 h | 0.998 | `wl_last_lag7` |

Every targeted feature has at least one unconstrained substitute with |ρ| ≥ 0.60, and
most have one above 0.85. A tree that cannot use `wl_last_lag4` uses `wl_last_lag3`
(ρ = 0.998); a tree that cannot use `ps_wet_root_anom` uses `ps_wet_top_anom`
(ρ = 0.929). The constraint is satisfied in feature space and nearly invisible in
prediction space.

This also explains why Part A's *whole-cluster* constraints were expensive: constraining
all water-level lags at once removes the substitutes, so the model must actually change
its ranking. Partial constraints do not.

---

## 4. Pre-registered decision

The rule was fixed before the run: within a horizon, `b1` is supported only if it
(1) matches or beats D16 on `val_2022` PR-AUC, (2) beats D16 on the worst cell,
(3) regresses no cell by more than 0.005, and (4) beats all three placebo arms.

| criterion | 48 h | 72 h |
| --- | --- | --- |
| 1. `val_2022` ≥ D16 | **FAIL** (−0.00047) | **FAIL** (−0.00065) |
| 2. worst cell > D16 | **FAIL** (−0.00123) | **FAIL** (−0.00122) |
| 3. no cell regression > 0.005 | **PASS** (−0.00123) | **PASS** (−0.00122) |
| 4. beats all placebos | **FAIL** | **FAIL** |
| **verdict** | **NOT SUPPORTED** | **NOT SUPPORTED** |

**D16 remains the champion. No test read was performed and none is authorised.**

The one positive is real but modest: targeting cuts the cost of imposing monotonicity by
an order of magnitude, from **−0.0067** (42 signs) to **−0.0004** (2–7 signs), and keeps
every cell within 0.0013 of D16. If sign consistency is required for regulatory or
trust reasons, this is the way to buy it cheaply. It does not make the model more
accurate.

---

## 5. Why the targeted prior is inert

1. **The audited effects are tiny relative to the ranking drivers.** At 72 h the largest
   targeted violation is `ps_wet_root_anom` with a total PDP swing of 0.016 (2020) and
   0.004 (2021); the other six are ≤ 0.008. The strong, already-correct features swing
   0.50 (`wl_dist_to_danger`), 0.036 (`wl_pct_danger`), 0.016 (`wl_rise_1d`) and 0.015
   (`precip_mm`). Constraining a feature whose entire effect is 0.002 cannot reorder a
   ranking dominated by effects two orders of magnitude larger.
2. **Collinear substitutes make partial constraints non-binding.** The audit identifies
   the *marginal* response of one feature holding the others fixed. In a correlated
   cluster, that marginal response is not the model's effective channel: removing it
   simply promotes a sibling. The 0.998 correlation among water-level lags is the
   clearest case.
3. **"Material" at 0.001 is a screening threshold, not a decision-relevant effect.**
   It is useful for flagging sign instability, but a 0.001 probability change is far
   below the resolution that can move PR-AUC. The 2020/2021 replication requirement
   makes the flag reliable; it does not make the flagged feature important.
4. **The Part A worst-cell gain was not a violation fix.** `a1_strict` improves
   `spatial_fold_1` at 72 h by +0.0037; `b1_targeted` is −0.0012 there. The gain is
   associated with constraining the broader strict set — which includes station-level
   attributes (`danger_level`, `elevation_m`, `slope_deg`) — not with correcting the
   audited dynamic lags. That is a spatial-extrapolation effect, and it costs average
   accuracy.

The general lesson: **a sign violation is evidence about the model, not necessarily a
lever on its ranking.** Before spending accuracy on a constraint, check (a) the size of
the feature's marginal effect, (b) whether collinear unconstrained substitutes exist, and
(c) whether the violation persists across regimes. Part B is the control that shows what
happens when only (c) is enforced.

---

## 6. What survives, and what to do next

**Survives as a result:**

* A **validated audit-selection rule.** Violations detected in 2020 and 2021 reappear in
  2022 for 6/7 targeted 72 h features and 2/2 at 48 h. The physical-consistency audit is
  temporally stable, which strengthens it as an XAI deliverable.
* A **leverage diagnostic.** PDP sign + effect size + collinearity + prediction churn
  separates "real but irrelevant" violations from "worth fixing" ones. On this panel,
  the audited violations are real but irrelevant.
* A **cheap way to impose monotonicity if it is required.** Target only the stable
  violations: 0 signs at 24 h, 2 at 48 h, 7 at 72 h. Cost −0.0004 mean PR-AUC
  (indistinguishable from zero) versus −0.0067 for the 42-sign strict arm. This is an
  engineering result, not an accuracy result.

**What not to do next:**

* Do not run more hard-constraint variants on this panel. The targeted arm is the
  strongest version of the idea — minimal, audited, temporally replicated — and it is
  behaviourally inert. More variants will reproduce the same mechanism.
* Do not expect per-basin sign tables to improve accuracy either. They address
  heterogeneity of *sign*, but the sign of the dominant features is already correct and
  the violations are in redundant features. Per-basin tables would mainly be a
  trust/consistency artefact.

**If the monotonicity line is continued, the only remaining question worth compute is
cluster-level rather than feature-level:** constrain an entire collinear cluster (e.g.
all water-level lags, all GloFAS lags) or none of it. That would make the constraint
binding, but Part A already shows the binding version costs ~0.007. The likely outcome is
a more expensive null. The higher-value next steps are the ones already queued:
Experiment 2 (SAR inundation) for new information, and the system layer for deployment.

---

## 7. Reproducibility

```bat
:: 1. pre-2022 audits (selection data only; no 2022 rows)
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py audit --task t2020 --suffix 2020
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py audit --task t2021 --suffix 2021

:: 2. mechanical target sets + pre-registration JSON
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py targeted

:: 3. Part B fits (a-arms skip via the shared ledger)
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py run --arms b1_targeted b2_targeted_placebo

:: 4. scorecard, per-horizon verdicts, bootstrap, figures
.venv\Scripts\python.exe -u scripts\mi12_monotonic.py report
```

The ledger is shared with Part A: every `(task, horizon, arm)` fit is appended once and
skipped on restart. `b1`/`b2` at 24 h are exact no-ops, so their rows reproduce `a0` to
machine precision — a pipeline integrity check as well as a design consequence.

**Guards enforced in code:** the target set may only contain features with a non-zero
physical sign in `monotonic_signs_v1.json`; the target JSON records the selection rule,
both source audits and per-feature ΔP for both years; every evaluation mask is asserted
disjoint from `split == "test"`; the summary records `test_rows_read = 0`. The Part B
JSON was written before the first Part B fit (20:50:42 vs 20:51:08).
