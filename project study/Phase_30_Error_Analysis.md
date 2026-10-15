# Phase 30 — Error Analysis (where and *why* the leaders still miss)
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase30_errors.md`, `scripts/phase30_errors.py`,
> `reports/phase30_patterns.csv`, `reports/phase30_fn_rows.csv`. Previous study: Phase 28–29
> (calibration & thresholds).

---

## 0. The 60-second version

The leaders are strong — at the frozen operating point **lightgbm (24 h)** misses only **16 of
891** floods (recall 0.982). Phase 30 asks: **what do those misses look like, and why?**

It turns the Phase-27 operational *counts* into **named, quantified patterns**, each labelled a
**fix** or a **stated limitation**:

- **(A) The dominant pattern — response lag.** **Every** leader miss happens while the river is
  **still below danger at issue time** (median 0.76 m below), but **68.7 %** are already rising.
  The rain has fallen (misses are *wetter* than hits) but the flood wave hasn't reached the
  gauge yet. **Limitation** of an observational-only signal.
- **(F) False alarms are the mirror image.** **80.9 %** of false alarms fire with the river
  *just below* danger (median 98.3 % of danger). This is the **intended price** of the 10:1
  cost-optimal, very-high-recall threshold — a **policy knob**, not a bug.
- **(E) Under-represented stations are missed ~14× more** (miss rate 0.230 vs 0.016, ρ = −0.501)
  — small/transboundary rivers with few training floods. A **fix**: targeted data.

Three hypotheses were **tested and not supported** — and stated as such: monsoon is *not*
harder for the leaders; missingness is a weak discriminator; broken DL sequence windows do not
drive misses.

---

## 1. Where Phase 30 sits

| Phase | What | Plain words |
|---|---|---|
| 28–29 | Calibration & thresholds | Honest probabilities, warning rule |
| **➜ 30** | **Error analysis** | **Where/why do we still miss?** |
| 31–32 | Robustness & ablation | Does it hold up? What matters? |
| 33 | Select + freeze + test | Pick champion, read test once |

Closes workflow **stage 54** (and supplies **stage 46** evidence).

---

## 2. Key words

| Word | Meaning |
|---|---|
| **FN** | False negative — a real flood we did **not** warn (a miss). |
| **FP** | False positive — a false alarm. |
| **TP / TN** | True positive / true negative. |
| **Recall** | Fraction of real floods warned. |
| **Precision** | Fraction of warnings that were real. |
| **FAR** | False-alarm rate. |
| **Response lag** | Delay between rainfall and the flood wave reaching the gauge. |
| **Event run length** | Consecutive days a flood lasts (brief events are hard). |
| **Under-represented station** | A station with few flood examples in training. |
| **wl_dist_to_danger** | Metres the water level is above (+) or below (−) the danger line. |
| **wl_pct_danger** | Water level as % of danger level. |
| **Disposition** | Is a pattern a **fix** or a **stated limitation**? |
| **Negative result** | A hypothesis the data did *not* support (reported honestly). |

---

## 3. Method (strictly on the frozen evidence)

- **No retraining, no new prediction path.** Probabilities come from the cached
  `phase28_predictions.npz` (the Phase-27 predictor).
- **Decision rule = the frozen Phase-29 operating point:** a row alarms when
  `p ≥ t_warn` for its *own period* (monsoon / non-monsoon), at `C_FN:C_FP = 10:1`.
- **Alignment proved:** the cached `station_id`/`date`/`y` arrays match the dataset rows
  **exactly** (17,502 / 17,300 / 17,089 rows, 84 stations).
- **Models:** the three frozen leaders — `lightgbm`@24 h, `xgboost`@48 h,
  `xgboost_selected`@72 h — plus contrasts `logreg_linear` (D7 bar), `fusion_L3` (best DL),
  `static_only` (control).
- **Test untouched** (`test_rows_read = 0`).

---

## 4. The leader operating points

| Model | Horizon | pos | TP | FP | **FN** | Recall | Precision | FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lightgbm | 24 h | 891 | 875 | 295 | **16** | 0.982 | 0.748 | 0.252 |
| xgboost | 48 h | 1,027 | 1,008 | 485 | **19** | 0.982 | 0.675 | 0.325 |
| xgboost_selected | 72 h | 1,124 | 1,076 | 518 | **48** | 0.957 | 0.675 | 0.325 |
| *static_only* (control) | 24 h | — | — | — | — | *0.751* | *0.302* | *—* |

The leader misses are the **small, hard tail** of an otherwise very-high-recall operating point.

**Threshold-convention sensitivity:** per-period vs all-season thresholds are nearly identical
at 24/48 h; at 72 h the per-period rule trades ~10 extra misses (48 vs 38) for ~120 fewer false
alarms — the Phase-29 cost trade-off, **not** a regression.

---

## 5. The missed-flood (FN) signature

All **83** leader FN rows are tabulated. Representative:

| Model | Horizon | station | river | date | p | t_warn | wl below danger (m) | wl % danger | rain_30d (mm) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| lightgbm | 24 h | 70 | Jadukata | 2022-06-11 | 0.003 | 0.0689 | 1.49 | 0.815 | 1,201 |
| lightgbm | 24 h | 59 | Surma | 2022-06-14 | 0.019 | 0.0689 | 0.52 | 0.952 | 1,233 |
| xgboost | 48 h | 96 | Surma Old Course | 2022-05-18 | 0.007 | 0.0645 | 0.76 | 0.884 | 833 |
| xgboost_selected | 72 h | 22 | Padma | 2022-08-22 | 0.007 | 0.0697 | 0.28 | 0.930 | 259 |

**The recurring signature: a wet catchment with the river *just below* danger at issue time.**

**False alarms** are the mirror image: median `wl_pct_danger` **0.986**, high `rain_7d`
(median 98 mm) — near-miss events where the river approaches but does not cross.

---

## 6. The named patterns

| # | Finding | Statistic | Contrast | Disposition |
|---|---|---|---|---|
| **A** | **Missed floods occur while the river is below danger but rising** | median **+0.76 m** below; **0.0 %** of FN had `wl≤danger` | TP median −0.35 m, 82.5 % below danger | **limitation** (response lag) |
| **F** | **False alarms fire with the river just below danger** | median `wl_pct_danger` **0.983**; 80.9 % below | TN median 0.710 | **limitation** (over-alarm) |
| **E** | **Low-training-flood stations are missed more** | miss rate **0.230 vs 0.016**; Spearman ρ **−0.501** | 36 stations | **fix** |
| B | Misses grow with lead time | 16 / 19 / 48 | recall 0.982 / 0.981 / 0.957 | limitation |
| G | Brief/isolated events are hard | FN in runs ≤ 2 rows **14.5 %** vs TP **2.8 %** | median run 7 vs 14 | limitation |
| G2 | Misses sit in **wetter** conditions | median `rain_7d` **102 mm** (FN) vs **86 mm** (TP) | `rain_30d` 477 vs 378 | limitation |
| I | Misses cluster spatially | **28.9 %** in top-2 rivers (Jamuna, Teesta) | — | limitation |
| C | Monsoon miss concentration is a **base-rate effect** | FN monsoon share 0.747 vs positive share 0.767; recall 0.973 vs 0.970 | — | **not a defect** |
| D | Missingness is a **weak** discriminator | identical median `nan_frac` (0.0645) | — | weak |
| H | Broken DL window is **not** a consistent miss driver | 0.143 vs 0.082 (24 h), reverses at 48/72 h | — | **not supported** |

### The top three in detail

**(A) Response-lag FN — the dominant, actionable pattern.** Essentially **every** miss happens
with the river still below danger (median 0.76 m) while **68.7 %** are already rising. The rain
has fallen (G2) but the **flood wave hasn't arrived**. The information needed — future rainfall,
upstream propagation — is **not in the feature set**. Fix candidates: upstream-lag features, or
accept as a stated limitation.

**(F) Over-alarm FP — the precision cost of FN ≫ FP.** 80.9 % of false alarms fire with the river
just below danger. Expected and **intended** under 10:1 — fix-by-policy (raise the ratio), not a
modelling defect.

**(E) Under-represented stations.** Miss rates ~**14× higher** where training floods are few —
small/transboundary rivers (Muhuri, Khowai, Atrai, Jadukata). **Fix:** targeted data /
per-station evaluation.

---

## 7. Problems & solutions

| Problem | Solution |
|---|---|
| "Counts" don't explain *why* we miss | Per-row diagnostics → **named, quantified patterns** (A–I). |
| Could have retrained / added a prediction path | Reused the cached Phase-28 probabilities; **no retraining**. |
| A pattern could be mistaken for a bug | Each labelled **fix** or **stated limitation**. |
| Hypotheses could be asserted not tested | C/D/H explicitly **tested and not supported**. |
| Test could creep in | Frozen to Phase 33; `test_rows_read = 0`. |

---

## 8. Files referenced

| File | Purpose |
|---|---|
| `reports/phase30_errors.md` | Full Phase 30 report |
| `scripts/phase30_errors.py` | Driver (exit 0, ~16 s) |
| `reports/phase30_row_metrics.csv` | 18 leader/contrast cells |
| `reports/phase30_fn_rows.csv` | All 83 FN rows |
| `reports/phase30_fp_rows.csv` | 1,095 top FP rows |
| `reports/phase30_by_station.csv` | 756 per-station rows |
| `reports/phase30_patterns.csv` | The 12 named patterns |
| `reports/phase30_summary.json` | `test_rows_read = 0` |
| `notebooks/30_errors.ipynb` | Verification notebook |

---

## 9. Verdict

- The leaders are **strongly early-stage**: their FN and FP are **two sides of one phenomenon** —
  the cost-optimal threshold sits where the river is *just below* danger, so early-stage floods
  are missed (A) and near-misses are alarmed (F).
- **Fixes identified:** (E) target low-prevalence stations; (A) add upstream / future-rain
  signal; (F) is a **policy knob**, not a fix.
- **Limitations stated:** A/G/G2 (an observational signal cannot see the not-yet-arrived wave);
  C/D/H (tested, not supported).
- **No phase invalidated** — Phase 30 adds the *mechanism* behind the Phase-27/29 numbers.
- **Test untouched.**

**Bottom line:** We now know *exactly* which floods we miss and why: mostly fast catchments where
the river is still below danger when we issue. That is a real, bounded limitation of an
observational early-warning system — stated, not hidden — and it points precisely at what extra
data (upstream gauges, forecast rainfall) would buy.

---

**Next in the series:** Phase 31–32 — Robustness (does the win hold under extreme rain, missing
data, and a stricter issue-time rule?) and Ablation (which feature groups actually carry the
skill?). Then Phase 33 — Selection + the single frozen test read.
