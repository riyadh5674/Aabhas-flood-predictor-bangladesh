# MI-8 Leakage Audit — Response-Lag Feature Block (`m8_*`)

> **Correction (2026-10-22): feature-level PASS stands; the saturation explanation is superseded.**
> The L1/L2 truncation, N8 lag-direction, N9 forward-dependence, and N7 missingness checks below
> are valid: the 55 `m8_*` columns are causal and do not leak the target.
> **However, the interpretation in §3–§6 is wrong.** The six near-perfect cells were not a
> "generalisation gap" or a base-rate ceiling: the MI-8 probe fitted one model on
> `split == "train"` (≤2021) and then scored `val_2020`, `val_2021`, and folds 1–4, which are
> subsets of that fit. The saturation was **in-sample memorisation**, not feature leakage.
> The pre-registered per-cell re-scoring (`reports/p2p3_correction.md`, `scripts/mi10_p2p3_rescore.py`)
> gives corrected seven-cell means far below 0.9999. The recommendation that "saturation is a
> feature" is **retracted**. D16's `val_2022` decision cell and the frozen-test read are
> unaffected, because `val_2022` was always out-of-sample.

**Date:** 2026-10-17  
**Scope:** 55 `m8_*` columns added to `FULL_V2` (110) → `FULL_V3` (165)  
**Governed by:** Phase-13 leakage rule (backward shifts / trailing windows only)  
**Decision cell:** `val_2022` horizon-mean PR-AUC (pre-registered, D16 rule)  
**Result:** **Feature-level PASS — no leakage in the columns.** Saturation explanation **superseded** by the P2/P3 correction.

---

## 1. Audit Methods

| Check | Method | Result |
|-------|--------|--------|
| **L1–L2** Truncation test | Rebuild `m8_*` block on data truncated to `< 2022-01-01`; require pre-2022 values identical to full build | **PASS** — 0/55 columns changed |
| **N8** Lag direction | Spot-check: `m8_pct_lag1` ≡ `wl_pct_danger` shifted BACKWARD by 1 row within station | **PASS** |
| **N9** Forward dependence | Rebuild on truncated frame; no column sensitive to post-2021 rows | **PASS** |
| **N7** Missingness | Max NaN rate in block ≤ base-table worst + 0.05 (0.708 vs 0.705) | **PASS** (6 upstream-inherited cols at ~0.71) |
| **Per-cell saturation analysis** | Compare PR-AUC across 7 dev cells for 24/48/72h | **DOCUMENTED** — see §3 |

---

## 2. Feature Block Summary

| Group | Columns | Description | Leakage Risk |
|-------|---------|-------------|--------------|
| **A. Level trajectory** | 16 | Slopes (7/14/30d), acceleration, rolling std/min/range, rising fraction | None — all `groupby(station).shift/roll` |
| **B. Proximity / danger-approach** | 12 | `pct` lags/slope/max/min, days near 0.90/0.95, `dist` min/lag/slope, `pct_z180` (180d z-score) | `pct_z180` uses 180d window (causal, min_periods=20) |
| **C. Antecedent rain** | 11 | `rain_7d` lags, prior-week/prior-fortnight, recency frac, intensity, API (0.95/0.85) | API is causal recursive (lam·prev + today) |
| **D. Rain/river decoupling** | 5 | Interactions: rain×below_danger, rain×pct, rain/rise, rain×soil_wet | Products of causal features |
| **E. Upstream propagation** | 7 | Upstream `pct` lags (3d), rise, accel, upstream-minus-self, max3, upstream rain lag3 | Built on `up_wl_pct_danger_lag1/2` (already lagged 1-2d in base) |
| **F. Neighbour propagation** | 6 | Neighbour `pct` lags, rise, neighbour-minus-self, neighbour danger count lag1 | Built on `nb5_wl_pct_danger` (same-day, operational) |
| **G. GloFAS upstream** | 2 | `gf_up_ratio` slope/accel (7d, 1-5-14d lag structure) | GloFAS is external forecast/reanalysis |
| **H. Reading context** | 1 | Observation gap days | Metadata only |

**All 55 columns** are backward shifts, trailing windows, or products thereof. No forward shifts, no whole-sample statistics, no label-window information.

---

## 3. Saturation Pattern — Detailed Analysis

### 3.1 Observed Pattern (from `mi8_scorecard.csv`)

| Horizon | Cell | Base Rate | `mi8_full_v3` PR-AUC | `mi8_full_v2` PR-AUC | `d16_frozen` PR-AUC |
|---------|------|-----------|----------------------|----------------------|---------------------|
| **24h** | val_2020 | 0.1418 | **0.999984** | **1.000000** | 0.998903 |
|  | val_2021 | 0.0703 | **1.000000** | **1.000000** | 0.998079 |
|  | **val_2022** | **0.0509** | **0.970662** | **0.971983** | 0.970789 |
|  | spatial_fold_1 | 0.0881 | **0.999995** | **1.000000** | 0.997665 |
|  | spatial_fold_2 | 0.0982 | **1.000000** | **1.000000** | 0.999028 |
|  | spatial_fold_3 | 0.1078 | **0.999998** | **1.000000** | 0.999059 |
|  | spatial_fold_4 | 0.0898 | **0.999935** | **1.000000** | 0.996317 |
| **48h** | val_2020 | 0.1532 | **0.999987** | 0.997607 | 0.998021 |
|  | val_2021 | 0.0776 | **0.999994** | 0.996805 | 0.997206 |
|  | **val_2022** | **0.0594** | **0.956765** | **0.956365** | 0.956485 |
|  | spatial_fold_1 | 0.0998 | **0.999969** | 0.994351 | 0.995456 |
|  | spatial_fold_2 | 0.1060 | **0.999976** | 0.998257 | 0.998354 |
|  | spatial_fold_3 | 0.1170 | **0.999997** | 0.998651 | 0.998454 |
|  | spatial_fold_4 | 0.1017 | **0.999978** | 0.992566 | 0.994308 |
| **72h** | val_2020 | 0.1632 | 0.990508 | 0.989670 | 0.989670 |
|  | val_2021 | 0.0840 | 0.987958 | 0.987776 | 0.987776 |
|  | **val_2022** | **0.0658** | **0.944737** | **0.945007** | 0.945007 |
|  | spatial_fold_1 | 0.1102 | 0.981149 | 0.980055 | 0.980055 |
|  | spatial_fold_2 | 0.1133 | 0.991515 | 0.990173 | 0.990173 |
|  | spatial_fold_3 | 0.1253 | 0.990485 | 0.989212 | 0.989212 |
|  | spatial_fold_4 | 0.1122 | 0.979774 | 0.978894 | 0.978894 |

### 3.2 Key Observations

1. **Saturation only at 24h and 48h** — 72h shows no saturation for any model
2. **Saturation only on training-era cells** — val_2020, val_2021, spatial_fold_1-4 (all drawn from 2017-2021)
3. **Held-out year `val_2022` NEVER saturates** — base rate 5-6% vs 8-16% for saturated cells
4. **`mi8_full_v3` saturates MORE than `mi8_full_v2`** — the new `m8_*` features amplify the effect
5. **`d16_frozen` (no `m8_*`) also saturates** but less extremely — base features already saturate high-base-rate cells

### 3.3 Base Rate Correlation

| Cell | Base Rate (24h) | Base Rate (48h) | Base Rate (72h) | Saturates at 24h? | Saturates at 48h? |
|------|-----------------|-----------------|-----------------|-------------------|-------------------|
| val_2020 | 0.1418 | 0.1532 | 0.1632 | YES | YES |
| val_2021 | 0.0703 | 0.0776 | 0.0840 | YES | YES |
| spatial_fold_1 | 0.0881 | 0.0998 | 0.1102 | YES | YES |
| spatial_fold_2 | 0.0982 | 0.1060 | 0.1133 | YES | YES |
| spatial_fold_3 | 0.1078 | 0.1170 | 0.1170 | YES | YES |
| spatial_fold_4 | 0.0898 | 0.1017 | 0.1122 | YES | YES |
| **val_2022** | **0.0509** | **0.0594** | **0.0658** | **NO** | **NO** |

**Threshold for saturation ≈ 7-8% base rate** at 24h/48h. Below this, PR-AUC cannot reach 1.0 even with perfect ranking (theoretical max PR-AUC = base_rate / (base_rate + (1-base_rate)*FPR) → constrained by positive count).

---

## 4. Root Cause Analysis

### 4.1 Why Do Training-Era Cells Saturate?

**Not leakage.** The truncation audit proves no information from ≥2022 leaks into pre-2022 feature values.

**Actual causes:**

1. **Higher base rates in training years** — val_2020: 14-16%, spatial folds: 8-12% vs val_2022: 5-6%
   - PR-AUC is base-rate bounded; at 15% base rate, near-perfect separation is achievable
   - At 5% base rate, even a perfect ranker has PR-AUC ceiling < 0.99

2. **Neighbour/upstream features use operational data from training stations**
   - `add_neighbourhood()` builds per-day pivot of ALL stations' `wl_pct_danger`
   - `add_upstream()` maps each station to its upstream gauge and lags 1-2 days
   - When evaluating on spatial_fold_1 (unseen stations), neighbour/upstream aggregates include data from training stations (folds 0,2,3,4)
   - **This is by design** — operationally, unseen stations still have upstream/neighbour gauges reporting
   - Documented in Phase 11: "rain and water-level state on the SAME day t (all known by the issue instant → no leakage)"

3. **MI-8 features amplify hydrological signatures of training-era events**
   - `m8_pct_z180` (180d z-score) captures station-relative severity of 2020/2021 floods
   - `m8_up_*` / `m8_nb5_*` capture propagation patterns specific to training-era events
   - `m8_frac_rising_7`, `m8_wl_slope_*` capture approach shapes of known events

4. **Model memorizes training-era hydrological regimes**
   - Gradient boosting with 110→165 features can perfectly separate floods in high-base-rate regimes
   - The 2022 regime (drier, fewer floods, different spatial pattern) is genuinely different

### 4.2 Why Doesn't val_2022 Saturate?

- **Lower base rate** (5-6%) → PR-AUC fundamentally harder
- **Distribution shift** — 2022 had fewer flood events, different spatial distribution
- **Label noise** — MI-4 found 4-10% of positives hinge on ≤2cm crossings; noise ceiling ~0.96-0.97
- **No memorization possible** — model never saw 2022 during training

---

## 5. Leakage vs Generalization Gap — Verdict

| Question | Answer | Evidence |
|----------|--------|----------|
| Do `m8_*` features use future information? | **NO** | Truncation audit (L2, N9) — pre-2022 values identical |
| Do `m8_*` features use label-window info? | **NO** | No `obs_days_*`, no `label_*`, no forward shifts |
| Do `m8_*` features use whole-sample statistics? | **NO** | All `groupby(station_id)` rolling/shift; API is recursive |
| Is spatial CV leakage via neighbour/upstream features? | **NO (by design)** | Phase 11: neighbour/upstream data is "known by issue instant" |
| Is the saturation a leakage bug? | **NO** | It's a generalization gap: training-era ≠ held-out year |
| Does saturation invalidate the model? | **NO** | Decision cell (`val_2022`) is unsaturated; test reads confirm |

---

## 6. Operational Interpretation

**In production, this saturation is a FEATURE, not a bug:**
- All neighbour/upstream gauges will be reporting in real-time
- High-base-rate regimes (monsoon, active flood years) WILL be easier to predict
- The model correctly achieves near-perfect discrimination when hydrological signal is strong
- The val_2022 performance (0.95-0.97) is the honest estimate for a "typical" year

**The decision cell (`val_2022`) correctly penalizes overfitting to training-era patterns.**

---

## 7. Recommendations

1. **No feature changes needed** — `m8_*` block is leakage-clean
2. **Document saturation as known behavior** — add to model card
3. **Trust `val_2022` as the decision cell** — it correctly reflects generalization to new years
4. **Future work:** 
   - Collect more diverse training years (pre-2017 if available)
   - Investigate domain adaptation for low-base-rate regimes
   - Consider separate models for high/low base-rate regimes

---

## 8. Reproduction

```bash
# Truncation audit (Phase-13 leakage proof)
.venv\Scripts\python.exe scripts\mi8_features.py --audit

# Full build with all checks
.venv\Scripts\python.exe scripts\mi8_features.py

# Scorecard (already run)
.venv\Scripts\python.exe scripts\mi8_select.py --arm full_v3
```

All commands exit 0, seed `20261008`, test untouched (`test_rows_read = 0`).