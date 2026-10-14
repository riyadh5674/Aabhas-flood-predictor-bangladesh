# Phase 30 — Error analysis (stage 54; stage 46 evidence)

**Scope.** Where and *when* do the frozen leaders miss floods (false negatives, FN) and
raise false alarms (false positives, FP) on the P1 primary validation cell `val_2022`, and
why? This deepens the Phase-27 operational block (`reports/phase27_operational.csv`, stage
46) from *counts* to **named, quantified patterns** — each one either a **fix** (re-run the
affected phase) or a **stated limitation**.

**Provenance.**
- Dataset: `data/processed/experimental_dataset.csv`, SHA-256
  `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e`.
- Probabilities: cached Phase-28 P1 `val_2022` scores `reports/phase28_predictions.npz`
  (Phase 27 predictor path). **No model is retrained and no new prediction path is added.**
- Decision rule: the **frozen Phase-29** operating point — a row alarms when
  `p >= t_warn` for the row's own period (`monsoon` / `non-monsoon`), cost ratio
  `C_FN : C_FP = 10 : 1` (`reports/phase29_chosen_thresholds.csv`).
- Seed `20261008`; run `scripts/phase30_errors.py`, **exit 0**, wall ≈ 16 s, log
  `scripts/phase30_run.log`. **`test_rows_read = 0`** (the ≥ 2023 period stays frozen to
  Phase 33).

**Models.** The three frozen leaders —
`lightgbm` @ 24 h, `xgboost` @ 48 h, `xgboost_selected` @ 72 h — plus three contrast rows:
the D7 bar `logreg_linear` (L1), the best DL model `fusion_L3` (L3), and the crippled
control `static_only` (L3).

---

## 1. Method

1. **Alignment proof.** For each horizon the P1 `val_2022` evaluation rows are
   `evi = flatnonzero((split=="val") & ~station_holdout & (year==2022) & ~isnan(y))`.
   The npz `station_id`, `date` and `y` arrays match `d.iloc[evi]` **exactly** for 24/48/72 h
   (17 502 / 17 300 / 17 089 rows, 84 stations). Every row is therefore joined to its
   `experimental_dataset.csv` features by `(station_id, date)` with no ambiguity.
2. **Decision application.** `pred = p >= t_warn(period)`; outcomes
   `FN = y==1 & !pred`, `FP = y==0 & pred`. The all-season threshold is applied as a
   **sensitivity** only (the frozen rule is period-specific).
3. **Per-row diagnostics** computed for every FN/FP/TP/TN row: rainfall at `t`
   (`precip_mm`, `rain_1d/3d/7d/14d/30d`, `rain_max_7d`), river condition at `t`
   (`wl_dist_to_danger`, `wl_pct_danger`, `wl_rise_1d/3d`), elevation/slope/distance,
   missingness (`nan_frac` over the 62 FULL features; `n_readings`), the **consecutive
   positive-label run length** (event brevity), and the station's **training-fold flood
   count** (under-representation). For the DL model `fusion_L3` the sequence-window mask
   `sequence_mask_L3.npy` gives the share of broken (incomplete) windows.

---

## 2. FN / FP counts on `val_2022` (per-period thresholds)

Leader operating points (`reports/phase30_row_metrics.csv`):

| model | horizon | n | pos | TP | FP | **FN** | TN | recall | precision | FAR | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 24 h | 17 502 | 891 | 875 | 295 | **16** | 16 316 | 0.982 | 0.748 | 0.252 | 0.0178 |
| xgboost | 48 h | 17 300 | 1 027 | 1 008 | 485 | **19** | 15 788 | 0.982 | 0.675 | 0.325 | 0.0298 |
| xgboost_selected | 72 h | 17 089 | 1 124 | 1 076 | 518 | **48** | 15 447 | 0.957 | 0.675 | 0.325 | 0.0324 |

Contrast rows (same rule):

| model | 24 h recall / precision | 48 h | 72 h |
| --- | --- | --- | --- |
| logreg_linear (D7 bar) | 0.975 / 0.678 | 0.967 / 0.622 | 0.940 | 0.633 |
| fusion_L3 (best DL) | 0.976 / 0.687 | 0.972 / 0.657 | 0.948 / 0.692 |
| static_only (control) | 0.751 / 0.302 | 0.738 / 0.356 | 0.706 | 0.421 |

**Sensitivity to the threshold convention** (frozen per-period vs a single all-season
threshold): lightgbm 24 h 16 vs 18 missed (per-period slightly better), xgboost 48 h 19 vs
19 (identical), xgboost_selected 72 h 48 vs 38 missed. The 72 h difference is the expected
cost trade-off: the per-period rule raises the non-monsoon threshold to 0.166, so it
alarms less off-season and trades ~10 extra misses for ~fewer false alarms (see Phase 29
§2–3). The pattern analysis below uses the **frozen per-period** rule.

---

## 3. The missed-flood (FN) table

`reports/phase30_fn_rows.csv` — all **83** FN rows of the three frozen leaders, with the
features above. The three leaders' misses are the small, hard tail of an otherwise
high-recall operating point. Representative rows:

| model | horizon | station | river | date | p | t_warn | wl below danger (m) | wl % danger | rain_30d (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 24 h | 70 | Jadukata | 2022-06-11 | 0.003 | 0.0689 | 1.49 | 0.815 | 1 201 |
| lightgbm | 24 h | 59 | Surma | 2022-06-14 | 0.019 | 0.0689 | 0.52 | 0.952 | 1 233 |
| lightgbm | 24 h | 21 | Muhuri | 2022-06-19 | 0.034 | 0.0689 | 1.14 | 0.909 | 685 |
| xgboost | 48 h | 96 | Surma Old Course | 2022-05-18 | 0.007 | 0.0645 | 0.76 | 0.884 | 833 |
| xgboost | 48 h | 65 | Surma | 2022-06-11 | 0.009 | 0.0524 | 0.41 | 0.947 | 1 201 |
| xgboost_selected | 72 h | 22 | Padma | 2022-08-22 | 0.007 | 0.0697 | 0.28 | 0.930 | 259 |

The recurring signature: a **wet catchment with the river *just below* danger at issue
time**. Full table: `reports/phase30_fn_rows.csv` (83 rows).

**All-season-threshold sensitivity** (frozen per-period vs a single all-season rule):

| model | horizon | FN (per-period) | FN (all-season) | recall (per-period) | recall (all-season) | FP (per-period) | FP (all-season) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 24 h | **16** | 18 | 0.982 | 0.980 | 295 | 297 |
| xgboost | 48 h | **19** | 19 | 0.982 | 0.982 | 485 | 503 |
| xgboost_selected | 72 h | **48** | 38 | 0.957 | 0.966 | 518 | 638 |

The per-period rule matches or beats the all-season rule on recall at 24 h and 48 h; at
72 h it trades 10 extra misses for 120 fewer false alarms (its non-monsoon threshold is
0.166, the all-season one 0.061). This is the Phase-29 cost trade-off, not a regression.

---

## 3b. FP tables

FP rows are numerous, so the table keeps the highest-confidence false alarms
(`reports/phase30_fp_rows.csv`, ≤ 400 per cell, 1 095 rows). Their profile is the mirror
image of the FN: median `wl_pct_danger` **0.986** (the river is *just below* danger), high
`rain_7d` (median 98 mm). Full per-station FN/FP behaviour in
`reports/phase30_by_station.csv` (756 rows).

---

## 4. Named patterns (quantified)

All numbers from `reports/phase30_patterns.csv` and
`reports/phase30_outcome_profile.csv`. "Frozen leaders pooled" = the three operating
points (83 FN, 83 = 16 + 19 + 48; 1 298 FP).

| # | finding | statistic | contrast | disposition |
| --- | --- | --- | --- | --- |
| **A** | **Missed floods occur while the river is still below danger but rising** | median `wl_dist_to_danger` **+0.76 m**; **0.0 %** of FN had `wl<=danger` | TP median −0.35 m, 82.5 % below danger | limitation (response-lag) |
| **F** | **False alarms fire while the river is still below danger at `t`** | median `wl_pct_danger` **0.983**; 80.9 % of FP below danger | TN median 0.710 | limitation (over-alarm) |
| **E** | **Stations seen with fewer training floods are missed more** | mean miss rate **0.230 vs 0.016** (low vs high train-flood quartile); Spearman ρ **−0.501** | 36 stations | limitation |
| B | Misses grow with lead time | **16 / 19 / 48** (24/48/72 h); recall 0.982 / 0.981 / 0.957 | — | limitation |
| G | Short/isolated flood events are the hard labels | FN in runs ≤ 2 rows **14.5 %** vs TP **2.8 %**; median run 7 vs 14 | — | limitation |
| G2 | Misses sit in **wetter** conditions, not drier | median `rain_7d` **102 mm** (FN) vs **86 mm** (TP); `rain_30d` 477 vs 378 mm | — | limitation |
| I | Misses cluster spatially | **28.9 %** of FN in the top-2 rivers (Jamuna, Teesta) | — | limitation |
| C | Monsoon concentration of misses is a **base-rate effect**, not a bias | FN monsoon share 0.747 vs positive-row share 0.767; recall monsoon **0.973** vs non-monsoon **0.970** | — | not a defect |
| D | Missingness is a **weak** discriminator | median `nan_frac` identical (0.0645); share with any NaN 0.735 vs 0.657 | — | weak |
| H | Broken DL sequence window is **not** a consistent miss driver | FN vs TP broken-window share: **0.143 vs 0.082** (24 h), **0.034 vs 0.080** (48 h), **0.034 vs 0.077** (72 h) | — | not supported |

### 4.1 The top three patterns in detail

**(A) Response-lag FN — the dominant, actionable pattern.** Essentially **every** leader
miss (0.0 % of 83 FN had `wl_pct_danger < 1`) happens with the river **still below danger**
at issue time (median 0.76 m below), while **68.7 %** are already rising (`wl_rise_3d > 0`).
These are genuine **early-stage** events: the rain has fallen (G2: FN are *wetter* than TP)
but the flood wave has not yet reached the gauge. This is a **limitation** of a purely
observational early-warning signal, not a bug — the information needed (future rainfall,
upstream propagation) is not in the feature set. Fix candidates: upstream-lag features
(Phase 11 space), or accept as a stated limitation.

**(F) Over-alarm FP — the precision cost of FN ≫ FP.** **80.9 %** of the 1 298 leader false
alarms fire with the river *just below* danger (median `wl_pct_danger` 0.983). They are the
direct price of the Phase-29 cost-optimal, very-high-recall threshold: near-miss events
(the river approaches but does not cross). This is **expected and intended** under
`C_FN : C_FP = 10 : 1`; it is a **fix-by-policy** (raise the ratio / threshold) rather than
a modelling defect.

**(E) Under-represented stations.** Miss rates are **~14× higher** on stations with the
fewest training-fold floods (0.230 vs 0.016; ρ = −0.501), concentrated on small/transboundary
rivers (Muhuri, Khowai, Atrai, Jadukata — all with few positives). This is a **fix**:
targeted data acquisition / per-station evaluation for low-prevalence stations.

**Secondary.** (B) recall decays with lead time and the extra 72 h misses are the same
response-lag type; (G) brief single-day events are over-represented among misses; (I)
misses cluster in the Jamuna/Teesta corridor.

**Negative results (stated limitations).** (C) the seasonal concentration of misses is
purely the monsoon base rate (FN monsoon share 0.747 vs positive-row share 0.767 — leader
recall is season-neutral, 0.973 vs 0.970), so "monsoon is harder" is **not** supported for
the leaders. (D) missingness is only a weak
discriminator (medians identical). (H) the "broken sequence window" hypothesis for the DL
model is **not supported** — at 24 h FN are slightly more likely to have an incomplete
window (0.143 vs 0.082) but the relationship reverses at 48/72 h.

---

## 5. Verdict and disposition

- The three frozen leaders are **strongly early-stage**: their FN and FP are two sides of
  one phenomenon — the cost-optimal operating point sits where the river is **just below
  danger**, so early-stage floods are missed (A) and near-misses are alarmed (F).
- **Fixes identified:** (E) target low-prevalence stations; (A) add upstream/future-rain
  signal; (F) is a policy knob, not a fix.
- **Limitations stated:** A/G/G2 (observational signal cannot see the not-yet-arrived
  wave), C/D/H (tested and not supported).
- No phase is invalidated — the Phase-16 threshold lesson, Phase-27 operational numbers and
  the Phase-29 frozen decisions all stand; Phase 30 adds the *mechanism*.
- Test untouched (`test_rows_read = 0`); the ≥ 2023 period stays frozen to Phase 33.

---

## 6. Provenance (files)

- Code: `scripts/phase30_errors.py`; run log `scripts/phase30_run.log` (exit 0, ~16 s,
  seed 20261008).
- Tables: `reports/phase30_row_metrics.csv` (18 cells),
  `reports/phase30_outcome_profile.csv` (72 rows),
  `reports/phase30_fn_rows.csv` (83), `reports/phase30_fp_rows.csv` (1 095),
  `reports/phase30_by_station.csv` (756), `reports/phase30_patterns.csv` (12).
- Figure: `figures/phase30_errors.png` (leader errors; `wl_dist_to_danger` FN vs TP;
  `rain_7d` FN vs TP; `fusion_L3` broken-window share).
- Summary: `reports/phase30_summary.json` (`test_rows_read = 0`).
- Verifying notebook: `notebooks/30_errors.ipynb` (generated by
  `scripts/build_phase30_notebook.py`).
