# Phase 5–9 — Exploratory Data Analysis (workflow stages 13–18)

| | |
| --- | --- |
| Module | `notebooks/03_eda.ipynb` (22 cells, **13 code executed end-to-end**) |
| Figures | `figures/phase5_*` (univariate, temporal, D5, D6, spatial, target-vs-feature, baseline, correlation, VIF) |
| Verdict | **20 PASS / 0 FAIL** · 8 findings recorded (E1–E7) |
| Data state | model-ready **200,665 × 54**, 133 stations, 2017-01-06 … 2026-10-05, post-F3+F8 |

Decisions baked in: **D1** JJAS monsoon · **D2** all horizons equal · **D5** drift decomposition ·
**D6** informative-missingness test · **D7** persistence baseline per horizon *and* per year.

> **Post-script (2026-10-08):** Phase 11 appended 19 spatial columns (→ 73 total). Labels are
> byte-identical, so every number below still holds; what changed is only the *candidate
> feature pool* — the §6 keep-list and §8 action 3 must be re-derived including the spatial
> groups (`reports/phase11_spatial_features.md` §7).

---

## 1. Univariate analysis (stage 13)

All key variables are finite, non-constant, right-skewed where expected (U1/U2 PASS).

| variable | median | mean | p99 | max | note |
| --- | --- | --- | --- | --- | --- |
| `wl_max` | 7.09 m | 10.69 m | 64.56 m | 70.81 m | highest danger level 70.3 m |
| `wl_pct_danger` | 0.74 | 0.70 | 1.13 | 3.35 | a *fraction* (finding F2) |
| `wl_dist_to_danger` | +2.85 m | +3.26 m | −0.82 m (p1) | −27.24 (max depth above danger) | |
| `rain_7d` | 70.6 mm | 90.6 mm | 424 mm | 2218 mm | p99 = "extreme rainfall" line |
| `rain_30d` | 342 mm | 380 mm | 1334 mm | 3720 mm | monsoon-accumulation |
| `wl_rise_1d` | −0.01 m | 0.006 m | **max 7.79 m/day** | — | F8-verified: 0 jumps >10 m/day |

`n_readings` 3–8, `wet_days_7` median 7/7, `tmin_c` 24.6 °C median. 244 rows carry NaN
weather (F1 residual window) — flagged for the logreg baseline (G2).

## 2. Temporal analysis (stage 14)

**Yearly flood frequency** (rate = mean label on labelled rows):

| year | 24h | 48h | 72h | n (24h) |
| --- | --- | --- | --- | --- |
| 2017 | 13.37 % | 15.20 % | 16.81 % | 17,447 |
| 2019 | 5.01 % | 5.77 % | 6.47 % | 18,306 |
| 2020 | 13.57 % | 14.77 % | 15.83 % | 20,049 |
| 2021 | 6.78 % | 7.59 % | 8.32 % | 21,743 |
| 2022 | 5.08 % | 5.96 % | 6.64 % | 21,878 |
| 2023 | 0.68 % | 0.93 % | 1.19 % | 23,858 |
| 2024 | 2.54 % | 3.11 % | 3.69 % | 24,341 |
| 2025 | 1.17 % | 1.49 % | 1.79 % | 28,250 |
| 2026 | 1.27 % | 1.60 % | 1.90 % | 24,689 |

**Season (D1 JJAS):** monsoon 72h rate 9.79 % vs non-monsoon **1.80 %** — 5.4×; 24 h 7.93 % vs
1.29 %. July peaks at 12.8 % (72h); Jan–Feb have literally **zero** floods in the labelled
record (but winter observation is sparse, §4 D6).

**Episodes** (consecutive positive days per station): median 3 d (24h) / 5 d (48h) / 6 d (72h);
longest single run 144–146 d (a prolonged 2020 tidal-station event, Mongla class). Extreme
rainfall days (station `rain_7d` > p99 = 424 mm): 2,007 station-days across 327 calendar days.

### D5 — drift decomposition (decision)

**Fixed panel = 31 stations** (≥200 observed days *and* labels in every year, 2018 excluded).
Stations reporting per year grew 86 → 141. Pooled 72h base rate falls **14.9 pp** 2017→2026,
but the **fixed panel also falls 19.1 pp** ⇒ the collapse is **mostly within-station regime
change** (2017/2020 mega-floods vs 2023–26 quiescence), *not* network composition; composition
actually contributes −4.2 pp (newer stations skew toward low-risk basins). Consequence: any
pooled-across-years number mixes regimes; **per-year, per-station reporting is mandatory** —
this is now a hard rule (D8), matching the earlier R1 verdict.

### D6 — is the observation process informative? (decision)

Observation is **heavily seasonal**: monsoon 99.0 % of station-days observed vs non-monsoon
44.7 % (+54.3 pp); unobserved days are dry (rain_30d median observed **340 mm** vs unobserved
**12 mm**; KS = 0.73). Month chi² p ≈ 0 (but with n≈330k, effect size is what matters).

Verdict: **informative — in the favourable direction.** Missing days are winter, not flood
days, so the labelled sample **over-represents the flood season**, meaning the model is
trained on the regime it must predict; but it also means (a) dry-season labels are scarce
accuracy is mostly wet-season accuracy, and (b) **unobserved ≠ 0**: coverage/`n_readings` are
observation-process signals, never label information (§9 action 5).

## 3. Spatial analysis (stage 15)

All 133 labelled stations in-bbox (SP1); all three horizons cover the same 133 stations (SP2).

Top stations by 72h flood rate (n ≥ 200 labelled days):

| station | division | rate |
| --- | --- | --- |
| Mongla | Khulna | **83.8 %** |
| Khulna | Khulna | 52.3 % |
| Daulatkhan | Barishal | 40.4 % |
| Companyganj | Chittagong | 35.8 % |
| Dalia | Rangpur | 29.2 % |
| Sureshswar | Dhaka | 24.8 % |
| Kalaroa | Khulna | 20.4 % |
| Barishal | Barishal | 20.0 % |

Khulna/Barishal (tidal) flood on tide + surge schedules — essentially standing water;
Rangpur (Teesta) floods on July–Sep rainfall. Flood risk is dominated by a handful of
basins, so **station identity is a first-order feature** (MI: `lat`, `lon`, `danger_level`
all ≤ 0.046 — location matters, but less than water-level state). Choropleth now unblocked — `admin/geoBoundaries-BGD-ADM{1,2}` acquired with Phase 11 (see `reports/phase1_data_inventory.md` D7).

## 4. Target vs feature analysis (stage 16)

Univariate (oriented) AUC on 72h — every feature × every horizon scored, all finite (TF1):

| feature | AUC | KS | med0 | med1 |
| --- | --- | --- | --- | --- |
| `wl_dist_to_danger` | **0.968** | 0.843 | +3.04 m | −0.26 m |
| `wl_pct_danger` | 0.959 | 0.807 | 0.72 | 1.03 |
| `wl_exceed_days_30` | 0.908 | 0.773 | 0 d | 11 d |
| `rain_30d` | 0.702 | 0.289 | 337 mm | 471 mm |
| `rain_14d` | 0.695 | 0.259 | 151 mm | 229 mm |
| `rain_7d` | 0.682 | 0.242 | 69 mm | 112 mm |
| `wet_days_7` | 0.663 | 0.294 | 7 | 7 |
| `tmin_c` | 0.642 | ... | 25.5 °C | 26.1 °C |

MI mirrors the ranking (`wl_dist_to_danger` 0.159, `wl_pct_danger` 0.157,
`wl_exceed_days_30` 0.105, then rain/location). Pearson on the label: `wl_exceed_days_30`
0.61, `wl_pct_danger` 0.37, rains ≤ 0.18. The three lenses agree: **today's water level
relative to the danger level is the single dominant signal; accumulation windows (30 d
exceedance, rain_30d) add the second layer; then raw rainfall; temperature contributes a
fraction of that.** `n_readings` alone scored below the 0.60 flag → the observation-process
warning (E7) did not trigger at 72h, but it stays on the stage-19 watchlist for the leak audit.

### D7 — persistence baseline (decision)

Persistence = "will it flood if it looks like it will flood": `wl_pct_danger` alone.
**(D7) Every Phase-16+ model must be reported against these per-year numbers.**

| horizon | pooled AUC | pooled PR-AUC | climatology | skill (PR-AUC ÷ base rate) |
| --- | --- | --- | --- | --- |
| 24 h | **0.987** | 0.904 | 0.050 | **18.2×** |
| 48 h | 0.974 | 0.869 | 0.057 | 15.3× |
| 72 h | 0.959 | 0.836 | 0.063 | 13.2× |

Per year, AUC is remarkably stable (0.912–0.990), but **PR-AUC is not** — it collapses from
0.86–0.93 (2017/2020) to **0.51–0.63 (2023)** purely because the base rate collapses while
the false-positive *count* barely moves. Any model claiming to add value must push per-year
PR-AUC (or precision-k) above these lines in the scarce years — the hard part of the problem.
This also validates D8: pooled metrics on this data are misleading.

## 5. Correlation analysis (stage 17)

CO1/CO2 PASS (38×38 matrices, unit diagonal, symmetric, finite; every feature correlated
with every label). Largest positive label correlations: `wl_exceed_days_30` 0.61, `wl_pct_danger`
0.37, `wl_dist_to_danger` 0.36, rains 0.15–0.18. Per workflow: high correlation ≠ importance
and low correlation ≠ uselessness (e.g. `doy_sin/doy_cos` matter through nonlinear interactions);
used correlation only to understand linear relationships.

## 6. Multicollinearity & redundancy (stage 18)

- **160 predictor pairs** |Spearman ρ| > 0.8, **137** > 0.95 (e.g. `precip_mm` ≡ `rain_1d`
  ρ=1.000; `wl_mean` ≡ `wl_first` ρ=0.999).
- **VIF > 10** for 22 features (all `wl_*` levels/lags/rolls, `rain_1d/7d`, `danger_level`).
- Constants confirmed and dropped (G3): `day_observed`, `source`. `obs_days_*` never admitted.
- Iterative VIF pruning leaves a **17-feature keep list**: `lat, lon, n_readings, wl_rise_1d,
  wl_rise_3d, wl_pct_danger, wl_exceed_days_30, precip_mm, rain_3d, rain_14d, rain_30d,
  rain_max_7d, wet_days_7, tmax_c, tmin_c, doy_sin, doy_cos` (max residual VIF 5.5).

Interpretation: the lags/levels are near-duplicates of `wl_pct_danger` (fine) — trees do not
care, but the **logistic-regression baseline must use a pruned set** (this list) or VIF will
instability. The kept set still contains the full *information* (today + rises + antecedent
30 d + rain + temperature + season + location).

## 7. Conclusions (analysis → decision)

1. **The problem is learnable in distribution:** `wl_pct_danger` alone (a *feature*, not info
   from the future) already reaches ROC-AUC 0.959–0.987 — but it is exactly the persistence
   baseline, so it is the **bar**, not the win. Models must beat it per year.
2. **The real difficulty is the rare/quiet years:** 2023–2026 base rates ~1–4 %, PR-AUC decays;
   evaluation must be rolling-origin per year (D8) with PR-AUC/recall emphasis.
3. **Drift is within-station, not network composition (D5/E2)** — a stable model built on
   today's hydrology state will transfer across years; station/season fixed effects still needed.
4. **Missingness is informative but safe (D6/E3):** gaps are winter/dry, labelled data
   over-represents the flood season; there is no flood-hidden blackout like 2018 in the data.
5. **Observation-process features (`n_readings`, coverage) must be handled explicitly in the
   stage-19 leakage audit** — they carry small hazard signal and could let a model learn
   FFWC's publication policy instead of the river.
6. **Data clean state reached:** F3 + F8 defects (32 + 17 false labels) fixed at data level;
   no further label-poisoning defects found in EDA.

## 8. Action plan (what the next module does)

| # | Action | Module/stage | Decision |
| --- | --- | --- | --- |
| 1 | Formal **data-leakage audit** — `obs_days_*` ban, `n_readings` admissibility ruling (E7/§5), feature-time-bound check; writes the whitelist again | Phase 11 / stage 19 | whitelist to be tracked |
| 2 | **Split design**: per-station-year rolling origin, monsoon block (JJAS), 2018 hole documented, bootstrap CI protocol | Phase 15 / stages 27–28 | D8 |
| 3 | **Feature engineering & selection** using the 17-keep list (logreg) vs full set (trees); re-derive cleanly in a notebook | Phase 10–12 / stages 21–25 | keep-17 |
| 4 | **Baseline fix + run**: persistence (this report's D7 numbers) as reference; fix `baseline_model.py` early-stopping-on-test leak first | Phase 16 / stage 30 | D7 |
| 5 | Model families (logreg, LightGBM, XGBoost) with **per-year macro+micro PR-AUC, recall at fixed precision, calibration**, monsoon subset, FN analysis | Phases 16–20 | D1/D2/D8 |

**Next invitation:** approve marking stages 13–18 DONE (already set in `phase0 §8`) and
approve Phase 11 (stage 19, leakage audit) — the natural next module, since split/model
results are only trustworthy after the boundary is formally re-pinned and `n_readings` ruled on.