# Phase 6–9 — Temporal, Spatial, Relationship & Redundancy Analysis
### Project study notes · Bangladesh flood-probability forecasting

> This continues the study series. The formal source is `reports/phase5_eda.md` — one report
> that covers workflow **stages 13–18** (univariate → temporal → spatial → target-vs-feature →
> correlation → multicollinearity), i.e. body **Phases 5–9**. The basic EDA half (label drift,
> monsoon structure, coverage) was already retold in the previous file; this file is the
> *deeper* half. Two later documents are cited where they finish a job these phases started:
> `reports/phase11_spatial_features.md` §7 and `reports/phase12_feature_selection.md` §1.
> Same rules as always: plain English, real numbers, real file names, what/why/how,
> problems + solutions.

---

## 0. The 60-second version

Phases 6–9 are the "look before you model" phases. Nobody trains a model here. The job is to
*understand the data well enough to make good decisions later*:

- **Phase 6 (univariate + temporal):** how each variable behaves, and how flooding behaves
  over years and seasons. The headline: **year-to-year drift is real and mostly happens
  *within* the same stations** — so pooling years into one number is misleading (this became
  hard rule **D8**). Missing days are mostly **winter/dry** days, not flood days — so gaps are
  "informative but safe", yet a missing reading must **never** be read as "no flood".
- **Phase 7 (spatial):** flooding is concentrated in a few basins (Mongla floods **83.8 %** of
  labelled days). So **which station** it is matters a lot — but the *current water level* still
  matters more.
- **Phase 8 (relationships):** today's water level **relative to the danger level**
  (`wl_dist_to_danger`, univariate AUC **0.968**) is by far the strongest single signal.
  Rainfall is a clear second layer. This is exactly why the **persistence baseline** becomes
  the bar every model must beat (**D7**).
- **Phase 9 (redundancy):** many columns are near-copies of each other — **137 feature pairs**
  correlate above ρ 0.95, and **22 features** have VIF > 10. Trees cope; a linear model does
  not, so a **17-feature keep list** was pruned for the logistic baseline.

Key takeaway: the data is clean and learnable; the difficulty is not "find a signal" but
"beat a very strong simple baseline, in years where floods are rare".

---

## 1. Where Phase 6–9 sit

| Phase | What | Plain words |
|---|---|---|
| 4–5 | Target investigation + basic EDA | Proved labels are right; saw drift/seasonality |
| **➜ 6** | **Univariate + temporal analysis** | **How each variable behaves; how flooding moves over years and months** |
| **➜ 7** | **Spatial analysis** | **Where flooding happens (which stations/basins)** |
| **➜ 8** | **Target-vs-feature + correlation** | **Which signals actually relate to the label** |
| **➜ 9** | **Multicollinearity & redundancy** | **Which columns are duplicates of each other** |
| 10–11 | Feature engineering (+ spatial features) | Build the predictors |
| 12+ | Selection → leakage audit → splits → models | Train, audit, pick |

Phases 6–9 correspond to **stages 13–18** of the master workflow (`final workflow.md`). They ran
on the model-ready table as it stood then: **200,665 × 54**, 133 stations, 2017-01-06 → 2026-10-05
(post-F3 + F8 label fixes). A post-script in the report matters: Phase 11 later appended **19
spatial columns → 73 total**, with labels byte-identical, so every number below still holds; only
the *candidate feature pool* changed. That handover is retold in §7.

---

## 2. Words you'll meet

| Word | Meaning |
|---|---|
| **Univariate** | Looking at one variable at a time (its range, skew, extremes). |
| **p99** | The 99th percentile — the "extreme" line: 1 % of days are above it. |
| **Drift** | The flood rate changing over time. Real, not a bug. |
| **Base rate** | Share of labelled days that are floods. |
| **Panel (fixed panel)** | A constant group of stations watched in *every* year — used to separate "data changed" from "the world changed". |
| **Informative missingness** | Gaps in the data are not random — they carry a pattern. |
| **VIF** | Variance Inflation Factor — a score for how much a feature is "explained" by the others. > 10 usually means redundancy. |
| **Spearman ρ** | A correlation that cares about *rank order*, not absolute values (robust). |
| **Collinear / near-duplicate** | Two columns that carry almost the same information. |
| **MI (mutual information)** | A non-linear "how much does this feature tell us about the label" score. |
| **Permutation importance** | Shuffle one column and see how much the model's score drops — a *model-based* importance. |
| **AUC / PR-AUC** | Ranking quality scores. PR-AUC is the one that stays honest when floods are rare. |
| **Persistence baseline (D7)** | The "it will flood because it already looks like it will flood" reference. The bar to beat. |

---

## 3. Phase 6 — Univariate and temporal analysis (stages 13, 14)

**Goal:** know exactly what each variable looks like, and how the *answer* behaves over time,
before any modelling choice is made.

### 3.1 Every key variable behaves plausibly

| variable | median | mean | p99 | max | note |
|---|---|---|---|---|---|
| `wl_max` | 7.09 m | 10.69 m | 64.56 m | 70.81 m | highest official danger level is 70.3 m |
| `wl_pct_danger` | 0.74 | 0.70 | 1.13 | 3.35 | it is a **fraction** (a finding, **F2**), not a percentage |
| `wl_dist_to_danger` | +2.85 m | +3.26 m | −0.82 m (p1) | −27.24 m | negative = already above danger |
| `rain_7d` | 70.6 mm | 90.6 mm | 424 mm | 2218 mm | the p99 line is the working definition of "extreme rainfall" |
| `rain_30d` | 342 mm | 380 mm | 1334 mm | 3720 mm | monsoon accumulation |
| `wl_rise_1d` | −0.01 m | +0.006 m | max **7.79 m/day** | — | **F8** verified: zero jumps above 10 m/day (spike filter D9) |

Also confirmed: `n_readings` ranges 3–8 per day, `wet_days_7` median is 7/7, `tmin_c` median
24.6 °C. **244 rows** still carry NaN weather (the residual window from defect **F1**, the
NASA POWER `-999` sentinel) — flagged so the linear baseline can impute them.

*Why it matters:* extremes must be physically possible values (they are), and a "7.79 m/day"
rise is a real flood wave, not a bad sensor. F2 (fraction vs percent) prevents a silent 100×
scaling error later.

### 3.2 Flood frequency by year (the drift, in one table)

Rate = share of labelled rows that are floods.

| year | 24 h | 48 h | 72 h | n (24 h) |
|---|---|---|---|---|
| 2017 | 13.37 % | 15.20 % | 16.81 % | 17,447 |
| 2019 | 5.01 % | 5.77 % | 6.47 % | 18,306 |
| 2020 | **13.57 %** | 14.77 % | 15.83 % | 20,049 |
| 2021 | 6.78 % | 7.59 % | 8.32 % | 21,743 |
| 2022 | 5.08 % | 5.96 % | 6.64 % | 21,878 |
| 2023 | **0.68 %** | 0.93 % | 1.19 % | 23,858 |
| 2024 | 2.54 % | 3.11 % | 3.69 % | 24,341 |
| 2025 | 1.17 % | 1.49 % | 1.79 % | 28,250 |
| 2026 | 1.27 % | 1.60 % | 1.90 % | 24,689 |

2018 is absent — the complete FFWC archive gap (**F2/R6**), never interpolated.

*Why it matters:* the same model can look brilliant in 2020 and weak in 2023 **without being
worse** — the base rate fell by ~20×. This single table is why PR-AUC is the primary metric and
why per-year reporting is mandatory.

### 3.3 Seasonality and episodes

- **Monsoon (JJAS, decision D1):** 72 h rate **9.79 %** vs non-monsoon **1.80 %** — **5.4×**;
  at 24 h it is 7.93 % vs 1.29 %. July peaks (**12.8 %** at 72 h). January–February have
  **zero** floods in the labelled record.
- **Episodes** (consecutive flood days, one station): median **3 d** (24 h) / **5 d** (48 h) /
  **6 d** (72 h); the longest single run is **144–146 days** — the prolonged 2020 tidal event
  (Mongla class), not an error.
- **Extreme rainfall:** `rain_7d` above the p99 (424 mm) occurs on **2,007 station-days across
  327 calendar days** — always in monsoon, almost always in major flood years. A small,
  precious sample: later robustness work (Phase 31) shows this bucket is the *hardest* to rank.

### 3.4 D5 — is the drift a data artefact or the world changing?

This is the sharpest question of the phase, and it was answered with a **fixed panel**: the
**31 stations** that have ≥ 200 observed days *and* labels in every year (2018 excluded).

| measure (72 h base rate) | fall, 2017 → 2026 |
|---|---|
| all stations, pooled | **14.9 pp** |
| fixed panel only | **19.1 pp** |
| contribution of *changing which stations exist* | **−4.2 pp** |

Verdict: the collapse is **mostly within-station regime change** (mega-flood years 2017/2020
vs quiescent 2023–26), *not* the growing station network (R7: 86 → 141 reporting stations).
Newer stations actually skew toward low-risk basins, so composition *hides* the drift slightly.

**Decision D5:** drift must be **explained, never pooled**. Together with **D8** ("never pool
years/seasons/stations/buckets") this becomes a hard reporting rule for every later phase.

### 3.5 D6 — is the observation process informative?

Observation is **heavily seasonal**: monsoon **99.0 %** of station-days observed vs
non-monsoon **44.7 %** (+54.3 pp). The unobserved dry days really are dry: median `rain_30d`
is **340 mm** on observed days vs **12 mm** on unobserved days (KS = 0.73).

Verdict: **informative — and in the *safe* direction.** Missing days are winter, not flood days,
so the labelled sample *over-represents* the flood season (the model trains on the regime it
must predict). Two consequences written into the rules:

1. dry-season labels are scarce → accuracy is mostly wet-season accuracy;
2. **unobserved ≠ 0** (the label contract), and `n_readings`/coverage are **observation-process
   signals** that must be ruled on explicitly in the leakage audit (finding **E7**).

---

## 4. Phase 7 — Spatial analysis (stage 15)

All 133 labelled stations fall inside the country bounding box (SP1), and all three horizons
cover the same 133 stations (SP2). Flood rate is dominated by a handful of basins:

| station | division | 72 h flood rate |
|---|---|---|
| Mongla | Khulna | **83.8 %** |
| Khulna | Khulna | 52.3 % |
| Daulatkhan | Barishal | 40.4 % |
| Companyganj | Chittagong | 35.8 % |
| Dalia | Rangpur | 29.2 % |
| Sureshswar | Dhaka | 24.8 % |
| Kalaroa | Khulna | 20.4 % |
| Barishal | Barishal | 20.0 % |

Two different physical stories share the same table: Khulna/Barishal (tidal) stations flood on
**tide + surge schedules** — effectively standing water — while Rangpur (Teesta) floods on
July–September **rainfall**. That is why per-station and per-basin reporting is mandatory and
why the later robustness phase (Phase 31) splits by station fold.

Location *is* a first-order feature, but a lesser one than water state: mutual information for
`lat`, `lon`, `danger_level` is **≤ 0.046**, against **0.255** for `wl_dist_to_danger`.

*Output unlocked:* the country choropleth became possible once admin boundaries
(`geoBoundaries-BGD-ADM1/ADM2`) were acquired in Phase 11 (see `reports/phase1_data_inventory.md` D7).

---

## 5. Phase 8 — Target-vs-feature and correlation (stages 16, 17)

**Goal:** which variables actually carry information about the label — measured three
independent ways, so no single lens can mislead.

### 5.1 Univariate "oriented AUC" at 72 h

| feature | AUC | KS | median non-flood | median flood |
|---|---|---|---|---|
| `wl_dist_to_danger` | **0.968** | 0.843 | +3.04 m | −0.26 m |
| `wl_pct_danger` | 0.959 | 0.807 | 0.72 | 1.03 |
| `wl_exceed_days_30` | 0.908 | 0.773 | 0 d | 11 d |
| `rain_30d` | 0.702 | 0.289 | 337 mm | 471 mm |
| `rain_14d` | 0.695 | 0.259 | 151 mm | 229 mm |
| `rain_7d` | 0.682 | 0.242 | 69 mm | 112 mm |
| `wet_days_7` | 0.663 | 0.294 | 7 | 7 |
| `tmin_c` | 0.642 | — | 25.5 °C | 26.1 °C |

The three lenses agree on the ordering: **today's water level relative to danger** dominates;
**accumulation windows** (`wl_exceed_days_30`, `rain_30d`) add the second layer; then raw
rainfall; temperature contributes a fraction of that. Mutual information mirrors the ranking
(`wl_dist_to_danger` 0.159, `wl_pct_danger` 0.157, `wl_exceed_days_30` 0.105), and Pearson on
the label gives `wl_exceed_days_30` 0.61, `wl_pct_danger` 0.37, rains ≤ 0.18.

One check *didn't* fire: `n_readings` alone scored below the 0.60 flag — so the
observation-process worry (**E7**) did not trigger at 72 h, but it stayed on the watchlist for
the leakage audit (Phase 13) rather than being dropped.

### 5.2 D7 — the persistence baseline becomes the bar

Persistence = *"will it flood because it already looks like it will flood"*, computed from
`wl_pct_danger` **alone**:

| horizon | pooled ROC-AUC | pooled PR-AUC | climatology | skill (PR-AUC ÷ base rate) |
|---|---|---|---|---|
| 24 h | **0.987** | **0.904** | 0.050 | **18.2×** |
| 48 h | 0.974 | 0.869 | 0.057 | 15.3× |
| 72 h | 0.959 | 0.836 | 0.063 | 13.2× |

Per year, ROC-AUC is remarkably stable (0.912–0.990) — but **PR-AUC is not**: it falls from
0.86–0.93 (2017/2020) to **0.51–0.63 (2023)** purely because the base rate collapsed while the
false-positive *count* barely moved.

**Decision D7:** every model from Phase 16 onward must be reported against these per-year
numbers. This is the bar, not a result — and it is a very high bar, which is the honest reason
the later "ML vs DL" story ends in "ML leads, by a real but modest margin" rather than a
blowout.

### 5.3 Correlation analysis (stage 17)

38 × 38 correlation matrices were built and validated (CO1/CO2 PASS: symmetric, unit diagonal,
finite, every feature correlated against every label). Largest positive label correlations:
`wl_exceed_days_30` 0.61, `wl_pct_danger` 0.37, `wl_dist_to_danger` 0.36, rainfall 0.15–0.18.

*How to read it:* correlation is used **only to understand linear structure**. High correlation
does **not** mean importance and low correlation does **not** mean uselessness —
`doy_sin`/`doy_cos` matter through non-linear interactions. That warning is why Phase 12 uses
four different selection lenses instead of one.

---

## 6. Phase 9 — Multicollinearity and redundancy (stage 18)

**Goal:** find columns that are near-copies, because they break linear models and inflate
coefficient variance (even though tree models are indifferent).

Findings:

- **160 predictor pairs** with |Spearman ρ| > 0.8, of which **137** exceed 0.95.
  Clean examples: `precip_mm` ≡ `rain_1d` (ρ = **1.000**), `wl_mean` ≡ `wl_first` (ρ = 0.999).
- **VIF > 10 for 22 features** — all the `wl_*` levels/lags/rolls, `rain_1d`/`rain_7d`, and
  `danger_level`.
- **Constants confirmed and dropped** (finding G3): `day_observed`, `source`. The
  label-window columns `obs_days_*` were never admitted at all.
- **Iterative VIF pruning** (remove the worst offender while VIF > 10) leaves a
  **17-feature keep list**:

  > `lat, lon, n_readings, wl_rise_1d, wl_rise_3d, wl_pct_danger, wl_exceed_days_30,
  > precip_mm, rain_3d, rain_14d, rain_30d, rain_max_7d, wet_days_7, tmax_c, tmin_c,
  > doy_sin, doy_cos` — max residual VIF **5.5**.

Interpretation: the water-level lags/levels are near-duplicates of `wl_pct_danger` — which is
**fine for trees**, but the **logistic-regression baseline must use the pruned set** or its
coefficients become unstable. The kept set still holds the full *information*: today + rises +
antecedent 30 days + rain + temperature + season + location.

*Important caveat (the post-script):* this 17-item list was derived **before** Phase 11 added 19
spatial columns. It is therefore a *provisional* result that Phase 12 had to re-derive — see §7.

---

## 7. Handover — what happened next (phases 10–12, for continuity)

These phases deliberately end with an action list; the study series picks it up in the next file.
The short version of what the deeper analysis triggered:

| Action handed over | What was actually done | Where |
|---|---|---|
| Re-derive the keep list now that geography exists | Phase 11 built **19 spatial columns** (7 static terrain/distance + 8 neighbourhood + 4 upstream); table 54 → **73 columns**, labels byte-identical | `reports/phase11_spatial_features.md` |
| Formal feature selection with group A/B/C included | Phase 12: 73 − 10 banned − 1 constant = **62 usable features**; MI ranking (step 19) put `wl_dist_to_danger` (0.2553) then `wl_pct_danger` (0.2537) on top; redundancy collapsed **23 of 62**; VIF removed 4 → **LINEAR = 35**; backward-elimination rule gave **SELECTED = 10**; sets written to `feature_sets.json` (**FULL 62 / LINEAR 35 / SELECTED 10**) | `reports/phase12_feature_selection.md` |
| Persistence as the reported bar (D7) | Kept in every later table, per year and per horizon | Phases 16–33 |
| Rule on `n_readings` / `nb*_n_wl_danger` (E7) | Formal leakage audit, `obs_days_*` ban enforced by `assert` | `reports/phase13_leakage_audit.md` |
| Never pool (D5/D8) | Temporal + spatial + seasonal cells in every phase; this is what Phase 27's "one table, one code path" reports | Phases 15–31 |

One number worth carrying forward from Phase 12, because it explains the whole modelling story:
permutation importance gives `wl_dist_to_danger` a mean PR-AUC drop of **0.526**, while the
second feature (`wl_pct_danger`) is **0.027** — the tree gain share of that single feature is
**77 %**. And the two selection lenses barely agree (Spearman perm↔MI = **0.087**), which is
exactly why multiple criteria were required.

---

## 8. Problems & solutions

| Problem | Solution |
|---|---|
| Base rate drifts 20× between years → pooled metrics lie | Fixed-panel decomposition (D5) proved the drift is real, then made per-year/per-cell reporting a hard rule (D8). |
| Growing station network (85 → 133) could fake a trend | Compared pooled vs fixed-panel rates; composition contributed only −4.2 pp, so the trend is genuine. |
| Missing days could hide floods | D6 test: gaps are winter/dry (rain_30d 12 mm vs 340 mm) → informative but safe; and the label contract still keeps missing as NaN, never 0. |
| A single "magic" feature could be mistaken for a real model | D7 persistence baseline computed and made the mandatory comparison; the strong feature *is* the baseline. |
| One importance lens could mislead | Three lenses (AUC, MI, Pearson) + later four (filter/MI/model/RFE); disagreements are documented, not hidden. |
| 137 near-duplicate pairs, 22 features with VIF > 10 | Iterative VIF pruning → 17-feature keep list; trees keep everything, linear models use the pruned set. |
| m²/percent confusion (`wl_pct_danger` stored as a fraction) | Recorded as F2 and fixed in the shared vocabulary before any scaling. |
| Impossible water-level jumps | D9 spike filter, F8 verified (0 jumps > 10 m/day). |
| Keep-list derived before the spatial features existed | Documented as provisional in the report's post-script; re-derived in Phase 12. |

---

## 9. Files referenced

| File | Purpose |
|---|---|
| `reports/phase5_eda.md` | Formal report for stages 13–18 (Phases 5–9); the figures and findings E1–E7 |
| `notebooks/03_eda.ipynb` | Executable notebook behind the report (22 cells) |
| `figures/phase5_univariate.png`, `phase5_temporal.png`, `phase5_spatial.png`, `phase5_target_vs_feature.png`, `phase5_corr.png`, `phase5_vif.png` | The visual evidence for §3–§6 |
| `reports/phase11_spatial_features.md` | The 19 spatial columns (post-script: pool grew to 73) |
| `reports/phase12_feature_selection.md` §1 | Admissible-pool arithmetic (73 → 62) and the MI step |
| `data/processed/flood_dataset_model_ready.csv` | The 200,665 × 73 table the numbers come from |
| `reports/phase0_research_definition.md` | Decisions D1–D10, risks R1–R8 |

---

## 10. Verdict

- **Phase 6:** every variable is physically plausible; flood frequency drifts hard across years,
  and the drift is **within-station regime change**, not a data artefact (D5 → D8).
- **Phase 7:** flooding is concentrated in a few basins/tidal stations, so **station identity is
  a first-order feature** — but smaller than water-level state.
- **Phase 8:** today's water level relative to danger is the dominant signal; the
  **persistence baseline (D7)** is the bar; the hard part is the quiet years (PR-AUC 0.51–0.63).
- **Phase 9:** the feature pool is heavily redundant (137 pairs > 0.95, 22 with VIF > 10); a
  17-feature pruned set serves linear models while trees keep all.

**Bottom line:** the data is learnable, clean and honest — and the *real* challenge was set here:
**beat a very strong simple baseline, in years when floods almost never happen.** Every later
phase is measured against that.

**Next in this series:** Phase 10–12 — feature engineering and feature selection (the MI step,
the 62/35/10 feature sets).
