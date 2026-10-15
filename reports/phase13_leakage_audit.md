# Phase 13 — Critical leakage audit (workflow body steps 22–24)

| | |
| --- | --- |
| **Date** | 2026-10-08 |
| **Module** | `notebooks/05_leakage_audit.ipynb` (source of truth = `scripts/build_phase13_notebook.py`) |
| **Inputs** | `data/processed/flood_dataset_model_ready.csv` (200,665 × 73) · `flood_dataset.csv` (357,250 × 73) · `feature_sets.json` (Phase 12) · `src/flood/features/build_dataset.py` (the code under audit) |
| **Outputs** | **`data/processed/feature_whitelist.csv`** (73 rows — the stage-19 deliverable that had been deleted) · `data/processed/leakage_audit.json` · `figures/phase13_auc_screen.png` · `feature_sets.json` → new `phase13` block |
| **Verdict** | **13/13 code cells executed, 0 errors, 11/11 checks green (`L1`–`L11`), every `assert` held.** No feature demoted → Phase 12 sets stand, no re-run required. |
| **Governing doc** | `flood prediction workflow.md` **Phase 13 — Critical leakage audit** (22 feature → 23 target → 24 duplicate/time); checklist 19 (data-leakage audit) + 20 (prediction-time information boundary); `reports/phase0_research_definition.md` §5; decisions **D10** (body-Phase order), **D8** (no pooled metrics) |

Run command (project venv kernel):

```
python scripts\build_phase13_notebook.py
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=flood-predictor notebooks\05_leakage_audit.ipynb
```

Nothing under `data/raw/` was modified. No model was trained beyond the small diagnostic
fits required by step 23 (L8) — splitting and modelling remain Phases 15–17 (D10).
The `assert`s abort the run, so **a green notebook is the evidence**, not a summary of it.

---

## 1. Method

For every stored value the claim to be *proved, not asserted* is the §5 boundary: the value
is a function of days `≤ t` only, with issue instant **24:00 UTC of day `t`**. Three
independent attack lines, one per workflow step:

| step | attack | checks |
| --- | --- | --- |
| 22 | re-derive the features from the raw calendar matrix with an **independent implementation**, and audit the builder source for forward references | L1–L4 |
| 23 | screen every feature against every label (univariate AUC), prove the screen can catch leakage (positive control), and test observation-process proxies | L5–L8 |
| 24 | keys, splits, holes and cross-part similarity in the row space | L9–L11 |

## 2. Step 22 — feature leakage audit

| check | evidence | result |
| --- | --- | --- |
| **L1** static source audit | every `.shift(-` in the module sits inside `add_labels` (2 calls, lines 145–181); the 4 feature builders (`add_features`, `add_static_spatial`, `add_neighbourhood`, `add_upstream`) contain **0** occurrences of `label_*`, `obs_days`, `LEADS`; no `year`/`month`/`bad_*` calendar columns in the pool; `day_observed` absent (dropped at Phase 12 step 18a) | **PASS** |
| **L2** calendar-formula recompute | **27 derived columns** (lags 1–7, `wl_max_lag1-3`, rises, `wl_dist_to_danger`, `wl_pct_danger`, `wl_max_roll7/30`, `wl_exceed_days_30`, rain windows, `doy_sin/cos`) rebuilt from the *full* matrix with `shift(+k)` / backward rolling formulas → **0 mismatches** over 200,665 rows (tol 1e-9) | **PASS** |
| **L3** neighbourhood recompute | nearest-15 haversine pool rebuilt from coordinates, nearest-k membership resolved per day, means/counts recomputed from **day-`t` values only** for 60 sampled dates → **47,984 cells, 0 mismatches** | **PASS** |
| **L4** upstream recompute | upstream rule re-derived independently (same river, elevation > +0.5 m, nearest ≤ 250 km) → 56 stations mapped; 4 columns × 200,665 rows = **802,660 cells, 0 mismatches** | **PASS** |

A formula audit is only meaningful because the builder's `shift(+k)` is a *calendar* shift
on a matrix with one row per calendar day — L10 re-verifies that directly in the data.

**Two grid subtleties the audit had to model exactly** (a first attempt scored 12,603
"mismatches" that were artefacts of the audit, not of the data):

1. `nb{3,5}_rain_7d/14d` are built from **`weather_station_daily.csv`'s own row grid**
   (rolling over that file's rows, `-999` sentinel → NaN), *not* from the calendar matrix —
   calendar-grid rolling differs wherever a station skips a weather day.
2. `nb{3,5}_n_wl_danger` is a `nansum` **count**, so an empty pool stores `0`, never NaN.

Both are documented in `add_neighbourhood()` and reproduced faithfully; with the right
grids the comparison is exact.

## 3. Step 23 — target leakage audit

| check | question answered | evidence | result |
| --- | --- | --- | --- |
| **L5** | *directly contains the target* | FULL(62) / LINEAR(35) / SELECTED(10) share **0** columns with the 3 labels, 3 `obs_days_*` and the 4 identity/date keys — and the banned columns exist in the table, so the ban is a choice | **PASS** |
| **L6** | *indirectly reveals the target* | univariate screen, 62 features × 3 horizons: **max oriented AUC 0.9901 (`wl_dist_to_danger`)**, 0 features at/above the 0.995 flag threshold | **PASS** |
| **L7** | *calculated using the target / after the event* | **positive control**: a deliberately future-derived feature (exceed on `t+1`, `t+1..t+3`) reproduces `label_24h` / `label_72h` on **100 %** of labelled rows, AUC **1.000** → the L6 threshold *would* catch it; its legitimate past-window twin `wl_exceed_days_30` scores 0.908 | **PASS** |
| **L8** | *observation-process signal* (phase5 **E7** + phase11 §7 item 1) | see table below | **PASS** |

Top of the L6 screen (`figures/phase13_auc_screen.png`):

| feature | 24 h | 48 h | 72 h | max |
| --- | ---: | ---: | ---: | ---: |
| `wl_dist_to_danger` | 0.9901 | 0.9791 | 0.9680 | **0.9901** |
| `wl_pct_danger` | 0.9874 | 0.9736 | 0.9590 | 0.9874 |
| `wl_exceed_days_30` | 0.9458 | 0.9253 | 0.9082 | 0.9458 |
| `up_wl_pct_danger_lag1` | 0.8915 | 0.8753 | 0.8606 | 0.8915 |
| `nb3_wl_pct_danger` | 0.8890 | 0.8739 | 0.8602 | 0.8890 |
| `nb5_wl_pct_danger` | 0.8814 | 0.8655 | 0.8515 | 0.8814 |
| `up_wl_pct_danger_lag2` | 0.8760 | 0.8583 | 0.8437 | 0.8760 |
| `nb5_n_wl_danger` | 0.8573 | 0.8379 | 0.8203 | 0.8573 |
| `nb3_n_wl_danger` | 0.8409 | 0.8198 | 0.8010 | 0.8409 |
| `up_rain7d_lag1` | 0.7090 | 0.7103 | 0.7091 | 0.7103 |

The two leaders are *definitionally* close to the target (stage vs the published danger
level) — that is persistence, not leakage: L2 rebuilds both from `wl_max`/`danger_level`
on day `t` with 0 mismatches, and L7 shows the screen scores a genuinely future-derived
feature 1.000 while these sit at 0.99 / 0.987.

### L8 — is the signal real, or is it just "a label exists"?

| feature | AUC → target (24 h) | AUC → **label availability** | Δ PR-AUC vs persistence (2022) |
| --- | ---: | ---: | --- |
| `n_readings` | 0.475 | **0.804** | −0.0013 / −0.0001 / +0.0018 (24/48/72 h) |
| `nb5_n_wl_danger` | **0.857** | 0.295 | **+0.0035 / +0.0040 / +0.0075** |

`n_readings` predicts *whether a label will exist* (0.804) but not *whether a flood will
happen* (0.475, i.e. chance), and adds nothing to the persistence baseline → **ADMIT-WATCH**.
`nb5_n_wl_danger` is the opposite: strong target signal, near-random availability, and it
*adds* PR-AUC → **ADMIT**. Both are computable at 24:00 UTC of `t`; neither is derived
from the label.

## 4. Step 24 — duplicate / time leakage

| check | evidence | result |
| --- | --- | --- |
| **L9** | 0 duplicate station-days in either table; provisional split (same dates as Phase 12) train 77,571 / val 21,920 / test 101,174 shares **0 dates and 0 station-days** | **PASS** |
| **L10** | holes derived from the data, not assumed: **52 zero-observation stretches**, longest **2017-12-10 → 2018-12-31**; 93,208 calendar rows sit in holes and **none** reaches the model-ready table (2018: 31,390 rows, 0 observed, 0 in table); rows are **consecutive calendar days for all 141 stations**; lag1–7 empty on 2019-01-01 (23 observed rows); label rules hold for all 3 horizons | **PASS** |
| **L11** | **0** identical 62-feature vectors in two parts (0 rows, 0.000 %); each cut sits inside a **model-ready-free stretch** (below) | **PASS** |

L10's label rules, per horizon (rows in the model-ready table):

| horizon | label = 0 | label = 1 | NaN (other label missing) |
| --- | ---: | ---: | ---: |
| `label_24h` | 190,577 | 9,982 | 106 |
| `label_48h` | 186,507 | 11,241 | 2,917 |
| `label_72h` | 182,834 | 12,340 | 5,491 |

asserted: a `0` requires **every** day of the window observed (`obs_days_h == h`), a `1`
requires an observed exceed day inside the window — so no label can be built from an
unobserved (or future-of-`t+h`) day.

### The split already has a natural embargo

| cut | last model-ready row before | first after | model-ready-free days | stations on both sides within 90 d (positive both sides) |
| --- | --- | --- | ---: | --- |
| 2022-01-01 | 2021-11-03 | 2022-03-15 | **131** | 66 (1) |
| 2023-01-01 | 2022-10-30 | 2023-01-27 | **88** | 43 (0) |

The date cut falls inside the dry-season non-reporting stretch in both places, so no
episode, storm or flood can straddle a boundary within ±7 d — an *incidental* embargo.
Phase 15 should still make the embargo explicit (decision remains open), but the risk it
exists to mitigate is already largely absent.

## 5. Rulings — `data/processed/feature_whitelist.csv`

One row per column of the model-ready table: group, **the latest day each value may depend
on**, availability at the issue instant, ruling.

| ruling | n | columns |
| --- | ---: | --- |
| `ADMIT` | 61 | everything approved by §5, formula-verified in L2/L3/L4 |
| `ADMIT-WATCH` | 1 | `n_readings` — label-availability proxy (L8); ablate in Phase 38 |
| `ADMIT-CONSTANT` | 1 | `day_observed` — available at `t`, constant in the table, dropped at Phase 12 step 18a |
| `BAN` | 10 | `station_id name date source`, `label_24h/48h/72h`, `obs_days_24h/48h/72h` |

Assertions enforced in the notebook: **BAN == the 10 banned columns**, **ADMIT ∪
{constant} == the 63 approved §5 columns**, **FULL == ADMIT − {day_observed} (62)**.

Open items closed here:

1. **`obs_days_*` / labels** — banned, and the ban is exercised (the columns exist in the
   table but appear in none of the three Phase-12 sets).
2. **`n_readings` (phase5 E7)** — ADMIT-WATCH with the L8 numbers above.
3. **19 Phase-11 columns (phase11 §7 item 1)** — all 19 ADMIT: static geography
   time-invariant; `nb*` recomputed from day-`t` neighbour state; `up_*` recomputed from
   the upstream gauge's `t-1`/`t-2` values. `nb*_n_wl_danger` explicitly ruled on via L8.
4. **`wl_dist_to_danger` (phase12 finding 1)** — ADMIT: `danger_level − wl_max`, both
   known at 24:00 UTC of `t`, reproduced exactly in L2; its 0.990 AUC is persistence, and
   L7 proves the screen distinguishes it from a real future-derived feature.
5. **Feature-time-bound check** — the `latest_day_used` column of the whitelist is the
   machine-readable answer for all 73 columns (static / `t-k` / `t` / `t+1..t+h` / key).

## 6. Consequence for Phase 12

No column was demoted, so `feature_sets.json`'s FULL/LINEAR/SELECTED sets are unchanged
and the Phase-12 notebook does **not** need a re-run (phase12 §9 item 3). A `phase13`
block was appended to `feature_sets.json` recording the ruling so the file stays
authoritative.

## 7. Findings raised by this phase

1. **F13-1 — the archive hole is wider than "2018".** Zero-observation stretches number
   **52**; the main one runs 2017-12-10 → 2018-12-31, but Nov 2017, Feb–Mar 2019,
   Nov 2019–Jan 2020, Nov 2020–Feb 2021, Nov 2021–Mar 2022, Nov 2022–Jan 2023 … are also
   empty. None reaches the model-ready table (L10) — this is a *coverage* fact for Phase 15
   (temporal split design), not a leakage defect.
2. **F13-2 — both date cuts already sit inside natural embargoes** (131 and 88 model-ready-free
   days). Phase 15 should record this rather than re-derive it.
3. **F13-3 — `n_readings` is a selection variable.** Availability AUC 0.804 with target AUC
   0.475: it says when FFWC published data, not when it flooded. Kept (available at `t`) but
   flagged for ablation; if a model leans on it, the gain will evaporate on a period with
   different publication habits.
4. **F13-4 — nearest-feature AUC 0.990 is below, but not far below, the 0.995 flag.** The
   screen's power is demonstrated (L7 = 1.000). No action; keep the threshold documented so
   a future feature at 0.996 gets challenged rather than trusted.
5. **Audit-tooling lesson:** the first L3 run reported 12,603 mismatches that were entirely
   the *audit's* fault (wrong source grid, count-vs-NaN semantics). Independent
   re-implementation must match the builder's data lineage, not just its formula.

## 8. Open items handed forward

1. **Phase 14 — define the experimental table**: whitelist is now authoritative; build the
   modelling matrix from `ADMIT` columns only (62 = FULL).
2. **Phase 15 — split design**: record the natural embargo (F13-2), decide whether an
   explicit gap is still required, and respect the base-rate drift (9.6 → 5.1 → 1.4 %,
   phase12 §2) under D8.
3. **Phase 38 — ablation**: drop `n_readings` and confirm no loss (F13-3).

**Next: Phase 14 — define the experimental dataset, then Phase 15 — train/validation/test
strategy (body steps 25–27 / stages 27–29).** Order per D10.
