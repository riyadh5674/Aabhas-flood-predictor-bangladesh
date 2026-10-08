# Phase 11 — Spatial feature engineering (workflow stages 21 & 23)

| | |
| --- | --- |
| **Date** | 2026-10-08 |
| **Modules** | `scripts/build_spatial_features.py` (static) · `src/flood/features/build_dataset.py` → `add_static_spatial()`, `add_neighbourhood()`, `add_upstream()` |
| **Inputs** | `data/raw/geo/srtm` (40 tiles), `data/raw/geo/osm/rivers_bangladesh.geojson`, `data/raw/ffwc/ffwc_stations.csv`, `data/raw/rainfall/weather_station_daily.csv` |
| **Outputs** | `data/processed/station_spatial_features.csv` (153 stations) · `flood_dataset.csv` and `flood_dataset_model_ready.csv` rebuilt **200,665 × 73** |
| **Verdict** | `audit_dataset.py --skip-live` **14/14 PASS incl. new A8b/A8c/A11** · `verify_model_ready.py` **PASS incl. two new Phase-11 hand-checks** |
| **Governing doc** | `flood prediction workflow.md` stages 21/23; `reports/phase0_research_definition.md` §5 information boundary |

No file under `data/raw/` was modified. Raw immutability contract holds (the FFWC/NASA raw
files are read-only inputs; everything is derived into `data/processed/`).

---

## 1. Group A — static geography (7 columns)

Built once by `scripts/build_spatial_features.py`, then joined per `station_id`
(`add_static_spatial`). **0 NaN cells in the model-ready table, 133/133 stations covered.**

| Column | mean | min | max | Source |
| --- | ---: | ---: | ---: | --- |
| `elevation_m` | 15.45 | −10.0 | 70.0 | SRTM 1″ (30 m) nearest sample |
| `slope_deg` | 3.07 | 0.00 | 13.20 | SRTM 3×3 finite difference |
| `dist_to_coast_km` | 8.26 | 0.05 | 100.87 | SRTM land/ocean boundary, equirectangular (~1 km) |
| `dist_river_km` | 0.41 | 0.001 | 7.82 | nearest OSM `waterway=river` line (UTM 46N) |
| `dist_major_river_km` | 1.48 | 0.003 | 24.32 | nearest line ≥ 20 km long (structural trunk proxy) |
| `river_density_10km_km_per_km2` | 0.245 | 0.015 | 0.633 | river length in a 10 km disc ÷ disc area |
| `n_river_lines_10km` | 17.96 | 1 | 90 | count of river lines in the disc |

Caveats (unchanged from `phase1_data_inventory.md` D7): `dist_to_coast` is a ~1 km
approximation; "major" is a *length* proxy, not a name join (OSM names are Bengali);
`elevation_m = −10` is a coastal/char depression, physically possible in the delta.

## 2. Group B — neighbourhood / regional context (8 columns)

`add_neighbourhood()` — **nearest-k gauges' state on day `t`.**

| Design choice | Value | Why |
| --- | --- | --- |
| Candidate pool | nearest **15** gauges by haversine, self excluded | keeps the search local and O(S²) once |
| `k` | **3 and 5** | two spatial scales, cheap |
| Membership | **resolved per day** — of the pool, the nearest *k* that actually reported that day | the station network grows over time (85 → 133, risk R7); a fixed neighbour list would silently use future installations |
| Rain signals | `nb{3,5}_rain_7d`, `nb{3,5}_rain_14d` | built from the **weather record**, not the WL-active matrix: rainfall is knowable at any gauge location any day, so a weather-only day still carries the areal signal (F1 sentinel rule applied) |
| WL signals | `nb{3,5}_wl_pct_danger` | mean wl/danger ratio of the nearest-k gauges that reported a stage on `t` |
| Count signal | `nb{3,5}_n_wl_danger` | how many of the nearest-k are **already at/above** their danger level on `t` |

Coverage in the model-ready table: rain neighbours **0 % NaN**; `nb*_wl_pct_danger`
**0.35 % NaN** (no gauge of the pool reported that day); counts 0 % NaN (empty ⇒ 0 is the
correct value, not missing).

## 3. Group C — upstream propagation (4 columns)

`add_upstream()` — **the nearest same-river gauge at clearly HIGHER elevation.**

* *Why not latitude?* The lower Ganges/Padma system flows **west**, so a latitude rule
  would pick the wrong "upstream". Rule: same `river` string, `elevation_m` higher by
  ≥ 0.5 m, nearest by haversine, within **250 km**.
* Result: **56 downstream stations** mapped (50 of them appear in the model-ready table);
  85 stations have no eligible upstream partner ⇒ `up_*` is NaN **by design** (67–70 % of
  rows), never imputed at build time.
* Features: `up_rain7d_lag{1,2}`, `up_wl_pct_danger_lag{1,2}` — the upstream gauge's value
  on **day t−1 and t−2**, i.e. strictly *before* the label window opens.

## 4. Leakage position (pre-audit argument)

| Feature group | Information used | Available at issue instant 24:00 UTC of `t`? |
| --- | --- | --- |
| static (A) | station coordinates + terrain | yes — time-invariant |
| neighbourhood rain (B) | neighbour rainfall over days ≤ `t` | yes |
| neighbourhood WL (B) | neighbour stage **on day `t`** | yes — same issue instant as the station's own `wl_*` |
| upstream (C) | upstream state on `t−1`, `t−2` | yes — strictly past |

Nothing in A/B/C touches `obs_days_*`, the label window, or any day `> t`. This is the
*argument*; the formal ruling still belongs to **stage 19** (leakage audit), which must add
all 19 columns to the tracked whitelist and rule on them explicitly.

## 5. Defect found and fixed during Phase 11

| | |
| --- | --- |
| Symptom | `audit_dataset.py` crashed with `KeyError: 'nb3_wl_pct_danger'` (check A11); model-ready had **71** columns instead of 73 |
| Root cause | `NEIGHB_SIGS` declared `wl_pct_danger` and the data dictionary documented it, but `add_neighbourhood()` only computed the rain means and the *count* — the wl-ratio `resolve()` call was never written |
| Why it stayed hidden | the `keep` filter was `df[[c for c in keep if c in df.columns]]` — a **declared-but-missing column was silently dropped**, so the build "succeeded" |
| Fix | (a) compute `nb{3,5}_wl_pct_danger` with the same `resolve()` path as the rain neighbours; (b) the build now **raises** if any declared column was not produced; (c) dead line (`... if False else ...`) and an unused dict removed |
| Proof of no collateral change | labels byte-identical to the pre-fix build: `label_24h` 200,559/9,982 · `label_48h` 197,748/11,241 · `label_72h` 195,174/12,340 (matches `phase0 §4`) |

## 6. Verification

| Check | Result |
| --- | --- |
| `audit_dataset.py --skip-live` | **14/14 PASS** — A1–A10 as before, plus **A8b** static join identity/range, **A8c** canonical division names, **A11** dynamic spatial plausibility (0 out-of-range cells; `nb3_wl_pct_danger` NaN 0.3 %) |
| `verify_model_ready.py` | hand recompute 60 rows / 180 labels **0 mismatches**; static join **0 mismatches**; `nb3_rain_7d` **0 mismatches** |
| **new** `nb3_wl_pct_danger` hand-check | 60 rows, nearest-3 pool rebuilt from master coordinates, values taken from the full matrix → **0 mismatches** |
| **new** `up_wl_pct_danger_lag1` hand-check | 32 sampled rows that have an upstream gauge (map size 56) → **0 mismatches** |

The verifier deliberately takes neighbour values from the *full matrix* and geometry from
the *station master*, i.e. a different code path from the builder.

## 7. Open items handed to the next stages

1. **Phase 13 / steps 22–24 (leakage audit, checklist 19–20)** — formally whitelist/rule on
   the 19 new columns; `nb*_n_wl_danger` and `n_readings` are observation-process signals and
   need an explicit admissibility ruling (phase5 findings E7/conclusion 5).
2. **Phase 12 / steps 18–21 (feature selection, checklist 24–25)** — runs **first**, per
   decision **D10** (body Phase order is binding): the 17-feature VIF keep-list predates this
   group and must be re-derived with A/B/C included; expect `nb*_wl_pct_danger` to be dropped
   as a near-duplicate of the station's own `wl_pct_danger` for the *linear* baseline while
   keeping it for trees.
3. **Missing data** — M4 (land cover / soil) still absent, so the static group is not
   final; risk R8 unchanged.
4. **Imputation** — 70 % NaN on `up_*` is structural: tree models handle it natively, the
   logistic baseline needs a fold-fitted imputer **plus a "no upstream gauge" indicator**.

**Next: Phase 12 — feature selection (body steps 18–21: statistical filter → MI → model-based
→ RFE), then Phase 13 — leakage audit (steps 22–24).** Order per D10.
