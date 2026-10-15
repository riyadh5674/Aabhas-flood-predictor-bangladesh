# Phase 2–3 — Data Quality & Exploratory Data Analysis (EDA)
### Project study notes · Bangladesh flood-probability forecasting

> **Purpose of this file:** Continue the study series (one file per phase-group). Phase 2–3 are
> described together here because they were tightly linked: *inspect structure (Phase 2)* →
> *find and fix defects (Phase 3)* → *understand patterns (EDA)*. The original material lives in
> `reports/phase2_3_data_quality.md` and the notebook `notebooks/02_data_quality_eda.ipynb`.
>
> As before: simple English, real numbers, real file names, explain *what/why/how*, cover
> problems+solutions, and be beginner-friendly.

---

## 0. The 60-second version

After taking stock in Phase 1, the next job was: **"Is what we have actually clean enough to trust?"**

- **Phase 2 (Load & structural inspection):** Check shapes, types, dates, keys, and whether the
  assembled table matches expectations.
- **Phase 3 (Data quality investigation + EDA):** Hunt for sneaky bugs (like `-999`), verify
  the label rules, and explore patterns (seasons, base rates, coverage).

The headline: **one silent bug (F1: −999 in NASA POWER) was caught and fixed the same day**, with
proof that nothing else changed. The data passed a strict set of quality checks (A1–A10) and is
ready for feature engineering.

---

## 1. Where Phase 2–3 sit

| Phase | What | Plain words |
|---|---|---|
| Phase 0 | Research definition | Fixed question, horizons, label rules, splits |
| Phase 1 | Data inventory | Took stock of every raw file |
| **➜ Phase 2–3** | **Quality + EDA** | **Checked structure, found/fixed defects, understood patterns** |
| Phase 4–11 | Target/EDA/features (spatial) | Built the modelling features |
| Phase 12+ | Selection, modelling, selection finalised (Phase 33) | Trained and chose the model |

**Why together?** Phase 2 is "does the table look structurally right?" Phase 3 is "what's
broken and what can we learn?". In practice they ran hand-in-hand (especially the F1 fix).

---

## 2. Key words (quick refresher)

| Word | Meaning |
|---|---|
| **Structural inspection** | Check column names, dtypes, uniqueness of keys, ranges — "does this table have the shape we expected?" |
| **QA checks (A1–A10)** | A numbered checklist of automatic sanity checks run by `scripts/audit_dataset.py`. |
| **Sanity check** | A cheap test that catches obvious mistakes (e.g. a date range looks impossible). |
| **Sentinel** | A fake code meaning "missing" (e.g. `-999`). Easy to miss if treated as a number. |
| **NaN / blank** | "Not a number" / missing value. Correct way to represent unknown. |
| **Panel data** | Repeated observations over time for the same entities (here: stations over days). |
| **Base rate** | How often "flood" happens (positives / total labelled). ~5% here — rare. |
| **Imbalance** | Many more "no flood" than "flood". Drives choice of **PR-AUC** as primary metric. |
| **Coverage** | How many station-days actually have readings/labels. |
| **EDA (exploratory)** | Looking at distributions, seasons, gaps to build intuition before modelling. |
| **Monsoon (JJAS)** | June–September — peak flood season in Bangladesh (D1 from Phase 0). |
| **Tidal stations** | Coastal stations where water level is strongly influenced by tides (e.g. Mongla, Khulna) — explains very high base rates there. |
| **Archive gap** | A real missing period in the source data (here **2018** has zero readings). |

---

## 3. Phase 2 — Load & structural inspection

**Goal:** Load the built tables and confirm the "skeleton" is correct.

**Main files inspected:**
- `data/processed/flood_dataset.csv` — full calendar (357,250 × 73)
- `data/processed/flood_dataset_model_ready.csv` — model-ready subset (200,665 × 73)
- `data/interim/ffwc_*_audit.csv` — coverage audits
- `data/raw/ffwc/harvest_manifest.csv` — provenance

### 3.1 What we checked

| Check idea | What we looked for | Result |
|---|---|---|
| **Shape** | Rows/cols match expectation | OK (357k full; 200,665 model-ready) |
| **Keys** | `(station_id, date)` should be unique | OK — no duplicates |
| **Dates** | Range plausible, no future dates | OK: 2017-01-06 → 2026-10-05 (model-ready) |
| **Stations** | Count matches master | OK: 133 stations with numeric danger level |
| **Label presence** | `label_24h/48h/72h` exist with correct types | OK |
| **Readings** | `n_readings` reasonable (0–8 typical) | OK |
| **Coverage logic** | Model-ready really only has days we can label | OK — matches Phase 0 contract |
| **Provenance** | Manifest matches what we fetched | OK |

**Takeaway:** Structurally the assembled table is sound. No obvious shape/type errors. The next step was to hunt for **value-level** defects.

---

## 4. Phase 3 — Data quality investigation (A1–A10 + findings F1–F7)

Phase 3 ran a strict audit via `scripts/audit_dataset.py` (checks **A1–A10**) and recorded findings **F1–F7**. The key one was **F1**.

### 4.1 The automated checks (A1–A10)

These are the "go/no-go" checks. They must all pass before moving on.

| ID | Check | What it means | Result |
|---|---|---|---|
| **A1** | Date continuity & range | Dates within expected span; no impossible gaps in a trivial sense | **PASS** |
| **A2** | Station coverage | Each station appears with expected date set | **PASS** |
| **A3** | No duplicate `(station_id,date)` | Key is unique | **PASS** |
| **A4** | Water-level sentinels handled | `≤ -1000` removed (F5) | **PASS** |
| **A5** | Label contract respected | Missing stays NaN (never becomes 0); danger level from master | **PASS** |
| **A6** | `obs_days_*` consistency | Window-observation counts make sense vs labels | **PASS** |
| **A7** | Danger-level consistency | Master danger level used (not per-reading) | **PASS** |
| **A8** | Plausible physical ranges | Water level/coords/rain/temp in sensible ranges | **PASS** |
| **A9** | Season assignment (JJAS) | Monsoon correctly defined | **PASS** |
| **A10** | No accidental leakage | Label-window columns not used as features later (enforced in Phase 13) | **PASS** |

**All A1–A10 passed after the F1 fix.**

### 4.2 Finding F1 — the `-999` sentinel (the big one)

As covered before, but stated clearly here:

- **Where:** `data/raw/rainfall/weather_station_daily.csv` (NASA POWER)
- **When it appears:** **2026-10-04 → 2026-10-06** (newest 3 days)
- **Count:** **459 rows** with `precip_mm = tmax_c = tmin_c = tmean_c = -999.0`
- **Why hidden:** `-999` is a **number**, not NaN. So `.isna()` checks said "0 missing".
- **Effect:** Rolling sums became huge negative (e.g. `rain_3d` down to **−1997.9 mm**) — corrupting exactly the "today" rows.
- **Fix:** Map `<= -999` → NaN on read in `build_dataset.py` (raw file untouched). Also fixed the ingest helper to not treat `-999` as truthy.
- **Proof:** Before/after snapshots identical on rows/stations/range/labels; **1,836 cells** became NaN. `audit_dataset.py` and `verify_model_ready.py` both **PASS** with 0 mismatches.

**Leftover:** 244 model-ready rows now have NaN rain/temp for those trailing days — correct (unknown), handled by imputation inside training folds later.

### 4.3 Findings F2–F7 (documented, accounted for)

| ID | Finding | Explanation |
|---|---|---|
| **F2** | **2018 is a complete archive gap** (0 readings) | Real. Not interpolated. Models must learn/work without 2018. |
| **F3** | **20 stations have no numeric danger level** | Cannot form labels → rows stay NaN forever (correct). |
| **F4** | **Tidal stations flood very often** (Mongla ~83.8%, Khulna ~52.3%) | Genuine tidal effect. Legitimate, not labelling error. |
| **F5** | **Water-level sentinel `≤ -1000` exists** | Already handled at build time (dropped). |
| **F6** | **Per-reading `dangerlevel` unreliable** | Overwritten by master value from station file. |
| **F7** | **Expected discharge stub empty** | Explained why BWDB discharge wasn't harvested at Phase 1 (later addressed via GloFAS in MI-1b). |

All recorded in `reports/phase2_3_data_quality.md`.

---

## 5. Exploratory Data Analysis (EDA) — what we learned

With clean data, we looked for patterns to build intuition. The goal here is *understanding*, not tuning.

### 5.1 Label counts & base rates

From Phase 0 and audits (model-ready):

| Label | n labelled | positives | base rate |
|---|---|---|---|
| `label_24h` | 200,559 | 9,982 | **4.98 %** |
| `label_48h` | 197,748 | 11,241 | **5.68 %** |
| `label_72h` | 195,174 | 12,340 | **6.32 %** |

So flooding is **rare** (about 1 in 20 days). As horizon grows, "any flood in window" becomes more likely (4.98% → 6.32%). The class imbalance is real: **~14–19:1** (no-flood : flood).

**Implication:** Accuracy is misleading. **PR-AUC** is the right primary metric (we kept this rule throughout).

### 5.2 Time coverage & the 2018 gap

- **Range:** 2017-01-06 → 2026-10-05
- **2018:** **completely missing** (F2). No readings returned by FFWC for any station in 2018.
- **Panel:** Mostly balanced across stations for the years that exist.

We never "filled" 2018. It's a real data gap and is respected by splits (train/val/test don't invent data).

### 5.3 Geography: basins & stations

- **133 stations** with usable danger levels
- **Basins:** Brahmaputra (57), Meghna (36), Ganges (25), plus smaller groups
- **Coastal/tidal effect (F4):** Mongla and Khulna show unusually high flood frequency — tidal influence, not errors.
- **Spatial spread:** Covers most of Bangladesh's major river network

### 5.4 Rainfall patterns (NASA POWER, with R5 caveat)

- **Mostly dry:** ~33% of station-days have exactly 0 mm
- **Highly skewed:** Occasional very heavy days (max seen ~662 mm/day)
- **Seasonal:** Strong monsoon peak (JJAS)
- **Sentinel fixed:** Trailing −999 removed (F1)

Remember **R5:** NASA POWER is **reanalysis** (modelled), not observed gauge rain. We keep this caveat in reports.

### 5.5 Water level behaviour

- **Danger levels vary** by station (1.8–70.3 m) — reflects different river sizes/locations.
- **`wl_pct_danger = wl_last / danger_level`** is a very intuitive clue: values near/above 1 mean "close to or above danger".
- **Rising vs falling** matters: `wl_rise_1d`, `wl_rise_3d` capture short-term change.
- **Memory helps:** lags (`wl_last_lag1..7`) and rolling max (`wl_max_roll7/30`) capture persistence.

### 5.6 Seasonality (monsoon = JJAS)

Flooding is overwhelmingly **monsoon-driven**. JJAS contains the bulk of flood events. This is why later phases report **all-season + monsoon/non-monsoon** breakdowns (and why threshold transfer between seasons can hurt if done blindly).

### 5.7 Coverage & "observed days"

The model-ready table only keeps rows where we can form a knowable label (enough readings in the window). The `obs_days_*` columns track this. They are **label-window info** and are **banned as features** — enforced later in Phase 13 (leakage audit).

### 5.8 EDA takeaways

- **Rare event:** PR-AUC > ROC-AUC as headline is correct.
- **Hydrology dominates:** Water level relative to danger (`wl_pct_danger`) and recent rise are strong signals.
- **Rain matters but is noisy/reanalysis:** Needs careful use.
- **Tidal/coastal are special cases** but legitimate.
- **2018 gap is real** — don't invent data.
- **Data is clean after F1 fix** and passes all audits.

---

## 6. Problems encountered & how we solved them

| Problem | Solution |
|---|---|
| **F1: `-999` sentinel hidden from `.isna()`** | Detected by inspecting the newest days. Added masking `<= -999 → NaN` in build + ingest. Proved fix only touched junk cells; audits re-passed. |
| **F2: 2018 missing** | Documented as real archive gap. Never interpolated. Reflected in reports and respected by splits. |
| **F3: 20 stations no danger level** | Correctly excluded from labelling (kept as NaN). Cannot be used as trainable labelled rows. |
| **F4: extreme base rates at tidal stations** | Recognised as genuine (tides), not error. Kept, and later analysis looks at per-station patterns. |
| **F5–F6: unreliable per-reading fields** | Used master station danger level; filtered sentinels. |
| **F7: empty discharge stub** | Tracked as M6 gap in Phase 1; later addressed via GloFAS in MI-1b. |

---

## 7. Files produced / referenced

| File | Purpose |
|---|---|
| `reports/phase2_3_data_quality.md` | Official quality report (findings, audits, EDA summary) |
| `notebooks/02_data_quality_eda.ipynb` | EDA notebook (plots, checks, F1 verification) |
| `scripts/audit_dataset.py` | Runs A1–A10 checks |
| `scripts/verify_model_ready.py` | Independent recomputation to prove correctness |
| `data/processed/flood_dataset_model_ready.csv` | Clean model-ready table (post-F1) |
| `data/interim/ffwc_*_audit.csv` | Coverage audits |

---

## 8. Verdict

**Phase 2–3: PASSED.**

- Table structure is correct.
- All automated checks **A1–A10 PASS** after fixing F1.
- The critical `-999` bug was caught, fixed, and **proven** not to affect labels or other fields.
- Findings F2–F7 are understood and documented.
- EDA gives clear intuition: rare event, monsoon-driven, hydrology strong, rain noisy/reanalysis.
- Data is **clean and trustworthy** to proceed to feature engineering (Phases 10–11) and beyond.

**Bottom line:** "Trust, but verify." We verified — and fixed one silent trap in the process.