# Phase 1 — Data Inventory
### Project study notes · Bangladesh flood-probability forecasting

> **How to read this series.** One file per phase, in order. Each file explains:
> (1) what we were trying to do, (2) *why* it came at this point, (3) the new words in plain
> English, (4) what we actually did, (5) the problems we hit and how we solved them, and
> (6) what the phase produced. No machine-learning background is assumed. I keep the real
> numbers and the real file names so you can open them and check for yourself.
>
> **The source document this summarises:** `reports/phase1_data_inventory.md` (the official,
> formal version). This study file is the friendly retelling.

---

## 0. The 60-second version

We are building a system that answers **one question**:

> **"What is the probability that a particular river station will flood within the next
> 24, 48, or 72 hours?"**

Before you can predict anything, you have to know **what information you actually have**.
Phase 1 was exactly that: a careful, *read-only* stocktake of every data file in the project —
what it is, where it came from, how big it is, how trustworthy it is, and where its holes are.

Think of a chef who, before cooking, opens the pantry and writes down every ingredient, how
much there is, and what is missing. That boring-sounding stocktake is Phase 1. It caught a
hidden bug that would have quietly wrecked every "today" forecast.

**Headline result of Phase 1:**
- Two data families were **complete and auditable**: government river data (FFWC) and NASA weather.
- **One silent defect (called "F1") was found and fixed the same day.**
- **Eight datasets we still needed were identified** (M1–M8), which told later phases what to go get.

---

## 1. Where Phase 1 sits in the journey

| Phase | What it did | In plain words |
|---|---|---|
| **Phase 0** | Research definition | Decided *what* we predict, *how far ahead*, and the ground rules |
| **➜ Phase 1** | **Data inventory** | **Took stock of every data file we have** |
| Phase 2–3 | Data quality & EDA | Hunted and fixed defects, explored patterns |
| Phase 4 → | Target, EDA, features, models | Built the actual predictor |

**Why right after Phase 0?** Because Phase 0 first nailed down the question — *"probability of
flooding at a station, 24/48/72 hours ahead"* — and the rules, e.g. *"a missing reading is never
treated as 'no flood'."* Only once you know the question can you check whether your data can
answer it. Phase 1 is that check. The question tells you **what data you need**; the inventory
proves **whether you actually have it, for how long, and how good it is.**

Phase 1 corresponds to **stage 05** of the project's master workflow
(`final workflow.md`), and it feeds directly into the Phase 2/3 quality notebook.

---

## 2. Words you'll meet (and what they really mean)

Come back to this box whenever a term is unclear. Everything here is used later in the notes.

| Word | Meaning in this project |
|---|---|
| **Dataset** | A table of data. Usually "rows" (one record each) × "columns" (one attribute each). |
| **Station** | A real river-measuring point in Bangladesh. A person/device reads the water height there. There are 153 official stations. |
| **River gauge / reading** | The measured water height (in metres) at a station at some moment. |
| **3-hourly** | Measured 8 times a day (every 3 hours), not continuously. |
| **Danger level** | The **official flood height** for a station, published by the government. When the water passes this line, that spot is officially "flooding". **This is our definition of a flood.** |
| **Target (or label)** | The thing we want to predict — here, "will it flood?" (yes/no). |
| **Feature (or predictor)** | A clue we use to predict — rain, water level, soil wetness, etc. |
| **Prediction horizon** | How far ahead we look: 24 h, 48 h, or 72 h. |
| **Station-day** | One station on one day. Our basic unit of prediction. |
| **Raw data** | The untouched originals, exactly as downloaded. Never edited. |
| **Processed data** | Raw data cleaned and combined into useful tables. |
| **API** | A web address a program calls to fetch data automatically (instead of a human clicking "download"). |
| **Reanalysis** | **Modelled** past weather — a best-guess reconstruction from satellites and models, *not* a direct rain-gauge measurement. Important honesty caveat. |
| **Sentinel / missing code** | A special fake number a provider uses to mean "no value", e.g. `-999`. Dangerous because software thinks it is a real number. |
| **Manifest** | A receipt: a file listing every download we made (which station, which date, when). |
| **Immutable** | "Do not change." Our core data files are frozen; to fix something you rebuild, you don't hand-edit. |
| **QA / QC** | Quality Assurance / Quality Control — systematic checking of data. |
| **EDA** | Exploratory Data Analysis — looking at data with tables and graphs to understand it. |
| **Metadata** | "Data about data" — identity and location info (name, coordinates), not the measurements. |
| **Provenance** | The audit trail — proof of where each piece of data came from. |
| **Base rate** | How often flooding happens on average; here about 5 % of station-days. It is rare, which makes the problem hard. |

---

## 3. The one job of Phase 1

**The job:** read every file under the `data/` folder and document it — **without changing a
single byte.**

Two deliberate rules made this trustworthy:

1. **Read-only.** Phase 1 never edits data. If something looks wrong, you *write it down as a
   finding* and fix it later, with a record. (The next phase, Phase 3, did the fixing.)
2. **Every file gets a role.** No file is left saying just "data" — each is labelled with what
   it is *for*.

That second rule uses four roles, which are worth understanding because they organise the whole
project:

| Role | Meaning | Everyday analogy |
|---|---|---|
| **TARGET** | The answer we want. | The exam answer key. |
| **PREDICTOR** | The clues we use. | The hints in the question. |
| **METADATA** | Identity & location. | The name and address on the exam paper. |
| **PROVENANCE** | The audit trail. | The receipt showing where the paper came from. |

The Phase 1 report used these tags (D1–D7) for every dataset below.

---

## 4. What we found — the inventory

### D1 — FFWC station master (the list of stations)

| Item | Value |
|---|---|
| File | `data/raw/ffwc/ffwc_stations.csv` (24 KB) |
| Source | Bangladesh government flood agency (FFWC) official API |
| Role | **METADATA + TARGET threshold** |
| Contents | 153 stations; for each: name, coordinates, river, basin, and **danger level** |

This is the **address book** of the project. It tells us *where* the stations are and, crucially,
what height counts as "flooding" at each one. Of 153 stations, **133 have a numeric danger
level** — the other 20 are marked with a dash ("-") and so can *never* be given a flood/no-flood
answer.

**Why it matters:** without the danger level, there is no definition of a flood. This one file
defines the entire target.

### D2 — FFWC observed water level (the actual river heights)

| Item | Value |
|---|---|
| Files | `data/raw/ffwc/waterlevel/{station}/{year}.csv` — **992 files, 143.6 MB** |
| Source | FFWC API, fetched station-by-station and day-by-day |
| Role | **PREDICTOR + TARGET source** |
| Size | **1,052,044 readings** |
| Period | 2017-01-01 → 2026-10-06 (**2018 is missing** — see F2) |

This is the **heart of the project**: the actual recorded river heights, every 3 hours. It is
both a *clue* (today's height helps predict tomorrow) and the *source of the answer* (we check
whether the height crossed the danger level).

Two things to know:
- **The 2018 hole is real.** The government archive simply has no 2018 readings. We never
  pretend otherwise and never "fill in" the gap.
- Readings sometimes contain a **sentinel value `≤ -1000`** to mean "broken/no data"; that gets
  removed when we build the clean table (recorded as finding F5).

### D3 — NASA POWER weather (rain and temperature)

| Item | Value |
|---|---|
| File | `data/raw/rainfall/weather_station_daily.csv` (30.9 MB) |
| Source | NASA POWER API, sampled at each station's coordinates |
| Role | **PREDICTOR** |
| Size | **550,341 rows** = 153 stations × 3,597 days (perfectly balanced) |
| Period | 2016-12-01 → 2026-10-06 |
| Variables | `precip_mm`, `tmax_c`, `tmin_c`, `tmean_c` |

Rain is an obvious flood clue, so we pull daily rainfall for every station.

**The honesty caveat that follows this file everywhere:** NASA POWER is **reanalysis**, i.e.
*modelled* rainfall, not rain-gauge measurements. This is stated in every report and is a
permanent caveat (tracked as risk **R5**). It also has a hidden trap — the `-999` bug (F1),
described in section 5.

### D4 — Harvest manifest (the receipt)

| Item | Value |
|---|---|
| File | `data/raw/ffwc/harvest_manifest.csv` (317 KB) |
| Role | **PROVENANCE** |
| Size | 20,655 rows — every `(station, date)` we ever requested |

This is the **audit trail**: a list of every download attempt. It serves two purposes: it makes
the download *resumable* (if it crashes, we know what is already done), and it lets a future
reviewer **prove which API calls were made** — essential for a thesis.

### D5 — Interim audits (internal report cards)

Two small summary files (`ffwc_station_audit.csv`, `ffwc_waterlevel_audit.csv`) that describe,
per station and per station-year, how many readings exist, how many days are covered, and how
often flooding occurred. These are our **report cards** on the raw data.

### D6 — Derived / processed datasets (the built table)

| File | Shape | Meaning |
|---|---|---|
| `data/processed/flood_dataset.csv` | 357,250 × 73 | **Every** station-day (whether observed or not) |
| `data/processed/flood_dataset_model_ready.csv` | 200,665 × 73 | Only days with **enough known info to have an answer** — the table we actually model on |

The difference matters. The "full calendar" has a row for *every* station *every* day. But some
days have no reading, so we cannot say flood/no-flood. The **model-ready** table keeps only the
rows where the answer is knowable (**200,665 station-days across 133 stations**). This table,
after a later cleanup, becomes the project's **immutable experimental table** — the frozen
centre of the whole study.

### D7 — Geospatial data (added a little later)

The first inventory pass (2026-10-07) did not yet have map data. Soon after, in **Phase 11
(2026-10-08)**, we acquired:
- **Elevation & slope** from NASA SRTM satellite terrain (30 m resolution),
- **River centre-lines** from OpenStreetMap (where rivers are),
- **Administrative boundaries** (district/division maps).

These become *static clues* (terrain and distance to rivers never change day to day). They were
recorded back into this inventory for completeness as **D7**.

> **The big picture of section 4:** we have (a) *where* the stations are and what flooding
> means there (D1), (b) the *actual river heights* over time (D2), (c) *rain and temperature*
> (D3), (d) an *audit trail* (D4–D5), (e) the *assembled table* (D6), and (f) *terrain maps*
> (D7). That is a complete, sensible toolkit for the question we asked.

---

## 5. The bug we caught: F1 — the “−999” sentinel

This is the single most valuable thing Phase 1 did. It is worth telling as a story.

### 5.1 What happened
NASA POWER returned, for three recent days (2026-10-04 → 2026-10-06, the newest days in the
file), the value **`-999` for rain and temperature on all 153 stations** — **459 rows** in total.
`-999` is NASA's code for *"no value available yet."*

### 5.2 Why it was invisible
A normal check for missing data asks: *"is this cell blank (NaN)?"* But `-999` is **not blank —
it is a number.** So every "0 missing values" check happily reported *"all good"*, even though
these cells were junk.

### 5.3 The damage it would have done
Our "recent rain" features are rolling sums, e.g. *rain over the last 3 days*. If one day in
that window is `-999`, the sum becomes a **huge negative number**. In the built table, `rain_3d`
plunged to **−1997.9 mm** (impossible — rain cannot be negative). A machine-learning model would
see this as a bizarre "anti-rain" signal on **exactly the newest days** — the days a real
early-warning system would be asked about *first*. So the bug sat precisely where it would do
the most harm.

### 5.4 How we fixed it
1. **Root cause found:** the download code used a shortcut that treated "no value" as a real
   number, and silently turned some missing rain into `0` ("assume dry"). Both are wrong.
2. **Code fix:** a new helper that turns "no value" and any value `≤ -999` into a proper blank
   (NaN) for rain and temperature.
3. **Build fix:** when reading the raw file, the builder now masks `≤ -999` **on read**, leaving
   the raw file untouched (respecting the "never edit raw data" rule).
4. **Result:** **1,836 junk cells became clean blanks** (459 rows × 4 columns).

### 5.5 How we *proved* we didn't break anything else
Before and after the fix, we compared snapshots:
- same number of rows (200,671), same stations (133), same date range,
- the **flood labels were identical** (same counts of flood / no-flood / unknown),
- only the junk "negative rain" lines changed.

Then two independent checkers re-ran (**all checks passed**, zero mismatches against a
hand-recomputed version). That is the standard we hold to: *fix it, then prove the fix changed
only the thing it was meant to change.*

### 5.6 The honest leftover
The 244 newest rows now legitimately have **blank** rain. That is the truth (NASA did not know
yet), and later modelling handles blanks properly instead of inventing a value.

> **Lesson in one line:** *"no data" is not the same as "zero", and a fake number that hides in
> plain sight is more dangerous than an obviously empty cell.*

---

## 6. The other findings (F2–F7), in plain words

| ID | Finding | Why it matters |
|---|---|---|
| **F2** | 2018 has **no readings at all** (0 out of 31,390 possible days). | A genuine archive gap. We never interpolate it; models must cope without it. |
| **F3** | **20 of 153 stations** have no numeric danger level. | For them, flood/no-flood can never be decided → their rows stay "unknown" forever. |
| **F4** | Two coastal stations flood *very* often (Mongla ≈ 84 %, Khulna ≈ 52 % of days). | This is real tidal behaviour, not a data error — but it means a few stations dominate the "flood" examples. |
| **F5** | River readings use a sentinel `≤ -1000` for "broken". | It is already filtered out at build time — same family of trap as F1, but known and handled. |
| **F6** | The per-reading danger level inside the files is unreliable. | We ignore it and use the official value from D1 instead. |
| **F7** | An expected discharge (river-flow) stub file was empty (3 bytes). | Told us the river-flow data source was not actually harvested — one of the missing items below. |

None of these were disasters — they were *known and documented*, which is exactly the point of
an inventory. Problems you have named are problems you can plan around.

---

## 7. What we were still missing (M1–M8)

Phase 1 also wrote down what we *needed but did not have*. This list is the to-do that drove
later phases.

| # | Missing dataset | Needed for | Status after Phase 1 |
|---|---|---|---|
| M1 | Administrative boundaries | map / location picker | **Acquired** in Phase 11 |
| M2 | Elevation (DEM) & slope | terrain clues | **Acquired** in Phase 11 |
| M3 | River network / distance-to-river | proximity clues | **Acquired** in Phase 11 |
| M4 | Land cover / soil / hazard zones | extra static clues | still open |
| M5 | Independent flood-extent maps | sanity-checking our results | still open |
| M6 | River **discharge** (upstream flow) | an independent upstream driver | still open at Phase 1 (important!) |
| M7 | Weather-model **forecasts** (rain ahead) | longer true lead time | optional upgrade |
| M8 | **Observed gauge rainfall** | replace the "reanalysis" caveat | still open |

Two of these mattered a lot later. **M6 (upstream discharge)** and **M8 (real soil moisture /
rain)** are exactly the kind of extra clues you'd hope would improve the model — and, as you'll
see in later phases, chasing them became the main improvement programme.

---

## 8. Verdict of Phase 1

In plain terms, Phase 1 concluded:

- ✅ **Two data families are complete and auditable:** government hydrology (D1–D2, D4–D5) and
  NASA weather (D3). The target (flood / no-flood) is **trustworthy by construction** — based on
  the official danger level, never on assumed zeros.
- 🛠️ **One new defect (F1) was found here and fixed the same day**, and both independent audits
  re-passed on the rebuilt data.
- 📋 **Eight needed datasets (M1–M8) were missing**, five of which held up the spatial/map work.

**Next up:** Phase 2 — load & structural inspection, and Phase 3 — data-quality investigation,
in `notebooks/02_data_quality_eda.ipynb`. F1 was already fixed and verified by then, and the
notebook re-checks it while hunting for further defects.

---

## 9. If you remember only five things

1. **Phase 1 = taking stock, not building.** It read every data file and wrote down what each
   one is, where it came from, and how good it is — without changing anything.
2. **Every dataset got a role:** the *answer* (target), the *clues* (predictors), the
   *identity* (metadata), or the *receipt* (provenance).
3. **A flood is defined by the official danger level** (D1), and the actual river heights (D2)
   are the heart of the project.
4. **The −999 bug (F1)** is the poster child for "always look at raw values, not just blank
   checks": a fake number masquerading as real rain would have corrupted the newest forecasts.
5. **Naming your gaps is progress.** Listing what's missing (M1–M8) is what let later phases go
   and get it.

---

### Quick file-reference map

| What | Where |
|---|---|
| Formal Phase 1 report | `reports/phase1_data_inventory.md` |
| Station master (danger levels) | `data/raw/ffwc/ffwc_stations.csv` |
| Observed water levels | `data/raw/ffwc/waterlevel/` |
| Weather (rain/temp) | `data/raw/rainfall/weather_station_daily.csv` |
| Download receipt | `data/raw/ffwc/harvest_manifest.csv` |
| Model-ready table | `data/processed/flood_dataset_model_ready.csv` |

*Written as a study companion. If any sentence here disagrees with the formal report, the
formal report wins — tell me and I'll fix this file.*
