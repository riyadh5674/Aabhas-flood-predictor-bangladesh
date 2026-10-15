# Phase 4–5 — Target Investigation & Exploratory Data Analysis (Deeper)
### Project study notes · Bangladesh flood-probability forecasting

> This continues the study series. The original sources: `Phase 0` (label contract), audits
> (A5 etc.), and the broader EDA in Phase 5 (referenced in the workflow ledger: Phase 4–5 appear
> together in places; Phase 4 = Target investigation, Phase 5 = EDA). One file per phase-group,
> simple English, real numbers/file names, what/why/how, problems+solutions.

---

## 0. The 60-second version

Before modelling, we need to be absolutely sure the **answer we're trying to predict (the "target" / label)** is defined correctly and doesn't have hidden traps.

- **Phase 4 (Target investigation):** Proved the label contract is sound, showed that *"missing ≠ no flood"*, and explained why we keep unknown rows as **NaN** (never convert them to 0).
- **Phase 5 (EDA, deeper):** Explored label behaviour over time (drift), seasons (monsoon), geography, and coverage. The big lesson: **base rates drift over years (R1)** and we must **never pool years/seasons/stations/buckets** (D8).

Key takeaway: the target is trustworthy by construction. The rare-event nature (~5% base rate) and year-to-year drift are real properties of the problem, not data bugs.

---

## 1. Where Phase 4–5 sit

| Phase | What | Plain words |
|---|---|---|
| 0 | Research definition | Locked in question, horizons, label contract |
| 1 | Data inventory | Took stock |
| 2–3 | Quality + EDA | Cleaned/fixed, basic patterns |
| **➜ 4–5** | **Target investigation + EDA** | **Prove labels are correct; understand how floods behave over time/space** |
| 6–11 | Temporal/spatial/relationships + features | Build predictors |
| 12+ | Selection/modelling | Train/pick model |

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Target / Label** | What we want to predict (flood within 24/48/72h). |
| **Label contract** | The exact rule that says when =1, =0, or =NaN. |
| **NaN / unknown** | We don't know the answer. **Rule: never treat unknown as "no flood".** |
| **Positive / negative** | Flood (=1), no flood (=0). |
| **Base rate** | % of labelled rows that are floods (~5%). |
| **Class imbalance** | Many more negatives than positives. |
| **Label drift** | Base rate changes over time (years/seasons). Not a mistake — real. |
| **Information boundary** | Only use info known at issue time `t` (no peeking into the future). |
| **Window (forecast window)** | The next 1/2/3 calendar days `(t, t+h]`. |
| **Leakage** | Accidentally giving the model info about the future/answer. Banned. |
| **Pooling** | Averaging across years/seasons/stations/buckets. **Forbidden** (D8). Must report per cell. |

---

## 3. Phase 4 — Target investigation

**Goal:** Prove the label definition is correct, defensible, and free of the common mistake ("missing = 0").

### 3.1 The label contract (non-negotiable)

From Phase 0 (§4), implemented in `src/flood/features/build_dataset.py` and verified by `scripts/audit_dataset.py` (A5) and `scripts/verify_model_ready.py`.

```text
y_h = 1   if ANY observed reading in (t, t+h] reaches the station's official FFWC danger level
y_h = 0   if EVERY day in that window has >= 3 readings AND a known danger level AND none reaches it
y_h = NaN otherwise   (window not fully observed, OR station has no official danger level)
```

**Key rules (plain):**
1. **Missing ≠ 0.** If we can't observe the whole window properly, the answer stays **unknown (NaN)** — never assumed "no flood".
2. **Official danger level only.** Comes from `data/raw/ffwc/ffwc_stations.csv` (master). Per-reading danger values are ignored (F6).
3. **A single exceedance is enough.** If it hits danger at any time in the window → flood (1).
4. **Need enough observations.** "Observed day" means **≥ 3 readings/day** (3-hourly source), contract constant `MIN_READINGS_PER_DAY = 3`.
5. **Label-window columns are banned as features.** `obs_days_24/48/72h` describe the window used to decide the label — that is future/label info → **leakage** if used as predictors (enforced in Phase 13).
6. **Stations with no numeric danger level can't be labelled.** 20 of 153 stations fall here → their labels stay NaN permanently (F3) — correct.

### 3.2 Why "missing ≠ 0" is critical

A common, dangerous shortcut is: *"if no data, assume no flood"*. The project explicitly rejects that. The discarded `flood-v3` dataset made that mistake — Phase 0 notes **~16,051 of 20,544 labels were assumed** (treating unknown as 0). That inflates performance and is scientifically wrong.

**Our approach:** If the window isn't fully/adequately observed, we say **"we don't know" (NaN)** and **drop that row from labelling** — not guess. This keeps the target **honest and defensible**.

### 3.3 Label counts (real numbers)

Model-ready table (200,665 station-days), post-F1:

| Label | n labelled | Positives (floods) | Base rate | NaN | Imbalance (0 : 1) |
|---|---|---|---|---|---|
| **`label_24h`** | 200,559 | 9,982 | **4.98 %** | 106 | **19.1 : 1** |
| **`label_48h`** | 197,748 | 11,241 | **5.68 %** | 2,917 | **16.6 : 1** |
| **`label_72h`** | 195,174 | 12,340 | **6.32 %** | 5,491 | **14.8 : 1** |

Notes:
- As horizon grows, the chance "any flood happens in window" increases (4.98% → 6.32%). Sensible.
- NaNs increase with longer windows (harder to have the whole window fully observed). That’s expected and correct.
- **Rare event** (~5%). Imbalance grows with test-years later (test hits ~70:1 at 24h) — this is why **PR-AUC is primary** (Phase 20), not accuracy.

### 3.4 Phase 4 verdict

- Target definition is **trustworthy by construction**.
- The rule *"missing stays NaN, never 0"* is proven and enforced (A5 passes).
- Label counts are consistent with the window definition.
- **No relabelling errors.** The contract matches the research question (probability of flooding *within* next 24/48/72h).

**Bottom line:** Phase 4 concluded the target is correct — we’re predicting the right thing.

---

## 5. Phase 5 — EDA (deeper)

**Goal:** Understand *how* floods behave over time, space, and seasons — without tuning models. Phase 5 looks at **label drift**, base rates by year/season, tidal effects, coverage, and explains why we must **never pool** (D8).

### 5.1 Label drift over years (R1)

A critical finding: **base rate changes over time**. This is **risk R1: base-rate drift (16.8% 2017 → 1.19% 2023)** as recorded in Phase 0/R1.

In plain words: some years saw many more flood days, others far fewer. That’s real climate/monsoon variability — not a data error.

**Why it matters:** If you train on a high-base-rate year and test on a very low one, raw accuracy/thresholds can misbehave. The project handles this by:
- **Reporting per year/cell (never pooling)** — D8
- **Deriving thresholds on validation (train-derived)** and re-deriving/assessing per period when appropriate (Phase 29)
- Keeping **PR-AUC** as primary (robust to base-rate shifts vs accuracy)

### 5.2 Seasonality — Monsoon (JJAS)

Flooding is **strongly seasonal**. **Monsoon = June–September (JJAS)** (D1).

- Bulk of flood events occur in JJAS.
- Non-monsoon has much lower base rate (often <1–2% in some cells).
- This creates a big seasonal split in difficulty.

Later phases therefore report **all-season + monsoon/non-monsoon** breakdowns. Transferring a single threshold across seasons can hurt recall (especially for weaker models) — this is checked in robustness/threshold work.

### 5.3 Geography & tidal stations (F4)

- Base rates vary by **station** (some rivers flood often, some rarely).
- **Tidal stations** flood very frequently: **Mongla ~83.8%, Khulna ~52.3%** (F4). This is **genuine tidal behaviour**, not labelling error.
- Geography (basins: Brahmaputra/Meghna/Ganges) shows different flood regimes.

Again → must **not pool stations**. Per-station and per-bucket reporting is required.

### 5.4 Coverage over time (2018 gap + trailing)

- **2018 is a complete archive gap (F2)** — 0 readings. We never filled it. Splits respect this reality.
- **Trailing days (2026-10-04→06)** had the NASA POWER `-999` issue (F1) — fixed to NaNs (correct/unknown). Only affects very recent trailing window, not historical signal.
- Longer horizons create more NaNs (window observation requirement) — visible in `label_48h/72h` NaN counts above. That’s correct by contract.

### 5.5 Label drift explained (never pooled) — D5

Phase 0/D5 states: **"Label drift must be explained, never pooled."**

What this means in plain words:
- The flood frequency changes year-to-year (real). We **explain** that (monsoon strength, geography, base-rate drift R1).
- We **do not average** different years together into a single misleading number. Instead we report per year/cell (val_2020/2021/2022, test years, spatial folds) — this is enforced throughout (D8).

### 5.6 EDA takeaways (Phase 5)

- **Base-rate drift is real (R1).** Treat years separately.
- **Strong seasonality (JJAS).** Monsoon dominates events.
- **Rare but structured.** ~5% average, varies by year/station/season.
- **Tidal stations are special** but legitimate (F4).
- **2018 gap is real** — no interpolation.
- **Contract is consistent.** Longer horizon → slightly higher base rate + slightly more "window not observed" (NaNs).
- **Never pool (D8).** The safest, fairest way is per-cell reporting.

---

## 6. Problems & solutions

| Problem | Solution |
|---|---|
| Risk of treating "missing" as 0 | Enforced label contract (rule 1), A5 passes, documented rejection of flood-v3 shortcut. |
| Base-rate drift across years (R1) | Never pool; report per year/season/cell (D8). Derive thresholds carefully per period (Phase 29). |
| Tidal stations look extreme | Documented as F4 (genuine). Keep as-is, analyse per-station/bucket. |
| 2018 gap | Leave as missing (no interpolation). Respect in splits. |
| Longer windows → more NaNs | Correct by contract (need full/adequate window observation). Accept as honest. |
| Potential leakage via window counts | `obs_days_*` explicitly banned as features (Phase 13). |

---

## 7. Files referenced

| File | Purpose |
|---|---|
| `reports/phase0_research_definition.md` | Label contract (§4), decisions D1–D10, risks R1–R8 |
| `src/flood/features/build_dataset.py` | Implements label contract |
| `scripts/audit_dataset.py` | Runs A5 (label contract) + A1–A10 |
| `scripts/verify_model_ready.py` | Independent recomputation/proof |
| `data/processed/flood_dataset_model_ready.csv` | Final labelled table |
| `data/raw/ffwc/ffwc_stations.csv` | Official danger levels (target thresholds) |
| `reports/phase2_3_data_quality.md` | QA/EDA context |

---

## 8. Verdict

- **Phase 4:** Target is **correct, defensible, and trustworthy by construction**. *Missing ≠ no flood* is strictly enforced.
- **Phase 5:** Flood behaviour is **seasonal and drifts over years** (real). We must **explain drift, never pool** (D5/D8). The rare-event, imbalanced nature is understood and will be handled properly (PR-AUC primary).

**Bottom line:** We know exactly what we're predicting, why the definition is right, and how the target behaves in reality — no hidden assumptions.