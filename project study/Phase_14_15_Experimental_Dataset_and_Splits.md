# Phase 14–15 — Experimental Dataset & Split Design
### Project study notes · Bangladesh flood-probability forecasting

> Covers Phase 14 (build immutable experimental table) and Phase 15 (split design: train/val/test + spatial folds). Sources: `reports/phase14_experimental_dataset.md`, `reports/phase15_split_design.md`, `feature_sets.json`, `split_assignments.csv`.

---

## 0. The 60-second version

Now that features are selected and leakage-checked, we freeze the **single modelling table** and decide **who trains, who validates, who is tested**.

- **Phase 14:** Assemble `data/processed/experimental_dataset.csv` — **immutable**. Join all approved features, enforce rules, verify.
- **Phase 15:** Create splits (`split_assignments.csv`) — **train ≤ 2021 / val 2022 / test ≥ 2023** (frozen). Add **7-day embargo**, **spatial folds** (station-disjoint), season flags. **Test is never used to select.**

This is the "rules of the game" before baselines/modelling.

---

## 1. Where Phase 14–15 sit

| Phase | What | Plain words |
|---|---|---|
| 12–13 | Selection + leakage audit | Chose subsets, proved no leakage |
| **➜ 14–15** | **Experimental dataset + splits** | **Freeze table; lock train/val/test + folds** |
| 16 | Baselines | First models on these splits |
| 17+ | Framework/modelling | Use same splits everywhere |

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Experimental dataset** | The single, final modelling table (joined, cleaned, rule-enforced). |
| **Immutable** | Never edited after creation. To fix, rebuild with new name/record. |
| **SHA-256** | Fingerprint to prove file identity. |
| **Split** | Which rows go to train/val/test. |
| **Temporal split** | Split by time: past (train), recent (val), future (test). |
| **Spatial fold** | Station-disjoint groups (hold out whole stations) to test spatial generalisation (P3). |
| **Station holdout / `station_holdout`** | Rows excluded from train/val (special folds) to avoid leakage. |
| **Embargo (7-day)** | Gap between blocks to reduce near-time leakage. |
| **Monsoon (JJAS)** | June–September flag. |
| **Test frozen** | Test set is **reserved** — never used for tuning/selection. |
| **P1/P2/P3** | Protocols: P1 temporal primary, P2 rolling origin, P3 spatial CV. |

---

## 3. Phase 14 — Experimental dataset

**Goal:** Build a single, **immutable** modelling table with all approved features.

### 3.1 What goes in

- Join **model-ready** base (`flood_dataset_model_ready.csv`) with **temporal** + **spatial** engineered features.
- Keep only rows with knowable labels where appropriate, and only approved features.
- Enforce label contract (NaN stays NaN; never 0).
- Enforce banned columns absent.
- Drop `day_observed` constant if present (noted in `feature_sets.json` as dropped).

### 3.2 Key properties (real)

From workflow/Phase 14 notes:
- **Shape:** **200,665 × 67** (as recorded in Phase 14 report) — this is the **immutable experimental table**.
- **Rows:** 200,665 station-days, **133 stations**.
- **Date range:** covers 2017–2026 (with 2018 gap known).
- **Labels:** `label_24h/48h/72h` present with NaNs where unknown.
- **Features:** aligned to approved sets (spatial 19 added).
- **SHA-256:** recorded (e.g. `a6748529083fcdce…`) to prove immutability.

**File:** `data/processed/experimental_dataset.csv` — **git-ignored** (large) but fingerprinted in reports.

### 3.3 Why immutable

Once frozen, **nothing** changes this file. If we need to add v2 features (MI-v2), we create **separate** files (`mi_v2_features.csv`) and merge at runtime — we don't rewrite `experimental_dataset.csv`. This prevents accidental drift and makes every result reproducible.

---

## 4. Phase 15 — Split design

**Goal:** Lock train/val/test + spatial folds so **every phase uses the same splits**.

### 4.1 Temporal split (main)

Two numbers describe "train" and "val", and it is worth keeping them straight.

**Full split** (all rows, before any station holdout):

| Split | Rule | Rows | Dates | Stations | Purpose |
|---|---|---|---:|---|---|
| **train** | **≤ 2021** | **77,571** | 2017-01-06 → 2021-11-03 | 105 | Fit models |
| **val** | **2022** | **21,920** | 2022-03-15 → 2022-10-30 | 105 | Tune / select / calibrate |
| **test** | **≥ 2023** | **101,174** | 2023-01-27 → 2026-10-05 | 133 | **Frozen — never used to select/tune** |

**Primary modelling subset** (used by P1 and by every script via `baseline_model.load()`): drop the
**fold-0 holdout stations** from train and val → **train 61,991 / val 17,538**, test unchanged at
**101,174**. These are the row counts you will see in run logs. The cuts are **2022-01-01** and
**2023-01-01**, and they are the same dates Phase 12 trained on and Phase 13 audited.

Base rate drifts hard across time: **9.60 %** (train) → **5.08 %** (val) → **1.41 %** (test) at 24 h,
and 16.81 % (2017) → 1.19 % (2023) per year — a **14×** swing. A single pooled score would be
meaningless, so everything is reported per year / season / bucket (**D8**, **R1**).

Seed **20261008** used consistently.

### 4.2 7-day embargo (a guard, not a remover)

An `EMBARGO_DAYS = 7` rule (label window 3 d + buffer) is coded between blocks so no label window can
straddle a cut. In practice it **flags 0 rows**: every cut lands inside a natural dry-season gap of
**88–131 days with no model-ready row at all** (e.g. 2021-11-03 → 2022-03-15 = 131 days). The embargo
stays in code as a safety net; the real gap does the work. 2018 is empty, so no cut can fall in the
archive hole (**R6**).

### 4.3 Spatial folds (station-disjoint) for P3

We need to test **spatial generalisation**: does it work on **stations never seen in training**?

- **5 station folds total** (`spatial_fold` ∈ {0,1,2,3,4}), balanced on label count within early/late cohorts.
- **Station-disjoint**: all rows of a station stay in the same fold.
- **Fold 0 (frozen)**: **26 stations** held out of **every** training set (`station_holdout`); folds **1–4** are the rotating CV folds.
- **`station_holdout`** flags the fold-0 rows excluded from train/val (this is the 77,571 → 61,991 drop above).

This gives **P3 (spatial CV)**: train on 4 folds, score the left-out fold — during development. **P4**
then reuses fold 0 for the final "unseen locations" claim on 2023+.

### 4.4 Season flags

- **`monsoon` / `season`** (JJAS = June–September) — matches **D1**. Used for monsoon/non-monsoon breakdowns.

### 4.5 Split assignments file

**File:** `data/processed/split_assignments.csv` — maps each `(station_id, date)` to:
- `split` ∈ {train,val,test}
- `spatial_fold` ∈ {0,1,2,3,4}
- `station_holdout` (bool)
- season flags

This is **frozen** and reused by **every** script/notebook from Phase 16 onward (and by MI-v2 scripts via `baseline_model.load()`).

### 4.6 Protocols (P1–P3)

| Protocol | Meaning |
|---|---|
| **P1 (temporal primary)** | Train on ≤2021, validate/select on 2022, test on ≥2023 — main result. |
| **P2 (rolling origin)** | Year-by-year rolling evaluation (robustness). |
| **P3 (spatial CV)** | Station-disjoint folds (generalisation to unseen stations), during development. |
| **P4 (frozen unseen test)** | Score fold-0 stations only in 2023+ — the final unseen-location claim. |
| **P5 (season)** | Every metric additionally split JJAS vs rest. |

### 4.7 Two traps this design names out loud

- **Year drift (R1):** base rate falls 16.81 % → 1.19 % (2017 → 2023). We never pool years.
- **New-network stations (R7):** 23 stations first appear *after* the split (first seen ≥ 2023 —
  Phase 16's `test_new_network` mask uses first-year > 2023, so it reports 22); their test base rate
  (**4.22 %**) is 3× the trained bucket's (**1.30 %**). They are **not** holdout stations, but no model
  ever trains on them, so they are reported as a **third test bucket** — never folded into "seen".

---

## 5. Why this matters

- **Prevents leakage:** test never seen in tuning/selection.
- **Fair comparison:** all models (classical + DL + MI-v2) use identical splits.
- **Reproducible:** immutable table + frozen splits + fixed seed.
- **Honest generalisation:** spatial folds test "new stations", temporal split tests "future time".
- **Matches contract:** enforces D8 (never pool years/seasons/stations/buckets).

---

## 6. Problems & solutions

| Problem | Solution |
|---|---|
| Risk of temporal leakage near boundaries | 7-day embargo rule in code; natural 88–131-day dry-season gaps actually separate the blocks (embargo flags 0 rows). |
| Need to test unseen locations | Station-disjoint spatial folds + `station_holdout`. |
| Drift across years (R1) | Keep temporal split as-is (val 2022 separate from test ≥2023); report per cell. |
| Drift between experiments | Immutable experimental dataset + frozen splits. |
| 2018 gap | Splits respect real dates (no interpolation). |

---

## 7. Files referenced

| File | Purpose |
|---|---|
| `reports/phase14_experimental_dataset.md` | Build details, shape, verification, SHA |
| `reports/phase15_split_design.md` | Split rules, embargo, folds, rationale |
| `data/processed/experimental_dataset.csv` | **Immutable** modelling table |
| `data/processed/split_assignments.csv` | Frozen split/fold assignments |
| `data/processed/feature_sets.json` | Feature subsets |
| `scripts/baseline_model.py` | Loads dataset+splits consistently |

---

## 8. Verdict

- **Phase 14:** Experimental dataset built immutably (**200,665 × 67**, 133 stations). Verified against rules; fingerprinted.
- **Phase 15:** Splits frozen: full **train≤2021 (77,571) / val 2022 (21,920) / test≥2023 (101,174)**; the primary modelling subset, excluding the fold-0 holdout stations, is **61,991 / 17,538 / 101,174**. 7-day embargo guard (flags 0 rows — natural gaps do the work), 5 station folds (fold 0 held out), `station_holdout`, season flags. Test reserved.
- **P1–P5** defined (primary ranking protocols P1–P3; P4/P5 are reporting protocols). Same splits used everywhere (including MI-v2).

**Bottom line:** The "playing field" is locked. From here on, evaluation is apples-to-apples.