# MI-4 — Label-quality audit: is PR-AUC 0.99 reachable?

| | |
| --- | --- |
| **Date** | 2026-10-16 |
| **Stage** | MI-4 (improvement plan) — label-quality / empirical-ceiling audit |
| **Module** | `scripts/mi4_label_audit.py` |
| **Inputs** | `data/processed/experimental_dataset.csv` (key + labels) · `reports/phase28_predictions.npz` (cached val_2022 probabilities, 24 models × 3 horizons) |
| **Outputs** | `reports/mi4_base_rates.csv` · `reports/mi4_label_audit.json` · log `scripts/mi4_run.log` |
| **Seed** | `20261008` · **run** exit `0` · wall ≈ 5.8 s · CPU |
| **Discipline** | Sections 1–5 are label statistics on the immutable key. Section 6 is **validation-only** (`val_2022`). The test-period *base rate* is a descriptive count (already published by Phases 4/5 and 33); **no test prediction is used and no decision depends on it.** |

**Bottom line:** a uniform PR-AUC of **0.99 is not reachable** — and it is limited *both* by
metric mathematics (very low base rate) *and*, secondarily, by genuine label noise at the
danger-level crossing. The leaders are already within ~0.03 of the information content of the
label on the primary cell. The honest ceiling is **0.96–0.97 on `val_2022`** and much lower on
the low-prevalence test years.

---

## 1. What the label actually is

`label_24h(t) = 1` **iff any observed reading in the window `(t, t+24h]` reaches the station's
official danger level**; `= 0` only if the whole window is observed (≥3 readings/day) *and* the
danger level is known; otherwise `NaN` (never an assumed "no flood"). From
`src/flood/features/build_dataset.py`: `LEADS = {24h:1, 48h:2, 72h:3}`,
`MIN_READINGS_PER_DAY = 3`.

**Two structural facts fall straight out of that definition:**

1. **The 24 h label *is* next-day nowcasting.** `label_24h(t)` uses a 1-day window, so it
   equals the observed exceedance on day `t+1` exactly. Verified: on 200,426 comparable rows the
   identity holds **99.98 %** of the time (39 mismatches, all at window edges).
2. **The horizons are nested, not independent.** Because `(t,t+24]` ⊂ `(t,t+48]` ⊂ `(t,t+72]`,
   we must have `label_24h ≤ label_48h ≤ label_72h` for the same row. Verified: **0 violations**
   across 195,051 rows where all three are defined. So Phase-27's horizons are *not* three
   independent problems — the longer windows simply contain the shorter ones.

---

## 2. Class-imbalance drift (question A)

### By split (24 h / 48 h / 72 h base rate)

| split | rows | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: |
| train (≤2021) | 77,571 | **9.60 %** | 10.72 % | 11.73 % |
| val (2022) | 21,920 | **5.08 %** | 5.96 % | 6.64 % |
| test (≥2023) | 101,174 | **1.41 %** | 1.77 % | 2.12 % |

The validation set is a *harder* problem than training (≈1.9× rarer), and the test set is
another ≈3.6× rarer again. Any PR-AUC "drop" from train to test is partly this, not model decay.

### By year (24 h)

| year | rows | 24 h base rate |
| --- | ---: | ---: |
| 2017 | 17,453 | 13.37 % |
| 2019 | 18,309 | 5.01 % |
| 2020 | 20,055 | **13.57 %** |
| 2021 | 21,754 | 6.78 % |
| 2022 | 21,920 | 5.08 % |
| 2023 | 23,869 | **0.68 %** |
| 2024 | 24,363 | 2.54 % |
| 2025 | 28,252 | 1.17 % |
| 2026 | 24,690 | 1.27 % |

- The year-to-year swing is **~20×** (13.57 % in 2020 vs 0.68 % in 2023) — the single biggest
  reason a *fixed* PR-AUC target is the wrong target.
- **2023's 0.68 % is exactly the "0.7 %-base-rate year" the plan warned about.** At that
  prevalence a no-skill ranker already scores PR-AUC 0.0068; the frozen test score (~0.75) is
  therefore a **~110× lift**, not a failure.
- **Data-coverage gap:** the key contains **no 2018 rows at all** (source archive gap) and
  begins 2017-01-06. Worth flagging, as it means "train ≤ 2021" is really 2017 + 2019–2021.

---

## 3. Label noise and fragility (question B)

### 3.1 Near-threshold crossings — the irreducible-ambiguity core

FFWC water levels are read to the **centimetre**, and the label flips on a single reading
crossing the danger level. A day (or a positive label) that hinges on a **≤2 cm** margin is a
coin-flip under realistic measurement error.

**Observed station-days sitting near the boundary** (n = 200,665):

| within | 1 cm | 2 cm | 5 cm | 10 cm |
| --- | ---: | ---: | ---: | ---: |
| share of observed days | 0.20 % | 0.31 % | 0.79 % | 1.52 % |

**Positive labels whose *smallest* crossing margin in the window is thin** — the share of
positives that are effectively a borderline call:

| horizon | n_pos | ≤1 cm | ≤2 cm | ≤5 cm | ≤10 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 h | 9,982 | 3.35 % | **4.23 %** | 9.00 % | 15.52 % |
| 48 h | 11,241 | 5.73 % | **7.22 %** | 14.49 % | 23.49 % |
| 72 h | 12,340 | 7.66 % | **9.62 %** | 18.80 % | 29.62 % |

So roughly **4 % (24 h) to 10 % (72 h) of positive labels hinge on a ≤2 cm crossing**. On
`val_2022`'s 891 positives that is ~38–64 rows whose label a ±2 cm measurement error can flip.
Shorter horizons are *less* fragile per positive but include more "hard" rows overall because
the window is smaller.

### 3.2 Observation density & label drop-out

| horizon | unlabelled rows | share |
| --- | ---: | ---: |
| 24 h | 106 | 0.05 % |
| 48 h | 2,917 | 1.45 % |
| 72 h | 5,491 | 2.74 % |

Every key row has ≥3 readings (`n_readings < 3` share = 0.00 %) and **all 133 stations have an
official danger level** (0 stations missing), so the drop-out is entirely the "whole window must
be observed" rule — it grows with horizon and is a mild, honest selection effect (a future
outage hides that window from evaluation).

---

## 4. Empirical ceiling (question C, validation-only)

A 12-model panel spanning every family (L1 linear → L4 graph/transformer) scored on the cached
`val_2022` probabilities. If a row is missed by *all* strong models, that flood is effectively
invisible to the current signal; if a negative row fools *all* of them at >0.99, the label
itself is suspect.

| horizon | base rate | n_pos | unanimous misses (all <0.01) | unanimous false alarms (all >0.99) | panel agreement | median-panel PR-AUC | xgboost PR-AUC | lift over base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 h | 5.09 % | 891 | **0** | **0** | 96.2 % | 0.9648 | 0.9681 | 19.0× |
| 48 h | 5.94 % | 1,027 | **0** | **0** | 94.9 % | 0.9540 | 0.9549 | 16.1× |
| 72 h | 6.58 % | 1,124 | **0** | **0** | 93.5 % | 0.9406 | 0.9408 | 14.3× |

**Reading it:**

- **No blind spot.** Zero floods are missed by *every* model, and zero negatives are
  unanimously over-confident. The leaders are not failing on a signal nobody can see — they are
  limited by genuine ambiguity and prevalence.
- **3.8–6.5 % of rows are genuinely contested** (the panel splits across 0.5). That is the
  irreducible-disagreement floor, and it lines up with the ~4 % fragility at 24 h.
- **The 12-model consensus ≈ the single best model** (0.9648 vs 0.9681 at 24 h) — mechanically
  consistent with MI-3c, where the ensemble failed to beat the best single config.
- **On the primary cell the model is already near the label's information ceiling:** 0.968 PR-AUC
  at a 5.1 % base rate leaves only ~0.03 on the table, and the fragility/disagreement floors
  argue most of that last 0.03 is unlearnable.

---

## 5. Verdict — is 0.99 label-limited?

**No honest route to a uniform 0.99 PR-AUC exists.** The two limiters are:

1. **Prevalence mathematics (dominant).** At fixed discrimination, PR-AUC compresses as the
   base rate falls; across the project's cells PR-AUC tracks the base rate at **+0.87**
   (page-33 base-rate study). With a 0.68 % test year, even a *perfect-ranking* model would
   still be reporting a number against 1.4 %-of-a-percent positives; a 0.99 target there is not
   a modelling problem, it is a metric artefact.
2. **Label noise (secondary).** 4–10 % of positives hinge on a ≤2 cm crossing at 1 cm
   resolution; ~4–6 % of rows are contested by the full model panel. This caps how much of the
   remaining head-room is real.

**Therefore:** MI-v2's honest target stands — **≈0.96–0.97 on `val_2022`**, higher at 24 h
(already 0.9708 with the MI-3b winner), **not** 0.99 on low-prevalence test years. MI-5/MI-6
will report the number the model achieves, not the number that was hoped for.

**Consequences for MI-5:** the scorecard should keep PR-AUC primary but be read *relative to
each cell's base rate*, not as one global figure; and the MI-3b winners (`lgbm7`/`lgbm7`/`xgb3`)
are the candidates to freeze.

---

## 6. Files & reproduce

| artefact | what |
| --- | --- |
| `reports/mi4_base_rates.csv` | base rate by horizon × {split, year, season, all} |
| `reports/mi4_label_audit.json` | nesting/identity, fragility, observation density, ceiling |
| `scripts/mi4_run.log` | full console output (exit 0, 5.8 s) |

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi4_label_audit.py   # -> mi4_base_rates.csv, mi4_label_audit.json
```

Deterministic (no randomness). Reads the immutable key and the cached Phase-28 probabilities;
trains nothing and reads no test prediction.
