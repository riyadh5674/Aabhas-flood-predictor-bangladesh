# Phase 18–19 — Dataset B (Sequence Windows) & Sequence Validation
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase18_dataset.md`, `reports/phase19_sequences.md`,
> `scripts/build_sequences.py`, `notebooks/18_sequences.ipynb`,
> `notebooks/09_sequence_validation.ipynb`, `data/processed/sequence_manifest.json`,
> `data/processed/sequence_policy.json`. Previous study: Phase 17 (framework).

---

## 0. The 60-second version

Classical ML (Phase 20) eats **one row per station-day** (Dataset A). But deep learning
(LSTM/GRU/TCN, Phases 21–22) wants to **look at a window of recent days** — a small movie
instead of a snapshot.

So Phase 18 built **Dataset B**: for every station-day, a tensor of the last **L days** of
**12 raw variables** — for **L = 3, 7, 14, 30**. Phase 19 then **validated** those windows
hard (the deep-learning twin of the Phase-13 leakage audit) and decided **what to do about
missing days**. Answer: **keep everything, add a mask channel** (P19-M1).

Shapes: `(200665, L, 12)` float32. Window always **ends on the anchor day t** — never peeks
forward. Missing calendar days become **all-NaN timesteps** with `mask = 0`, never a silent
shift and never fabricated data.

Gate result: **PASS** — no sequence model trains until this validation is green.

---

## 1. Where Phase 18–19 sit

| Phase | What | Plain words |
|---|---|---|
| 17 | Framework / registry | Wrote the rulebook |
| **➜ 18–19** | **Dataset B + sequence validation** | **Build & prove the DL input** |
| 20 | Classical ML (L2) | Run XGBoost/LightGBM/etc. on Dataset A |
| 21–22 | Temporal DL (L3) | Run LSTM/GRU/TCN on Dataset B |
| 23 | Fusion | Combine sequence + static + engineered |

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Dataset A** | One row per station-day, 67 engineered columns (for classical ML). |
| **Dataset B** | Windows of raw values: `(200665, L, 12)` (for deep learning). |
| **Tensor** | A multi-dimensional array, e.g. `(samples, days, variables)`. |
| **Window / sequence** | The last L calendar days ending on the anchor day *t*. |
| **Anchor day *t*** | The day we're forecasting *from*. Window ends here (never beyond). |
| **Front-pad** | Filling days **before** a station's first row with NaN. |
| **Mask channel** | An extra 0/1 channel saying which timesteps were really observed. |
| **Gap / absent calendar day** | A day with no data (e.g. the whole of 2018, R6). |
| **Imputation** | Inventing values for missing days. **We do not do this.** |
| **Scatter build** | Laying each station onto a full daily calendar before windowing. |
| **SHA-256 / manifest** | Fingerprint + record to prove the build is reproducible. |
| **P19-M1** | The chosen policy: keep-all + observation mask. |
| **Forward reach** | A window accidentally including day > t → leakage. Banned. |

---

## 3. Phase 18 — Dataset A vs Dataset B

| | Dataset A (Phase 14) | Dataset B (this phase) |
|---|---|---|
| Form | 200,665 × 67 engineered columns | `(200665, L, 12)` float32 windows + index |
| Consumers | L1/L2 models, fusion branch | L3/L4 models (LSTM/GRU/TCN/…) |
| Contents | lags, rolls, neighbours, `up_*`, labels | **12 raw/near-raw variables only** |
| Status | **immutable** (SHA re-checked) | reproducible from A/raw in one command |

**Why keep them separate?** Because it makes the **E13 experiment** interpretable:
*raw sequence → deep learning* vs *engineered features → XGBoost*. If Dataset B contained
the engineered rolling features, that comparison would be meaningless.

---

## 4. The 12 sequence variables (and why each is allowed)

| Variable | Why it's in the sequence |
|---|---|
| `wl_first wl_mean wl_max wl_min wl_last` | The day's observed hydrograph (all readings) |
| `wl_pct_danger` | Station's danger utilisation (per-row normalisation, not a roll) |
| `precip_mm` | The driver of the whole system |
| `tmax_c tmin_c` | Thermal season / snowmelt-proxy context |
| `n_readings` | Observation metadata — a real missingness signal |
| `doy_sin doy_cos` | Narrow season encoding (D1), legal at *t* |

**Excluded by contract:**
- all lag/roll/`rain_*_d`/`rise`/`exceed`/`nb3_`/`nb5_`/`up_*` (keeps E13 clean),
- all labels,
- every static/geographic column (`lat lon danger_level river elevation_m …`) — those travel
  on the **Phase-23 fusion branch** instead.

---

## 5. Window semantics (the crucial detail)

```
X[i, k, :] = variable values of station s on day (date_i − L + 1 + k)
window     = L calendar days ending ON the anchor day t   (never reaching past t)
```

How it's built:
- **Calendar scatter:** each station is laid onto its full daily grid
  (**2017-01-06 … 2026-10-05 = 3,560 days**), then windowed. So an absent calendar day
  becomes an **all-NaN timestep** — never a silent shift (proved on 30 anchors × 4 lengths).
- Days before a station's first row are **front-padded** with NaN.
- **No statistic is fitted anywhere** — raw I/O only (`fitted_statistics: none` in manifest).
- Row order = `experimental_dataset.csv` order, so labels/splits join by `sample_idx`.

**Plain point:** "missing day" is represented honestly as "empty day", not skipped or guessed.

---

## 6. What the tensors look like (real numbers)

| L | File | Shape | Size | Samples with ≥1 empty day | Front-padded |
|---:|---|---|---:|---:|---:|
| 3 | `sequences_L3.npy` | (200665, 3, 12) | 28.9 MB | **2.9 %** | 19 |
| 7 | `sequences_L7.npy` | (200665, 7, 12) | 67.4 MB | **7.5 %** | 19 |
| 14 | `sequences_L14.npy` | (200665, 14, 12) | 134.8 MB | **14.1 %** | 35 |
| 30 | `sequences_L30.npy` | (200665, 30, 12) | 289.0 MB | **26.7 %** | 62 |

**Coverage:** only **200,665 / 473,480 = 42.4 %** of station-days exist in the table. The other
**57.6 % are gaps** (the 2018 archive hole R6 + station outages). They are **not imputed
here** — Phase 19 picks the missing-timestep rule **on evidence**.

**Key number:** as L grows, more windows contain an empty day (2.9 % → 26.7 %). That
fraction drives the Phase-19 decision.

---

## 7. Phase 19 — the missing-timestep decision (S3)

Four candidate rules and their cost (samples dropped):

| Rule | L=3 | L=7 | L=14 | L=30 |
|---|---:|---:|---:|---:|
| **R1 keep-all + mask** (chosen) | **0 %** | **0 %** | **0 %** | **0 %** |
| R2 reject any gap | 2.9 % | 7.5 % | 14.1 % | 26.7 % |
| R3 min observed (> L/4 empty → drop) | 2.9 % | 6.2 % | 6.8 % | 11.8 % |
| R4 ffill age ≤ 2 then reject | 2.8 % | 5.6 % | 9.8 % | 18.6 % |

**Bias probe on the strongest alternative (R2 at L=30):**
- Drops **26.69 %** of samples, and **unevenly**: monsoon 12.9 % vs non-monsoon 43.9 %
  — a **3.4× season asymmetry**.
- Positive rate among dropped windows **3.18 %** vs **5.63 %** among kept — rejecting would
  **shift the label distribution** in an L-dependent way (breaks D12 fairness).

### Decision — P19-M1 "keep-all + observation mask channel"

> No timestep and no sample is dropped or imputed. An absent calendar day stays **all-NaN**
> in the tensor and is handed to the model as **`mask = 0`**
> (`sequence_mask_L{L}.npy`, uint8, observed = 1). Variable-level NaNs on *present* days
> (`precip_mm`, `tmax_c`, `tmin_c` — 0.027 % of observed timesteps) are replaced by 0
> **after** the training fold's standardisation. Phase 21 input =
> **`concat(12 variables, mask)` = 13 channels**. A plain NaN must never reach a layer.

**Why this is right:**
1. **Identical sample sets across L** keeps the D12 window sweep comparable.
2. **Nothing is fabricated** — interpolation would invent driver variables.
3. Even the cheapest rejected rule costs 2.9–26.7 % with **season/split bias**.

---

## 8. Phase 19 validation checks (28/28 green, gate PASS)

| Check | What it proved |
|---|---|
| **S1** chronological order | 200,665 rows, `sample_idx` contiguous; inside each of 133 stations rows are strictly chronological & unique (0 duplicates). **2018 absent (R6 confirmed).** |
| **S2** gap distribution | An empty timestep is *exactly* an absent calendar day — **0 mismatches** vs an independently rebuilt grid at every L. |
| **S3** missing-timestep policy | P19-M1 chosen; mask files written; original 5 hashes re-verified unchanged. |
| **S4** no station mixing | 2,036 sampled timesteps each matched their own `(station, day)` row — **0 violations**. |
| **S5** no forward reach | Newest day in every window **is** the anchor t; no day later than t. Label window `t+1…t+3` never inside a tensor. |
| **S6** label alignment | 100 windows × 3 horizons, labels rebuilt from **raw 3-hourly readings** — **292 comparable labels, 0 mismatches**. |
| **S7** split/holdout inheritance | All 200,665 windows inherit split/fold/holdout/season **1:1** with Phase 15. Every training cell contains **0 holdout, 0 frozen-test rows**. |

**S7 real numbers:** P1 train<2022 = 61,991; P2 ≤2019/2020/2021 = 28,711 / 44,614 / 61,991;
P3 excl_fold 1–4 = 46,793 / 45,727 / 47,207 / 46,246. 26 holdout stations appear in **0**
training-cell station sets. `test_unseen` = 19,763 rows, all holdout.

**Legal history:** windows may reach **back** across a split boundary (1,310 L=30 windows
start in an earlier year) — past days at issue time are legal (D5). Only **forward** reach is
banned.

---

## 9. Reproducibility

```powershell
python -u scripts\build_sequences.py > scripts\phase18_run.log      # 17/17, 9.0 s
```

`sequence_manifest.json` records, for every artifact: shape, dtype, byte size and **SHA-256**,
plus source table hash, variable order, calendar range, gap stats, build timestamp. The
notebook recomputes all four tensor hashes and the index hash and compares to the manifest —
**a rebuild must be byte-identical**.

**Environment:** torch **2.14.1+cpu**, numpy 2.3.2, pandas 3.0.6, scikit-learn 1.9.1,
**CUDA not available (CPU-only machine)**. Determinism: a seeded 5-step tiny training run
reproduces its five losses **exactly** on a second run.

---

## 10. Problems & solutions

| Problem | Solution |
|---|---|
| Deep nets need sequences, classical ML needs rows | Build Dataset B separately; keep E13 comparison interpretable. |
| Missing calendar days could silently shift windows | **Calendar scatter**: absent day = all-NaN timestep (0 mismatches proved). |
| Dropping gap-windows would bias by season/split | P19-M1 keep-all + mask; rejection costs up to 26.7 % with 3.4× season asymmetry. |
| Imputation would invent driver values | **No imputation** — mask channel instead. |
| Windows could leak future days | S5: newest day is always the anchor t; no forward reach. |
| Windows could mix stations | S4: every timestep matched its own station — 0 violations. |
| Splits could drift for DL | S7: 1:1 inheritance; 0 holdout/test rows in training cells. |
| Non-reproducible tensors | SHA-256 manifest; notebook re-verifies byte-identity. |
| Whole-year 2018 hole | Represented as empty timesteps (R6); never fabricated. |

---

## 11. Files referenced

| File | Purpose |
|---|---|
| `reports/phase18_dataset.md` | Dataset B build report |
| `reports/phase19_sequences.md` | Sequence validation report (gate PASS) |
| `scripts/build_sequences.py` | Builds the four tensors (17/17, 9.0 s) |
| `data/processed/sequences_L{3,7,14,30}.npy` | The four window tensors |
| `data/processed/sequence_mask_L{3,7,14,30}.npy` | Observation masks (uint8) |
| `data/processed/sequence_index.csv` | 200,665 rows (joins labels/splits) |
| `data/processed/sequence_manifest.json` | Shapes, hashes, policy record |
| `data/processed/sequence_policy.json` | Counterfactual costs of rejected rules |
| `notebooks/18_sequences.ipynb` | 32/32 checks |
| `notebooks/09_sequence_validation.ipynb` | 28/28 checks (S1a…S7d) |

---

## 12. Verdict

- **Phase 18:** Built Dataset B — four window tensors `(200665, L, 12)` from **12 raw
  variables**, calendar-scattered (no silent shifts), SHA-fingerprinted. **17/17** script
  checks, **32/32** notebook checks.
- **Phase 19:** Validated it hard — chronological, no station mixing, **no forward reach**,
  labels recomputed independently (0 mismatches), splits inherited 1:1. Chose the
  **keep-all + mask** policy (P19-M1) on evidence, avoiding a 3.4× season bias.
- **Gate: PASS.** Deep learning may now proceed (Phases 21–22).

**Bottom line:** We built the "movie" version of the data (windows) and proved — with hard
checks — that it never peeks into the future, never mixes stations, and never invents missing
values. Only then does deep learning start.

---

**Next in the series:** Phase 20 — Classical ML (XGBoost/LightGBM/etc., L2 rows, budget 40
configs, parity grid). Then Phases 21–22 (temporal DL), 23 (fusion), 24–25 (gated graph /
transformer), 26–33 (calibration, thresholds, selection, frozen test).
