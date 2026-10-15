# Phase 31–32 — Robustness & Ablation (does it hold up? what actually matters?)
### Project study notes · Bangladesh flood-probability forecasting

> Sources: `reports/phase31_robustness.md`, `reports/phase32_ablation.md`,
> `scripts/phase31_robustness.py`, `scripts/phase32_ablation.py`. Previous study: Phase 30
> (error analysis).

---

## 0. The 60-second version

Two "stress-test and explain" phases.

**Phase 31 — Robustness.** Take the frozen leaders and ask: do they still work when conditions
change? Different years, unseen stations, monsoon vs dry, **extreme rainfall**, missing data, and
a **stricter issue-time rule** (only information through *t−1*). Most robust; two honest
caveats:
- **Extreme rainfall is the hardest ranking regime** — PR-AUC *falls* ~0.04–0.05 versus
  low-rain, and the ML edge over the D7 persistence bar narrows (leaders win only **28/63**
  extreme-bucket points).
- **Under a stricter issue-time rule** (same-day observation withheld) the leaders lose
  0.013–0.068 PR-AUC and **can slip below D7** at 24–48 h on unseen stations (bounded ≤ 0.015).

**Phase 32 — Ablation.** Which feature groups actually carry the skill? The answer is a clean
**hydrology story**: weather+rain alone is nearly skill-less; **geography** jumps skill to ~0.48;
**hydrology** (water levels/danger) is the single biggest layer (**+0.470**, significant 9/9) and
is what beats D7. The **station-network** layer and the `up_*` upstream columns add
**essentially nothing**.

---

## 1. Where Phase 31–32 sit

| Phase | What | Plain words |
|---|---|---|
| 30 | Error analysis | Where we still miss, and why |
| **➜ 31** | **Robustness** | **Does it hold under stress?** |
| **➜ 32** | **Ablation** | **Which inputs carry the skill?** |
| 33 | Select + freeze + test | Pick champion, read test once |

Closes **stage 53** (robustness) and **stages 55–56** (ablation matrix).

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Robustness** | Does performance hold when the evaluation conditions change? |
| **Slice** | A sub-group (year, station fold, season, rain bucket, …). |
| **Rainfall bucket** | Cell-relative rain quantiles: low ≤ p50, high (p50,p90], extreme > p90. |
| **Missingness bucket** | Fraction of NaN features: none / low / high / very-high. |
| **Issue-time variant** | Using only information up to *t−1* instead of *t*. |
| **Threshold transfer** | Applying a threshold derived elsewhere (to separate ranking loss from cut loss). |
| **Ablation** | Remove/withhold part of the input to measure its contribution. |
| **Nested group** | Cumulative feature groups G1 ⊂ G2 ⊂ … ⊂ G5 (= FULL). |
| **Increment (paired)** | The PR-AUC gain of adding one layer, with a paired bootstrap CI. |
| **Paired bootstrap CI** | Resample stations; CI on the **difference**. |
| **Macro vs micro** | Per-station average vs pooled (D8 — both reported). |
| **D7 bar** | The persistence benchmark every model must beat. |
| **Redundant** | Adds no measurable skill (e.g. the `up_*` columns). |

---

## 3. Phase 31 — robustness

### Method

- **Surviving panel:** the Phase-27/28/29/30 leaders `lightgbm`@24 h, `xgboost`@48 h,
  `xgboost_selected`@72 h, plus `logreg_linear`, `persistence` (D7) and `climatology`.
- **No new search or architecture** — the *frozen* winners re-fit per Phase-15 protocol.
- **7 cells × 3 seasons × 3 horizons**, each with its own exact 10:1 cost threshold; a second
  column applies the frozen Phase-29 threshold to every cell to separate **ranking loss** from
  **threshold-transfer loss**.
- **Rainfall / missingness buckets** are **cell-relative** (adapt to each regime).
- **Stricter issue-time variant:** every feature replaced by its **previous calendar day** value
  (valid for 98.5 % of rows), model re-fit and re-scored (labels unchanged).

### Cross-check (validates reuse)

A fresh re-fit of the six-model panel reproduces `phase27_generalisation.csv` to
**max |Δ PR-AUC| = 1.1 × 10⁻¹⁶** over 126 shared rows — so the 24-model Phase-27 table is sound
context for the models Phase 31 doesn't re-fit.

### Rainfall intensity — the hardest regime

| Horizon | bucket | base rate | leader PR-AUC (median) | leader recall (min) |
|---|---|---|---:|---:|
| 24 h | low ≤ p50 | 0.063 | 0.9885 | 0.9483 |
| 24 h | high | 0.113 | 0.9836 | 0.9545 |
| 24 h | **extreme > p90** | 0.154 | **0.9484** | 0.9399 |
| 72 h | low | 0.074 | 0.9705 | 0.8942 |
| 72 h | **extreme > p90** | 0.185 | **0.9219** | 0.8927 |

Extreme-rain windows **double the base rate** yet **lower** PR-AUC (extreme − low = −0.040 /
−0.046 / −0.049 at 24/48/72 h). Because average precision normally *rises* with prevalence, a
fall despite ~2.5× the positives means **score separation genuinely degrades** — many non-flood
rows also carry high scores. **The ML edge over D7 is smallest here**: leaders beat persistence
in only **28/63** extreme-bucket points. When rain is already extreme, yesterday's water level is
a strong carry-forward predictor.

### Missingness — flat

Median PR-AUC is **flat** across missingness buckets (0.9664–1.0000). Every low minimum is a
**small-`n` instance** (≤ 5 positives) — noise, not a missingness-driven collapse. The leaders
beat persistence in 192/231 bucket points.

### Stricter issue-time (features through *t−1*)

| Variant | leader PR-AUC min | median | max | recall min |
|---|---:|---:|---:|---:|
| standard | 0.9155 | 0.9655 | 0.9912 | — |
| **stricter** *t−1* | **0.8480** | **0.9380** | 0.9787 | 0.9072 |

The loss is **always negative** (median −0.027), grows with horizon, and concentrates in the
hardest cells (`spatial_fold_1` −0.059, `val_2022` −0.049). **The D7 margin is not robust to
this convention:** the stricter variant slips below persistence in **19/63** points (worst
−0.015). So the claim must be qualified: *"always beats D7 under the standard convention, and is
marginal at 24–48 h on unseen stations if the same-day observation is withheld."*

### Named findings

- **F31-1** reuse validated (|Δ| = 1.1e−16).
- **F31-2** leaders beat D7 in **63/63** all-season points (min margin +0.0143).
- **F31-3** year difficulty ordering val_2020 > val_2021 > val_2022 for all leaders.
- **F31-4** season edge case: on unseen-station fold 1 (72 h) the monsoon/dry gap reaches
  **0.259** (dry base rate collapses to ~0.01); leaders still beat persistence 126/126.
- **F31-5** extreme rain is the hardest regime and narrows the D7 edge (28/63).
- **F31-6** missingness does not degrade the median.
- **F31-7** stricter issue-time loses 0.013–0.068 and can slip below D7 (bounded).
- **F31-8** every result is per cell (D8).

---

## 4. Phase 32 — ablation

### The nested groups (FULL 62 = 5 disjoint layers)

| Group | Layers | n |
|---|---|---:|
| `G1_weather` | weather | 5 |
| `G2_weather_rainhist` | + rainfall history | 12 |
| `G3_plus_geography` | + geography (admin text + terrain/distances) | 26 |
| `G4_plus_hydrology` | + local hydrology (danger + water level) | 50 |
| `G5_full` | + station network (`nb3_*`/`nb5_*` + `up_*`) | 62 |
| `G5_no_up` | FULL minus the four `up_*` (E11) | 58 |

**The learner is held fixed and only the input changes** — two fixed learners (`xgboost_c01`,
`lightgbm_c00`) with their FULL-tuned settings. 6 groups × 2 learners × 7 cells × 3 seasons × 3
horizons = **756 rows (252 fits)**.

### The hydrology story (temporal macro PR-AUC, xgboost)

| Group | n | 24 h | 48 h | 72 h |
|---|---:|---:|---:|---:|
| `G1_weather` | 5 | 0.1585 | 0.1717 | 0.1852 |
| `G2_weather_rainhist` | 12 | 0.1673 | 0.1797 | 0.1986 |
| `G3_plus_geography` | 26 | 0.4759 | 0.4995 | 0.5095 |
| `G4_plus_hydrology` | 50 | **0.9775** | **0.9660** | **0.9520** |
| `G5_full` | 62 | 0.9782 | 0.9664 | 0.9534 |
| `G5_no_up` | 58 | 0.9783 | 0.9665 | 0.9531 |
| *D7 persistence* | — | *0.9429* | *0.9250* | *0.9048* |
| *climatology* | — | *0.0877* | *0.0967* | *0.1043* |

### Increments (paired station-bootstrap, 9 cell×horizon points)

| Increment | mean ΔPR-AUC | range | significant |
|---|---:|---|---:|
| `G2 − G1` (+ rainfall history) | +0.0101 | −0.012 … +0.050 | 3/9 |
| `G3 − G2` (+ geography) | **+0.3131** | +0.185 … +0.424 | 9/9 |
| `G4 − G3` (+ hydrology) | **+0.4702** | +0.389 … +0.539 | 9/9 |
| `G5 − G4` (+ station network) | +0.0009 | −0.0002 … +0.004 | 2/9 |
| `G5_no_up − G5` (drop `up_*`) | −0.0001 | −0.001 … +0.0004 | 2/9 |

**The ablation is a hydrology story:**
- **Weather alone (G1) and weather+rain (G2)** are only marginally above the no-skill climatology
  bar and **far below the D7 bar** — no usable skill.
- **Geography (G3)** recovers ~half the achievable skill (0.16 → ~0.48).
- **Hydrology (G4)** is the **single largest layer** (+0.470, significant 9/9) — it is hydrology
  that **clears the D7 bar** and reaches 0.95–0.98.
- **The station-network layer (G5−G4)** buys **+0.0009** (significant only 2/9) — marginal.
- **`up_*` is redundant** (E11 = −0.0001, CI [−0.0023, +0.0016]) — **F14-2 confirmed at full
  capacity**.

### Full-model cross-check

- **Full beats D7 in 21/21** (cell × horizon) points (min margin +0.0144).
- Monsoon vs non-monsoon gap ≤ 0.0155 on temporal cells (unlike Phase-31's unseen-station folds).
- Monotone decay 24 h > 48 h > 72 h on every protocol.

### Reused rows (E13–E16) and one honest block

- **E13** engineered vs raw-sequence DL gap: **+0.020 / +0.018 / +0.021** temporal, widening to
  **+0.060 / +0.042 / +0.032** on unseen stations (Phase-26 interaction reproduced).
- **E14** sequence length: L=3 best (0.9519 vs 0.946–0.949) — longer windows don't help.
- **E15/E16** fusion/graph/attention ablations reused from Phases 23–25.
- **E12 (soil/landcover) BLOCKED by R8** — the table carries no such groups; recorded as blocked,
  **not hidden**.

---

## 5. Problems & solutions

| Problem | Solution |
|---|---|
| Robustness claims could be cherry-picked | Fixed panel, **all 7 cells × 3 seasons × 3 horizons**, per-cell (D8). |
| Threshold change could be mistaken for ranking loss | Two columns: slice-optimal vs frozen (transfer) threshold. |
| Rain/missingness buckets could be arbitrary | **Cell-relative** quantiles, printed per cell. |
| Ablation could be confounded by tuning | Learner **held fixed**; two learners agree to ±0.001. |
| A layer could look useful by noise | **Paired station-bootstrap CI** on each increment. |
| A missing input group could be silently dropped | **E12 recorded as BLOCKED**, not skipped. |
| D7 claim could be over-stated | Qualified: standard convention yes; stricter *t−1* marginal at 24–48 h. |
| Test could creep in | `test_rows_read = 0` in both phases. |

---

## 6. Files referenced

| File | Purpose |
|---|---|
| `reports/phase31_robustness.md` | Robustness report |
| `reports/phase32_ablation.md` | Ablation report |
| `reports/phase31_cells.csv`, `phase31_rainfall.csv`, `phase31_missingness.csv`, `phase31_issuetime.csv` | Robustness slices |
| `reports/phase32_groups.csv` (756) | Nested groups |
| `reports/phase32_increments.csv` (210) | Paired increments with CIs |
| `reports/phase32_matrix.csv` (22) | E0–E16 matrix (status tagged) |
| `notebooks/31_robustness.ipynb` (10/10), `notebooks/32_ablation.ipynb` | Verification |

---

## 7. Verdict

**Phase 31 — Robustness.**
- The leaders are robust across years, seasons and stations and **beat D7 in 63/63** all-season
  points (min margin +0.014) and **21/21** in Phase 32.
- **Two honest caveats:** (1) **extreme rainfall** is the hardest ranking regime and narrows the
  D7 edge (28/63); (2) under a **stricter issue-time rule** the leaders lose 0.013–0.068 and can
  slip **below D7** at 24–48 h on unseen stations — bounded (≤ 0.015) but real, so the claim is
  qualified.
- Missingness does **not** degrade the median.

**Phase 32 — Ablation.**
- Skill is a **hydrology story**: weather+rain alone is near skill-less; **geography** helps a
  lot; **hydrology** carries the rest (**+0.470, 9/9 significant**) and is what beats D7.
- **Station network (+0.0009)** and the **`up_*` columns (−0.0001)** are **redundant**.
- Two independent learners agree — the ordering is a property of the **features**, not one model.

**Bottom line:** the model's power comes from **water-level hydrology (and geography)**, not from
weather or the station network; and while it is robust in normal conditions, it is **vulnerable
exactly where flooding is extreme and where same-day observations are withheld** — both stated as
explicit, bounded limitations rather than buried.

---

**Next in the series:** Phase 33 — Model selection, freezing the champion, and the **single
frozen test read** (the one time the ≥ 2023 period is opened).
