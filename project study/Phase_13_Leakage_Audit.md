# Phase 13 — Critical Leakage Audit
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase13_leakage_audit.md` (the executed notebook is the evidence).
> Module: `notebooks/05_leakage_audit.ipynb` (built by `scripts/build_phase13_notebook.py`).
> Previous study file: Phase 10–12 (feature engineering & selection). Simple English, real
> numbers, what/why/how + problems & solutions.

---

## 0. The 60-second version

We now have a modelling table and chosen feature sets. Before any model is trained, we must
answer one uncomfortable question: **is any input secretly a peek at the answer?**

*Leakage* is when a feature contains information that would not really be available at the
moment a forecast must be issued. It makes a model look brilliant in testing and then fail in
the real world — the single most common way flood/ML projects fool themselves.

Phase 13 attacked the table along **three independent lines** and ran **11 checks (L1–L11)**.
Every `assert` had to hold; a green notebook (not a prose claim) is the evidence.

**Result: 11/11 PASS. No feature was demoted, so the Phase-12 sets stay valid.** One feature
(`n_readings`) was admitted but flagged for a later ablation.

---

## 1. Where Phase 13 sits and the rule it enforces

The binding rule is §5 of `reports/phase0_research_definition.md`: every stored value must be
a function of **days ≤ t only**, where the forecast is issued at **24:00 UTC of day `t`**.

| Phase | What | Plain words |
|---|---|---|
| 12 | Feature selection | Chose the columns |
| **➜ 13** | **Leakage audit** | **Prove none of them peeks at the future** |
| 14–16 | Experimental table → splits → baselines | Freeze the data and set the bar |

No model was trained beyond the small diagnostic fits that check L8 requires; nothing under
`data/raw/` was touched. Splitting and real modelling stay in Phases 15–17.

---

## 2. The three attack lines

| Workflow step | Attack | Checks |
|---|---|---|
| 22 | Re-derive features from the raw calendar matrix with an **independent implementation**; audit the builder source for forward references | L1–L4 |
| 23 | Screen every feature against every label (univariate AUC); prove the screen *can* catch leakage (positive control); test observation-process proxies | L5–L8 |
| 24 | Keys, splits, holes and cross-part similarity in the row space | L9–L11 |

### 2.1 Step 22 — rebuilding every derived feature from scratch

| check | what it did | result |
|---|---|---|
| **L1** static source audit | every `.shift(-` in the builder sits inside `add_labels` (2 calls); the 4 feature builders contain **0** uses of `label_*`, `obs_days`, `LEADS`; no calendar columns | **PASS** |
| **L2** calendar-formula recompute | **27 derived columns** (lags 1–7, `wl_max_lag1-3`, rises, danger ratios, rolling 7/30, exceed days, rain windows, `doy_sin/cos`) rebuilt with `shift(+k)` / backward rolling → **0 mismatches over 200,665 rows** | **PASS** |
| **L3** neighbourhood recompute | nearest-15 pool rebuilt from coordinates, membership resolved per day, means/counts recomputed from **day-`t` values only** → **47,984 cells, 0 mismatches** | **PASS** |
| **L4** upstream recompute | upstream rule re-derived (same river, +0.5 m, ≤ 250 km) → 56 stations; **802,660 cells, 0 mismatches** | **PASS** |

*Why it matters:* a formula audit is only meaningful if the "shift" is a true **calendar**
shift — L10 verifies that directly against the data.

### 2.2 Step 23 — could any feature reveal the label?

| check | question | evidence | result |
|---|---|---|---|
| **L5** | directly contains the target? | FULL/LINEAR/SELECTED share **0** columns with the 3 labels, 3 `obs_days_*`, 4 keys — and those columns *exist*, so the ban is a real choice | **PASS** |
| **L6** | indirectly reveals the target? | 62 features × 3 horizons screened: **max AUC 0.9901**, none at/above the 0.995 flag | **PASS** |
| **L7** | calculated using the target? | **positive control**: a deliberately future-derived feature reproduces the label on **100 %** of rows, AUC **1.000** → the L6 threshold *would* catch leakage; its legitimate past twin `wl_exceed_days_30` = 0.908 | **PASS** |
| **L8** | observation-process proxy? | `n_readings` vs `nb5_n_wl_danger` table below | **PASS** |

Top of the L6 screen (oriented AUC, 24 h / 48 h / 72 h → max):

| feature | 24 h | 48 h | 72 h | max |
|---|---:|---:|---:|---:|
| `wl_dist_to_danger` | 0.9901 | 0.9791 | 0.9680 | **0.9901** |
| `wl_pct_danger` | 0.9874 | 0.9736 | 0.9590 | 0.9874 |
| `wl_exceed_days_30` | 0.9458 | 0.9253 | 0.9082 | 0.9458 |
| `up_wl_pct_danger_lag1` | 0.8915 | 0.8753 | 0.8606 | 0.8915 |
| `nb3_wl_pct_danger` | 0.8890 | 0.8739 | 0.8602 | 0.8890 |

The two leaders sit close to the target **by definition** (current stage vs the published danger
level) — that is *persistence*, not leakage. L2 rebuilds both from day-`t` values with 0
mismatches, and L7 proves the screen distinguishes them from a genuinely future-derived feature.

**L8 — is the signal real, or just "a label exists"?**

| feature | AUC → target (24 h) | AUC → **label availability** | Δ PR-AUC vs persistence (2022) |
|---|---:|---:|---|
| `n_readings` | 0.475 | **0.804** | −0.0013 / −0.0001 / +0.0018 |
| `nb5_n_wl_danger` | **0.857** | 0.295 | **+0.0035 / +0.0040 / +0.0075** |

`n_readings` predicts *whether a label will exist* (0.804) but not *whether it will flood*
(0.475 = chance) and adds nothing → **ADMIT-WATCH** (ablate in Phase 38).
`nb5_n_wl_danger` is the opposite → **ADMIT**.

### 2.3 Step 24 — duplicates, holes and split hygiene

| check | evidence | result |
|---|---|---|
| **L9** | 0 duplicate station-days; train/val/test share **0 dates and 0 station-days** | **PASS** |
| **L10** | holes derived from data: **52 zero-observation stretches** (longest 2017-12-10 → 2018-12-31); **none** reaches the model-ready table; rows are consecutive calendar days for all 141 stations; label rules hold for all 3 horizons | **PASS** |
| **L11** | **0** identical 62-feature vectors across parts; each cut sits inside a model-ready-free stretch | **PASS** |

L10's label rules, per horizon (rows in the model-ready table):

| horizon | label = 0 | label = 1 | NaN |
|---|---:|---:|---:|
| `label_24h` | 190,577 | 9,982 | 106 |
| `label_48h` | 186,507 | 11,241 | 2,917 |
| `label_72h` | 182,834 | 12,340 | 5,491 |

A `0` requires **every** day of the window observed; a `1` requires an observed exceed day
inside the window — so no label can be built from an unobserved or out-of-window day.

**A useful surprise — the split already has a natural embargo:**

| cut | last model-ready row before | first after | model-ready-free days |
|---|---|---:|---:|
| 2022-01-01 | 2021-11-03 | 2022-03-15 | **131** |
| 2023-01-01 | 2022-10-30 | 2023-01-27 | **88** |

The date cut falls inside the dry-season non-reporting stretch both times, so no flood episode
can straddle a boundary. Phase 15 still makes the embargo explicit, but the risk it guards
against is largely already absent.

---

## 3. The rulings — `data/processed/feature_whitelist.csv`

One row per column: group, **the latest day each value may depend on**, availability at the
issue instant, ruling.

| ruling | n | columns |
|---|---:|---|
| `ADMIT` | 61 | everything approved by §5, formula-verified in L2–L4 |
| `ADMIT-WATCH` | 1 | `n_readings` (ablate in Phase 38) |
| `ADMIT-CONSTANT` | 1 | `day_observed` (dropped at Phase 12) |
| `BAN` | 10 | `station_id name date source`, `label_24h/48h/72h`, `obs_days_24h/48h/72h` |

Asserted: **BAN == the 10 banned columns**, **ADMIT ∪ {constant} == the 63 approved columns**,
**FULL == ADMIT − {day_observed} = 62**.

Open items closed here: the `obs_days_*`/label ban (exercised, not just declared); `n_readings`
(ADMIT-WATCH with numbers); all 19 Phase-11 columns (ADMIT, incl. `nb*_n_wl_danger`); and the
top feature `wl_dist_to_danger` (ADMIT — 0.990 AUC is persistence, proven by L7).

---

## 4. Consequence and findings

Because nothing was demoted, **FULL 62 / LINEAR 35 / SELECTED 10 are unchanged** and the
Phase-12 notebook does **not** need a re-run; a `phase13` block was appended to
`feature_sets.json` so the file stays authoritative.

Findings raised:

1. **F13-1 — the archive hole is wider than "2018".** There are **52** empty stretches, not
   one. None reaches the model-ready table — a *coverage* fact for Phase 15, not a defect.
2. **F13-2 — both date cuts already sit inside natural embargoes** (131 and 88 free days).
3. **F13-3 — `n_readings` is a selection variable**: it says when FFWC published, not when it
   flooded (availability AUC 0.804 vs target 0.475). Kept but flagged.
4. **F13-4 — the nearest-feature AUC 0.990 is below, but not far below, the 0.995 flag.** Keep
   the threshold so a future 0.996 feature gets challenged rather than trusted.
5. **Tooling lesson — the first L3 run reported 12,603 "mismatches" that were the audit's
   fault** (wrong source grid; count-vs-NaN semantics). An independent re-implementation must
   match the builder's **data lineage**, not just its formula.

---

## 5. Problems & solutions

| Problem | Solution |
|---|---|
| A feature could quietly use tomorrow's water level | Independent rebuild of all 27 derived columns; **0 mismatches**. |
| Neighbour/upstream features could use the future | Rebuilt from day-`t` (and `t-1`/`t-2`) only; 47,984 + 802,660 cells, 0 mismatches. |
| "No leakage" is an assertion, not proof | Positive control (L7): a real future-derived feature scores **1.000**, so the screen demonstrably works. |
| A high AUC (0.99) might be leakage | Shown to be persistence — rebuilt exactly and separated by L7. |
| A feature might predict *data availability*, not floods | L8 splits the two: `n_readings` → ADMIT-WATCH; `nb5_n_wl_danger` → ADMIT. |
| The 2018 gap might hide a split problem | L10 derived **52** holes from the data and proved the split lands in dry-season gaps. |
| An audit flaw masquerading as data flaw | Traced 12,603 false mismatches to the audit's own grid and fixed the method. |

---

## 6. Files referenced

| File | Purpose |
|---|---|
| `reports/phase13_leakage_audit.md` | The full audit record and L1–L11 results |
| `notebooks/05_leakage_audit.ipynb` | Executed evidence (13/13 cells, 11/11 checks green) |
| `data/processed/feature_whitelist.csv` | Per-column ruling + `latest_day_used` (73 rows) |
| `data/processed/leakage_audit.json` | Machine-readable audit output |
| `figures/phase13_auc_screen.png` | The univariate screen plot |
| `data/processed/feature_sets.json` | New `phase13` block recording the ruling |

---

## 7. Verdict

- **11/11 leakage checks green; every `assert` held.** The 62-feature set is provably a
  function of days ≤ `t`.
- **Nothing demoted** → Phase 12's FULL 62 / LINEAR 35 / SELECTED 10 stand.
- **One watch item** (`n_readings`) and **four findings** handed forward; the split's incidental
  88–131-day embargo is documented for Phase 15.

**Bottom line:** the project can now honestly say its inputs do not peek at the future — proven
by independent re-derivation and a positive control, not by assertion. This is the gate that
lets every later number be trusted.
