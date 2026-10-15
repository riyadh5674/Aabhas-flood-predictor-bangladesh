# Phase 14 — Define the experimental dataset

| | |
| --- | --- |
| **Workflow** | `flood prediction workflow.md` **Phase 14** — *create the final modelling table (Location / Timestamp / Predictor 1..N / Target / prediction horizon). Keep a copy of the raw dataset separately. Never overwrite the original.* |
| **Checklist** | stage 26 (experimental / model-ready table) |
| **Notebook** | `notebooks/06_experimental_dataset.ipynb` — **10/10 checks (E1–E10) green, 0 errors** |
| **Output** | `data/processed/experimental_dataset.csv` — **200,665 × 67, 119.9 MB** |
| **Dictionary** | `data/processed/experimental_dataset_dictionary.csv` — 67 rows, one per column |
| **Source** | `data/processed/flood_dataset_model_ready.csv` (200,665 × 73) — **read-only**, SHA-256 `15a0b78518cba0ca…` unchanged before/after (E9) |

**This is the table Phases 15–16 split and train on.**

---

## 1. The schema

Exactly the workflow's conceptual layout, in order:

| block | columns | count |
| --- | --- | ---: |
| **Location** | `station_id` (join key; `lat`/`lon` travel as predictors) | 1 |
| **Timestamp** | `date` — the day `t` the prediction is issued for (24:00 UTC convention, phase0 §5) | 1 |
| **Predictors** | the Phase-12 `FULL` set = Phase-13 whitelist `ADMIT` (61) + `ADMIT-WATCH` (1) | **62** |
| **Target** | `label_24h`, `label_48h`, `label_72h` (one column per prediction horizon) | 3 |
| | | **67** |

Five predictors are text (`river basin division district upazilla`) and are shipped **raw**;
frequency encoding stays a Phase-16 concern (LINEAR set only — LightGBM handles text).
The table is built by **column selection only** — every retained cell is byte-identical to
the audited source (`assert EXPL[c].equals(mr[c])`, E3), so the Phase-13 leakage audit
applies to this table without re-running.

## 2. Checks

| # | check | evidence |
| --- | --- | --- |
| **E1** | provenance | source 73 cols ↔ whitelist 73 rows (`ADMIT` 61 / `BAN` 10 / `ADMIT-WATCH` 1 / `ADMIT-CONSTANT` 1) ↔ Phase-12 `FULL` 62 are mutually consistent; `admit == FULL`, the 10 banned columns and the 3 targets all exist → every exclusion below is a *choice* |
| **E2** | schema contract | 67 = 1 + 1 + 62 + 3; `set(predictors) == ADMIT ∪ ADMIT-WATCH`; **0 banned columns among predictors** |
| **E3** | column selection | exactly 6 columns leave, each with a stated reason (§3); all 67 retained columns identical to source cells |
| **E4** | rows & keys | 200,665 rows, **0 duplicate station-days**, 133 stations, 2017-01-06 → 2026-10-05, time strictly increasing within every station, **0 rows in 2018** |
| **E5** | dtypes | 57 numeric + 5 text predictors; `station_id` int64, `date` datetime64; targets binary float with NaN |
| **E6** | missingness | 0 fully-empty predictors; **37/62 complete**; structural NaN `up_*` 67–70 %; gap-driven `wl_last_lag1` 0.5 % (the 2018 hole) |
| **E7** | target contract | 200,559 / 197,748 / 195,174 labelled rows, 9,982 / 11,241 / 12,340 positives (4.98 / 5.68 / 6.32 %) — **matches phase0 §4 exactly** |
| **E8** | write + round-trip | CSV written and re-read in 44.6 s; `assert_frame_equal` on values, dtypes and column order — no silent coercion |
| **E9** | immutability | SHA-256 of `flood_dataset_model_ready.csv` **unchanged** after the run (workflow: *never overwrite the original*) |
| **E10** | dictionary | 67-row `experimental_dataset_dictionary.csv`: position, block, role, whitelist group, latest day used, ruling, dtype, null profile |

## 3. What was dropped, and why (6 of 73)

| column | reason |
| --- | --- |
| `name` | identity/date key (banned); redundant with `station_id` |
| `source` | identity/date key (banned) |
| `obs_days_24h` `obs_days_48h` `obs_days_72h` | **label-window information** (banned, phase0 §5) — label availability is already visible as `NaN` in the target |
| `day_observed` | `ADMIT-CONSTANT`: available at `t` but constant in the model-ready table; dropped at Phase 12 step 18a |

Note the asymmetry that matters: **`station_id` and `date` are banned as *features* but kept
as *keys*** — the table must say which station-day a row is; feature matrices are built from
`FULL` only, which Phase 13 L5 already asserts and Phase 15/16 re-assert.

## 4. Missingness of the final table

| kind | columns | behaviour |
| --- | --- | --- |
| complete (0 % NaN) | **37/62** predictors — today's hydrology, rain windows, static geo, season, `nb*` counts | no handling needed |
| gap-driven | `wl_last_lag1..7`, `wl_max_lag1..3`, `wl_rise_*`, `wl_max_roll*` — **0.5 % → 3.5 %** | the 2018 archive hole (Phase 13 L10); lags never bridge it |
| structural | `up_rain7d_lag1/2`, `up_wl_pct_danger_lag1/2` — **67–70 %** | no upstream gauge for those stations; stays `NaN` (tree models) |
| sparse | `nb*_wl_pct_danger` 0.35 %, `upazilla` 0.16 % | no neighbour/gazetteer match |

Targets: `label_24h` 0.05 % NaN, `label_48h` 1.45 %, `label_72h` 2.74 %.

## 5. Targets (unchanged contract, re-verified)

| target | n | positives | base rate | NaN | imbalance 0:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_24h` | 200,559 | 9,982 | 4.98 % | 106 | 19.1 : 1 |
| `label_48h` | 197,748 | 11,241 | 5.68 % | 2,917 | 16.6 : 1 |
| `label_72h` | 195,174 | 12,340 | 6.32 % | 5,491 | 14.8 : 1 |

Rare-but-not-degenerate ⇒ **PR-AUC is the primary metric** (D8/phase0 §4).

Rows per year: 2017 17,453 · 2018 **0** · 2019 18,309 · 2020 20,055 · 2021 21,754 ·
2022 21,920 · 2023 23,869 · 2024 24,363 · 2025 28,252 · 2026 24,690.

## 6. Consequences handed to Phase 15 / 16

1. **No split column exists yet** — deliberately. Phase 15 (steps 25–27) assigns
   train/validation/test; the date cuts it must respect (2022/2023) already sit inside the
   natural 131-day / 88-day embargoes measured in Phase 13 L11.
2. **Label availability differs per horizon** (106 / 2,917 / 5,491 NaN) — each horizon's
   training subset is a *different row set*; never re-use an index built for one horizon.
3. **Missingness must not be imputed before the split** (Phase 13/F7): any statistic fitted
   on train only, in Phase 16+.
4. **The 5 text predictors** stay raw in this table; LINEAR/RFE paths frequency-encode them
   from **train** statistics only.
5. **`n_readings` rides along as `ADMIT-WATCH`** — it must be ablated (phase5 E7) rather
   than trusted; `nb5_n_wl_danger` was admitted on evidence.

## 7. Findings

| id | finding | action |
| --- | --- | --- |
| **F14-1** | The 73-column source carries 6 columns that must never reach a model (`name`, `source`, 3×`obs_days_*`, `day_observed`) — and they are now *absent* by construction, not by discipline | closed: exclusion enforced in the table itself |
| **F14-2** | 4 predictors are structurally NaN for 67–70 % of rows (no upstream gauge) | Phase 16: report LINEAR/SELECTED results **with and without** `up_*`; never impute from the full table |
| **F14-3** | Gap-driven lag NaN peaks at 3.5 % (`wl_max_roll30`) — the hole's footprint is small but non-zero and clusters in 2019 | Phase 15: do not treat these rows as ordinary missing-at-random in the split audit |
| **F14-4** | Round-trip through CSV is lossless (E8) → downstream code may read the table directly without a pickle | no action |

**Open items for later phases:** split assignment (Phase 15), text encoding + imputation
policy (Phase 16), `n_readings` ablation (Phase 38), per-horizon row sets (Phase 17).

---

**Next: Phase 15 — train/validation/test strategy (body steps 25–27 / checklist 27–29)**:
temporal split, spatial validation on unseen stations, monsoon vs non-monsoon evaluation —
using the embargo and base-rate-drift evidence from Phase 13. Order per **D10**.
