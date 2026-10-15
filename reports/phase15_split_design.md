# Phase 15 — Train/validation/test strategy

| | |
| --- | --- |
| **Workflow** | **Phase 15 — Train/validation/test strategy** — *do NOT start with a normal random 80/20 split* → `25` temporal → `26` spatial (unseen locations) → `27` monsoon |
| **Checklist** | stage 27 (temporal split, respect **R1** + **R6**) · stage 28 (spatial design, respect **R7**) |
| **Notebook** | `notebooks/07_split_design.ipynb` — **10/10 checks (S1–S10) green, 0 errors** |
| **Outputs** | `data/processed/split_assignments.csv` (200,665 × 9, 11 MB) · `data/processed/splits.json` (design + evidence manifest) |
| **Untouched** | `experimental_dataset.csv` — SHA-256 `a6748529083fcdce…` unchanged (S9) |

No model was trained here. The notebook *defines and proves* the strategy; Phases 16+ fit on it.

---

## 1. Step 25 — temporal split

| part | rows | date range | stations | 24 h base rate | 48 h | 72 h |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| **train** | 77,571 | 2017-01-06 → 2021-11-03 | 105 | **9.60 %** | 10.72 % | 11.73 % |
| **val** | 21,920 | 2022-03-15 → 2022-10-30 | 105 | **5.08 %** | 5.96 % | 6.64 % |
| **test** (frozen) | 101,174 | 2023-01-27 → 2026-10-05 | 133 | **1.41 %** | 1.77 % | 2.12 % |

0 shared dates between any two parts (asserted). Cuts are **2022-01-01** and **2023-01-01** —
the same dates Phase 12 trained on and Phase 13 audited.

**Why those dates hold (S4, measured not assumed):**

| cut | last row before | first row after | model-ready-free days | rows within 7 d |
| --- | --- | --- | ---: | ---: |
| 2020-01-01 | 2019-11-12 | 2020-02-19 | 98 | 0 |
| 2021-01-01 | 2020-11-09 | 2021-02-18 | 100 | 0 |
| 2022-01-01 | 2021-11-03 | 2022-03-15 | **131** | 0 |
| 2023-01-01 | 2022-10-30 | 2023-01-27 | **88** | 0 |

**Every** cut — including the rolling-fold cuts — sits inside a dry-season stretch of 88–131
days with no model-ready row at all, so the explicit `EMBARGO_DAYS = 7` rule (label window
3 d + buffer) flags **0 rows** and no label window can straddle a cut. The embargo code stays
as a safety net; the natural gap does the work. 2018 is empty, so no cut can land in the
archive hole (**R6**).

## 2. Rolling origin (D8, R1)

Three pre-test folds, each train strictly before its validation year minus the embargo,
holdout stations excluded from both sides:

| fold | train rows | val rows | val 24 h | val 72 h |
| --- | ---: | ---: | ---: | ---: |
| train ≤ 2019 → val 2020 | 28,711 | 15,903 | 14.18 % | 16.32 % |
| train ≤ 2020 → val 2021 | 44,614 | 17,377 | 7.03 % | 8.40 % |
| train ≤ 2021 → val 2022 | 61,991 | 17,538 | 5.09 % | 6.58 % |

Fold-2022's train set is **byte-identical** to the primary train set (asserted).

Per-year `label_72h` base rate (**R1** — never pool years):

| 2017 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16.81 % | 6.47 % | 15.83 % | 8.32 % | 6.64 % | **1.19 %** | 3.69 % | 1.79 % | 1.90 % |

**14× drift** 2017 → 2023: a single pooled score would be meaningless, which is exactly why
D8 mandates per-fold/per-year reporting with bootstrap CIs (Phase 20).

## 3. Step 26 — spatial validation (unseen locations)

**5 station folds**, balanced on label count *within* early/late cohorts (first observed
≤ 2017 / ≥ 2018) so **R7**'s network growth cannot masquerade as spatial generalisation:

| fold | stations | early | late | rows | pos (24 h) | pos (72 h) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 (frozen)** | 26 | 17 | 9 | 39,725 | 1,926 | 2,501 |
| 1 | 27 | 17 | 10 | 40,481 | 1,854 | 2,407 |
| 2 | 27 | 17 | 10 | 41,511 | 1,913 | 2,206 |
| 3 | 26 | 17 | 9 | 38,246 | 2,390 | 2,802 |
| 4 | 27 | 17 | 10 | 40,702 | 1,899 | 2,424 |

Positive balance max/mean = **1.20**; 85 early + 48 late stations = 133.

| protocol | trains on | tests on | measures |
| --- | --- | --- | --- |
| **P1** primary | train years, no holdout | val 2022 | model selection |
| **P2** rolling origin | growing window ≤ Y−1 | val year Y ∈ {2020, 2021, 2022} | temporal generalisation, per year |
| **P3** spatial CV | train years, 4 of 5 folds | left-out fold, same years | unseen-location transfer **during development** (14,784–16,264 test rows/fold, station-disjoint every time) |
| **P4** frozen unseen test | *nothing* | fold-0 stations in 2023+ | the final unseen-location claim |
| **P5** season | — | — | every metric additionally split JJAS vs rest |

**Test period in three buckets (S6)** — the R7 trap, caught:

| bucket | rows | stations | 24 h rate | 72 h rate |
| --- | ---: | ---: | ---: | ---: |
| trained stations (in P1 train) | 75,889 | 84 | 1.30 % | 1.97 % |
| **new-network stations** (first observed after 2022) | 5,522 | 23 | **4.22 %** | **6.16 %** |
| **UNSEEN** (frozen holdout, fold 0) | 19,763 | 26 | 1.03 % | 1.57 % |

The 23 new-network stations (1 first seen 2023, 13 in 2025, 9 in 2026) are **not** holdout
stations, but no model ever trains on them — under a naive "seen vs unseen" split they would
have been scored as *seen*, and their base rate (4.22 %) is 3× the trained bucket's, so they
are a **non-random sample**. Reported separately from now on.

## 4. Step 27 — monsoon validation (D1)

| season | rows | stations | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: |
| monsoon (JJAS) | 111,410 (55.5 %) | 123 | 7.93 % | 8.91 % | **9.79 %** |
| non-monsoon | 89,255 | 133 | 1.29 % | 1.56 % | **1.80 %** |

Re-verified against phase0 §6 (55.5 %, 9.79 % / 1.83 % — 5.4× contrast). Every split holds
> 1 000 monsoon and > 1 000 non-monsoon rows, so all three horizons can be reported per
season:

| split | monsoon | non-monsoon |
| --- | ---: | ---: |
| train | 44,146 | 33,425 |
| val | 12,498 | 9,422 |
| test | 54,766 | 46,408 |

## 5. How Phase 16 uses this

```python
tbl  = pd.read_csv("data/processed/experimental_dataset.csv")          # 67 cols
splt = pd.read_csv("data/processed/split_assignments.csv")             # 9 cols
d    = tbl.merge(splt, on=["station_id", "date"], validate="one_to_one")   # S9 proved the keys match
tr   = d[(d.split == "train") & ~d.station_holdout]                    # 61,991 rows
va   = d[(d.split == "val")   & ~d.station_holdout]                    # 17,538 rows
te   = d[d.split == "test"]                                            # 101,174 rows, FROZEN
```

Rules handed forward: **test never used for selection or early stopping** (the existing
`scripts/baseline_model.py` has exactly that flaw — stage 30 says fix it first); metrics per
year/season/bucket, never pooled (D8); persistence baseline alongside every model (D7).

## 6. Findings

| id | finding | action |
| --- | --- | --- |
| **F15-1** | all four cut dates sit in natural 88–131-day dry-season gaps; the 7-day embargo flags 0 rows | cuts are structurally safe; keep the explicit embargo rule in code as a guard |
| **F15-2** | base rate drifts 14× (16.81 % → 1.19 %, 2017 → 2023) and train/val/test differ 9.60 / 5.08 / 1.41 % (24 h) | rolling origin + per-year reporting is mandatory, not optional (R1/D8) |
| **F15-3** | 23 stations first appear *after* the split point; their test base rate is 3× the trained bucket's | third bucket in all reporting; never fold them into "seen" (R7) |
| **F15-4** | holdout fold 0 is balanced on both cohort and label count (17/9 early/late, pos ratio 1.20) | unseen-location claim is not confounded with era |
| **F15-5** | only 84 stations are ever trainable; spatial CV test folds hold 19–22 stations each | CIs over folds will be station-limited — bootstrap by station (Phase 20) |
| **F15-6** | monsoon carries 55.5 % of rows and 5.4× the base rate | headline metric must be reported per season (D1) |

**Open items handed on:** fix the early-stopping-on-test flaw before Phase 16 baselines
(stage 30); class-imbalance analysis (stage 29) is still PARTIAL and belongs with Phase 16;
imputation/encoding policy for LINEAR (F14-2) decided at model time.

---

**Next: Phase 16 — establish baseline models (body step 28 / checklist stage 30)**, with the
persistence one-feature baseline (D7) mandatory from the first number onward. Order per **D10**.
