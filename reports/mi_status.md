# MI-v2 — live status

| | |
| --- | --- |
| **Date** | 2026-10-17 |
| **Programme** | Model Improvement Programme v2 (`reports/improvement_plan.md`) |
| **Seed** | `20261008` (`baseline_model.B.SEED`) — every stage |
| **Python** | `D:\Flood predictor\.venv\Scripts\python.exe`, cwd `D:\Flood predictor` |
| **Governing rules** | test (`>= 2023`) untouched until MI-6 · per-cell only, never pooled (D8) · PR-AUC primary, accuracy banned as headline · Phase-13 leakage rule · frozen model until MI-5 |

> **Correction (2026-10-22).** The MI-sprint seven-cell means and 21/21 claims in this file
> came from a single ≤2021 fit and are **in-sample for six of the seven cells**. They are
> withdrawn unless explicitly re-scored per cell in `reports/p2p3_correction.md`. The
> `val_2022` selection and D16 are unchanged. The frozen test was **not read once at
> programme level**: it was opened six times (five predictive reads, one invalid); see
> `reports/test_read_ledger.md`. The paper quotes the MI-6 D16 read.

**One-line state:** **MI-v2 complete** — MI-1 done (ERA5 **complete**, 133/133), MI-2 done,
MI-3 done (ensemble rejected), MI-4 done (labels bound the ceiling), MI-5 done (frozen
`hpo_perhorizon_FULL_V2`, D16), **MI-6 done — the frozen test read once** (D16 test
horizon-mean **0.77076** vs Phase-33 `xgboost` 0.75601 vs D7 0.66755); **MI-3a(ERA5) probe
done — ERA5 is a wash on the primary cell (−0.00025 val_2022), so D16 stands**.

**Post-v2 extensions (2026-10-17):** **MI-7 complete** — the fair DL rematch (equal features,
equal tuning budget) confirms **DL does not beat ML** (`MI7_fusion_v2_L3` 0.933991 vs `xgboost`
0.954634 vs D16 0.957427; paired DL−ML CIs exclude zero in ML's favour at all three horizons);
stacking is a wash → the **DL queue is closed**. **MI-8 complete -- the score push did NOT beat D16**: the 55-column `m8_*` block is a
**fixed-config win on `val_2022` (+0.00058)**; the archived **+0.00135 / 21-21**
seven-cell claim is **corrected** (in-sample) to **+0.00187 on 20/21** by per-cell refits
(`reports/p2p3_correction.md`). Under a 71-config/horizon HPO the arm lands at
**0.957388 vs D16 0.957427 (-0.00004)** on `val_2022`, and the equal-budget `FULL_V2`
control reaches **0.957785 (+0.00036)** -- **no paired station-bootstrap CI excludes zero
at any horizon**. On the corrected seven-cell mean the arm is **+0.00203 above D16**, but
the pre-registered `val_2022` rule keeps **D16; champion unchanged at 0.957427**. The
disagreement is reported, not hidden.

**MI-4 (FULL_V4) complete** -- the original D17 hybrid and its test read are **RETRACTED** (see `reports/mi4_test_read_decision.md` and `reports/mi9_test_read_decision.md`). MI-9 reproduced D16 bit-for-bit; D16 stands as the governed model. The FULL_V4 HPO remains a development result only.

---

## ⚠️ 2026-10-21 — **D17 SUSPENDED: the MI-4 test read is invalid**

While building Experiment 1 of the advanced programme, an audit of
`scripts/mi4_test_read.py` found that the D17 comparison is **not comparable
with MI-6**. Full report: `reports/d17_comparability_audit.md`.

| Defect | Consequence |
| --- | --- |
| Complete-case filter `~d[feats].isna().any(axis=1)` on the 187-feature FULL_V4 matrix | only **31.5 %** of test rows scored, and the retained subset has a **1.5–1.8× higher base rate** in the year cells (`test_2024`: 4.55 % vs 2.54 %; `test_unseen`: 1.80 % vs 1.03 %) → PR-AUC inflated mechanically |
| `test_new_network` mask = `year >= 2023 & ~holdout` (26,292 rows) instead of the Phase-15 `first_year > 2023` (4,805 rows) | the claimed new-network gain is not a comparison of the same cell |
| `mi4_select_v4.py` used a hand-written "D16" config, not the frozen D16 hyper-parameters | the "D16 refit" in the selection scorecard was a different model |
| `mi4_hpo_v4.py` **was** correct (imputes, all rows) | only the HPO objective is trustworthy — and it showed Δ ≤ 0.001 |

**Status changes**

* `models/final_v3/` is retained on disk but is **not** the champion.
* The champion reverts to **D16** (`models/final_v2/`, val_2022 0.957427).
* Test read #3 is pre-registered in `scripts/mi9_test_read_corrected.py`, which
  mirrors the MI-6 protocol exactly and **aborts** unless re-scoring the frozen
  D16 model reproduces `mi6_test_metrics.csv` to < 1e-9.

**Standing rule (added to the checklist):** any evaluation that drops rows must
report the fraction retained and the base rate of the retained subset, and must
never be compared against a result computed on all rows. Prefer the pipeline
imputer.

### Outcome — test read #3 (MI-9): **D17 RETRACTED, D16 champion (decision D18)**

`reports/mi9_test_read_decision.md`. The D16 sanity gate reproduced MI-6 exactly
(max \|Δ PR-AUC\| = 0 over 63 shared rows), so the comparison is provably on the
same protocol.

| Horizon | D16 | D17 hybrid | Δ |
| --- | ---: | ---: | ---: |
| 24 h | 0.836226 | 0.835923 | **−0.00030** |
| 48 h | 0.768958 | 0.768958 | 0.00000 |
| 72 h | 0.707088 | 0.707088 | 0.00000 |
| **horizon-mean** | **0.770757** | 0.770657 | **−0.00010** |

The claimed +0.065/+0.064/+0.078 does not exist. The only real V4 gains are
`test_2026` (+0.0175) and `test_unseen` (+0.0153); the 24 h bootstrap on
`test_2024` gives −0.0013 [−0.0061, +0.0031], P(Δ>0) = 0.283. **V4 is closed as
a null result; the champion is D16 (`models/final_v2/`).**

---

## 1. Stage ledger

| stage | state | command | exit | wall | artefacts | headline |
| --- | --- | --- | ---: | ---: | --- | --- |
| **MI-1a** ERA5-Land | **DONE** | `scripts/fetch_openmeteo_batch.py` | `0` | run2 ≈ 15 m 46 s; run4 ≈ 18 m | `data/raw/openmeteo/era5land_station_daily.csv` (669,389 rows, **133** stations, sha `b22d0a92…`) · `era5land_manifest.json` | 133/133 — block now **active** (`FULL_V2` 110 → **136**); tagged build `mi_v2_features_era5.csv` |
| **MI-1b** GloFAS | **DONE** | `scripts/fetch_glofas.py` | `0` | not captured | `data/raw/glofas/discharge_anchors_daily.csv` (55,363 rows, 11 anchors) · `chosen_cells.json` · `glofas_manifest.json` | 11 transboundary/in-country anchors, 2013-01-01 → 2026-10-12 |
| **MI-1c** POWER soil | **DONE** | `scripts/fetch_power_soil.py` | `0` | not captured | `data/raw/power_soil/power_soil_daily.csv` (669,389 rows, 133 stations) · `power_soil_manifest.json` | complete coverage (133/133) |
| **MI-2** v2 features | **DONE** | `scripts/build_mi_features.py` | `0` | deterministic (~s) | `data/processed/mi_v2_features.csv` (200,665 × 48 new) · `feature_sets_v2.json` (`FULL_V2=110`) · `reports/mi2_features.md` | 48 new cols; leakage audit **ADMIT** |
| **MI-3a** v1-vs-v2 compare | **DONE** | `scripts/mi3_compare.py` | `0` | 153 s | `reports/mi3_featureset_compare.csv` · `_summary.json` | val_2022 horizon-mean **0.95463 → 0.95589** |
| **MI-3b** HPO | **DONE** | `scripts/mi3_search.py` | `0` | 4,132 s | `reports/mi3_hpo.csv` (123 rows) · `models/mi/mi_hpo_{24h,48h,72h}.json` | best configs: `lgbm7`/`lgbm7`/`xgb3` |
| **MI-3c** ensemble | **DONE (negative)** | `scripts/mi3_ensemble.py` | `0` | 675 s | `reports/mi3_ensemble.csv` (21 rows) | **ensemble rejected** — best single wins on val_2022 |
| **MI-4** label audit | **DONE** | `scripts/mi4_label_audit.py` | `0` | 5.8 s | `reports/mi4_label_audit.md` · `mi4_base_rates.csv` · `mi4_label_audit.json` | 0.99 unreachable (base-rate + noise); ceiling ≈0.96–0.97 on val_2022 |
| **MI-5** re-select & freeze | **DONE** | `scripts/mi5_selection.py` | `0` | 256 s | `reports/mi5_selection.md` · `mi5_scorecard.csv` · `mi5_selection.json` · `models/final_v2/` | selected **`hpo_perhorizon_FULL_V2`** (lgbm7/lgbm7/xgb3), val_2022 horizon-mean **0.95743** (+0.0028 vs Phase 33); decision **D16** |
| **MI-6** single test read | **DONE** | `scripts/mi6_test_read.py` | `0` | 119 s | `reports/mi6_test_read.md` · `mi6_test_metrics.csv` · `mi6_test_read_log.json` · `mi6_thresholds.csv` · `mi6_summary.json` | frozen test read **once**; D16 test horizon-mean **0.77076** vs P33 0.75601 vs D7 0.66755; D7 cross-check exact (5.6e-17) |
| **MI-3a(ERA5)** probe | **DONE — negative/marginal** | `scripts/mi3_era5_probe.py` | `0` | 169 s | `reports/mi3_era5_probe.md` · `mi3_era5_probe.csv` · `mi3_era5_probe.json` | ERA5 = **wash on `val_2022`** (−0.00025; −0.0012 at 24 h), archived +0.00075 7-dev claim **withdrawn** (in-sample, not re-scored) -> **D16 stands**
| **MI-v2** notebook | **DONE — 14/14** | `scripts/build_mi_notebook.py` → `jupyter nbconvert --execute` | `0` | 15.9 s | `notebooks/mi_v2.ipynb` (19 cells, executed) | self-verifying: re-derives all MI headline facts independently (NM-01…NM-14) |
| **MI-7** fair DL rematch | **DONE — negative** | `scripts/mi7_dl_rematch.py` · `mi7_mlpreds.py` · `mi7_evaluate.py` · `mi7_stack.py` | `0` | multi-run | `project study/MI7_Fair_DL_Rematch.md` · `reports/mi7_report.md` · `mi7_ladder.csv` · `mi7_metrics.csv` · `mi7_scorecard.csv` · `mi7_paired_deltas.csv` · `mi7_stack.csv` · `mi7_summary.json` | fair rig (equal features + 288-config space, 5 anchors, K=3 seeds): best DL `MI7_fusion_v2_L3` **0.933991** vs ML `xgboost` 0.954634 vs D16 0.957427 → **DL does not beat ML** (`mi7 − ML = −0.0206`, `mi7 − D16 = −0.0234`); stacking a wash → **DL queue closed** |
| **MI-8** score push | **DONE — no champion change** | `scripts/mi8_features.py` · `mi8_probe.py` · `mi8_search.py --arm full_v3`/`full_v2` · `mi8_select.py` | `0` | probe 169 s · search 11 685 s (v3) + 3 587 s resume (v2) · select 298 s | `reports/mi8_selection.md` · `mi8_probe.{csv,json}` · `mi8_hpo_full_{v2,v3}.csv` · `mi8_scorecard.csv` · `mi8_paired_deltas.csv` · `mi8_summary.json` | `FULL_V3` = 165 (+55 `m8_*`): fixed-config **+0.00058** val_2022; archived 7-dev **+0.00135/21-21** is in-sample, corrected to **+0.00187/20-21**; 71-cfg HPO → **0.957388 vs D16 0.957427 (−0.00004)**; control `FULL_V2` 0.957785 (+0.00036); **all paired CIs straddle 0** → **D16 stands** |
| **MI-4 (FULL_V4)** extended HPO | **DONE** | `scripts/mi4_hpo_v4.py` · `scripts/mi4_select_v4.py` · `scripts/mi4_test_read.py` | `0` | search ~12 ks · select ~5 min · test ~1 min | `reports/mi4_hpo_v4.csv` (213 rows) · `models/mi/mi4_hpo_v4_<horizon>.json` · `reports/mi4_test_read_log.json` · `reports/mi4_test_metrics.csv` · `models/final_v3/` | FULL_V4 = 187 (FULL_V2 110 + WorldCover 22 + MI-8 55); 71 cfg/horizon; **Decision D17: Hybrid adopted** — V4 lgbm13 (FULL_V4) 24h, D16 lgbm7/xgb3 (FULL_V2) 48/72h; test horizon-mean **0.7983** vs D16 0.7337 vs D7 0.6427; unseen-station +0.13…+0.20 — **RETRACTED, see MI-9** |

| **MI-9** corrected test read | **DONE — D17 RETRACTED (D18)** | `scripts/mi9_test_read_corrected.py` | `0` | ~170 s | `reports/mi9_test_read_decision.md` · `reports/d17_comparability_audit.md` · `mi9_test_metrics.csv` · `mi9_headline.csv` · `mi9_test_thresholds.csv` · `mi9_sanity.json` · `mi9_summary.json` · `mi9_test_read_log.json` | MI-6 protocol (all rows, imputed, Phase-15 cells); **D16 reproduced exactly** (max \|ΔPR-AUC\| = 0, 63 rows); **D16 0.770757 vs hybrid 0.770657 (−0.00010)**; 24 h per-cell wash; **champion = D16** |
| **MI-10** P2/P3 correction | **DONE - 18/18 gates** | `scripts/mi10_p2p3_rescore.py` | `0` | 2,216 s | `reports/p2p3_rescore/summary.json` + `probe_p2p3.csv` + `arms_p2p3.csv` + `d7_p2p3.csv` + `d7_margins_p2p3.csv`; `reports/p2p3_correction.md` | per-cell refits: FULL 0.964051, FULL_V2 0.965930, FULL_V3 0.967796; V3-V2 +0.001867 (20/21); D16 arm 0.966532; **D16 stands** (val_2022 0.957427) |
| **MI-11** event-level evaluation | **DONE - dev only** | `scripts/mi11_event_eval.py` | `0` | 322 s | `reports/event_level/summary.{md,json}` + `event_metrics.csv` | 845 events at 96 stations; strict pre-onset recall D16 0.509/0.724/0.881 vs D7 0.452/0.649/0.767; FA/station-year D16 0.58/0.98/2.30 vs D7 1.56/1.97/2.44; `test_rows_read = 0` |

Run logs: `scripts/mi1_run.log` / `mi1_run2.log` / `mi1_run3.log` (MI-1a), `mi2_run.log`,
`mi3_run.log`, `mi3_hpo_run.log`, `mi3_ensemble_run.log`, `mi5_run.log`, `mi6_run.log`.
(MI-1b/1c were driven before the log-retention habit; their provenance lives in the two
manifests.)

---

## 2. Headline numbers

### MI-2 — what the v2 table added

`FULL_V2 = 62 (v1) + 48 = 110` features: POWER soil wetness (`ps_`, 19), GloFAS discharge
(`gf_`, 18), derived water-level memory + API (`wl_last_lag*`, `api_*`, 6), interactions
(`ix_`, 5), ERA5-Land (`e5_`, **0 — this is the frozen table**; ERA5 was landed later as a
separate tagged variant, `FULL_V2 = 136`, evaluated in MI-3a(ERA5)). Worst null rate
`wl_last_lag30 = 1.99 %`.
All 48 columns cleared the Phase-13 audit (backward shifts only; climatologies computed
strictly before 2022-01-01).

### MI-3a — does v2 help at all? (frozen xgboost, FULL vs FULL_V2)

| horizon | FULL mean | FULL_V2 mean | Δ |
| --- | ---: | ---: | ---: |
| 24 h | 0.98911 | 0.99118 | **+0.00207** |
| 48 h | 0.98392 | 0.98699 | **+0.00307** |
| 72 h | 0.97719 | 0.98237 | **+0.00518** |
| **val_2022 horizon-mean** | **0.95463** | **0.95589** | **+0.00126** |

Every cell improves; gains grow with lead time. Identical to the earlier cached run
(reproducible).

### MI-3b — HPO (41 configs/horizon, budget-trimmed)

| horizon | best config | val_2022 | frozen val_2022 | Δ | mean PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 h | `lgbm7` | **0.9708** | 0.9695 | +0.0013 | 0.9943 |
| 48 h | `lgbm7` | **0.9565** | 0.9549 | +0.0016 | 0.9912 |
| 72 h | `xgb3` | **0.9450** | 0.9432 | +0.0018 | 0.9801 |

**Horizon-mean of the best singles: 0.9574** (vs Phase-33 frozen 0.95463 → **+0.0028**).
Gains are small but consistent and grow with horizon. Pool: 12 xgboost + 12 lightgbm +
8 hist-gradient-boosting + 8 extra-trees + 1 frozen.

### MI-3c — ensemble vs best single (val_2022)

| horizon | best single | ENS_mean | ENS_rank |
| --- | ---: | ---: | ---: |
| 24 h | **0.9708** | 0.9654 | 0.9635 |
| 48 h | **0.9565** | 0.9495 | 0.9525 |
| 72 h | **0.9450** | 0.9374 | 0.9401 |

**Verdict: the ensemble is rejected.** On the primary cell it loses to the best single
config at every horizon (−0.004 to −0.008). Its `dev_mean` is marginally higher
(0.9950 vs 0.9943 at 24 h) only because the near-saturated spatial folds (0.998–1.000) and
easy validation years dominate that average — a good illustration of why `dev_mean` is a
poor selector and **val_2022 remains the decision cell**. The RF/ET members drag the
average down on the hard cell. **Decision: ship the best single HPO config per horizon**
(`lgbm7`, `lgbm7`, `xgb3`) as the MI-5 candidates.

### MI-5 — re-selection & freeze (v2, validation-only)

The pre-registered Phase-33 rule (D16) re-applied to the MI-3b winners on `FULL_V2`, scored
on the 7 dev cells (`val_2020/21/22` + `spatial_fold_1..4`), test never read:

| rank | model | val_2022 24/48/72 h | **horizon-mean** | archived floor* | missed@10:1 | worst ECE |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| **1** | **`hpo_perhorizon_FULL_V2`** | 0.9708 / 0.9565 / 0.9450 | **0.95743** | +0.0225 | 90 | 0.0023 |
| 2 | `lgbm7_FULL_V2` | 0.9708 / 0.9565 / 0.9429 | 0.95671 | +0.0225 | **84** | 0.0025 |
| 3 | `xgboost_FULL_V2` | 0.9695 / 0.9549 / 0.9432 | 0.95589 | +0.0210 | 100 | 0.0027 |
| 4 | `xgb3_FULL_V2` | 0.9675 / 0.9537 / 0.9450 | 0.95540 | +0.0204 | 104 | 0.0019 |
| 5 | `xgboost_FULL` *(Phase-33)* | 0.9681 / 0.9549 / 0.9408 | 0.95463 | +0.0200 | 91 | 0.0025 |

*Archived floor is in-sample. The corrected D16 floor is **+0.0156 (21/21)**; see
`reports/p2p3_correction.md`. All five clear Gate 1 on the archived table, and Gate 2 (ECE ≪ 0.02). **Selected:
`hpo_perhorizon_FULL_V2`** = lightgbm `lgbm7` @ 24 h & 48 h + xgboost `xgb3` @ 72 h on 110
`FULL_V2` features → **+0.00280** over the Phase-33 winner. Bootstrap CIs overlap at every
horizon, so this is a consistent point shift, **not a CI-separable jump**. Frozen to
`models/final_v2/` as **decision D16**; details in `reports/mi5_selection.md`.

### MI-6 — the single frozen-test read (D16)

Test read **once** (guarded, `n_reads = 1`); D7 reproduces the Phase-16 read exactly
(max|Δ| = 5.6e-17). All-season PR-AUC, mean over the 7 test cells (horizon-mean):

| model | test horizon-mean | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: |
| **`hpo_perhorizon_FULL_V2` (D16)** | **0.77076** | **0.83623** | **0.76896** | 0.70709 |
| Phase-33 `xgboost` (v1) | 0.75601 | 0.80624 | 0.75255 | **0.70923** |
| D7 persistence | 0.66755 | 0.73396 | 0.66252 | 0.60617 |

D16 beats D7 on **all 21** test points and beats the Phase-33 model on **6 of 7** cells (loses
`test_2025` by −0.0037; at 72 h the two are a wash). So the MI-v2 gain **does carry to the
test** — and is larger there (+0.0148 vs +0.0028 on val_2022) — but it is concentrated at
24–48 h and the absolute test numbers stay well below validation. Full detail:
`reports/mi6_test_read.md`.

### MI-3a(ERA5) — probing the landed ERA5 block (dev cells only)

Fixed Phase-33 model, same rows, three feature sets (`reports/mi3_era5_probe.md`):

| set | cols | `val_2022` horizon-mean | 7-dev horizon-mean (archived*) |
| --- | ---: | ---: | ---: |
| `FULL` (v1) | 62 | 0.95463 | 0.98341 |
| `FULL_V2_pre` | 110 | **0.95589** | 0.98685 |
| `FULL_V2_era5` | 136 | 0.95564 | **0.98760** |
| Delta (ERA5) | +26 | **-0.00025** | +0.00075* |

ERA5 is a **wash on the primary cell** (-0.00025; -0.0012 at 24 h). The archived
seven-cell gain (+0.00075, mostly 72 h) is **in-sample and withdrawn**; it was not
re-scored. *Archived column, in-sample. The probe reproduces MI-3a exactly (0.95463 / 0.95589),
so the measurement is sound. **Conclusion: ERA5 not adopted; the D16 freeze stands.**

---

## 3. Honest read against the plan's target

Plan expectation: val_2022 horizon-mean **0.95 → 0.97–0.98**. Actual:
**0.95463 → 0.95589 (v2 features) → 0.95743 (MI-5 frozen per-horizon model)** = **+0.0028**.

- The signal is real (every cell moved the right way) but **much smaller than hoped**.
- **It carries to the frozen test**: D16 test horizon-mean **0.77076** vs Phase-33 `xgboost`
  **0.75601** (+0.01475) and D7 0.66755 — larger than the validation gain, but concentrated at
  24–48 h (72 h is a wash) and still far from 0.99.
- **ERA5-Land is now fully landed (133/133) and evaluated — and it is not the lever.** On the
  primary `val_2022` cell the block is a wash (-0.00025 horizon-mean; -0.0012 at 24 h); the archived seven-cell gain (+0.00075) is **in-sample and withdrawn** (not re-scored). D16 stands; no HPO re-run was
  justified (`reports/mi3_era5_probe.md`).
- A uniform 0.99 PR-AUC at the 0.7 %-base-rate test years **is not reachable** and is not
  claimed (base-rate ↔ PR-AUC correlation +0.87; MI-4 ceiling).

---

## 4. Deviations recorded

1. **Optuna / catboost unavailable** (must not be installed) → MI-3b samples with sklearn
   `ParameterSampler`; budget trimmed to **41 configs/horizon** (`N_TRIALS=24`,
   `N_TRIALS_SK=8`). The MI-3b pool is therefore sklearn HGB/ET rather than catboost.
2. **ERA5-Land now COMPLETE** — 133/133 stations, 669,389 rows (sha `b22d0a92…`); the last
   10 landed in run4 after the Open-Meteo 429 window cleared (`wait_for_api` recovered at
   1045 s). The builder's `era5_block()` gate is **active**, adding 24 `e5_*` + 2 `ix_*`
   (`FULL_V2` 110 → **136**). Built **non-destructively** as `mi_v2_features_era5.csv` /
   `feature_sets_v2_era5.json` (`build_mi_features.py --tag era5`), leaving the frozen v2
   table and all MI-5/MI-6 artefacts intact.
3. **MI-3c ensemble is a negative result** — reported as-is, not buried.

---

## 5. Still to do

**The MI-v2 programme is complete** (MI-1 → MI-6). Outstanding optional follow-ups:

- **MI-1a ERA5-Land is DONE and evaluated** (133/133): the MI-3a(ERA5) probe found it a
  **wash on the primary `val_2022` cell** (-0.00025); the archived seven-cell gain is **withdrawn** (in-sample, not re-scored) → **not adopted**; the D16 freeze stands (`reports/mi3_era5_probe.md`).
- **No further frozen-test reads** without a new pre-registered decision (the authoritative ledger records **six opens**: five predictive reads, one of them
  invalidated and retracted, plus one label-statistics read; see
  `reports/test_read_ledger.md`). Since ERA5 did not change the model, none is due.
- MI-4 summary (for the record): labels are nested, `label_24h` = next-day exceedance,
  4–10 % of positives hinge on a ≤2 cm crossing; the val_2022 ceiling is 0.96–0.97 and a
  uniform 0.99 is not reachable or claimable.
- **MI-7 closed the DL queue** — the fair rematch left DL ~0.021 behind ML on equal features
  and an equal tuning budget (`project study/MI7_Fair_DL_Rematch.md`). No further deep-learning
  work is justified.
- **MI-8 exhausted the feature and budget levers on `val_2022`** — D16 sits on top of the
  71-config plateau (0 of 71 configs beat it at 24 h and 72 h; 4 at 48 h, by ≤0.0003) and the
  cell is bounded above by label noise (`reports/mi8_selection.md`).
- **Open, cheap, and recommended:** the §7 leakage audit of the 55 `m8_*` columns (causal
  windows only; no feature computed over rows that include a dev cell) before the corrected seven-cell gain is trusted. The corrected arm gain is **+0.00203**
  (7-cell mean) against -0.00004 on `val_2022`; the audit found no feature leakage, only
  in-sample evaluation (`reports/p2p3_correction.md`).
- **A real score move now requires a pre-registered additional test read** — the seven dev
  cells have no headroom left in this feature family and model class. Owner decision.

---

## 6. Reproduce

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\build_mi_features.py      # MI-2 -> mi_v2_features.csv, feature_sets_v2.json
.venv\Scripts\python.exe -u scripts\mi3_compare.py            # MI-3a -> mi3_featureset_compare.csv / _summary.json
.venv\Scripts\python.exe -u scripts\mi3_search.py             # MI-3b -> mi3_hpo.csv, models/mi/mi_hpo_*.json
.venv\Scripts\python.exe -u scripts\mi3_ensemble.py           # MI-3c -> mi3_ensemble.csv
.venv\Scripts\python.exe -u scripts\mi4_label_audit.py        # MI-4 -> mi4_base_rates.csv, mi4_label_audit.json
.venv\Scripts\python.exe -u scripts\mi5_selection.py          # MI-5 -> mi5_scorecard.csv, mi5_selection.json, models/final_v2/
.venv\Scripts\python.exe -u scripts\mi6_test_read.py          # MI-6 -> mi6_test_metrics.csv, mi6_test_read_log.json (the ONE test read)
.venv\Scripts\python.exe -u scripts\build_mi_features.py --tag era5  # tagged ERA5 table -> mi_v2_features_era5.csv, feature_sets_v2_era5.json (needs >=130 stations)
.venv\Scripts\python.exe -u scripts\mi3_era5_probe.py         # MI-3a(ERA5) -> mi3_era5_probe.csv / .json (dev cells only, no test)
.venv\Scripts\python.exe scripts\build_mi_notebook.py        # build notebooks/mi_v2.ipynb (14 self-verifying checks)
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\mi_v2.ipynb --ExecutePreprocessor.kernel_name=flood-predictor
```

The verification notebook `notebooks/mi_v2.ipynb` (`scripts/build_mi_notebook.py`, executed with the
`flood-predictor` kernel) re-derives every MI headline fact independently from the artefacts —
**14/14 checks pass** (the MI equivalent of the `notebooks/NN_*.ipynb` phase notebooks).

The MI-2 table/`feature_sets_v2.json` above are the **frozen, pre-ERA5** artefacts (`FULL_V2 = 110`);
the ERA5 variant is emitted separately under the `--tag era5` suffix and is **not** the frozen model.

All deterministic given seed `20261008` (MI-2 has no randomness). Every stage MI-1 → MI-5 and the
MI-3a(ERA5) probe keep the test untouched (`test_rows_read = 0`); the test is read exactly once,
in MI-6 (`n_reads = 1`). MI-7 and MI-8 also keep the test untouched (`test_rows_read = 0`).

---

## 7. Post-v2 extensions — MI-7 (fair DL rematch) & MI-8 (score push)

Both were run on the **seven dev cells only**; the frozen test stayed sealed (`test_rows_read = 0`).

### MI-7 — the fair DL rematch (negative, DL queue closed)

Hosted in `project study/MI7_Fair_DL_Rematch.md`. The earlier DL-vs-ML comparison was invalidated
by a **feature asymmetry** (ML saw `FULL`/`FULL_V2`, DL saw a reduced set), so MI-7 rebuilt the
comparison on one pre-registered rig (`SPACE_SIZE = 288`, 5 anchors, K = 3 seed averaging,
500-resample station bootstrap, plus a `mi7_mlpreds` reproduction diff of `0.000000`).

| model | val_2022 horizon-mean |
| --- | ---: |
| D16 (champion, ML) | **0.957427** |
| `xgboost` (ML reference) | 0.954634 |
| best fair DL `MI7_fusion_v2_L3` | **0.933991** |

`mi7 − ML = −0.020643`, `mi7 − D16 = −0.023435`, `mi7_beats_ml = false`; the paired DL−ML CIs
exclude zero **negative** at all three horizons (except 72 h vs lightgbm, which straddles by
0.0007). DL is **well calibrated** (ECE 0.00158–0.00259) but **ranks worse** with 1.3–1.9× wider
CIs and **more false alarms at every horizon**. Stacking ML+DL is a wash. Scope honesty: only
`fusion_v2_L3` reached 20/20 runs; `lstm_v2_L7` was killed at 2/20; six families were never run.

### MI-8 — the score push (no champion change)

Hosted in `reports/mi8_selection.md`. Two levers, each measured separately:

| lever | evidence |
| --- | --- |
| **Features** — 55 `m8_*` columns (`FULL_V3` = 165) | **fixed config:** +0.00058 `val_2022`; corrected 7-cell **+0.00187**, **20/21** (archived +0.00135/21-21 is in-sample); reproduction `diff = 0.0` |
| **Budget** — 71 configs/horizon (vs MI-3b's 41), run on *both* feature sets | 11 685 s (v3) + 3 587 s resumed (v2) |

| arm | 24 h | 48 h | 72 h | horizon-mean | vs D16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D16 frozen | 0.970789 | 0.956485 | 0.945007 | **0.957427** | — |
| `full_v3` (new feats) | 0.970662 | 0.956765 | 0.944737 | 0.957388 | **−0.00004** |
| `full_v2` (control) | 0.971983 | 0.956365 | 0.945007 | 0.957785 | +0.00036 |

`mi8_select.py` refits every winner, reproduces D16 exactly (Δ ≤ 4.7e-07 at all horizons),
and reports a **paired station bootstrap**: **every contrast straddles zero at every horizon**.
A free check: `full_v2` @ 72 h selects the identical spec to D16 and is bit-identical to it
(delta `0.0`, sd `0.0`).

**Two recordings that matter more than the score:**

1. **The 7-cell mean disagrees with the decision cell -- and the disagreement survives
   correction.** On the corrected seven-cell mean, `full_v3` is **+0.00203** above D16,
   while on `val_2022` it is **-0.00004** below. The pre-registered cell governs, so D16
   stands; both numbers are reported.
2. **The saturation pattern is resolved.** The six near-1.0 cells were **in-sample**: the
   probe fitted one model on <=2021 and scored six cells that are subsets of that fit.
   The per-cell refits remove the saturation; the feature effect survives in 20/21 points.
   See `reports/p2p3_correction.md`.


### MI-11 - event-level evaluation (development only)

845 events at 96 stations on the non-test timeline. The event catalog bridges only exact
one-day gaps. Recall uses only events with at least one observable issue day in the lead
window (`n_eligible`); events with no prior row are excluded as unverifiable and reported
separately, not counted as misses. Thresholds are cost-optimal (C_FN=10, C_FP=1) on each
cell's training rows, matching MI-6/Phase 29.

| horizon | D16 strict pre-onset recall | D7 recall | D16 FA/station-year | D7 FA/station-year |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 0.509 | 0.452 | 0.58 | 1.56 |
| 48 h | 0.724 | 0.649 | 0.98 | 1.97 |
| 72 h | 0.881 | 0.767 | 2.30 | 2.44 |

At 24 h D16 loses strict recall on two cells (`val_2020`, `spatial_fold_1`), so the result
is a lead-time and false-alarm advantage, not uniform superiority. The test event numbers
(MI-6 definition: warning anywhere in the lead window or event span) are higher and more
permissive; they are not directly comparable to this strict pre-onset metric.

### Reproduce (post-v2 additions)

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi8_features.py                 # -> data/processed/mi8_features.csv (FULL_V3 = 165)
.venv\Scripts\python.exe -u scripts\mi8_probe.py                    # -> reports/mi8_probe.{csv,json}    (~169 s)
.venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v3     # -> reports/mi8_hpo_full_v3.csv     (~11 685 s)
.venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v2     # -> reports/mi8_hpo_full_v2.csv     (resumable)
.venv\Scripts\python.exe -u scripts\mi8_select.py   --arm full_v3   # -> mi8_scorecard.csv, mi8_paired_deltas.csv, mi8_summary.json (~298 s)
```

MI-7's rig and evaluation commands are listed in `project study/MI7_Fair_DL_Rematch.md` §8.
`mi8_search.py` is resumable (skips `(horizon, tag)` pairs already in the partial ledger).
