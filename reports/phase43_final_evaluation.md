# Phase 43 — Final scientific evaluation (stage 71)

**Status:** COMPLETE · **Written:** 2026-11-01 · **Seed:** `20261008`

This document answers, with citations to the frozen reports, the five validity questions
required by the workflow (final workflow §0.3, Part D — Closure):

1. Data validity — labels, missingness, leakage
2. Model validity — temporal, spatial, monsoon, imbalance
3. Prediction validity — calibration, false negatives, 24/48/72 h
4. Explainability — sensible features, per-prediction explanation
5. System validity — location selection, map, ingestion, alert

All numbers are drawn from frozen artefacts; the test split remains closed
(`test_rows_read = 0`). Nothing here re-opens the test.

---

## 1. Data validity

### 1.1 Labels (phase0 §4, phase14, phase13)

The research question is a **probabilistic lead-time forecast**: P(flood at station × issue
day within the next 24/48/72 h), not a classifier or susceptibility map
(`reports/phase0_research_definition.md` §1–3).

- **Label contract:** `label_24h/48h/72h` = any observed reading in `(t, t+h]` ≥ FFWC danger
  level; missing stays NaN, never 0; `obs_days_*` columns banned as features
  (`final workflow.md` §0.3). Verified by `tests/test_label_contract.py` (32 tests, all pass).
- **Label drift explained, never pooled** (D5): base rate drifts 16.8% (2017) → 1.19% (2023)
  (`reports/phase15_split_design.md` §1). All results are per-cell, never pooled (D8).
- **Label ceiling:** MI-4 label audit bounds the achievable PR-AUC ceiling at ~0.96–0.97 on
  `val_2022`; a uniform 0.99 is not reachable (`reports/mi4_label_audit.md`,
  `reports/mi_status.md` §3).

### 1.2 Missingness (phase2, phase13, phase31)

- Missingness may be informative (D6). Water-level spike filter applied on read (D9).
- 2018 archive gap (R6) never interpolated. All cuts fall in dry-season gaps of 88–131 days
  with 0 rows within the 7-day embargo (`reports/phase15_split_design.md` §1).
- Median PR-AUC is flat across missingness buckets (0.966–1.000); minima are ≤5-positive
  noise in the hardest fold, not a collapse (`reports/phase31_robustness.md`).

### 1.3 Leakage (phase13, phase19)

- Phase 13 leakage audit: banned columns asserted absent; label-window features excluded;
  7-day embargo between train/val/test (`reports/phase13_leakage_audit.md`).
- Phase 19 sequence validation: S1–S7 all green; missing-timestep policy P19-M1 (keep-all +
  mask, 0 samples dropped); labels recomputed from raw with 0 mismatches
  (`reports/phase19_sequences.md`, `sequence_policy.json`).
- No banned/label-window column in any horizon's top-20 SHAP attribution
  (`reports/phase34_xai.md`).

---

## 2. Model validity

### 2.1 Temporal validity (P1, P2)

- **P1 temporal primary:** train <2022 / val 2022 / test 2023+ (frozen).
- D16 (per-horizon LightGBM/XGBoost on FULL_V2 110 features) val_2022 horizon-mean PR-AUC
  **0.957427** (24 h 0.970789 / 48 h 0.956485 / 72 h 0.945007)
  (`reports/FREEZE.md`, `reports/mi6_summary.json`).
- Year difficulty: val_2020 > val_2021 > val_2022 for all leaders
  (`reports/phase31_robustness.md`).
- Difficulty grows monotonically with lead time for every competitive model
  (`reports/phase26_multihorizon.md`).

### 2.2 Spatial validity (P3)

- **P3 spatial CV:** 5 station-disjoint folds, fold 0 = 26 stations held out
  (`reports/phase15_split_design.md` §3).
- Leaders beat D7 persistence in 63/63 cell×horizon points (min margin +0.0143)
  (`reports/phase31_robustness.md`).
- Best DL `fusion_L3` collapses to 0.7657 on fold-1 24 h vs leader min fold 0.9147 — a
  robustness gap the pooled headline hides (`reports/phase31_robustness.md`).
- Unseen-station seasonal gap reaches 0.2586 on fold-1 72 h (non-monsoon base rate
  0.0098–0.0155 vs 0.148–0.179); leaders still ≥ D7 in 126/126 seasonal points
  (`reports/phase31_robustness.md`).

### 2.3 Monsoon validity (D1)

- Monsoon = JJAS. Temporal monsoon/non-monsoon PR-AUC gap ≤ 0.0435
  (`reports/phase31_robustness.md`).
- Thresholds re-derived per period: carrying a dry-season threshold into monsoon loses up to
  16.6 points of recall for weaker models (`reports/phase29_thresholds.md`).

### 2.4 Class imbalance (phase16 §10, phase27)

- Imbalance ratios: 9.4:1 train → 18.6:1 val → **70:1** test at 24 h
  (`final workflow.md` §0.3).
- Resampling/class weighting inside training only; thresholds train-derived and
  period-specific.
- Accuracy banned as headline metric (Phase 16 §10); PR-AUC primary, ROC-AUC secondary.

### 2.5 ML vs DL (phase27, MI-7)

- Fair comparison on identical splits, labels, features-as-available, metric code, seeds,
  tuning budget (D14). One table, one code path (`reports/phase27_ml_vs_dl.md`).
- L2 gradient boosting leads every horizon on every protocol; best DL `fusion_L3` is
  0.016–0.024 behind on val_2022.
- MI-7 fair DL rematch (equal features + equal budget): DL does not beat ML
  (`MI7_fusion_v2_L3` 0.933991 vs `xgboost` 0.954634 vs D16 0.957427; DL–ML CIs exclude
  zero in ML's favour at all 3 horizons). DL queue closed (`reports/mi_status.md` §7).

---

## 3. Prediction validity

### 3.1 Calibration (phase28)

- Raw calibration good for most models (17/24 clear ECE ≤ 0.02 all-season; median raw ECE
  0.0067). Miscalibration is structural over-confidence, worst in monsoon and growing with
  lead time (`reports/phase28_calibration.md`).
- Platt/isotonic recalibration fitted on validation only, scored out-of-fold over 5 station
  folds, brings every skill model within tolerance (`reports/phase28_calibration.md`).
- D16 OOF ECE 0.0031 (`reports/phase33_selection.md`).

### 3.2 False negatives / operational (phase29, phase30, phase27)

- Cost-derived thresholds: C_FN:C_FP = 10:1, sweep 0.01–0.60, exact cost argmin
  (`reports/phase29_thresholds.md`).
- Leaders: lightgbm 24 h (t=0.060, recall 0.980, 18 missed), xgboost 48 h (t=0.052,
  recall 0.982, 19 missed), xgboost_selected 72 h (t=0.061, recall 0.966, 38 missed)
  (`reports/phase29_thresholds.md`).
- Error analysis: 10 named patterns. Top three:
  - **A** missed floods occur while river still below danger but rising (median
    `wl_dist_to_danger` +0.76 m; 0.0% of FN below danger vs 82.5% of TP)
  - **F** false alarms fire just below danger (median `wl_pct_danger` 0.983; 80.9% below
    danger — the price of FN ≫ FP)
  - **E** stations with fewest training floods missed ~14× more (miss rate 0.230 vs 0.016)
    (`reports/phase30_errors.md`).
- Frozen test: D16 test horizon-mean **0.770758** (24 h 0.836227 / 48 h 0.768959 / 72 h
  0.707087) vs D7 0.667549; D16 beats D7 on all 21 test points
  (`reports/FREEZE.md`, `reports/mi6_test_read.md`).
- Missed floods: 2,478 (D7) → 1,044 (D16, −58%); mean row recall 0.769 → 0.894; event
  recall 0.901 → 0.967 at comparable FAR (`reports/final workflow.md` §1, Phase 33 row).

### 3.3 Risk bands (phase29)

- Cost-derived bands: watch = ½·t_warn, warning = t_warn, high = precision ≥ 0.90.
- Top band: 90.0–90.1% observed flood rate, 13.7–17.7× lift (~90% of rows low, <0.2% base
  rate). Quartile default rejected (50–75% of rows in zero-flood bands)
  (`reports/phase29_thresholds.md`, `phase29_risk_bands.csv`).

---

## 4. Explainability (phase34)

- Global exact TreeSHAP of D16 on P1 val_2022 (17,502 rows @ 24 h, no new test read):
  - **G4_hydrology = 64.0%** of 24 h attribution (led by `wl_dist_to_danger` 1.005,
    `wl_pct_danger` 0.539); then G3_geography 8.4%, G5_network 6.1%, G1_weather 4.7%
    (`reports/phase34_xai.md`).
  - No banned/label-window column in any horizon's top-20.
  - Cross-model: D16 reproduces Phase-33 xgboost ordering on shared 62 features
    (Spearman ρ 0.76/0.86/0.88).
- Local explanations for TP, missed flood (pattern A), false alarm
  (`phase34_local_explanations.csv`).
- DL attribution (integrated gradients) on Transformer (L=7) and LSTM (L=3): 74.7%/70.6%
  on G4_hydrology — same dominant layer as tabular SHAP (`phase34_dl_attribution.csv`).

---

## 5. System validity

### 5.1 Deployed system (DEPLOYMENT.md, README.md)

- **Live app:** HF Static Space at
  https://riyadhuggy-aabhas-flood-outlook.static.hf.space — verified live 2026-11-01
  (`DEPLOYMENT.md` §1): Space `RUNNING`, `/`, `/index.html`, hashed assets,
  `/data/snapshot/health.json`, one date's predictions, station trend all HTTP 200.
  `health.json`: model_version D16, 2,611 feature dates, latest 2026-10-30, 141 stations.
- **Architecture:** static snapshot (precomputed forecasts for 2,611 dates through
  2026-10-30) + frozen D16 model; no live API container (HF free cpu-basic requires PRO for
  Docker Spaces). Live FastAPI available locally/Docker (`Dockerfile`, `docker-compose.yml`).
- **API endpoints:** `/health`, `/model/info`, `/thresholds`, `/dates`, `/predict`,
  `/predict/batch`, `/predict/all` (`DEPLOYMENT.md` §2.4).
- **CI/CD:** GitHub Actions runs API import checks, TypeScript typecheck, production build
  on every push; snapshot rebuild + publish is manual (`workflow_dispatch`)
  (`DEPLOYMENT.md` §3).

### 5.2 Location selection

- Prediction unit: one FFWC station × one issue day (133 stations, station point +
  district/upazila/river/basin attributes) (`reports/phase0_research_definition.md` §2).
- Location-selection UI: choropleth unblocked (ADM1/ADM2 boundaries in `data/raw/geo/admin/`)
  (`reports/phase0_research_definition.md` stage 15).

### 5.3 Ingestion

- FFWC station water levels, NASA POWER rainfall/soil, Open-Meteo ERA5-Land, GloFAS
  transboundary discharge, ESA WorldCover landcover (`reports/mi_status.md` §1, MI-1).
- Feature store: HF Dataset `riyadhuggy/aabhas-flood-features` (2,611 partitions,
  ~262 MB) (`README.md`).
- Automatic FFWC ingest resumable; weather acquisition not fully automated
  (`final workflow.md` stage 64 PARTIAL).

### 5.4 Alert engine

- Risk bands + thresholds drive the alert logic (`reports/phase29_thresholds.md`,
  `system/config/alert_config.json`).
- Early-warning engine is a prototype; production label mandatory before full rollout
  (`final workflow.md` stage 66 TODO).

---

## 6. Overall assessment

| Validity dimension | Verdict | Evidence |
|---|---|---|
| Data validity | PASS | Labels per contract; missingness NaN-never-0; leakage audited; embargo 0 rows |
| Model validity (temporal) | PASS | D16 val_2022 PR-AUC 0.9574; beats D7 on 21/21 dev points |
| Model validity (spatial) | PASS | Leaders ≥ D7 in 63/63 points; 126/126 seasonal |
| Model validity (monsoon) | PASS | Gap ≤ 0.0435 temporal; thresholds re-derived per period |
| Model validity (imbalance) | PASS | 70:1 test ratio handled; accuracy banned as headline |
| Prediction validity (calibration) | PASS | OOF ECE 0.0031; 17/24 models ECE ≤ 0.02 raw |
| Prediction validity (FN/operational) | PASS | Test PR-AUC 0.7708 vs D7 0.6675; FN −58%; recall 0.894 |
| Prediction validity (risk bands) | PASS | Top band 90% flood rate, 13.7–17.7× lift |
| Explainability | PASS | Hydrology-first SHAP; no leakage in top-20; local explanations |
| System validity | PARTIAL | Live static Space verified; API endpoints defined; alert engine prototype |

**Overall:** The project achieves its core goal — a leakage-aware, probabilistic,
station-level 24/48/72-hour flood outlook for Bangladesh, frozen as D16, deployed on
HF Static Space, and verified live. The test split remains closed. Remaining system work
(stages 62–70) is partially built (API, web UI, map, ingestion) but not fully documented
per workflow phases 35–42; stage 73 (research report/thesis) is pending.
