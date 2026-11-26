# Phase 43 — Final scientific evaluation (stage 71)

**Status:** COMPLETE · **Written:** 2026-11-01 · **Revised:** 2026-11-10 · **Seed:** `20261008`

This document answers, with citations to the frozen reports, the five validity questions
required by the workflow (final workflow §0.3, Part D — Closure):

1. Data validity — labels, missingness, leakage
2. Model validity — temporal, spatial, monsoon, imbalance
3. Prediction validity — calibration, false negatives, 24/48/72 h
4. Explainability — sensible features, per-prediction explanation
5. System validity — location selection, map, ingestion, alert

All numbers are drawn from frozen artefacts. The frozen test is **closed**; this document
performs no new test read. The programme-level read history is recorded in
`reports/test_read_ledger.md`: **six opens — five predictive reads (one invalid and retracted)
plus one label-statistics read**. The reported D16 headline is the valid **MI-6** read.

---

## 1. Data validity

### 1.1 Labels (phase0 §4, phase14, phase13)

The research question is a **probabilistic lead-time forecast**: P(flood at station × issue
day within the next 24/48/72 h), not a classifier or susceptibility map
(`reports/phase0_research_definition.md` §1–3).

- **Label contract:** `label_24h/48h/72h` = any observed reading in `(t, t+h]` ≥ FFWC danger
  level; missing stays NaN, never 0; `obs_days_*` columns banned as features
  (`final workflow.md` §0.3). Verified by `tests/test_label_contract.py` and the full suite
  (`tests/run_tests.py`, **89 tests, all pass**).
- **Label drift explained, never pooled** (D5): base rate drifts 16.8 % (2017) → 1.19 % (2023)
  (`reports/phase15_split_design.md` §1). All results are per-cell, never pooled (D8).
- **Label ceiling:** the MI-4 label audit bounds the achievable PR-AUC ceiling at ~0.96–0.97 on
  `val_2022`; a uniform 0.99 is not reachable (`reports/mi4_label_audit.md`,
  `reports/mi_status.md` §3).

### 1.2 Missingness (phase2, phase13, phase31)

- Missingness may be informative (D6). Water-level spike filter applied on read (D9).
- 2018 archive gap (R6) never interpolated. All cuts fall in dry-season gaps of 88–131 days
  with 0 rows within the 7-day embargo (`reports/phase15_split_design.md` §1).
- Median PR-AUC is flat across missingness buckets (0.966–1.000); minima are ≤5-positive
  noise in the hardest fold, not a collapse (`reports/phase31_robustness.md`).
- The October–November 2026 silent-null incident is documented in
  `reports/phase45_thesis.md` §7. It is now guarded by the incremental source refreshers and the
  mandatory feature-quality gate; it did not change the frozen model or the frozen test result.

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
- D16 (per-horizon LightGBM/LightGBM/XGBoost on FULL_V2 110 features) val_2022 horizon-mean
  PR-AUC **0.957427** (24 h 0.970789 / 48 h 0.956485 / 72 h 0.945007)
  (`reports/FREEZE.md`, `reports/mi6_summary.json`).
- Year difficulty: val_2020 > val_2021 > val_2022 for all leaders
  (`reports/phase31_robustness.md`).
- Difficulty grows monotonically with lead time for every competitive model
  (`reports/phase26_multihorizon.md`).

### 2.2 Spatial validity (P3)

- **P3 spatial CV:** 5 station-disjoint folds, fold 0 = 26 stations held out
  (`reports/phase15_split_design.md` §3).
- Leaders beat D7 persistence in 63/63 cell × horizon points (min margin +0.0143)
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

### 2.5 ML vs DL (phase27, MI-7, MI-8)

- Fair comparison on identical splits, labels, features-as-available, metric code, seeds,
  tuning budget (D14). One table, one code path (`reports/phase27_ml_vs_dl.md`).
- L2 gradient boosting leads every horizon on every protocol; best DL `fusion_L3` is
  0.016–0.024 behind on val_2022.
- MI-7 fair DL rematch (equal features + equal budget): DL does not beat ML
  (`MI7_fusion_v2_L3` 0.933991 vs `xgboost` 0.954634 vs D16 0.957427; DL–ML CIs exclude
  zero in ML's favour at all 3 horizons). DL queue closed (`reports/mi_status.md` §7).
- MI-8 score push: the new feature block won on the 7-cell mean (+0.00164) but not on the
  pre-registered `val_2022` decision cell, and no paired station-bootstrap CI excluded zero at
  any horizon. **D16 stands** (`reports/mi8_selection.md`).

### 2.6 Post-freeze audits (MI-9 … MI-12)

- **MI-9 / D18:** the MI-4 test read was invalid (complete-case filter retained 31.5 % of test
  rows with a 1.5–1.8× higher base rate; wrong new-network mask; non-frozen "D16" config).
  MI-9 reproduced MI-6 **bit-for-bit** (max |Δ PR-AUC| = 0, 63 shared rows). The D17 hybrid was
  **−0.00010** behind D16 (0.770657 vs 0.770757). **D17 retracted; D16 champion.**
  `models/final_v3/` is retained for traceability only.
- **MI-10:** the P2/P3 seven-cell record was corrected. FULL **0.964051**, FULL_V2 **0.965930**,
  FULL_V3 **0.967796**; V3−V2 **+0.001867** on 20/21 cells. D16 still stands on the
  pre-registered decision cell (`reports/p2p3_correction.md`).
- **MI-11:** event-level evaluation is **development-only** (845 events, 96 stations). D16
  strict pre-onset event recall **0.509 / 0.724 / 0.881** vs D7 **0.452 / 0.649 / 0.767**.
  The frozen test was not touched (`reports/event_level/summary.md`).
- **MI-12:** monotonic physical priors failed the pre-registered criteria (strict arm −0.0067,
  broad arm −0.0070 mean `val_2022` PR-AUC). The partial-dependence audit is useful: D16 is
  physically consistent on 9/9 strong features at 72 h but inverts 15 secondary features.
  **D16 stands** (`reports/mi12_monotonic.md`).

---

## 3. Prediction validity

### 3.1 Calibration (phase28)

- Raw calibration good for most models (17/24 clear ECE ≤ 0.02 all-season; median raw ECE
  0.0067). Miscalibration is structural over-confidence, worst in monsoon and growing with
  lead time (`reports/phase28_calibration.md`).
- Platt/isotonic recalibration fitted on validation only, scored out-of-fold over 5 station
  folds, brings every skill model within tolerance (`reports/phase28_calibration.md`).
- D16 OOF ECE 0.0031 (`reports/phase33_selection.md`).

### 3.2 False negatives / operational (phase29, phase30, MI-6)

- Cost-derived thresholds: C_FN:C_FP = 10:1, sweep 0.01–0.60, exact cost argmin
  (`reports/phase29_thresholds.md`).
- Phase-29 validation leaders: lightgbm 24 h (t = 0.060, recall 0.980, 18 missed),
  xgboost 48 h (t = 0.052, recall 0.982, 19 missed), xgboost_selected 72 h (t = 0.061,
  recall 0.966, 38 missed) (`reports/phase29_thresholds.md`).
- Error analysis: 10 named patterns. Top three:
  - **A** missed floods occur while river still below danger but rising (median
    `wl_dist_to_danger` +0.76 m; 0.0 % of FN below danger vs 82.5 % of TP)
  - **F** false alarms fire just below danger (median `wl_pct_danger` 0.983; 80.9 % below
    danger — the price of FN ≫ FP)
  - **E** stations with fewest training floods missed ~14× more (miss rate 0.230 vs 0.016)
    (`reports/phase30_errors.md`).
- **Frozen test (the valid MI-6 read):** D16 test horizon-mean **0.770758** (24 h 0.836227 /
  48 h 0.768959 / 72 h 0.707087) vs D7 **0.667549**. D16 beats D7 on **21/21** cell × horizon
  points (`reports/FREEZE.md`, `reports/mi6_test_read.md`).
- **Operational behaviour at the chosen thresholds, pooled over the 7 test cells:**

  | horizon | recall | precision | FAR | missed flood rows (D16 / D7) | TP / FP / FN |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 24 h | 0.918 | 0.549 | 0.451 | 234 / 474 | 2,616 / 2,152 / 234 |
  | 48 h | 0.903 | 0.388 | 0.612 | 342 / 762 | 3,194 / 5,030 / 342 |
  | 72 h | 0.856 | 0.351 | 0.649 | 602 / 1,242 | 3,586 / 6,624 / 602 |

  The 10:1 cost stance buys high recall at the price of many false alarms — the expected
  operating point. Pooled across horizons, D16 misses **1,178** flood rows vs D7's **2,478**
  (**−52.5 %**). Event recall (D16) is **0.929 / 0.966 / 0.966** vs D7 **0.861 / 0.913 /
  0.891** (`reports/mi6_test_read.md`, `reports/mi6_test_metrics.csv`).
- The earlier Phase-33 statement "missed floods 2,478 → 1,044 (−58 %)" refers to the Phase-33
  `xgboost` selection model, **not D16**, and must not be quoted as the D16 headline.

### 3.3 Risk bands (phase29, current config)

- Four cost-derived bands per horizon. Current frozen cuts
  (`system/config/model_config.json`): 24 h **0.0303 / 0.0605 / 0.434**, 48 h
  **0.0434 / 0.0868 / 0.563**, 72 h **0.0482 / 0.0963 / 0.5277**.
- Top band: 90.0–90.1 % observed flood rate, 13.7–17.7× lift (~90 % of rows low, <0.2 % base
  rate). Quartile default rejected (50–75 % of rows in zero-flood bands)
  (`reports/phase29_thresholds.md`, `phase29_risk_bands.csv`).
- Research-prototype threshold sign-off: `docs/THRESHOLD_SIGN_OFF.md` (2026-11-10). External /
  agency sign-off and alert-delivery credentials remain open.

---

## 4. Explainability (phase34)

- Global exact TreeSHAP of D16 on P1 val_2022 (17,502 rows @ 24 h, no new test read):
  - **G4_hydrology = 64.0 %** of 24 h attribution (led by `wl_dist_to_danger` 1.005,
    `wl_pct_danger` 0.539); then G3_geography 8.4 %, G5_network 6.1 %, G1_weather 4.7 %
    (`reports/phase34_xai.md`).
  - No banned/label-window column in any horizon's top-20.
  - Cross-model: D16 reproduces Phase-33 xgboost ordering on shared 62 features
    (Spearman ρ 0.76/0.86/0.88).
- Local explanations for TP, missed flood (pattern A), false alarm
  (`phase34_local_explanations.csv`).
- DL attribution (integrated gradients) on Transformer (L=7) and LSTM (L=3): 74.7 %/70.6 %
  on G4_hydrology — same dominant layer as tabular SHAP (`phase34_dl_attribution.csv`).

---

## 5. System validity

### 5.1 Public deployments (verified 2026-11-10)

- **Static website / PWA:** https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/
  — HTTP 200. The GitHub Pages build ships a precomputed snapshot built from the full local
  feature store (**2,621 dates, 2017-01-01 → 2026-11-09**), frozen D16, and a browser-side
  risk-band layer. It needs no server or credentials. `health.json` reports
  `model_version: D16`, `feature_store_dates: 2,621`, `latest_feature_date: 2026-11-09`,
  `station_count: 81` on the latest date and `trend_station_count: 127`, `degraded: false`.
- **Live API + app:** https://aabhas-flood-predictor-xckw.onrender.com — `/health` HTTP 200
  on 2026-11-10: `model_loaded: true`, `model_version: D16`, **120** bundled feature dates
  through **2026-11-05**, all three critical sources `ok`, `degraded: false`. This is a
  **recent-current demo slice**, not the full 2,621-date history. Render's free tier sleeps
  after inactivity; the first request can take about a minute to wake the service.
- **Architecture:** one FastAPI inference service (Docker) serving the React PWA and the JSON
  API, plus a separate static snapshot build for zero-cost public hosting. The frozen D16
  model is the same artefact in both paths.
- **API endpoints:** `/health`, `/model/info`, `/thresholds`, `/dates`, `/predict`,
  `/predict/batch`, `/predict/all`, `/predict/trend`, `/monitoring`, `/alerts/feed`
  (`DEPLOYMENT.md` §2.4).
- **CI/CD:** GitHub Actions runs API import checks, TypeScript typecheck, production build and
  the test suites on push; the static snapshot publish and daily data refresh are manual /
  opt-in workflows (`DEPLOYMENT.md` §3, `.github/workflows/`).

### 5.2 Location selection

- Prediction unit: one FFWC station × one issue day (133 stations in the modelling network,
  station point + district/upazila/river/basin attributes)
  (`reports/phase0_research_definition.md` §2).
- Location-selection UI: station list + choropleth risk map (ADM1/ADM2 boundaries in
  `data/raw/geo/admin/`) (`reports/phase0_research_definition.md` stage 15).
- Latest snapshot station coverage: 81 stations with predictions on the latest feature date,
  127 in the 90-day trend window; the UI states the scope explicitly.

### 5.3 Ingestion and automation

- FFWC station water levels, NASA POWER rainfall/soil, Open-Meteo ERA5-Land, GloFAS
  transboundary discharge, ESA WorldCover landcover (`reports/mi_status.md` §1, MI-1).
- Local feature store: 2,621 dated partitions, current through **2026-11-09**. Feature-quality
  gate writes `system/data/processed/quality/<date>.json`; latest local partition passed with
  critical null share 0.0.
- GloFAS, NASA POWER soil and ERA5-Land are idempotent incremental refreshers;
  `system/ingestion/refresh_sources.py` runs as **Step 0** of the daily pipeline and writes
  `source_status/<date>.json`.
- A free-tier GitHub Actions daily refresh exists (`.github/workflows/refresh.yml`, 04:30 UTC
  cron) but is **disabled until** `AABHAS_REFRESH_ENABLED=true` and the HF/Render secrets are
  configured. Windows Task Scheduler remains the local fallback. See
  `docs/FREE_TIER_OPERATIONS.md`.
- The Render demo slice is rebuilt with `scripts/build_demo_store.py --days 120` and refreshed
  by redeploying; it is not fed by the GitHub Actions cron in the current free-tier setup.

### 5.4 Alert engine

- Risk bands + thresholds drive the alert logic (`reports/phase29_thresholds.md`,
  `system/config/alert_config.json`); draft → human approval → dispatch → hash-chained audit.
- External delivery is disabled (`notification.enabled = false`) by design. A research-prototype
  threshold sign-off is recorded (`docs/THRESHOLD_SIGN_OFF.md`, 2026-11-10); an agency/named
  duty-officer sign-off and SMTP/webhook credentials are still required before any public
  warning can be sent.

---

## 6. Overall assessment

| Validity dimension | Verdict | Evidence |
|---|---|---|
| Data validity | PASS | Labels per contract; missingness NaN-never-0; leakage audited; embargo 0 rows; incident guarded |
| Model validity (temporal) | PASS | D16 val_2022 PR-AUC 0.9574; beats D7 on 21/21 dev points |
| Model validity (spatial) | PASS | Leaders ≥ D7 in 63/63 points; 126/126 seasonal |
| Model validity (monsoon) | PASS | Gap ≤ 0.0435 temporal; thresholds re-derived per period |
| Model validity (imbalance) | PASS | 70:1 test ratio handled; accuracy banned as headline |
| Prediction validity (calibration) | PASS | OOF ECE 0.0031; 17/24 models ECE ≤ 0.02 raw |
| Prediction validity (FN/operational) | PASS | Test PR-AUC 0.7708 vs D7 0.6675; missed flood rows 1,178 vs 2,478 (−52.5 %) |
| Prediction validity (risk bands) | PASS | Top band ~90 % flood rate, 13.7–17.7× lift |
| Explainability | PASS | Hydrology-first SHAP; no leakage in top-20; local explanations |
| System validity | PASS (research prototype) | Public static site and live API verified 2026-11-10; API, map, ingestion, quality gate, tests and alert workflow implemented. Production remains PARTIAL: managed always-on scheduling, external sign-off and delivery credentials are open |

**Overall:** The project achieves its core research goal — a leakage-aware, probabilistic,
station-level 24/48/72-hour flood outlook for Bangladesh, frozen as D16 and verifiable from
committed artefacts. The public static site and the live API are both online and serving D16
output. The frozen test is closed, and the programme-level read history is disclosed in
`reports/test_read_ledger.md`; the reported headline is the valid MI-6 read, not the maximum of
several opens.

The remaining work is **operational, not scientific**: managed always-on scheduling, an
external/agency threshold sign-off, real alert-delivery credentials, and a named institutional
recipient. Until then the system is a **research prototype and not an official warning
service**; FFWC advisories always take precedence. Stage 73 (research report / thesis) is
complete in draft (`reports/phase45_thesis.md`); PDF/typesetting is a publication task.
