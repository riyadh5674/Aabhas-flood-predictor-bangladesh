# Model Improvement Programme v2 (MI‑v2)

**Status:** COMPLETE (2026-10-16) — MI-1 done (ERA5 **complete + evaluated**, 133/133), MI-2 done, MI-3 done (ensemble rejected), MI-4 done, MI-5 done (frozen `hpo_perhorizon_FULL_V2`, decision D16), **MI-6 done — the frozen test read once** (D16 test horizon-mean 0.77076 vs Phase-33 0.75601 vs D7 0.66755); ERA5 probe = wash on the primary cell, so D16 stands. Live state: `reports/mi_status.md`.
**Created:** 2026-10-15 · **Updated:** 2026-10-16
**Supersedes for selection purposes:** the Phase-33 record (`xgboost` L2-03) is superseded by the MI-5 selected model **`hpo_perhorizon_FULL_V2`** (decision D16, `models/final_v2/`), confirmed on the single MI-6 test read. Phase 33 stays as the historical baseline.
**Governing rules:** `final workflow.md` (decisions D8 up-cell-only, D15 frozen-model discipline, Phase-16 §10 accuracy-banned-as-headline, leakage audit Phase 13).

### Progress snapshot (2026-10-16)

| stage | state | evidence |
|---|---|---|
| MI-1a ERA5-Land | **DONE** 133/133 stations, 669,389 rows; block active (`FULL_V2` 110 → 136) | `data/raw/openmeteo/era5land_station_daily.csv` |
| MI-1b GloFAS | **DONE** 11 anchors, 55,363 rows | `data/raw/glofas/discharge_anchors_daily.csv` |
| MI-1c POWER soil | **DONE** 133 stations, 669,389 rows | `data/raw/power_soil/power_soil_daily.csv` |
| MI-2 v2 features | **DONE** 48 new cols, `FULL_V2=110`; leakage audited | `data/processed/mi_v2_features.csv`, `reports/mi2_features.md` |
| MI-3a compare | **DONE** val_2022 horizon-mean 0.95463 → 0.95589 | `reports/mi3_featureset_summary.json` |
| MI-3a(ERA5) probe | **DONE — negative/marginal** ERA5 = wash on `val_2022` (−0.00025); +0.00075 across dev cells (mostly 72 h) → **not adopted, D16 stands** | `reports/mi3_era5_probe.md` |
| MI-3b HPO | **DONE** 41 configs/horizon; best `lgbm7`/`lgbm7`/`xgb3` → horizon-mean 0.9574 (+0.0028) | `reports/mi3_hpo.csv`, `models/mi/mi_hpo_*.json` |
| MI-3c ensemble | **DONE — REJECTED** loses to best single on val_2022 at every horizon | `reports/mi3_ensemble.csv` |
| MI-4 label audit | **DONE** 0.99 unreachable (base rate + 4–10 % near-threshold noise); ceiling 0.96–0.97 | `reports/mi4_label_audit.md` |
| MI-5 re-select & freeze | **DONE (D16)** selected `hpo_perhorizon_FULL_V2` (lgbm7/lgbm7/xgb3), val_2022 horizon-mean 0.95743 (+0.0028); gates G1/G2 passed | `models/final_v2/`, `reports/mi5_selection.md` |
| MI-6 test read | **DONE** frozen test read **once** (`n_reads=1`): D16 test horizon-mean 0.77076 vs Phase-33 0.75601 vs D7 0.66755; D7 cross-check exact | `reports/mi6_test_read.md`, `mi6_test_metrics.csv`, `mi6_test_read_log.json` |

**Deviations recorded:** Optuna (≥300 trials) and `catboost` are unavailable and must not be
installed, so MI-3b samples with sklearn `ParameterSampler` (41 configs/horizon). ERA5-Land is
now **complete** — 133/133 stations, 669,389 rows (sha `b22d0a92…`); the last 10 landed in
run4 once the Open-Meteo 429 window cleared. The builder gate is **active**, adding 24 `e5_*`
+ 2 `ix_*` (`FULL_V2` 110 → **136**), built non-destructively as the tagged
`mi_v2_features_era5.csv`. The **MI-3a(ERA5) probe is a negative/marginal result**: against the
fixed Phase-33 model the block is a **wash on the primary `val_2022` cell** (−0.00025
horizon-mean; −0.0012 at 24 h) and only +0.00075 across the 7 dev cells (mostly 72 h), so ERA5
was **not adopted** and the D16 freeze stands. The MI-3c ensemble is a **negative result** and
was rejected: it loses to the best single config on the primary cell, so MI-5 used the best
single config per horizon.

---

## 1. Why

The Phase‑33 winner (`xgboost`, L2‑03) is strong on ranking and calibration, but the **primary metric (PR‑AUC) reads 0.95463 horizon‑mean on validation** and drops to 0.71–0.80 on the frozen test. The owner wants this as high as possible ("minimum 0.99").

**Honest framing of the target (must be stated, not hidden):**

| metric | current (val_2022, horizon‑mean) | realistic after MI‑v2 |
|---|---|---|
| **ROC‑AUC** | 0.9854 | already ≥0.99 on most cells; maintain |
| **Accuracy** | 0.9728 | already ~0.99 on easy cells; banned as headline |
| **PR‑AUC** | 0.95463 | **≥0.97** plausible; **0.99 on high‑base‑rate cells** plausible |
| **PR‑AUC on 0.7 %‑base‑rate test years** | 0.75 | **cannot honestly reach 0.99** — base rate caps it |
| **Recall / FN** | 0.901 / 1,044 | improve further |

**Key evidence (page‑33 base‑rate study):** the same model scores PR‑AUC **0.99 where floods are ~10 % of days** and **0.75 where they are 0.7 %**, at constant ROC‑AUC. Base‑rate ↔ PR‑AUC correlation = **+0.87**. A uniform 0.99 PR‑AUC at a 0.7 % base rate is not reachable; the plan below maximises what *is* reachable, honestly.

**Levers not yet used:** the model is trained on a **62‑feature set** built entirely from FFWC water levels + **NASA POWER (MERRA‑2, ~50 km) reanalysis rainfall** + static geography. There is **no soil moisture, no landcover, no observed/satellite rainfall, and no true transboundary signal.** Those are the gaps this programme closes.

---

## 2. Data sources — verified reachable (probe 2026‑10‑15)

| source | what | access | status |
|---|---|---|---|
| **Open‑Meteo archive (ERA5‑Land)** | daily precipitation, ET₀, **soil moisture 0‑7 / 7‑28 / 28‑100 cm**, runoff, temperature, radiation | `archive-api.open-meteo.com`, no auth | ✅ 200 |
| **Open‑Meteo flood API (GloFAS)** | **river discharge** on a 5 km grid incl. India/Nepal/Bhutan/China upstream | `flood-api.open-meteo.com`, no auth | ✅ 200, 1984→present |
| **NASA POWER** | soil wetness `GWETTOP/GWETROOT/GWETPROF` | `power.larc.nasa.gov`, no auth | ✅ 200 |
| **CHIRPS v2.0** | observed/blended daily rainfall (tif.gz) | `data.chc.ucsb.edu` | ✅ 200 (heavy: 1 file/day) |
| Copernicus CDS / GES DISC | ERA5 native / IMERG / SMAP | requires account/token | ⚠️ auth needed — not used |

**Chosen primaries:** Open‑Meteo ERA5‑Land (soil + higher‑res rain) and Open‑Meteo GloFAS (transboundary discharge). CHIRPS and POWER soil wetness are optional reinforcements.

---

## 3. Stages, deliverables, definition‑of‑done

Every stage must produce, per `final workflow.md` §0.2: **code + run log (exit code, wall‑clock, seed) + report + checklist/phase entry**, all agreeing.

### MI‑1 — Data expansion (Tier 1)
- **MI‑1a** ERA5‑Land daily for all **133** stations, 2013‑01‑01 → present: `precipitation_sum, et0_fao_evapotranspiration, soil_moisture_0_to_7cm_mean, soil_moisture_7_to_28cm_mean, soil_moisture_28_to_100cm_mean, temperature_2m_mean, runoff, shortwave_radiation_sum`.
- **MI‑1b** GloFAS discharge at upstream/transboundary anchor points (Farakka, Patna, Tezpur, Dibrugarh, Chisapani, Bhimnagar, Silchar, Dharla) + in‑country (Bahadurabad, Hardinge, Bhairab Bazar) and at each station's best GloFAS river cell.
- **MI‑1c** (optional) CHIRPS daily rainfall sampled to station coords; NASA POWER soil wetness.
- **Deliverable:** `data/raw/openmeteo/era5land_station_daily.csv`, `data/raw/glofas/discharge_points_daily.csv`, provenance manifest with SHA‑256.
- **DoD:** row counts match expected dates × stations; ≤0.5 % unexpected nulls; monotonic dates; provenance log committed.

### MI‑2 — Feature engineering (Tier 2)
Build **v2 feature set** joined onto the existing key `(station_id, date)`:
- longer hydrology memory: `wl_last_lag10/14/21/30`, `wl_max_lag10/14`;
- antecedent precipitation index (API) with 3 decay constants;
- soil‑moisture levels + anomalies vs. per‑station day‑of‑year climatology;
- **upstream discharge** features lagged over travel times 1–10 d (per basin), plus discharge rate‑of‑rise and anomaly;
- rate‑of‑rise windows (2/5/10 d), `wl_pct_danger` slope;
- interactions: `rain_7d × soil_moisture_0_7`, `rain_7d × slope_deg`, `up_discharge_rate × wl_rise`;
- basin/day aggregates.
- **Deliverable:** `data/processed/mi_v2_features.csv`, `data/processed/feature_sets_v2.json`.
- **DoD:** leakage audit (Phase‑13 rule: nothing at ≥ t+1); null rates documented; v1 `FULL` and v2 subsets both emitted so ablation is clean.

### MI‑3 — Model search (Tier 3)
- **MI‑3a** Optuna HPO per horizon (n_trials ≥ 300, seed fixed), objective = horizon‑mean PR‑AUC on validation.
- **MI‑3b** Model pool expansion: add `catboost`, `hist_gradient_boosting`, `extra_trees` to the existing L2 family.
- **MI‑3c** Stacking/ensemble (out‑of‑fold on train; meta‑learner = logistic or ridge), plus optional calibrated DL blend.
- **Deliverable:** `models/mi/*.pkl`, `reports/mi3_hpo.csv`, `reports/mi3_ensemble.json`.
- **DoD:** every candidate scored on the same validation cells as Phase 27; no test contact.

### MI‑4 — Label‑quality audit (Tier 4)
- Quantify label noise (FFWC warning‑level crossings), class‑imbalance drift, and an **empirical ceiling** (e.g. Bayes‑error proxy via inter‑model disagreement vs. label). State whether PR‑AUC 0.99 is label‑limited.
- **Deliverable:** `reports/mi4_label_audit.md`.

### MI‑5 — Re‑selection & freeze — **DONE (D16)**
- Re-ran the Phase‑33 scorecard on **frozen validation evidence only** (7 dev cells) with the v2 features/models; applied gates G1/G2 and the D15 rule; froze the winner to `models/final_v2/`.
- **Result:** `hpo_perhorizon_FULL_V2` (lightgbm `lgbm7` @ 24/48 h + xgboost `xgb3` @ 72 h on `FULL_V2`) — val_2022 horizon-mean **0.95743** (+0.00280 vs Phase 33); all five candidates cleared both gates; test never read.
- **DoD met:** `reports/mi5_scorecard.csv`, `reports/mi5_selection.json`, `models/final_v2/{README.md, final_config_v2.json, selected_model_v2.pkl}`, `reports/mi5_selection.md`, decision **D16** recorded.

### MI‑6 — Single logged test read — **DONE**
- Opened the frozen test (`>= 2023`) **exactly once** for the MI-5/D16 model + the D7 bar; guarded (`n_reads = 1`), same schema as `phase33_test_read_log.json`.
- **Result:** D16 test horizon-mean **0.77076** (24 h 0.83623 / 48 h 0.76896 / 72 h 0.70709) vs Phase‑33 `xgboost` **0.75601** vs D7 **0.66755**; D16 beats D7 on all 21 test points and the Phase‑33 model on 6/7 cells. D7 reproduces the Phase‑16 read exactly (max|Δ| = 5.6e‑17).
- **DoD met:** `reports/mi6_test_metrics.csv`, `reports/mi6_test_read_log.json`, `reports/mi6_thresholds.csv`, `reports/mi6_summary.json`, `reports/mi6_test_read.md`.

---

## 4. Evaluation discipline (non‑negotiable)
1. Test (2023+) is **untouched** until MI‑6. Any decision uses train/validation/spatial‑CV only.
2. Improvements are claimed only if they clear the same **common 95 % CI** bar Phase 33 used.
3. Accuracy is reported but **never the headline** (§10).
4. Cells never pooled (D8).
5. Every stage is reproducible from a seed and logged.

## 5. Expected gains (honest, ordered by leverage)
| lever | est. val PR‑AUC Δ | effort | risk |
|---|---|---|---|
| GloFAS transboundary discharge (travel‑time lags) | **+0.010 … +0.025** | medium | low |
| ERA5‑Land soil moisture + higher‑res rain | +0.005 … +0.015 | low | low |
| Feature engineering (lags, API, interactions) | +0.005 … +0.010 | low | low |
| Optuna HPO | +0.005 … +0.015 | medium | low |
| Stacking/ensemble | +0.003 … +0.008 | medium | low |
| CHIRPS observed rain | +0.003 … +0.010 | high | low |
| Label audit | 0 (defines ceiling) | low | — |

**Cumulative honest expectation:** val_2022 horizon‑mean PR‑AUC **0.95 → 0.97‑0.98**; 24 h arguably **>0.98**; ROC‑AUC stays ≥0.99; FN falls further. **A uniform 0.99 PR‑AUC across all cells including 0.7 %‑base‑rate test years is not expected and must not be promised.**

## 6. Risks
- **R‑MI1** Open‑Meteo rate limits / schema changes → cache aggressively, retry, manifest.
- **R‑MI2** GloFAS grid‑cell mismatch for a station → pick max‑discharge cell within 0.2°, record the chosen cell.
- **R‑MI3** ERA5‑Land latency (~5 d) vs. issue time → provide both same‑day and lag‑3 variants; document operational caveat.
- **R‑MI4** Overfitting validation with HPO → limit trials, use spatial‑CV as the tie‑breaker, keep the CI bar.
- **R‑MI5** Silent leakage from reanalysis → Phase‑13 audit on every new column.
