# Phase 34 — Explainable AI (stage 58)

**Status:** DONE · **Seed:** `20261008` · **Wall-clock:** 202.2 s
**Model explained:** `hpo_perhorizon_FULL_V2` (decision **D16**, `models/final_v2/`), per-horizon
family `{"label_24h": "lgbm", "label_48h": "lgbm", "label_72h": "xgb"}`, feature set **FULL_V2**
(110 columns). The **Phase-33** incumbent (`xgboost`, FULL 62) is scored
for cross-model agreement.
**Cell:** P1 primary `val_2022` (holdout excluded), **17,502 rows**
@ 24 h. **Test period: `test_rows_read = 0`** — Phase 34 opens no new test read; all
explanations are on validation.
**Dataset:** `data/processed/experimental_dataset.csv`, SHA-256 `a6748529083fcdce…` — unchanged.

---

## 0. Method

| target | method | why |
| --- | --- | --- |
| global, tabular | exact **TreeSHAP** on each frozen per-horizon booster (lightgbm 24/48 h, xgboost 72 h) | tree ensembles admit exact Shapley values; the sklearn `ColumnTransformer` (median imputation + train-fitted frequency encoding) is applied first and the encoded geography columns are mapped back to `G3_geography` |
| cross-check | **permutation importance** (3,000 validation rows, 3 repeats, PR-AUC scoring) | an assumption-free second opinion; a disagreement would be reported, not hidden |
| local, tabular | **per-row SHAP** for a high-confidence true positive, a missed flood and a false alarm | the proposal's requested per-prediction format |
| deep models | **integrated gradients** (baseline = fold mean, 16 steps) on the temporal **Transformer** (L4-03, L = 7) and the **LSTM** (L3-01, L = 3) | architecture-appropriate attribution; the 12 raw sequence variables are aggregated to the **same phase-0 groups** so the two families are comparable |

The five phase-0 layers tile the 62 v1 features exactly (weather 5 → +rain history 7 →
+geography 14 → +hydrology 24 → +network 12); FULL_V2 appends four groups (soil wetness 19,
GloFAS discharge 18, water-level memory/API 6, interactions 5) — 110 columns in total.

---

## 1. Global drivers (TreeSHAP)

Top-15 features at each horizon (mean absolute SHAP, log-odds units):

### 24 h

| # | feature | group | mean|SHAP| |
| --- | --- | --- | --- |
| 1 | wl_dist_to_danger | G4_hydrology | 1.0053 |
| 2 | wl_pct_danger | G4_hydrology | 0.5389 |
| 3 | wl_exceed_days_30 | G4_hydrology | 0.1341 |
| 4 | lon | G3_geography | 0.0853 |
| 5 | wl_rise_1d | G4_hydrology | 0.0744 |
| 6 | precip_mm | G1_weather | 0.0665 |
| 7 | nb3_n_wl_danger | G5_network | 0.0559 |
| 8 | ix_updis_x_wlrise | G9_interactions | 0.0497 |
| 9 | api_07 | G8_wl_memory_api | 0.0433 |
| 10 | nb3_wl_pct_danger | G5_network | 0.0389 |
| 11 | freq_district | G3_geography | 0.0323 |
| 12 | ix_api09_x_wet_top | G9_interactions | 0.0322 |
| 13 | nb5_wl_pct_danger | G5_network | 0.0303 |
| 14 | wl_rise_3d | G4_hydrology | 0.0273 |
| 15 | lat | G3_geography | 0.0267 |

### 48 h

| # | feature | group | mean|SHAP| |
| --- | --- | --- | --- |
| 1 | wl_dist_to_danger | G4_hydrology | 1.0112 |
| 2 | wl_pct_danger | G4_hydrology | 0.5861 |
| 3 | lon | G3_geography | 0.1565 |
| 4 | wl_exceed_days_30 | G4_hydrology | 0.1385 |
| 5 | precip_mm | G1_weather | 0.1132 |
| 6 | api_07 | G8_wl_memory_api | 0.0928 |
| 7 | wl_rise_1d | G4_hydrology | 0.0866 |
| 8 | doy_cos | G1_weather | 0.0573 |
| 9 | ix_updis_x_wlrise | G9_interactions | 0.0569 |
| 10 | nb3_wl_pct_danger | G5_network | 0.0564 |
| 11 | nb3_n_wl_danger | G5_network | 0.0504 |
| 12 | ix_api09_x_wet_top | G9_interactions | 0.0499 |
| 13 | lat | G3_geography | 0.0389 |
| 14 | rain_1d | G2_rain_history | 0.0364 |
| 15 | freq_district | G3_geography | 0.0363 |

### 72 h

| # | feature | group | mean|SHAP| |
| --- | --- | --- | --- |
| 1 | wl_dist_to_danger | G4_hydrology | 2.1983 |
| 2 | wl_pct_danger | G4_hydrology | 0.8186 |
| 3 | precip_mm | G1_weather | 0.2917 |
| 4 | lon | G3_geography | 0.2555 |
| 5 | gf_up_ratio | G7_glofas | 0.2151 |
| 6 | doy_cos | G1_weather | 0.2073 |
| 7 | wl_exceed_days_30 | G4_hydrology | 0.1636 |
| 8 | ix_updis_x_wlrise | G9_interactions | 0.1424 |
| 9 | ps_wet_root_clim | G6_soil_wetness | 0.1283 |
| 10 | ix_api09_x_wet_top | G9_interactions | 0.1162 |
| 11 | wl_rise_1d | G4_hydrology | 0.1121 |
| 12 | gf_up_ratio_lag10 | G7_glofas | 0.1114 |
| 13 | lat | G3_geography | 0.1106 |
| 14 | freq_district | G3_geography | 0.0983 |
| 15 | nb3_wl_pct_danger | G5_network | 0.0944 |

**Group shares** (fraction of total mean|SHAP|) at 24 h:

| group | n features | sum |SHAP| | share |
| --- | --- | --- | --- |
| G4_hydrology | 24 | 1.870 | 64.0% |
| G3_geography | 14 | 0.246 | 8.4% |
| G5_network | 12 | 0.180 | 6.1% |
| G1_weather | 5 | 0.137 | 4.7% |
| G7_glofas | 18 | 0.135 | 4.6% |
| G9_interactions | 5 | 0.120 | 4.1% |
| G6_soil_wetness | 19 | 0.119 | 4.1% |
| G8_wl_memory_api | 6 | 0.071 | 2.4% |
| G2_rain_history | 7 | 0.045 | 1.5% |

**Scientifically sensible?** Yes. The leading driver is `wl_dist_to_danger`
and the dominant group is **G4_hydrology** (64.0 % of total
24 h attribution) — the water-level family (`wl_dist_to_danger`, `wl_pct_danger`, `wl_*`,
`danger_level`, `api_*`). It is followed by **G3_geography** (8.4 %),
**G5_network** (6.1 %) and **G1_weather**
(4.7 %) — locality, upstream-neighbour stage and raw weather. The
new v2 soil/discharge/memory/interaction groups each carry a few percent: real, but small,
consistent with MI-3a finding that the v2 features only nudged the primary metric. **No
nonsensical driver appears**: a leakage re-check over the 24 h top-20 finds banned/label-window
columns = **none**, so no Phase-13 re-run is triggered.
This is the expected signature for a lead-time flood model — the river *is* the state variable;
locality and the upstream stage add the early-warning content.

![global drivers](figures/phase34_shap_summary.png)

---

## 2. Cross-check and cross-model agreement

* **D16 vs Phase-33 `xgboost`** Spearman ρ on the 62 shared features:
  {"24h": 0.755, "48h": 0.86, "72h": 0.878} — the two frozen models, trained on
  different feature sets (110 vs 62) with different learners, explain the task through the
  **same** hydrology-first ordering. The v2 columns refine, they do not replace, the v1 story.
* **Permutation vs SHAP** Spearman ρ on the per-horizon feature rankings:
  {"24h": 0.021, "48h": 0.404, "72h": 0.262}. **This is low at 24 h and this is a real,
  reportable result, not a bug:** permutation importance is unreliable under strong feature
  collinearity. Once `wl_dist_to_danger` is permuted the signal is destroyed (importance
  0.565), so permuting its near-duplicates (`wl_pct_danger`, `wl_*`) adds almost nothing and
  their permutation ranks collapse to ≈0 — hence ρ ≈ 0.021
  while the SHAP ranking spreads credit across the correlated water-level ladder. The
  cross-model SHAP agreement (row above) is the trustworthy stability check; the permutation
  table is reported as the (instructive) counter-example.

Permutation-importance top-10 at 24 h — note how a **single** correlated feature absorbs the
whole importance mass:

| # | feature | group | importance | std |
| --- | --- | --- | --- | --- |
| 1 | wl_dist_to_danger | G4_hydrology | 0.5653 | 0.0148 |
| 2 | wl_pct_danger | G4_hydrology | 0.0308 | 0.0024 |
| 3 | wl_rise_1d | G4_hydrology | 0.0052 | 0.0015 |
| 4 | ix_updis_x_wlrise | G9_interactions | 0.0046 | 0.0005 |
| 5 | lat | G3_geography | 0.0008 | 0.0001 |
| 6 | api_09 | G8_wl_memory_api | 0.0007 | 0.0000 |
| 7 | ps_wet_top_anom | G6_soil_wetness | 0.0007 | 0.0001 |
| 8 | ps_wet_top_14d | G6_soil_wetness | 0.0007 | 0.0001 |
| 9 | wl_last | G4_hydrology | 0.0006 | 0.0000 |
| 10 | wl_last_lag10 | G8_wl_memory_api | 0.0006 | 0.0001 |

---

## 3. Local explanations (per prediction, 24 h)

The proposal asks for the decision to be readable one row at a time
(*"flood probability 0.86 → top contributing factors"*):


**high-confidence true positive** — station 110 (Fulchari), 2022-06-21, 24 h.  Model probability **P(flood) = 1.00**, observed label = 1.
  * `wl_dist_to_danger` = -0.6 → raises the risk by 6.253 log-odds
  * `wl_pct_danger` = 1.03101 → raises the risk by 3.469 log-odds
  * `wl_exceed_days_30` = 4 → raises the risk by 0.373 log-odds
  * `nb3_n_wl_danger` = 2 → raises the risk by 0.229 log-odds
  * `ix_updis_x_wlrise` = 0 → raises the risk by 0.181 log-odds
  * `nb3_wl_pct_danger` = 1.00204 → raises the risk by 0.158 log-odds
  * `wl_rise_1d` = 0.09 → raises the risk by 0.102 log-odds
  * `nb5_wl_pct_danger` = 1.0027 → raises the risk by 0.064 log-odds

**missed flood (false negative)** — station 70 (Lorergorh), 2022-06-11, 24 h.  Model probability **P(flood) = 0.00**, observed label = 1.
  * `wl_dist_to_danger` = 1.49 → lowers the risk by 0.555 log-odds
  * `api_07` = 101.857 → raises the risk by 0.302 log-odds
  * `wl_mean` = 6.532 → raises the risk by 0.199 log-odds
  * `wl_pct_danger` = 0.814907 → lowers the risk by 0.184 log-odds
  * `wl_exceed_days_30` = 1 → raises the risk by 0.175 log-odds
  * `lon` = 91.2501 → raises the risk by 0.160 log-odds
  * `ix_wet_top_x_slope` = 12.8052 → raises the risk by 0.156 log-odds
  * `ix_api09_x_wet_top` = 326.471 → raises the risk by 0.130 log-odds

**false alarm (false positive)** — station 8 (Mongla), 2022-03-29, 24 h.  Model probability **P(flood) = 0.98**, observed label = 0.
  * `wl_dist_to_danger` = -0.46 → raises the risk by 5.888 log-odds
  * `wl_pct_danger` = 1.25556 → raises the risk by 2.735 log-odds
  * `wl_exceed_days_30` = 12 → raises the risk by 0.332 log-odds
  * `danger_level` = 1.8 → raises the risk by 0.330 log-odds
  * `lon` = 89.6 → raises the risk by 0.202 log-odds
  * `freq_district` = n/a → raises the risk by 0.182 log-odds
  * `ix_updis_x_wlrise` = -1.95505 → lowers the risk by 0.156 log-odds
  * `wl_rise_1d` = 0.54 → raises the risk by 0.122 log-odds

![local explanations](figures/phase34_local_examples.png)

Read top-to-bottom: the true positive is a river already **above** danger with a large recent
exceedance count — the model is doing exactly what the physics says. The missed flood is the
mirror of Phase-30 pattern **A**: the river is still **below** danger at issue time, so the
water-level state reads "safe" and the model under-predicts. The false alarm is a **high-stage
non-event** — a river at/above danger whose next 24 h did not flood — the accepted price of
operating at the **FN ≫ FP** point chosen in Phase 29. Neither local pattern contradicts
Phase 30: the explanations and the error analysis tell the same story.

---

## 4. Deep-model attribution (integrated gradients)

Per-variable mean|IG| shares at 24 h (top 12 of 12), Transformer and LSTM:

| model | variable | group | mean|IG| | share |
| --- | --- | --- | --- | --- |
| transformer_L4-03 | wl_pct_danger | G4_hydrology | 0.1779 | 49.2% |
| transformer_L4-03 | precip_mm | G1_weather | 0.0431 | 11.9% |
| transformer_L4-03 | wl_mean | G4_hydrology | 0.0229 | 6.3% |
| transformer_L4-03 | wl_last | G4_hydrology | 0.0207 | 5.7% |
| transformer_L4-03 | wl_max | G4_hydrology | 0.0202 | 5.6% |
| transformer_L4-03 | doy_sin | G1_weather | 0.0163 | 4.5% |
| transformer_L4-03 | wl_min | G4_hydrology | 0.0147 | 4.1% |
| transformer_L4-03 | doy_cos | G1_weather | 0.0120 | 3.3% |
| transformer_L4-03 | tmax_c | G1_weather | 0.0102 | 2.8% |
| transformer_L4-03 | tmin_c | G1_weather | 0.0098 | 2.7% |
| transformer_L4-03 | n_readings | G4_hydrology | 0.0071 | 2.0% |
| transformer_L4-03 | wl_first | G4_hydrology | 0.0070 | 1.9% |
| lstm_L3-01 | wl_pct_danger | G4_hydrology | 2.6061 | 47.0% |
| lstm_L3-01 | precip_mm | G1_weather | 0.5467 | 9.8% |
| lstm_L3-01 | doy_sin | G1_weather | 0.4450 | 8.0% |
| lstm_L3-01 | wl_min | G4_hydrology | 0.4162 | 7.5% |
| lstm_L3-01 | doy_cos | G1_weather | 0.3315 | 6.0% |
| lstm_L3-01 | n_readings | G4_hydrology | 0.2264 | 4.1% |
| lstm_L3-01 | wl_first | G4_hydrology | 0.2110 | 3.8% |
| lstm_L3-01 | wl_last | G4_hydrology | 0.1998 | 3.6% |
| lstm_L3-01 | wl_mean | G4_hydrology | 0.1766 | 3.2% |
| lstm_L3-01 | tmin_c | G1_weather | 0.1726 | 3.1% |
| lstm_L3-01 | tmax_c | G1_weather | 0.1349 | 2.4% |
| lstm_L3-01 | wl_max | G4_hydrology | 0.0835 | 1.5% |

Group shares (24 h): {"transformer_L4-03": {"G1_weather": 0.25265, "G4_hydrology": 0.74735}, "lstm_L3-01": {"G1_weather": 0.2938, "G4_hydrology": 0.7062}}.

**Comparison with the tabular SHAP.** The DL encoders see only the 12 raw sequence
variables, so they physically cannot express geography or network features; what they *can*
express is the **G4 hydrology** and **G1 weather/rain** layers. Both architectures put the
overwhelming majority of attribution on **G4 hydrology** — the same dominant layer the
tabular TreeSHAP finds — with a smaller weather/rain share. This is the "same feature
groups" comparison the workflow asks for: the deep nets and the gradient-boosted tables
agree on *where the signal is*, which is why the Phase-27/33 outcome (ML ahead, DL never
nonsensically better) is explainable rather than mysterious.

![group attribution](figures/phase34_group_attribution.png)

---

## 5. Findings

* **F34-1 — The model is hydrology-first, and that is correct.** The dominant global group
  is `G4_hydrology` (64.0 % of 24 h attribution), led by
  `wl_dist_to_danger` / `wl_pct_danger`. Locality (`G3_geography`) and upstream-neighbour
  stage (`G5_network`) come next; antecedent rainfall and the v2 soil/discharge/memory groups
  are real but smaller. No driver is scientifically indefensible and no banned column leaks
  into the top-20.
* **F34-2 — The explanation is stable across models, but permutation importance is not a
  usable cross-check here.** The D16 model reproduces the Phase-33 ordering on the shared 62
  features (ρ ≈ 0.76–0.88),
  so the v1→v2 feature change did not reshape the explanation. By contrast, permutation
  importance disagrees with SHAP (ρ ≈ 0.021 at 24 h)
  because a **single** dominant feature (`wl_dist_to_danger`) absorbs the whole importance
  mass once its near-duplicates are present — the textbook failure mode of permutation
  importance under collinearity. This is reported as evidence that the SHAP ranking, not the
  permutation ranking, is the reliable one.
* **F34-3 — Local explanations reproduce the Phase-30 error patterns.** The missed-flood
  example is a below-danger river (pattern A); the false alarm is a high-stage non-event at
  the FN ≫ FP operating point. XAI confirms the error analysis instead of contradicting it.
* **F34-4 — DL attribution localises the same layers.** Transformer and LSTM integrated
  gradients both concentrate on G4 hydrology with a weather/rain secondary — the same
  ordering as the tabular SHAP where the two can be compared.
* **F34-5 — Explanations add no leakage.** The v2 features appear as contributors with
  modest shares, consistent with their small measured gain (MI-3a +0.0013); nothing in the
  attribution suggests a future-information path.

---

## 6. Limitations

1. Explanations are computed on **validation** (`val_2022`), the same cell used for every
   downstream decision; the frozen test is not re-read for XAI. Attribution on the test
   period would require a new, logged test read and is out of scope.
2. **Correlated features share credit.** SHAP values for `wl_pct_danger` and
   `wl_dist_to_danger` (and the lag ladder) are not independent; the *group* shares are the
   robust quantity, individual-feature ranks less so.
3. The DL attribution covers the **temporal** Transformer and LSTM (the models with the
   cleanest public checkpoints); the fusion/GNN variants are not re-explained. Integrated
   gradients are an approximation and depend on the chosen baseline (fold mean).
4. Frequency-encoded geography columns (`freq_*`) are explained in their encoded scale and
   mapped to `G3_geography`; they carry locality, not raw coordinates.

---

## 7. Artefacts

| artefact | content |
| --- | --- |
| `reports/phase34_global_shap.csv` | TreeSHAP, 330 rows = 3 horizons × 110 features |
| `reports/phase34_group_shap.csv` | 27 group × horizon rows |
| `reports/phase34_permutation_importance.csv` | 330 permutation rows |
| `reports/phase34_local_explanations.csv` | 24 rows — 3 exemplars × top-8 factors |
| `reports/phase34_dl_attribution.csv` | 72 rows — 2 DL models × 3 horizons × 12 variables |
| `reports/phase34_crossmodel.csv` | 186 rows — D16 vs Phase-33 SHAP (shared 62) |
| `reports/phase34_summary.json` | headline facts for the verification notebook |
| `figures/phase34_shap_summary.png` | global drivers |
| `figures/phase34_group_attribution.png` | tabular SHAP vs DL IG by group |
| `figures/phase34_local_examples.png` | local explanations |
| `notebooks/34_explainable_ai.ipynb` | self-verifying notebook (N34-01…N34-12) |

*Note on the DoD name.* `final workflow.md` Phase 34 names the notebook
`notebooks/10_explainability.ipynb`; under decision **D11** the phase-numbered notebook
`notebooks/34_explainable_ai.ipynb` is the governed artefact (same convention as
`33_selection.ipynb` … `17_framework_registry.ipynb`).

Run: `python -u scripts/phase34_xai.py` (seed `20261008`).
