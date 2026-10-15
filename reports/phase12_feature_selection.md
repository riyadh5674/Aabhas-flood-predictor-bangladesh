# Phase 12 — Feature selection (workflow body steps 18–21)

| | |
| --- | --- |
| **Date** | 2026-10-08 (executed 15:58–16:08, one clean pass) |
| **Module** | `notebooks/04_feature_selection.ipynb` (source of truth = `scripts/build_phase12_notebook.py`) |
| **Input** | `data/processed/flood_dataset_model_ready.csv` — 200,665 × 73, 133 stations, 2017-01-06 → 2026-10-05 |
| **Outputs** | `data/processed/feature_sets.json`, `feature_selection_mi.csv`, `feature_selection_permimportance.csv`, `figures/phase12_mi.png`, `phase12_perm_importance.png`, `phase12_selection_curve.png` |
| **Verdict** | **13/13 code cells executed, 0 errors, every `assert` green.** Test years 2023+ never read by any selection decision. |
| **Governing doc** | `flood prediction workflow.md` **Phase 12 — Feature selection** (steps 18 statistical filter → 19 MI → 20 model-based → 21 recursive/sequential); `reports/phase0_research_definition.md` §5 information boundary; decisions **D2** (3 horizons equal), **D7** (persistence is the bar), **D8** (no pooled metrics), **D10** (body-Phase order) |

Run command (kernel must be the project venv — an earlier attempt in a non-venv kernel died on
`No module named 'lightgbm'`):

```
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=flood-predictor notebooks\04_feature_selection.ipynb
```

Nothing under `data/raw/` was touched.

---

## 1. Step 18a — admissible pool and constants

§5 boundary arithmetic, not a re-invented whitelist: 73 columns − 10 banned
(`station_id name date source`, three `label_*`, three `obs_days_*`) = **63 approved** →
`day_observed` is constant in the model-ready table (finding G3) → **62 usable features**
(5 text: `river basin division district upazilla`; 57 numeric).

## 2. Provisional temporal split (needed by steps 20–21)

| part | dates | rows | role |
| --- | --- | ---: | --- |
| train | 2017-01-06 → 2021-12-31 (2018 absent, FFWC gap F2) | 77,571 | fit encoder/imputer, MI, models |
| val | 2022-01-01 → 2022-12-31 | 21,920 | all importance/selection decisions |
| test | ≥ 2023-01-01 | 101,174 | **untouched** — reserved for Phase 15/16 |

Label base rates (a Phase 15/16 warning already visible here):

| | train | val | test |
| --- | ---: | ---: | ---: |
| `label_24h` | 9.60 % | 5.08 % | 1.41 % |
| `label_48h` | 10.73 % | 5.96 % | 1.77 % |
| `label_72h` | 11.73 % | 6.64 % | 2.12 % |

Preprocessing is fitted on **train only**: frequency encoding for the five text columns
(deliberately *not* target encoding — that would leak the label), median imputation from
train medians. `assert`ed 0 NaN / all-finite in all three matrices.

## 3. Step 19 — Mutual information (the "MI step")

`mutual_info_classif`, k=5, train rows only, all three horizons, ranked by `mi_mean`.

| rank | feature | MI 24h | MI 48h | MI 72h | mi_mean |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `wl_dist_to_danger` | 0.2560 | 0.2565 | 0.2533 | 0.2553 |
| 2 | `wl_pct_danger` | 0.2553 | 0.2539 | 0.2518 | 0.2537 |
| 3 | `wl_exceed_days_30` | 0.1541 | 0.1538 | 0.1533 | 0.1537 |
| 4 | `rain_max_7d` | 0.1204 | 0.1296 | 0.1368 | 0.1290 |
| 5 | `nb5_n_wl_danger` | 0.1219 | 0.1222 | 0.1228 | 0.1223 |
| 6 | `nb3_wl_pct_danger` | 0.1212 | 0.1229 | 0.1224 | 0.1221 |
| 7 | `nb3_n_wl_danger` | 0.1196 | 0.1206 | 0.1189 | 0.1197 |
| 8 | `nb5_wl_pct_danger` | 0.1133 | 0.1145 | 0.1147 | 0.1142 |
| 9 | `lon` | 0.0711 | 0.0752 | 0.0794 | 0.0752 |
| 10 | `river_density_10km_km_per_km2` | 0.0696 | 0.0750 | 0.0802 | 0.0749 |

Full table: `feature_selection_mi.csv`. Figure: `figures/phase12_mi.png`.

## 4. Step 18b — redundancy collapse + VIF → the LINEAR set

* **Redundancy**: Spearman |ρ| ≥ 0.95 on train, walked in MI order so the survivor of each
  near-duplicate group is the higher-MI feature. **23 of 62 collapsed** (12 shown in the
  notebook output), e.g. `nb5_wl_pct_danger`→`nb3_wl_pct_danger` (0.978),
  `rain_1d`→`precip_mm` (1.000), `wl_max/wl_last/wl_first/wl_mean`→`wl_min` (0.997–0.999).
* **VIF**: iterative removal of the worst offender while VIF > 10 (correlation-matrix
  inverse) → **4 removed** (`wl_min`, `up_rain7d_lag1`, `danger_level`, `rain_7d`).
* Result: **LINEAR = 35 features** — the set the logistic-regression baseline in Phase 16
  must use (phase5 §6: unpruned logreg is numerically unstable). Tree models keep all 62.

## 5. Step 20 — model-based importance vs the persistence bar (D7)

LightGBM (400 trees, lr 0.05, leaves 64) on the 62 features, scored PR-AUC on **2022 only**:

| horizon | persistence (`wl_pct_danger`) | LightGBM 62 feats | ratio |
| --- | ---: | ---: | ---: |
| `label_24h` | 0.9064 | 0.9591 | 1.058 |
| `label_48h` | 0.8761 | 0.9442 | 1.078 |
| `label_72h` | 0.8446 | 0.9258 | 1.096 |

Permutation importance (5 repeats, mean PR-AUC drop) top 10 — `wl_dist_to_danger` is
dominant (**0.526** mean drop) and `wl_pct_danger` second (**0.027**); the tail is flat.

| feature | perm 24h | perm 48h | perm 72h | mean |
| --- | ---: | ---: | ---: | ---: |
| `wl_dist_to_danger` | 0.6555 | 0.5377 | 0.3847 | 0.5259 |
| `wl_pct_danger` | 0.0174 | 0.0262 | 0.0372 | 0.0269 |
| `wl_rise_1d` | 0.0162 | 0.0169 | 0.0183 | 0.0172 |
| `precip_mm` | 0.0043 | 0.0086 | 0.0089 | 0.0073 |
| `wl_exceed_days_30` | 0.0027 | 0.0029 | 0.0015 | 0.0024 |
| `nb3_wl_pct_danger` | 0.0012 | 0.0022 | 0.0024 | 0.0019 |
| `wl_last` | 0.0005 | 0.0017 | 0.0034 | 0.0019 |
| `danger_level` | 0.0007 | 0.0020 | 0.0027 | 0.0018 |
| `wl_rise_3d` | 0.0009 | 0.0019 | 0.0022 | 0.0017 |
| `lon` | 0.0012 | 0.0012 | 0.0022 | 0.0015 |

Tree split-gain agrees directionally (`wl_dist_to_danger` = 0.77 of total gain, then
`wl_rise_1d` 0.028, `precip_mm` 0.018, `wl_pct_danger` 0.017, `doy_sin/cos` next).
Rank agreement across the three methods is only moderate — Spearman on the 62 ranks:
**perm vs gain 0.541, MI vs gain 0.309, perm vs MI 0.087**. MI is a *filter* criterion,
perm/gain are *model* criteria; the disagreement is expected and is exactly why the
workflow asks for multiple approaches rather than one.

## 6. Step 21 — recursive/sequential selection (backward elimination curve)

Top-k by permutation importance, refit on train, scored on val 2022, all three horizons:

| k | 24h | 48h | 72h | mean |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.9426 | 0.9233 | 0.9130 | 0.9263 |
| **10** | 0.9543 | 0.9389 | 0.9283 | **0.9405** |
| 15 | 0.9566 | 0.9427 | 0.9304 | 0.9432 |
| 20 | 0.9589 | 0.9439 | 0.9273 | 0.9434 |
| 30 | 0.9598 | 0.9438 | 0.9320 | **0.9452 (best)** |
| 40 | 0.9598 | 0.9423 | 0.9319 | 0.9447 |
| 62 | 0.9591 | 0.9426 | 0.9258 | 0.9425 |

Documented rule (not eyeballed): **smallest k within 0.01 mean PR-AUC of the best k**
→ best mean 0.9452 at k=30, threshold 0.9352 → **k\* = 10**. Note k=10 also *beats* the
full 62-feature model on 48h/72h — the flat tail is noise, not signal.

**SELECTED (10):** `wl_dist_to_danger, wl_pct_danger, wl_rise_1d, precip_mm,
wl_exceed_days_30, nb3_wl_pct_danger, wl_last, danger_level, wl_rise_3d, lon`

## 7. Outputs

`data/processed/feature_sets.json` — machine-readable record of everything above
(banned list, thresholds, split, selection rule, counts, validation PR-AUC):

| set | n | who uses it |
| --- | ---: | --- |
| **FULL** | 62 | tree models (LightGBM/XGBoost/RF) |
| **LINEAR** | 35 | logistic-regression baseline (Phase 16) |
| **SELECTED** | 10 | compact/interpretable model + cross-check |

## 8. Findings raised by this phase

1. **F/G-NEW — one feature dominates.** `wl_dist_to_danger` carries 77 % of tree gain and
   a 0.53 mean perm-drop; the persistence bar (`wl_pct_danger`) alone already reaches
   0.84–0.91 PR-AUC. The model's value over persistence is real but **modest (×1.06–1.10)**
   and concentrated in rainfall-rise + spatial terms. Phase 13 must rule on
   `wl_dist_to_danger`'s admissibility with particular care: it is derived from `wl_last`
   and `danger_level`, both measured on day `t`.
2. **Label base rate drifts hard across the split** (9.6 % → 5.1 % → 1.4 % for 24 h).
   Unrelated to selection, but Phase 15/16 must not report pooled metrics (D8) and Phase 22
   calibration will be tested by this drift.
3. **Rank agreement perm↔MI = 0.087.** Both are retained as complementary criteria; the
   SELECTED set follows perm (model-based), MI justified the redundancy survivors.
4. **Redundancy is severe**: 23/62 pairs at |ρ| ≥ 0.95 — confirms phase5 finding on
   antecedent-rainfall and gauge-level collinearity; the LINEAR set is the practical answer
   for the logreg baseline.

## 9. Open items handed to Phase 13 (steps 22–24)

1. Leakage audit must rule explicitly on all 62 FULL columns — especially
   `wl_dist_to_danger`, `wl_pct_danger`, `wl_last`, `wl_exceed_days_30` (all use day-`t`
   stage vs `danger_level`) and the neighbourhood `nb*_n_wl_danger` observation-process
   signals flagged in phase5 E7.
2. `obs_days_*` and labels stay banned (already enforced by `assert`).
3. After Phase 13 demotes/promotes any column, **re-run this notebook**
   (`build_phase12_notebook.py` → nbconvert) so `feature_sets.json` stays authoritative.

**Next: Phase 13 — Critical leakage audit (body steps 22–24: feature / target / duplicate-time
leakage), then Phase 14 — define the experimental dataset.** Order per D10.
