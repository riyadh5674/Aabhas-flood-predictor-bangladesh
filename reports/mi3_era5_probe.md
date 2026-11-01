# MI-3a(ERA5) — does the ERA5-Land block improve the v2 model? (dev cells only)

| | |
| --- | --- |
| **Stage** | MI-1a follow-up (ERA5 landed) → MI-3a probe |
| **Command** | `.venv\Scripts\python.exe -u scripts\mi3_era5_probe.py` |
| **Run log** | `scripts/mi3_era5_probe.log` |
| **Exit / wall** | `0` / **169 s** |
| **Seed** | `20261008` |
| **Model** | frozen Phase-33 xgboost config (L2-03), *not* re-tuned |
| **Artefacts** | `reports/mi3_era5_probe.csv` · `reports/mi3_era5_probe.json` |
| **Test** | **never read** (`test_rows_read = 0`) — validation + spatial-CV only |

## 1. Setup

ERA5-Land is now **complete** (133/133 stations, 669,389 rows, sha `b22d0a92…`), so the
builder's `era5_block()` gate is active. Built **non-destructively** as the tagged table
`data/processed/mi_v2_features_era5.csv` + `feature_sets_v2_era5.json`
(`build_mi_features.py --tag era5`), leaving the frozen v2 table and every MI-5/MI-6 artefact
untouched.

Three feature sets are compared **on the same rows** with the same fixed model:

| set | cols | meaning |
| --- | ---: | --- |
| `FULL` | 62 | v1 features |
| `FULL_V2_pre` | 110 | MI-2 block **without** ERA5 (read from the frozen `feature_sets_v2.json`) |
| `FULL_V2_era5` | 136 | MI-2 block **with** ERA5 (24 `e5_*` + 2 `ix_*`) |

## 2. Results

**Primary cell — `val_2022` horizon-mean PR-AUC:**

| | `FULL` | `FULL_V2_pre` | `FULL_V2_era5` | Δ ERA5 |
| --- | ---: | ---: | ---: | ---: |
| horizon-mean | 0.95463 | 0.95589 | **0.95564** | **−0.00025** |

Per horizon on `val_2022` (`pre → era5`):

| horizon | `FULL` | `FULL_V2_pre` | `FULL_V2_era5` | Δ |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 0.9681 | 0.9695 | 0.9683 | **−0.0012** |
| 48 h | 0.9549 | 0.9549 | 0.9552 | +0.0003 |
| 72 h | 0.9408 | 0.9432 | 0.9434 | +0.0002 |

**All 7 dev cells horizon-mean PR-AUC:**

| | `FULL` | `FULL_V2_pre` | `FULL_V2_era5` | Δ ERA5 |
| --- | ---: | ---: | ---: | ---: |
| horizon-mean | 0.98341 | 0.98685 | **0.98760** | **+0.00075** |
| 24 h | 0.98911 | 0.99118 | 0.99142 | +0.00024 |
| 48 h | 0.98392 | 0.98699 | 0.98765 | +0.00066 |
| 72 h | 0.97719 | 0.98237 | 0.98374 | **+0.00137** |

*Reproducibility check:* `FULL` 0.95463 and `FULL_V2_pre` 0.95589 reproduce the recorded MI-3a
numbers to the 5th decimal → the probe is measuring exactly what MI-3a measured.

## 3. Verdict

**ERA5-Land is not the lever we hoped for.** It is a **wash on the primary metric** (−0.00025
on `val_2022`, and *negative* at 24 h) and only a **marginal positive elsewhere** (+0.00075
across the 7 dev cells), where the gain is concentrated at **72 h** and on the spatial folds —
not the 24 h short lead that an operational early-warning system cares about most.

Consequences:

- The MI-5 selection rule ranks on **horizon-mean `val_2022` PR-AUC**; on that primary cell the
  ERA5 block does not help, so the **D16 freeze (`hpo_perhorizon_FULL_V2`, 110 cols) stands**.
- A full MI-3b HPO re-run on the ERA5 table (~70 min) is **not justified** by this evidence:
  the fixed-config probe is already neutral-to-negative on the cell the HPO optimises, so the
  expected change is inside noise.
- The tagged ERA5 table/feature-set remain on disk as a **tested-and-marginal** artefact for
  any future pre-registered attempt (e.g. if the label ceiling is raised first).
- **No test read** — and none should occur, since the model choice is unchanged.

Reproduce:

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\build_mi_features.py --tag era5   # the ERA5 table (needs >=130 stations)
.venv\Scripts\python.exe -u scripts\mi3_era5_probe.py                 # this probe
```
