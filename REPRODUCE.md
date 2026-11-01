# Reproducing the paper's results

**Project:** *Leakage-Aware Probabilistic Flood Forecasting for Bangladesh*
**Freeze:** `reports/freeze_manifest.json` · `reports/FREEZE.md`
**Seed:** `20261008` everywhere · **OS used:** Windows 11, Python 3.13.7
**Test policy:** the frozen test is **closed** for this paper. The reported D16 test
numbers come from the valid MI-6 read; no script below touches `split == "test"`.

---

## 0. What can be reproduced, and at what level

| level | starting point | time | what it proves |
| --- | --- | ---: | --- |
| **A. Verify the paper** | frozen artefacts in this repo | ~1 min | every hashed file and headline number is unchanged (`freeze_results.py --check`) |
| **B. Re-run the correction** | `data/processed/experimental_dataset.csv` + MI feature tables | ~35 min | the P2/P3 corrected numbers and all reproduction gates |
| **C. Re-run event analysis** | same, plus the D16 config | ~10 min | the event-level table and figure |
| **D. Re-run the full model programme** | processed dataset | hours | Phases 16–34, MI-1…MI-9, from the tabular models to the test reads |
| **E. Rebuild the dataset from raw** | raw FFWC/NASA/OSM/SRTM archives | days | data acquisition and QC; raw archives are not shipped in this repository |

A–C are the paper-reproduction path. D is documented for audit. E is outside the repo
because the raw archives are large and some sources require registration.

---

## 1. Environment

```powershell
cd "D:\Flood predictor"
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# PyTorch CPU build (only needed for the DL phases):
# .venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

The exact versions that produced the frozen numbers are in `reports/FREEZE.md`
(`environment.packages`) and `requirements.txt`. The core stack is numpy 2.3.2,
pandas 3.0.6, scikit-learn 1.9.1, LightGBM, XGBoost, and PyTorch 2.14.1+cpu.

---

## 2. Verify the freeze (level A)

```powershell
.venv\Scripts\python.exe scripts\freeze_results.py --check
```

This recomputes SHA-256 for every file in the manifest and re-derives the headline
numbers from the CSVs/JSONs (D16 `val_2022` per horizon and mean; D16 test per horizon
and mean; D7 test mean; the event-analysis flags; the dataset row count). It fails if
any value or hash drifted. `reports/FREEZE.md` lists the checks and hashes.

Expected: `checks 14/14; changed files 0`.

---

## 3. Re-run the P2/P3 correction (level B)

```powershell
.venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py   # ~35 min
.venv\Scripts\python.exe scripts\mi10_report.py            # ~1 s
```

* `mi10_p2p3_rescore.py` refits **per cell** under the P2/P3 masks and writes
  `reports/p2p3_rescore/{probe,arms,d7,d7_margins}_p2p3.csv`, `summary.json`, `run.log`.
  It refuses to report if any `val_2022` reproduction gate fails (tolerance 1e-5).
* `mi10_report.py` reads `summary.json` and writes `reports/p2p3_correction.md`.
* The script asserts `test_rows_read = 0` and that no evaluation mask contains
  `split == "test"`.
* Pre-registration: `reports/p2p3_correction_protocol.md` (written before the run).

The old in-sample artefacts (`mi8_probe.csv`, `mi8_scorecard.csv`, `mi3_*.csv`) are
**not overwritten**; they remain as the historical record and are quoted only with the
correction banner.

---

## 4. Re-run the event-level analysis (level C)

```powershell
.venv\Scripts\python.exe -u scripts\mi11_event_eval.py
```

Writes `reports/event_level/{event_metrics.csv, summary.json, summary.md}`. It refits
D16 and D7 per cell for the seven development cells, builds observed danger-level
events on the non-test timeline, and measures event recall, lead time, and
false-alarm episodes. The frozen test is not read.

A quick single-cell check:

```powershell
.venv\Scripts\python.exe -u scripts\mi11_event_eval.py --smoke
```

---

## 5. Re-run the model programme (level D)

The paper's discrimination tables come from the phases below. Run them in order; all
use seed `20261008` and write to `reports/` and `figures/`.

| stage | command | what it produces |
| --- | --- | --- |
| data + features | `scripts/phase14_dataset.py` → `phase18_dataset.py` | model-ready table |
| v2 features | `scripts/mi2_features.py` | `mi_v2_features.csv` (FULL_V2 = 110) |
| MI-8 features | `scripts/mi8_features.py` | `mi8_features.csv` (FULL_V3 = 165) |
| baselines | `scripts/phase16_baseline.py` | D7 and six L1 references |
| classical | `scripts/phase20_classical.py` | LightGBM, XGBoost, RF, GB, selected |
| DL | `scripts/phase21_temporal_dl.py`, `phase22_tcn.py`, `phase23_fusion.py`, `phase24_gnn.py`, `phase25_transformer.py` | 12 DL configurations |
| fair ML–DL | `scripts/phase27_ml_vs_dl.py` | 24-model comparison on common cells |
| calibration / thresholds / errors / robustness / ablation / XAI | `phase28_calibration.py`, `phase29_thresholds.py`, `phase30_errors.py`, `phase31_robustness.py`, `phase32_ablation.py`, `phase34_xai.py` | paper's Sections V–VI |
| selection | `scripts/phase33_selection.py` | Phase-33 incumbent |
| MI sprint | `scripts/mi3_*.py`, `mi5_selection.py`, `mi6_test_read.py`, `mi7_*.py`, `mi8_*.py` | v2 features, D16, fair DL rematch, MI-8 |
| test correction | `scripts/mi9_test_read_corrected.py` | D18: D17 retracted, D16 champion |

The test scripts are **guarded** and must not be re-run for this paper. Their outputs
are frozen in `reports/mi6_summary.json`, `mi9_summary.json`, and `test_read_ledger.md`.

---

## 6. Artefact map (paper → file)

| paper element | source |
| --- | --- |
| Dataset (200,665 rows, 133 stations) | `data/processed/experimental_dataset.csv` |
| Label counts / prevalence | `reports/phase14_experimental_dataset.md`, `phase1_data_inventory.md` |
| D7 definition | `scripts/baseline_model.py` (`d7` pipeline), `reports/phase16_baseline.md` |
| ML vs DL table | `reports/phase27_ml_vs_dl.md`, `phase27_metrics.csv` |
| D16 selection | `reports/mi5_selection.md`, `models/final_v2/` |
| D16 test read | `reports/mi6_test_read.md`, `mi6_test_metrics.csv` |
| Test-read disclosure | `reports/test_read_ledger.md` |
| P2/P3 correction | `reports/p2p3_correction.md`, `reports/p2p3_rescore/` |
| Event-level table | `reports/event_level/summary.md` |
| Calibration | `reports/phase28_calibration.md`, `figures/phase28_reliability.png` |
| Robustness | `reports/phase31_robustness.md`, `figures/phase31_robustness.png` |
| Ablation | `reports/phase32_ablation.md`, `figures/phase32_ablation.png` |
| SHAP | `reports/phase34_xai.md`, `figures/phase34_shap_summary.png`, `phase34_group_attribution.png` |
| Thresholds / risk bands | `reports/phase29_thresholds.md` |
| Error analysis | `reports/phase30_errors.md` |

---

## 7. Determinism notes

* All scripts set `baseline_model.B.SEED = 20261008`; tree models use fixed
  `random_state`/`seed`; DL uses the same seed with deterministic CPU kernels.
* Thresholds are fitted on **training** rows of the same cell (`best_f1_threshold`),
  never on evaluation rows.
* `mi10_p2p3_rescore.py` checkpoints after every cell, so an interrupted run resumes
  from `*_p2p3.csv` without losing completed cells.
* Station-level bootstrap resamples use a fixed seed and a shared resample index for
  paired comparisons.
* Floating-point equality in the gates is not required; tolerances are stated in the
  scripts and in `FREEZE.md`.

---

## 8. Test-read policy (repeat of the important part)

Six opens, five predictive reads, one invalidated:

1. Phase 16 baseline (2026-10-09)
2. Phase 33 selection (2026-10-15)
3. MI-6 D16 champion read (2026-10-15/16)
4. MI-4 label statistics (2026-10-16; no predictions)
5. MI-4/D17 comparison (2026-10-21) — **invalid, retracted**
6. MI-9 correction (2026-10-21) — **reproduced MI-6 bit-for-bit; D16 champion**

The paper quotes the **MI-6** read (D16 0.77076 vs D7 0.66755) and states the
programme-level caveat in Limitations. No new read is permitted for this paper.
