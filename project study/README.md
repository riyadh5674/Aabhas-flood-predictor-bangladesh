# Project Study — a beginner-friendly guide to the flood-forecasting project

This folder explains, **phase by phase in simple English**, how the Bangladesh
flood-probability model was built — what we did, *why*, and what the real numbers were.
Each file is self-contained and uses the same structure:

> **60-second version → where it sits → words glossary → what we did → problems & solutions →
> files → verdict.**

Every claim is tied to a real report, script, table or notebook in the repository, and the
honest limitations are stated (not hidden). Numbers come from the frozen artefacts — the
**test period (≥ 2023) is never used to make a decision**.

---

## Read in order

| # | Study file | Covers | One-line idea |
|---|---|---|---|
| 1 | `Phase_01_Data_Inventory.md` | Phase 1 | What data do we even have? Stations, sources, coverage. |
| 2 | `Phase_02_03_Data_Quality_and_EDA.md` | Phases 2–3 | Cleaning, quality checks, first look at the data. |
| 3 | `Phase_04_05_Target_Investigation_and_EDA.md` | Phases 4–5 | What exactly are we predicting, and how rare is it? |
| 4 | `Phase_06_09_Temporal_Spatial_Relationships_and_Redundancy.md` | Phases 6–9 | Time patterns, space patterns, redundant columns. |
| 5 | `Phase_10_12_Feature_Engineering_and_Selection.md` | Phases 10–12 | Building features, then choosing the best ones. |
| 6 | `Phase_13_Leakage_Audit.md` | Phase 13 | Proving no future information sneaks into the model. |
| 7 | `Phase_14_15_Experimental_Dataset_and_Splits.md` | Phases 14–15 | The frozen table + train/validate/test splits. |
| 8 | `Phase_16_Baseline_Models.md` | Phase 16 | How good is "simple"? (persistence bar, imbalance). |
| 9 | `Phase_17_Experimental_Framework.md` | Phase 17 | Writing the rulebook before the race. |
| 10 | `Phase_18_19_Dataset_B_and_Sequence_Validation.md` | Phases 18–19 | Building & proving the deep-learning inputs. |
| 11 | `Phase_20_Classical_ML.md` | Phase 20 | The workhorse models — XGBoost wins. |
| 12 | `Phase_21_22_Temporal_Deep_Learning.md` | Phases 21–22 | LSTM / GRU / TCN on day-sequences. |
| 13 | `Phase_23_Feature_Fusion.md` | Phase 23 | Does knowing *where* help? (mostly no). |
| 14 | `Phase_24_25_Gated_Graph_and_Transformer.md` | Phases 24–25 | Gated GNN and Transformer — small, honest gains. |
| 15 | `Phase_26_27_MultiHorizon_and_ML_vs_DL.md` | Phases 26–27 | The fair head-to-head: ML beats DL. |
| 16 | `Phase_28_29_Calibration_and_Thresholds.md` | Phases 28–29 | Honest probabilities → flood warnings. |
| 17 | `Phase_30_Error_Analysis.md` | Phase 30 | Where we still miss, and why (response lag). |
| 18 | `Phase_31_32_Robustness_and_Ablation.md` | Phases 31–32 | Stress tests + which features carry the skill. |
| 19 | `Phase_33_Selection_and_Frozen_Test.md` | Phase 33 | Pick the champion, open the test **once**. |

---

## The story in five sentences

1. The team assembled decades of rainfall, water-level and geography data and audited it hard
   against leakage.
2. Simple models were barely better than "today's water level" — so **persistence became the bar**
   every model had to beat.
3. Many model families were compared **fairly** (one grid, fixed budgets, validation only), and
   the **gradient-boosting table model** won.
4. Deep learning never beat it on the headline metric, though it helped a little at longer lead
   times on unseen stations — reported honestly at ~80× the cost.
5. On the frozen test, the champion **halved the missed floods** and raised event recall to
   **0.967** — and openly admitted the one cell where it only ties persistence.

---

## Ground rules the whole project follows

- **PR-AUC is the headline metric**, never accuracy (accuracy is meaningless at a 0.7 % base rate).
- **Never pool** years, seasons or station groups — every number is reported per cell (D8).
- **The test set is opened once, at the end**, and never used to choose anything.
- **Validation-only tuning**, with budgets fixed before results are seen.
- **Limitations are stated**: ties, miscalibration, NaN-heavy inputs, transfer losses.

---

## What comes next (outside this study series)

- **Phase 34 — Explainable AI** (needs `pip install shap`).
- **MI-v2 improvement sprint** — a separate, post-freeze effort (ERA5-Land soil moisture, GloFAS
  discharge, NASA POWER soil wetness, hyper-parameter search, probability ensembling) that
  produces `models/final_v2/` without ever re-reading the frozen test.
