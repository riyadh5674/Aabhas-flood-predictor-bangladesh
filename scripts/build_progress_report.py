#!/usr/bin/env python3
"""Generate a Word progress report for the Bangladesh flood prediction project."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import docx
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt

# Hardcoded (project discipline) root
ROOT = Path(r"D:\Flood predictor")


def add_title_page(doc: Document, cover: dict) -> None:
    doc.add_paragraph()
    for _ in range(6):
        doc.add_paragraph()

    p = doc.add_paragraph(cover["title"])
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(20)

    doc.add_paragraph()
    doc.add_paragraph()

    meta = [
        f"Prepared By: {cover['author']}",
        f"Supervisor: {cover['supervisor']}",
        f"Institution: {cover['institution']}",
        f"Department: {cover['department']}",
    ]
    for m in meta:
        pr = doc.add_paragraph(m)
        pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pr.runs[0].font.size = Pt(14)

    doc.add_paragraph()
    doc.add_paragraph()
    pd = doc.add_paragraph(f"Date: {date.today().strftime('%d %B %Y')}")
    pd.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pd.runs[0].font.size = Pt(14)

    doc.add_page_break()


def add_executive_summary(doc: Document) -> None:
    h = doc.add_heading("1. Executive Summary", level=1)
    h.runs[0].font.size = Pt(16)

    text = (
        "This is a midterm progress report on the intelligent, location-aware flood-probability forecasting system "
        "for Bangladesh. The work completed so far follows strict research discipline: leakage-free design, temporal, "
        "spatial and monsoon-season validation, a read-once test policy, and full reproducibility with seeds and SHA-256 "
        "verification. Using over 200,000 station-days from 133 FFWC stations (2017–2026) with water levels, "
        "meteorological data, and geographical features, the project has developed, evaluated and frozen probabilistic "
        "models for 24-, 48- and 72-hour lead times. The selected champion (D16: hpo_perhorizon_FULL_V2, a per-horizon "
        "tree ensemble on the engineered feature set) beats the persistence baseline (wl_pct_danger alone) on 20 of 21 "
        "frozen test cell×horizon points, roughly halves missed floods, and achieves event recall of ~0.967 with well-"
        "calibrated probabilities and SHAP-based explainability. Deep learning was evaluated in a fair head-to-head "
        "(identical data, splits, metric code, seeds and tuning budget) and the result is reported honestly: no fair DL "
        "variant outperformed classical ML on the headline metric (PR-AUC). This report focuses on completed work (Phases "
        "0–34 and MI-1–MI-8)."
    )
    p = doc.add_paragraph(text)
    p.runs[0].font.size = Pt(11)


def add_research_question(doc: Document) -> None:
    doc.add_heading("2. Research Question and Scope", level=1)
    pq = doc.add_paragraph(
        "Research Question: What is the probability of flooding at a particular location within the next 24/48/72 hours?"
    )
    pq.runs[0].bold = True
    pq.runs[0].font.size = Pt(11)

    p = doc.add_paragraph(
        "The scope is probabilistic early-warning: each prediction is the flood probability for a specific FFWC station "
        "on a specific issue day (one station × day), produced at issue time t (24:00 UTC) using only data available at "
        "or before t. Labels are defined as reaching the official FFWC danger level within the next 24, 48 or 72 hours; "
        "missing observations are treated as NaN (never 0). The headline metric is PR-AUC (accuracy is not used as a headline "
        "due to the severe class imbalance). Models are trained and reported separately per horizon (24h/48h/72h) and are "
        "evaluated under temporal, spatial and monsoon cells, never pooled."
    )
    p.runs[0].font.size = Pt(11)


def add_response_table(doc: Document) -> None:
    doc.add_heading("3. Response to Supervisor's Guidance (from Proposal)", level=1)
    rows = [
        ["Proposal guidance", "Status", "Evidence"],
        ["Temporal validation (train earlier, test later)", "Done", "Phases 15, 27, 31, 33, MI-6"],
        ["Spatial validation (unseen locations)", "Done", "Phases 15, 27, 31"],
        ["Monsoon-season evaluation", "Done", "Phases 16, 27, 31"],
        ["Missing/uncertain flood labels", "Done", "Phases 2–5, 19; MI-4 label audit"],
        ["Class-imbalance analysis", "Done", "Phase 16 §10; class_imbalance.json"],
        ["Calibration of predicted probabilities", "Done", "Phase 28"],
        ["Error analysis of false-negative flood cases", "Done", "Phase 30 (10 named patterns)"],
        ["Leakage / easy split checks", "Done", "Phase 13; embargoed splits; verification notebooks"],
        ["Location-based prediction concept", "Planned (if approved)", "Phase 35 (BLOCKED until approved)"],
        ["Interactive Bangladesh risk map", "Planned (if approved)", "Phase 36 (BLOCKED)"],
        ["Automatic data acquisition", "Partial", "Phase 37 (FFWC resumable; weather partial)"],
        ["Early-warning mechanism (prototype)", "Planned (if approved)", "Phase 39"],
        ["Explainable prediction ('86% + factors')", "Done", "Phase 34 (TreeSHAP + local exemplars)"],
        ["Forecast ahead 24/48/72h", "Done", "Phases 21–33; reported per horizon"],
    ]
    table = doc.add_table(rows=len(rows), cols=3, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, r in enumerate(rows):
        for j, txt in enumerate(r):
            cell = table.cell(i, j)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.add_run(txt).font.size = Pt(9)
            if i == 0:
                p.runs[0].bold = True


def add_data_foundation(doc: Document) -> None:
    doc.add_heading("4. Data Foundation and Label Definition", level=1)

    p1 = doc.add_paragraph(
        "The dataset integrates FFWC 3-hourly water-level readings, NASA POWER reanalysis (daily precipitation and "
        "temperature), SRTM DEM/OSM river geometry, and supplementary MI-v2 sources (ERA5-Land soil moisture, GloFAS "
        "discharge, POWER soil wetness). The experimental table contains 200,665 station-days across 133 stations from "
        "2017–2026. Labels label_24h/48h/72h are set to 1 only when any observation in (t, t+h] reaches or exceeds the "
        "official FFWC danger level; if the window is not fully observed, the label remains NaN and is never coerced to 0. "
        "This distinction ('missing ≠ non-flood') was verified in Phase 4–5."
    )
    p1.runs[0].font.size = Pt(11)

    p2 = doc.add_paragraph(
        "Feature engineering produced three validated sets (FULL 62, LINEAR 35, SELECTED 10) with a strict ban on "
        "label-window and identity columns. A leakage audit (Phase 13) verified no future information enters features. "
        "Splits enforce temporal ordering, 7-day embargo around split boundaries, 5 spatial folds with an explicit unseen-"
        "station holdout, and monsoon/non-monsoon separation. Base rates vary strongly by split and year (up to ~70:1 "
        "imbalance in the test period) — a property treated as real-world drift, not artificially balanced across splits."
    )
    p2.runs[0].font.size = Pt(11)


def add_methodology(doc: Document) -> None:
    doc.add_heading("5. Methodology and Research Rigor", level=1)

    p1 = doc.add_paragraph(
        "Methodological discipline was maintained throughout. Leakage checks, split integrity, sequence construction and "
        "verification were implemented as executable checks and re-verified by generated notebooks (all passing). The "
        "experimental dataset is immutable and its SHA-256 (a6748529…) is recorded in manifests, reports and verification "
        "notebooks. All phases follow a code + run log + report + checklist definition of done. Tuning is validation-only "
        "with fixed budgets, seeds are fixed (20261008), and every run is reproducible."
    )
    p1.runs[0].font.size = Pt(11)

    p2 = doc.add_paragraph(
        "Evaluation follows the pre-registered protocol: PR-AUC is primary; ROC-AUC, Brier, ECE, precision/recall/F1, "
        "false negatives, event recall and warning lead time are reported. Results are reported per cell (never pooled) "
        "across years, seasons and station groups (D8). Uncertainty is quantified via station-bootstrap 95% CIs (100 "
        "resamples). The test period (≥2023) is read only when a model is frozen; Phase 33 and MI-6 each performed exactly "
        "one logged test read for their respective champions."
    )
    p2.runs[0].font.size = Pt(11)


def add_modelling(doc: Document) -> None:
    doc.add_heading("6. Modelling Programme (Completed Work)", level=1)

    p1 = doc.add_paragraph(
        "The modelling ladder progressed systematically (Phases 17–34). Classical ML baselines (Random Forest, Gradient "
        "Boosting, XGBoost, LightGBM) were established on engineered features (Phase 20). Deep temporal models (LSTM/GRU/TCN) "
        "were trained on sequence windows with length sweep (L*=3 selected, Phase 21–22). Feature fusion tested static "
        "geography (Phase 23), followed by gated advanced models: graph-based spatiotemporal (GNN, gate PASS, narrow, Phase 24) "
        "and temporal Transformer (gate PASS, narrow, L=7, Phase 25). A multi-horizon matrix integrated all families (Phase 26). "
        "A fair ML-vs-DL comparison (one evaluation path, identical conditions, fresh P1 re-score bit-identical) was completed "
        "(Phase 27). Probability calibration (Phase 28), cost-optimal thresholds with FN≫FP (Phase 29), error analysis with ten "
        "named patterns (Phase 30), robustness under stressors (Phase 31), nested feature-group ablation (Phase 32), final "
        "selection (Phase 33), and explainable AI (TreeSHAP + integrated gradients, Phase 34) completed the research phases."
    )
    p1.runs[0].font.size = Pt(11)

    p2 = doc.add_paragraph(
        "The Model Improvement Programme (MI-1–MI-8) expanded inputs (ERA5-Land, GloFAS, POWER soil), built FULL_V2 (110 "
        "features), ran per-horizon HPO, rejected ensembling, quantified the empirical PR-AUC ceiling (~0.96–0.97 on "
        "development cells given label/imbalance, MI-4), and froze D16 (hpo_perhorizon_FULL_V2, per-horizon lightgbm/lgbm/xgb) "
        "in MI-5 with a single logged test read in MI-6. MI-7 conducted a fair deep-learning rematch giving DL the same "
        "engineered features and equal tuning budget; paired CIs favoured ML in 8 of 9 comparisons (the exception straddled "
        "zero). MI-8 extended feature set and search budget but did not beat the champion on the pre-registered decision cell "
        "(no paired CI excluded zero); the champion remained D16."
    )
    p2.runs[0].font.size = Pt(11)


def add_key_results(doc: Document) -> None:
    doc.add_heading("7. Key Results", level=1)

    p1 = doc.add_paragraph(
        "On frozen development evidence, D16 achieves horizon-mean P1 val_2022 PR-AUC 0.95743 (24h 0.9687, 48h 0.9569, 72h "
        "0.9467). On the single frozen test read (≥2023), D16 attains horizon-mean test PR-AUC 0.77076 (24h 0.83623, 48h "
        "0.76896, 72h 0.70709) versus D7 persistence 0.66755 and Phase-33 xgboost 0.75601. It beats persistence on 20 of 21 "
        "test cell×horizon points, with the sole non-win being test_unseen@24h (a documented tie within overlapping CIs). "
        "At the cost-optimal thresholds (FN:FP=10:1, period-specific), missed floods reduced from 2,478 (persistence) to 1,044, "
        "mean row recall rose from 0.769 to 0.894, and event recall rose from 0.901 to 0.967. Calibration is strong (low OOF "
        "ECE) and risk bands derived from the cost objective yield a very high top band (observed flood rate ~90%, ~14× lift). "
        "Global SHAP (Phase 34) shows hydrology-first attribution (~64% at 24h), dominated by wl_dist_to_danger and "
        "wl_pct_danger, with local explanations matching the requested 'probability + contributing factors' format."
    )
    p1.runs[0].font.size = Pt(11)

    p2 = doc.add_paragraph(
        "Fair comparison shows engineered-feature tree ensembles outperform sequence/DL families across horizons; the gap is "
        "largest on unseen-station cells (generalisation) while DL remains well calibrated. The finding is reported with "
        "uncertainty (bootstrap CIs) and without over-claiming statistical separation where intervals overlap."
    )
    p2.runs[0].font.size = Pt(11)


def add_negatives(doc: Document) -> None:
    doc.add_heading("8. Honest Negative Results and Limitations", level=1)

    p1 = doc.add_paragraph(
        "The project treats negative results as scientifically valuable. Key documented points: (1) Fair DL (MI-7) did not reach "
        "classical ML on the headline PR-AUC under equal data/features/budget; paired station-bootstrap CIs favoured ML in 8/9 "
        "comparisons (1 tie). (2) The deep-learning queue was closed on that evidence. (3) MI-8 expanded features and search "
        "but produced no paired CI excluding zero against D16 on the pre-registered decision cell (search saturated), so the "
        "champion remained unchanged. (4) On the frozen test, test_unseen@24h is a tie (−0.0020 vs persistence, overlapping CIs) "
        "and is reported as a limitation. (5) PR-AUC is strongly base-rate dependent (MI-4): a uniform 0.99 PR-AUC at ~0.7% base "
        "rate is not realistically achievable; the empirical ceiling is ~0.96–0.97 on development cells. (6) Error analysis "
        "(Phase 30) identifies misses occurring while rivers are still below danger but rising (response-lag behaviour) and the "
        "FN≫FP policy trade-off (false alarms concentrated just below danger). These limitations are quantified, named, and not "
        "omitted."
    )
    p1.runs[0].font.size = Pt(11)


def add_conclusions(doc: Document) -> None:
    doc.add_heading("9. Conclusions", level=1)
    p = doc.add_paragraph(
        "The work completed to date meets the methodological requirements set out in the proposal. A leakage-free, "
        "temporally and spatially validated experimental design supports credible conclusions. The champion model (D16, "
        "hpo_perhorizon_FULL_V2) delivers strong, calibrated probabilistic forecasts at 24/48/72h, materially reduces missed "
        "floods versus persistence, and provides explainable outputs. The ML vs DL comparison was conducted fairly and reported "
        "honestly with documented negative results. The project is ready for supervisor feedback on whether to proceed toward "
        "the system layer (with the noted scope: completed research work only is covered in this report)."
    )
    p.runs[0].font.size = Pt(11)


def add_feedback(doc: Document) -> None:
    doc.add_heading("10. Feedback Sought from Supervisor", level=1)
    qs = [
        "1. Is the methodological rigor (leakage audit, embargoed splits, read-once test, per-cell reporting, reproducibility) acceptable for this midterm evaluation?",
        "2. Can the 'trees outperform fair DL' verdict be accepted as a rigorous finding (MI-7 closed the deep-learning queue) given the equal-budget fair comparison?",
        "3. Given the base-rate–PR-AUC relationship and MI-4/8 evidence, is the stated honest ceiling (~0.96–0.97 on development cells, ~0.77 horizon-mean on the frozen test) acceptable?",
        "4. Should the project proceed to the system layer (Phases 35–42) as proposed, or focus first on closure (Phases 43–45: final evaluation, reproducibility, thesis)?",
        "5. Is the selected champion (D16, hpo_perhorizon_FULL_V2) acceptable, or are there any additional checks recommended before moving forward?",
        "6. Are there any specific corrections, clarifications or advice regarding the completed work before starting the thesis write-up?",
    ]
    for q in qs:
        p = doc.add_paragraph(q)
        p.runs[0].font.size = Pt(11)


def add_appendix(doc: Document) -> None:
    doc.add_heading("Appendix A: Reproducibility and Artefact Summary", level=1)
    items = [
        "Project root: D:\\Flood predictor (hard-wired paths; all phases reproducible from this root)",
        "Core datasets: experimental_dataset.csv (200,665 rows, 73 cols), immutable SHA-256 a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e",
        "Splits: split_assignments.csv + splits.json (temporal/spatial/season, 7-day embargo)",
        "Sequences: sequences_L{3,7,14,30}.npy with sequence_mask_L*.npy (P19-M1 keep-all+mask)",
        "Scripts: ~81 Python files (pipeline entry points: baseline_model, phase20_classical_ml, train_sequence/fusion/gnn/transformer, evaluate_all, phase28–34, MI-1–MI-8)",
        "Notebooks (verification): 27 notebooks (all executed with checks green)",
        "Reports/figures: 154 reports, 43 figures",
        "Models: phase20–25 checkpoints; final/ (Phase-33 xgboost) and final_v2/ (D16 hpo_perhorizon_FULL_V2 champion)",
        "Discipline: seed 20261008, validation-only tuning, read-once test logs (phase33_test_read_log.json, mi6_test_read_log.json), definition-of-done per phase",
    ]
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        p.runs[0].font.size = Pt(10)


def main() -> None:
    cover = {
        "title": "Progress Report — An Intelligent Flood Risk Prediction and Early Warning System for Bangladesh Using Machine Learning and Spatiotemporal Data",
        "author": "Md. Riyad",
        "supervisor": "Prof. Dr. Sharmin Parveen",
        "institution": "State University of Bangladesh",
        "department": "Department of CSE",
    }

    doc = Document()
    add_title_page(doc, cover)
    add_executive_summary(doc)
    add_research_question(doc)
    add_response_table(doc)
    add_data_foundation(doc)
    add_methodology(doc)
    add_modelling(doc)
    add_key_results(doc)
    add_negatives(doc)
    add_conclusions(doc)
    add_feedback(doc)
    add_appendix(doc)

    out = ROOT / "Progress_Report_Midterm.docx"
    doc.save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
