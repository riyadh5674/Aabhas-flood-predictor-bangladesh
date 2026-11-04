#!/usr/bin/env python3
"""Generate FINAL full progress report for Bangladesh flood prediction project."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import docx
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.shared import Inches, Pt

ROOT = Path(r"D:\Flood predictor")


def add_cover(doc: Document):
    for _ in range(7):
        doc.add_paragraph()
    p = doc.add_paragraph("Progress Report on \"An Intelligent Flood Risk Prediction and Early Warning System for Bangladesh Using Machine Learning and Spatiotemporal Data\"")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(16)
    doc.add_paragraph()
    p = doc.add_paragraph("Submitted by")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size = Pt(12)
    p = doc.add_paragraph("Md. Riyad")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(14)
    doc.add_paragraph()
    p = doc.add_paragraph("Under the supervision of")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size = Pt(12)
    p = doc.add_paragraph("Prof. Dr. Sharmin Parveen")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(14)
    doc.add_paragraph()
    doc.add_paragraph()
    p = doc.add_paragraph("Department of Computer Science and Engineering")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size = Pt(12)
    p = doc.add_paragraph("State University of Bangladesh")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size = Pt(12)
    doc.add_paragraph()
    p = doc.add_paragraph(f"Date of Submission: {date.today().strftime('%d %B %Y')}")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size = Pt(12)
    doc.add_page_break()


def add_toc_placeholder(doc: Document):
    h = doc.add_heading("Table of Contents", level=1)
    h.runs[0].font.size = Pt(16)
    items = ["1. Executive Summary", "2. Introduction", "3. Research Question and Objectives", "4. Review of Supervisor's Guidelines",
             "5. Data Collection and Preparation", "6. Research Methodology and Rigor", "7. Modelling Work Done (Phases 0–34 & MI-1–MI-8)",
             "8. Results and Analysis", "9. Honest Negative Results and Limitations", "10. Conclusion", "11. Feedback Sought from Supervisor",
             "Appendix A: Reproducibility and Artefact Inventory"]
    for it in items:
        p = doc.add_paragraph(it)
        p.paragraph_format.left_indent = Inches(0.5)
        p.runs[0].font.size = Pt(11)
    doc.add_page_break()


def add_sections(doc: Document):
    # 1. Executive Summary
    doc.add_heading("1. Executive Summary", level=1)
    p = doc.add_paragraph(
        "This progress report presents the work completed on the development of an intelligent, location-aware flood probability "
        "forecasting and early warning system for Bangladesh. The study adopts a rigorous, reproducible research approach with "
        "strict leakage control, temporal/spatial/monsoon validation, read-once test discipline, and full traceability of artefacts. "
        "Using over 200,000 station-days from 133 FFWC stations (2017–2026), the system predicts flood probability at 24, 48 and 72 "
        "hours for each station×day using only information available at issue time. The frozen champion model (D16: hpo_perhorizon_FULL_V2, "
        "per-horizon tree ensemble on engineered features) beats the persistence baseline on 20 of 21 frozen test cell×horizon points, "
        "roughly halves missed floods, raises event recall to ~0.967, and provides calibrated probabilities with SHAP-based explanations. "
        "A fair head-to-head comparison (identical data, splits, metric code, seeds and tuning budget) showed classical ML outperformed "
        "all fair deep learning variants on PR-AUC. Negative results are documented transparently. This report focuses on completed work "
        "(Phases 0–34 and MI-1–MI-8) as requested.")
    p.runs[0].font.size = Pt(11)

    # 2. Introduction
    doc.add_heading("2. Introduction", level=1)
    p = doc.add_paragraph(
        "Flooding is a recurrent and devastating natural hazard in Bangladesh. Timely, location-specific probabilistic forecasts can "
        "support preparedness and early warning. The present work addresses this need by developing a probabilistic flood prediction "
        "framework with explicit focus on methodological rigor: correct label definition ('missing ≠ non-flood'), information-leakage "
        "prevention, realistic evaluation via temporal/spatial/monsoon splits, calibration of probabilities, error analysis of false "
        "negatives, and model explainability.")
    p.runs[0].font.size = Pt(11)

    # 3. Research Question & Objectives
    doc.add_heading("3. Research Question and Objectives", level=1)
    p = doc.add_paragraph("Research Question: What is the probability of flooding at a particular location within the next 24/48/72 hours?")
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(11)
    p = doc.add_paragraph("Objectives:")
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(11)
    for obj in [
        "To develop probabilistic flood prediction models for 24, 48 and 72 hours lead time.",
        "To ensure leakage-free, realistic evaluation (temporal, spatial, monsoon).",
        "To compare classical ML and deep learning fairly with fixed budgets and identical protocol.",
        "To calibrate probabilities and optimize thresholds with explicit cost (FN >> FP).",
        "To perform error analysis focusing on false negatives.",
        "To provide explainable predictions (SHAP/local exemplars).",
        "To freeze a reproducible champion and validate on a single logged test read."
    ]:
        pr = doc.add_paragraph(obj, style="List Bullet")
        pr.runs[0].font.size = Pt(11)

    # 4. Review of Supervisor's Guidelines
    doc.add_heading("4. Review of Supervisor's Guidelines", level=1)
    rows = [
        ["Supervisor's Guidance", "Status", "Evidence"],
        ["Temporal validation", "Done", "Phases 15,27,31,33,MI-6"],
        ["Spatial validation (unseen locations)", "Done", "Phases 15,27,31"],
        ["Monsoon-season evaluation", "Done", "Phases 16,27,31"],
        ["Missing/uncertain flood labels", "Done", "Phases 2–5,19; MI-4"],
        ["Class-imbalance analysis", "Done", "Phase 16 §10"],
        ["Calibration of probabilities", "Done", "Phase 28"],
        ["FN error analysis", "Done", "Phase 30 (10 patterns)"],
        ["Leakage/easy split checks", "Done", "Phase 13; embargoed splits; verification notebooks"],
        ["Location-based prediction", "Planned (if approved)", "Phase 35"],
        ["Bangladesh risk map", "Planned (if approved)", "Phase 36"],
        ["Automatic data acquisition", "Partial", "Phase 37"],
        ["Early-warning mechanism (prototype)", "Planned (if approved)", "Phase 39"],
        ["Explainable prediction", "Done", "Phase 34"],
        ["24/48/72h forecasting", "Done", "Phases 21–33"],
    ]
    t = doc.add_table(rows=len(rows), cols=3, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, r in enumerate(rows):
        for j, txt in enumerate(r):
            c = t.cell(i, j)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.add_run(txt).font.size = Pt(9)
            if i == 0:
                p.runs[0].bold = True

    # 5. Data
    doc.add_heading("5. Data Collection and Preparation", level=1)
    p = doc.add_paragraph(
        "Data sources: FFWC 3-hourly water levels (133 stations), NASA POWER (precipitation/temperature), SRTM/OSM (DEM/rivers), "
        "and MI-v2 additions (ERA5-Land soil moisture, GloFAS discharge, POWER soil wetness). Experimental dataset: 200,665 station-days "
        "(2017–2026), SHA-256 a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e. Label rule: flood if danger level "
        "exceeded in (t,t+h]; missing window -> NaN (never 0). Feature sets: FULL 62, LINEAR 35, SELECTED 10 with banned columns "
        "enforced. Leakage audit passed (Phase 13). Splits: temporal train/val/test with 7-day embargo, 5 spatial folds (unseen holdout), "
        "monsoon/JJAS separation. Strong base-rate drift (up to ~70:1) treated as real.")
    p.runs[0].font.size = Pt(11)

    # 6. Methodology
    doc.add_heading("6. Research Methodology and Rigor", level=1)
    for txt in [
        "Reproducibility: fixed seed 20261008, versioned artefacts, verification notebooks executed with green checks.",
        "Evaluation: PR-AUC primary (accuracy not headline); ROC-AUC, Brier, ECE, precision/recall/F1, FN count, event recall, lead time.",
        "Uncertainty: station-bootstrap 95% CIs; cells never pooled (years/seasons/stations).",
        "Test discipline: read-once only when frozen (Phase 33 and MI-6 each logged once).",
        "Calibration & thresholds: Phase 28 calibration; Phase 29 cost-optimal thresholds (FN:FP=10:1), period-specific.",
        "Fair comparison: single evaluation path; MI-7 gave DL same engineered features + equal tuning budget.",
        "Explainability: TreeSHAP global + local exemplars; IG for DL (Phase 34).",
    ]:
        pr = doc.add_paragraph(txt, style="List Bullet")
        pr.runs[0].font.size = Pt(11)

    # 7. Modelling Done
    doc.add_heading("7. Modelling Work Done (Phases 0–34 & MI-1–MI-8)", level=1)
    p = doc.add_paragraph(
        "Foundations (0–16): data audit, leakage audit, splits, baselines (D7 persistence). DL prep (17–19): registry, Dataset B "
        "sequences (L*=3). Classical ML (20): RF/GBM/XGB/LGBM. Temporal DL (21–22): LSTM/GRU/TCN. Fusion (23): static context. "
        "GNN (24, gated PASS) and Transformer (25, gated PASS, L=7). Multi-horizon (26), fair ML-vs-DL (27), calibration (28), "
        "thresholds/bands (29), error analysis (30), robustness (31), ablation (32), selection (33, D16 frozen), XAI (34). MI-1–MI-8: "
        "ERA5/GloFAS/POWER-soil, FULL_V2, HPO, ensemble rejected, label ceiling (0.96–0.97 dev), D16 frozen, single test read, "
        "fair DL rematch (DL queue closed), MI-8 search saturated (champion unchanged).")
    p.runs[0].font.size = Pt(11)

    # 8. Results
    doc.add_heading("8. Results and Analysis", level=1)
    p = doc.add_paragraph(
        "Frozen development (D16, P1 val_2022): horizon-mean PR-AUC 0.95743 (24h 0.9687, 48h 0.9569, 72h 0.9467). Frozen test (single "
        "logged read, ≥2023): horizon-mean PR-AUC 0.77076 (24h 0.83623, 48h 0.76896, 72h 0.70709) vs D7 0.66755 and Phase-33 0.75601. "
        "D16 beats D7 on 20/21 test cell×horizon points (only test_unseen@24h is a documented tie). Operational: missed floods 2,478→1,044 "
        "(−58%), mean row recall 0.769→0.894, event recall 0.901→0.967 at FN:FP=10:1 (period-specific). Calibration strong; risk bands "
        "yield ~90% observed flood rate in top band. Fair comparison: tree ensembles lead every horizon; DL gap largest on unseen stations "
        "but CIs show separation only where quantified. SHAP confirms hydrology-first (~64% at 24h): wl_dist_to_danger, wl_pct_danger "
        "dominant; local exemplars show prediction + contributing factors.")
    p.runs[0].font.size = Pt(11)

    # 9. Negative results
    doc.add_heading("9. Honest Negative Results and Limitations", level=1)
    for txt in [
        "Fair DL rematch (MI-7): under equal conditions, no fair DL variant outperformed classical ML on PR-AUC; paired CIs favoured ML in 8/9 (1 tie). DL queue closed.",
        "MI-8: expanded features/search gave no paired CI excluding zero vs D16 on pre-registered decision cell (search saturated); champion unchanged.",
        "Test_unseen@24h: tie vs D7 (overlapping CIs) — documented limitation.",
        "Base-rate effect: PR-AUC strongly base-rate dependent (MI-4); uniform 0.99 PR-AUC at ~0.7% base rate not realistic; empirical ceiling ~0.96–0.97 (dev).",
        "Error behaviour: misses occur while river below danger but rising (response-lag); FN≫FP trade-off produces alarms just below danger.",
    ]:
        pr = doc.add_paragraph(txt, style="List Bullet")
        pr.runs[0].font.size = Pt(11)

    # 10. Conclusion
    doc.add_heading("10. Conclusion", level=1)
    p = doc.add_paragraph(
        "All completed work meets the methodological requirements from the proposal. Leakage-free design, realistic splits, read-once test, "
        "calibration, thresholds, error/robustness/ablation analyses, XAI and fair ML-vs-DL comparison have been completed. D16 is frozen, "
        "reproducible, and validated with documented gains over persistence and transparent limitations. The project is ready for supervisor "
        "feedback on next steps.")
    p.runs[0].font.size = Pt(11)

    # 11. Feedback
    doc.add_heading("11. Feedback Sought from Supervisor", level=1)
    for q in [
        "1. Is the methodological rigor (leakage audit, embargoed splits, read-once test, per-cell reporting, reproducibility) acceptable?",
        "2. Can the 'trees outperform fair DL' verdict be accepted as rigorous (MI-7 closed DL queue)?",
        "3. Is the honest ceiling (~0.96–0.97 dev, ~0.77 test horizon-mean) acceptable given base-rate dependence?",
        "4. Should the project proceed to system layer (Phases 35–42) or focus on closure (Phases 43–45)?",
        "5. Is D16 (hpo_perhorizon_FULL_V2) acceptable, or any additional checks before moving forward?",
        "6. Any specific corrections/advice before thesis write-up?",
    ]:
        pr = doc.add_paragraph(q)
        pr.runs[0].font.size = Pt(11)

    doc.add_page_break()

    # Appendix
    doc.add_heading("Appendix A: Reproducibility and Artefact Inventory", level=1)
    for txt in [
        "Root: D:\\Flood predictor (all reproducible).",
        "Immutable experimental dataset SHA-256: a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e",
        "Splits: split_assignments.csv + splits.json (7-day embargo, 5 spatial folds).",
        "Sequences: sequences_L{3,7,14,30}.npy + masks (P19-M1).",
        "Scripts: ~81 Python files; verification notebooks: 27 (all green).",
        "Reports/figures: 154 / 43. Models: final/ (Phase-33), final_v2/ (D16 champion), mi/ HPO JSONs.",
        "Seed: 20261008; validation-only tuning; read-once test logs present.",
        "DoD: code + run log + report + checklist per phase.",
    ]:
        pr = doc.add_paragraph(txt, style="List Bullet")
        pr.runs[0].font.size = Pt(10)


def add_page_numbers(doc: Document):
    # Add footer with page numbers
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run()
    try:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        fld = OxmlElement('w:fldSimple')
        fld.set(qn('w:instr'), 'PAGE')
        run._r.append(fld)
    except Exception:
        run.add_text('Page')
    p.runs[0].font.size = Pt(10)


def main():
    doc = Document()
    add_cover(doc)
    add_toc_placeholder(doc)
    add_sections(doc)
    # add page numbers to all sections
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run = p.add_run('Page ')
        try:
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            fld = OxmlElement('w:fldSimple')
            fld.set(qn('w:instr'), 'PAGE')
            run2 = p.add_run()
            run2._r.append(fld)
        except Exception:
            pass
        p.add_run(' of ')
        try:
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            fld = OxmlElement('w:fldSimple')
            fld.set(qn('w:instr'), 'NUMPAGES')
            run3 = p.add_run()
            run3._r.append(fld)
        except Exception:
            pass
        p.runs[0].font.size = Pt(9)

    out = ROOT / "Progress_Report_Final.docx"
    doc.save(out)
    print('Saved:', out)


if __name__ == "__main__":
    main()
