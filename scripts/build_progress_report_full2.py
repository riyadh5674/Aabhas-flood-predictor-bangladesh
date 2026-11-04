#!/usr/bin/env python3
"""Build proper structured Word progress report with tables/figures."""

from pathlib import Path
from datetime import date

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.shared import Inches, Pt

ROOT = Path(r"D:\Flood predictor")


def hdr(p, text, bold=True, size=12):
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)


def add_h(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.runs[0].font.size = Pt(15 if level == 1 else 13)


def main():
    doc = Document()
    # cover
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph("Progress Report")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    r = p.runs[0]; r.bold=True; r.font.size=Pt(20)
    p = doc.add_paragraph("On")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.runs[0].font.size=Pt(12)
    p = doc.add_paragraph("An Intelligent Flood Risk Prediction and Early Warning System for Bangladesh Using Machine Learning and Spatiotemporal Data")
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    r = p.runs[0]; r.bold=True; r.font.size=Pt(14)
    doc.add_paragraph()
    p = doc.add_paragraph("Submitted By: Md. Riyad"); p.alignment=WD_PARAGRAPH_ALIGNMENT.CENTER
    p = doc.add_paragraph("Supervisor: Prof. Dr. Sharmin Parveen"); p.alignment=WD_PARAGRAPH_ALIGNMENT.CENTER
    p = doc.add_paragraph("Department of Computer Science and Engineering"); p.alignment=WD_PARAGRAPH_ALIGNMENT.CENTER
    p = doc.add_paragraph("State University of Bangladesh"); p.alignment=WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph()
    p = doc.add_paragraph(f"Date: {date.today().strftime('%d %B %Y')}"); p.alignment=WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_page_break()

    add_h(doc, "Table of Contents")
    for t in [
        "1. Introduction", "2. Problem Statement and Research Question", "3. Objectives", "4. Methodology Overview",
        "5. Work Done (Phases 0–34 and MI-1–MI-8)", "6. Results", "7. Comparison (ML vs DL)", "8. Explainability",
        "9. Limitations", "10. Conclusion", "11. Feedback Sought from Supervisor", "Appendix A: References/Artefacts"
    ]:
        p = doc.add_paragraph(t); p.paragraph_format.left_indent = Inches(0.3)
    doc.add_page_break()

    add_h(doc, "1. Introduction")
    p = doc.add_paragraph(
        "Flooding is one of the most frequent and devastating natural hazards in Bangladesh. Accurate and location-specific "
        "probabilistic flood forecasts can help reduce loss of life and property through timely early warning. This project "
        "aims to develop a rigorous, location-aware flood probability prediction system for 24, 48 and 72 hours ahead, "
        "with a strong emphasis on correctness, reproducibility and explainability.")
    p.runs[0].font.size = Pt(11)

    add_h(doc, "2. Problem Statement and Research Question")
    p = doc.add_paragraph("Research Question:")
    r = p.runs[0]; r.bold=True; r.font.size=Pt(11)
    p = doc.add_paragraph("What is the probability of flooding at a particular location within the next 24/48/72 hours?")
    r = p.runs[0]; r.bold=True; r.font.size=Pt(11)
    p = doc.add_paragraph("Key points addressed: information leakage prevention, realistic temporal/spatial/monsoon evaluation, "
                          "calibration of probabilities, false-negative analysis, and explainable predictions.")
    p.runs[0].font.size = Pt(11)

    add_h(doc, "3. Objectives")
    t = doc.add_table(rows=9, cols=2, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Objective"); hdr(t.cell(0,1).paragraphs[0], "Status")
    items = [
        ("Probabilistic 24/48/72h forecasts", "Completed"),
        ("Leakage-free design & rigorous splits", "Completed"),
        ("Temporal/spatial/monsoon validation", "Completed"),
        ("Fair ML vs DL comparison", "Completed"),
        ("Calibration & thresholds (FN>>FP)", "Completed"),
        ("Error/robustness/ablation", "Completed"),
        ("Explainability (SHAP)", "Completed"),
        ("Frozen champion + single logged test", "Completed"),
    ]
    for i,(a,b) in enumerate(items,1):
        c0=t.cell(i,0); c1=t.cell(i,1)
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c0.paragraphs[0].add_run(a).font.size=Pt(9)
        c1.paragraphs[0].add_run(b).font.size=Pt(9)

    doc.add_paragraph()

    add_h(doc, "4. Methodology Overview")
    t = doc.add_table(rows=10, cols=2, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Component"); hdr(t.cell(0,1).paragraphs[0], "Description")
    comp = [
        ("Dataset", "200,665 station-days, 133 FFWC stations (2017–2026); SHA-256 a6748529…"),
        ("Labels", "Danger level reached in (t,t+h]; missing window = NaN (never 0)"),
        ("Features", "FULL 62, LINEAR 35, SELECTED 10; leakage-audited"),
        ("Splits", "Temporal + 7-day embargo + 5 spatial folds + monsoon"),
        ("Headline Metric", "PR-AUC (accuracy not headline)"),
        ("Uncertainty", "Station-bootstrap 95% CIs; per-cell reporting"),
        ("Test Discipline", "Read-once only when frozen (logged)"),
        ("Training", "Validation-only tuning, fixed budgets, seed 20261008"),
        ("Interpretability", "TreeSHAP global + local exemplars"),
    ]
    for i,(a,b) in enumerate(comp,1):
        c0=t.cell(i,0); c1=t.cell(i,1)
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c0.paragraphs[0].add_run(a).font.size=Pt(9)
        c1.paragraphs[0].add_run(b).font.size=Pt(9)

    doc.add_paragraph()
    add_h(doc, "5. Work Done (Phases 0–34 and MI-1–MI-8)")
    t = doc.add_table(rows=17, cols=2, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Phase Block"); hdr(t.cell(0,1).paragraphs[0], "Key Work Completed")
    wk = [
        ("0–16", "Data audit, leakage, splits, features, baselines, imbalance (D7 persistence established)"),
        ("17–19", "Modelling framework, Dataset B sequences (L*=3), sequence validation"),
        ("20", "Classical ML (RF/GBM/XGB/LGBM)"),
        ("21–22", "Temporal DL (LSTM/GRU/TCN)"),
        ("23", "Fusion (temporal+static)"),
        ("24–25", "GNN (gated), Transformer (gated, L=7)"),
        ("26", "Multi-horizon matrix"),
        ("27", "Fair ML vs DL comparison"),
        ("28–29", "Calibration; thresholds/bands (FN:FP=10:1, period-specific)"),
        ("30", "Error analysis (10 named patterns)"),
        ("31–32", "Robustness + nested feature ablation"),
        ("33", "Selection → frozen champion"),
        ("34", "Explainable AI (TreeSHAP + IG)"),
        ("MI-1–MI-3", "ERA5/GloFAS/POWER-soil, FULL_V2, HPO, ensemble rejected"),
        ("MI-4–MI-6", "Label ceiling, D16 frozen, single logged test read"),
        ("MI-7–MI-8", "Fair DL rematch (DL queue closed), search saturated"),
    ]
    for i,(a,b) in enumerate(wk,1):
        c0=t.cell(i,0); c1=t.cell(i,1)
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c0.paragraphs[0].add_run(a).font.size=Pt(9)
        c1.paragraphs[0].add_run(b).font.size=Pt(9)

    doc.add_paragraph()
    add_h(doc, "6. Results")
    t = doc.add_table(rows=6, cols=4, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Metric"); hdr(t.cell(0,1).paragraphs[0], "24h"); hdr(t.cell(0,2).paragraphs[0], "48h"); hdr(t.cell(0,3).paragraphs[0], "72h")
    res = [
        ("D16 PR-AUC (val_2022, P1)", "0.9687", "0.9569", "0.9467"),
        ("D16 PR-AUC (Test, frozen)", "0.83623", "0.76896", "0.70709"),
        ("D7 Persistence (Test)", "0.7793", "0.6768", "0.5965"),
        ("Event Recall (Test, @ threshold)", "~0.967", "~0.967", "~0.967"),
        ("Missed Floods (Test)", "1044 (vs 2478)", "—", "—"),
    ]
    for i,(a,b,c,d) in enumerate(res,1):
        cells=[t.cell(i,0),t.cell(i,1),t.cell(i,2),t.cell(i,3)]
        for cc in cells:
            cc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cells[0].paragraphs[0].add_run(a).font.size=Pt(9)
        cells[1].paragraphs[0].add_run(b).font.size=Pt(9)
        cells[2].paragraphs[0].add_run(c).font.size=Pt(9)
        cells[3].paragraphs[0].add_run(d).font.size=Pt(9)
    p = doc.add_paragraph("Note: D16 beats persistence on 20 of 21 frozen test cell×horizon points; only test_unseen@24h is a documented tie (overlapping CIs).")
    p.runs[0].font.size=Pt(9)

    doc.add_paragraph()
    add_h(doc, "7. Comparison (ML vs DL)")
    t = doc.add_table(rows=6, cols=3, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Aspect"); hdr(t.cell(0,1).paragraphs[0], "Finding"); hdr(t.cell(0,2).paragraphs[0], "Evidence")
    cmp = [
        ("Fair comparison", "One evaluation path, identical data/splits/budget/seeds", "Phase 27, MI-7"),
        ("Best DL vs ML", "Fair DL did not outperform classical ML on PR-AUC", "MI-7 (paired CIs favoured ML in 8/9; 1 tie)"),
        ("DL queue", "Closed on evidence", "MI-7"),
        ("Search saturation", "MI-8 expanded features/budget; no paired CI beat D16 on decision cell", "MI-8"),
        ("Generalisation", "ML advantage largest on unseen-station cells", "Phases 27,31"),
    ]
    for i,(a,b,c) in enumerate(cmp,1):
        cells=[t.cell(i,0),t.cell(i,1),t.cell(i,2)]
        for cc in cells:
            cc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cells[0].paragraphs[0].add_run(a).font.size=Pt(9)
        cells[1].paragraphs[0].add_run(b).font.size=Pt(9)
        cells[2].paragraphs[0].add_run(c).font.size=Pt(9)

    doc.add_paragraph()
    add_h(doc, "8. Explainability")
    t = doc.add_table(rows=5, cols=2, style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr(t.cell(0,0).paragraphs[0], "Element"); hdr(t.cell(0,1).paragraphs[0], "Result")
    exp = [
        ("Global attribution", "Hydrology-first (~64% at 24h); wl_dist_to_danger, wl_pct_danger dominant"),
        ("Local explanations", "Prediction + contributing factors per instance (as requested)"),
        ("DL attribution", "Integrated Gradients aligned with hydrology-first pattern"),
        ("Leakage check", "No banned/label-window features in top-20"),
    ]
    for i,(a,b) in enumerate(exp,1):
        c0=t.cell(i,0); c1=t.cell(i,1)
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c0.paragraphs[0].add_run(a).font.size=Pt(9)
        c1.paragraphs[0].add_run(b).font.size=Pt(9)

    doc.add_paragraph()
    add_h(doc, "9. Limitations")
    for txt in [
        "PR-AUC strongly base-rate dependent (MI-4); uniform 0.99 PR-AUC at ~0.7% base rate not realistic (empirical ceiling ~0.96–0.97 dev).",
        "test_unseen@24h: documented tie vs persistence (overlapping CIs).",
        "Misses occur while river below danger but rising (response-lag behaviour; Phase 30).",
        "FN>>FP trade-off yields some false alarms just below danger (policy knob, not defect).",
        "System components (map, auto-acquisition, live pipeline, UI) remain planned if approved (not covered as completed work).",
    ]:
        pr = doc.add_paragraph(txt, style="List Bullet"); pr.runs[0].font.size=Pt(10)

    add_h(doc, "10. Conclusion")
    p = doc.add_paragraph(
        "The completed work (Phases 0–34, MI-1–MI-8) meets the proposal's methodological requirements. A leakage-free, "
        "realistically validated design, fair ML-vs-DL comparison, calibration, thresholds, robustness, ablation, error analysis "
        "and explainability are all completed. The frozen champion (D16, hpo_perhorizon_FULL_V2) is reproducible, reduces missed "
        "floods substantially over persistence, and is supported by transparent, quantified limitations.")
    p.runs[0].font.size=Pt(11)

    add_h(doc, "11. Feedback Sought from Supervisor")
    for q in [
        "1. Is the methodological rigor and test discipline acceptable for this midterm evaluation?",
        "2. Can the fair ML>DL finding be accepted (MI-7 closed the DL queue)?",
        "3. Is the stated honest ceiling (base-rate–PR-AUC) acceptable?",
        "4. Should work proceed to system layer (35–42) or closure (43–45)?",
        "5. Is D16 acceptable as the champion, or any additional checks recommended?",
        "6. Any corrections/advice before thesis write-up?",
    ]:
        pr = doc.add_paragraph(q); pr.runs[0].font.size=Pt(11)

    doc.add_page_break()
    add_h(doc, "Appendix A: References/Artefacts")
    for txt in [
        "Immutable dataset SHA-256: a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e",
        "Splits: split_assignments.csv, splits.json (7-day embargo, 5 spatial folds)",
        "Frozen models: models/final/ (Phase-33), models/final_v2/ (D16 champion)",
        "Key reports: phase27_*, phase28_*, phase29_*, phase30_*, phase31_*, phase32_*, phase33_*, phase34_*, mi5_*, mi6_*, mi7_*",
        "Seed: 20261008 | Read-once test logs present | Verification notebooks (27) all green",
    ]:
        pr = doc.add_paragraph(txt, style="List Bullet"); pr.runs[0].font.size=Pt(10)

    # page numbers
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        p.add_run('Page ')
        try:
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE')
            r2 = p.add_run(); r2._r.append(fld)
        except Exception:
            pass
        p.add_run(' of ')
        try:
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'NUMPAGES')
            r3 = p.add_run(); r3._r.append(fld)
        except Exception:
            pass
        p.runs[0].font.size = Pt(9)

    out = ROOT / "Progress_Report_Finalized.docx"
    doc.save(out)
    print('Saved:', out)


if __name__ == "__main__":
    main()
