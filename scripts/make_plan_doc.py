"""Build D:\\Flood predictor\\plan.doc (real .doc via Word COM if available)."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(r"D:\Flood predictor")
DOCX = OUT_DIR / "_plan.docx"
DOC = OUT_DIR / "plan.doc"

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

d = Document()
styles = d.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(11)


def H(text, level=1):
    d.add_heading(text, level=level)


def P(text, bold=False, italic=False):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def B(text):
    d.add_paragraph(text, style="List Bullet")


def N(text):
    d.add_paragraph(text, style="List Number")


def TBL(rows):
    t = d.add_table(rows=1, cols=len(rows[0]))
    t.style = "Table Grid"
    for i, c in enumerate(rows[0]):
        t.rows[0].cells[i].text = str(c)
        for p in t.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows[1:]:
        cells = t.add_row().cells
        for i, c in enumerate(row):
            cells[i].text = str(c)
    d.add_paragraph()


# ---------------------------------------------------------------- title
title = d.add_heading("Flood Probability Forecasting for Bangladesh", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = d.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Research plan: 24 / 48 / 72-hour probability of flooding at a given location\n"
                "Project folder: D:\\Flood predictor   |   Duration: 2–3 months (intensive)")
r.italic = True

# ---------------------------------------------------------------- 1
H("1. Problem definition and research question", 1)
P("Primary research question: given everything known at issue time t, what is the probability "
  "that a given location (station / river reach / district-upazila) will experience flooding "
  "within the next 24, 48 and 72 hours?")
P("This is a probabilistic, lead-time forecast — not a static flood-susceptibility map, and not "
  "the monthly water-level regression used in flood-v3. Every prediction is a calibrated "
  "probability p, which is then mapped to Low / Moderate / High / Very High risk bands with an "
  "explicit operational threshold (the threshold must be tunable, because the cost of a missed "
  "flood far exceeds the cost of a false alarm).")
P("Deliverables", bold=True)
B("Model: calibrated probability of exceedance at lead h = 24/48/72 h at each station, with SHAP explanations.")
B("Web platform: interactive Bangladesh risk map, district/upazila selection, risk levels, "
  "SHAP panel per location, alert/early-warning view for citizens and administrators.")
B("Evidence: rigorous chronological + station-out validation, false-negative analysis, "
  "comparison against strong baselines, a reproducible data pipeline, and plan.doc / final report.")

# ---------------------------------------------------------------- 2
H("2. Why the old flood-v3 dataset cannot be used", 1)
P("The flood-v3 project (C:\\Users\\user\\Desktop\\flood-v3) is kept only for methodology. Its data is unfit:")
TBL([["Problem", "Consequence for 24/48/72 h forecasting"],
     ["16,051 of 20,544 labels were ASSUMED (empty record → 0 = 'no flood')",
      "Labels are unreliable; the model would learn the reporting pattern of the form, not flooding"],
     ["Monthly time resolution", "A 72-hour forecast is physically impossible to define on monthly steps"],
     ["year_norm / bad_years_boost features", "Calendar leakage: the model reads the date, not the hydrology"],
     ["Rainfall documented in cm, actually mm", "Feature units silently wrong; feature importance is meaningless"],
     ["No lead-time framing, no warning threshold", "Cannot produce a calibrated probability or an operating point"]])
P("Carried over from flood-v3 (these were good): chronological splits, station-hold-out split, "
  "monsoon lens, calibration (Brier / ECE / reliability curves), false-negative analysis, "
  "label-uncertainty audit, DeLong + bootstrap significance tests, SHAP explanations.")

# ---------------------------------------------------------------- 3
H("3. Dataset strategy", 1)
P("Rule adopted: labels and core predictors must come from measured hydrology with a real lead time. "
  "Satellite flood maps are used as a spatial map layer and a sanity check only — they cannot "
  "support a 72-hour forecast (they are observations of a flood after it happened).")

H("3.1 Primary source — FFWC / BWDB river data (labels + core predictors)", 2)
P("API base https://api.ffwc.gov.bd/data_load (authenticated: header "
  "x-ffwc-internal-key + browser Origin/Referer; discovered from the ffwc.gov.bd bundle).")
TBL([["Endpoint", "Content", "Status"],
     ["/stations", "153 stations: id, lat/long, river, basin, district, upazilla, "
      "danger level, record high water level", "harvested → data/raw/ffwc/ffwc_stations.csv"],
     ["/observed-waterlevel-by-station-and-date/{id}/{YYYY-MM-DD}",
      "Observed 3-hourly water level; each call returns a ~40.5-day trailing window (~205 readings)",
      "harvested → data/raw/ffwc/waterlevel/{station_id}/{year}.csv"],
     ["messages/, scroller-messages/", "FFWC flood warnings/river bulletins (Bengali) — useful for "
      "warning-threshold ground truth", "available"],
     ["Archive depth", "2017 and 2019-2026 present (~10 years); 2018 and earlier return empty "
      "(verified as a genuine gap in FFWC's own archive, not a crawl bug)", "verified"]])
P("CSV is the working format: a small ingestion module (src/flood/ingest/ffwc.py) converts the "
  "JSON responses into tidy, resumable CSVs. Full history (2014-01-01 to 2026-10-06, stride 35 days) "
  "ran in 619 s for 20,655 API calls at ~26 req/s, producing 992 station-year CSVs and "
  "1,052,044 readings under data/raw/ffwc/waterlevel/.")

H("3.2 Rainfall (predictor)", 2)
B("CHIRPS 2.5° daily (ClimHaz/IHazards or UCSB) — 2001→present, free, no key: accumulated rainfall "
  "at the station and upstream over 1/3/7-day windows.")
B("ERA5-Land hourly (Copernicus CDS) — antecedent soil moisture / precipitation for baseline conditions.")
B("Optional Phase 5: NWP (ECMWF open data / GFS) rainfall forecasts — if not obtained, the honest "
  "claim is 'empirical lead-time forecast from antecedent hydrology + observed rainfall'.")

H("3.3 Geography (static features + map layer)", 2)
B("Bangladesh administrative boundaries (upazila/district, PU/UN Mapped or GADM/geoBoundaries) — for selection UI.")
B("Copernicus GLO-30 / SRTM DEM — elevation, distance to river, slope; HydroRIVERS — drainage network/order.")
B("CEGIS flood-hazard maps (1% / 2% AEP) — hazard zoning reference, not a label.")
B("ESA WorldCover (land cover), SoilGrids (soil), WorldPop (population at risk).")

H("3.4 Spatial flood observation (map layer + validation, NOT a label)", 2)
B("Global Flood Database (MODIS 2000–2018, 250 m, GEE) — 913 flood events, Bangladesh subset → map "
  "layer and independent sanity check that mapped risk matches observed floods.")
B("UNOSAT Sentinel-1 FloodAI BGD (https://unosat-rm.cern.ch/FloodAI/apps/BGD/) — recent inundation "
  "extents for the map layer and qualitative validation.")
B("GRDC / GMB discharge (Hardinge-Bridge 1969→, Bahadurabad 1976→) and Harvard GMB gridded streamflow "
  "(1951–2023) as optional independent upstream-driver features if time allows.")

# ---------------------------------------------------------------- 4
H("4. Label definition (the scientific core)", 1)
N("Issue time t (daily grid), location s (FFWC station), lead h ∈ {24, 48, 72} hours.")
N("y = 1 if observed water level at s reaches or exceeds the published FFWC Danger Level "
  "anywhere in the window (t, t + h]; else y = 0.")
N("Rows with missing / unverified observations are dropped — NEVER coerced to 0 (this is the "
  "central flaw of flood-v3).")
N("Because levels are 3-hourly, the same issue day yields three honest targets (one per lead).")
N("Store a label-confidence flag (observed vs. assumed) so every metric can be recomputed on the "
  "high-confidence subset only.")
P("Note: 'location' is a station. Stations map to district/upazila in the master table, which is "
  "how the web map will do location selection; the same model scores any point near a station.")

# ---------------------------------------------------------------- 5
H("5. Predictors", 1)
TBL([["Group", "Features"],
     ["Hydrology (core)", "Target-station water level lags 1–7 d, rise rate, distance to danger level, "
      "same-river upstream station lags, 7 d percentile"],
     ["Rainfall", "CHIRPS accumulation at target and upstream catchment for 1/3/5/7/10 d, "
      "max 24 h intensity, anomaly vs. local climatology"],
     ["Antecedent", "ERA5-Land soil moisture, prior 30 d wetness, discharge from GMB/GRDC if available"],
     ["Seasonality", "Day-of-year harmonics (sin/cos), monsoon phase — deliberately narrow to avoid "
      "the calendar leakage of flood-v3"],
     ["Static", "Elevation, slope, distance to river, drainage order, land cover, soil, flood-hazard zone, population"],
     ["Optional Phase 5", "NWP predicted rainfall for the next 0–72 h"]])

# ---------------------------------------------------------------- 6
H("6. Honest evaluation protocol", 1)
B("Baselines that must be beaten: persistence (tomorrow = today), month × station climatology, "
  "logistic regression on water-level lags, and FFWC's own operational forecast where retrievable.")
B("Splits: (a) chronological — train ≤ year X, validate X+1, test X+2…, always forward; "
  "(b) station-out — hold out whole stations to prove geographic generalisation; "
  "(c) monsoon-only test window.")
B("Metrics: PR-AUC as primary (flood events are rare), plus recall at a fixed precision, "
  "false-negative rate and estimated cost, Brier score + ECE with reliability diagrams, "
  "lead-time-stratified curves (24 vs 48 vs 72 h), and warning lead time (hours before exceedance).")
B("False-negative drill-down: every missed event is listed with its SHAP explanation and the "
  "feature values at issue time — this is required by the proposal.")
B("Uncertainty: DeLong tests vs. baselines, bootstrap CIs over seasons, label-uncertainty audit "
  "(metrics recomputed on the confident-label subset).")
B("No leakage policy: features must be computable at issue time; no future windows, no year indices, "
  "no 'bad years boost', unit tests assert units (mm) and time order.")

# ---------------------------------------------------------------- 7
H("7. Twelve-week roadmap", 1)
TBL([["Phase", "Weeks", "Work", "Gate / output"],
     ["0 — Data feasibility", "1", "Crack FFWC API (done), confirm archive depth, station inventory, "
      "danger levels; pull station + history CSVs; download CHIRPS + boundaries",
      "Go/no-go: ≥10 y of labels with a usable base rate"],
     ["1 — Label & feature pipeline", "2–3", "Build daily station-day table, labels at 3 leads, "
      "dedupe/QC, CHIRPS + static features, leakage tests",
      "Feature table with audit report and label base rates per station"],
     ["2 — Baseline models", "4", "Persistence / climatology / logistic, first LightGBM, "
      "chronological split, calibration",
      "PR-AUC and Brier beats baselines on test season"],
     ["3 — Full modelling", "5–7", "Tuning, station-out split, threshold selection for alert levels, "
      "SHAP, false-negative analysis, ablations (no-rainfall / no-upstream)",
      "Final model card with all protocol metrics"],
     ["4 — Platform", "8–10", "FastAPI service (forecast + SHAP + risk levels), interactive Leaflet map, "
      "district/upazila selection, auto-updating data, alert view",
      "Working web demo end-to-end"],
     ["5 — Optional NWP", "10–11", "Add weather-model rainfall forecasts; compare empirical vs. "
      "NWP-augmented lead time", "Optional improvement, honest comparison"],
     ["6 — Write-up", "11–12", "Final report/paper, reproducible pipeline, demo polish",
      "Report + plan.doc (this) + video/demo"]])
P("Working rule: every phase ends with a written artifact (CSV, report, notebook or working demo) "
  "so the model class and claims can change course early without losing work.")

# ---------------------------------------------------------------- 8
H("8. Repository layout", 1)
P("data/raw (untouched source data) → data/interim → data/processed (model-ready tables) | "
  "src/flood/{ingest,features,models,evaluate,api} | notebooks (exploration) | scripts "
  "(reproducible downloads) | web (front-end) | reports, figures | tests (leakage + unit tests)")
P("Ingestion scripts are parameterised and resumable; re-running them must not duplicate rows "
  "(manifest + dedupe on station/water-level timestamp).")

# ---------------------------------------------------------------- 9
H("9. Risks and mitigations", 1)
TBL([["Risk", "Mitigation"],
     ["FFWC archive shallower than expected", "Early gate in Phase 0; fallback to GRDC/GMB discharge "
      "as the hydrological driver + rainfall-based labels for the map layer"],
     ["API throttling / key rotation", "Resumable manifest crawl, low concurrency, cached raw JSON on first pass"],
     ["Few exceedance events per station (class imbalance)", "PR-AUC objective, event-based sampling, "
      "multi-station pooling with station embeddings/fixed effects"],
     ["Seasonality confounding the model", "Narrow season features, monsoon-only test, climatology baseline"],
     ["Claiming a forecast that is really nowcasting", "State explicitly whether NWP is used; "
      "report skill vs. persistence at each lead"],
     ["Map/label mismatch (station-based labels, raster map)", "Map shows station buffers + hazard zones; "
      "GFD/UNOSAT used to check that the mapped risk agrees with observed floods"]])

# ---------------------------------------------------------------- 10
H("10. Current status (Phase 0, complete for the FFWC leg)", 1)
B("FFWC API authenticated (x-ffwc-internal-key + Origin headers); 153-station master with "
  "coordinates and danger levels: data/raw/ffwc/ffwc_stations.csv.")
B("Full water-level history harvested: 1,052,044 readings, 992 station-year CSVs, 141 stations "
  "with data — 127 stations have >=1,000 readings and 106 cover >=5 years (data/interim/*_audit.csv).")
B("Label gate passed: published danger level exists for 133 stations; mean exceedance rate 4.0% "
  "(median 1.4%), rising to 6-10% on the major floodplain stations (Bahadurabad, Chandpur, "
  "Sreepur, Sirajganj) — rare but not degenerate, so PR-AUC framing is correct.")
B("Known gap: no 2018 data in FFWC's archive (empty responses confirmed for that year) - "
  "evaluation must therefore split 2017 vs 2019-2026 and never bridge the hole.")
B("Next: CHIRPS rainfall download, administrative boundaries + static geo layers, then the "
  "label/feature table (Phase 1) and baselines (Phase 2).")

d.save(str(DOCX))
print("wrote", DOCX)

try:
    import win32com.client  # type: ignore

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(DOCX), ReadOnly=False)
    if DOC.exists():
        DOC.unlink()
    doc.SaveAs2(str(DOC), FileFormat=0)  # wdFormatDocument -> real binary .doc
    doc.Close(False)
    word.Quit()
    DOCX.unlink(missing_ok=True)
    print("converted ->", DOC)
except Exception as exc:
    print("Word COM failed (%s: %s)" % (type(exc).__name__, exc))
    if DOCX.exists():
        if DOC.exists():
            DOC.unlink()
        DOCX.rename(DOC)
        print("renamed ->", DOC)
