"""Build notebooks/08_baseline_models.ipynb (workflow Phase 16 + checklist stage 29).

Phase 16 (body step 28): establish simple baselines before sophisticated ML.
Stage 29: class-imbalance analysis - Phase 15 handed it forward as "belongs with
Phase 16"; phase0 stage 29 was PARTIAL ("formal analysis pending").

The baselines themselves are fit by scripts/baseline_model.py (312 s: six models x
three horizons x eight development tasks). This notebook does NOT refit anything - it
verifies the Phase-16 contract from the written outputs, then performs the
class-imbalance analysis that closes stage 29.

Verification checks S1-S7:
    S1  input contract: Phase-14 table SHA-256, Phase-15 split table, Phase-12 feature
        sets, Phase-16 output files
    S2  split contract: mask sizes, no shared dates, holdout rows never trainable
    S3  metric-file structure: 8 development tasks x 6 models x 3 horizons x 3 seasons
        = 432 rows; test file 7 cells x 6 x 3 x 3 = 378 rows; no duplicate cells
    S4  arithmetic: tp+fp+fn+tn == n, recall/precision/specificity recomputed, and
        n / base_rate / train_rows identical across the six models of a cell
        (every model saw exactly the same rows)
    S5  prevalence identity: climatology PR-AUC == cell prevalence, ROC-AUC == 0.5,
        recall == 0
    S6  thresholds: exactly one threshold per (task, model, horizon) - train-derived,
        reused unchanged in every eval cell and in the test file
    S7  D7 + isolation: persistence PR-AUC == average precision of the raw
        wl_pct_danger score (the logistic link is monotone); the development metrics
        file contains no test cell and the test file carries only the P1 training set

Class-imbalance checks I1-I5 (stage 29):
    I1  class distribution + imbalance ratio per split, phase0 section 4 re-checked
    I2  drift: per-year ratios (R1), monsoon vs non-monsoon (D1), split x season cells
    I3  spatial concentration of positives, zero-positive stations (R2/R3 re-measured)
    I4  minority-class performance: accuracy trap, ROC-vs-PR gap, threshold transfer
    I5  metric policy for Phases 17+ (PR-AUC primary, accuracy never a headline) +
        writes data/processed/class_imbalance.json

Build and run:

    python scripts/build_phase16_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\08_baseline_models.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "08_baseline_models.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 16 - Establish baseline models (+ checklist stage 29: class imbalance)

| | |
| --- | --- |
| **Workflow** | `flood prediction workflow.md` **Phase 16 - Establish baseline models**: *before sophisticated ML, establish simple baselines* -> `28` baseline models; class-imbalance analysis is checklist **stage 29** |
| **Checklist #** | 29 (class-imbalance analysis - PARTIAL, closed here) + 30 (baseline models) |
| **Fit by** | `scripts/baseline_model.py` -> `scripts/phase16_run.log` (exit 0, 312 s) and `reports/phase16_baseline.md` |
| **Input** | `data/processed/experimental_dataset.csv` (Phase 14) + `split_assignments.csv` (Phase 15) + `feature_sets.json` (Phase 12) |
| **Outputs read** | `reports/phase16_baseline_metrics.csv` (432 rows) · `phase16_test_metrics.csv` (378) · `phase16_test_per_station.csv` (3,612) |
| **Output written** | `data/processed/class_imbalance.json` (stage-29 evidence) · `figures/phase16_class_imbalance.png` |
| **Binding decisions** | **D1** monsoon = JJAS · **D2** all three horizons · **D7** persistence bar · **D8** no pooled headline · **R1** base-rate drift · **R4** imbalance 15-19:1, PR-AUC primary, resampling inside the training fold only |

Twelve check groups, **S1-S7** (Phase-16 contract) and **I1-I5** (stage 29), 59 checks
in total. **No model is refit here** - the notebook verifies what the script wrote and
then analyses the class imbalance. The frozen test file is read for reporting only;
every threshold it carries was derived from training predictions (S6), and no decision
in this notebook selects anything from it."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
TBL = PROC / "experimental_dataset.csv"
SPLITS = PROC / "split_assignments.csv"
FSETS = PROC / "feature_sets.json"
MET = REPORTS / "phase16_baseline_metrics.csv"
MET_TEST = REPORTS / "phase16_test_metrics.csv"
PER_STATION = REPORTS / "phase16_test_per_station.csv"
MANIFEST = PROC / "class_imbalance.json"
REPORT_MD = REPORTS / "phase16_baseline.md"

TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
MODELS = ["climatology", "persistence", "logreg_linear", "logreg_linear_no_up",
          "logreg_selected", "tree_depth4"]
REAL = [m for m in MODELS if m != "climatology"]
SEASONS = ["all", "monsoon", "non-monsoon"]
SEED = 20261008


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


RES, WARN = [], []


def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid


def warn(cid, detail):
    WARN.append({"id": cid, "note": detail})
    print(f"[WARN] {cid}: {detail}")


t0 = time.time()
TBL_SHA = sha256(TBL)
exp = pd.read_csv(TBL, parse_dates=["date"])
spl = pd.read_csv(SPLITS, parse_dates=["date"])
d = exp.merge(spl, on=["station_id", "date"], validate="one_to_one")
st = pd.read_csv(PROC.parent / "raw" / "ffwc" / "ffwc_stations.csv",
                 usecols=["id", "name"]).rename(
    columns={"id": "station_id", "name": "station_name"})
d = d.merge(st, on="station_id", how="left", validate="many_to_one")
fsets = json.loads(FSETS.read_text(encoding="utf-8"))["sets"]
met = pd.read_csv(MET)
mtest = pd.read_csv(MET_TEST)
per_station = pd.read_csv(PER_STATION)

print(f"merged {len(d):,} rows x {d.shape[1]} cols | metrics {met.shape} | "
      f"test {mtest.shape} | per-station {per_station.shape} | sha {TBL_SHA[:16]}...")"""))

# --------------------------------------------------------------------- S1
cells.append(code(r"""# S1 - input contract: every phase this one depends on is intact
check("S1a", f"experimental_dataset.csv SHA-256 unchanged since Phase 14 "
             f"({TBL_SHA[:16]}...)", TBL_SHA == TBL_SHA_EXPECTED)
check("S1b", f"Phase-14 table {exp.shape[0]:,} x {exp.shape[1]}", exp.shape == (200665, 67))
check("S1c", f"Phase-15 split table {spl.shape[0]:,} x {spl.shape[1]}",
      spl.shape == (200665, 9))
check("S1d", f"merge 1:1 on (station_id, date) -> {len(d):,} rows", len(d) == 200665)
banned = {"station_id", "station_name", "date", "label_24h", "label_48h",
          "label_72h", "split", "spatial_fold", "cohort", "first_year",
          "station_holdout", "season", "in_embargo_window"}
check("S1e", f"feature sets FULL {len(fsets['FULL'])} / LINEAR {len(fsets['LINEAR'])} / "
             f"SELECTED {len(fsets['SELECTED'])}; no banned column in FULL",
      len(fsets["FULL"]) == 62 and len(fsets["LINEAR"]) == 35
      and len(fsets["SELECTED"]) == 10 and not (set(fsets["FULL"]) & banned))
check("S1f", f"Phase-16 outputs present: dev {met.shape[0]} rows, test {mtest.shape[0]} "
             f"rows, per-station {per_station.shape[0]} rows",
      met.shape == (432, 28) and mtest.shape == (378, 28)
      and per_station.shape == (3612, 11))"""))

# --------------------------------------------------------------------- S2
cells.append(code(r"""# S2 - split contract (Phase 15): masks used by the baseline script, rebuilt here
hold = d.station_holdout.astype(bool)
raw_hold = d.split.isin(["train", "val"]) & hold
tr = (d.split == "train") & ~hold
va = (d.split == "val") & ~hold
te = d.split == "test"

check("S2a", f"mask sizes train {tr.sum():,} / val {va.sum():,} / test {te.sum():,}",
      tr.sum() == 61991 and va.sum() == 17538 and te.sum() == 101174)
check("S2b", "zero shared dates between train, val and test",
      not (set(d.loc[tr, "date"]) & set(d.loc[va, "date"]))
      and not (set(d.loc[tr | va, "date"]) & set(d.loc[te, "date"])))
check("S2c", f"train ends {d.loc[tr, 'date'].max().date()}, val year "
             f"{sorted(set(d.loc[va, 'date'].dt.year))}, test starts "
             f"{d.loc[te, 'date'].min().date()}",
      d.loc[tr, "date"].max().year <= 2021
      and sorted(set(d.loc[va, "date"].dt.year)) == [2022]
      and d.loc[te, "date"].min().year >= 2023)
check("S2d", f"holdout stations: {hold.sum():,} rows, of which "
             f"{raw_hold.sum():,} rows / {d.loc[raw_hold, 'station_id'].nunique()} "
             f"stations sit inside split train/val (must never be trainable)",
      d.loc[hold, "station_id"].nunique() == 26
      and d.loc[raw_hold, "station_id"].nunique() == 21)"""))

# --------------------------------------------------------------------- S3
cells.append(code(r"""# S3 - the metric files have exactly the cells Phase 16 promised (D2 x D8)
want_tasks = {("P1", "train_<2022", "val_2022"),
              ("P2", "train_<=2019", "val_2020"),
              ("P2", "train_<=2020", "val_2021"),
              ("P2", "train_<=2021", "val_2022")}
want_tasks |= {("P3", f"excl_fold_{k}", f"spatial_fold_{k}") for k in (1, 2, 3, 4)}
got_tasks = set(map(tuple, met[["protocol", "train", "cell"]].drop_duplicates().values))
check("S3a", f"{len(got_tasks)} development tasks = P1 + P2(3 years) + P3(4 folds)",
      got_tasks == want_tasks)
check("S3b", "models / horizons / seasons are complete",
      set(met.model) == set(MODELS) and set(met.horizon) == set(HORIZONS)
      and set(met.season) == set(SEASONS)
      and set(mtest.model) == set(MODELS) and set(mtest.season) == set(SEASONS))
check("S3c", f"row arithmetic 8 tasks x 6 x 3 x 3 = {len(met)}; "
             f"7 test cells x 6 x 3 x 3 = {len(mtest)}",
      len(met) == 432 and len(mtest) == 378)
check("S3d", "no duplicate cell: (protocol, train, cell, season, model, horizon) unique",
      met.duplicated(["protocol", "train", "cell", "season", "model", "horizon"]).sum() == 0
      and mtest.duplicated(["train", "cell", "season", "model", "horizon"]).sum() == 0)
check("S3e", "every development cell reports all three horizons and all seasons",
      (met.groupby(["protocol", "train", "cell", "model"]).size() == 9).all()
      and (mtest.groupby(["train", "cell", "model"]).size() == 9).all())"""))

# --------------------------------------------------------------------- S4
cells.append(code(r"""# S4 - arithmetic: the confusion counts, the metrics and the shared data are consistent
for tag, df in (("dev", met), ("test", mtest)):
    s = df.tp + df.fp + df.fn + df.tn
    check(f"S4a-{tag}", f"tp+fp+fn+tn == n for all {len(df)} rows", (s == df.n).all())

    rec = df.tp / (df.tp + df.fn).replace(0, np.nan)
    pre = df.tp / (df.tp + df.fp).replace(0, np.nan)
    spe = df.tn / (df.tn + df.fp).replace(0, np.nan)
    check(f"S4b-{tag}", "recall / precision / specificity recomputed from the counts",
          np.allclose(rec, df.recall, equal_nan=True)
          and np.allclose(pre, df.precision, equal_nan=True)
          and np.allclose(spe, df.specificity, equal_nan=True))

    g = df.groupby(["train", "cell", "season", "horizon"])[["n", "base_rate"]].nunique()
    check(f"S4c-{tag}", "n and base_rate identical across the six models of a cell "
                        "- every model scored the same rows",
          (g == 1).all().all())

g = met.groupby(["protocol", "train", "horizon"]).train_rows.nunique()
check("S4d", "train_rows identical across models and across eval cells within a task",
      (g == 1).all())
p1_rows = (met[(met.protocol == "P1") & (met.train == "train_<2022")]
           .groupby("horizon").train_rows.first().to_dict())
check("S4e", f"P1 training rows per horizon {p1_rows}",
      p1_rows == {"label_24h": 61977, "label_48h": 60992, "label_72h": 60144})"""))

# --------------------------------------------------------------------- S5
cells.append(code(r"""# S5 - the climatology is the trivial reference: a constant forecast scores prevalence
for tag, df in (("dev", met), ("test", mtest)):
    c = df[df.model == "climatology"]
    diff = float((c.pr_auc - c.base_rate).abs().max())
    check(f"S5a-{tag}", f"climatology PR-AUC == cell prevalence in all {len(c)} cells "
                        f"(max |diff| {diff:.1e})", diff < 1e-12)
    check(f"S5b-{tag}", "climatology ROC-AUC == 0.5 and recall == 0 at its 0.5 "
                        "threshold (0.5 > train base rate)",
          (c.roc_auc == 0.5).all() and (c.recall == 0).all())
    key = [k for k in ("protocol", "train", "cell", "season", "horizon") if k in df.columns]
    w = df.pivot_table(index=key, columns="model", values="pr_auc")
    ratio = w["persistence"] / w["climatology"]
    check(f"S5c-{tag}", f"persistence beats climatology in all {len(ratio)} cells "
                        f"(weakest ratio {ratio.min():.1f}x)", (ratio > 1.0).all())"""))

# --------------------------------------------------------------------- S6
cells.append(code(r"""# S6 - thresholds: derived from TRAIN predictions only (phase0 test policy)
g = met.groupby(["protocol", "train", "model", "horizon"]).threshold.nunique()
check("S6a", f"one threshold per (task, model, horizon) over {len(g)} groups - "
             f"max distinct {int(g.max())}", int(g.max()) == 1)
g2 = mtest.groupby(["train", "model", "horizon"]).threshold.nunique()
check("S6b", f"test file: one threshold per (train, model, horizon), max distinct "
             f"{int(g2.max())}", int(g2.max()) == 1)
a = met.groupby(["train", "model", "horizon"]).threshold.first()
b = mtest.groupby(["train", "model", "horizon"]).threshold.first()
j = a.to_frame("dev").join(b.to_frame("test"), how="inner")
delta = float((j.dev - j.test).abs().max())
check("S6c", f"the 18 test thresholds equal the training-derived ones exactly "
             f"(max |diff| {delta:.1e}) - nothing was retuned on test", delta == 0.0)
thr = (met[(met.protocol == "P1") & (met.cell == "val_2022") & (met.season == "all")]
       .pivot_table(index="model", columns="horizon", values="threshold"))
print("\ntrain-derived thresholds (P1, used for every cell including the test file):")
print(thr.round(4).to_string())"""))

# --------------------------------------------------------------------- S7
cells.append(code(r"""# S7 - D7 persistence ranks exactly like the raw score, and test never enters development
worst = 0.0
for h in HORIZONS:
    sub = d[va].dropna(subset=[h])          # the same rows the script scored
    raw_ap = average_precision_score(sub[h], sub["wl_pct_danger"])
    csv_ap = float(met[(met.protocol == "P1") & (met.cell == "val_2022")
                       & (met.season == "all") & (met.model == "persistence")
                       & (met.horizon == h)].pr_auc.iloc[0])
    worst = max(worst, abs(raw_ap - csv_ap))
check("S7a", f"D7 persistence PR-AUC == average precision of raw wl_pct_danger on "
             f"validation 2022 (max |diff| {worst:.1e}, float32 fit noise) - the "
             f"logistic link is monotone, so the bar is the D7 score", worst < 1e-6)
check("S7b", "the development metrics file contains no test cell",
      not met.cell.str.startswith("test_").any())
check("S7c", "the test file carries only the P1 training set and the same 6 models",
      set(mtest.train) == {"train_<2022"} and set(mtest.model) == set(MODELS))
check("S7d", f"per-station rows cover the two reference models x 7 cells x 3 horizons "
             f"({per_station.model.unique().tolist()})",
      set(per_station.model) == {"persistence", "logreg_linear"}
      and set(per_station.cell) == set(mtest.cell.unique()))"""))

cells.append(md("""---

## Stage 29 - class-imbalance analysis

The workflow's *Phase 19 - Class imbalance* asks for the class distribution, the
imbalance ratio, minority-class performance, and the candidate remedies (class
weighting, resampling, threshold adjustment), with the warning that oversampling must
happen **inside the training process, never before the split**. phase0 **R4** fixes the
measurement: imbalance 15-19:1, PR-AUC primary, resampling only inside the training
fold. This section measures all of it on the Phase-14 table, using the Phase-15 masks."""))

# --------------------------------------------------------------------- I1
cells.append(code(r"""# I1 - class distribution and imbalance ratio per split
def cell_stats(frame, h):
    n = int(frame[h].notna().sum())
    p = int(frame[h].sum())
    r = p / n if n else np.nan
    return {"labelled": n, "pos": p, "rate_%": round(100 * r, 2),
            "ratio_0_to_1": round((1 - r) / r, 1) if r else np.nan}

rows = []
for k, m in (("train", tr), ("val", va), ("test", te)):
    for h in HORIZONS:
        rows.append({"split": k, "horizon": h, **cell_stats(d[m], h)})
SPLIT_TBL = pd.DataFrame(rows).set_index(["split", "horizon"])
print("imbalance by split (Phase-15 masks, holdout excluded from train/val)")
print(SPLIT_TBL.to_string())

# phase0 section 4 - the whole-table numbers this project was built on
P0 = {"label_24h": (200559, 9982, 4.98, 19.1),
      "label_48h": (197748, 11241, 5.68, 16.6),
      "label_72h": (195174, 12340, 6.32, 14.8)}
ok = True
for h, (n0, p0, r0, ratio0) in P0.items():
    s = cell_stats(d, h)
    ok &= (s["labelled"] == n0 and s["pos"] == p0
           and abs(s["rate_%"] - r0) < 0.005 and abs(s["ratio_0_to_1"] - ratio0) < 0.05)
check("I1a", "whole-table label counts / rates / ratios match phase0 section 4 "
             "(4.98 % 19.1:1, 5.68 % 16.6:1, 6.32 % 14.8:1)", ok)
check("I1b", "split sizes 61,991 / 17,538 / 101,174",
      len(d[tr]) == 61991 and len(d[va]) == 17538 and len(d[te]) == 101174)
r_tr = SPLIT_TBL.loc[("train", "label_24h"), "ratio_0_to_1"]
r_va = SPLIT_TBL.loc[("val", "label_24h"), "ratio_0_to_1"]
r_te = SPLIT_TBL.loc[("test", "label_24h"), "ratio_0_to_1"]
check("I1c", f"imbalance worsens into deployment: 24 h ratio {r_tr} (train) -> "
             f"{r_va} (val) -> {r_te} (test)", r_tr < r_va < r_te)
check("I1d", "every split x horizon cell carries at least 800 positives (worst case: "
             "validation 2022 at 24 h, 891) - minority-class metrics stay estimable",
      (SPLIT_TBL.pos >= 800).all())"""))

# --------------------------------------------------------------------- I2
cells.append(code(r"""# I2 - drift (R1) and season (D1): the ratio is not one number
YEAR = d.assign(year=d.date.dt.year).groupby("year")
year_rows = []
for y, g in YEAR:
    row = {"year": y, "rows": len(g)}
    for h in HORIZONS:
        s = cell_stats(g, h)
        row[h + "_rate_%"] = s["rate_%"]
        row[h + "_ratio"] = s["ratio_0_to_1"]
    year_rows.append(row)
YEAR_TBL = pd.DataFrame(year_rows).set_index("year")
print("per-year class distribution (whole table - years are never pooled, R1/D8)")
print(YEAR_TBL.to_string())

DOC_72 = {2017: 16.81, 2019: 6.47, 2020: 15.83, 2021: 8.32, 2022: 6.64,
          2023: 1.19, 2024: 3.69, 2025: 1.79, 2026: 1.90}
got = {y: YEAR_TBL.loc[y, "label_72h_rate_%"] for y in DOC_72}
check("I2a", "per-year label_72h rates reproduce the Phase-15 table "
             "(16.81 % 2017 -> 1.19 % 2023 -> 1.90 % 2026)",
      all(abs(got[y] - v) < 0.005 for y, v in DOC_72.items()))
check("I2b", "2018 absent - the archive gap is visible in the class distribution, "
             "never interpolated across (R6)", 2018 not in YEAR_TBL.index)
r72 = YEAR_TBL["label_72h_ratio"]
check("I2c", f"72 h imbalance ratio spans {r72.min()}:{r72.max()} ({r72.idxmin()} vs "
             f"{r72.idxmax()}) - a single ratio would be meaningless", r72.max() >= 80)

season_rows = []
for k, m in (("train", tr), ("val", va), ("test", te)):
    for seas in ("monsoon", "non-monsoon"):
        g = d[m & (d.season == seas)]
        row = {"split": k, "season": seas, "rows": len(g)}
        for h in HORIZONS:
            s = cell_stats(g, h)
            row[h + "_rate_%"] = s["rate_%"]
            row[h + "_labelled"] = s["labelled"]
        season_rows.append(row)
SEASON_TBL = pd.DataFrame(season_rows).set_index(["split", "season"])
print("\nsplit x season (D1 - JJAS vs the rest)")
print(SEASON_TBL.to_string())

check("I2d", "every split x season cell carries > 1 000 labelled rows, so all three "
             "horizons can be reported per season (D1)",
      (SEASON_TBL[[h + "_labelled" for h in HORIZONS]] > 1000).all().all())
m24 = SEASON_TBL.loc[("test", "monsoon"), "label_24h_rate_%"]
n24 = SEASON_TBL.loc[("test", "non-monsoon"), "label_24h_rate_%"]
check("I2e", f"test period: monsoon {m24} % vs non-monsoon {n24} % at 24 h "
             f"({m24 / n24:.1f}x) - season split is mandatory", m24 / n24 > 5.0)"""))

# --------------------------------------------------------------------- I3
cells.append(code(r"""# I3 - where the positives live: spatial concentration (R2 / R3)
per_st = (d.groupby(["station_id", "station_name"])
          .agg(rows=("label_24h", lambda s: int(s.notna().sum())),
               pos=("label_24h", "sum"),
               mean=("label_24h", "mean"))
          .reset_index())
per_st["rate_%"] = 100 * per_st["mean"]
per_st["ratio_0_to_1"] = (100 - per_st["rate_%"]) / per_st["rate_%"].replace(0, np.nan)
per_st = per_st.sort_values("pos", ascending=False)
total_pos = int(per_st.pos.sum())
share10 = float(per_st.pos.head(10).sum() / total_pos)
share20 = float(per_st.pos.head(20).sum() / total_pos)
with_pos = int((per_st.pos > 0).sum())
big = per_st[per_st.rows >= 200]
zero_big = big[big.pos == 0].sort_values("rows", ascending=False)

print(f"133 stations, {total_pos:,} 24 h positives, {with_pos} stations ever positive")
print(f"top-10 stations hold {share10:.1%} of all positives, top-20 {share20:.1%}")
print(f"\nstations with >= 200 labelled rows and ZERO positives ({len(zero_big)}):")
print(zero_big[["station_name", "rows", "rate_%"]].to_string(index=False))
print("\nhighest positive rates:")
print(per_st.head(6)[["station_name", "rows", "pos", "rate_%"]].round(1).to_string(index=False))

check("I3a", f"{with_pos}/133 stations ever produce a positive; "
             f"{len(big)} stations have >= 200 labelled rows",
      with_pos == 111 and len(big) == 122)
check("I3b", f"{len(zero_big)} zero-positive stations with >= 200 rows - they are kept, "
             f"never dropped, and macro averages must be reported next to micro (R3)",
      len(zero_big) == 14)
check("I3c", f"top-10 stations hold {share10:.1%} of positives - a micro average is "
             f"dominated by a handful of tidal stations (R2)", 0.30 < share10 < 0.50)
if len(zero_big) != 13:
    warn("W-R3", f"phase0 R3 records 13 zero-positive stations, the Phase-14 table "
                 f"has {len(zero_big)} (>= 200 labelled rows) - re-measured here, "
                 f"phase0 not overwritten")
top1 = per_st.iloc[0]
if top1["rate_%"] < 80:
    warn("W-R2", f"phase0 R2 quotes Mongla 83.8 %; the Phase-14 table gives "
                 f"{top1['rate_%']:.1f} % - the earlier figure is not reproducible "
                 f"from the current table, flagged for phase0 review")"""))

# --------------------------------------------------------------------- I4
cells.append(code(r"""# I4 - minority-class performance, and why accuracy / ROC-AUC cannot be the headline
v = met[(met.protocol == "P1") & (met.cell == "val_2022") & (met.season == "all")].copy()
v["accuracy"] = (v.tp + v.tn) / v.n
v["always_negative_accuracy"] = 1 - v.base_rate
v["accuracy_gain"] = v.accuracy - v.always_negative_accuracy
print("validation 2022 - the accuracy trap")
print(v.pivot_table(index="model", columns="horizon",
                    values=["accuracy", "accuracy_gain"]).round(4).to_string())

check("I4a", f"best accuracy gain over the trivial always-negative classifier is "
             f"{v.accuracy_gain.max():.3f} (< 0.05) while PR-AUC gains are far larger "
             f"- accuracy is useless here and is banned as a headline",
      v.accuracy_gain.max() < 0.05)

for tag, df in (("dev", met), ("test", mtest)):
    real = df[df.model.isin(REAL)]
    gap = (real.roc_auc - real.pr_auc)
    check(f"I4b-{tag}", f"PR-AUC < ROC-AUC in all {len(real)} real-model cells "
                        f"(min gap {gap.min():.3f}) - ROC-AUC is inflated by the "
                        f"majority class, PR-AUC stays primary (R4)", (gap > 0).all())

rec_val = float(v[v.model == "logreg_linear"].set_index("horizon")
                .loc["label_24h", "recall"])
t23 = mtest[(mtest.cell == "test_2023") & (mtest.season == "all")]
rec_t23 = float(t23[t23.model == "logreg_linear"].set_index("horizon")
                .loc["label_24h", "recall"])
base_val = float(v[v.model == "logreg_linear"].set_index("horizon")
                 .loc["label_24h", "base_rate"])
base_t23 = float(t23[t23.model == "logreg_linear"].set_index("horizon")
                 .loc["label_24h", "base_rate"])
check("I4c", f"threshold transfer: logreg_linear 24 h recall {rec_val:.3f} on "
             f"validation 2022 (base {base_val:.1%}) collapses to {rec_t23:.3f} on "
             f"test 2023 (base {base_t23:.2%}) - a train-derived threshold does not "
             f"follow the base rate",
      rec_val - rec_t23 > 0.20)

minor = (mtest[(mtest.season == "all") & (mtest.model.isin(REAL))]
         .groupby(["model", "cell"])
         .agg(recall=("recall", "mean"), precision=("precision", "mean"),
              pr_auc=("pr_auc", "mean")))
print("\nfrozen test - minority-class performance per cell (never pooled, D8)")
print(minor.round(3).to_string())
check("I4d", "recall / precision / PR-AUC finite for every real model x test cell",
      minor.notna().all().all())"""))

# --------------------------------------------------------------------- figure
cells.append(code(r"""# stage-29 figure: drift of the positive rate + concentration of the positives
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
for h, mk in zip(HORIZONS, ["o-", "s--", "^:"]):
    ax.plot(YEAR_TBL.index, YEAR_TBL[h + "_rate_%"], mk, label=h)
ax.set_title("positive rate per year (R1 - never pooled)")
ax.set_xlabel("year"); ax.set_ylabel("positives / labelled rows [%]")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
cum = np.cumsum(per_st.pos.to_numpy()) / total_pos
ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=3)
ax.axvline(10, color="r", ls="--", lw=1, label=f"top 10 = {share10:.0%}")
ax.axvline(20, color="g", ls="--", lw=1, label=f"top 20 = {share20:.0%}")
ax.set_title("cumulative share of 24 h positives by station (R2)")
ax.set_xlabel("stations (sorted by positives)"); ax.set_ylabel("cumulative share")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("Phase 16 / stage 29 - class imbalance on the Phase-14 table")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "phase16_class_imbalance.png", dpi=130)
plt.close(fig)
check("I3d", "figures/phase16_class_imbalance.png written",
      (FIGURES / "phase16_class_imbalance.png").exists())"""))

# --------------------------------------------------------------------- I5
cells.append(code(r"""# I5 - metric policy carried into Phases 17+ (R4 / phase0) and the stage-29 manifest
check("I5a", "'pr_auc' is recorded for every cell in both metric files",
      met.pr_auc.notna().all() and mtest.pr_auc.notna().all())
check("I5b", "'accuracy' is not a column of either metric file - Phase 16 never "
             "reports it (workflow: do not use accuracy as the primary metric)",
      "accuracy" not in met.columns and "accuracy" not in mtest.columns)
check("I5c", "the Phase-16 report exists and carries the D7 persistence bar",
      REPORT_MD.exists() and "persistence" in REPORT_MD.read_text(encoding="utf-8"))
check("I5d", "the Phase-16 report records the frozen test as written once and read "
             "by no decision - this notebook does not contradict it",
      "recorded once" in REPORT_MD.read_text(encoding="utf-8"))

manifest = {
    "phase": 16,
    "stage": 29,
    "date": time.strftime("%Y-%m-%d"),
    "inputs": {"experimental_dataset.csv": TBL_SHA,
               "split_assignments.csv": sha256(SPLITS),
               "phase16_baseline_metrics.csv": sha256(MET),
               "phase16_test_metrics.csv": sha256(MET_TEST)},
    "distributions": {
        "by_split": SPLIT_TBL.reset_index().to_dict("records"),
        "by_year": YEAR_TBL.reset_index().to_dict("records"),
        "by_split_season": SEASON_TBL.reset_index().to_dict("records"),
    },
    "concentration": {
        "stations": 133, "stations_ever_positive": with_pos,
        "stations_ge_200_rows": int(len(big)),
        "zero_positive_stations_ge_200_rows": zero_big.station_name.tolist(),
        "top10_share_positives": round(share10, 4),
        "top20_share_positives": round(share20, 4),
        "top_station_rates": dict(zip(per_st.station_name.head(5),
                                      per_st["rate_%"].head(5).round(1))),
    },
    "imbalance_ratio_0_to_1": {
        "label_24h": {"all": 19.1, "train": float(r_tr), "val": float(r_va),
                      "test": float(r_te)},
        "label_48h": {"all": 16.6},
        "label_72h": {"all": 14.8, "worst_year": float(r72.max()),
                      "best_year": float(r72.min())},
    },
    "policy": {
        "primary_metric": "pr_auc (PR-AUC)",
        "never_headline": ["accuracy", "roc_auc alone"],
        "reporting": "per year / per bucket / per season / per station, macro + micro "
                     "(D8), never pooled",
        "thresholds": "derived from training predictions only; stage 39 sets the "
                      "cost of FN >> FP",
        "resampling": "inside the training fold only, never before the split (R4)",
        "class_weight": "candidate variant for Phase 17, fitted on the same folds",
        "open_risk": "train-derived thresholds do not transfer to the low-base-rate "
                     "test years (I4c) - recall falls from 0.85 to 0.53 at 24 h",
    },
    "checks": RES,
    "warnings": WARN,
}
MANIFEST.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
check("I5e", f"stage-29 evidence written to {MANIFEST.name} "
             f"({MANIFEST.stat().st_size // 1024} KB)", MANIFEST.exists())"""))

# --------------------------------------------------------------------- summary
cells.append(code(r"""print("=" * 78)
print("PHASE 16 + STAGE 29   baseline contract verified, class imbalance measured")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
for w in WARN:
    print(f"  WARN  {w['id']}: {w['note']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, {len(WARN)} warnings, "
      f"{time.time() - t0:.1f}s")
print("outputs: data/processed/class_imbalance.json, figures/phase16_class_imbalance.png")
print("unchanged: experimental_dataset.csv (SHA re-checked at the end below)")
assert all(r["pass"] for r in RES)
assert sha256(TBL) == TBL_SHA == TBL_SHA_EXPECTED, "experimental table changed!"
print("ALL PHASE-16 / STAGE-29 CHECKS PASSED - experimental_dataset.csv byte-stable")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
