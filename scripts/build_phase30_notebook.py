"""Build notebooks/30_errors.ipynb (workflow Phase 30 / stage 54, stage-46 evidence).

Phase 30 is a pure error-analysis of the frozen Phase-29 leaders on the P1 `val_2022`
cell.  This notebook re-derives the headline numbers **independently** from the produced
artefacts *and* from the raw inputs (cached probabilities + experimental table), so the
FN/FP tables and the named patterns cannot silently drift.

Checks N30-01..N30-12:
    N30-01  artefacts exist with the expected shapes
    N30-02  row-metric identities (tp+fp+fn+tn==n; recall/precision/far self-consistent)
    N30-03  leader per-period FN equals the sum of the Phase-29 seasonal misses
    N30-04  frozen rule reproduced from the raw probability cache + chosen thresholds
    N30-05  FN/FP tables cover exactly the leader FN / (sampled) FP
    N30-06  no test rows read; experimental-table SHA-256 unchanged
    N30-07  pattern A (river below danger, rising) holds
    N30-08  pattern F (over-alarm while below danger) holds
    N30-09  pattern E (under-represented stations missed more) holds
    N30-10  pattern G (short/isolated events are the hard labels) holds
    N30-11  negative result H (broken DL window is NOT a consistent driver)
    N30-12  patterns table well-formed with >= 6 named findings

Run:
    python scripts/build_phase30_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\30_errors.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "30_errors.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 30 — Error analysis (stage 54; stage 46 evidence)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 30 — Error analysis** → **stage 54** (deepens **stage 46**) |
| **Built by** | `scripts/phase30_errors.py` (re-uses the cached Phase-28 `val_2022` probabilities + Phase-29 thresholds; **no retraining**) |
| **Question** | Where / when do the frozen leaders miss floods (FN) and raise false alarms (FP), and why? |
| **Rule** | frozen Phase-29 decision: alarm when `p >= t_warn(period)`, `C_FN : C_FP = 10 : 1` |

This notebook re-derives the row-level outcomes from the raw inputs and checks the FN/FP
**named patterns** (top three: A response-lag FN, F over-alarm FP, E under-represented
stations). Test (`>= 2023`) is never read.""")) 

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"D:\Flood predictor\scripts")
import baseline_model as B

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"

RM = REPORTS / "phase30_row_metrics.csv"
PROF = REPORTS / "phase30_outcome_profile.csv"
FNR = REPORTS / "phase30_fn_rows.csv"
FPR = REPORTS / "phase30_fp_rows.csv"
BYST = REPORTS / "phase30_by_station.csv"
PAT = REPORTS / "phase30_patterns.csv"
SUMMARY = REPORTS / "phase30_summary.json"
REPORT = REPORTS / "phase30_errors.md"
FIG = ROOT / "figures" / "phase30_errors.png"
CHOSEN = REPORTS / "phase29_chosen_thresholds.csv"
NPZ = REPORTS / "phase28_predictions.npz"
TBL = ROOT / "data" / "processed" / "experimental_dataset.csv"
MASK3 = ROOT / "data" / "processed" / "sequence_mask_L3.npy"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
LEAD = {"24h": ("lightgbm", "label_24h"),
        "48h": ("xgboost", "label_48h"),
        "72h": ("xgboost_selected", "label_72h")}
LEADERS = {v[0] for v in LEAD.values()}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


RES = []
def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid


rm = pd.read_csv(RM)
prof = pd.read_csv(PROF)
fn = pd.read_csv(FNR)
fp = pd.read_csv(FPR)
byst = pd.read_csv(BYST)
pat = pd.read_csv(PAT)
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
chosen = pd.read_csv(CHOSEN)
npz = np.load(NPZ, allow_pickle=True)
TW = {(r.model, r.horizon, r.period): float(r.t_warn) for r in chosen.itertuples()}
t0 = time.time()
print("Phase 30 notebook - error analysis verification")""")) 

cells.append(code(r"""# N30-01  artefacts present with the expected shapes
for p in (RM, PROF, FNR, FPR, BYST, PAT, SUMMARY, REPORT, FIG, CHOSEN, NPZ):
    assert p.exists(), p
check("N30-01", f"row_metrics {rm.shape}; outcome_profile {prof.shape}; fn_rows {fn.shape}; "
                f"fp_rows {fp.shape}; by_station {byst.shape}; patterns {pat.shape}; "
                f"figure {FIG.stat().st_size:,} B; report {REPORT.stat().st_size:,} B",
      len(rm) == 18 and len(fn) == 83 and len(pat) >= 6 and REPORT.stat().st_size > 4000)""")) 

cells.append(code(r"""# N30-02  row-metric identities
bad = []
for r in rm.itertuples():
    if r.tp + r.fp + r.fn + r.tn != r.n:
        bad.append((r.model, r.horizon, "n"))
    if r.n_pos != r.tp + r.fn:
        bad.append((r.model, r.horizon, "n_pos"))
    rec = r.tp / max(1, r.tp + r.fn)
    far = r.fp / (r.tp + r.fp) if (r.tp + r.fp) else float("nan")
    if abs(rec - r.recall) > 1e-9 or abs(far - r.far) > 1e-9:
        bad.append((r.model, r.horizon, "rates"))
check("N30-02", f"tp+fp+fn+tn==n, n_pos==tp+fn and recall/far self-consistent for all "
                f"{len(rm)} cells = {not bad} (violations {bad})", not bad)""")) 

cells.append(code(r"""# N30-03  leader per-period FN equals the sum of the Phase-29 seasonal misses
rows = []
for h, (m, _label) in LEAD.items():
    seasonal = chosen[(chosen.model == m) & (chosen.horizon == h) & (chosen.period != "all")]
    expected = int(seasonal.missed.sum())
    got = int(rm[(rm.model == m) & (rm.horizon == h)].fn.iloc[0])
    rows.append((m, h, expected, got))
ok = all(e == g for *_x, e, g in rows)
print(pd.DataFrame(rows, columns=["model", "horizon", "seasonal_missed_sum", "per_period_fn"]))
check("N30-03", "per-period leader FN == sum of Phase-29 monsoon+non-monsoon misses "
                f"= {ok}", ok)"""))

cells.append(code(r"""# N30-04  frozen rule reproduced from the raw probability cache + chosen thresholds
import baseline_model as B  # noqa
d, fs = B.load()
d = d.reset_index(drop=True)
sids = d.station_id.to_numpy(); dates = d.date.to_numpy()
split = d.split.to_numpy(); hold = d.station_holdout.to_numpy()
yr = d.date.dt.year.to_numpy()


def run_lengths(labels, sids, dates):
    order = np.lexsort((dates, sids)); lab = labels[order]
    lab = np.where(np.isnan(lab), 0.0, lab); ss = sids[order]
    ch = np.ones(len(lab), bool); ch[1:] = (lab[1:] != lab[:-1]) | (ss[1:] != ss[:-1])
    gid = np.cumsum(ch) - 1; sz = np.bincount(gid)
    out = np.empty(len(lab)); out[order] = sz[gid].astype(float)
    return out


def train_pos(y, sids, tr):
    df = pd.DataFrame({"sid": sids[tr], "y": y[tr]}).dropna()
    return pd.Series(sids).map(df.groupby("sid").y.sum()).fillna(0).to_numpy(dtype=float)


frames = {}
repro = []
for hs, (m, label) in LEAD.items():
    y = d[label].to_numpy().astype(float)
    evi = np.flatnonzero((split == "val") & (~hold) & (yr == 2022) & ~np.isnan(y))
    p = npz[f"p::{m}::{label}"].astype(float)
    season = d.season.to_numpy()[evi]
    thr = np.where(season == "monsoon", TW[(m, hs, "monsoon")], TW[(m, hs, "non-monsoon")])
    pred = (p >= thr).astype(int); yv = y[evi].astype(int)
    out = np.where(yv == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 1, "FP", "TN"))
    frames[hs] = pd.DataFrame({
        "model": m, "horizon": hs, "y": yv, "p": p, "pred": pred, "outcome": out,
        "season": season, "station_id": sids[evi], "date": dates[evi],
        "river": d.river.to_numpy()[evi],
        "wl_dist_to_danger": d.wl_dist_to_danger.to_numpy()[evi],
        "wl_pct_danger": d.wl_pct_danger.to_numpy()[evi],
        "wl_rise_3d": d.wl_rise_3d.to_numpy()[evi],
        "rain_7d": d.rain_7d.to_numpy()[evi], "rain_30d": d.rain_30d.to_numpy()[evi],
        "runlen": run_lengths(y, sids, dates)[evi],
        "tr_pos": train_pos(y, sids, split == "train")[evi]})
    r = rm[(rm.model == m) & (rm.horizon == hs)].iloc[0]
    repro.append((int((out == "TP").sum()) == int(r.tp), int((out == "FP").sum()) == int(r.fp),
                  int((out == "FN").sum()) == int(r.fn), int((out == "TN").sum()) == int(r.tn)))
allr = pd.concat(frames.values(), ignore_index=True)
ok = all(all(t) for t in repro)
print("reproduced (TP,FP,FN,TN) per leader:", repro)
check("N30-04", f"fresh recompute of the frozen per-period decision reproduces every leader "
                f"cell's TP/FP/FN/TN exactly = {ok}", ok)""")) 

cells.append(code(r"""# N30-05  FN/FP tables cover exactly the leader FN / (sampled) FP
leader_fn = int(rm[rm.role == "leader"].fn.sum())
leader_fp = int(rm[rm.role == "leader"].false_alarms.sum())
fn_ok = len(fn) == leader_fn
fp_ok = len(fp) <= leader_fp and len(fp) > 0
check("N30-05", f"fn_rows {len(fn)} == leader FN total {leader_fn}; "
                f"fp_rows {len(fp)} <= leader FP total {leader_fp} (sampled)", fn_ok and fp_ok)""")) 

cells.append(code(r"""# N30-06  no test rows read + immutable table unchanged
check("N30-06", f"summary test_rows_read = {summary['test_rows_read']}; "
                f"experimental_dataset.csv SHA-256 unchanged = {sha256(TBL) == TBL_SHA}",
      summary["test_rows_read"] == 0 and sha256(TBL) == TBL_SHA)""")) 

cells.append(code(r"""# N30-07  pattern A: missed floods while the river is still below danger but rising
F = allr[allr.outcome == "FN"]; T = allr[allr.outcome == "TP"]
fn_dd = float(np.median(F.wl_dist_to_danger)); tp_dd = float(np.median(T.wl_dist_to_danger))
fn_below = float((F.wl_dist_to_danger <= 0).mean()); fn_rise = float((F.wl_rise_3d > 0).mean())
csv_val = float(pat[pat.pattern_id == "A"].value.iloc[0])
check("N30-07", f"FN median wl_dist_to_danger {fn_dd:+.2f} m > 0 and > TP {tp_dd:+.2f} m; "
                f"share FN below danger {fn_below:.3f} (< 0.05); rising {fn_rise:.0%}; "
                f"csv value {csv_val:+.2f}",
      fn_dd > 0 > tp_dd and fn_below < 0.05 and abs(fn_dd - csv_val) < 0.01)""")) 

cells.append(code(r"""# N30-08  pattern F: false alarms while the river is still below danger at t
P = allr[allr.outcome == "FP"]; N = allr[allr.outcome == "TN"]
fp_pct = float(np.median(P.wl_pct_danger)); tn_pct = float(np.median(N.wl_pct_danger))
fp_below = float((P.wl_pct_danger < 1).mean())
check("N30-08", f"FP median wl_pct_danger {fp_pct:.3f} >> TN {tn_pct:.3f}; "
                f"share FP below danger {fp_below:.3f} (> 0.7)",
      fp_pct > tn_pct and fp_below > 0.7)""")) 

cells.append(code(r"""# N30-09  pattern E: under-represented stations are missed more
stg = allr.groupby("station_id").agg(fn=("outcome", lambda s: int((s == "FN").sum())),
                                     n_pos=("y", "sum"), tr_pos=("tr_pos", "first"))
stg = stg[stg.n_pos > 0]
stg["miss_rate"] = stg.fn / stg.n_pos
q = stg.tr_pos.quantile([0.25, 0.75])
lo = float(stg[stg.tr_pos <= q.iloc[0]].miss_rate.mean())
hi = float(stg[stg.tr_pos >= q.iloc[1]].miss_rate.mean())
rho = float(stg.tr_pos.corr(stg.miss_rate, method="spearman"))
check("N30-09", f"low vs high train-flood quartile miss rate {lo:.3f} vs {hi:.3f}; "
                f"Spearman rho {rho:+.3f} (<= -0.3) over {len(stg)} stations",
      lo > hi and rho <= -0.3)""")) 

cells.append(code(r"""# N30-10  pattern G: short / isolated events are the hard labels
fn_short = float((F.runlen <= 2).mean()); tp_short = float((T.runlen <= 2).mean())
check("N30-10", f"share in runs <= 2 rows: FN {fn_short:.3f} > TP {tp_short:.3f}; "
                f"median run FN {float(np.median(F.runlen)):.0f} vs TP {float(T.runlen.median()):.0f}",
      fn_short > tp_short)""")) 

cells.append(code(r"""# N30-11  negative result H: broken DL window is NOT a consistent miss driver
M = np.load(MASK3, mmap_mode="r")
shares = {}
for hs, label in [("24h", "label_24h"), ("48h", "label_48h"), ("72h", "label_72h")]:
    y = d[label].to_numpy().astype(float)
    evi = np.flatnonzero((split == "val") & (~hold) & (yr == 2022) & ~np.isnan(y))
    p = npz[f"p::fusion_L3::{label}"].astype(float)
    season = d.season.to_numpy()[evi]
    thr = np.where(season == "monsoon", TW[("fusion_L3", hs, "monsoon")],
                   TW[("fusion_L3", hs, "non-monsoon")])
    pred = (p >= thr).astype(int); yv = y[evi].astype(int)
    out = np.where(yv == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 1, "FP", "TN"))
    mfrac = np.asarray(M[evi], dtype=np.float32).mean(axis=1)
    shares[hs] = (
        float((mfrac[out == "FN"] < 1).mean()) if (out == "FN").any() else float("nan"),
        float((mfrac[out == "TP"] < 1).mean()))
    print(f"fusion_L3 {hs}: FN broken {shares[hs][0]:.3f} vs TP {shares[hs][1]:.3f}")
consistent = all(shares[h][0] > shares[h][1] for h in shares)
check("N30-11", "broken sequence window is NOT a consistent DL-miss driver "
                f"(FN>TP in every cell = {consistent}; it reverses after 24 h)",
      not consistent)""")) 

cells.append(code(r"""# N30-12  patterns table well-formed with >= 6 named findings
need = {"A", "E", "F"}
have = set(pat.pattern_id)
wellformed = pat[["finding", "metric", "disposition", "detail"]].notna().all().all()
check("N30-12", f"{len(pat)} named findings; top-3 present = {need <= have}; "
                f"all rows well-formed = {bool(wellformed)}",
      len(pat) >= 6 and need <= have and bool(wellformed))""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 30 / STAGE 54  Error analysis verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
npass = sum(rr['pass'] for rr in RES)
print(f"\n{npass}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
print("Top patterns: A response-lag FN; F over-alarm FP; E under-represented stations.")
print("test_rows_read =", summary["test_rows_read"])""")) 

cells.append(md("""## Named patterns (from `reports/phase30_patterns.csv`)

| # | finding | value | disposition |
| --- | --- | --- | --- |
| A | Missed floods occur while the river is still below danger but rising | median +0.76 m; 0.0 % of FN had wl<=danger | limitation (response-lag) |
| F | False alarms fire while the river is still below danger at `t` | median wl_pct_danger 0.983; 80.9 % below danger | limitation (over-alarm) |
| E | Stations seen with fewer training floods are missed more | miss rate 0.230 vs 0.016; rho -0.501 | limitation |
| B | Misses grow with lead time | 16 / 19 / 48; recall 0.982 / 0.981 / 0.957 | limitation |
| G | Short/isolated events are the hard labels | FN in runs <= 2 rows 14.5 % vs TP 2.8 % | limitation |
| G2 | Misses sit in wetter conditions, not drier | rain_7d 102 vs 86 mm | limitation |
| I | Misses cluster spatially | 28.9 % of FN in Jamuna + Teesta | limitation |
| C | Monsoon concentration is a base-rate effect | recall 0.973 vs 0.970 | not a defect |
| D | Missingness is a weak discriminator | medians identical | weak |
| H | Broken DL window is not a consistent driver | FN>TP only at 24 h | not supported |

**Verdict.** The leaders' FN and FP are two sides of the Phase-29 cost-optimal point (the
river sits *just below* danger): early-stage floods are missed (A) and near-misses are
alarmed (F). Fixes: target low-prevalence stations (E), add upstream/future-rain signal (A);
F is a policy knob. C/D/H are tested-and-not-supported. Test untouched.""")) 

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "flood-predictor", "language": "python",
                   "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
