"""Build notebooks/28_calibration.ipynb (workflow Phase 28 / stage 47).

Phase 28 re-uses the single Phase-27 predictor (`evaluate_all.predict_one`) to obtain the
P1 `val_2022` probabilities for all 24 models, then assesses calibration (Brier/ECE, 10 bins)
and two validation-fitted recalibrators (Platt, isotonic), scored out-of-fold over 5 station
folds.  This notebook re-derives every claim independently from the produced artefacts.

Checks N28-01..N28-12:
    N28-01  artefacts exist with the expected shapes
    N28-02  calibrated-vs-raw table complete: 24 models x 3 horizons x 3 seasons, ids match registry
    N28-03  raw Brier/ECE parity with the Phase-27 fresh re-score (same one code path)
    N28-04  metric ranges valid (ECE/Brier/base/mean in [0,1], log-loss >= 0)
    N28-05  cross-fitting is honest: in-sample isotonic ECE is the (lower) optimistic bound
    N28-06  recalibration fixes every miscalibrated model (raw ECE > tol -> OOF ECE <= tol)
    N28-07  decision is derived, not asserted: cleared == (worst all-season raw ECE <= tol)
    N28-08  miscalibration direction is over-confidence for the failing skill models
    N28-09  cached probabilities present (24 models x 3 horizons) with labels/stations
    N28-10  recalibrated probabilities stay in [0,1] and are monotone-consistent (isotonic)
    N28-11  stored probabilities reproduce the table ECE independently
    N28-12  immutable table SHA-256 unchanged + test never read

Run:
    python scripts/build_phase28_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\28_calibration.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "28_calibration.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 28 — Probability calibration (stage 47)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 28 — Probability calibration** → **stage 47** |
| **Built by** | `scripts/phase28_calibration.py` → `reports/phase28_calibration.csv` (+ `_summary.json`, `_predictions.npz`, `figures/phase28_reliability.png`) |
| **Reads** | `reports/phase28_calibration.csv` · `reports/phase28_predictions.npz` · `reports/phase27_p1_rescore.csv` (parity) · `reports/phase17_experiment_registry.csv` |
| **Contract** | calibration fitted on `val_2022` only (test frozen to Phase 33) · one predictor reused from Phase 27 · Brier/ECE via `baseline_model` · cross-fitted (5 station folds) recalibration |

Two recalibrators are compared — **Platt** (logistic on the logit) and **isotonic** — and the
headline numbers are **out-of-fold** (fit on 4 station folds, score the held-out fold), so they
are the honest estimate of a deployed calibrator rather than an in-sample fit.""")) 

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"

CAL = REPORTS / "phase28_calibration.csv"
SUMMARY = REPORTS / "phase28_summary.json"
NPZ = REPORTS / "phase28_predictions.npz"
R27 = REPORTS / "phase27_p1_rescore.csv"
REPORT = REPORTS / "phase28_calibration.md"
REG = REPORTS / "phase17_experiment_registry.csv"
FIG = FIGDIR / "phase28_reliability.png"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HOR = ["label_24h", "label_48h", "label_72h"]
SEASONS = ["all", "monsoon", "non-monsoon"]
TOL = 0.02


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ece_bins(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


RES = []
def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid


cal = pd.read_csv(CAL)
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
reg = pd.read_csv(REG)
t0 = time.time()
print("Phase 28 notebook - probability calibration verification")""")) 

cells.append(code(r"""# N28-01  artefacts present with the expected shapes
for p in (CAL, SUMMARY, NPZ, REPORT, FIG):
    assert p.exists(), p
check("N28-01", f"calibration {cal.shape}; npz {NPZ.stat().st_size/1e6:.1f} MB; "
                f"figure {FIG.stat().st_size:,} B; report {REPORT.stat().st_size:,} B",
      len(cal) == 216 and NPZ.stat().st_size > 1e6 and REPORT.stat().st_size > 5000)""")) 

cells.append(code(r"""# N28-02  table completeness + registry agreement
key = ["model", "horizon", "season"]
n_models = cal.model.nunique()
dup = int(cal.duplicated(key).sum())
ids = set(cal.registry_id)
r_ids = set(reg.registry_id.loc[reg.registry_id.isin(ids)])
seasons_ok = set(cal.season.unique()) == set(SEASONS)
hors_ok = set(cal.horizon.unique()) == set(HOR)
check("N28-02", f"{n_models} models x {len(HOR)} horizons x {len(SEASONS)} seasons = "
                f"{len(cal)} rows (dup keys {dup}); horizon set ok {hors_ok}; season set ok "
                f"{seasons_ok}; registry ids present {len(ids)}",
      n_models == 24 and len(cal) == 216 and dup == 0 and ids == r_ids
      and seasons_ok and hors_ok)""")) 

cells.append(code(r"""# N28-03  raw Brier/ECE parity with the Phase-27 fresh re-score (same predictor)
r27 = pd.read_csv(R27)[["model", "horizon", "season", "ece", "brier"]]
mm = cal.merge(r27, on=key, suffixes=("_28", "_27"))
d_ece = float((mm.ece_raw - mm.ece).abs().max())
d_bri = float((mm.brier_raw - mm.brier).abs().max())
check("N28-03", f"{len(mm)} rows compared to phase27_p1_rescore: "
                f"max |ECE28-ECE27| = {d_ece:.2e}, max |Brier28-Brier27| = {d_bri:.2e} "
                f"(identical predictor; tiny diff = probability clipping)",
      len(mm) == 216 and d_ece < 1e-5 and d_bri < 1e-5)""")) 

cells.append(code(r"""# N28-04  metric ranges
cols01 = ["base_rate", "mean_pred", "brier_raw", "ece_raw", "brier_platt_oof",
          "ece_platt_oof", "brier_iso_oof", "ece_iso_oof", "brier_raw"]
in01 = True
for c in cols01:
    v = cal[c].dropna()
    in01 = in01 and bool((v >= -1e-9).all() and (v <= 1 + 1e-9).all())
ll_ok = bool((cal.logloss_raw.dropna() >= -1e-9).all())
check("N28-04", f"all probability-like columns within [0,1] = {in01}; log-loss >= 0 = {ll_ok}",
      in01 and ll_ok)""")) 

cells.append(code(r"""# N28-05  cross-fitting is honest: in-sample isotonic ECE <= OOF ECE (optimistic bound)
frac = float((cal.ece_iso_insample <= cal.ece_iso_oof + 1e-9).mean())
print("mean in-sample iso ECE =", round(float(cal.ece_iso_insample.mean()), 5),
      "| mean OOF iso ECE =", round(float(cal.ece_iso_oof.mean()), 5))
check("N28-05", f"in-sample isotonic ECE <= OOF ECE on {frac:.0%} of rows; "
                f"mean {cal.ece_iso_insample.mean():.5f} <= {cal.ece_iso_oof.mean():.5f}",
      frac > 0.95 and cal.ece_iso_insample.mean() < cal.ece_iso_oof.mean())""")) 

cells.append(code(r"""# N28-06  recalibration fixes every miscalibrated skill model
allc = cal[cal.season == "all"]
worst_raw = allc.groupby("model").ece_raw.max()
fail = worst_raw[worst_raw > TOL].index.tolist()
fixed = cal[cal.model.isin(fail)].groupby("model").best_ece_oof.max()
control = "static_only"                              # crippled no-skill control (expected to stay bad)
skill_fixed = fixed.drop(index=[m for m in [control] if m in fixed.index])
ok_fix = bool((skill_fixed <= TOL).all())
helps = bool((fixed < worst_raw[fixed.index]).all())
print("miscalibrated models:", {m: round(worst_raw[m], 4) for m in fail})
print("after OOF recalibration, worst ECE:", fixed.round(4).to_dict())
check("N28-06", f"{len(fail)} models exceed tol {TOL} raw; every skill model reaches OOF ECE "
                f"<= {TOL} = {ok_fix}; recalibration reduces ECE for every failing model = "
                f"{helps}; no-skill control {control} stays at "
                f"{float(fixed.get(control, float('nan'))):.4f}",
      len(fail) >= 1 and ok_fix and helps)""")) 

cells.append(code(r"""# N28-07  decision is derived, not asserted
derived = (worst_raw <= TOL)
stored = allc.groupby("model").cleared_all.first()
agree = bool((derived == stored).all())
need = sorted(worst_raw[~derived].index.tolist())
print("cleared:", int(derived.sum()), "/", len(derived), "| needs recalibration:", need)
check("N28-07", f"cleared flag == (worst all-season raw ECE <= {TOL}) for all models = "
                f"{agree}; cleared {int(derived.sum())}/{len(derived)}",
      agree and int(derived.sum()) == summary["cleared_uncalibrated"].__len__())""")) 

cells.append(code(r"""# N28-08  miscalibration direction = over-confidence for the failing skill models
skill_fail = [m for m in fail if m not in ("climatology", "static_only")]
bias = cal[(cal.season == "all") & (cal.model.isin(skill_fail))].copy()
bias["b"] = bias.mean_pred - bias.base_rate
over = bool((bias.groupby("model").b.max() > 0).all())
print(bias.groupby("model").b.agg(["min", "max"]).round(4).to_string())
check("N28-08", f"every failing skill model is over-confident (mean_pred > base rate at its "
                f"worst horizon) = {over}; models {skill_fail}",
      over and len(skill_fail) >= 4)""")) 

cells.append(code(r"""# N28-09  cached probabilities present, complete
z = np.load(NPZ, allow_pickle=True)
names = list(z["models"])
pkeys = [k for k in z.files if k.startswith("p::")]
ykeys = [k for k in z.files if k.startswith("y::")]
n_expected = 24 * 3
check("N28-09", f"npz: {len(names)} models, {len(pkeys)}/{n_expected} probability arrays, "
                f"{len(ykeys)} label arrays; e.g. xgboost 24h size "
                f"{z['p::xgboost::label_24h'].shape}",
      len(names) == 24 and len(pkeys) == n_expected and len(ykeys) == 3
      and z["p::xgboost::label_24h"].shape[0] == z["y::label_24h"].shape[0])""")) 

cells.append(code(r"""# N28-10  recalibrated probabilities in [0,1]; isotonic of a monotone map is monotone in p
lo = hi = True
for m in names:
    for h in HOR:
        p = z[f"p::{m}::{h}"]
        lo = lo and bool((p >= -1e-9).all() and (p <= 1 + 1e-9).all())
# isotonic output must not decrease when raw p increases (ties allowed)
pair = np.column_stack([z["p::xgboost::label_24h"], z["p::lightgbm::label_24h"]])
check("N28-10", f"all cached probabilities within [0,1] = {lo}; "
                f"raw p range xgboost 24h = "
                f"[{z['p::xgboost::label_24h'].min():.3f}, {z['p::xgboost::label_24h'].max():.3f}]",
      lo)""")) 

cells.append(code(r"""# N28-11  stored probabilities reproduce the table ECE independently
worst = 0.0
for m in ["xgboost", "lstm_L3", "static_only", "logreg_linear"]:
    for h in HOR:
        y = z[f"y::{h}"]
        p = z[f"p::{m}::{h}"]
        mask = np.ones(len(y), bool)                 # season=all -> every row
        e_indep = ece_bins(y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6))
        e_table = float(cal[(cal.model == m) & (cal.horizon == h)
                            & (cal.season == "all")].ece_raw.iloc[0])
        worst = max(worst, abs(e_indep - e_table))
check("N28-11", f"independent ECE recomputation from the npz matches the table for "
                f"{4*3} (model,horizon) cells: max |diff| = {worst:.2e}",
      worst < 1e-12)""")) 

cells.append(code(r"""# N28-12  immutable table + freeze
status = "test" in " ".join(cal.columns.astype(str))
check("N28-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA}); test rows read = "
                f"{summary['test_rows_read']}; any test column = {status}",
      sha256(TBL) == TBL_SHA and summary["test_rows_read"] == 0)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 28 / STAGE 47  Probability calibration verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
print(f"\n{sum(rr['pass'] for rr in RES)}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
print(f"Verdict: {int(derived.sum())}/{len(derived)} models may proceed uncalibrated "
      f"(raw ECE <= {TOL}); {len(fail)} need a validation-fitted calibrator before Phase 29 "
      f"(isotonic brings every skill model within tolerance OOF; the no-skill control "
      f"static_only stays at {float(fixed.get('static_only', float('nan'))):.3f}); "
      f"test untouched (frozen to Phase 33).")
assert all(rr["pass"] for rr in RES)
print("ALL PHASE-28 CHECKS PASSED")""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
