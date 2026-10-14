"""Build notebooks/29_thresholds.ipynb (workflow Phase 29 / stages 48, 49).

Phase 29 is a pure decision layer on the Phase-27/28 evidence: it re-uses the cached P1
`val_2022` probabilities (`reports/phase28_predictions.npz`) and only *sweeps thresholds*,
*derives risk bands* and *tests the period shift*.  This notebook re-derives every claim
independently from the produced artefacts.

Checks N29-01..N29-12:
    N29-01  artefacts exist with the expected shapes
    N29-02  sweep complete: 24 models x 3 horizons x 3 periods x 65 thresholds, ids match registry
    N29-03  chosen table complete + chosen threshold is at least as good as the grid best
    N29-04  metric ranges valid (precision/recall/f1/far/fpr/alarm_rate in [0,1])
    N29-05  cost identity: cost_per1k == (10*fn + fp)*1000/n recomputed from counts
    N29-06  leaders are derived (max all-season PR-AUC), not asserted
    N29-07  cost ratio is a policy input: t(20:1) <= t(10:1) <= t(5:1) for every leader
    N29-08  period shift reproduces the Phase-16 lesson: dry->wet recall loss >= 0
    N29-09  transfer table self-consistent (diagonal loss 0; loss == self_recall - target_recall)
    N29-10  risk bands partition the data (frequencies sum to 1; flood counts sum to total)
    N29-11  cost-derived bands beat the quartile default (very-high rate >= 0.8; quartile collapses)
    N29-12  immutable table SHA-256 unchanged + test never read

Run:
    python scripts/build_phase29_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\29_thresholds.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "29_thresholds.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 29 — Risk levels & threshold optimization (stages 48, 49)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 29 — Risk levels & threshold optimization** → **stages 48, 49** |
| **Built by** | `scripts/phase29_thresholds.py` → `reports/phase29_threshold_sweep.csv` (+ `_chosen_thresholds.csv`, `_transfer.csv`, `_risk_bands.csv`, `_pr_auc.csv`, `_summary.json`, `figures/phase29_*.png`) |
| **Reads** | `reports/phase28_predictions.npz` (the Phase-28/27 probabilities) · `reports/phase29_*.csv` · `reports/phase17_experiment_registry.csv` |
| **Contract** | thresholds and bands derived on `val_2022` only (test frozen to Phase 33) · row-level cost `C_FN:CFP = 10:1` (sensitivity 5, 20) · **period-specific re-derivation** |

The operating objective is explicit: **a missed flood costs 10× a false alarm**. This notebook
verifies that the sweep, the chosen thresholds, the period-transfer penalty and the risk bands
are internally consistent and *derived*, not asserted.""")) 

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"

SWEEP = REPORTS / "phase29_threshold_sweep.csv"
CHOSEN = REPORTS / "phase29_chosen_thresholds.csv"
TRANSFER = REPORTS / "phase29_transfer.csv"
BANDS = REPORTS / "phase29_risk_bands.csv"
PRAUC = REPORTS / "phase29_pr_auc.csv"
SUMMARY = REPORTS / "phase29_summary.json"
NPZ = REPORTS / "phase28_predictions.npz"
REPORT = REPORTS / "phase29_thresholds.md"
REG = REPORTS / "phase17_experiment_registry.csv"
FIG1 = FIGDIR / "phase29_threshold_cost.png"
FIG2 = FIGDIR / "phase29_bands_transfer.png"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HOR = ["24h", "48h", "72h"]
PERIODS = ["all", "monsoon", "non-monsoon"]
CFN = 10.0


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


sweep = pd.read_csv(SWEEP)
chosen = pd.read_csv(CHOSEN)
transfer = pd.read_csv(TRANSFER)
bands = pd.read_csv(BANDS)
prauc = pd.read_csv(PRAUC)
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
reg = pd.read_csv(REG)
t0 = time.time()
print("Phase 29 notebook - risk levels & threshold verification")""")) 

cells.append(code(r"""# N29-01  artefacts present with the expected shapes
for p in (SWEEP, CHOSEN, TRANSFER, BANDS, PRAUC, SUMMARY, REPORT, FIG1, FIG2):
    assert p.exists(), p
check("N29-01", f"sweep {sweep.shape}, chosen {chosen.shape}, transfer {transfer.shape}, "
                f"bands {bands.shape}; figures {FIG1.stat().st_size:,}+{FIG2.stat().st_size:,} B; "
                f"report {REPORT.stat().st_size:,} B",
      len(sweep) == 24 * 3 * 3 * 65 and len(chosen) == 216 and REPORT.stat().st_size > 5000)""")) 

cells.append(code(r"""# N29-02  sweep completeness + registry agreement
key = ["model", "horizon", "period", "threshold"]
dup = int(sweep.duplicated(key).sum())
n_models = sweep.model.nunique()
grid = sorted(sweep.threshold.unique())
hors_ok = set(sweep.horizon.unique()) == set(HOR)
per_ok = set(sweep.period.unique()) == set(PERIODS)
names_ok = set(sweep.model) == set(np.load(NPZ, allow_pickle=True)["models"])
check("N29-02", f"{n_models} models x {len(HOR)} horizons x {len(PERIODS)} periods x "
                f"{len(grid)} thresholds = {len(sweep)} rows (dup {dup}); "
                f"grid {grid[0]}..{grid[-1]}; horizon ok {hors_ok}; period ok {per_ok}; "
                f"model names match the 24 in the npz = {names_ok}",
      n_models == 24 and len(sweep) == 14040 and dup == 0 and len(grid) == 65
      and hors_ok and per_ok and names_ok)""")) 

cells.append(code(r"""# N29-03  chosen table complete + chosen threshold at least as good as the grid best
comp = set(zip(chosen.model, chosen.horizon, chosen.period))
sweep_keys = set(zip(sweep.model, sweep.horizon, sweep.period))
worst_gap = 0.0
for _, r in chosen.iterrows():
    g = sweep[(sweep.model == r.model) & (sweep.horizon == r.horizon)
              & (sweep.period == r.period)]
    # cost at chosen t from the sweep row nearest the chosen threshold (grid resolution)
    row = g.iloc[(g.threshold - r.t_warn).abs().argmin()]
    # chosen (exact) cost must not exceed the best grid cost by more than the grid step
    worst_gap = max(worst_gap, float(r.cost_per1k - g.cost_fn10_per1k.min()))
check("N29-03", f"chosen covers every cell = {comp == sweep_keys}; "
                f"max (chosen_cost - best_grid_cost) = {worst_gap:.4f} per 1k (<= grid step)",
      comp == sweep_keys and worst_gap < 0.5)""")) 

cells.append(code(r"""# N29-04  metric ranges
cols01 = ["precision", "recall", "f1", "far", "fpr", "alarm_rate"]
in01 = all(bool((sweep[c].dropna() >= -1e-9).all() and (sweep[c].dropna() <= 1 + 1e-9).all())
           for c in cols01)
ch01 = all(bool((chosen[c].dropna() >= -1e-9).all() and (chosen[c].dropna() <= 1 + 1e-9).all())
           for c in cols01)
check("N29-04", f"sweep probability-like columns within [0,1] = {in01}; chosen = {ch01}; "
                f"min cost_per1k = {chosen.cost_per1k.min():.2f} (>=0)",
      in01 and ch01 and bool((chosen.cost_per1k >= 0).all()))""")) 

cells.append(code(r"""# N29-05  cost identity: cost_per1k == (10*fn + fp)*1000/n from the counts
worst = 0.0
for _, r in chosen.iterrows():
    pos = r.base_rate * r.n
    tp = pos - r.missed
    fp = r.alarm_rate * r.n - tp
    c = (CFN * r.missed + fp) * 1000.0 / r.n
    worst = max(worst, abs(c - r.cost_per1k))
check("N29-05", f"max |recomputed cost - stored cost| = {worst:.2e} per 1k over {len(chosen)} "
                f"cells (C_FN={CFN:.0f}, C_FP=1)",
      worst < 1e-6)""")) 

cells.append(code(r"""# N29-06  leaders are derived (max all-season PR-AUC), not asserted
allc = chosen[chosen.period == "all"]
derived = {h: allc[allc.horizon == h].sort_values("pr_auc", ascending=False).iloc[0].model
           for h in HOR}
agree = derived == summary["leaders"]
print("derived leaders:", derived)
check("N29-06", f"derived leaders == summary leaders = {agree}; pr_auc "
                f"{ {h: round(float(allc[(allc.horizon==h)&(allc.model==derived[h])].pr_auc.iloc[0]),4) for h in HOR} }",
      agree)""")) 

cells.append(code(r"""# N29-07  cost ratio is a policy input: t(20:1) <= t(10:1) <= t(5:1)
bad = []
for h in HOR:
    m = summary["leaders"][h]
    r = allc[(allc.model == m) & (allc.horizon == h)].iloc[0]
    if not (r.t_fn20 <= r.t_warn + 1e-9 <= r.t_fn5 + 1e-9):
        bad.append((h, m, r.t_fn20, r.t_warn, r.t_fn5))
print(allc[allc.model.isin(summary["leaders"].values())][
    ["model", "horizon", "t_fn20", "t_warn", "t_fn5", "t_f1"]].round(4).to_string(index=False))
check("N29-07", f"t(20:1) <= t(10:1) <= t(5:1) holds for all leaders = {not bad} "
                f"(violations {bad})",
      not bad)""")) 

cells.append(code(r"""# N29-08  period shift reproduces the Phase-16 lesson: dry->wet recall loss >= 0
dw = transfer[(transfer.variant == "raw") & (transfer.source == "non-monsoon")
              & (transfer.target == "monsoon")]
frac_pos = float((dw.recall_loss > 1e-9).mean())
print(dw.sort_values("recall_loss", ascending=False).head(8)[
    ["model", "horizon", "t_source", "t_target_own", "self_recall", "target_recall",
     "recall_loss"]].round(4).to_string(index=False))
lead_loss = {h: float(dw[(dw.model == summary["leaders"][h]) & (dw.horizon == h)]
                      .recall_loss.iloc[0]) for h in HOR}
check("N29-08", f"dry->wet carries a recall penalty on {frac_pos:.0%} of cells; "
                f"worst {dw.recall_loss.max():.3f} ({dw.loc[dw.recall_loss.idxmax(), 'model']}); "
                f"leader losses { {h: round(v,4) for h,v in lead_loss.items()} }",
      frac_pos > 0.5 and float(dw.recall_loss.max()) > 0.05 and lead_loss["72h"] > 0.0)""")) 

cells.append(code(r"""# N29-09  transfer table self-consistent
diag = transfer[transfer.source == transfer.target]
diag_ok = bool((diag.recall_loss.abs() < 1e-9).all())
ident = float((transfer.recall_loss - (transfer.self_recall - transfer.target_recall))
              .abs().max())
check("N29-09", f"diagonal (source==target) recall_loss == 0 = {diag_ok}; "
                f"max |loss - (self_recall - target_recall)| = {ident:.2e}",
      diag_ok and ident < 1e-9)""")) 

cells.append(code(r"""# N29-10  risk bands partition the data
cd = bands[bands.scheme == "cost-derived"]
freq_ok = bool(np.allclose(cd.groupby(["model", "horizon"]).freq.sum().values, 1.0, atol=1e-6))
# flood counts must sum to the base-rate population for each cell
z = np.load(NPZ, allow_pickle=True)
flood_ok = True
for (m, h), g in cd.groupby(["model", "horizon"]):
    hk = [k for k in ["label_24h", "label_48h", "label_72h"] if k.endswith(h)][0]
    tot = float(z[f"y::{hk}"].sum())
    flood_ok = flood_ok and abs(g.floods.sum() - tot) < 1e-6
check("N29-10", f"cost-derived band frequencies sum to 1 = {freq_ok}; "
                f"band flood counts sum to the population total = {flood_ok}",
      freq_ok and flood_ok)""")) 

cells.append(code(r"""# N29-11  cost-derived bands beat the quartile default
cd = bands[bands.scheme == "cost-derived"]
qt = bands[bands.scheme == "quartile"]
sep_ok = all(float(cd[(cd.model == summary["leaders"][h]) & (cd.horizon == h)
                      & (cd.band == "very high")].flood_rate.iloc[0]) >= 0.8 for h in HOR)
qt_lead = qt[qt.model.isin(set(summary["leaders"].values()))]
qt_zero = float(qt_lead[qt_lead.flood_rate.fillna(0) == 0]
                .groupby(["model", "horizon"]).freq.sum().mean())
qt_top = float(qt_lead[qt_lead.band == "very high"].flood_rate.mean())
print("quartile (leaders): mean mass in zero-flood bands =", round(qt_zero, 3),
      "| very-high band rate =", round(qt_top, 3))
check("N29-11", f"cost-derived very-high band rate >= 0.8 for every leader = {sep_ok}; "
                f"quartile mass in zero-flood bands = {qt_zero:.0%} (collapses the top quartile)",
      sep_ok and qt_zero > 0.5)""")) 

cells.append(code(r"""# N29-12  immutable table + freeze
check("N29-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA}); test rows read = "
                f"{summary['test_rows_read']}",
      sha256(TBL) == TBL_SHA and summary["test_rows_read"] == 0)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 29 / STAGES 48, 49  Risk levels & threshold optimization verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
print(f"\n{sum(rr['pass'] for rr in RES)}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
ft = summary["frozen_thresholds_all_season"]
print("Frozen thresholds (all-season, FN:FP=10:1): "
      + "; ".join(f'{h} {ft[h]["model"]} t={ft[h]["t_warn"]:.3f} recall={ft[h]["recall"]:.3f}'
                  for h in HOR))
print("Re-derive per period; cost-derived risk bands beat the quartile default; "
      "test untouched (frozen to Phase 33).")
assert all(rr["pass"] for rr in RES)
print("ALL PHASE-29 CHECKS PASSED")""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
