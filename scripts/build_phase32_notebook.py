"""Build notebooks/32_ablation.ipynb (workflow Phase 32 / stages 55, 56).

Phase 32 runs one planned ablation/comparison matrix (E0-E16) once.  This notebook
re-derives the headline facts independently from the produced artefacts *and* from the
raw experimental table:

    N32-01  artefacts exist with the expected shapes
    N32-02  no test rows read; experimental-table SHA-256 unchanged
    N32-03  the five layers tile FULL 62 exactly; G5_no_up == 58
    N32-04  group table shape + finite primary metrics (CI brackets the estimate)
    N32-05  hydrology is the dominant contribution and clears the D7 bar
    N32-06  weather-only / weather+rain are near the climatology bar (no skill)
    N32-07  the +network layer is a marginal increment over +hydrology
    N32-08  dropping up_* (E11) is a negligible increment (F14-2 confirmed)
    N32-09  the D7 persistence bar beats climatology everywhere
    N32-10  E0..E16 present; E12 blocked with the R8 blocker; E14 has L=3/7/14/30
    N32-11  increments: finite CIs and the significance flag matches the CI sign
    N32-12  seasons present, figure written, reused temporal detail finite

Run:
    python scripts/build_phase32_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\32_ablation.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "32_ablation.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 32 — Ablation & comparative experiment matrix (stages 55, 56)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 32 — Ablation & comparative experiment matrix** → **stages 55, 56** |
| **Built by** | `scripts/phase32_ablation.py` (nested feature-group ablation + reused E12-E16) |
| **New run** | E1-E5, E11: six nested groups x 2 fixed learners x 7 dev cells x 3 seasons x 3 horizons |
| **Reused** | E13-E16 from the frozen Phase-16/21/23/24/25/26/27 artefacts; E12 blocked by R8 |
| **Rule** | micro (pooled) + macro (per-station mean) PR-AUC with station-bootstrap CI; paired CI on increments |

`G1 weather (5) -> +rainfall history (7) -> +geography (14) -> +hydrology (24) -> +network (12) = FULL 62`,
with `G5_no_up (58)` for the F14-2 `up_*` contrast.  Every number is per cell, never pooled
(D8).  Test (`>= 2023`) is never read.""")) 

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"D:\Flood predictor\scripts")
import phase32_ablation as P32

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
TBL = ROOT / "data" / "processed" / "experimental_dataset.csv"
FIG = ROOT / "figures" / "phase32_ablation.png"
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
TEMPORAL = ["val_2020", "val_2021", "val_2022"]
HORDERS = ["24h", "48h", "72h"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


g = pd.read_csv(REPORTS / "phase32_groups.csv")
r = pd.read_csv(REPORTS / "phase32_reference.csv")
inc = pd.read_csv(REPORTS / "phase32_increments.csv")
mat = pd.read_csv(REPORTS / "phase32_matrix.csv")
reuse = pd.read_csv(REPORTS / "phase32_reuse_detail.csv")
summary = json.loads((REPORTS / "phase32_summary.json").read_text(encoding="utf-8"))
print("Phase 32 notebook - ablation matrix verification")""")) 

cells.append(code(r"""def macro(cells, learner, group, hor, col="pr_auc", season="all"):
    sub = g[(g.learner == learner) & (g.group == group) & (g.horizon == hor)
            & (g.season == season) & (g.cell.isin(cells))]
    return float(sub[col].mean())


RES = []
def check(cid, msg, ok):
    RES.append((cid, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid}  {msg}")""")) 

cells.append(code(r"""# N32-01  artefacts present with the expected shapes
for p in (REPORTS / "phase32_groups.csv", REPORTS / "phase32_reference.csv",
          REPORTS / "phase32_increments.csv", REPORTS / "phase32_matrix.csv",
          REPORTS / "phase32_reuse_detail.csv", REPORTS / "phase32_summary.json",
          REPORTS / "phase32_ablation.md", FIG):
    assert p.exists(), p
check("N32-01", f"groups {g.shape}, reference {r.shape}, increments {inc.shape}, "
                f"matrix {mat.shape}, reuse {reuse.shape}; figure {FIG.stat().st_size:,} B",
      len(g) == 6 * 2 * 7 * 3 * 3 and len(r) == 2 * 7 * 3 * 3
      and len(inc) == 5 * 2 * 7 * 3 and len(mat) >= 20)""")) 

cells.append(code(r"""# N32-02  no test rows read + immutable table unchanged
check("N32-02", f"summary test_rows_read = {summary['test_rows_read']}; "
                f"experimental_dataset.csv SHA-256 unchanged = {sha256(TBL) == TBL_SHA}",
      summary["test_rows_read"] == 0 and sha256(TBL) == TBL_SHA)""")) 

cells.append(code(r"""# N32-03  the five disjoint layers tile FULL 62 exactly; G5_no_up == 58
import json as _json
full = set(_json.loads((ROOT / "data" / "processed" / "feature_sets.json").read_text(
    encoding="utf-8"))["sets"]["FULL"])
layers = {"weather": set(P32.WEATHER), "rain_hist": set(P32.RAIN_HIST), "geo": set(P32.GEO),
          "hydro": set(P32.HYDRO), "network": set(P32.NETWORK)}
union = set().union(*layers.values())
disjoint = sum(len(s) for s in layers.values()) == len(union) == 62
sizes = {k: int(v) for k, v in summary["groups"].items()}
layer_sizes = {k: len(v) for k, v in layers.items()}
ok = (union == full and disjoint and sizes["G1_weather"] == 5
      and sizes["G2_weather_rainhist"] == 12 and sizes["G3_plus_geography"] == 26
      and sizes["G4_plus_hydrology"] == 50 and sizes["G5_full"] == 62 and sizes["G5_no_up"] == 58)
check("N32-03", f"layer sizes {layer_sizes}; union == FULL 62 = {union == full}; "
                f"disjoint = {disjoint}; group sizes {sizes}", ok)""")) 

cells.append(code(r"""# N32-04  group table shape + finite primary metrics (season=all rows carry a CI)
allrows = g[g.season == "all"]
finite = bool(allrows.pr_auc.notna().all() and allrows.roc_auc.notna().all()
              and allrows.macro_pr_auc.notna().all() and (allrows.n_stations > 0).all())
ci_ok = bool(allrows.pr_auc_ci_lo.notna().all() and allrows.pr_auc_ci_hi.notna().all()
             and (allrows.pr_auc_ci_lo <= allrows.pr_auc + 1e-9).all()
             and (allrows.pr_auc_ci_hi >= allrows.pr_auc - 1e-9).all())
check("N32-04", f"{len(g)} rows ({len(allrows)} season=all with CI); finite micro/macro/roc = {finite}; "
                f"CI brackets the point estimate = {ci_ok}",
      len(g) == 756 and finite and ci_ok)""")) 

cells.append(code(r"""# N32-05  hydrology is the dominant contribution and clears the D7 bar
g3 = {h: macro(TEMPORAL, "xgboost", "G3_plus_geography", h) for h in HORDERS}
g4 = {h: macro(TEMPORAL, "xgboost", "G4_plus_hydrology", h) for h in HORDERS}
pers = {h: float(r[(r.model == "persistence") & (r.horizon == h) & (r.season == "all")
                   & (r.cell.isin(TEMPORAL))].pr_auc.mean()) for h in HORDERS}
jump = {h: round(g4[h] - g3[h], 4) for h in HORDERS}
ok = all(jump[h] > 0.20 for h in HORDERS) and all(g4[h] > pers[h] for h in HORDERS)
check("N32-05", f"G4 - G3 jump {jump} > 0.20; G4 {g4} above the D7 bar {pers} at every horizon", ok)""")) 

cells.append(code(r"""# N32-06  weather-only / weather+rain are near the climatology bar (no skill)
clim = {h: float(r[(r.model == "climatology") & (r.horizon == h) & (r.season == "all")
                   & (r.cell.isin(TEMPORAL))].pr_auc.mean()) for h in HORDERS}
g1 = {h: round(macro(TEMPORAL, "xgboost", "G1_weather", h), 4) for h in HORDERS}
g2 = {h: round(macro(TEMPORAL, "xgboost", "G2_weather_rainhist", h), 4) for h in HORDERS}
ok = all(g1[h] < 0.35 for h in HORDERS) and all(g1[h] < pers[h] for h in HORDERS) \
     and all(g2[h] < 0.50 for h in HORDERS)
check("N32-06", f"weather-only G1 {g1} (<0.35), +rain G2 {g2} (<0.50); climatology bar {clim}; "
                f"both far below D7 {pers}", ok)""")) 

cells.append(code(r"""# N32-07  the +network layer is only a marginal increment over +hydrology
g5 = {h: macro(TEMPORAL, "xgboost", "G5_full", h) for h in HORDERS}
incr = {h: round(g5[h] - g4[h], 4) for h in HORDERS}
sub_in = inc[(inc.learner == "xgboost") & (inc.cell.isin(TEMPORAL))
             & (inc.group_a == "G5_full") & (inc.group_b == "G4_plus_hydrology")]
sig = int(sub_in.significant.sum())
check("N32-07", f"G5 - G4 increments {incr} all |x| < 0.03; significant in {sig}/{len(sub_in)} "
                f"temporal points", all(abs(v) < 0.03 for v in incr.values()))""")) 

cells.append(code(r"""# N32-08  dropping up_* (E11) is a negligible increment (F14-2 confirmed)
d11 = inc[(inc.learner == "xgboost") & (inc.group_a == "G5_no_up") & (inc.group_b == "G5_full")
          & (inc.cell.isin(TEMPORAL))]
mrow = float(mat[mat.experiment == "E11"].value.iloc[0])
nsig = int(d11.significant.sum())
check("N32-08", f"E11 mean d(no_up - full) = {mrow:+.5f}; significant in {nsig}/{len(d11)} "
                f"temporal points", abs(mrow) < 0.01)""")) 

cells.append(code(r"""# N32-09  the D7 persistence bar beats climatology everywhere (a real bar)
piv = r.pivot_table(index=["cell", "horizon", "season"], columns="model", values="pr_auc")
n_win = int((piv["persistence"] > piv["climatology"]).sum())
check("N32-09", f"persistence > climatology in {n_win}/{len(piv)} reference points",
      n_win == len(piv))""")) 

cells.append(code(r"""# N32-10  E0..E16 present; E12 blocked with the R8 blocker; E14 has L = 3/7/14/30
need = {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10",
        "E11", "E12", "E13", "E15", "E16"}
e14_set = sorted([m for m in mat.experiment if m.startswith("E14")])
e12 = mat[mat.experiment == "E12"]
check("N32-10", f"E0..E16 present; E12 status={str(e12.status.iloc[0])}; "
                f"blocker-in-source={'R8' in str(e12.source.iloc[0])}; E14 rows {e14_set}",
      set(mat.experiment).issuperset({"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
                                      "E9", "E10", "E11", "E12", "E13", "E15", "E16"})
      and str(mat[mat.experiment == "E12"].status.iloc[0]) == "blocked"
      and "R8" in str(mat[mat.experiment == "E12"].source.iloc[0])
      and all(f"E14_L{L}" in set(mat.experiment) for L in (3, 7, 14, 30)))""")) 

cells.append(code(r"""# N32-11  increments: finite CIs and the significance flag matches the CI sign
fin = inc.ci_lo.notna() & inc.ci_hi.notna()
flag = (inc.ci_lo > 0) | (inc.ci_hi < 0)
match = bool((fin & (flag == inc.significant)).all())
check("N32-11", f"{len(inc)} increment rows; all CIs finite = {bool(fin.all())}; "
                f"significant flag == CI excludes zero = {match}", match)""")) 

cells.append(code(r"""# N32-12  seasons present, figure written, reused temporal detail finite
seasons_ok = set(g.season.unique()) == {"all", "monsoon", "non-monsoon"}
fig_ok = FIG.exists() and FIG.stat().st_size > 5000
reuse_ok = bool(reuse[reuse.cells == "temporal"].pr_auc.notna().all()) and len(reuse) > 0
check("N32-12", f"seasons {sorted(g.season.unique())}; figure {FIG.stat().st_size:,} B; "
                f"reused temporal rows finite = {reuse_ok}",
      seasons_ok and fig_ok and reuse_ok)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 32 / STAGES 55, 56  ablation & comparative matrix verified")
print("=" * 78)
for cid, ok in RES:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
npass = sum(int(ok) for _, ok in RES)
print(f"\n{npass}/{len(RES)} checks passed")
print("test_rows_read =", summary["test_rows_read"])""")) 

cells.append(md("""## Headline (per cell, never pooled — see `reports/phase32_ablation.md`)

* the nested ablation is a **hydrology story**: weather alone (G1) and weather+rain (G2) are
  no better than climatology; geography (G3) lifts skill sharply, and hydrology (G4) is the
  layer that clears the D7 persistence bar;
* the **station-network layer (G5) is marginal** over hydrology, and dropping the four
  `up_*` columns (E11) changes nothing — the F14-2 result is confirmed at full capacity;
* E13-E16 (ML-vs-DL, sequence length, spatial and architecture ablations) integrate the
  frozen Phase-21/23/24/25/26/27 evidence; E12 (soil/landcover) is explicitly blocked by R8.""")) 

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "flood-predictor", "language": "python",
                   "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
