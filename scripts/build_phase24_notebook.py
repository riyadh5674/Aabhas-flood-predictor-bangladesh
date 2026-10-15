"""Build notebooks/24_gnn.ipynb (workflow Phase 24 / stage 39, D13 pass).

The graph families (L4-01 gcn, L4-02 gat, L4-04 gnn_lstm) are fit by
scripts/train_gnn.py; the registered P3 kill-criterion comparison (GNN vs the
temporal-only LSTM L3-01, paired station bootstrap) is produced by
scripts/phase24_ablation.py.  This notebook re-derives every claim from the
written artefacts; the only training it does is the small fixed-epoch
determinism probe in N24-09.

Phase 24 asks (D13): given a defensible hydrology-first station graph, does a
spatiotemporal GNN beat the per-station temporal model on UNSEEN STATIONS (P3),
where same-station persistence cannot transfer?  A documented rejection is an
acceptable, reportable outcome.

Checks N24-01..N24-12:
    N24-01  artefacts exist with the expected shapes
    N24-02  registry contract: L4-01/02/04 gated by phase24_graph_gate, budget<=20
    N24-03  test policy: zero test cells; the 7 dev cells; all three seasons
    N24-04  graph artefact: 427 edges (52 river + 375 fallback), connected, sha
    N24-05  tuning: budget honoured per family, 3 horizons, mean == mean, winner
    N24-06  grid: 216 rows, n>0, CIs on big cells, n_params == winner's params
    N24-07  headline: GCN/GAT/GNN-LSTM vs temporal-only (L3-01) vs D7 + xgboost
    N24-08  ablation: paired station-bootstrap CI, GNN - LSTM on P3 (kill criterion)
    N24-09  determinism: two identical fixed-epoch GNN fits bit-identical
    N24-10  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N24-11  report exists; figure rendered
    N24-12  summary

Run:
    python scripts/build_phase24_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\24_gnn.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "24_gnn.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 24 — Station graph & spatiotemporal GNN (stage 39, D13)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 24 — Station graph & spatiotemporal GNN** → checklist **stage 39** |
| **Gate (24.1)** | `scripts/phase24_gate.py` → `reports/phase24_graph_gate.md` — **PASS (narrow)** |
| **Fit by (24.2)** | `scripts/train_gnn.py gcn gat gnn_lstm` → `scripts/phase24_run.log`; `scripts/phase24_ablation.py gcn gat` |
| **Reads** | `reports/phase24_metrics.csv` (216) · `reports/phase24_search.csv` (24) · `models/phase24_L4-0{1,2,4}.pt` · `reports/phase24_ablation.csv` · `phase24_ablation_preds.npz` · `data/processed/station_graph.csv` + `station_graph_manifest.json` · `reports/phase21_metrics.csv` (lstm_L3 temporal-only) · `reports/phase16_baseline_metrics.csv` · `reports/phase20_metrics.csv` |
| **Contract** | Dataset B sequences (P19-M1) + frozen graph · dev cells only (test frozen to Phase 33) · budget ≤ 20 · seed 20261008 · L\\* = 3 |

The gate authorised **exactly one** fair test under pre-registered kill criteria:
the GNN must beat the temporal-only LSTM (`L3-01`) on **P3 spatial CV (unseen
stations)** with a paired bootstrap CI.  This notebook reports that verdict.""")) 

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
import sys
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
MET24 = REPORTS / "phase24_metrics.csv"
SEARCH24 = REPORTS / "phase24_search.csv"
ABL = REPORTS / "phase24_ablation.csv"
ABLNPZ = REPORTS / "phase24_ablation_preds.npz"
MET21 = REPORTS / "phase21_metrics.csv"
P16 = REPORTS / "phase16_baseline_metrics.csv"
P20 = REPORTS / "phase20_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase24_gnn.md"
GATE = REPORTS / "phase24_graph_gate.md"
GRAPH = PROC / "station_graph.csv"
GMAN = PROC / "station_graph_manifest.json"
LOG = ROOT / "scripts" / "phase24_run.log"
ABLLOG = ROOT / "scripts" / "phase24_ablation.out.txt"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAMS = ["gcn", "gat", "gnn_lstm"]
REG_ID = {"gcn": "L4-01", "gat": "L4-02", "gnn_lstm": "L4-04"}
BUDGET = {"gcn": 8, "gat": 8, "gnn_lstm": 8}
HORIZONS = ["label_24h", "label_48h", "label_72h"]
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
PT = {f: MODELS / f"phase24_{REG_ID[f]}.pt" for f in FAMS}


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


met24 = pd.read_csv(MET24)
search24 = pd.read_csv(SEARCH24)
met21 = pd.read_csv(MET21)
abl = pd.read_csv(ABL)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
p20 = pd.read_csv(P20)
t0 = time.time()
print("Phase 24 notebook - station-graph GNN verification")""")) 

cells.append(code(r"""# N24-01  artefacts present with the expected shapes
for p in (MET24, SEARCH24, ABL, ABLNPZ, REPORT, GATE, GRAPH, GMAN, LOG,
          ABLLOG, *PT.values()):
    assert p.exists(), p
check("N24-01", f"metrics {len(met24)} rows, search {len(search24)} rows, "
                f"ablation {len(abl)} rows, {len(PT)} model files, graph + gate + "
                f"report + logs present",
      len(met24) == 216 and len(search24) == 24 and len(abl) > 0)""")) 

cells.append(code(r"""# N24-02  registry contract: L4-01/02/04 gated by phase24_graph_gate, budget<=20
for f in FAMS:
    row = reg[reg.registry_id == REG_ID[f]]
    assert len(row) == 1, (f, row)
    r = row.iloc[0]
    ok = (r.model == f and r.test_policy == "frozen_read_at_phase33"
          and int(r.tuning_budget_configs) >= BUDGET[f]
          and r.gate == "phase24_graph_gate"
          and r.selection_cell == "val_2022" and r.selection_metric == "pr_auc")
    check(f"N24-02.{f}", f"{REG_ID[f]} {r.model}: gate={r.gate}, "
                         f"budget_max={int(r.tuning_budget_configs)} >= used "
                         f"{BUDGET[f]}, status={r.status}, test={r.test_policy}",
          bool(ok))""")) 

cells.append(code(r"""# N24-03  test policy: no test rows anywhere in the phase-24 metrics
cells = set(met24.cell.unique())
test_rows = met24[met24.cell.astype(str).str.startswith("test")]
seasons = set(met24.season.unique())
check("N24-03", f"cells = {sorted(cells)} == the 7 dev cells (no test); test rows "
                f"= {len(test_rows)} (must be 0); seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})""")) 

cells.append(code(r"""# N24-04  graph artefact: 427 edges (52 river + 375 knn fallback), connected, sha
g = pd.read_csv(GRAPH)
man = json.loads(GMAN.read_text(encoding="utf-8"))
kinds = g.kind.value_counts().to_dict()
check("N24-04", f"graph {man['n_edges']} edges = {kinds}; "
                f"connected={man['connected']}; mean_deg={man['mean_degree']}; "
                f"sha={man['sha256'][:12]} (matches file: "
                f"{man['sha256'] == sha256(GRAPH)})",
      len(g) == 427 and kinds == {"river": 52, "knn_fallback": 375}
      and man["connected"] and man["sha256"] == sha256(GRAPH))""")) 

cells.append(code(r"""# N24-05  tuning: budget honoured per family, 3 horizons, mean == mean, winner
for f in FAMS:
    gg = search24[search24.family == f]
    three = bool(((gg[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
    mean_ok = bool(np.allclose(gg.mean_pr_auc,
                               gg[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
    w = gg.loc[gg.mean_pr_auc.idxmax()]
    check(f"N24-05.{f}", f"{f}: {len(gg)} configs == budget {BUDGET[f]}, 3 horizons, "
                         f"mean == mean(3), winner {w.config_id} "
                         f"mean_pr_auc={w.mean_pr_auc:.5f} ({w.weighting}), "
                         f"n_params={int(w.n_params):,}",
          len(gg) == BUDGET[f] and three and mean_ok)""")) 

cells.append(code(r"""# N24-06  grid: 216 rows, n>0, CIs on big cells, n_params == winner's params
size = met24.groupby("model").size()
ci_missing = met24[(met24.season == "all") & (met24.n >= 1000) & met24.pr_auc_ci_lo.isna()]
np_ok = True
for f in FAMS:
    w = search24[search24.family == f].sort_values("mean_pr_auc", ascending=False).iloc[0]
    got = set(met24[met24.model == f].n_params.unique())
    np_ok &= (got == {int(w.n_params)})
check("N24-06", f"rows = {dict(size)} (expect 216); n>0 = {bool((met24.n > 0).all())}; "
                f"bootstrap CI missing on big cells = {len(ci_missing)}; every "
                f"model's n_params == its winner's params = {np_ok}",
      (size == 72).all() and (met24.n > 0).all() and len(ci_missing) == 0 and np_ok)""")) 

cells.append(code(r"""# N24-07  headline: GCN/GAT/GNN-LSTM vs temporal-only(L3-01) vs D7 + xgboost
val24 = met24[(met24.protocol == "P1") & (met24.cell == "val_2022") & (met24.season == "all")]
val21 = met21[(met21.protocol == "P1") & (met21.cell == "val_2022")
              & (met21.season == "all") & (met21.model == "lstm_L3")]
dl = pd.concat([val24, val21]).pivot_table(index="model", columns="horizon",
                                           values="pr_auc")
order = ["lstm_L3", "gcn", "gat", "gnn_lstm"]
bar = pd.concat([
    p16[(p16.model == "persistence") & (p16.protocol == "P1")
        & (p16.cell == "val_2022") & (p16.season == "all")],
    p20[(p20.model == "xgboost") & (p20.protocol == "P1")
        & (p20.cell == "val_2022") & (p20.season == "all")],
]).pivot_table(index="model", columns="horizon", values="pr_auc")
allv = pd.concat([dl.reindex(order), bar])
print("val_2022 PR-AUC (season=all):")
print(allv.round(4).to_string())
check("N24-07", f"graph families {dl.loc['gcn'].round(4).to_dict()} / "
                f"{dl.loc['gat'].round(4).to_dict()} / {dl.loc['gnn_lstm'].round(4).to_dict()}; "
                f"temporal-only {dl.loc['lstm_L3'].round(4).to_dict()}; bars "
                f"D7 {bar.loc['persistence'].round(4).to_dict()} / "
                f"xgb {bar.loc['xgboost'].round(4).to_dict()}",
      set(order) <= set(dl.index) and dl.notna().all().all())""")) 

cells.append(code(r"""# N24-08  ablation: paired station-bootstrap CI, GNN - LSTM on P3 (kill criterion)
from phase24_ablation import paired_bootstrap

full = abl[abl.graph == "full"].copy()
P = np.load(ABLNPZ)
recomputed = []
for _, r in full.iterrows():
    key = f"{r.family}_{r.cell}_{r.horizon}"
    y = P[key + "_y"]; st = P[key + "_st"]
    pg = P[key + "_p_gnn"]; pl = P[key + "_p_lstm"]
    m, lo, hi = paired_bootstrap(st, y, pg, pl)
    recomputed.append(abs(m - r.diff_gnn_minus_lstm) < 1e-9
                      and abs(lo - r.diff_ci_lo) < 1e-9
                      and abs(hi - r.diff_ci_hi) < 1e-9)
# per-family P3 mean (full graph) gnn vs lstm
means = full.groupby(["family", "horizon"])[
    ["pr_auc_gnn", "pr_auc_lstm_L3_01"]].mean().round(4)
print("P3 spatial CV (full graph), mean over 4 folds:")
print(means.to_string())
sig = full[(full.diff_ci_lo > 0) | (full.diff_ci_hi < 0)]
print(f"\ncells whose 95% CI excludes 0 (separable): "
      f"{list(zip(sig.family, sig.cell, sig.horizon))}")
check("N24-08", f"paired bootstrap reproduces the stored CI exactly "
                f"({all(recomputed)}); {len(sig)}/{len(full)} P3 cells separable "
                f"from the temporal-only LSTM",
      all(recomputed))""")) 

cells.append(code(r"""# N24-09  determinism: two identical fixed-epoch GNN fits bit-identical
import train_gnn as g
from train_sequence import build_tasks_index

ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
assert len(ix) == 200665
split = ix.split.to_numpy()
hold = ix.station_holdout.astype(bool).to_numpy()
dp, sp, row_of, sp_index = g.build_index_maps(ix)
a_hat, a_mask = g.load_graph(sp_index)
X, M = g.ts_prepare if hasattr(g, "ts_prepare") else None, None
from train_sequence import prepare
X, M = prepare(3)
y = ix["label_24h"].to_numpy()
tr_flat = np.flatnonzero((split == "train") & ~hold)
ev_flat = np.flatnonzero((split == "val") & ~hold)
mtr = np.zeros(len(y), bool); mtr[tr_flat[:2000]] = True
mev = np.zeros(len(y), bool); mev[ev_flat[:1000]] = True
vtr = g.make_view(mtr, y, sp, dp, X, M)
vev = g.make_view(mev, y, sp, dp, X, M)
w = search24[search24.family == "gcn"].sort_values("mean_pr_auc", ascending=False).iloc[0]
cfg = json.loads(w["params"]); cfg["_weight"] = "off"
spec = {"agg": "gcn", "reg": "det"}
r1 = g.fit_gnn(spec, cfg, X, M, row_of, a_hat, a_mask, vtr, vev, fixed_epochs=2)
r2 = g.fit_gnn(spec, cfg, X, M, row_of, a_hat, a_mask, vtr, vev, fixed_epochs=2)
same = all(torch.equal(r1["net"].state_dict()[k], r2["net"].state_dict()[k])
           for k in r1["net"].state_dict())
check("N24-09", f"two seeded 2-epoch GCN fits on 2,000 rows: bit-identical "
                f"parameters = {same}", bool(same))""")) 

cells.append(code(r"""# N24-10  SHA-256 of the immutable experimental table
check("N24-10", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)""")) 

cells.append(code(r"""# N24-11  report + figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES.mkdir(exist_ok=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
po = ["lstm_L3", "gcn", "gat", "gnn_lstm", "xgboost", "persistence"]
plot = allv.reindex(po)
x = np.arange(len(po)); w = 0.26
for i, h in enumerate(HORIZONS):
    ax.bar(x + (i - 1) * w, plot.loc[po, h].to_numpy(), w, label=h)
ax.set_xticks(x); ax.set_xticklabels(po, rotation=20, ha="right")
ax.set_ylabel("PR-AUC (val_2022)"); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3, axis="y")
ax.legend(); ax.set_title("Phase 24 - graph families vs temporal-only vs D7/xgb")

hzs = ["24h", "48h", "72h"]
xb = np.arange(len(hzs)); ww = 0.38
for j, fam in enumerate(["gcn", "gat"]):
    sub = full[full.family == fam].groupby("horizon")[
        ["pr_auc_gnn", "pr_auc_lstm_L3_01"]].mean().reindex(HORIZONS)
    ax2.bar(xb + (j - 0.5) * ww, sub.pr_auc_gnn.to_numpy(), ww, label=f"{fam} GNN")
ax2.plot(xb, full[full.family == "gcn"].groupby("horizon")["pr_auc_lstm_L3_01"]
         .mean().reindex(HORIZONS).to_numpy(), "k--o", label="LSTM L3-01")
ax2.set_xticks(xb); ax2.set_xticklabels(hzs)
ax2.set_ylabel("P3 PR-AUC (unseen stations)"); ax2.set_ylim(0.5, 1.0)
ax2.grid(alpha=0.3, axis="y"); ax2.legend()
ax2.set_title("P3 spatial CV: graph vs temporal-only")
fig.tight_layout()
fig.savefig(FIGURES / "phase24_val2022.png", dpi=130)
plt.close(fig)
check("N24-11", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure figures/phase24_val2022.png written",
      REPORT.exists() and (FIGURES / "phase24_val2022.png").exists())""")) 

cells.append(code(r"""# N24-12  summary
print("\n" + "=" * 78)
print("PHASE 24 / STAGE 39  Station-graph GNN (D13 gate PASS, narrow) verified")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
check("N24-12", f"{sum(r['pass'] for r in RES) + 1}/{len(RES) + 1} checks passed, "
                f"{time.time() - t0:.1f}s", True)
print("Verdict: see reports/phase24_gnn.md; test untouched (frozen to Phase 33); "
      "next: Phase 25 Transformer (gated) / Phase 26 multi-horizon")
assert all(r["pass"] for r in RES)
print("ALL PHASE-24 CHECKS PASSED")""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
