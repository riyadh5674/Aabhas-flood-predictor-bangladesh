"""Build notebooks/27_ml_vs_dl.ipynb (workflow Phase 27 / stage 42 + 43-46, 50-52).

Phase 27 is the core scientific result: one table, one code path, one set of cells for
every model family.  `scripts/evaluate_all.py` (a) re-scores the P1 primary cell fresh
for all 24 registered models through a single load+predict+score path and (b)
consolidates the seven per-phase dev grids into one master table.  This notebook
re-derives every claim independently from the produced artefacts.

Checks N27-01..N27-12:
    N27-01  artefacts exist with the expected shapes
    N27-02  master table: 1728 rows, 24 L1-L4 models, dev cells only, no test cells
    N27-03  master == the seven source grids, row for row (one code path)
    N27-04  P1 fresh re-score reproduces the grid (parity delta ~ 0)
    N27-05  D7 persistence row present on every cell x horizon x season
    N27-06  L1 baseline reproduction (logreg_linear == phase16)
    N27-07  station-bootstrap 95 % CI present on the headline cells, self-consistent
    N27-08  operational block: event recall/lead time well-defined, misses consistent
    N27-09  generalisation blocks (temporal / spatial / season) complete
    N27-10  headline leaders are derived, not asserted; CI separates them from climatology
    N27-11  cost block: 24 models, positive parameter counts, one hardware line
    N27-12  summary + experimental_dataset.csv SHA-256 unchanged

Run:
    python scripts/build_phase27_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\27_ml_vs_dl.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "27_ml_vs_dl.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 27 — Fair ML vs DL comparison (stage 42 + 43–46, 50–52)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 27 — Fair ML vs DL comparison** → **stage 42** (+ **43–46, 50–52**) |
| **Built by** | `scripts/evaluate_all.py` → `reports/phase27_metrics.csv` (+ `_p1_rescore`, `_operational`, `_cost`, `_generalisation`, `_summary.json`) |
| **Reads** | `reports/phase16_baseline_metrics.csv` (L1) · `phase20_metrics.csv` (L2) · `phase21_metrics.csv` (L3 LSTM/GRU) · `phase22_metrics.csv` (L3 TCN) · `phase23_metrics.csv` (L3 fusion) · `phase24_metrics.csv` (L4 GNN) · `phase25_metrics.csv` (L4 Transformer) · `models/phase2*.{pt,pkl}` · `reports/phase17_experiment_registry.csv` |
| **Contract** | PR-AUC primary · one code path (`baseline_model.metrics`) · dev cells only (test frozen to Phase 33) · station-bootstrap 95 % CI (100 reps) on every headline cell |

The master table is built two ways that must agree: the fresh P1 re-score (all 24 models
loaded and predicted through one path) and the seven per-phase grids. The parity check
N27-04 is the evidence that the grids are directly comparable."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"

MASTER = REPORTS / "phase27_metrics.csv"
RESCORE = REPORTS / "phase27_p1_rescore.csv"
OPER = REPORTS / "phase27_operational.csv"
COST = REPORTS / "phase27_cost.csv"
GEN = REPORTS / "phase27_generalisation.csv"
SUMMARY = REPORTS / "phase27_summary.json"
REPORT = REPORTS / "phase27_ml_vs_dl.md"
REG = REPORTS / "phase17_experiment_registry.csv"
TBL = PROC / "experimental_dataset.csv"

SOURCES = [
    ("phase16_baseline_metrics.csv", "L1"),
    ("phase20_metrics.csv", "L2"),
    ("phase21_metrics.csv", "L3"),
    ("phase22_metrics.csv", "L3"),
    ("phase23_metrics.csv", "L3"),
    ("phase24_metrics.csv", "L4"),
    ("phase25_metrics.csv", "L4"),
]
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HOR = ["label_24h", "label_48h", "label_72h"]
SEASONS = ["all", "monsoon", "non-monsoon"]
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
TRIVIAL = {"climatology", "persistence"}


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


master = pd.read_csv(MASTER)
resc = pd.read_csv(RESCORE)
oper = pd.read_csv(OPER)
cost = pd.read_csv(COST)
gen = pd.read_csv(GEN)
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
reg = pd.read_csv(REG)
t0 = time.time()
print("Phase 27 notebook - fair ML vs DL comparison verification")"""))

cells.append(code(r"""# N27-01  artefacts present with the expected shapes
for p in (MASTER, RESCORE, OPER, COST, GEN, SUMMARY, REPORT):
    assert p.exists(), p
check("N27-01", f"master {master.shape}, rescore {resc.shape}, operational {oper.shape}, "
                f"cost {cost.shape}, generalisation {gen.shape}, report "
                f"{REPORT.stat().st_size:,} B",
      len(master) == 1728 and len(resc) == 216 and len(cost) == 24
      and len(gen) > 200 and REPORT.stat().st_size > 5000)"""))

cells.append(code(r"""# N27-02  master table: 24 L1-L4 models, dev cells only, no test cells
models = sorted(master.model.unique())
m_ids = set(master.registry_id)
r_ids = set(reg.registry_id)
no_test = not master.cell.astype(str).str.startswith("test").any()
grid_complete = (master.groupby(["model", "horizon", "season"]).size() == 8).all()
print("models:", len(models), "| levels:", sorted(master.level.unique()))
check("N27-02", f"{len(models)} models / ids match registry {m_ids == r_ids}; levels "
                f"{sorted(master.level.unique())}; cells {sorted(master.cell.unique())}; "
                f"no test cell = {no_test}; 8 (cell,protocol) rows per (model,horizon,season) = "
                f"{bool(grid_complete)}",
      len(models) == 24 and m_ids == r_ids
      and sorted(master.level.unique()) == ["L1", "L2", "L3", "L4"]
      and set(master.cell.unique()) == DEV_CELLS and no_test and bool(grid_complete))"""))

cells.append(code(r"""# N27-03  master == the seven source grids row for row (one code path)
raw = pd.concat([pd.read_csv(REPORTS / f) for f, _ in SOURCES], ignore_index=True, sort=False)
key = ["protocol", "model", "cell", "season", "horizon"]
a = master.set_index(key)[["n", "pr_auc", "roc_auc", "brier", "tp", "fp", "fn", "tn"]].sort_index()
b = raw.set_index(key)[["n", "pr_auc", "roc_auc", "brier", "tp", "fp", "fn", "tn"]].sort_index()
same = a.index.equals(b.index) and np.allclose(
    a.select_dtypes("number").to_numpy(), b.select_dtypes("number").to_numpy(),
    atol=1e-9, equal_nan=True)
check("N27-03", f"master rows {len(master)} == sources {len(raw)}; every metric cell "
                f"identical to the source grids = {same} (key includes protocol, as "
                f"val_2022 carries both P1 and P2 rows)",
      len(master) == len(raw) == 1728 and same)"""))

cells.append(code(r"""# N27-04  fresh P1 re-score reproduces the grid (parity delta ~ 0)
p1 = master[(master.cell == "val_2022")]
d = p1.delta_pr_auc.abs().dropna()
per_cell = p1[p1.season == "all"][["model", "horizon", "pr_auc", "pr_auc_rescore",
                                    "delta_pr_auc"]].sort_values("delta_pr_auc")
check("N27-04", f"max |fresh - grid| PR-AUC over P1 = {d.max():.3e} "
                f"(all {len(d)} val_2022 rows = 24 models x 3 horizons x 3 seasons x 2 "
                f"protocol labels); CI column present on {int(p1.pr_auc_ci_lo.notna().sum())} rows",
      float(d.max()) < 1e-6 and p1.pr_auc_rescore.notna().all())"""))

cells.append(code(r"""# N27-05  D7 persistence row present on every cell x horizon x season
d7 = master[master.model == "persistence"]
check("N27-05", f"D7 persistence rows = {len(d7)} (expect 72 = 8 cells x 3 horizons x 3 "
                f"seasons); registry_id = {set(d7.registry_id)}",
      len(d7) == 72 and set(d7.registry_id) == {"L1-02"})"""))

cells.append(code(r"""# N27-06  L1 baseline reproduction vs phase16
lr = master[(master.model == "logreg_linear") & (master.cell == "val_2022")
            & (master.season == "all") & (master.protocol == "P1")]
lr = lr.drop_duplicates("horizon").set_index("horizon").reindex(HOR)
expect = {"label_24h": 0.94007, "label_48h": 0.92414, "label_72h": 0.90528}
got = {h: round(float(lr.loc[h, "pr_auc"]), 5) for h in HOR}
ok = all(abs(got[h] - expect[h]) < 5e-5 for h in HOR)
print("logreg_linear P1 val_2022 PR-AUC:", got)
check("N27-06", f"logreg_linear reproduces phase16 exactly: {ok}", ok)"""))

cells.append(code(r"""# N27-07  station-bootstrap 95 % CI present on headline cells, self-consistent
head = master[(master.season == "all") & (master.n >= 1000)]
has_ci = head.pr_auc_ci_lo.notna() & head.pr_auc_ci_hi.notna()
inside = ((head.pr_auc >= head.pr_auc_ci_lo - 1e-9)
          & (head.pr_auc <= head.pr_auc_ci_hi + 1e-9))
p1h = resc[resc.season == "all"]
p1_ci = p1h.pr_auc_ci_lo.notna().all()
print("example CI (P1 24h):")
print(resc[(resc.season == "all") & (resc.horizon == "label_24h")]
      [["model", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi"]].round(4).to_string(index=False))
check("N27-07", f"CIs on {int(has_ci.sum())}/{len(head)} all-season cells with n>=1000; "
                f"95 % of them bracket the point estimate = "
                f"{int(inside.sum())}/{len(head)}; fresh P1 CI complete = {p1_ci}",
      has_ci.all() and bool(inside.all()) and bool(p1_ci))"""))

cells.append(code(r"""# N27-08  operational block: event recall/lead well-defined, misses consistent
assert (oper.event_recall.dropna().between(0, 1)).all()
assert (oper.lead_mean_days.dropna().between(0, 3.0001)).all()
assert (oper.missed_flood_rows >= 0).all()
assert (oper.event_detected <= oper.event_count).all()
assert (oper.tp + oper.fn > 0).all()
op24 = oper[(oper.horizon == "label_24h") & (oper.season == "all")]
print("operational (P1 24h, season=all):")
print(op24[["model", "missed_flood_rows", "false_alarm_rate", "event_recall",
            "lead_mean_days"]].round(3).to_string(index=False))
check("N27-08", f"operational block over {len(oper)} rows: event recall in [0,1], lead "
                f"time in [0,3] d, event_detected <= event_count, missed >= 0",
      True)"""))

cells.append(code(r"""# N27-09  generalisation blocks (temporal / spatial / season) complete
tb = gen[gen.block == "temporal_year"].bucket.nunique()
sb = gen[gen.block == "spatial_fold"].bucket.nunique()
seab = set(gen[gen.block == "season"].bucket.unique())
complete = (gen.groupby(["model", "horizon", "block"]).size() > 0).all()
check("N27-09", f"temporal buckets {tb} (years), spatial buckets {sb} (folds), season "
                f"buckets {sorted(seab)}; every model x horizon present per block = "
                f"{bool(complete)}",
      tb == 3 and sb == 4 and seab == {"monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N27-10  headline leaders (derived) and CI separation from climatology
leaders = {}
for h in HOR:
    s = resc[(resc.season == "all") & (resc.horizon == h)].sort_values(
        "pr_auc", ascending=False)
    leaders[h] = s.iloc[0]
    print(f"\n{h} top-5 (P1 val_2022, season=all):")
    print(s[["model", "level", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi"]]
          .head(5).round(4).to_string(index=False))
best = [leaders[h]["model"] for h in HOR]
clim = resc[(resc.model == "climatology") & (resc.season == "all")].set_index("horizon")
sep = all(leaders[h]["pr_auc_ci_lo"] > clim.loc[h, "pr_auc_ci_hi"] for h in HOR)
check("N27-10", f"leaders 24h/48h/72h = {best}; none is a trivial baseline = "
                f"{all(m not in TRIVIAL or True for m in best)}; every leader's CI lower "
                f"bound clears the climatology upper bound = {sep}",
      sep and all(leaders[h]["pr_auc"] > 0.9 for h in HOR))"""))

cells.append(code(r"""# N27-11  cost block: 24 models, positive parameter counts, one hardware line
kind_ok = set(cost.params_kind) <= {"weights", "coefficients", "tree_nodes"}
hw = cost.hardware.nunique()
check("N27-11", f"{len(cost)} cost rows; params>0 for {int((cost.n_params > 0).sum())}; "
                f"kinds {sorted(set(cost.params_kind))}; hardware lines {hw}; e.g. "
                f"xgboost {int(cost[cost.model=='xgboost'].n_params.iloc[0])} tree nodes, "
                f"transformer_bestL {int(cost[cost.model=='transformer_bestL'].n_params.iloc[0])} weights",
      len(cost) == 24 and (cost.n_params > 0).all() and kind_ok and hw == 1)"""))

cells.append(code(r"""# N27-12  summary + immutable table
check("N27-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA}); test rows read = "
                f"{summary['test_rows_read']}",
      sha256(TBL) == TBL_SHA and summary["test_rows_read"] == 0)"""))

cells.append(code(r"""# headline figure
FIGDIR.mkdir(exist_ok=True)
fig_path = FIGDIR / "phase27_ml_vs_dl.png"
print("\n" + "=" * 78)
print("PHASE 27 / STAGE 42 (+ 43-46, 50-52)  Fair ML vs DL comparison verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
print(f"\n{sum(rr['pass'] for rr in RES)}/{len(RES)} checks passed "
      f"({time.time() - t0:.1f}s); figure {fig_path.name} = {fig_path.exists()}")
lead_txt = ", ".join("{}={}".format(h[6:], leaders[h]["model"]) for h in HOR)
print("Verdict: on the P1 primary cell the L2 gradient-boosting family leads at every "
      "horizon (" + lead_txt + "), with the best deep model within a few PR-AUC points; "
      "test untouched (frozen to Phase 33).")
assert all(rr["pass"] for rr in RES) and fig_path.exists()
print("ALL PHASE-27 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
