"""Build notebooks/25_transformer.ipynb (workflow Phase 25 / stage 40, D13 pass).

The temporal Transformer (L4-03) is fit by scripts/train_transformer.py; the
registered head-to-head against the temporal baselines (paired station
bootstrap) is produced by scripts/phase25_ablation.py.  This notebook re-derives
every claim from the written artefacts; the only training it does is the small
fixed-epoch determinism probe in N25-09.

Phase 25 asks (D13): given a solid temporal baseline (LSTM/GRU/TCN) and a
answered static-context question (Phase 23 fusion), does a temporal Transformer
(self-attention over the past-L days) improve 24/48/72 h PR-AUC — or does it tie
/ lose?  A documented negative result is an acceptable, reportable outcome.

Checks N25-01..N25-12:
    N25-01  artefacts exist with the expected shapes
    N25-02  registry contract: L4-03 gated by phase25_transformer_gate, budget<=20
    N25-03  test policy: zero test cells; the 7 dev cells; all three seasons
    N25-04  gate report: PASS (narrow), preconditions + pre-registered rules
    N25-05  tuning: budget honoured, 3 horizons, mean == mean, winner, window L
    N25-06  grid: 72 rows, n>0, CIs on big cells, n_params == winner's params
    N25-07  headline: transformer vs LSTM/TCN/fusion vs D7 + xgboost (val_2022)
    N25-08  ablation: paired station-bootstrap CI, tx vs LSTM/TCN on P1 + P3
    N25-09  determinism: two identical fixed-epoch Transformer fits bit-identical
    N25-10  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N25-11  report exists; figure rendered
    N25-12  summary

Run:
    python scripts/build_phase25_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\25_transformer.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "25_transformer.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 25 — Temporal Transformer (stage 40, D13)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 25 — Transformer models** → checklist **stage 40** |
| **Gate (25.1)** | `scripts/phase25_gate.py` → `reports/phase25_transformer_gate.md` — **PASS (narrow, pre-registered)** |
| **Fit by (25.2)** | `scripts/train_transformer.py` (L4-03) → `scripts/phase25_run.log`; `scripts/phase25_ablation.py` |
| **Reads** | `reports/phase25_metrics.csv` (72) · `reports/phase25_search.csv` (16) · `models/phase25_L4-03.pt` · `reports/phase25_ablation.csv` · `phase25_ablation_preds.npz` · `reports/phase25_transformer_gate.{md,json}` · `reports/phase21_metrics.csv` / `phase22_metrics.csv` / `phase23_metrics.csv` · `reports/phase16_baseline_metrics.csv` · `reports/phase20_metrics.csv` |
| **Contract** | Dataset B sequences (P19-M1) · dev cells only (test frozen to Phase 33) · budget ≤ 20 · seed 20261008 · window L in {3,7,14,30} |

The gate authorised **exactly one** fair test under pre-registered acceptance/kill
criteria: the temporal Transformer must beat the temporal baselines on `val_2022`
(primary) and must not lose on **P3 spatial CV (unseen stations)** (secondary),
each with a paired station-bootstrap CI. This notebook reports that verdict."""))

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
MET25 = REPORTS / "phase25_metrics.csv"
SEARCH25 = REPORTS / "phase25_search.csv"
ABL = REPORTS / "phase25_ablation.csv"
ABLNPZ = REPORTS / "phase25_ablation_preds.npz"
GATE_MD = REPORTS / "phase25_transformer_gate.md"
GATE_JSON = REPORTS / "phase25_transformer_gate.json"
MET21 = REPORTS / "phase21_metrics.csv"
MET22 = REPORTS / "phase22_metrics.csv"
MET23 = REPORTS / "phase23_metrics.csv"
P16 = REPORTS / "phase16_baseline_metrics.csv"
P20 = REPORTS / "phase20_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase25_transformer.md"
SEQIX = PROC / "sequence_index.csv"
PT = MODELS / "phase25_L4-03.pt"
LOG = ROOT / "scripts" / "phase25_run.log"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAM = "transformer_bestL"
REG_ID = "L4-03"
BUDGET = 16
HORIZONS = ["label_24h", "label_48h", "label_72h"]
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}


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


met25 = pd.read_csv(MET25)
search25 = pd.read_csv(SEARCH25)
abl = pd.read_csv(ABL)
met21 = pd.read_csv(MET21)
met22 = pd.read_csv(MET22)
met23 = pd.read_csv(MET23)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
p20 = pd.read_csv(P20)
gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
t0 = time.time()
print("Phase 25 notebook - temporal Transformer verification")"""))

cells.append(code(r"""# N25-01  artefacts present with the expected shapes
for p in (MET25, SEARCH25, ABL, ABLNPZ, REPORT, GATE_MD, GATE_JSON, LOG, PT):
    assert p.exists(), p
check("N25-01", f"metrics {len(met25)} rows, search {len(search25)} rows, "
                f"ablation {len(abl)} rows, model {PT.stat().st_size:,} B, "
                f"gate + report + log present",
      len(met25) == 72 and len(search25) == BUDGET and len(abl) == 12)"""))

cells.append(code(r"""# N25-02  registry contract: L4-03 gated by phase25_transformer_gate, budget<=20
row = reg[reg.registry_id == REG_ID]
assert len(row) == 1, row
r = row.iloc[0]
ok = (r.model == "transformer_temporal" and r.test_policy == "frozen_read_at_phase33"
      and int(r.tuning_budget_configs) >= BUDGET
      and r.gate == "phase25_transformer_gate" and r.status == "planned"
      and r.selection_cell == "val_2022" and r.selection_metric == "pr_auc")
check(f"N25-02", f"{REG_ID} {r.model}: gate={r.gate}, status={r.status}, "
                 f"budget_max={int(r.tuning_budget_configs)} >= used {BUDGET}, "
                 f"test={r.test_policy}", bool(ok))"""))

cells.append(code(r"""# N25-03  test policy: no test rows anywhere in the phase-25 metrics
cells = set(met25.cell.unique())
test_rows = met25[met25.cell.astype(str).str.startswith("test")]
seasons = set(met25.season.unique())
check("N25-03", f"cells = {sorted(cells)} == the 7 dev cells (no test); test rows "
                f"= {len(test_rows)} (must be 0); seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N25-04  gate report: PASS (narrow), preconditions + pre-registered rules
dec = gate["decision"]
rules = gate["rules"]
check("N25-04", f"gate {gate['registry']['id']} / {gate['registry']['gate']}: "
                f"preconditions_ok={gate['preconditions_ok']}, L*={gate['lstar']}, "
                f"decision='{dec[:40]}...'; acceptance/kill rules "
                f"{sorted(rules)} recorded before any Transformer was fit",
      gate["preconditions_ok"] and dec.startswith("PASS")
      and set(rules) == {"primary", "secondary", "kill", "expectation"}
      and GATE_MD.stat().st_size > 1000)"""))

cells.append(code(r"""# N25-05  tuning: budget honoured, 3 horizons, mean == mean, winner, window L
g = search25[search25.family == FAM]
assert len(g) == BUDGET
three = bool(((g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
mean_ok = bool(np.allclose(g.mean_pr_auc,
                           g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
win = g.loc[g.mean_pr_auc.idxmax()]
win_cfg = json.loads(win["params"])
win_L = int(win_cfg["L"])
Ls = sorted({int(json.loads(s)["L"]) for s in g["params"]})
check("N25-05", f"{FAM}: {len(g)} configs == budget {BUDGET}, 3 horizons, mean == "
                f"mean(3); windows swept {Ls}; winner {win.config_id} L={win_L} "
                f"mean_pr_auc={win.mean_pr_auc:.5f} ({win.weighting}), "
                f"n_params={int(win.n_params):,}, epochs={win.best_epochs}",
      len(g) == BUDGET and three and mean_ok and set(Ls) <= {3, 7, 14, 30}
      and int(g.n_configs_budget.unique()[0]) == BUDGET)"""))

cells.append(code(r"""# N25-06  grid: 72 rows, n>0, CIs on big cells, n_params == winner's params
size = met25.groupby("model").size()
ci_missing = met25[(met25.season == "all") & (met25.n >= 1000) & met25.pr_auc_ci_lo.isna()]
got = set(met25[met25.model == FAM].n_params.unique())
np_ok = (got == {int(win.n_params)})
check("N25-06", f"rows = {dict(size)} (expect 72); n>0 = {bool((met25.n > 0).all())}; "
                f"bootstrap CI missing on big cells = {len(ci_missing)}; "
                f"n_params == winner's params ({int(win.n_params):,}) = {np_ok}",
      (size == 72).all() and (met25.n > 0).all() and len(ci_missing) == 0 and np_ok)"""))

cells.append(code(r"""# N25-07  headline: transformer vs LSTM/TCN/fusion vs D7 + xgboost (val_2022)
def val2022(df, model):
    return df[(df.protocol == "P1") & (df.cell == "val_2022")
              & (df.season == "all") & (df.model == model)]

tx = val2022(met25, FAM).pivot_table(index="model", columns="horizon", values="pr_auc")
base = pd.concat([val2022(met21, "lstm_L3"), val2022(met22, "tcn_bestL"),
                  val2022(met23, "fusion_L3")]).pivot_table(
    index="model", columns="horizon", values="pr_auc")
bar = pd.concat([
    val2022(p16, "persistence"),
    val2022(p20, "xgboost"),
]).pivot_table(index="model", columns="horizon", values="pr_auc")
order = [FAM, "lstm_L3", "tcn_bestL", "fusion_L3"]
allv = pd.concat([tx, base]).reindex(order)
print("val_2022 PR-AUC (season=all):")
print(pd.concat([allv, bar]).round(4).to_string())
best_temporal = base.loc[["lstm_L3", "tcn_bestL"]].mean().idxmax()
challenger = base.loc[["lstm_L3", "tcn_bestL"]].mean()
check("N25-07", f"transformer {allv.loc[FAM].round(4).to_dict()} vs temporal "
                f"baselines {challenger.round(4).to_dict()}; bars D7 "
                f"{bar.loc['persistence'].round(4).to_dict()} / xgb "
                f"{bar.loc['xgboost'].round(4).to_dict()}",
      set(order) <= set(allv.index) and allv.notna().all().all()
      and bar.notna().all().all())"""))

cells.append(code(r"""# N25-08  ablation: paired station-bootstrap CI, tx vs LSTM/TCN on P1 + P3
from phase25_ablation import paired_bootstrap

ix = pd.read_csv(SEQIX, parse_dates=["date"])
assert len(ix) == 200665
st_col = ix.station_id.to_numpy()
split = ix.split.to_numpy()
fold = ix.spatial_fold.to_numpy()
hold = ix.station_holdout.astype(bool).to_numpy()


def p1_y_st(h):
    yv = ix[f"label_{h}"].to_numpy()
    m = (split == "val") & ~hold & ~np.isnan(yv)
    return yv[m].astype(int), st_col[m]


def p3_y_st(h):
    yv = ix[f"label_{h}"].to_numpy()
    ys, ss = [], []
    for k in (1, 2, 3, 4):
        m = (split == "train") & (fold == k) & ~np.isnan(yv)
        ys.append(yv[m].astype(int)); ss.append(st_col[m])
    return np.concatenate(ys), np.concatenate(ss)


P = np.load(ABLNPZ)
recomputed = []
for _, rr in abl.iterrows():
    base_name = rr.comparison.replace("transformer_vs_", "")
    h = rr.horizon
    if rr.protocol == "P1":
        y, st = p1_y_st(h)
        pt = P[f"P1_transformer_{h}"]
        pb = P[f"P1_{base_name}_{h}"]
    else:
        y, st = p3_y_st(h)
        pt = np.concatenate([P[f"P3_transformer_spatial_fold_{k}_{h}"] for k in (1, 2, 3, 4)])
        pb = np.concatenate([P[f"P3_{base_name}_spatial_fold_{k}_{h}"] for k in (1, 2, 3, 4)])
    m, lo, hi = paired_bootstrap(st, y, pt, pb)
    recomputed.append(abs(m - rr["diff"]) < 1e-9
                      and abs(lo - rr.ci_lo) < 1e-9
                      and abs(hi - rr.ci_hi) < 1e-9)
print(abl[["comparison", "protocol", "horizon", "pr_auc_transformer",
           "pr_auc_base", "diff", "ci_lo", "ci_hi"]].round(5).to_string(index=False))
sig = abl[(abl.ci_lo > 0) | (abl.ci_hi < 0)]
check("N25-08", f"paired bootstrap reproduces the stored CI exactly "
                f"({all(recomputed)}); {len(sig)}/{len(abl)} cells separable "
                f"from the baseline (transformer win: "
                f"{int(((abl.ci_lo > 0)).sum())}, loss: {int(((abl.ci_hi < 0)).sum())})",
      all(recomputed))"""))

cells.append(code(r"""# N25-09  determinism: two identical fixed-epoch Transformer fits bit-identical
import train_transformer as T
from train_sequence import prepare, fold_stats

X, M = prepare(win_L)
y = ix["label_24h"].to_numpy()
tr_flat = np.flatnonzero((split == "train") & ~hold & ~np.isnan(y))
ev_flat = np.flatnonzero((split == "val") & ~hold & ~np.isnan(y))
tr = tr_flat[:2000]
ev = ev_flat[:1000]
mu, sd = fold_stats(X, M, tr)
cfg = json.loads(win["params"])
cfg["_weight"] = "on" if str(win["weighting"]).startswith("pos") else "off"
r1 = T.fit_transformer(cfg, X, M, mu, sd, tr, y[tr], ev, y[ev], fixed_epochs=2)
r2 = T.fit_transformer(cfg, X, M, mu, sd, tr, y[tr], ev, y[ev], fixed_epochs=2)
same = all(torch.equal(r1["net"].state_dict()[k], r2["net"].state_dict()[k])
           for k in r1["net"].state_dict())
check("N25-09", f"two seeded 2-epoch Transformer fits (L={win_L}) on 2,000 rows: "
                f"bit-identical parameters = {same}", bool(same))"""))

cells.append(code(r"""# N25-10  SHA-256 of the immutable experimental table
check("N25-10", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)"""))

cells.append(code(r"""# N25-11  report + figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES.mkdir(exist_ok=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
po = [FAM, "lstm_L3", "tcn_bestL", "fusion_L3", "xgboost", "persistence"]
plot = pd.concat([allv, bar]).reindex(po)
x = np.arange(len(po)); w = 0.26
for i, h in enumerate(HORIZONS):
    ax.bar(x + (i - 1) * w, plot.loc[po, h].to_numpy(), w, label=h)
ax.set_xticks(x); ax.set_xticklabels(po, rotation=20, ha="right")
ax.set_ylabel("PR-AUC (val_2022)"); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3, axis="y")
ax.legend(); ax.set_title("Phase 25 - Transformer vs temporal baselines vs D7/xgb")

hzs = ["24h", "48h", "72h"]
xb = np.arange(len(hzs)); ww = 0.2
for j, base_name in enumerate(["lstm_L3", "tcn_bestL"]):
    sub = abl[(abl.comparison == f"transformer_vs_{base_name}") & (abl.protocol == "P3")]
    sub = sub.set_index("horizon").reindex(hzs)
    ax2.bar(xb + (j - 1.5) * ww + ww / 2, sub.pr_auc_transformer.to_numpy(), ww,
            label=f"transformer vs {base_name}")
    ax2.bar(xb + (j - 1.5) * ww + 1.5 * ww, sub.pr_auc_base.to_numpy(), ww,
            label=f"{base_name}", alpha=0.6)
ax2.set_xticks(xb); ax2.set_xticklabels(hzs)
ax2.set_ylabel("P3 PR-AUC (unseen stations)"); ax2.set_ylim(0.5, 1.0)
ax2.grid(alpha=0.3, axis="y"); ax2.legend(fontsize=8)
ax2.set_title("P3 spatial CV: Transformer vs temporal baselines")
fig.tight_layout()
fig.savefig(FIGURES / "phase25_val2022.png", dpi=130)
plt.close(fig)
check("N25-11", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure figures/phase25_val2022.png written",
      REPORT.exists() and (FIGURES / "phase25_val2022.png").exists())"""))

cells.append(code(r"""# N25-12  summary
print("\n" + "=" * 78)
print("PHASE 25 / STAGE 40  Temporal Transformer (D13 gate PASS, narrow) verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
check("N25-12", f"{sum(rr['pass'] for rr in RES) + 1}/{len(RES) + 1} checks passed, "
                f"{time.time() - t0:.1f}s", True)
print("Verdict: see reports/phase25_transformer.md; test untouched (frozen to Phase 33); "
      "next: Phase 26 multi-horizon matrix")
assert all(rr["pass"] for rr in RES)
print("ALL PHASE-25 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
