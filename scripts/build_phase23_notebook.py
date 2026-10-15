"""Build notebooks/23_fusion.ipynb (workflow Phase 23 / stage 38).

The fusion variants are fit by scripts/train_fusion.py (static_only fusion_L3
hybrid_L3) and the matched-capacity ablation by scripts/phase23_ablation.py. This
notebook re-derives every claim from the written artefacts; the only training it
does is the small fixed-epoch determinism probe in N23-09.

Phase 23 asks whether the temporal model, allowed to see WHERE it is (the static
geographic vector), gains skill over the sequence alone - and by how much, with a
station-bootstrap CI on the gain (DoD), not just point estimates.

Checks N23-01..N23-14:
    N23-01  artefacts exist with the expected shapes
    N23-02  registry contract: L3-07/08/09 (frozen test, budget <= 16, selection)
    N23-03  test policy: zero test cells; the 7 dev cells; all three seasons
    N23-04  grid identity: rebuilt masks 1:1; no holdout/test rows in training
    N23-05  tuning: budget honoured per family, 3 horizons, mean == mean, winner
    N23-06  grid: 216 rows, n>0, CIs on big cells, n_params == winner's params
    N23-07  headline: fused vs temporal-only (L3-01) vs static-only per horizon,
            against D7 persistence and Phase-20 xgboost
    N23-08  matched-capacity ablation + paired station-bootstrap gain CI
    N23-09  determinism: two identical fixed-epoch fusion fits bit-identical
    N23-10  static vector: 133 x 15, per-station constant, covers the stations
    N23-11  report exists; figures rendered
    N23-12  run log records a clean exit for the driver calls
    N23-13  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N23-14  summary

Run:
    python scripts/build_phase23_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\23_fusion.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "23_fusion.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 23 — Multi-input / feature-fusion deep learning (stage 38)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 23 — Multi-input / feature-fusion deep learning** → checklist **stage 38** |
| **Fit by** | `scripts/train_fusion.py static_only fusion_L3 hybrid_L3` → `scripts/phase23_run.log`; `scripts/phase23_ablation.py` → `scripts/phase23_ablation.log` |
| **Reads** | `reports/phase23_metrics.csv` (216) · `reports/phase23_search.csv` (44) · `models/phase23_L3-0{7,8,9}.pt` · `reports/phase23_ablation*.csv` · `reports/phase21_metrics.csv` (lstm_L3 temporal-only) · `data/processed/static_vectors.csv` + `static_manifest.json` · `reports/phase17_experiment_registry.csv` · `reports/phase16_baseline_metrics.csv` · `reports/phase20_metrics.csv` |
| **Contract** | Dataset B sequences (P19-M1) + static group · dev cells only (test frozen to Phase 33) · budgets 16/16/12 (≤ 16) · seed 20261008 · L\\* = 3 |

The phase answers the project's point — **does knowing *where* a station is add
skill over the sequence alone?** — with the fused / temporal-only / static-only
three-way, plus a matched-capacity ablation and a station-bootstrap CI on the
gain."""))

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
MET23 = REPORTS / "phase23_metrics.csv"
SEARCH23 = REPORTS / "phase23_search.csv"
MET21 = REPORTS / "phase21_metrics.csv"
ABL = REPORTS / "phase23_ablation.csv"
ABLBOOT = REPORTS / "phase23_ablation_boot.csv"
ABLPRED = REPORTS / "phase23_ablation_preds.npz"
P16 = REPORTS / "phase16_baseline_metrics.csv"
P20 = REPORTS / "phase20_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase23_fusion.md"
STATIC = PROC / "static_vectors.csv"
MANIFEST = PROC / "static_manifest.json"
LOG = ROOT / "scripts" / "phase23_run.log"
ABLLOG = ROOT / "scripts" / "phase23_ablation.log"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAMS = ["fusion_L3", "static_only", "hybrid_L3"]
REG_ID = {"fusion_L3": "L3-07", "static_only": "L3-08", "hybrid_L3": "L3-09"}
BUDGET = {"fusion_L3": 16, "static_only": 16, "hybrid_L3": 12}
HORIZONS = ["label_24h", "label_48h", "label_72h"]
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
PT = {f: MODELS / f"phase23_{REG_ID[f]}.pt" for f in FAMS}


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


met23 = pd.read_csv(MET23)
search23 = pd.read_csv(SEARCH23)
met21 = pd.read_csv(MET21)
abl = pd.read_csv(ABL)
ablboot = pd.read_csv(ABLBOOT)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
p20 = pd.read_csv(P20)
t0 = time.time()
print("Phase 23 notebook - feature-fusion verification")"""))

cells.append(code(r"""# N23-01  artefacts present with the expected shapes
fig = FIGURES / "phase23_val2022.png"
for p in (MET23, SEARCH23, ABL, ABLBOOT, ABLPRED, REPORT, LOG, ABLLOG, STATIC,
          MANIFEST, *PT.values()):
    assert p.exists(), p
check("N23-01", f"phase23_metrics {len(met23)} rows, phase23_search {len(search23)} "
                f"rows, ablation {len(abl)} rows, {len(PT)} model files, static "
                f"vector + manifest + report + logs present",
      len(met23) == 216 and len(search23) == 16 + 16 + 12 and len(abl) == 144)"""))

cells.append(code(r"""# N23-02  registry contract: L3-07/08/09 fused / static-only / hybrid
for f in FAMS:
    row = reg[reg.registry_id == REG_ID[f]]
    assert len(row) == 1, (f, row)
    r = row.iloc[0]
    ok = (r.model == f and r.test_policy == "frozen_read_at_phase33"
          and int(r.tuning_budget_configs) >= BUDGET[f]
          and "phase23_metrics.csv" in r.metrics_out
          and r.selection_cell == "val_2022" and r.selection_metric == "pr_auc"
          and str(r.gate) in ("nan", ""))
    check(f"N23-02.{f}", f"{REG_ID[f]} {r.model}: budget_max="
                        f"{int(r.tuning_budget_configs)} >= used {BUDGET[f]}, "
                        f"test_policy={r.test_policy}, selection="
                        f"{r.selection_metric}/{r.selection_cell}", bool(ok))"""))

cells.append(code(r"""# N23-03  test policy: no test rows anywhere in the phase-23 metrics
cells = set(met23.cell.unique())
test_rows = met23[met23.cell.astype(str).str.startswith("test")]
seasons = set(met23.season.unique())
check("N23-03", f"cells = {sorted(cells)} == the 7 dev cells (no test); test rows "
                f"= {len(test_rows)} (must be 0); seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N23-04  grid identity vs split_assignments + holdout/test exclusion
import baseline_model  # noqa: F401
from train_sequence import build_tasks_index

ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
assert len(ix) == 200665
hold = ix.station_holdout.astype(bool).to_numpy()
split = ix.split.to_numpy()
year = ix.date.dt.year.to_numpy()
fold = ix.spatial_fold.to_numpy()

rebuilt = {"train_<2022": (split == "train") & ~hold}
dev = {"val_2022": (split == "val") & ~hold}
for Y in (2020, 2021, 2022):
    rebuilt[f"train_<={Y-1}"] = (year <= Y - 1) & ~hold
    dev[f"val_{Y}"] = (year == Y) & ~hold
for k in (1, 2, 3, 4):
    rebuilt[f"excl_fold_{k}"] = (split == "train") & ~hold & (fold != k)
    dev[f"spatial_fold_{k}"] = (split == "train") & (fold == k)

tasks = build_tasks_index(ix)
mism = 0
for t in tasks:
    mism += int(not np.array_equal(t["train"], rebuilt[t["name"]]))
    for _, cell, m in t["evals"]:
        mism += int(not np.array_equal(m, dev[cell]))
leak = sum(int((t["train"] & hold).sum() + (t["train"] & (split == "test")).sum())
           for t in tasks)
check("N23-04", f"8 training masks == masks rebuilt from split_assignments "
                f"(mismatches={mism}); dev eval masks 1:1; training masks contain "
                f"{leak} holdout/test rows (must be 0)",
      mism == 0 and leak == 0)"""))

cells.append(code(r"""# N23-05  tuning: budget honoured per family, 3 horizons, mean == mean, winner
for f in FAMS:
    g = search23[search23.family == f]
    three = bool(((g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
    mean_ok = bool(np.allclose(g.mean_pr_auc,
                               g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
    w = g.loc[g.mean_pr_auc.idxmax()]
    check(f"N23-05.{f}", f"{f}: {len(g)} configs == budget {BUDGET[f]}, 3 horizons, "
                        f"mean == mean(3), winner {w.config_id} "
                        f"mean_pr_auc={w.mean_pr_auc:.5f} ({w.weighting}), "
                        f"n_params={int(w.n_params):,}",
          len(g) == BUDGET[f] and three and mean_ok)"""))

cells.append(code(r"""# N23-06  grid: 216 rows, n>0, CIs on big cells, n_params == winner's params
size = met23.groupby("model").size()
ci_missing = met23[(met23.season == "all") & (met23.n >= 1000) & met23.pr_auc_ci_lo.isna()]
np_ok = True
for f in FAMS:
    w = search23[search23.family == f].sort_values("mean_pr_auc", ascending=False).iloc[0]
    got = set(met23[met23.model == f].n_params.unique())
    np_ok &= (got == {int(w.n_params)})
check("N23-06", f"rows = {dict(size)} (expect 216); n>0 = {bool((met23.n > 0).all())}; "
                f"bootstrap CI missing on big cells = {len(ci_missing)}; every "
                f"model's n_params == its winner's params = {np_ok}",
      (size == 72).all() and (met23.n > 0).all() and len(ci_missing) == 0 and np_ok)"""))

cells.append(code(r"""# N23-07  headline: fused vs temporal-only(L3-01) vs static-only, vs D7 + xgboost
val23 = met23[(met23.protocol == "P1") & (met23.cell == "val_2022") & (met23.season == "all")]
val21 = met21[(met21.protocol == "P1") & (met21.cell == "val_2022")
              & (met21.season == "all") & (met21.model == "lstm_L3")]
dl = pd.concat([val23, val21]).pivot_table(index="model", columns="horizon",
                                           values="pr_auc")
order = ["static_only", "lstm_L3", "hybrid_L3", "fusion_L3"]
bar = pd.concat([
    p16[(p16.model == "persistence") & (p16.protocol == "P1")
        & (p16.cell == "val_2022") & (p16.season == "all")],
    p20[(p20.model == "xgboost") & (p20.protocol == "P1")
        & (p20.cell == "val_2022") & (p20.season == "all")],
]).pivot_table(index="model", columns="horizon", values="pr_auc")
allv = pd.concat([dl.reindex(order), bar])
print("val_2022 PR-AUC (season=all):")
print(allv.round(4).to_string())
check("N23-07", f"three-way present: fused {dl.loc['fusion_L3'].round(4).to_dict()}, "
                f"temporal-only {dl.loc['lstm_L3'].round(4).to_dict()}, "
                f"static-only {dl.loc['static_only'].round(4).to_dict()}; bars "
                f"D7 {bar.loc['persistence'].round(4).to_dict()} / "
                f"xgb {bar.loc['xgboost'].round(4).to_dict()}",
      set(order) <= set(dl.index) and dl.notna().all().all())"""))

cells.append(code(r"""# N23-08  matched-capacity ablation + paired station-bootstrap gain CI
from sklearn.metrics import average_precision_score
from phase23_ablation import paired_boot

P = np.load(ABLPRED)
recomputed = []
for hor in HORIZONS:
    y = P[f"fusion_matched|{hor}|y"]
    st = P[f"fusion_matched|{hor}|st"]
    pf = P[f"fusion_matched|{hor}|p"]
    pt = P[f"temporal_only_matched|{hor}|p"]
    m, lo, hi = paired_boot(st, y, pf, pt)
    rec = ablboot[ablboot.horizon == hor].iloc[0]
    recomputed.append(abs(m - rec.gain_mean_boot) < 1e-9
                      and abs(lo - rec.gain_ci_lo) < 1e-9
                      and abs(hi - rec.gain_ci_hi) < 1e-9)
print("matched-capacity fused - temporal-only gain (val_2022, season=all):")
print(ablboot.round(4).to_string(index=False))
sig = ablboot[(ablboot.gain_ci_lo > 0) | (ablboot.gain_ci_hi < 0)]
check("N23-08", f"paired bootstrap reproduces the stored CI exactly "
                f"({recomputed}); horizons with a CI excluding 0: "
                f"{sig.horizon.tolist()}",
      all(recomputed) and len(ablboot) == 3)"""))

cells.append(code(r"""# N23-09  determinism: two identical fixed-epoch fusion fits bit-identical
import train_fusion as tf
from train_sequence import prepare

tf.STATION_OF = ix.station_id.to_numpy()
S = pd.read_csv(STATIC).set_index("station_id")
X, M = prepare(3)
y = ix["label_24h"].to_numpy()
tri = (split == "train") & ~hold
ev = (split == "val") & ~hold
tr_idx = np.flatnonzero(tri & ~np.isnan(y))[:3000]
ev_idx = np.flatnonzero(ev & ~np.isnan(y))[:1500]
spec = {"kind": "lstm", "use_seq": True, "use_static": True, "lin": False}
enc = tf.fit_encoders(spec, X, M, S, None, tr_idx)
tr_t = tf.encode(enc, spec, X, M, S, None, tr_idx)
ev_t = tf.encode(enc, spec, X, M, S, None, ev_idx)
cfg = {"_cell": 0, "lr": 1e-3, "hidden": 32, "layers": 2, "dropout": 0.0,
       "batch_size": 256, "_weight": "off"}
r1 = tf.fit_net(spec, cfg, tr_t, y[tr_idx], ev_t, y[ev_idx], fixed_epochs=2)
r2 = tf.fit_net(spec, cfg, tr_t, y[tr_idx], ev_t, y[ev_idx], fixed_epochs=2)
same = all(torch.equal(r1["net"].state_dict()[k], r2["net"].state_dict()[k])
           for k in r1["net"].state_dict())
check("N23-09", f"two seeded 2-epoch fusion fits on 3,000 rows: bit-identical "
                f"parameters = {same}", bool(same))"""))

cells.append(code(r"""# N23-10  static vector: 133 x 15, per-station constant, covers the stations
Sd = pd.read_csv(STATIC)
man = json.loads(MANIFEST.read_text(encoding="utf-8"))
d = pd.read_csv(TBL, usecols=["station_id"] + man["numeric"] + man["categorical"])
const = all(int(d.groupby("station_id")[c].nunique(dropna=False).max()) == 1
            for c in man["numeric"] + man["categorical"])
cover = set(ix.station_id.unique()) <= set(Sd.station_id)
check("N23-10", f"static {Sd.shape[0]} x {man['static_dim']} "
                f"({len(man['numeric'])} numeric + {len(man['categorical'])} "
                f"categorical); per-station constant = {const}; covers stations = "
                f"{cover}; manifest hash matches = "
                f"{man['output_sha256'] == sha256(STATIC)}",
      Sd.shape[0] == 133 and man["static_dim"] == 15 and const and cover
      and man["output_sha256"] == sha256(STATIC))"""))

cells.append(code(r"""# N23-11  figures: val_2022 bars + matched gain-with-CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES.mkdir(exist_ok=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
po = ["static_only", "lstm_L3", "hybrid_L3", "fusion_L3", "xgboost", "persistence"]
plot = allv.reindex(po)
x = np.arange(len(po)); w = 0.26
for i, h in enumerate(HORIZONS):
    ax.bar(x + (i - 1) * w, plot.loc[po, h].to_numpy(), w, label=h)
ax.set_xticks(x); ax.set_xticklabels(po, rotation=20, ha="right")
ax.set_ylabel("PR-AUC (val_2022)"); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3, axis="y")
ax.legend(); ax.set_title("Phase 23 - fused vs temporal-only vs static-only")

xb = np.arange(len(HORIZONS))
g = ablboot.set_index("horizon").reindex(HORIZONS)
ax2.axhline(0, color="k", lw=1)
ax2.bar(xb, g.gain, 0.5, color="#4477aa")
ax2.errorbar(xb, g.gain, yerr=[g.gain - g.gain_ci_lo, g.gain_ci_hi - g.gain],
             fmt="none", ecolor="k", capsize=4)
ax2.set_xticks(xb); ax2.set_xticklabels([h.replace("label_", "") for h in HORIZONS])
ax2.set_ylabel("PR-AUC gain (fused - temporal-only)")
ax2.set_title("Matched capacity: static-branch gain, 95% station-bootstrap CI")
ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIGURES / "phase23_val2022.png", dpi=130)
plt.close(fig)
check("N23-11", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure figures/phase23_val2022.png written",
      REPORT.exists() and (FIGURES / "phase23_val2022.png").exists())"""))

cells.append(code(r"""# N23-12/N23-13/N23-14  logs clean, byte-stability, summary
txt = LOG.read_text(encoding="utf-8", errors="replace")
clean = [ln for ln in txt.splitlines() if "exit=0" in ln]
abltxt = ABLLOG.read_text(encoding="utf-8", errors="replace")
check("N23-12", f"{LOG.name}: {len(clean)} clean-finish markers; "
                f"{ABLLOG.name} has 'fused - temporal-only gain' = "
                f"{'fused - temporal-only gain' in abltxt}",
      len(clean) >= 1 and "fused - temporal-only gain" in abltxt)
check("N23-13", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)
print("\n" + "=" * 78)
print("PHASE 23 / STAGE 38  Multi-input feature fusion verified")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
check("N23-14", f"{sum(r['pass'] for r in RES) + 1}/{len(RES) + 1} checks passed, "
                f"{time.time() - t0:.1f}s", True)
print("Answer: does WHERE add skill over the sequence alone? see "
      "reports/phase23_fusion.md; test untouched (frozen to Phase 33); "
      "next: Phase 24 GNN gate (stage 39)")
assert all(r["pass"] for r in RES)
print("ALL PHASE-23 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
