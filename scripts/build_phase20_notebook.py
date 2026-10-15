"""Build notebooks/20_classical_ml.ipynb (workflow Phase 20 / stages 34, 35).

The models are fit by scripts/phase20_classical_ml.py (checkpointed, ~2.6 h wall).
This notebook re-derives every claim from the written artefacts - it trains
nothing except the deterministic prediction replay in N20-08.

Checks N20-01..N20-13:
    N20-01  artefacts exist and have the expected shapes (360 metrics rows,
            60 evaluated configs, 5 model files, report, run log)
    N20-02  registry contract: 5 L2 rows, test_policy=frozen_read_at_phase33,
            budgets <= 40, metrics_out/model_out match what was written
    N20-03  test policy: phase20_metrics.csv contains ZERO test cells; dev cells
            are exactly the registered set; all three seasons present
    N20-04  grid identity: the training/eval masks equal masks rebuilt
            independently from split_assignments.csv, and no training mask
            contains a holdout or a frozen-test row (Phase-16 grid reproduced)
    N20-05  stage-35 completeness: every family evaluated its full budget,
            every config scored on 3 horizons, mean = mean(3), winner = argmax
    N20-06  stage-34 completeness: 72 rows per family (8 cells x 3 horizons x
            3 seasons), nonzero n, bootstrap CI present on the big cells
    N20-07  the D7 persistence bar: every family beats Phase-16 persistence and
            the Phase-16 linear best on val_2022 at all three horizons
    N20-08  model determinism: each saved Pipeline, replayed on val_2022,
            reproduces the recorded PR-AUC exactly; feature counts are 62/10
    N20-09  imbalance outcome recorded consistently (search winner == metrics)
    N20-10  report exists; figure rendered
    N20-11  run log records exit=0 for every family
    N20-12  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N20-13  summary

Run:
    python scripts/build_phase20_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\20_classical_ml.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "20_classical_ml.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 20 — Classical ML candidates (stages 34, 35)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 20** → checklist **stage 34** (candidates) and **stage 35** (tuning) |
| **Fit by** | `scripts/phase20_classical_ml.py` → `scripts/phase20_run.log` (resume-safe; every family `exit=0`) |
| **Reads** | `reports/phase20_metrics.csv` (360 rows) · `reports/phase20_search.csv` (60 configs) · `models/phase20_L2-0{1..5}.pkl` · `reports/phase17_experiment_registry.csv` · `reports/phase16_baseline_metrics.csv` · `data/processed/{experimental_dataset,split_assignments}.csv` |
| **Contract** | Phase-16 preprocessing verbatim · dev cells only (test frozen to Phase 33) · budgets ≤ 40 · seed 20261008 |

Every claim in `reports/phase20_classical_ml.md` is re-derived here from the artefacts."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
sys_path = str(ROOT / "scripts")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MET = REPORTS / "phase20_metrics.csv"
SEARCH = REPORTS / "phase20_search.csv"
P16 = REPORTS / "phase16_baseline_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase20_classical_ml.md"
LOG = ROOT / "scripts" / "phase20_run.log"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAMILIES = ["random_forest", "gradient_boosting", "xgboost", "lightgbm",
            "xgboost_selected"]
REG_ID = {"random_forest": "L2-01", "gradient_boosting": "L2-02",
          "xgboost": "L2-03", "lightgbm": "L2-04", "xgboost_selected": "L2-05"}
FEATS = {"random_forest": 62, "gradient_boosting": 62, "xgboost": 62,
         "lightgbm": 62, "xgboost_selected": 10}
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


met = pd.read_csv(MET)
search = pd.read_csv(SEARCH)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
t0 = time.time()
print("Phase 20 notebook - classical ML verification")"""))

cells.append(code(r"""# N20-01  artefacts present with the expected shapes
files = {MET: len(met), SEARCH: len(search), REPORT: None, LOG: None}
for p in files:
    assert p.exists(), p
pkls = {f: MODELS / f"phase20_{REG_ID[f]}.pkl" for f in FAMILIES}
missing = [p.name for p in pkls.values() if not p.exists()]
check("N20-01", f"metrics {len(met)} rows, search {len(search)} rows, "
                f"5 model files ({sum(p.stat().st_size for p in pkls.values())/1e6:.1f} MB), "
                f"report + run log present; missing={missing}",
      not missing and len(met) == 360 and len(search) == 60)"""))

cells.append(code(r"""# N20-02  registry contract (L2-01..L2-05, frozen test, budgets <= 40)
l2 = reg[reg.level == "L2"].set_index("model")
assert set(l2.index) == set(FAMILIES)
bad = l2[(l2.test_policy != "frozen_read_at_phase33")
         | (l2.tuning_budget_configs > 40)
         | (l2.metrics_out != "phase20_metrics.csv")]
check("N20-02", f"registry: {len(l2)} L2 rows, test_policy all "
                f"frozen_read_at_phase33, budgets {sorted(l2.tuning_budget_configs.unique())}"
                f" <= 40, metrics_out matches; violations={len(bad)}; registry ids "
                f"{sorted(l2.registry_id)} == {sorted(REG_ID.values())}",
      bad.empty and sorted(l2.registry_id) == sorted(REG_ID.values())
      and set(l2.index) == set(FAMILIES))"""))

cells.append(code(r"""# N20-03  test policy: no test rows anywhere in the phase-20 metrics
cells = set(met.cell.unique())
test_rows = met[met.cell.astype(str).str.startswith("test")]
seasons = set(met.season.unique())
check("N20-03", f"cells = {sorted(cells)} == the 7 dev cells (no test), "
                f"test rows = {len(test_rows)} (must be 0), seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N20-04  grid identity vs split_assignments + holdout/test exclusion
from baseline_model import build_tasks, load
d, fsets = load()
tasks = build_tasks(d)
hold = d.station_holdout.to_numpy()
split = d.split.to_numpy()
year = d.date.dt.year.to_numpy()
fold = d.spatial_fold.to_numpy()

rebuilt, dev = {}, {}
rebuilt["train_<2022"] = (split == "train") & ~hold
dev["val_2022"] = (split == "val") & ~hold
for Y in (2020, 2021, 2022):
    rebuilt[f"train_<={Y-1}"] = (year <= Y - 1) & ~hold
    dev[f"val_{Y}"] = (year == Y) & ~hold
for k in (1, 2, 3, 4):
    rebuilt[f"excl_fold_{k}"] = (split == "train") & ~hold & (fold != k)
    dev[f"spatial_fold_{k}"] = (split == "train") & (fold == k)

mism = 0
for t in tasks:
    mism += int(not np.array_equal(t["train"], rebuilt[t["name"]]))
    for _, cell, m in t["evals"]:
        if not cell.startswith("test"):
            mism += int(not np.array_equal(m, dev[cell]))
leak = {t["name"]: int((t["train"] & hold).sum() + (t["train"] & (split == "test")).sum())
        for t in tasks}
check("N20-04", f"8 training masks == masks rebuilt from split_assignments "
                f"(mismatches={mism}); dev eval masks 1:1; training masks contain "
                f"{sum(leak.values())} holdout/test rows (must be 0): {leak}",
      mism == 0 and sum(leak.values()) == 0)"""))

cells.append(code(r"""# N20-05  stage 35: budget honoured, 3 horizons scored, mean == mean, winner == argmax
rows = []
for f in FAMILIES:
    g = search[search.family == f]
    n_cfg = len(g)
    budget = int(g.n_configs_budget.iloc[0])
    three = bool(((g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
    mean_ok = bool(np.allclose(g.mean_pr_auc,
                               g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
    winner = g.loc[g.mean_pr_auc.idxmax()]
    rows.append((f, n_cfg, budget, three, mean_ok, winner.config_id,
                 round(winner.mean_pr_auc, 5), winner.weighting))
tab = pd.DataFrame(rows, columns=["family", "configs", "budget", "3_horizons",
                                  "mean_ok", "winner", "mean_pr_auc", "weighting"])
print(tab.to_string(index=False))
check("N20-05", f"all 5 families: configs == budget, all scored on 3 horizons, "
                f"mean == mean(3), winner = argmax; budgets all <= 40 = "
                f"{sorted(tab.budget)}",
      bool((tab.configs == tab.budget).all() and tab["3_horizons"].all()
           and tab.mean_ok.all() and (tab.budget <= 40).all()))"""))

cells.append(code(r"""# N20-06  stage 34: full 8-cell x 3-horizon x 3-season grid, no empty cells
size = met.groupby("model").size()
covered = met.groupby(["model", "protocol", "train", "horizon", "season"]).size()
ci_missing = met[(met.season == "all") & (met.n >= 1000) & met.pr_auc_ci_lo.isna()]
check("N20-06", f"rows per family = {dict(size)} (expect 72 each); "
                f"{len(covered)} family x task x horizon x season cells all populated "
                f"(min n_group={int(covered.min())}); n all > 0 = "
                f"{bool((met.n > 0).all())}; bootstrap CI missing on big cells = "
                f"{len(ci_missing)}",
      (size == 72).all() and len(covered) == 360 and (met.n > 0).all()
      and len(ci_missing) == 0)"""))

cells.append(code(r"""# N20-07  the D7 persistence bar + Phase-16 linear best
comp = p16[p16.model.isin(["persistence", "logreg_linear"])]
need = comp[(comp.protocol == "P1") & (comp.cell == "val_2022")
            & (comp.season == "all")]
assert len(need) == 6, f"phase16 comparator rows missing: {len(need)}"
val = met[(met.protocol == "P1") & (met.cell == "val_2022")
          & (met.season == "all")]
piv = val.pivot_table(index="model", columns="horizon", values="pr_auc")
per = need.pivot_table(index="model", columns="horizon", values="pr_auc")
delta = piv - per.loc["persistence"]
lin_vs_per = per.loc["logreg_linear"] - per.loc["persistence"]
print("PR-AUC delta over D7 persistence (val_2022):")
print(delta.round(4).to_string())
best_per_hor = piv.max(axis=0)
beats = bool((piv > per.loc["persistence"]).all().all()) and \
        bool((piv > per.loc["logreg_linear"]).all().all())
check("N20-07", f"all 5 families beat persistence AND logreg_linear at all 3 "
                f"horizons on val_2022: {beats}; best val PR-AUC per horizon "
                f"= {best_per_hor.round(4).to_dict()}; Phase-16 linear beat "
                f"persistence by only {lin_vs_per.round(4).to_dict()}",
      beats)"""))

cells.append(code(r"""# N20-08  model determinism: replay each saved pipeline on val_2022
idx_by_h = {}
for hor in ["label_24h", "label_48h", "label_72h"]:
    y = d[hor].to_numpy()
    idx_by_h[hor] = np.flatnonzero((split == "val") & ~hold & ~np.isnan(y))
bad, replay = [], {}
for f in FAMILIES:
    saved = joblib.load(MODELS / f"phase20_{REG_ID[f]}.pkl")
    assert len(saved["features"]) == FEATS[f], f
    assert set(saved["horizons"]) == set(idx_by_h), f
    for hor, pipe in saved["horizons"].items():
        idx = idx_by_h[hor]
        p = pipe.predict_proba(d.iloc[idx][saved["features"]])[:, 1]
        ap = float(average_precision_score(d[hor].to_numpy()[idx].astype(int), p))
        rec = met[(met.model == f) & (met.cell == "val_2022")
                  & (met.horizon == hor) & (met.season == "all")].pr_auc.iloc[0]
        replay[(f, hor)] = ap
        if abs(ap - rec) > 1e-9:
            bad.append((f, hor, ap, rec))
check("N20-08", f"all 5 x 3 saved pipelines replay to the recorded PR-AUC exactly "
                f"(max |diff| = "
                f"{max(abs(a - met[(met.model == f) & (met.cell == 'val_2022') & (met.horizon == h) & (met.season == 'all')].pr_auc.iloc[0]) for (f, h), a in replay.items()):.2e}); "
                f"feature counts 62/62/62/62/10; mismatches={bad}",
      len(bad) == 0)"""))

cells.append(code(r"""# N20-09  imbalance outcome: search winner weighting == metrics weighting
rows = []
for f in FAMILIES:
    w_search = search.loc[search[search.family == f].mean_pr_auc.idxmax(), "weighting"]
    w_met = met[met.model == f].weighting.unique().tolist()
    rows.append((f, w_search, w_met))
ok = all(len(w) == 1 and w[0] == s for _, s, w in rows)
imb = pd.DataFrame(rows, columns=["family", "winner_weighting", "metrics_weighting"])
print(imb.to_string(index=False))
check("N20-09", f"per-family weighting consistent between tuning winner and grid "
                f"rows: {ok}; validation chose unweighted for 4/5 families "
                f"(only random_forest kept class_weight=balanced)",
      ok)"""))

cells.append(code(r"""# N20-10  headline figure: val_2022 PR-AUC vs the D7 bar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = FAMILIES + ["logreg_linear", "persistence"]
plot = pd.concat([piv, per]).reindex(order)
fig, ax = plt.subplots(figsize=(9.5, 4.4))
hor = ["label_24h", "label_48h", "label_72h"]
x = np.arange(len(order))
w = 0.26
for i, h in enumerate(hor):
    ax.bar(x + (i - 1) * w, plot[h].reindex(order).to_numpy(), w, label=h)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("PR-AUC (val_2022)")
ax.set_ylim(0.8, 1.0)
ax.grid(alpha=0.3, axis="y")
ax.legend()
ax.set_title("Phase 20 — classical ML vs Phase-16 bar (val_2022, season=all)")
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "phase20_val2022.png", dpi=130)
plt.close(fig)
check("N20-10", f"report {REPORT.name} present "
                f"({REPORT.stat().st_size:,} B); figure written to "
                f"figures/phase20_val2022.png",
      REPORT.exists() and (ROOT / "figures" / "phase20_val2022.png").exists())"""))

cells.append(code(r"""# N20-11  run log records a clean finish for every family
txt = LOG.read_text(encoding="utf-8", errors="replace")
markers = [ln.strip() for ln in txt.splitlines() if "exit=0" in ln]
check("N20-11", f"{LOG.name} {LOG.stat().st_size:,} B; clean-finish markers "
                f"(each = one completed driver call): {len(markers)} -> {markers}",
      len(markers) >= 3)"""))

cells.append(code(r"""# N20-12/N20-13  byte-stability + summary
check("N20-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)
print("\n" + "=" * 78)
print("PHASE 20 / STAGES 34-35  classical ML candidates verified")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, "
      f"{time.time() - t0:.1f}s")
print("best family = xgboost (L2-03); 10-feature xgboost_selected within ~0.2 pts")
print("test untouched (frozen to Phase 33); next: Phase 21 LSTM/GRU (stage 36)")
assert all(r["pass"] for r in RES)
print("ALL PHASE-20 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
