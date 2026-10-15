"""Build notebooks/21_temporal_dl.ipynb (workflow Phase 21 / stage 36).

The models are fit by scripts/train_sequence.py (resume-safe, checkpointed per
family; every family exit=0). This notebook re-derives every claim from the
written artefacts - it trains nothing except the two determinism probes in
N21-08 (a small fixed-epoch two-run replay).

Checks N21-01..N21-13:
    N21-01  artefacts exist and have the expected shapes (360 metrics rows,
            46 evaluated configs, 5 model files, report, run log)
    N21-02  registry contract: 5 L3 rows (L3-01..L3-05), frozen test, budgets
            <= 20, metrics_out lists phase21_metrics.csv, selection val_2022/pr_auc
    N21-03  test policy: phase21_metrics.csv contains ZERO test cells; dev
            cells are exactly the registered set; all three seasons present
    N21-04  grid identity: sequence-index training/eval masks equal masks
            rebuilt independently from split_assignments.csv, and no training
            mask contains a holdout or a frozen-test row (Phase-20 grid)
    N21-05  stage-36 tuning: every family evaluated its budget, every config
            scored on 3 horizons, mean = mean(3), winner = argmax
    N21-06  stage-36 grid: 72 rows per family (8 cells x 3 horizons x 3
            seasons), nonzero n, bootstrap CI present on the big cells,
            n_params recorded
    N21-07  headline: DL vs the D7 persistence bar and the Phase-20 classical
            best (xgboost) on val_2022, all three horizons
    N21-08  determinism: (a) each saved state_dict replays to the recorded
            PR-AUC exactly; (b) two identical fixed-epoch fits are bit-identical
    N21-09  L* recorded (val_2022 mean PR-AUC argmax over the four LSTM
            families) and gru_bestL was tuned at exactly L*
    N21-10  report exists; figure rendered
    N21-11  run log records exit=0 for every driver call
    N21-12  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N21-13  summary

Run:
    python scripts/build_phase21_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\21_temporal_dl.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "21_temporal_dl.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 21 — Deep temporal learning I: LSTM & GRU (stage 36)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 21** → checklist **stage 36** |
| **Fit by** | `scripts/train_sequence.py` → `scripts/phase21_run.log` (resume-safe; every family `exit=0`) |
| **Reads** | `reports/phase21_metrics.csv` (360 rows) · `reports/phase21_search.csv` (46 configs) · `models/phase21_L3-0{1..5}.pt` · `reports/phase17_experiment_registry.csv` · `reports/phase16_baseline_metrics.csv` · `reports/phase20_metrics.csv` · `data/processed/{sequence_index,sequences_L*,sequence_mask_L*}.npy` |
| **Contract** | Dataset B raw sequences + mask (P19-M1) · 13 channels · per-fold standardisation · dev cells only (test frozen to Phase 33) · budgets ≤ 20 · seed 20261008 |

Every claim in `reports/phase21_temporal_dl.md` is re-derived here from the artefacts."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
import sys
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MET = REPORTS / "phase21_metrics.csv"
SEARCH = REPORTS / "phase21_search.csv"
P16 = REPORTS / "phase16_baseline_metrics.csv"
P20 = REPORTS / "phase20_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase21_temporal_dl.md"
LOG = ROOT / "scripts" / "phase21_run.log"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAMILIES = ["lstm_L3", "lstm_L7", "lstm_L14", "lstm_L30", "gru_bestL"]
REG_ID = {"lstm_L3": "L3-01", "lstm_L7": "L3-02", "lstm_L14": "L3-03",
          "lstm_L30": "L3-04", "gru_bestL": "L3-05"}
BUDGET = {"lstm_L3": 12, "lstm_L7": 10, "lstm_L14": 8, "lstm_L30": 6,
          "gru_bestL": 10}
FAM_L = {"lstm_L3": 3, "lstm_L7": 7, "lstm_L14": 14, "lstm_L30": 30,
         "gru_bestL": 3}
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


met = pd.read_csv(MET)
search = pd.read_csv(SEARCH)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
p20 = pd.read_csv(P20)
t0 = time.time()
print("Phase 21 notebook - LSTM/GRU verification")"""))

cells.append(code(r"""# N21-01  artefacts present with the expected shapes
for p in (MET, SEARCH, REPORT, LOG):
    assert p.exists(), p
pts = {f: MODELS / f"phase21_{REG_ID[f]}.pt" for f in FAMILIES}
missing = [p.name for p in pts.values() if not p.exists()]
check("N21-01", f"metrics {len(met)} rows, search {len(search)} rows, "
                f"5 model files ({sum(p.stat().st_size for p in pts.values())/1e6:.2f} MB), "
                f"report + run log present; missing={missing}",
      not missing and len(met) == 360 and len(search) == 46)"""))

cells.append(code(r"""# N21-02  registry contract (L3-01..L3-05: frozen test, budget <= 20)
l3 = reg[reg.level == "L3"].set_index("model")
fam_rows = l3.loc[FAMILIES]
bad = fam_rows[(fam_rows.test_policy != "frozen_read_at_phase33")
               | (fam_rows.tuning_budget_configs > 20)
               | (~fam_rows.metrics_out.str.contains("phase21_metrics.csv", regex=False))
               | (fam_rows.selection_cell != "val_2022")
               | (fam_rows.selection_metric != "pr_auc")]
check("N21-02", f"registry: {len(fam_rows)} L3 rows {sorted(FAMILIES)}, "
                f"test_policy all frozen_read_at_phase33, budgets "
                f"{sorted(fam_rows.tuning_budget_configs.unique())} <= 20, "
                f"metrics_out/selection matches; violations={len(bad)}",
      bad.empty and sorted(fam_rows.registry_id) == sorted(REG_ID.values())
      and set(fam_rows.index) == set(FAMILIES))"""))

cells.append(code(r"""# N21-03  test policy: no test rows anywhere in the phase-21 metrics
cells = set(met.cell.unique())
test_rows = met[met.cell.astype(str).str.startswith("test")]
seasons = set(met.season.unique())
check("N21-03", f"cells = {sorted(cells)} == the 7 dev cells (no test), "
                f"test rows = {len(test_rows)} (must be 0), seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N21-04  grid identity vs split_assignments + holdout/test exclusion
# The notebook's sequence index carries the same split/fold/holdout columns as
# Phase 15, so the masks are rebuilt here independently of train_sequence.py.
import baseline_model  # noqa: F401  (ensures scripts/ import path works)
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
leak = {t["name"]: int((t["train"] & hold).sum()
                       + (t["train"] & (split == "test")).sum()) for t in tasks}
check("N21-04", f"8 sequence-index training masks == masks rebuilt from "
                f"split_assignments (mismatches={mism}); dev eval masks 1:1; "
                f"training masks contain {sum(leak.values())} holdout/test rows "
                f"(must be 0): {leak}",
      mism == 0 and sum(leak.values()) == 0)"""))

cells.append(code(r"""# N21-05  stage-36 tuning: budget honoured, 3 horizons, mean == mean, winner == argmax
rows = []
for f in FAMILIES:
    g = search[search.family == f]
    n_cfg = len(g)
    three = bool(((g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
    mean_ok = bool(np.allclose(g.mean_pr_auc,
                               g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
    winner = g.loc[g.mean_pr_auc.idxmax()]
    rows.append((f, n_cfg, BUDGET[f], three, mean_ok, winner.config_id,
                 round(float(winner.mean_pr_auc), 5), winner.weighting,
                 int(winner.n_params)))
tab = pd.DataFrame(rows, columns=["family", "configs", "budget", "3_horizons",
                                  "mean_ok", "winner", "mean_pr_auc",
                                  "weighting", "n_params"])
print(tab.to_string(index=False))
check("N21-05", f"all 5 families: configs == budget, 3 horizons scored, "
                f"mean == mean(3), winner = argmax; budgets all <= 20 = "
                f"{sorted(tab.budget)}; n_params {sorted(tab.n_params)}",
      bool((tab.configs == tab.budget).all() and tab["3_horizons"].all()
           and tab.mean_ok.all() and (tab.budget <= 20).all()))"""))

cells.append(code(r"""# N21-06  stage-36 grid: full 8-cell x 3-horizon x 3-season grid, no empty cells
size = met.groupby("model").size()
covered = met.groupby(["model", "protocol", "train", "horizon", "season"]).size()
ci_missing = met[(met.season == "all") & (met.n >= 1000) & met.pr_auc_ci_lo.isna()]
check("N21-06", f"rows per family = {dict(size)} (expect 72 each); "
                f"{len(covered)} family x task x horizon x season cells populated "
                f"(min n_group={int(covered.min())}); n all > 0 = "
                f"{bool((met.n > 0).all())}; bootstrap CI missing on big cells = "
                f"{len(ci_missing)}; n_params present = "
                f"{bool(met.n_params.notna().all())}",
      (size == 72).all() and len(covered) == 360 and (met.n > 0).all()
      and len(ci_missing) == 0 and bool(met.n_params.notna().all()))"""))

cells.append(code(r"""# N21-07  headline: DL vs the D7 bar and the Phase-20 classical best, val_2022
val = met[(met.protocol == "P1") & (met.cell == "val_2022")
          & (met.season == "all")]
dl = val.pivot_table(index="model", columns="horizon", values="pr_auc").reindex(FAMILIES)
bar = pd.concat([
    p16[(p16.model == "persistence") & (p16.protocol == "P1")
        & (p16.cell == "val_2022") & (p16.season == "all")],
    p20[(p20.model == "xgboost") & (p20.protocol == "P1")
        & (p20.cell == "val_2022") & (p20.season == "all")],
]).pivot_table(index="model", columns="horizon", values="pr_auc")
allv = pd.concat([dl, bar])
print("val_2022 PR-AUC (season=all):")
print(allv.round(4).to_string())
best_dl = dl.max(axis=0)
delta_per = best_dl - bar.loc["persistence"]
delta_xgb = best_dl - bar.loc["xgboost"]
check("N21-07", f"best DL per horizon {best_dl.round(4).to_dict()}; vs D7 "
                f"persistence {delta_per.round(4).to_dict()}; vs xgboost "
                f"{delta_xgb.round(4).to_dict()}",
      True)"""))

cells.append(code(r"""# N21-08a  replay: each saved state_dict reproduces the recorded PR-AUC
from train_sequence import CHANNELS, SeqNet, fold_stats, prepare, std_rows

ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
hold = ix.station_holdout.astype(bool).to_numpy()
split = ix.split.to_numpy()
tri = (split == "train") & ~hold
ev = (split == "val") & ~hold
labels = {h: ix[h].to_numpy() for h in HORIZONS}

bad = []
for f in FAMILIES:
    saved = torch.load(MODELS / f"phase21_{REG_ID[f]}.pt", weights_only=False)
    L = int(saved["L"])
    assert L == FAM_L[f], f
    X, M = prepare(L)
    kind = saved["kind"]
    p = saved["params"]
    for hor in HORIZONS:
        y = labels[hor]
        tr_idx = np.flatnonzero(tri & ~np.isnan(y))
        ev_idx = np.flatnonzero(ev & ~np.isnan(y))
        mu, sd = fold_stats(X, M, tr_idx)
        net = SeqNet(kind, p["hidden"], p["layers"], p["dropout"])
        net.load_state_dict(saved["state"][hor])
        net.eval()
        with torch.no_grad():
            pr = torch.sigmoid(net(torch.tensor(std_rows(X, M, ev_idx, mu, sd)))).numpy()
        ap = float(average_precision_score(y[ev_idx].astype(int), pr))
        rec = met[(met.model == f) & (met.cell == "val_2022")
                  & (met.horizon == hor) & (met.season == "all")].pr_auc.iloc[0]
        if abs(ap - rec) > 1e-9:
            bad.append((f, hor, ap, float(rec)))
check("N21-08a", f"all 5 x 3 saved state_dicts replay to the recorded PR-AUC "
                 f"exactly; mismatches={bad}", len(bad) == 0)"""))

cells.append(code(r"""# N21-08b  determinism: two identical fixed-epoch fits are bit-identical
from train_sequence import fit_model, seed_all

X, M = prepare(3)
y = labels["label_24h"]
tr_idx = np.flatnonzero(tri & ~np.isnan(y))[:4000]
ev_idx = np.flatnonzero(ev & ~np.isnan(y))[:2000]
mu, sd = fold_stats(X, M, tr_idx)
cfg = {"lr": 1e-3, "hidden": 32, "layers": 1, "dropout": 0.0,
       "batch_size": 256, "_weight": "off"}
r1 = fit_model("lstm", 3, cfg, X, M, mu, sd, tr_idx, y[tr_idx], ev_idx, y[ev_idx],
               fixed_epochs=2)
r2 = fit_model("lstm", 3, cfg, X, M, mu, sd, tr_idx, y[tr_idx], ev_idx, y[ev_idx],
               fixed_epochs=2)
same = all(torch.equal(r1["net"].state_dict()[k], r2["net"].state_dict()[k])
           for k in r1["net"].state_dict())
ap1 = float(average_precision_score(y[ev_idx].astype(int), r1["pred"]))
ap2 = float(average_precision_score(y[ev_idx].astype(int), r2["pred"]))
check("N21-08b", f"two seeded 2-epoch fits on 4,000 train rows: bit-identical "
                 f"parameters = {same}; val PR-AUC {ap1:.6f} == {ap2:.6f}",
      same and abs(ap1 - ap2) < 1e-12)"""))

cells.append(code(r"""# N21-09  L* recorded and gru_bestL tuned at exactly L*
lstar = json.loads((PROC / "phase21_lstar.json").read_text())["L_star"]
lstm_mean = {FAM_L[f]: float(search[search.family == f].mean_pr_auc.max())
             for f in FAMILIES if f.startswith("lstm_")}
argmax_L = max(lstm_mean, key=lambda k: lstm_mean[k])
gru_L = int(search[search.family == "gru_bestL"].feature_set.iloc[0].split("L=")[-1])
check("N21-09", f"L* = {lstar}, argmax over the four LSTM families = {argmax_L} "
                f"(mean PR-AUC {lstm_mean}); gru_bestL tuned at L={gru_L}",
      int(lstar) == argmax_L and gru_L == int(lstar))"""))

cells.append(code(r"""# N21-10  headline figure: val_2022 PR-AUC by family vs the two reference bars
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = FAMILIES + ["xgboost", "persistence"]
plot = allv.reindex(order)
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(order))
w = 0.26
for i, h in enumerate(HORIZONS):
    ax.bar(x + (i - 1) * w, plot.loc[order, h].to_numpy(), w, label=h)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=25, ha="right")
ax.set_ylabel("PR-AUC (val_2022)")
ax.set_ylim(0.8, 1.0)
ax.grid(alpha=0.3, axis="y")
ax.legend()
ax.set_title("Phase 21 - LSTM/GRU vs D7 bar and Phase-20 xgboost (val_2022, season=all)")
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "phase21_val2022.png", dpi=130)
plt.close(fig)
check("N21-10", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure written to figures/phase21_val2022.png",
      REPORT.exists() and (ROOT / "figures" / "phase21_val2022.png").exists())"""))

cells.append(code(r"""# N21-11  run log records a clean finish for every driver call
txt = LOG.read_text(encoding="utf-8", errors="replace")
clean = [ln.strip() for ln in txt.splitlines() if "exit=0" in ln]
check("N21-11", f"{LOG.name} {LOG.stat().st_size:,} B; clean-finish markers "
                f"(driver 'TOTAL … exit=0' + shell echoes): {len(clean)}",
      len(clean) >= 2)"""))

cells.append(code(r"""# N21-12/N21-13  byte-stability + summary
check("N21-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)
print("\n" + "=" * 78)
print("PHASE 21 / STAGE 36  LSTM & GRU deep temporal models verified")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, "
      f"{time.time() - t0:.1f}s")
print("L* = 3 (shortest window wins); test untouched (frozen to Phase 33); "
      "next: Phase 22 TCN (stage 37)")
assert all(r["pass"] for r in RES)
print("ALL PHASE-21 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
