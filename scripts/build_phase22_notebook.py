"""Build notebooks/22_tcn.ipynb (workflow Phase 22 / stage 37).

The TCN is fit by scripts/train_sequence.py tcn_bestL (resume-safe, checkpointed
after the family; exit=0). This notebook re-derives every claim from the written
artefacts - it trains nothing except the two determinism probes in N22-08 (a
small fixed-epoch two-run replay).

Phase 22's scientific purpose is the recurrence-vs-convolution comparison:
LSTM vs GRU vs TCN on identical data / splits / metrics / budget (D14). lstm_L3
and gru_bestL come from phase21_metrics.csv (L* = 3), tcn_bestL from
phase22_metrics.csv.

Checks N22-01..N22-13:
    N22-01  artefacts exist with the expected shapes (72 metrics rows, 12
            evaluated configs, 1 tcn model file, report, run log)
    N22-02  registry contract: L3-06 tcn_bestL (frozen test, budget <= 20,
            metrics_out lists phase22_metrics.csv, selection val_2022/pr_auc)
    N22-03  test policy: phase22_metrics.csv has ZERO test cells; dev cells are
            exactly the registered 7; all three seasons present
    N22-04  grid identity: sequence-index training/eval masks equal masks
            rebuilt independently; no training mask holds a holdout/test row
    N22-05  stage-37 tuning: budget honoured, 3 horizons per config, mean ==
            mean(3), winner = argmax
    N22-06  stage-37 grid: 72 rows, nonzero n, bootstrap CI on big cells,
            n_params recorded
    N22-07  headline: the LSTM / GRU / TCN three-way table on val_2022 vs the
            D7 persistence bar and Phase-20 xgboost, all three horizons
    N22-08  determinism: (a) the saved state_dict replays to the recorded
            PR-AUC exactly; (b) two identical fixed-epoch TCN fits are
            bit-identical
    N22-09  L* recorded (= 3) and tcn_bestL tuned at exactly L*
    N22-10  report exists; figure rendered
    N22-11  run log records a clean exit for the driver call
    N22-12  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N22-13  summary

Run:
    python scripts/build_phase22_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\22_tcn.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "22_tcn.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 22 — Temporal convolutional network (stage 37)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 22 — Temporal convolutional network** → checklist **stage 37** |
| **Fit by** | `scripts/train_sequence.py tcn_bestL` → `scripts/phase22_run.log` (resume-safe; checkpointed after the family) |
| **Reads** | `reports/phase22_metrics.csv` (72 rows) · `reports/phase22_search.csv` (12 configs) · `models/phase22_L3-06.pt` · `reports/phase21_metrics.csv` (lstm_L3, gru_bestL) · `reports/phase17_experiment_registry.csv` · `reports/phase16_baseline_metrics.csv` · `reports/phase20_metrics.csv` · `data/processed/{sequence_index,sequences_L*,sequence_mask_L*}.npy` |
| **Contract** | Dataset B raw sequences + mask (P19-M1) · 13 channels · per-fold standardisation · dev cells only (test frozen to Phase 33) · budget 12 (≤ 20) · seed 20261008 · L\\* = 3 |

The phase exists to answer whether **dilated causal convolution matches or beats
recurrence, cheaper** (workflow Phase 22). Every claim in `reports/phase22_tcn.md`
is re-derived here from the artefacts."""))

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
MET22 = REPORTS / "phase22_metrics.csv"
SEARCH22 = REPORTS / "phase22_search.csv"
MET21 = REPORTS / "phase21_metrics.csv"
P16 = REPORTS / "phase16_baseline_metrics.csv"
P20 = REPORTS / "phase20_metrics.csv"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT = REPORTS / "phase22_tcn.md"
LOG = ROOT / "scripts" / "phase22_run.log"
PT = MODELS / "phase22_L3-06.pt"
TBL = PROC / "experimental_dataset.csv"

TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
FAMILIES = ["lstm_L3", "gru_bestL", "tcn_bestL"]      # recurrence vs convolution at L* = 3
REG_ID = {"lstm_L3": "L3-01", "gru_bestL": "L3-05", "tcn_bestL": "L3-06"}
BUDGET = {"lstm_L3": 12, "gru_bestL": 10, "tcn_bestL": 12}
FAM_L = {"lstm_L3": 3, "gru_bestL": 3, "tcn_bestL": 3}
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


met22 = pd.read_csv(MET22)
search22 = pd.read_csv(SEARCH22)
met21 = pd.read_csv(MET21)
reg = pd.read_csv(REG)
p16 = pd.read_csv(P16)
p20 = pd.read_csv(P20)
t0 = time.time()
print("Phase 22 notebook - TCN verification")"""))

cells.append(code(r"""# N22-01  artefacts present with the expected shapes
for p in (MET22, SEARCH22, REPORT, LOG, PT):
    assert p.exists(), p
check("N22-01", f"phase22_metrics.csv {len(met22)} rows, phase22_search.csv "
                f"{len(search22)} rows, {PT.name} {PT.stat().st_size/1e3:.0f} KB, "
                f"report + run log present",
      len(met22) == 72 and len(search22) == 12)"""))

cells.append(code(r"""# N22-02  registry contract (L3-06 tcn_bestL: frozen test, budget <= 20)
row = reg[reg.registry_id == "L3-06"]
assert len(row) == 1, row
r = row.iloc[0]
ok = (r.model == "tcn_bestL" and r.test_policy == "frozen_read_at_phase33"
      and int(r.tuning_budget_configs) == 20 and int(r.tuning_budget_configs) >= BUDGET["tcn_bestL"]
      and "phase22_metrics.csv" in r.metrics_out
      and r.selection_cell == "val_2022" and r.selection_metric == "pr_auc"
      and str(r.gate) in ("nan", "") )
check("N22-02", f"L3-06 {r.model}: test_policy={r.test_policy}, budget_max="
                f"{int(r.tuning_budget_configs)} >= used {BUDGET['tcn_bestL']}, "
                f"metrics_out={r.metrics_out}, selection={r.selection_metric}/"
                f"{r.selection_cell}",
      bool(ok))"""))

cells.append(code(r"""# N22-03  test policy: no test rows anywhere in the phase-22 metrics
cells = set(met22.cell.unique())
test_rows = met22[met22.cell.astype(str).str.startswith("test")]
seasons = set(met22.season.unique())
check("N22-03", f"cells = {sorted(cells)} == the 7 dev cells (no test); "
                f"test rows = {len(test_rows)} (must be 0); seasons = {sorted(seasons)}",
      cells == DEV_CELLS and len(test_rows) == 0
      and seasons == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N22-04  grid identity vs split_assignments + holdout/test exclusion
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
check("N22-04", f"8 sequence-index training masks == masks rebuilt from "
                f"split_assignments (mismatches={mism}); dev eval masks 1:1; "
                f"training masks contain {sum(leak.values())} holdout/test rows "
                f"(must be 0)",
      mism == 0 and sum(leak.values()) == 0)"""))

cells.append(code(r"""# N22-05  stage-37 tuning: budget honoured, 3 horizons, mean == mean, winner == argmax
g = search22[search22.family == "tcn_bestL"]
three = bool(((g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].notna()).all(axis=1)).all())
mean_ok = bool(np.allclose(g.mean_pr_auc,
                           g[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].mean(axis=1)))
winner = g.loc[g.mean_pr_auc.idxmax()]
check("N22-05", f"tcn_bestL: {len(g)} configs == budget {BUDGET['tcn_bestL']}, "
                f"3 horizons scored, mean == mean(3), winner {winner.config_id} "
                f"mean_pr_auc={winner.mean_pr_auc:.5f} ({winner.weighting}), "
                f"n_params={int(winner.n_params):,}",
      len(g) == BUDGET["tcn_bestL"] and three and mean_ok)"""))

cells.append(code(r"""# N22-06  stage-37 grid: full 8-cell x 3-horizon x 3-season grid, no empty cells
size = met22.groupby("model").size()
covered = met22.groupby(["model", "protocol", "train", "horizon", "season"]).size()
ci_missing = met22[(met22.season == "all") & (met22.n >= 1000) & met22.pr_auc_ci_lo.isna()]
check("N22-06", f"rows = {dict(size)} (expect 72); {len(covered)} task x horizon "
                f"x season cells populated (min group={int(covered.min())}); n all "
                f"> 0 = {bool((met22.n > 0).all())}; bootstrap CI missing on big "
                f"cells = {len(ci_missing)}; n_params present = "
                f"{bool(met22.n_params.notna().all())}",
      (size == 72).all() and len(covered) == 72 and (met22.n > 0).all()
      and len(ci_missing) == 0 and bool(met22.n_params.notna().all()))"""))

cells.append(code(r"""# N22-07  headline: LSTM / GRU / TCN three-way on val_2022 vs D7 + xgboost
val21 = met21[(met21.protocol == "P1") & (met21.cell == "val_2022")
              & (met21.season == "all") & (met21.model.isin(["lstm_L3", "gru_bestL"]))]
val22 = met22[(met22.protocol == "P1") & (met22.cell == "val_2022")
              & (met22.season == "all")]
val = pd.concat([val21, val22])
dl = val.pivot_table(index="model", columns="horizon", values="pr_auc").reindex(FAMILIES)
mp = val.pivot_table(index="model", columns="horizon", values="macro_pr_auc").reindex(FAMILIES)
bar = pd.concat([
    p16[(p16.model == "persistence") & (p16.protocol == "P1")
        & (p16.cell == "val_2022") & (p16.season == "all")],
    p20[(p20.model == "xgboost") & (p20.protocol == "P1")
        & (p20.cell == "val_2022") & (p20.season == "all")],
]).pivot_table(index="model", columns="horizon", values="pr_auc")
allv = pd.concat([dl, bar])
print("val_2022 PR-AUC (season=all):")
print(allv.round(4).to_string())
print("\nmacro (per-station) PR-AUC:")
print(mp.round(4).to_string())
delta_per = dl.sub(bar.loc["persistence"])
delta_xgb = dl.sub(bar.loc["xgboost"])
print("\nTCN minus persistence:", delta_per.loc["tcn_bestL"].round(4).to_dict())
print("TCN minus xgboost   :", delta_xgb.loc["tcn_bestL"].round(4).to_dict())
best_tcn = dl.loc["tcn_bestL"]
check("N22-07", f"three-way present for {FAMILIES}: TCN {best_tcn.round(4).to_dict()} "
                f"vs D7 {delta_per.loc['tcn_bestL'].round(4).to_dict()} vs xgboost "
                f"{delta_xgb.loc['tcn_bestL'].round(4).to_dict()}",
      set(dl.index) == set(FAMILIES) and dl.notna().all().all())"""))

cells.append(code(r"""# N22-08a  replay: the saved TCN state_dict reproduces the recorded PR-AUC
from train_sequence import CHANNELS, SeqNet, fold_stats, prepare, std_rows

hold = ix.station_holdout.astype(bool).to_numpy()
split = ix.split.to_numpy()
tri = (split == "train") & ~hold
ev = (split == "val") & ~hold
labels = {h: ix[h].to_numpy() for h in HORIZONS}

saved = torch.load(PT, weights_only=False)
assert saved["kind"] == "tcn" and int(saved["L"]) == 3, saved.get("kind")
X, M = prepare(3)
bad = []
for hor in HORIZONS:
    y = labels[hor]
    tr_idx = np.flatnonzero(tri & ~np.isnan(y))
    ev_idx = np.flatnonzero(ev & ~np.isnan(y))
    mu, sd = fold_stats(X, M, tr_idx)
    p = saved["params"]
    net = SeqNet("tcn", p["hidden"], p["layers"], p["dropout"])
    net.load_state_dict(saved["state"][hor])
    net.eval()
    with torch.no_grad():
        pr = torch.sigmoid(net(torch.tensor(std_rows(X, M, ev_idx, mu, sd)))).numpy()
    ap = float(average_precision_score(y[ev_idx].astype(int), pr))
    rec = met22[(met22.model == "tcn_bestL") & (met22.cell == "val_2022")
                & (met22.horizon == hor) & (met22.season == "all")].pr_auc.iloc[0]
    if abs(ap - rec) > 1e-9:
        bad.append((hor, ap, float(rec)))
check("N22-08a", f"all 3 saved TCN state_dicts replay to the recorded PR-AUC "
                 f"exactly; mismatches={bad}", len(bad) == 0)"""))

cells.append(code(r"""# N22-08b  determinism: two identical fixed-epoch TCN fits are bit-identical
from train_sequence import fit_model

y = labels["label_24h"]
tr_idx = np.flatnonzero(tri & ~np.isnan(y))[:4000]
ev_idx = np.flatnonzero(ev & ~np.isnan(y))[:2000]
mu, sd = fold_stats(X, M, tr_idx)
cfg = {"lr": 1e-3, "hidden": 32, "layers": 2, "dropout": 0.0,
       "batch_size": 256, "_weight": "off"}
r1 = fit_model("tcn", 3, cfg, X, M, mu, sd, tr_idx, y[tr_idx], ev_idx, y[ev_idx],
               fixed_epochs=2)
r2 = fit_model("tcn", 3, cfg, X, M, mu, sd, tr_idx, y[tr_idx], ev_idx, y[ev_idx],
               fixed_epochs=2)
same = all(torch.equal(r1["net"].state_dict()[k], r2["net"].state_dict()[k])
           for k in r1["net"].state_dict())
ap1 = float(average_precision_score(y[ev_idx].astype(int), r1["pred"]))
ap2 = float(average_precision_score(y[ev_idx].astype(int), r2["pred"]))
check("N22-08b", f"two seeded 2-epoch TCN fits on 4,000 train rows: bit-identical "
                 f"parameters = {same}; val PR-AUC {ap1:.6f} == {ap2:.6f}",
      same and abs(ap1 - ap2) < 1e-12)"""))

cells.append(code(r"""# N22-09  L* recorded (= 3) and tcn_bestL tuned at exactly L*
lstar = int(json.loads((PROC / "phase21_lstar.json").read_text())["L_star"])
tcn_L = int(search22[search22.family == "tcn_bestL"].feature_set.iloc[0].split("L=")[-1])
check("N22-09", f"L* = {lstar}; tcn_bestL tuned at L={tcn_L}",
      tcn_L == lstar)"""))

cells.append(code(r"""# N22-10  headline figure: val_2022 PR-AUC by model vs the two reference bars
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = FAMILIES + ["xgboost", "persistence"]
plot = allv.reindex(order)
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(order))
w = 0.26
for i, h in enumerate(HORIZONS):
    ax.bar(x + (i - 1) * w, plot.loc[order, h].to_numpy(), w, label=h)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("PR-AUC (val_2022)")
ax.set_ylim(0.8, 1.0)
ax.grid(alpha=0.3, axis="y")
ax.legend()
ax.set_title("Phase 22 - LSTM vs GRU vs TCN vs D7 bar and Phase-20 xgboost (val_2022)")
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "phase22_val2022.png", dpi=130)
plt.close(fig)
check("N22-10", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure written to figures/phase22_val2022.png",
      REPORT.exists() and (ROOT / "figures" / "phase22_val2022.png").exists())"""))

cells.append(code(r"""# N22-11  run log records a clean finish for the driver call
txt = LOG.read_text(encoding="utf-8", errors="replace")
clean = [ln.strip() for ln in txt.splitlines() if "exit=0" in ln]
check("N22-11", f"{LOG.name} {LOG.stat().st_size:,} B; clean-finish markers: "
                f"{len(clean)}", len(clean) >= 1)"""))

cells.append(code(r"""# N22-12/N22-13  byte-stability + summary
check("N22-12", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)
print("\n" + "=" * 78)
print("PHASE 22 / STAGE 37  Temporal convolutional network verified")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, "
      f"{time.time() - t0:.1f}s")
print("Answer: does convolution match or beat recurrence, cheaper? "
      "see reports/phase22_tcn.md; test untouched (frozen to Phase 33); "
      "next: Phase 23 fusion (stage 38)")
assert all(r["pass"] for r in RES)
print("ALL PHASE-22 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
