"""Build notebooks/26_multihorizon.ipynb (workflow Phase 26 / stage 41 + 59/60/61).

Phase 26 is an integration phase: it assembles every surviving L1-L4 model into one
model x horizon matrix and answers how difficulty grows with lead time and whether
it grows differently for engineered-feature (L1/L2) than sequence/DL (L3/L4) models.
The matrix is built by scripts/phase26_matrix.py; this notebook re-derives every
claim independently from the seven per-phase metrics files (no training at all).

Checks N26-01..N26-12:
    N26-01  artefacts exist with the expected shapes
    N26-02  registry coverage: 24 L1-L4 models, ids match, 4 levels, dev cells only
    N26-03  consolidated grid: 1728 rows, one row per source row, no test cells
    N26-04  P1 val_2022 matrix re-derived from the sources == phase26_summary.csv
    N26-05  P2 rolling-mean and P3 spatial-mean matrices re-derived == summary
    N26-06  lead-time decay: monotone for every competitive model; drop table
    N26-07  interaction (temporal cells): group means/medians == interaction.json
    N26-08  interaction (unseen stations): engineered decays faster; fusion flat
    N26-09  stage 59/60/61: per-horizon leaders re-derived
    N26-10  experimental_dataset.csv SHA-256 unchanged (byte-stable)
    N26-11  report exists; figure rendered
    N26-12  summary

Run:
    python scripts/build_phase26_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\26_multihorizon.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "26_multihorizon.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 26 — Multi-horizon strategy (stage 41 + stages 59–61)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 26 — Multi-horizon strategy** → **stage 41** (+ **59, 60, 61**) |
| **Built by** | `scripts/phase26_matrix.py` → `reports/phase26_matrix.csv`, `phase26_summary.csv`, `phase26_interaction.json` |
| **Reads** | `reports/phase16_baseline_metrics.csv` (L1) · `phase20_metrics.csv` (L2) · `phase21_metrics.csv` (L3 LSTM/GRU) · `phase22_metrics.csv` (L3 TCN) · `phase23_metrics.csv` (L3 fusion) · `phase24_metrics.csv` (L4 GNN) · `phase25_metrics.csv` (L4 Transformer) · `reports/phase17_experiment_registry.csv` |
| **Contract** | integration only (no training) · dev cells only (test frozen to Phase 33) · 3 horizons, never pooled (D8) · pr_auc primary |

Every claim in `reports/phase26_multihorizon.md` is re-derived here directly from the
seven source metrics files — the consolidated grid is *rebuilt*, not merely read back."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"

MATRIX = REPORTS / "phase26_matrix.csv"
SUMMARY = REPORTS / "phase26_summary.csv"
INTER = REPORTS / "phase26_interaction.json"
REPORT = REPORTS / "phase26_multihorizon.md"
REG = REPORTS / "phase17_experiment_registry.csv"
TBL = PROC / "experimental_dataset.csv"
FIGDIR = ROOT / "figures"

# source metrics files -> registry level
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
SEED = 20261008
HOR = ["label_24h", "label_48h", "label_72h"]
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}

REG_ID = {
    "climatology": "L1-01", "persistence": "L1-02", "logreg_linear": "L1-03",
    "logreg_linear_no_up": "L1-04", "logreg_selected": "L1-05", "tree_depth4": "L1-06",
    "random_forest": "L2-01", "gradient_boosting": "L2-02", "xgboost": "L2-03",
    "lightgbm": "L2-04", "xgboost_selected": "L2-05",
    "lstm_L3": "L3-01", "lstm_L7": "L3-02", "lstm_L14": "L3-03", "lstm_L30": "L3-04",
    "gru_bestL": "L3-05", "tcn_bestL": "L3-06", "fusion_L3": "L3-07",
    "static_only": "L3-08", "hybrid_L3": "L3-09",
    "gcn": "L4-01", "gat": "L4-02", "transformer_bestL": "L4-03", "gnn_lstm": "L4-04",
}
LVL = {m: rid.split("-")[0] for m, rid in REG_ID.items()}
TRIVIAL = {"climatology", "persistence"}
ENGINEERED = {"L1", "L2"}
SEQUENCE = {"L3", "L4"}


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


# --- rebuild the consolidated grid independently from the seven sources ---
frames = []
for fn, lvl in SOURCES:
    d = pd.read_csv(REPORTS / fn)
    d["level"] = lvl
    d["source"] = fn
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
raw["registry_id"] = raw["model"].map(REG_ID)

matrix = pd.read_csv(MATRIX)
summary = pd.read_csv(SUMMARY)
inter = json.loads(INTER.read_text(encoding="utf-8"))
reg = pd.read_csv(REG)
t0 = time.time()
print("Phase 26 notebook - multi-horizon matrix verification")"""))

cells.append(code(r"""# N26-01  artefacts present with the expected shapes
for p in (MATRIX, SUMMARY, INTER, REPORT):
    assert p.exists(), p
check("N26-01", f"matrix {len(matrix)} rows, summary {len(summary)} rows, "
                f"interaction json {len(inter)} keys, report {REPORT.stat().st_size:,} B",
      len(matrix) == 1728 and len(summary) == 72
      and REPORT.stat().st_size > 5000)"""))

cells.append(code(r"""# N26-02  registry coverage: 24 models, ids match, 4 levels, dev cells only
m_ids = set(matrix.registry_id)
r_ids = set(reg.registry_id)
models = sorted(matrix.model.unique())
levels = sorted(matrix.level.unique())
check("N26-02", f"{len(models)} models across levels {levels}; registry ids match "
                f"({len(m_ids)} == {len(r_ids)}): {m_ids == r_ids}; "
                f"seasons {sorted(matrix.season.unique())}; cells "
                f"{sorted(matrix.cell.unique())} == the 7 dev cells",
      len(models) == 24 and m_ids == r_ids
      and levels == ["L1", "L2", "L3", "L4"]
      and set(matrix.cell.unique()) == DEV_CELLS
      and set(matrix.season.unique()) == {"all", "monsoon", "non-monsoon"})"""))

cells.append(code(r"""# N26-03  consolidated grid == the sources, row for row; no test cells anywhere
per_src = matrix.groupby("source").size().to_dict()
raw_src = raw.groupby("source").size().to_dict()
no_test = not matrix.cell.astype(str).str.startswith("test").any()
cols_ok = {"model", "protocol", "cell", "season", "horizon", "n", "pr_auc",
           "roc_auc", "brier", "level", "family", "registry_id", "source",
           "horizon_h"} <= set(matrix.columns)
check("N26-03", f"rows {len(matrix)} == {len(raw)} rebuilt; per-source counts match "
                f"= {per_src == raw_src}; pr_auc non-null "
                f"{int(matrix.pr_auc.notna().sum())}/{len(matrix)}; no test cell = "
                f"{no_test}; horizons {sorted(matrix.horizon.unique())}",
      len(matrix) == len(raw) == 1728 and per_src == raw_src
      and matrix.pr_auc.notna().all() and no_test and cols_ok
      and set(matrix.horizon.unique()) == set(HOR))"""))

cells.append(code(r"""# N26-04  P1 val_2022 matrix, re-derived from the sources == phase26_summary.csv
def pivot(df, protocol, cell):
    d = df[(df.protocol == protocol) & (df.season == "all")]
    d = d[d.cell == cell] if cell else d.groupby(["model", "horizon_h"])["pr_auc"].mean().reset_index()
    return d.pivot_table(index="model", columns="horizon_h", values="pr_auc").reindex(columns=[24, 48, 72])

raw["horizon_h"] = raw["horizon"].str.extract(r"(\d+)").astype(int)
p1 = pivot(raw, "P1", "val_2022")
s1 = summary[summary.protocol == "P1"].set_index("model")
s1 = s1.reindex(p1.index)
recon = np.allclose(p1.to_numpy(), s1[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].to_numpy(),
                    atol=1e-5)
drop_ok = np.allclose((p1[72] - p1[24]).to_numpy(), s1["drop_72_24"].to_numpy(), atol=2e-5)
print("P1 val_2022 PR-AUC (season=all):")
print(p1.round(4).to_string())
check("N26-04", f"rebuilt P1 matrix matches phase26_summary.csv exactly "
                f"(values {recon}, drops {drop_ok}); 24 rows", bool(recon and drop_ok
      and len(p1) == 24))"""))

cells.append(code(r"""# N26-05  P2 rolling-mean and P3 spatial-mean matrices, re-derived
p2 = raw[(raw.protocol == "P2") & (raw.season == "all")].groupby(
    ["model", "horizon_h"])["pr_auc"].mean().unstack().reindex(columns=[24, 48, 72])
p3 = raw[(raw.protocol == "P3") & (raw.season == "all")].groupby(
    ["model", "horizon_h"])["pr_auc"].mean().unstack().reindex(columns=[24, 48, 72])
s2 = summary[summary.protocol == "P2"].set_index("model").reindex(p2.index)
s3 = summary[summary.protocol == "P3"].set_index("model").reindex(p3.index)
ok2 = np.allclose(p2.to_numpy(), s2[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].to_numpy(), atol=1e-5)
ok3 = np.allclose(p3.to_numpy(), s3[["pr_auc_24h", "pr_auc_48h", "pr_auc_72h"]].to_numpy(), atol=1e-5)
print("P3 spatial-CV mean PR-AUC:")
print(p3.round(4).to_string())
check("N26-05", f"rebuilt P2 (3 years) {ok2} and P3 (4 folds) {ok3} matrices match "
                f"summary; P2/P3 each 24 rows",
      bool(ok2 and ok3 and len(p2) == 24 and len(p3) == 24))"""))

cells.append(code(r"""# N26-06  lead-time decay: monotone for every competitive model
competitive = [m for m in p1.index if m not in TRIVIAL and m != "static_only"]
mono = {m: (p1.loc[m, 24] >= p1.loc[m, 48] >= p1.loc[m, 72]) for m in competitive}
drops = (p1[72] - p1[24]).sort_values()
print("lead-time decay (P1, 24h -> 72h), most -> least decay:")
for m, v in drops.items():
    print(f"  {m:22s} {LVL[m]}  {p1.loc[m,24]:.4f} -> {p1.loc[m,72]:.4f}  {v:+.4f}")
check("N26-06", f"every competitive model decays monotonically with lead time = "
                f"{all(mono.values())} ({len(competitive)} models); steepest = "
                f"{drops.index[0]} {drops.iloc[0]:+.4f}; gentlest competitive = "
                f"{drops[drops.index.isin(competitive)].index[-1]} "
                f"{drops[drops.index.isin(competitive)].iloc[-1]:+.4f}",
      all(mono.values()) and drops.index[0] == "persistence")"""))

cells.append(code(r"""# N26-07  interaction on the temporal cells: group stats == interaction.json
def group_stats(pv):
    nt = [m for m in pv.index if m not in TRIVIAL]
    eng = [m for m in nt if LVL[m] in ENGINEERED]
    seq = [m for m in nt if LVL[m] in SEQUENCE]
    e, s = pv.loc[eng], pv.loc[seq]
    return {
        "engineered_24_48_72": [round(float(e[h].mean()), 5) for h in (24, 48, 72)],
        "sequence_24_48_72": [round(float(s[h].mean()), 5) for h in (24, 48, 72)],
        "engineered_drop_mean": round(float((e[72] - e[24]).mean()), 5),
        "sequence_drop_mean": round(float((s[72] - s[24]).mean()), 5),
        "engineered_drop_median": round(float(np.median(e[72] - e[24])), 5),
        "sequence_drop_median": round(float(np.median(s[72] - s[24])), 5),
        "best_engineered": e.mean(axis=1).idxmax(),
        "best_engineered_drop": round(float(e.loc[e.mean(axis=1).idxmax(), 72]
                                            - e.loc[e.mean(axis=1).idxmax(), 24]), 5),
        "best_sequence": s.mean(axis=1).idxmax(),
        "best_sequence_drop": round(float(s.loc[s.mean(axis=1).idxmax(), 72]
                                          - s.loc[s.mean(axis=1).idxmax(), 24]), 5),
    }


gs1 = group_stats(p1)
ref1 = inter["interaction"]["P1"]
scalar1 = all(abs(gs1[k] - ref1[k]) < 1e-9 for k in gs1
              if isinstance(gs1[k], (int, float)))
seq1 = all(np.allclose(gs1[k], ref1[k])
           for k in ("engineered_24_48_72", "sequence_24_48_72"))
name1 = all(gs1[k] == ref1[k] for k in ("best_engineered", "best_sequence"))
match1 = scalar1 and seq1 and name1
print("P1 interaction:", json.dumps(gs1, indent=2))
check("N26-07", f"P1 group means/medians and best-per-group reproduce interaction.json "
                f"= {match1}; engineered mean {gs1['engineered_drop_mean']:+.4f} vs "
                f"sequence {gs1['sequence_drop_mean']:+.4f} (medians "
                f"{gs1['engineered_drop_median']:+.4f} / {gs1['sequence_drop_median']:+.4f})",
      match1)"""))

cells.append(code(r"""# N26-08  interaction on unseen stations (P3): engineered decays faster; fusion flat
gs3 = group_stats(p3)
fusion_p3_drop = float(p3.loc["fusion_L3", 72] - p3.loc["fusion_L3", 24])
eng_faster = gs3["engineered_drop_mean"] < gs3["sequence_drop_mean"]
print(f"P3: engineered mean drop {gs3['engineered_drop_mean']:+.4f} vs sequence "
      f"{gs3['sequence_drop_mean']:+.4f}; fusion_L3 P3 drop {fusion_p3_drop:+.4f}; "
      f"best engineered {gs3['best_engineered']} ({gs3['best_engineered_drop']:+.4f}) / "
      f"best sequence {gs3['best_sequence']} ({gs3['best_sequence_drop']:+.4f})")
check("N26-08", f"on P3 engineered-feature models decay faster than sequence models "
                f"= {eng_faster} ({gs3['engineered_drop_mean']:+.4f} < "
                f"{gs3['sequence_drop_mean']:+.4f}); fusion_L3 is flat/rising across "
                f"horizons on P3 (drop {fusion_p3_drop:+.4f} > 0)",
      eng_faster and fusion_p3_drop > 0
      and gs3["best_sequence"] == "fusion_L3")"""))

cells.append(code(r"""# N26-09  stage 59/60/61: per-horizon leaders re-derived
leaders = {}
for h in (24, 48, 72):
    leaders[h] = p1[h].idxmax()
print("P1 leaders:", {f"{h}h": leaders[h] for h in (24, 48, 72)})
print("P3 leaders:", {f"{h}h": p3[h].idxmax() for h in (24, 48, 72)})
stages_ok = (leaders[24] == "lightgbm" and leaders[48] == "xgboost"
             and leaders[72] == "xgboost_selected")
check("N26-09", f"P1 24h leader {leaders[24]}, 48h {leaders[48]}, 72h {leaders[72]} "
                f"(stage 59/60/61); P1 best is an L2 gradient-boosting model at every "
                f"horizon = {all(LVL[leaders[h]] == 'L2' for h in (24, 48, 72))}",
      stages_ok and all(LVL[leaders[h]] == "L2" for h in (24, 48, 72)))"""))

cells.append(code(r"""# N26-10  SHA-256 of the immutable experimental table
check("N26-10", f"experimental_dataset.csv SHA-256 = {sha256(TBL)[:12]}... "
                f"(unchanged: {sha256(TBL) == TBL_SHA})",
      sha256(TBL) == TBL_SHA)"""))

cells.append(code(r"""# N26-11  report + figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR.mkdir(exist_ok=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.9))

po = ["xgboost", "random_forest", "fusion_L3", "gcn", "lstm_L3", "tcn_bestL",
      "gru_bestL", "persistence"]
x = np.arange(len(po)); w = 0.26
for i, h in enumerate((24, 48, 72)):
    ax.bar(x + (i - 1) * w, p1.loc[po, h].to_numpy(), w, label=f"{h}h")
ax.set_xticks(x); ax.set_xticklabels(po, rotation=20, ha="right")
ax.set_ylabel("PR-AUC (val_2022)"); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3, axis="y")
ax.legend(); ax.set_title("Phase 26 - model x horizon matrix (P1)")

# interaction: mean PR-AUC curves per group, P1 vs P3
hs = np.array([24, 48, 72])
for j, (pv, lab, ls) in enumerate([(p1, "P1 temporal", "-"), (p3, "P3 unseen", "--")]):
    gs = group_stats(pv)
    ax2.plot(hs, gs["engineered_24_48_72"], "o" + ls, color="C0",
             label=f"engineered (L1/L2) {lab}")
    ax2.plot(hs, gs["sequence_24_48_72"], "s" + ls, color="C1",
             label=f"sequence (L3/L4) {lab}")
ax2.set_xticks(hs); ax2.set_xlabel("lead time (h)"); ax2.set_ylabel("mean PR-AUC")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
ax2.set_title("ML vs DL across lead time: P1 vs P3")
fig.tight_layout()
fig.savefig(FIGDIR / "phase26_horizon_matrix.png", dpi=130)
plt.close(fig)
check("N26-11", f"report {REPORT.name} present ({REPORT.stat().st_size:,} B); "
                f"figure figures/phase26_horizon_matrix.png written",
      REPORT.exists() and (FIGDIR / "phase26_horizon_matrix.png").exists())"""))

cells.append(code(r"""# N26-12  summary
print("\n" + "=" * 78)
print("PHASE 26 / STAGE 41 (+ 59, 60, 61)  Multi-horizon matrix verified")
print("=" * 78)
for rr in RES:
    print(f"  {'PASS' if rr['pass'] else 'FAIL'}  {rr['id']}")
check("N26-12", f"{sum(rr['pass'] for rr in RES) + 1}/{len(RES) + 1} checks passed, "
                f"{time.time() - t0:.1f}s", True)
print("Verdict: difficulty grows with lead time for every competitive model; the "
      "ML-vs-DL interaction is small on P1/P2 and real on unseen stations (P3).")
print("See reports/phase26_multihorizon.md; test untouched (frozen to Phase 33); "
      "next: Phase 27 fair ML-vs-DL comparison")
assert all(rr["pass"] for rr in RES)
print("ALL PHASE-26 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
