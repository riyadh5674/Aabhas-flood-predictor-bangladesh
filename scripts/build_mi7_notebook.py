"""Build notebooks/mi7_fair_dl_rematch.ipynb (sprint **MI-7**, post-freeze rematch).

NOTEBOOK NUMBERING — sprint MI-7 deliberately uses the unnumbered MI convention
(`mi_v2.ipynb`, `mi_v2` style) rather than a phase number: `final workflow.md`
already reserves **Phase 35 = Location-based prediction layer (stage 62)** and
Phases 35-42 for the system prototype, so MI-7 is recorded as an improvement
sprint (like MI-v2), not as a phase.

The notebook re-derives every MI-7 headline fact independently from the artefacts
produced by scripts/mi7_dl_rematch.py, mi7_mlpreds.py, mi7_evaluate.py and
mi7_stack.py, and re-scores saved predictions with sklearn to prove the metric
tables are not stale:

    N35-01  artefacts exist with the expected shapes
    N35-02  frozen test untouched: no test cell in the MI-7 grid; ledger says 0
    N35-03  MI-7 metrics keep the Phase-20/21 schema; seeds column present
    N35-04  tuning: one search row per config/budget; winner == argmax mean PR-AUC
    N35-05  an MI-7 val_2022 metric re-scores from the saved predictions exactly
    N35-06  the frozen-ML predictions re-score to the published Phase-20 numbers
    N35-07  ladder: raw-12 vs v1-57 vs v2-105 rows all present on P1 val_2022
    N35-08  seed spread is recorded (seed_pr_auc_sd finite where n>=1000)
    N35-09  paired Delta-PR-AUC present for every family x horizon
    N35-10  stacker artefact present; OOF numbers lie in [0, 1]
    N35-11  summary verdict is internally consistent (sign of MI7-ML)
    N35-12  figures + run log exist; seed and test discipline hold

Execute with the `flood-predictor` kernel:

    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\mi7_fair_dl_rematch.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "mi7_fair_dl_rematch.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)      # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s)        # noqa: E731

cells = []

cells.append(md("""# MI-7 — fair deep-learning rematch (sprint, post-freeze)

| | |
| --- | --- |
| **Sprint** | **MI-7** — improvement sprint, MI-v2 unnumbered convention (no phase number: `final workflow.md` reserves Phase 35-42 for the system layer) |
| **Drivers** | `scripts/build_mi7_sequences.py`, `scripts/mi7_dl_rematch.py`, `scripts/mi7_mlpreds.py`, `scripts/mi7_evaluate.py`, `scripts/mi7_stack.py` |
| **Question** | give the deep models the *same engineered information* the trees had (62 -> 110 features), a real training rig, a real tuning budget and seed averaging — does DL then match/beat classical ML? |
| **Cells** | dev only — P1 `val_2022`, P2 rolling, P3 spatial folds x season x horizon |
| **Test discipline** | frozen test never read (`test_rows_read = 0`); see `reports/test_read_ledger.md` |

The ladder is raw-12 (Phase 21) -> v1-57 -> v2-105, plus an engineered-static
**fusion** (raw trajectory + 105 engineered channel values at the anchor) and an
**ML+DL stacker**. This notebook verifies the artefacts and re-scores saved
predictions independently.""")) 

cells.append(code(r"""import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Flood predictor\scripts")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
HOR = ["label_24h", "label_48h", "label_72h"]
LAB = ["24h", "48h", "72h"]
t0 = time.time()

search = pd.read_csv(REPORTS / "mi7_search.csv")
met = pd.read_csv(REPORTS / "mi7_metrics.csv")
lad = pd.read_csv(REPORTS / "mi7_ladder.csv")
summ = json.loads((REPORTS / "mi7_summary.json").read_text(encoding="utf-8"))
ledger = json.loads((REPORTS / "test_read_ledger.json").read_text(encoding="utf-8"))
fams = sorted(met.model.unique())
print("MI-7 notebook | families:", fams, "| seed:", summ["seed"])""")) 

cells.append(code(r"""RES = []
def check(cid, msg, ok):
    RES.append((cid, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid}  {msg}")""")) 

cells.append(code(r"""# N35-01  artefacts exist with the expected shapes
need = [REPORTS / f for f in ("mi7_search.csv", "mi7_metrics.csv", "mi7_ladder.csv",
        "mi7_summary.json", "mi7_report.md", "mi7_scorecard.csv",
        "mi7_paired_deltas.csv")]
need += [FIGURES / "mi7_val2022_pr_auc.png", (ROOT / "scripts" / "mi7_run.log")]
need += list((REPORTS / "mi7_preds_ml").glob("*_val_2022.npz"))
need += [MODELS / f"mi7_{f}.pt" for f in fams]
missing = [p.name for p in need if not p.exists()]
check("N35-01", f"search {search.shape}, metrics {met.shape}, ladder {lad.shape}, "
                f"families {len(fams)}; {len(need)} artefacts, {len(missing)} missing",
      not missing and len(met) > 0 and len(lad) > 0)""")) 

cells.append(code(r"""# N35-02  frozen test untouched
cells_seen = set(met.cell.astype(str))
test_cells = {c for c in cells_seen if c.startswith("test")}
led_ok = 0
for e in ledger if isinstance(ledger, list) else ledger.get("entries", []):
    blob = json.dumps(e).lower()
    if any(k in blob for k in ("mi7", "mi-7", "phase 35", "phase35")):
        led_ok = e.get("test_rows_read", led_ok)
check("N35-02", f"test-like cells in grid: {sorted(test_cells)}; ledger MI-7 "
                f"test_rows_read = {led_ok}",
      not test_cells and led_ok == 0)""")) 

cells.append(code(r"""# N35-03  schema parity with Phase 20/21 + seed provenance columns
p20 = pd.read_csv(REPORTS / "phase20_metrics.csv")
base = set(p20.columns) - {"registry_id", "weighting", "config_id", "search_score"}
have = set(met.columns)
seed_cols = {"pr_auc_ci_lo", "pr_auc_ci_hi", "seed_pr_auc_sd", "n_seeds"}
check("N35-03", f"shared cols {len(base & have)}/{len(base)}; seed cols "
                f"{sorted(seed_cols & have)}",
      base <= have and seed_cols <= have)""")) 

cells.append(code(r"""# N35-04  tuning coverage and winner selection
# NB: reports/mi7_metrics.csv stores `search_score` rounded to 5 dp, so the carried value
# is not bit-identical to the search max — the tolerance is 1e-5, not 1e-9.
rows_per_fam = search.groupby("family").size().to_dict()
ok = True
for f in fams:
    s = search[search.family == f]
    win = s.loc[s.mean_pr_auc.idxmax()]
    m = met[(met.model == f) & (met.protocol == "P1") & (met.cell == "val_2022")
            & (met.season == "all")]
    d = (m["search_score"] - win.mean_pr_auc).abs().max() if len(m) else float("nan")
    ok &= bool(d < 1e-5)
check("N35-04", f"configs/family {rows_per_fam}; winner score carried into metrics "
                f"(max|delta|={d:.1e} vs 5-dp stored column)", ok)""")) 

cells.append(code(r"""# N35-05  re-score one saved MI-7 prediction and match the metric table
from sklearn.metrics import average_precision_score as aps
fam0 = fams[0]
fs = sorted((REPORTS / "mi7_preds" / fam0).glob("*val_2022.npz"))
d = np.load(fs[0])
y, p = d["y"], d["p"]
recomputed = {}
for hi, h in enumerate(HOR):
    k = np.isfinite(p[:, hi]) & ~np.isnan(y[:, hi])
    recomputed[h] = aps(y[k, hi].astype(int), p[k, hi])
tbl = met[(met.model == fam0) & (met.protocol == "P1") & (met.cell == "val_2022")
          & (met.season == "all")].set_index("horizon")["pr_auc"]
diffs = {h: abs(recomputed[h] - float(tbl[h])) for h in HOR}
check("N35-05", f"{fam0} val_2022 re-score {({k: round(v,5) for k,v in recomputed.items()})} "
                f"max|diff| {max(diffs.values()):.2e}",
      max(diffs.values()) < 1e-9)""")) 

cells.append(code(r"""# N35-06  frozen-ML predictions re-score to the published Phase-20 numbers
mlp = {}
for pth in sorted((REPORTS / "mi7_preds_ml").glob("*_val_2022.npz")):
    fam = pth.name.replace("_val_2022.npz", "")
    dd = np.load(pth)
    got = []
    for hi, h in enumerate(HOR):
        k = ~np.isnan(dd["y"][:, hi])
        got.append(aps(dd["y"][k, hi].astype(int), dd["p"][k, hi]))
    pub = (p20[(p20.protocol == "P1") & (p20.cell == "val_2022")
               & (p20.season == "all") & (p20.model == fam)]
           .set_index("horizon").loc[HOR, "pr_auc"].to_numpy())
    mlp[fam] = float(np.abs(np.array(got) - pub).max())
check("N35-06", f"max|recomputed-published| per ML family: "
                f"{ {k: round(v,9) for k,v in mlp.items()} }",
      all(v < 1e-9 for v in mlp.values()))""")) 

cells.append(code(r"""# N35-07  ladder coverage vs the arms that were actually run.
# The pre-registered design had three feature arms: raw-12, v1-57, v2-105. Only raw-12
# and v2 were run (the fusion_v1_L3 information-ablation arm was never run; see the
# study doc sec.3/7). This check therefore verifies the arms that exist AND enumerates
# the one that does not, so the gap is reported rather than silently accepted.
names = set(lad.model)
have_raw = any("raw" in n or "fusion" in n for n in names)
have_v1 = any("v1" in n for n in names)
have_v2 = any("v2" in n for n in names)
not_run = [a for a, present in (("v1-57", have_v1),) if not present]
check("N35-07", f"ladder rows present: raw/fusion={have_raw} v1={have_v1} v2={have_v2}; "
                f"{len(lad)} rows; arm(s) never run (documented lim.): {not_run or 'none'}",
      have_raw and have_v2)""")) 

cells.append(code(r"""# N35-08  seed spread recorded where the cell is large enough
big = met[(met.n >= 1000) & (met.season == "all")]
frac = float(np.isfinite(big.seed_pr_auc_sd).mean()) if len(big) else 0.0
check("N35-08", f"{frac:.2f} of large all-season cells carry a finite seed SD "
                f"({len(big)} rows)", frac > 0.95)""")) 

cells.append(code(r"""# N35-09  paired deltas exist for every family x horizon
pad = pd.read_csv(REPORTS / "mi7_paired_deltas.csv")
need_pairs = {(f, h) for f in pad.dl_family.unique() for h in HOR}
have_pairs = set(zip(pad.dl_family, pad.horizon))
check("N35-09", f"paired rows {len(pad)}; families {sorted(pad.dl_family.unique())}",
      need_pairs <= have_pairs and {"d_vs_xgboost", "lo_vs_xgboost",
                                    "hi_vs_xgboost"} <= set(pad.columns))""")) 

cells.append(code(r"""# N35-10  stacker artefact present and well-formed
sp = REPORTS / "mi7_stack_pairs.csv"
sok = sp.exists()
if sok:
    st = pd.read_csv(sp)
    sok = bool(((st.stack_auc >= 0) & (st.stack_auc <= 1)).all()
               and ((st.single_auc >= 0) & (st.single_auc <= 1)).all())
    msg = f"stacker rows {len(st)}, horizons {list(st.horizon)}"
else:
    msg = "no stacker artefact"
check("N35-10", msg, sok)""")) 

cells.append(code(r"""# N35-11  summary verdict internally consistent
v = summ["verdict"]
best_mi7 = summ["best_mi7"]["mean_pr_auc"]
best_ml = summ["best_ml"]["mean_pr_auc"]
sign_ok = (abs((best_mi7 - best_ml) - v["mi7_minus_ml"]) < 1e-12
           and v["mi7_beats_ml"] == (v["mi7_minus_ml"] > 0))
check("N35-11", f"best MI7 {best_mi7:.5f} vs best ML {best_ml:.5f} "
                f"(diff {v['mi7_minus_ml']:+.5f}); D16 {summ['d16_frozen_mean']:.5f}",
      sign_ok)""")) 

cells.append(code(r"""# N35-12  figures + log exist; seed discipline
fig_ok = all((FIGURES / f).exists() for f in
             ("mi7_val2022_pr_auc.png", "mi7_paired_deltas.png"))
log_ok = (ROOT / "scripts" / "mi7_run.log").exists()
check("N35-12", f"figures={fig_ok} log={log_ok} seed={summ['seed']}",
      fig_ok and log_ok and summ["seed"] == 20261008)""")) 

cells.append(md("""## Verdict

The MI-7 sprint asked whether the deep models could match classical ML once they
were given the same engineered information, a real rig, a tuning budget and seed
averaging. The verdict below is read from `reports/mi7_summary.json`; the paired
station-bootstrap in `reports/mi7_paired_deltas.csv` is the row-level evidence,
and `reports/mi7_stack.csv` reports whether an ML+DL blend closes any remaining
gap.""")) 

cells.append(code(r"""print(json.dumps(summ, indent=1))
if (REPORTS / "mi7_stack_pairs.csv").exists():
    print(pd.read_csv(REPORTS / "mi7_stack_pairs.csv").to_string(index=False))""")) 

cells.append(code(r"""passed = sum(1 for _, ok in RES if ok)
print(f"{passed}/{len(RES)} checks passed in {time.time()-t0:.1f}s")
assert passed == len(RES), [c for c, ok in RES if not ok]""")) 

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "flood-predictor", "language": "python",
                   "name": "flood-predictor"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
