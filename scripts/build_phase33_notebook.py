"""Build notebooks/33_selection.ipynb (workflow Phase 33 / stage 57).

Phase 33 selects the final model from the frozen development evidence, freezes it, then
reads the frozen test **once** for the selected model + the D7 bar.  This notebook
re-derives the headline facts independently from the produced artefacts *and* from the raw
experimental table:

    N33-01  artefacts exist with the expected shapes
    N33-02  experimental-table SHA-256 unchanged (immutable)
    N33-03  the frozen test was read exactly once; D7 reproduces the Phase-16 read
    N33-04  scorecard: 24 models, exactly 4 eligible (all L2), ranks 1..4
    N33-05  the selection is xgboost (L2-03) with the pre-registered rule recorded
    N33-06  primary + co-primary both favour the selected model among the eligible set
    N33-07  selected beats D7 on 20/21 frozen-test cell x horizon points (1 tie/loss)
    N33-08  test metrics finite; all buckets/seasons present; recall gain over D7
    N33-09  the frozen model reloads and reproduces the P1 val_2022 PR-AUC (independence)
    N33-10  frozen artefacts exist; config carries seed, config_id, feature set, SHA
    N33-11  the four eligible models are statistically inseparable (CIs overlap)
    N33-12  seasons + figure present; thresholds frozen (Phase-29) before the test read
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "33_selection.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 33 — Final model selection (stage 57)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 33 — Final model selection** → **stage 57** |
| **Built by** | `scripts/phase33_selection.py` (scorecard on dev evidence → freeze → one test read) |
| **Selection evidence** | the frozen Phase-16/20-32 dev grid (validation + spatial CV only) |
| **Test** | read **once**, for the selected model **and** the D7 bar, then logged |
| **Rule** | gates (beat D7 everywhere, calibrated) → rank by horizon-mean P1 PR-AUC → fewest missed floods |

A multi-criteria scorecard, not "highest accuracy wins": PR-AUC vs the D7 bar is primary,
flood recall / missed-flood count at the Phase-29 cost-optimal threshold is co-primary
(FN >> FP), calibration is a gate, temporal (P2) and spatial (P3) generalisation are strong
evidence, and interpretability + cost are tie-breakers.""")) 

cells.append(code(r"""import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Flood predictor\scripts")
import baseline_model as B

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TABLE = ROOT / "data" / "processed" / "experimental_dataset.csv"
FIG = ROOT / "figures" / "phase33_scorecard.png"
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HOR = ["label_24h", "label_48h", "label_72h"]
DISP = ["24h", "48h", "72h"]
YEARS = ["test_2023", "test_2024", "test_2025", "test_2026"]
t0 = time.time()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


sc = pd.read_csv(REPORTS / "phase33_scorecard.csv")
tm = pd.read_csv(REPORTS / "phase33_test_metrics.csv")
sel = json.loads((REPORTS / "phase33_selection.json").read_text(encoding="utf-8"))
summary = json.loads((REPORTS / "phase33_summary.json").read_text(encoding="utf-8"))
log = json.loads((REPORTS / "phase33_test_read_log.json").read_text(encoding="utf-8"))
cfg = json.loads((MODELS / "final" / "final_config.json").read_text(encoding="utf-8"))
CHOSEN = summary["model"]
D7 = "persistence"
print("Phase 33 notebook - final selection verification | selected =", CHOSEN)""")) 

cells.append(code(r"""RES = []
def check(cid, msg, ok):
    RES.append((cid, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid}  {msg}")""")) 

cells.append(code(r"""# N33-01  artefacts present with the expected shapes
need = [REPORTS / f for f in ("phase33_scorecard.csv", "phase33_selection.json",
        "phase33_test_metrics.csv", "phase33_test_read_log.json", "phase33_summary.json")]
need += [MODELS / "final" / "selected_model.pkl", MODELS / "final" / "final_config.json",
         MODELS / "final" / "README.md", FIG]
for p in need:
    assert p.exists(), p
check("N33-01", f"scorecard {sc.shape}, test_metrics {tm.shape}, "
                f"{len(need)} artefacts present incl. figure {FIG.stat().st_size:,} B",
      sc.shape[0] == 24 and tm.shape[0] > 0)""")) 

cells.append(code(r"""# N33-02  the experimental table is immutable
check("N33-02", f"experimental_dataset.csv SHA-256 = {sha256(TABLE)[:16]}... "
                f"unchanged = {sha256(TABLE) == TBL_SHA}",
      sha256(TABLE) == TBL_SHA and cfg["dataset_sha256"] == TBL_SHA)""")) 

cells.append(code(r"""# N33-03  the frozen test was read exactly once; D7 reproduces the Phase-16 read
d7_delta = log["reads"][-1]["d7_phase16_crosscheck_max_abs_delta_pr_auc"]
p16 = pd.read_csv(REPORTS / "phase16_test_metrics.csv")
p16 = p16[p16.model == D7]
j = tm[tm.model == D7].merge(p16[["cell", "horizon", "season", "pr_auc"]],
                             on=["cell", "horizon", "season"], suffixes=("", "_p16"))
repro = float(np.nanmax(np.abs(j.pr_auc - j.pr_auc_p16)))
check("N33-03", f"n_reads = {log['n_reads']}; distinct test rows = "
                f"{log['reads'][-1]['distinct_test_rows']:,}; D7 vs Phase-16 max|d| = "
                f"{d7_delta:.2e} (recomputed {repro:.2e}, {len(j)} shared rows)",
      log["n_reads"] == 1 and log["reads"][-1]["distinct_test_rows"] == 101174
      and d7_delta < 1e-6 and repro < 1e-6)""")) 

cells.append(code(r"""# N33-04  scorecard: 24 models, exactly 4 eligible (all L2), ranks 1..4
elig = sc[sc.eligible].sort_values("rank")
check("N33-04", f"{len(sc)} models scored; {len(elig)} eligible -> "
                f"{list(elig.model)} (all L2 = {set(elig.level) == {'L2'}}); "
                f"ranks {sorted(elig['rank'].tolist())}",
      len(sc) == 24 and len(elig) == 4 and set(elig.level) == {"L2"}
      and sorted(elig["rank"].tolist()) == [1, 2, 3, 4])""")) 

cells.append(code(r"""# N33-05  the selection is xgboost (L2-03) with the pre-registered rule recorded
row = sc[sc.model == CHOSEN].iloc[0]
check("N33-05", f"selected {CHOSEN} / {row.registry_id} / config {cfg['config_id']}; "
                f"rule = '{sel['rule'][:48]}...'",
      CHOSEN == "xgboost" and row.registry_id == "L2-03"
      and "gate1" in sel["rule"] and len(sel["eligible"]) == 4)""")) 

cells.append(code(r"""# N33-06  primary + co-primary both favour the selected model among the eligible set
prim = elig.p1_pr_auc_mean.max()
miss = elig.missed_warn_total.min()
check("N33-06", f"primary: max horizon-mean P1 PR-AUC = {prim:.5f} ({CHOSEN} = "
                f"{row.p1_pr_auc_mean:.5f}); co-primary: fewest missed floods = "
                f"{int(miss)} ({CHOSEN} = {int(row.missed_warn_total)})",
      abs(row.p1_pr_auc_mean - prim) < 1e-9 and int(row.missed_warn_total) == int(miss))""")) 

cells.append(code(r"""# N33-07  selected beats D7 on 20/21 frozen-test cell x horizon points
a = tm[tm.season == "all"]
win = loss = 0
loss_detail = []
for h in HOR:
    for c in YEARS + ["test_trained", "test_new_network", "test_unseen"]:
        xs = a[(a.model == CHOSEN) & (a.horizon == h) & (a.cell == c)].iloc[0]
        ds = a[(a.model == D7) & (a.horizon == h) & (a.cell == c)].iloc[0]
        if xs.pr_auc >= ds.pr_auc:
            win += 1
        else:
            loss += 1
            loss_detail.append((c, h, xs.pr_auc - ds.pr_auc, xs.pr_auc_ci_lo,
                                xs.pr_auc_ci_hi, ds.pr_auc_ci_lo, ds.pr_auc_ci_hi))
check("N33-07", f"{CHOSEN} >= D7 in {win}/21 cell x horizon points; "
                f"{loss} not ({loss_detail[0][0]}/{loss_detail[0][1]} "
                f"d={loss_detail[0][2]:+.4f}, CIs overlap)",
      win == 20 and loss == 1 and loss_detail[0][0] == "test_unseen"
      and loss_detail[0][1] == "label_24h"
      and loss_detail[0][3] <= loss_detail[0][6] and loss_detail[0][5] <= loss_detail[0][4])""")) 

cells.append(code(r"""# N33-08  test metrics finite; all buckets/seasons/horizons present; recall gain
cells_ok = set(a["cell"]) == set(YEARS + ["test_trained", "test_new_network", "test_unseen"])
seas_ok = set(tm.season) == {"all", "monsoon", "non-monsoon"}
fin = bool(a.pr_auc.notna().all() and a.roc_auc.notna().all())
rec_gain = float(a[a.model == CHOSEN].recall.mean() - a[a.model == D7].recall.mean())
miss_sel = int(a[a.model == CHOSEN].missed_flood_rows.sum())
miss_d7 = int(a[a.model == D7].missed_flood_rows.sum())
check("N33-08", f"cells = {cells_ok}; seasons = {seas_ok}; finite = {fin}; "
                f"mean recall +{rec_gain:.3f}; missed floods {miss_sel} vs D7 {miss_d7}",
      cells_ok and seas_ok and fin and rec_gain > 0.10 and miss_sel < miss_d7)""")) 

cells.append(code(r"""# N33-09  the frozen model reloads and reproduces the P1 val_2022 PR-AUC (independence)
import joblib
frozen = joblib.load(MODELS / "final" / "selected_model.pkl")
d, fsets = B.load()
split = d.split.to_numpy(); hold = d.station_holdout.to_numpy()
yr = d.date.dt.year.to_numpy()
evi = np.flatnonzero((split == "val") & ~hold & (yr == 2022))
repro = {}
for h, dd in zip(HOR, DISP):
    y = d[h].to_numpy()
    e = evi[~np.isnan(y[evi])]
    p = frozen["horizons"][h].predict_proba(d.iloc[e][frozen["features"]])[:, 1]
    repro[dd] = round(float(B.metrics(y[e].astype(int), p, 0.5)["pr_auc"]), 5)
ok = all(abs(repro[dd] - float(sc[sc.model == CHOSEN][f"p1_pr_auc_{dd}"].iloc[0])) < 1e-4
         for dd in DISP)
check("N33-09", f"reloaded frozen pipelines reproduce P1 val_2022 PR-AUC {repro} "
                f"(scorecard {[float(sc[sc.model == CHOSEN][f'p1_pr_auc_{d}'].iloc[0]) for d in DISP]})",
      ok and frozen["model"] == CHOSEN and len(frozen["features"]) == 62)""")) 

cells.append(code(r"""# N33-10  frozen artefacts carry seed, config_id, feature set, dataset SHA
n_thr = len(cfg["thresholds"])
check("N33-10", f"seed {cfg['seed']}, config {cfg['config_id']}, "
                f"feature_set {cfg['feature_set']} ({len(cfg['features'])}), "
                f"fit_rows@24h {cfg['fit_rows']['label_24h']:,}, "
                f"thresholds {n_thr} (2 models x 3 horizons x 3 seasons)",
      cfg["seed"] == 20261008 and cfg["feature_set"] == "FULL"
      and len(cfg["features"]) == 62 and cfg["fit_rows"]["label_24h"] == 61977
      and n_thr == 18)""")) 

cells.append(code(r"""# N33-11  the four eligible models are statistically inseparable (CIs overlap)
insep = {}
for dd in DISP:
    lo = elig[f"p1_ci_lo_{dd}"].max()
    hi = elig[f"p1_ci_hi_{dd}"].min()
    insep[dd] = bool(lo <= hi)
check("N33-11", f"common CI overlap among the 4 eligible at each horizon: {insep} "
                f"(differences are within sampling noise, so tie-breakers decide)",
      all(insep.values()))""")) 

cells.append(code(r"""# N33-12  thresholds frozen (Phase-29) before the test read; monsoon present
p29 = pd.read_csv(REPORTS / "phase29_chosen_thresholds.csv")
thr_ok = True
for h, dd in zip(HOR, DISP):
    q = p29[(p29.model == CHOSEN) & (p29.horizon == dd) & (p29.period == "all")]
    thr_ok = thr_ok and abs(float(q.t_warn.iloc[0]) -
                            float(cfg["thresholds"][f"{CHOSEN}|{h}|all"])) < 1e-9
mon = tm[(tm.season == "monsoon") & (tm.model == CHOSEN)]
check("N33-12", f"frozen thresholds match Phase-29 = {thr_ok}; monsoon rows = {len(mon)}; "
                f"figure {FIG.stat().st_size:,} B",
      thr_ok and len(mon) == 21 and FIG.stat().st_size > 5000)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 33 / STAGE 57  final model selection verified")
print("=" * 78)
for cid, ok in RES:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
npass = sum(int(ok) for _, ok in RES)
print(f"\n{npass}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
print(f"selected = {CHOSEN} ({summary['registry_id']}); test read {log['n_reads']}x; "
      f"seed {summary['seed']}")
print("test: xgboost >= D7 on 20/21 cell x horizon (only test_unseen/24h is a CI-overlap tie)")
assert npass == len(RES)
print("ALL PHASE-33 CHECKS PASSED")""")) 

cells.append(md(f"""## Headline (see `reports/phase33_selection.md`)

* **Selection (dev evidence only):** four models clear both gates (beat D7 on all
  7 development cells x 3 horizons **and** are calibrated) — `xgboost`, `xgboost_selected`,
  `lightgbm`, `random_forest`, all L2 gradient boosting / forests.  Ranked by the
  pre-registered rule (primary = horizon-mean P1 `val_2022` PR-AUC, then fewest missed
  floods), the winner is **`xgboost` (L2-03, FULL 62 features)**.
* **Statistically separated?  No.**  The four eligible models share a common
  95 % station-bootstrap CI at every horizon, so the choice is settled by the
  co-primary (missed floods: **87** for xgboost vs 88 / 96 / 112) and the tie-breakers
  (interpretability, cost) — never by a noise-level PR-AUC difference.
* **Frozen test (read once):** xgboost beats the D7 persistence bar on **20/21**
  cell x horizon points (year-wise PR-AUC 24 h 0.75-0.90 vs 0.63-0.80), cuts missed
  floods from **2,478 to 1,044**, and lifts mean recall **0.77 -> 0.89** (event recall
  0.90 -> 0.97).  The single non-win is `test_unseen`/24 h (-0.002, CIs overlap) — the
  honest limit on completely unseen stations at the shortest lead time.
* **Frozen artefact:** `models/final/selected_model.pkl` (config `xgboost_c01`, seed
  `20261008`, dataset SHA `a67485...`), plus `final_config.json` and a load recipe.""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
