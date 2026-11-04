"""Build notebooks/mi_v2.ipynb (Model Improvement Programme v2).

The MI-v2 programme (`reports/improvement_plan.md`) is a follow-on to the 1-33 workflow; it
has no phase number, so this notebook is named after the programme rather than the sequence.
It follows the same convention as `scripts/build_phaseNN_notebook.py`: markdown intro, a
`check()` helper, one code cell per numbered assertion (NM-xx) that re-derives the headline
facts independently from the produced artefacts and the immutable table, then a pass/fail
tally.  Execute it with the `flood-predictor` kernel.

    NM-01  every MI artefact exists with the expected shape
    NM-02  experimental-table SHA-256 unchanged (immutable); freeze records both SHAs
    NM-03  the v2 block merges 1:1 (200,665 rows, 48 new cols)
    NM-04  frozen feature sets: FULL 62 -> FULL_V2 110, block counts 19/18/6/5
    NM-05  MI-3a  v2 lifts val_2022 (0.95463 -> 0.95589)
    NM-06  MI-3b  HPO 41 configs/horizon; winners lgbm7 / lgbm7 / xgb3
    NM-07  MI-3c  the ensemble is rejected (best single beats ENS at every horizon)
    NM-08  MI-4  label audit bounds the ceiling (nested labels exact, no unanimous miss)
    NM-09  MI-5  pre-registered rule D16 selected + froze hpo_perhorizon_FULL_V2
    NM-10  the frozen model reloads and reproduces P1 val_2022 PR-AUC (independence)
    NM-11  MI-6  the frozen test was read once; D7 reproduces the Phase-16 read
    NM-12  MI-6  headline: D16 vs Phase-33 vs D7 (0.77076 / 0.75601 / 0.66755)
    NM-13  MI-3a(ERA5)  the ERA5 block is a wash on the primary cell -> not adopted
    NM-14  no test leakage before MI-6; the one read is MI-6
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "mi_v2.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# MI-v2 — Model Improvement Programme (verification)

| | |
| --- | --- |
| **Programme** | Model Improvement Programme v2 — `reports/improvement_plan.md` |
| **Stages** | MI-1 data → MI-2 features → MI-3 compare/HPO/ensemble → MI-4 label audit → MI-5 re-select & freeze → MI-6 single test read → MI-3a(ERA5) probe |
| **Governing rules** | test (`>= 2023`) untouched until MI-6 · per-cell only, never pooled (D8) · PR-AUC primary, accuracy banned as headline · Phase-13 leakage rule |
| **Frozen model** | `models/final_v2/` — decision **D16**, `hpo_perhorizon_FULL_V2`, `FULL_V2` (110 features) |
| **Test** | read **once**, in MI-6 (`n_reads = 1`), for D16 and the D7 bar |

This notebook does not re-run the programme; it **re-derives the headline facts
independently** from the produced artefacts and the immutable `experimental_dataset.csv`,
asserting each one (NM-01 … NM-14).  Every number below is recomputed here, not copied.""")) 

cells.append(code(r"""import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Flood predictor\scripts")
import baseline_model as B

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
TABLE = PROC / "experimental_dataset.csv"
V1_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
V2_SHA = "16287675df99eb5afb521c5c99252efd119c480fd1c7166435519fd43ff62a3c"
HOR = ["label_24h", "label_48h", "label_72h"]
DISP = ["24h", "48h", "72h"]
CELLS = ["test_2023", "test_2024", "test_2025", "test_2026",
         "test_trained", "test_new_network", "test_unseen"]
DEV_CELLS = ["val_2020", "val_2021", "val_2022",
             "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
CHOSEN = "hpo_perhorizon_FULL_V2"
D7 = "persistence"
t0 = time.time()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---- artefacts ------------------------------------------------------------
fs2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))
fse = json.loads((PROC / "feature_sets_v2_era5.json").read_text(encoding="utf-8"))
sum3 = json.loads((REPORTS / "mi3_featureset_summary.json").read_text(encoding="utf-8"))
era5 = json.loads((REPORTS / "mi3_era5_probe.json").read_text(encoding="utf-8"))
lab = json.loads((REPORTS / "mi4_label_audit.json").read_text(encoding="utf-8"))
sel = json.loads((REPORTS / "mi5_selection.json").read_text(encoding="utf-8"))
cfg = json.loads((MODELS / "final_v2" / "final_config_v2.json").read_text(encoding="utf-8"))
mlog = json.loads((REPORTS / "mi6_test_read_log.json").read_text(encoding="utf-8"))
msum = json.loads((REPORTS / "mi6_summary.json").read_text(encoding="utf-8"))

hpo = pd.read_csv(REPORTS / "mi3_hpo.csv")
ens = pd.read_csv(REPORTS / "mi3_ensemble.csv")
sc = pd.read_csv(REPORTS / "mi5_scorecard.csv")
tm = pd.read_csv(REPORTS / "mi6_test_metrics.csv")
era5_csv = pd.read_csv(REPORTS / "mi3_era5_probe.csv")
p16 = pd.read_csv(REPORTS / "phase16_test_metrics.csv")
p33 = pd.read_csv(REPORTS / "phase33_test_metrics.csv")

# ---- the model table: immutable v1 + v2 block merged 1:1 ------------------
d, fsets = B.load()
v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
d2 = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
print("MI-v2 notebook - verification | selected =", CHOSEN)""")) 

cells.append(code(r"""RES = []
def check(cid, msg, ok):
    RES.append((cid, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid}  {msg}")""")) 

cells.append(code(r"""# NM-01  every MI artefact exists with the expected shape
need = [REPORTS / f for f in (
    "improvement_plan.md", "mi_status.md", "mi2_features.md",
    "mi3_featureset_summary.json", "mi3_hpo.csv", "mi3_ensemble.csv",
    "mi4_label_audit.json", "mi4_base_rates.csv", "mi5_scorecard.csv",
    "mi5_selection.json", "mi6_test_metrics.csv", "mi6_test_read_log.json",
    "mi6_summary.json", "mi6_thresholds.csv", "mi3_era5_probe.json")]
need += [PROC / "mi_v2_features.csv", PROC / "feature_sets_v2.json",
         PROC / "feature_sets_v2_era5.json"]
need += [MODELS / "final_v2" / "selected_model_v2.pkl",
         MODELS / "final_v2" / "final_config_v2.json",
         MODELS / "final_v2" / "README.md"]
missing = [str(p.name) for p in need if not p.exists()]
check("NM-01", f"{len(need)} MI artefacts present (missing {len(missing)}); "
               f"hpo {hpo.shape}, ensemble {ens.shape}, scorecard {sc.shape}, "
               f"test_metrics {tm.shape}",
      not missing and hpo.shape[0] == 123 and ens.shape[0] == 21
      and sc.shape[0] == 7 and tm.shape[0] == 126)""")) 

cells.append(code(r"""# NM-02  the experimental table is immutable; the freeze records both SHAs
tbl = sha256(TABLE)
check("NM-02", f"experimental_dataset.csv SHA-256 = {tbl[:16]}... unchanged = {tbl == V1_SHA}; "
               f"freeze records v1 = {cfg['dataset_sha256_v1'][:8]}..., "
               f"v2 = {cfg['dataset_sha256_v2'][:8]}...",
      tbl == V1_SHA and cfg["dataset_sha256_v1"] == V1_SHA
      and cfg["dataset_sha256_v2"] == V2_SHA)""")) 

cells.append(code(r"""# NM-03  the v2 block merges 1:1 onto the immutable table (200,665 rows, 48 new cols)
check("NM-03", f"v2 table {v2.shape} = station_id,date + {v2.shape[1] - 2} new cols; "
               f"merge 1:1 -> {d2.shape[0]:,} rows == table {len(d):,}; "
               f"v2_new_columns = {len(fs2['v2_new_columns'])}",
      d2.shape[0] == len(d) == 200665 and v2.shape[1] - 2 == 48
      and len(fs2["v2_new_columns"]) == 48)""")) 

cells.append(code(r"""# NM-04  frozen feature sets: FULL 62 -> FULL_V2 110, with the documented block counts
F, FV2 = fs2["sets"]["FULL"], fs2["sets"]["FULL_V2"]
new = [c for c in FV2 if c not in set(F)]
ps = sum(c.startswith("ps_") for c in new)
gf = sum(c.startswith("gf_") for c in new)
wl = sum(c.startswith(("wl_", "api_")) for c in new)
ix = sum(c.startswith("ix_") for c in new)
check("NM-04", f"FULL {len(F)} -> FULL_V2 {len(FV2)} (+{len(new)}); "
               f"ps_ {ps}, gf_ {gf}, wl_/api_ {wl}, ix_ {ix}",
      len(F) == 62 and len(FV2) == 110 and len(new) == 48
      and set(F).issubset(set(FV2)) and (ps, gf, wl, ix) == (19, 18, 6, 5))""")) 

cells.append(code(r"""# NM-05  MI-3a  the v2 block lifts val_2022 (reproduces the MI-3a summary)
vm = sum3["val_2022_horizon_mean"]
deltas = {h: round(sum3[h]["delta"], 5) for h in sum3 if h.startswith("label_")}
check("NM-05", f"val_2022 horizon-mean FULL {vm['FULL']:.5f} -> FULL_V2 {vm['FULL_V2']:.5f} "
               f"(+{vm['FULL_V2'] - vm['FULL']:.5f}); per-horizon deltas {deltas}",
      abs(vm["FULL"] - 0.95463) < 1e-5 and abs(vm["FULL_V2"] - 0.95589) < 1e-5
      and all(v > 0 for v in deltas.values()))""")) 

cells.append(code(r"""# NM-06  MI-3b  HPO = 41 configs/horizon; winners lgbm7 / lgbm7 / xgb3
best = {}
for h, g in hpo.groupby("horizon"):
    b = g.sort_values("val_2022", ascending=False).iloc[0]
    best[h] = (b.tag, b.family, float(b.val_2022))
mean_best = sum(v[2] for v in best.values()) / 3
check("NM-06", f"41 cfg/horizon = {hpo.groupby('horizon').size().to_dict()}; "
               f"winners {[best[h][0] for h in HOR]}; per-horizon mean {mean_best:.5f}",
      (hpo.groupby("horizon").size() == 41).all()
      and [best[h][0] for h in HOR] == ["lgbm7", "lgbm7", "xgb3"]
      and abs(mean_best - 0.95743) < 1e-5)""")) 

cells.append(code(r"""# NM-07  MI-3c  the ensemble is rejected (best single beats ENS_mean/ENS_rank at every horizon)
rej = {}
for h, g in ens.groupby("horizon"):
    single = g[~g.model.str.startswith("ENS")].val_2022.max()
    ens_best = g[g.model.str.startswith("ENS")].val_2022.max()
    rej[h] = bool(single > ens_best)
check("NM-07", f"best single > best ensemble at every horizon = {rej} "
               f"(so MI-5 used single models, not an ensemble)",
      all(rej.values()) and len(rej) == 3)""")) 

cells.append(code(r"""# NM-08  MI-4  the label audit bounds the ceiling (nested labels exact, no unanimous miss)
nest = lab["nesting_identity"]
cel = lab["ceiling"]["cells"]
no_unanimous_miss = all(cel[h]["unanimous_miss_pos"] == 0 for h in HOR)
check("NM-08", f"nesting violations "
               f"{nest['viol_24_implies_48']}/{nest['viol_48_implies_72']}/{nest['viol_24_implies_72']}; "
               f"ceiling panel {len(lab['ceiling']['panel'])} models; unanimous miss = {no_unanimous_miss}; "
               f"24h base rate {cel['label_24h']['base_rate']:.4f}",
      nest["viol_24_implies_48"] == nest["viol_48_implies_72"] == nest["viol_24_implies_72"] == 0
      and no_unanimous_miss and len(lab["ceiling"]["panel"]) == 12
      and abs(cel["label_24h"]["base_rate"] - 0.050908) < 1e-4)""")) 

cells.append(code(r"""# NM-09  MI-5  the pre-registered D16 rule selected hpo_perhorizon_FULL_V2 and froze it
fam = cfg["per_horizon_family"]
check("NM-09", f"decision {sel['decision']} -> {sel['selected']} (rank1 {sel['ranked'][0]['model']}); "
               f"eligible {len(sel['eligible'])}; families {fam}; {len(cfg['features'])} feats; "
               f"test_rows_read {cfg['test_rows_read']}",
      sel["decision"] == "D16" and sel["selected"] == CHOSEN
      and sel["ranked"][0]["model"] == CHOSEN and len(sel["eligible"]) == 5
      and cfg["model"] == CHOSEN and cfg["feature_set"] == "FULL_V2"
      and len(cfg["features"]) == 110
      and fam == {"label_24h": "lgbm", "label_48h": "lgbm", "label_72h": "xgb"}
      and cfg["test_rows_read"] == 0 and cfg["dataset_sha256_v2"] == V2_SHA)""")) 

cells.append(code(r"""# NM-10  the frozen model reloads and reproduces P1 val_2022 PR-AUC (independence)
import joblib
frozen = joblib.load(MODELS / "final_v2" / "selected_model_v2.pkl")
split = d2.split.to_numpy(); hold = d2.station_holdout.to_numpy()
evi = np.flatnonzero((split == "val") & ~hold)
assert not (split[evi] == "test").any(), "test leaked into the dev cell"
repro = {}
for h, dd in zip(HOR, DISP):
    y = d2[h].to_numpy()
    e = evi[~np.isnan(y[evi])]
    p = frozen["pipelines"][h].predict_proba(d2.iloc[e][frozen["features"]])[:, 1]
    repro[dd] = round(float(B.metrics(y[e].astype(int), p, 0.5)["pr_auc"]), 5)
score = {dd: float(sc[sc.model == CHOSEN][f"p1_pr_auc_{dd}"].iloc[0]) for dd in DISP}
ok = all(abs(repro[dd] - score[dd]) < 1e-4 for dd in DISP)
check("NM-10", f"reloaded frozen pipelines reproduce P1 val_2022 PR-AUC {repro} "
               f"(scorecard {[round(v, 5) for v in score.values()]})",
      ok and frozen["model"] == CHOSEN and len(frozen["features"]) == 110)""")) 

cells.append(code(r"""# NM-11  MI-6  the frozen test was read exactly once; D7 reproduces the Phase-16 read
rd = mlog["reads"][-1]
p16d7 = p16[p16.model == D7]
j = tm[tm.model == D7].merge(p16d7[["cell", "horizon", "season", "pr_auc"]],
                             on=["cell", "horizon", "season"], suffixes=("", "_p16"))
repro_d7 = float(np.nanmax(np.abs(j.pr_auc - j.pr_auc_p16)))
check("NM-11", f"n_reads = {mlog['n_reads']}; distinct test rows = {rd['distinct_test_rows']:,}; "
               f"scored rows = {rd['test_rows_read']:,}; D7 vs Phase-16 max|d| = {repro_d7:.2e} "
               f"({len(j)} shared rows)",
      mlog["n_reads"] == 1 and rd["distinct_test_rows"] == 101174
      and rd["test_rows_read"] == 1198056 and repro_d7 < 1e-6)""")) 

cells.append(code(r"""# NM-12  MI-6 headline: D16 beats D7 on 21/21 and Phase-33 on 6/7 cells
def cellmean(df, m, c):
    return float(df[(df.season == "all") & (df.model == m) & (df.cell == c)].pr_auc.mean())

d16 = {h: tm[(tm.season == "all") & (tm.model == CHOSEN) &
             (tm.horizon == h) & (tm.cell.isin(CELLS))].pr_auc.mean() for h in HOR}
d7m = {h: tm[(tm.season == "all") & (tm.model == D7) &
             (tm.horizon == h) & (tm.cell.isin(CELLS))].pr_auc.mean() for h in HOR}
p33m = {h: p33[(p33.season == "all") & (p33.model == "xgboost") &
               (p33.horizon == h) & (p33.cell.isin(CELLS))].pr_auc.mean() for h in HOR}
m16, m7, m33 = (round(sum(v.values()) / 3, 5) for v in (d16, d7m, p33m))
win_d7 = sum(float(tm[(tm.season == "all") & (tm.model == CHOSEN) & (tm.horizon == h)
                      & (tm.cell == c)].pr_auc.iloc[0])
             >= float(tm[(tm.season == "all") & (tm.model == D7) & (tm.horizon == h)
                         & (tm.cell == c)].pr_auc.iloc[0]) for h in HOR for c in CELLS)
loss = [c for c in CELLS if cellmean(tm, CHOSEN, c) < cellmean(p33, "xgboost", c)]
p33_win = len(CELLS) - len(loss)
check("NM-12", f"test horizon-mean: D16 {m16} vs Phase-33 {m33} vs D7 {m7}; "
               f"D16>=D7 {win_d7}/21; D16>=P33 cells {p33_win}/7 (loses {loss})",
      abs(m16 - 0.77076) < 1e-5 and abs(m33 - 0.75601) < 1e-5 and abs(m7 - 0.66755) < 1e-5
      and win_d7 == 21 and p33_win == 6 and loss == ["test_2025"])""")) 

cells.append(code(r"""# NM-13  MI-3a(ERA5)  the ERA5 block is a wash on the primary cell -> not adopted
evm = era5["val_2022_horizon_mean"]
ead = era5["all_dev_horizon_mean"]
check("NM-13", f"ERA5 +{era5['n_new_cols']} cols -> FULL_V2_era5 {len(fse['sets']['FULL_V2'])}; "
               f"val_2022 delta {evm['delta_era5_vs_pre']:+.5f}, all-dev delta "
               f"{ead['FULL_V2_era5'] - ead['FULL_V2_pre']:+.5f}; frozen FULL_V2 still "
               f"{len(fs2['sets']['FULL_V2'])}",
      era5["n_new_cols"] == 26 and len(fse["sets"]["FULL_V2"]) == 136
      and abs(evm["delta_era5_vs_pre"] + 0.00025) < 1e-4
      and ead["FULL_V2_era5"] > ead["FULL_V2_pre"]
      and len(fs2["sets"]["FULL_V2"]) == 110)""")) 

cells.append(code(r"""# NM-14  no test leakage before MI-6; the one read is MI-6
probe_cells = set(era5_csv["cell"].unique())
check("NM-14", f"MI-5 freeze test_rows_read = {cfg['test_rows_read']}; ERA5 probe cells = "
               f"{sorted(probe_cells)} (dev only = {probe_cells <= set(DEV_CELLS)}); "
               f"total test reads = {mlog['n_reads']}",
      cfg["test_rows_read"] == 0 and probe_cells <= set(DEV_CELLS)
      and mlog["n_reads"] == 1)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("MI-v2  improvement programme verified")
print("=" * 78)
for cid, ok in RES:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
npass = sum(int(ok) for _, ok in RES)
print(f"\n{npass}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
print(f"selected = {CHOSEN} (decision {sel['decision']}, seed {cfg['seed']}); "
      f"test read {mlog['n_reads']}x")
print("test: D16 0.77076 vs Phase-33 0.75601 vs D7 0.66755 (horizon-mean over 7 cells)")
assert npass == len(RES)
print("ALL MI-v2 CHECKS PASSED")""")) 

cells.append(md("""## Headline (see `reports/mi_status.md` and `reports/improvement_plan.md`)

* **MI-1/MI-2 — data & features.** ERA5-Land (133/133), GloFAS (11 anchors) and POWER soil
  (133 stations) land cleanly; the v2 block adds 48 columns and lifts `FULL` 62 → `FULL_V2` 110.
* **MI-3a — the gain is real but thin.** val_2022 horizon-mean **0.95463 → 0.95589**
  (+0.00126), growing with lead time.  **MI-3b** HPO (41 configs/horizon) picks
  `lgbm7`/`lgbm7`/`xgb3` → **0.95743**.  **MI-3c**: the ensemble is **rejected** — it loses to
  the best single at every horizon.
* **MI-4 — why not 0.99.** The labels are exactly nested, unambiguous, and unanimous across a
  12-model panel; the residual near-threshold noise (~4-10 %) plus the low base rate cap a
  uniform 0.99.  The honest ceiling is ~0.96-0.97 on `val_2022`.
* **MI-5 — frozen (D16).** The pre-registered rule selects **`hpo_perhorizon_FULL_V2`**
  (`FULL_V2`, 110 features; lgbm/lgbm/xgb per horizon) and freezes it to `models/final_v2/`.
* **MI-6 — the single test read.** D16 test horizon-mean **0.77076** beats Phase-33
  (`xgboost`) **0.75601** and D7 **0.66755**; D16 ≥ D7 on **21/21** points, ≥ Phase-33 on
  **6/7** cells (loses `test_2025` by 0.0037, CIs overlap).  The direction is consistent but
  the improvement is **not** bootstrap-separable — an honest, thin gain.
* **MI-3a(ERA5).** The completed ERA5 block adds 26 columns (`FULL_V2` 110 → 136) but is a
  **wash on the primary cell** (−0.00025) with only a marginal +0.00075 across the dev cells;
  it was **not adopted**, so **D16 stands** and no second test read is due.""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
