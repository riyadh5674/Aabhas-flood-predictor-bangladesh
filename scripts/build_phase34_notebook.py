"""Build notebooks/34_explainable_ai.ipynb (workflow Phase 34 / stage 58).

Phase 34 explains the frozen D16 model (`models/final_v2/selected_model_v2.pkl`): global
TreeSHAP, a permutation cross-check, three local exemplars, and integrated-gradient
attribution for the temporal Transformer (L4-03) and LSTM (L3-01).  This notebook
**re-derives the headline facts independently** from the produced artefacts *and* by
recomputing SHAP on the frozen pipelines:

    N34-01  artefacts exist with the expected shapes
    N34-02  experimental-table SHA unchanged; frozen D16 carries seed/feature-set/110 cols
    N34-03  global SHAP: 3 x 110 rows; top driver per horizon is the G4 hydrology leader
    N34-04  TreeSHAP is additive and the 24 h ranking reproduces from an independent recompute
    N34-05  group shares sum to 1; hydrology dominates at 24 h
    N34-06  D16 vs Phase-33 SHAP agree on the shared 62 features (rho > 0.5, all horizons)
    N34-07  permutation-vs-SHAP rho is low at 24 h (documented collinearity failure)
    N34-08  three local exemplars (TP/FN/FP) with the expected labels and probabilities
    N34-09  DL attribution: 2 models x 3 horizons x 12 variables; hydrology leads at 24 h
    N34-10  summary top-15 reproduces the SHAP table; wall-clock/seed recorded
    N34-11  no banned/label-window column appears in any top-20 (no leakage)
    N34-12  figures + run log exist; seed and test discipline hold

Execute with the `flood-predictor` kernel:

    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\34_explainable_ai.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "34_explainable_ai.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 34 — Explainable AI (stage 58)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 34 — Explainable AI** → **stage 58** |
| **Built by** | `scripts/phase34_xai.py` |
| **Model explained** | **D16** `hpo_perhorizon_FULL_V2` (`models/final_v2/`), per-horizon lightgbm/lgbm/xgb, FULL_V2 110 features |
| **Reference** | Phase-33 `xgboost` (FULL 62) — cross-model explanation agreement |
| **Cell** | `val_2022` (holdout excluded) — **no new test read** (`test_rows_read = 0`) |

Global **TreeSHAP** + a **permutation** cross-check, **local** per-prediction explanations,
and **integrated gradients** for the temporal Transformer (L4-03) and LSTM (L3-01),
aggregated to the same phase-0 feature groups. This notebook re-derives every headline
fact from the artefacts and from the frozen pipelines.""")) 

cells.append(code(r"""import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Flood predictor\scripts")
import baseline_model as B
import joblib
import shap

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
TABLE = ROOT / "data" / "processed" / "experimental_dataset.csv"
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
HOR = ["label_24h", "label_48h", "label_72h"]
DISP = ["24h", "48h", "72h"]
TEXT = list(B.TEXT_COLS)
BANNED = {"station_id", "station_name", "name", "date", "source", "label_24h", "label_48h",
          "label_72h", "obs_days_24h", "obs_days_48h", "obs_days_72h", "split", "spatial_fold",
          "cohort", "first_year", "station_holdout", "season", "in_embargo_window"}
t0 = time.time()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def spearman(a, b):
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


summary = json.loads((REPORTS / "phase34_summary.json").read_text(encoding="utf-8"))
FS = json.loads((ROOT / "data" / "processed" / "feature_sets_v2.json")
                .read_text(encoding="utf-8"))["sets"]
FULL62, FULLV2 = FS["FULL"], FS["FULL_V2"]
gs = pd.read_csv(REPORTS / "phase34_global_shap.csv")
grp = pd.read_csv(REPORTS / "phase34_group_shap.csv")
perm = pd.read_csv(REPORTS / "phase34_permutation_importance.csv")
loc = pd.read_csv(REPORTS / "phase34_local_explanations.csv")
dl = pd.read_csv(REPORTS / "phase34_dl_attribution.csv")
cm = pd.read_csv(REPORTS / "phase34_crossmodel.csv")
print("Phase 34 notebook - explainable AI verification | model =", summary["model"])""")) 

cells.append(code(r"""RES = []
def check(cid, msg, ok):
    RES.append((cid, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid}  {msg}")""")) 

cells.append(code(r"""# N34-01  artefacts present with the expected shapes
need = [REPORTS / f for f in ("phase34_xai.md", "phase34_global_shap.csv",
        "phase34_group_shap.csv", "phase34_permutation_importance.csv",
        "phase34_local_explanations.csv", "phase34_dl_attribution.csv",
        "phase34_crossmodel.csv", "phase34_summary.json")]
need += [FIGURES / "phase34_shap_summary.png", FIGURES / "phase34_group_attribution.png",
         FIGURES / "phase34_local_examples.png", MODELS / "final_v2" / "selected_model_v2.pkl"]
missing = [str(p) for p in need if not p.exists()]
check("N34-01", f"global {gs.shape}, group {grp.shape}, dl {dl.shape}, local {loc.shape}, "
                f"crossmodel {cm.shape}; {len(need)} artefacts",
      not missing and gs.shape[0] == 330 and dl.shape[0] == 72 and loc.shape[0] == 24
      and cm.shape[0] == 186 and grp.shape[0] == 27)""")) 

cells.append(code(r"""# N34-02  experimental table immutable; frozen D16 metadata
fr = joblib.load(MODELS / "final_v2" / "selected_model_v2.pkl")
sha_ok = sha256(TABLE) == TBL_SHA
fam = fr["per_horizon_family"]
check("N34-02", f"table SHA unchanged = {sha_ok}; model {fr['model']} seed {fr['seed']}, "
                f"feature_set {fr['feature_set']} ({len(fr['features'])} cols), "
                f"families {fam}, test_rows_read = {fr['test_rows_read']}",
      sha_ok and fr["dataset_sha256_v1"] == TBL_SHA and fr["model"] == "hpo_perhorizon_FULL_V2"
      and fr["seed"] == 20261008 and fr["feature_set"] == "FULL_V2"
      and len(fr["features"]) == 110 and fr["test_rows_read"] == 0)""")) 

cells.append(code(r"""# N34-03  global SHAP: top driver per horizon is the G4 hydrology leader
tops = {}
for dd in DISP:
    r = gs[gs.horizon == dd].nsmallest(1, "rank").iloc[0]
    tops[dd] = (r.feature, r.group)
same_as_summary = all(tops[dd][0] == summary["top_feature"][dd] for dd in DISP)
all_hydro = all(tops[dd][1] == "G4_hydrology" for dd in DISP)
check("N34-03", f"24h {tops['24h']}, 48h {tops['48h']}, 72h {tops['72h']}; "
                f"matches summary = {same_as_summary}",
      same_as_summary and all_hydro and set(gs.horizon) == set(DISP)
      and gs.groupby("horizon").size().eq(110).all())""")) 

cells.append(code(r"""# N34-04  TreeSHAP is additive; the 24 h ranking reproduces from an independent recompute
d, _ = B.load()
v2 = pd.read_csv(ROOT / "data" / "processed" / "mi_v2_features.csv", parse_dates=["date"])
d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
mask = (d.split.to_numpy() == "val") & ~d.station_holdout.to_numpy()
y = d["label_24h"].to_numpy()
idx = np.flatnonzero(mask & ~pd.isna(y))
rng = np.random.default_rng(20261008)
pick = np.sort(rng.choice(idx, size=300, replace=False))
pipe = fr["pipelines"]["label_24h"]
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
Xt = pre.transform(d.iloc[pick][fr["features"]])
ex = shap.TreeExplainer(clf)
sv = np.asarray(ex.shap_values(Xt))
if sv.ndim == 3:
    sv = sv[:, :, 1]
recon = sv.sum(1) + float(np.asarray(ex.expected_value).ravel()[0])
add = float(np.max(np.abs(recon - clf.predict(Xt, raw_score=True))))
names = [c for c in fr["features"] if c not in TEXT] + [f"freq_{c}" for c in fr["features"] if c in TEXT]
sub = pd.Series(np.abs(sv).mean(0), index=names)
a = gs[gs.horizon == "24h"].set_index("feature")["mean_abs_shap"]
j = pd.concat([a, sub], axis=1).dropna()
rho = spearman(j.iloc[:, 0], j.iloc[:, 1])
check("N34-04", f"max|sum(SHAP)+base - margin| = {add:.1e} (additive); independent 300-row "
                f"24 h ranking rho vs table = {rho:.3f}",
      add < 1e-6 and rho > 0.9)""")) 

cells.append(code(r"""# N34-05  group shares sum to 1; hydrology dominates at 24 h
sums = {dd: float(grp[grp.horizon == dd]["share"].sum()) for dd in DISP}
g4 = float(grp[(grp.horizon == "24h") & (grp.group == "G4_hydrology")]["share"].iloc[0])
rank0 = grp[grp.horizon == "24h"].sort_values("share", ascending=False).iloc[0]["group"]
check("N34-05", f"share sums {({k: round(v, 6) for k, v in sums.items()})}; 24 h leader = "
                f"{rank0} ({100 * g4:.1f} %)",
      all(abs(v - 1.0) < 1e-6 for v in sums.values()) and rank0 == "G4_hydrology" and g4 > 0.5)""")) 

cells.append(code(r"""# N34-06  D16 vs Phase-33 SHAP agree on the shared 62 features
cm_rho = {}
for dd in DISP:
    a = gs[(gs.horizon == dd) & gs.feature.isin(cm.feature)].set_index("feature")["mean_abs_shap"]
    b = cm[cm.horizon == dd].set_index("feature")["p33_mean_abs_shap"]
    jj = pd.concat([a, b], axis=1).dropna()
    cm_rho[dd] = round(spearman(jj.iloc[:, 0], jj.iloc[:, 1]), 3)
match = cm_rho == summary["crossmodel_spearman_shared62"]
check("N34-06", f"shared-62 Spearman {cm_rho}; matches summary = {match}; "
                f"shared features = {cm.feature.nunique()}",
      match and all(v > 0.5 for v in cm_rho.values()) and cm.feature.nunique() == 62)""")) 

cells.append(code(r"""# N34-07  permutation-vs-SHAP rho is low at 24 h (documented collinearity failure)
perm_rho = {}
for dd in DISP:
    a = gs[gs.horizon == dd].set_index("feature")["mean_abs_shap"]
    b = perm[perm.horizon == dd].set_index("feature")["importance_mean"]
    jj = pd.concat([a, b], axis=1).dropna()
    perm_rho[dd] = round(spearman(jj.iloc[:, 0], jj.iloc[:, 1]), 3)
match = perm_rho == summary["permutation_spearman"]
absorbed = float(perm[perm.horizon == "24h"].importance_mean.max())
check("N34-07", f"permutation-vs-SHAP {perm_rho}; single-feature absorbed mass = {absorbed:.4f}; "
                f"matches summary = {match}",
      match and perm_rho["24h"] < 0.5 and absorbed > 0.5)""")) 

cells.append(code(r"""# N34-08  three local exemplars (TP/FN/FP) with expected labels and probabilities
cases = loc.groupby("case").agg(n=("feature", "size"), lab=("label", "first"),
                                p=("predicted_probability", "first")).to_dict("index")
tp = cases.get("high-confidence true positive", {})
fn = cases.get("missed flood (false negative)", {})
fp = cases.get("false alarm (false positive)", {})
check("N34-08", f"cases = { {k: (v['n'], v['lab'], round(v['p'], 2)) for k, v in cases.items()} }",
      len(cases) == 3 and all(v["n"] == 8 for v in cases.values())
      and tp.get("lab") == 1 and tp.get("p", 0) >= 0.5
      and fn.get("lab") == 1 and fn.get("p", 1) < 0.5
      and fp.get("lab") == 0 and fp.get("p", 0) >= 0.5)""")) 

cells.append(code(r"""# N34-09  DL attribution: 2 models x 3 horizons x 12 vars; hydrology leads at 24 h
per = dl.groupby(["model", "horizon"]).agg(n=("variable", "nunique"),
                                           s=("share", "sum")).reset_index()
topvar = {}
g4share = {}
for m in dl.model.unique():
    s = dl[(dl.model == m) & (dl.horizon == "24h")].sort_values("share", ascending=False)
    topvar[m] = s.iloc[0]["variable"]
    g4share[m] = round(float(s[s.group == "G4_hydrology"]["share"].sum()), 3)
check("N34-09", f"{len(dl)} rows; per-cell counts ok = {bool((per.n == 12).all())}; "
                f"share sums ok = {bool((per.s.sub(1).abs() < 1e-6).all())}; "
                f"24 h top var {topvar}; G4 share {g4share}",
      len(dl) == 72 and (per.n == 12).all() and (per.s.sub(1).abs() < 1e-6).all()
      and all(v == "wl_pct_danger" for v in topvar.values())
      and all(v > 0.5 for v in g4share.values()))""")) 

cells.append(code(r"""# N34-10  summary top-15 reproduces the SHAP table; wall-clock/seed recorded
ok = True
for dd in DISP:
    a = gs[gs.horizon == dd].nsmallest(15, "rank").set_index("feature")["mean_abs_shap"]
    b = pd.Series({r["feature"]: r["mean_abs_shap"] for r in summary["top15"][dd]})
    ok = ok and bool((a - b).abs().max() < 1e-6)
check("N34-10", f"top-15 match table (all horizons) = {ok}; wall = {summary['wall_s']} s; "
                f"rows explained {summary['n_rows_explained']}; seed {summary['seed']}",
      ok and summary["wall_s"] and summary["wall_s"] > 0 and summary["seed"] == 20261008
      and summary["test_rows_read"] == 0)""")) 

cells.append(code(r"""# N34-11  no banned/label-window column appears in any top-20 (no leakage)
leaks = {}
for dd in DISP:
    top20 = set(gs[gs.horizon == dd].nsmallest(20, "rank").feature)
    leaks[dd] = sorted(top20 & BANNED)
feats = fr["features"]
v2cols = [c for c in FULLV2 if c not in set(FULL62)]
check("N34-11", f"banned in top-20 = {leaks}; features {len(feats)} unique "
                f"{len(set(feats))}; FULL {len(FULL62)} + new {len(v2cols)} = FULL_V2 "
                f"{len(FULLV2)}",
      not any(leaks.values()) and len(feats) == len(set(feats)) == 110
      and len(FULL62) == 62 and len(FULLV2) == 110 and len(v2cols) == 48
      and set(v2cols) <= set(feats) and set(FULL62) <= set(feats)
      and not (set(feats) & BANNED))""")) 

cells.append(code(r"""# N34-12  figures + run log exist; seed and test discipline hold
figs = {p.name: p.stat().st_size for p in FIGURES.glob("phase34_*.png")}
log = ROOT / "scripts" / "phase34_run.log"
check("N34-12", f"figures {figs}; run log {log.stat().st_size:,} B; seed {summary['seed']}",
      len(figs) == 3 and all(v > 5000 for v in figs.values()) and log.exists()
      and log.stat().st_size > 0 and summary["cell"] == "val_2022"
      and summary["test_rows_read"] == 0)""")) 

cells.append(code(r"""print("\n" + "=" * 78)
print("PHASE 34 / STAGE 58  explainable AI verified")
print("=" * 78)
for cid, ok in RES:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
npass = sum(int(ok) for _, ok in RES)
print(f"\n{npass}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
print(f"model = {summary['model']} (D16); cell = {summary['cell']}; test_rows_read = "
      f"{summary['test_rows_read']}")
print("global: hydrology-first (G4 ~64% @24h); local: TP/FN/FP explained; "
      "DL: IG on Transformer+LSTM -> G4 leads")
assert npass == len(RES)
print("ALL PHASE-34 CHECKS PASSED")""")) 

cells.append(md("""## Headline (see `reports/phase34_xai.md`)

* **Global.** The frozen **D16** model is **hydrology-first**: `G4_hydrology` carries
  **~64 %** of the 24 h TreeSHAP mass, led by `wl_dist_to_danger` / `wl_pct_danger`; locality,
  upstream-neighbour stage, raw rainfall and the v2 soil/discharge/memory groups follow. No
  banned column leaks into any top-20, so no Phase-13 re-check is triggered.
* **Cross-model.** D16 reproduces the Phase-33 `xgboost` ordering on the shared 62 features
  (ρ ≈ 0.76–0.88) — the v1→v2 feature change refined the explanation, it did not reshape it.
* **Permutation cross-check is instructive, not confirming.** Under the collinear
  water-level ladder a *single* feature (`wl_dist_to_danger`) absorbs the whole permutation
  importance, so its ranking barely correlates with SHAP (ρ ≈ 0.02 at 24 h) — a textbook
  failure mode reported honestly.
* **Local.** Three exemplars (high-confidence TP, missed flood, false alarm) are explained in
  the proposal's one-row format; the miss is a below-danger river (Phase-30 pattern A) and the
  false alarm is a high-stage non-event at the FN ≫ FP operating point.
* **Deep models.** Integrated gradients on the temporal **Transformer (L4-03)** and **LSTM
  (L3-01)** put ~70–75 % of attribution on `G4_hydrology` (`wl_pct_danger` leading) with a
  weather/rain remainder — the **same** dominant layer the tabular SHAP finds, using the same
  feature groups.
* **Discipline.** Every explanation uses `val_2022`; **no new test read** (`test_rows_read = 0`).""")) 

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
