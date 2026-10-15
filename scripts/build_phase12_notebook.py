"""Build notebooks/04_feature_selection.ipynb (workflow Phase 12, body steps 18-21).

The notebook is the analysis artifact; this script only assembles it, so the code has a
single source of truth and can be regenerated after any change:

    python scripts/build_phase12_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\04_feature_selection.ipynb

Cells cover, in binding body-Phase order (decision D10):
    step 18  statistical filtering  (admissible pool, constants, redundancy, VIF)
    step 19  mutual information     (computed first only because MI is the tie-break)
    step 20  model-based importance (provisional temporal split, LightGBM)
    step 21  recursive/sequential feature selection (backward elimination curve)
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "04_feature_selection.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 12 - Feature selection (workflow body steps 18-21)

| | |
| --- | --- |
| **Workflow** | `flood prediction workflow.md` **Phase 12 - Feature selection**: `18` statistical filtering -> `19` Mutual Information -> `20` model-based importance -> `21` recursive/sequential selection |
| **Checklist #** | 24-25 (cross-reference only - body Phase order is law, decision **D10**) |
| **Input** | `data/processed/flood_dataset_model_ready.csv` (200,665 x 73, post Phase 11) |
| **Boundary** | `reports/phase0_research_definition.md` §5 (63 approved columns) - labels, `obs_days_*`, keys are banned |
| **Decisions** | D1 (JJAS), D2 (all three horizons equal), D7 (persistence baseline is the bar), D8 (no pooled metrics), D10 (body Phase order) |
| **Outputs** | `data/processed/feature_sets.json`, `feature_selection_mi.csv`, `feature_selection_permimportance.csv`, `figures/phase12_*` |

Nothing under `data/raw/` is modified. The `assert`s abort the run, so a green notebook
*is* the evidence.""" ))

cells.append(code("""import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
import lightgbm as lgb

ROOT = Path(r"D:\\Flood predictor")
FIGDIR, OUTDIR = ROOT / "figures", ROOT / "data" / "processed"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEED = 20261008

df = pd.read_csv(OUTDIR / "flood_dataset_model_ready.csv")
df["date"] = pd.to_datetime(df["date"])
assert df.shape == (200665, 73), df.shape
assert str(df["date"].min().date()) == "2017-01-06" and str(df["date"].max().date()) == "2026-10-05"
print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols | {df.station_id.nunique()} stations | "
      f"{df.date.min().date()}..{df.date.max().date()}")"""))

cells.append(md("""## Step 18a - statistical filtering: admissible pool and constants

The pool is taken from the §5 information boundary, not re-invented here: 73 columns minus
10 banned (`station_id name date source`, three labels, three `obs_days_*`) = **63 approved**.
Anything not in that list can never enter a feature matrix.""" ))

cells.append(code("""BANNED = ["station_id", "name", "date", "source"] + HORIZONS + \\
          [f"obs_days_{h[-3:]}" for h in HORIZONS]

cands = [c for c in df.columns if c not in BANNED]
assert len(cands) == 63, f"§5 says 63 approved columns, got {len(cands)}"

# zero / near-zero variance removal (finding G3)
const = [c for c in cands if df[c].nunique(dropna=False) <= 1]
assert const == ["day_observed"], const          # constant in model-ready by construction
FEATS = [c for c in cands if c not in const]     # 62

TEXT = df[FEATS].select_dtypes(exclude=[np.number]).columns.tolist()
NUM = [c for c in FEATS if c not in TEXT]
assert TEXT == ["river", "basin", "division", "district", "upazilla"], TEXT

print(f"63 approved - {len(const)} constant ({const}) = {len(FEATS)} usable features")
print(f"  text  ({len(TEXT)}): {TEXT}")
print(f"  numeric ({len(NUM)})")"""))

cells.append(md("""## Provisional temporal split (needed by steps 20-21)

Feature selection needs a model, so a split is needed *before* Phase 15 formalises it.
This is a **provisional, documented** partition used only to rank features; Phase 15 keeps
the right to redesign it, and the **test years 2023+ are never used for any selection
decision** (D8: no pooled numbers, per-year reporting from Phase 16 on).

| part | dates | role |
| --- | --- | --- |
| train | 2017-01-06 .. 2021-12-31 (2018 empty: FFWC archive gap, F2) | fit imputer/encoder/MI/models |
| val | 2022-01-01 .. 2022-12-31 | importance + selection decisions |
| test | >= 2023-01-01 | **untouched** - reserved for Phase 15/16 |""" ))

cells.append(code("""train = df[df["date"] < "2022-01-01"].copy()
val = df[(df["date"] >= "2022-01-01") & (df["date"] < "2023-01-01")].copy()
test = df[df["date"] >= "2023-01-01"].copy()
assert len(train) + len(val) + len(test) == len(df)
assert train.index.intersection(val.index).size == 0 and train.index.intersection(test.index).size == 0
assert 2018 not in set(train["date"].dt.year)          # FFWC archive gap is respected
assert set(val["date"].dt.year) == {2022} and min(test["date"].dt.year) == 2023

print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} (untouched)")
for h in HORIZONS:
    print(f"  {h}: base rate train={train[h].mean():.3%} val={val[h].mean():.3%} "
          f"test={test[h].mean():.3%} | NaN train={int(train[h].isna().sum()):,}")"""))

cells.append(md("""## Preprocessing fitted on TRAIN only

* **Frequency encoding** for the five text columns (a count per level, unknown level -> 0).
  It is deliberately *not* target encoding: that would leak the label into the feature.
* **Median imputation** fitted on train, applied to val/test (finding G2: 244 rows with NaN
  weather; tree models do not need it, MI and VIF do).
* A *no-upstream-gauge* indicator is **not** added here - that is a Phase 16 modelling
  decision, recorded as an open item in `reports/phase11_spatial_features.md`.""" ))

cells.append(code("""enc_maps = {c: train[c].value_counts(dropna=False) for c in TEXT}

def encode(frame):
    out = frame.copy()
    for c in TEXT:
        out[c] = out[c].map(enc_maps[c]).fillna(0.0).astype(float)
    return out

Xtr_raw = encode(train)[FEATS].astype(float)
med = Xtr_raw.median()                                  # train medians only
Xtr = Xtr_raw.fillna(med)
Xva = encode(val)[FEATS].astype(float).fillna(med)
Xte = encode(test)[FEATS].astype(float).fillna(med)
for nm, X in (("train", Xtr), ("val", Xva), ("test", Xte)):
    assert X.isna().sum().sum() == 0 and np.isfinite(X.to_numpy()).all(), nm
print(f"encoded+imputed: {Xtr.shape} / {Xva.shape} / {Xte.shape}")"""))

cells.append(md("""## Step 19 - Mutual information (the "MI step")

`mutual_info_classif` (kNN entropy estimator, `k=5`, train rows only, all three horizons -
D2). MI is computed *before* the redundancy collapse below purely because MI is the
tie-break when two features are near-duplicates: **the higher-MI one survives**.

Ranking is horizon-neutral: `mi_mean` = mean of the three MI values (D2).""" ))

cells.append(code("""MI = pd.DataFrame(index=FEATS, columns=HORIZONS, dtype=float)
for h in HORIZONS:
    m = train[h].notna()
    y = train.loc[m, h].astype(int)
    t0 = time.time()
    MI[h] = mutual_info_classif(Xtr.loc[m], y, random_state=SEED, n_neighbors=5, n_jobs=-1)
    print(f"{h}: MI computed in {time.time() - t0:5.1f}s | n={m.sum():,} positives={int(y.sum()):,}")

MI["mi_mean"] = MI[HORIZONS].mean(axis=1)
MI = MI.sort_values("mi_mean", ascending=False)
assert MI.isin([np.inf, -np.inf]).sum().sum() == 0
top = MI.head(15)
top"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 7))
top25 = MI.head(25)[::-1]
ax.barh(range(len(top25)), top25["mi_mean"], color="#2b6cb0")
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=8)
ax.set_xlabel("mutual information (mean over 24/48/72 h, train)")
ax.set_title("Phase 12 step 19 - mutual information, top 25 of 62 features")
fig.tight_layout()
fig.savefig(FIGDIR / "phase12_mi.png", dpi=150)
plt.close(fig)
print("wrote figures/phase12_mi.png")"""))

cells.append(md("""## Step 18b - redundancy collapse + VIF

1. **Redundancy**: Spearman `|rho| >= 0.95` on train. Walk the features in MI order and
   drop any that is already represented by a kept feature - so the survivor of every
   near-duplicate group is the higher-MI one (EDA found 137 such pairs, stage 18).
2. **VIF**: iterative removal of the worst `VIF > 10` (correlation-matrix inverse; statsmodels
   is not installed, so `diag(inv(corr))` is computed directly). This produces the
   **LINEAR** set that the logistic-regression baseline must use (phase5 §6: VIF makes an
   unpruned logreg numerically unstable).

Tree models do not need either filter - they keep the full set.""" ))

cells.append(code("""order = [f for f in MI.index if f in FEATS]              # MI mean, descending
R = Xtr[FEATS].corr(method="spearman").abs()

kept, red_drop, red_pairs = [], [], []
for f in order:
    hit = [g for g in kept if R.loc[f, g] >= 0.95]
    if hit:
        red_drop.append(f)
        red_pairs.append((f, hit[0], round(float(R.loc[f, hit[0]]), 3)))
    else:
        kept.append(f)

def vif(cols, data):
    C = data[cols].corr().to_numpy()
    return np.diag(np.linalg.inv(C))

lin = list(kept)
dropped_vif = []
while len(lin) > 5:
    v = vif(lin, Xtr)
    i = int(np.argmax(v))
    if v[i] <= 10.0:
        break
    dropped_vif.append((lin[i], round(float(v[i]), 1)))
    lin.pop(i)

print(f"redundancy: {len(red_drop)} features collapsed at |rho|>=0.95 (survivor = higher MI)")
for f, g, r in red_pairs[:12]:
    print(f"   drop {f:28s} kept {g:28s} rho={r}")
print(f"VIF: {len(dropped_vif)} removed above 10 -> LINEAR set = {len(lin)} features")
print("   ", [f for f, _ in dropped_vif])
LINEAR = lin
assert not set(LINEAR) & set(red_drop) and not set(LINEAR) & set(BANNED)"""))

cells.append(md("""## Step 20 - model-based importance

LightGBM on the **provisional** split, 62 features, `label_*` non-NaN rows only.
Reported against the **persistence baseline** (D7): `wl_pct_danger` alone, scored by PR-AUC
on the same validation rows. Nothing here claims a model result - Phase 16/17 owns that;
this step only ranks features.""" ))

cells.append(code("""def fit_lgb(feats, h):
    ytr, yv = train[h], val[h]
    mtr, mv = ytr.notna(), yv.notna()
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                             min_child_samples=50, objective="binary", n_jobs=-1,
                             random_state=SEED, verbose=-1)
    mdl.fit(Xtr.loc[mtr, feats], ytr[mtr].astype(int))
    p = mdl.predict_proba(Xva.loc[mv, feats])[:, 1]
    return mdl, average_precision_score(yv[mv].astype(int), p)

MODELS, ap_full, ap_pers = {}, {}, {}
for h in HORIZONS:
    MODELS[h], ap_full[h] = fit_lgb(FEATS, h)
    mv = val[h].notna()
    ap_pers[h] = average_precision_score(val.loc[mv, h].astype(int),
                                         val.loc[mv, "wl_pct_danger"])

cmp_tbl = pd.DataFrame({"persistence (wl_pct_danger)": pd.Series(ap_pers),
                        "LightGBM, 62 features": pd.Series(ap_full)})
cmp_tbl["ratio"] = cmp_tbl["LightGBM, 62 features"] / cmp_tbl["persistence (wl_pct_danger)"]
print("PR-AUC on validation year 2022 only (never pooled - D8)")
cmp_tbl"""))

cells.append(code("""PI = pd.DataFrame(index=FEATS)
for h in HORIZONS:
    yv, mv = val[h], val[h].notna()
    r = permutation_importance(MODELS[h], Xva.loc[mv, FEATS], yv[mv].astype(int),
                               n_repeats=5, scoring="average_precision",
                               n_jobs=-1, random_state=SEED)
    PI[f"perm_{h}"] = r.importances_mean
PI["perm_mean"] = PI[[f"perm_{h}" for h in HORIZONS]].mean(axis=1)
PI = PI.sort_values("perm_mean", ascending=False)
assert PI["perm_mean"].notna().all()

fig, ax = plt.subplots(figsize=(9, 7))
t = PI.head(25)[::-1]
ax.barh(range(len(t)), t["perm_mean"], color="#2f855a")
ax.set_yticks(range(len(t)))
ax.set_yticklabels(t.index, fontsize=8)
ax.set_xlabel("mean drop in PR-AUC when the feature is permuted (val 2022, 5 repeats)")
ax.set_title("Phase 12 step 20 - permutation importance, top 25")
fig.tight_layout()
fig.savefig(FIGDIR / "phase12_perm_importance.png", dpi=150)
plt.close(fig)
PI.head(10)"""))

cells.append(code("""# tree-based importance: LightGBM split-gain, normalised per model, mean over horizons
from scipy.stats import spearmanr

GAIN = pd.DataFrame(index=FEATS)
for h in HORIZONS:
    imp = MODELS[h].booster_.feature_importance(importance_type="gain")
    GAIN[f"gain_{h}"] = imp / imp.sum()
GAIN["gain_mean"] = GAIN[[f"gain_{h}" for h in HORIZONS]].mean(axis=1)
GAIN = GAIN.sort_values("gain_mean", ascending=False)

ranks = pd.DataFrame({"perm": [list(PI.index).index(f) + 1 for f in FEATS],
                      "gain": [list(GAIN.index).index(f) + 1 for f in FEATS],
                      "mi": [list(MI.index).index(f) + 1 for f in FEATS]}, index=FEATS)
print("rank agreement across the three methods (spearman on 62 ranks):")
print("   perm vs gain", round(float(spearmanr(ranks["perm"], ranks["gain"]).statistic), 3),
      "| perm vs MI", round(float(spearmanr(ranks["perm"], ranks["mi"]).statistic), 3),
      "| MI vs gain", round(float(spearmanr(ranks["mi"], ranks["gain"]).statistic), 3))
print("top 10 by mean gain importance")
GAIN.head(10).round(4)"""))

cells.append(md("""## Step 21 - recursive / sequential selection

Backward elimination on the permutation ranking: refit on train for the top
`k in {5, 10, 15, 20, 30, 40, 62}` and score on validation (2022), all three horizons.

Rule for the chosen set (documented, not eyeballed): **smallest `k` whose mean PR-AUC over
the three horizons is within 0.01 of the best `k`.** The test years stay untouched.""" ))

cells.append(code("""sizes = [5, 10, 15, 20, 30, 40, len(FEATS)]
rank_mean = PI["perm_mean"].sort_values(ascending=False).index.tolist()

curve = pd.DataFrame(index=sizes, columns=HORIZONS, dtype=float)
for h in HORIZONS:
    order_h = PI[f"perm_{h}"].sort_values(ascending=False).index.tolist()
    for k in sizes:
        curve.loc[k, h] = fit_lgb(order_h[:k], h)[1]
    print(f"{h}: " + "  ".join(f"k={k}:{v:.4f}" for k, v in curve[h].items()))

curve["mean"] = curve[HORIZONS].mean(axis=1)
best = curve["mean"].max()
k_star = int(curve.index[curve["mean"] >= best - 0.01].min())
SELECTED = [f for f in rank_mean[:k_star]]
print(f"best mean PR-AUC = {best:.4f} -> smallest k within 0.01 = {k_star}")
SELECTED"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 5.5))
for h, c in zip(HORIZONS, ["#2b6cb0", "#2f855a", "#c05621"]):
    ax.plot(curve.index, curve[h], marker="o", color=c, label=f"LightGBM {h[-3:]}")
    ax.axhline(ap_pers[h], color=c, ls=":", lw=1)
ax.axvline(k_star, color="black", ls="--", lw=1, label=f"chosen k = {k_star}")
ax.set_xscale("log")
ax.set_xticks(sizes); ax.set_xticklabels(sizes)
ax.set_xlabel("number of features (top-k by permutation importance)")
ax.set_ylabel("PR-AUC on validation year 2022")
ax.set_title("Phase 12 step 21 - feature-count curve (dotted = persistence baseline, D7)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "phase12_selection_curve.png", dpi=150)
plt.close(fig)
print("wrote figures/phase12_selection_curve.png")"""))

cells.append(md("""## Outputs - feature sets for Phases 13+

Three sets, written to `data/processed/feature_sets.json` (tracked in git, unlike the bulk
data). Every set is asserted to contain **no banned column**.""" ))

cells.append(code("""FULL = list(FEATS)
LINEAR_SET = list(LINEAR)
SELECTED_SET = list(SELECTED)

for nm, s in (("FULL", FULL), ("LINEAR", LINEAR_SET), ("SELECTED", SELECTED_SET)):
    assert not set(s) & set(BANNED), f"{nm} contains banned columns"
    assert len(set(s)) == len(s), f"{nm} has duplicates"
    assert all(c in FEATS for c in s), f"{nm} has unknown columns"
    print(f"{nm:9s} n={len(s):3d}")

payload = {
    "created": "2026-10-08",
    "phase": "Phase 12 Feature selection (workflow body steps 18-21; checklist 24-25)",
    "source": "data/processed/flood_dataset_model_ready.csv",
    "rows": int(len(df)),
    "columns_before_selection": 73,
    "banned": BANNED,
    "constant_dropped": const,
    "redundancy_threshold": "|spearman| >= 0.95 on train, survivor = higher MI",
    "vif_threshold": 10,
    "split": {"train": "< 2022-01-01", "val": "2022 (selection decisions only)",
              "test": ">= 2023-01-01 - UNTOUCHED, reserved for Phase 15/16"},
    "selection_rule": f"smallest top-k within 0.01 mean PR-AUC of the best k; k={k_star}",
    "sets": {"FULL": FULL, "LINEAR": LINEAR_SET, "SELECTED": SELECTED_SET},
    "counts": {"FULL": len(FULL), "LINEAR": len(LINEAR_SET), "SELECTED": len(SELECTED_SET)},
    "pr_auc_val_2022": {"persistence": {h: round(float(ap_pers[h]), 4) for h in HORIZONS},
                        "lightgbm_full": {h: round(float(ap_full[h]), 4) for h in HORIZONS}},
    "notes": [
        "LINEAR = VIF-pruned set for the logistic-regression baseline (Phase 16).",
        "FULL = everything admissible; tree models use it.",
        "SELECTED = step-21 outcome; used for the compact/interpretable model and as a check.",
        "test years 2023+ were never read by any selection decision.",
        "Phase 13 (leakage audit) may still demote columns; re-run this notebook after it.",
    ],
}
(OUTDIR / "feature_sets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
MI.to_csv(OUTDIR / "feature_selection_mi.csv")
PI.to_csv(OUTDIR / "feature_selection_permimportance.csv")
print("wrote data/processed/feature_sets.json, feature_selection_mi.csv, "
      "feature_selection_permimportance.csv")
print(json.dumps(payload["counts"]))
print("validation PR-AUC (2022):")
print(cmp_tbl.round(4).to_string())"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
