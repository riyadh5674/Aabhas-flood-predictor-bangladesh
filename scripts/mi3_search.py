"""MI-3b/c: randomised hyper-parameter search + probability ensemble.

No new dependencies (optuna/catboost unavailable): uses sklearn ParameterSampler and
the already-installed xgboost / lightgbm families.  Objective = val_2022 PR-AUC
(the Phase-33 primary cell); mean PR-AUC over the seven development cells is also
recorded.  Validation-only; the test period is never touched.

MI-3b deviation from the plan (recorded in reports/mi_status.md): the plan asked for
Optuna (>=300 trials) over a pool of xgb / lgbm / catboost / hist_gradient_boosting /
extra_trees.  optuna and catboost are not installed and must not be installed, so this
script samples with sklearn.model_selection.ParameterSampler (seed fixed) and adds the
two sklearn families that need no new dependency: HistGradientBoostingClassifier and
ExtraTreesClassifier (N_TRIALS_SK sampled configs each, below).  sklearn's HGB/ET
expose no scale_pos_weight, so none is passed for them (dropped, not crashed).

Budget: 12 xgb + 12 lgbm + 8 hgb + 8 et + 1 frozen = 41 configs per horizon (123 fits).
This is a deliberate runtime cap agreed after the first out-of-order attempt; the plan's
Optuna >=300-trial target is not met and the gap is recorded in reports/mi_status.md.

RESUMABLE / MEMORY-LIGHT (2026-10-16): the first background attempts died silently
after ~10 configs on a 15.7 GB / 3.4 GB-free host, with no traceback -- consistent with
an out-of-memory kill.  Two hardening changes:
  * every finished config is appended immediately to reports/mi3_hpo_partial.csv, and a
    restart skips (horizon, tag) pairs already present, so repeated launches make progress;
  * predictions are taken only on the union of development rows (never all 200,665 rows),
    n_jobs is reduced to 4, and memory is released between configs.
CLI: `python -u scripts/mi3_search.py [24h 48h 72h ...]` runs only the given horizons
(default: all three).  The final reports/mi3_hpo.csv and models/mi/mi_hpo_<horizon>.json
are (re)written from the partial file at the end, so a partially-finished run still
produces usable output.

Writes:
  reports/mi3_hpo_partial.csv    appended result per finished config (resume ledger)
  reports/mi3_hpo.csv            every tried config, sorted by horizon then mean PR-AUC
  models/mi/mi_hpo_<horizon>.json  best config per horizon (by val_2022)
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B          # noqa: E402
from sklearn.pipeline import Pipeline          # noqa: E402
from sklearn.model_selection import ParameterSampler  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              ExtraTreesClassifier)
from xgboost import XGBClassifier              # noqa: E402
from lightgbm import LGBMClassifier            # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "mi"
MODELS.mkdir(parents=True, exist_ok=True)
PARTIAL = REPORTS / "mi3_hpo_partial.csv"
TAG = ""                     # set from --tag in main(); "" reproduces the pre-ERA5 paths
HOR = B.HORIZONS
DEV = ["val_2020", "val_2021", "val_2022",
       "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
N_TRIALS = 24
N_TRIALS_SK = 8           # MI-3b: sampled configs per sklearn family (budget-bounded)
N_JOBS = 4                # memory-light: 8 threads on a 3 GB-free host invites the OOM kill
SEED = B.SEED

XGB_SPACE = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
    "scale_pos_weight": [1.0, 3.0, 10.0],
}
LGBM_SPACE = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}
# MI-3b sklearn families (no new dependency; no scale_pos_weight available/exists)
HGB_SPACE = {
    "max_iter": [200, 400],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 1.0, 3.0],
}
ET_SPACE = {
    "n_estimators": [300, 500],
    "max_depth": [None, 12, 20],
    "min_samples_leaf": [1, 3, 5],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.5],
}

# Fixed partial-ledger schema.  Families expose DIFFERENT hyper-parameter keys, so every
# appended row is reindexed to this union; otherwise to_csv(mode="a") writes ragged rows
# (e.g. a sampled xgb row carries reg_lambda that the frozen row lacks) and the resume
# read_csv fails with "Expected N fields, saw N+1" (observed 2026-10-16).
META_COLS = ["horizon", "family", "tag", "mean_pr_auc", "val_2022", "secs"]
CFG_COLS = ["n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "reg_lambda", "scale_pos_weight",
            "num_leaves", "min_child_samples",
            "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
            "min_samples_split", "max_features"]


def load_v2():
    suffix = f"_{TAG}" if TAG else ""
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / f"mi_v2_features{suffix}.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    fs2 = json.loads((PROC / f"feature_sets_v2{suffix}.json").read_text(encoding="utf-8"))["sets"]
    return d, fs2


def dev_masks(d):
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()
    return {"val_2020": (year == 2020) & ~holdout,
            "val_2021": (year == 2021) & ~holdout,
            "val_2022": (split == "val") & ~holdout,
            **{f"spatial_fold_{k}": (split == "train") & (fold == k) for k in (1, 2, 3, 4)}}


def build(family, cfg, feats):
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    if family == "xgb":
        clf = XGBClassifier(**cfg, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss",
                            random_state=SEED, verbosity=0)
    elif family == "lgbm":
        clf = LGBMClassifier(**cfg, n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    elif family == "hgb":
        # sklearn HGB: no scale_pos_weight parameter exists -> never passed
        clf = HistGradientBoostingClassifier(**cfg, random_state=SEED)
    elif family == "et":
        clf = ExtraTreesClassifier(**cfg, n_jobs=N_JOBS, class_weight="balanced_subsample",
                                   random_state=SEED)
    else:
        raise ValueError(f"unknown family {family}")
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", clf)])


def score_cells(pipe, d, masks, y_all, train_mask, feats, keep_pos):
    """Predict only on the union of development rows (keep_pos), then score each cell.

    p_all is a full-length array with NaN outside keep_pos; every dev cell and the
    training threshold rows are subsets of keep_pos by construction.
    """
    p_sub = pipe.predict_proba(d.iloc[keep_pos][feats])[:, 1]
    p_all = np.full(len(d), np.nan)
    p_all[keep_pos] = p_sub
    thr = B.best_f1_threshold(y_all[train_mask], p_all[train_mask])
    out = {}
    for cell in DEV:
        m = masks[cell] & ~pd.isna(y_all)
        if m.sum() == 0 or len(np.unique(y_all[m])) < 2:
            continue
        out[cell] = B.metrics(y_all[m], p_all[m], thr)["pr_auc"]
    return out


def _read_partial():
    """Read the ledger, tolerating a truncated final line left by a hard kill."""
    try:
        return pd.read_csv(PARTIAL, on_bad_lines="skip")
    except TypeError:            # older pandas: no on_bad_lines
        return pd.read_csv(PARTIAL, engine="python")
    except Exception:
        return pd.DataFrame()


def load_done():
    if PARTIAL.exists() and PARTIAL.stat().st_size > 0:
        prev = _read_partial()
        if {"horizon", "tag"}.issubset(prev.columns):
            return {(r.horizon, r.tag) for r in prev.itertuples()}
    return set()


def append_row(row: dict):
    # Reindex to the fixed union schema so every row has the same columns regardless of
    # which family produced it (see META_COLS/CFG_COLS).
    df = pd.DataFrame([row]).reindex(columns=META_COLS + CFG_COLS)
    header = (not PARTIAL.exists()) or PARTIAL.stat().st_size == 0
    df.to_csv(PARTIAL, mode="a", header=header, index=False)


def write_outputs():
    """(Re)build the final sorted table and best-config JSONs from the partial ledger."""
    if not PARTIAL.exists():
        print("no partial ledger; nothing to write")
        return
    hpo = _read_partial().drop_duplicates(subset=["horizon", "tag"], keep="last")
    hpo = hpo.sort_values(["horizon", "mean_pr_auc"], ascending=[True, False])
    suffix = f"_{TAG}" if TAG else ""
    hpo.to_csv(REPORTS / f"mi3_hpo{suffix}.csv", index=False)
    for h, g in hpo.groupby("horizon"):
        b = g.sort_values("val_2022", ascending=False).iloc[0]
        cfg = {k: v for k, v in b.items()
               if k not in ("horizon", "family", "tag", "mean_pr_auc", "val_2022", "secs")
               and not (isinstance(v, float) and np.isnan(v))}
        (MODELS / f"mi_hpo{suffix}_{h}.json").write_text(
            json.dumps({"family": b.family, "tag": b.tag, "val_2022": float(b.val_2022),
                        "mean_pr_auc": float(b.mean_pr_auc), "cfg": cfg}, indent=2),
            encoding="utf-8")
    print(f"WROTE {REPORTS/f'mi3_hpo{suffix}.csv'}  ({len(hpo)} rows over {hpo.horizon.nunique()} horizons)")


def main(argv) -> int:
    global PARTIAL, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="optional suffix; 'era5' reads mi_v2_features_era5.csv + "
                         "feature_sets_v2_era5.json and writes mi3_hpo_era5.csv + "
                         "models/mi/mi_hpo_era5_<horizon>.json (separate resume ledger)")
    a, rest = ap.parse_known_args(argv)
    TAG = a.tag
    PARTIAL = REPORTS / f"mi3_hpo_partial{'_' + TAG if TAG else ''}.csv"
    t0 = time.time()
    want = [h for h in rest if h in HOR] or HOR
    done = load_done()
    print(f"resume ledger: {len(done)} configs already done; running horizons {want}")

    d, fs2 = load_v2()
    feats = fs2["FULL_V2"]
    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()

    # predict only on training + development rows (never the frozen test rows)
    dev_union = np.zeros(len(d), dtype=bool)
    for m in masks.values():
        dev_union |= m
    keep_pos = np.where(train | dev_union)[0]
    print(f"loaded {len(d):,} rows x {len(feats)+13} cols; "
          f"train {int(train.sum()):,} / keep {len(keep_pos):,}")

    for h in want:
        y_all = d[h].to_numpy()
        tr = train & ~pd.isna(y_all)
        Xtr = d.loc[tr, feats]
        ytr = y_all[tr]

        base_cfg = dict(n_estimators=300, learning_rate=0.03, max_depth=6,
                        min_child_weight=10, subsample=1.0, colsample_bytree=0.8,
                        scale_pos_weight=1.0)
        trials = [("xgb", base_cfg, "frozen")] + \
                 [("xgb", c, f"xgb{i}") for i, c in enumerate(
                     ParameterSampler(XGB_SPACE, n_iter=N_TRIALS // 2, random_state=SEED))] + \
                 [("lgbm", c, f"lgbm{i}") for i, c in enumerate(
                     ParameterSampler(LGBM_SPACE, n_iter=N_TRIALS // 2, random_state=SEED))] + \
                 [("hgb", c, f"hgb{i}") for i, c in enumerate(
                     ParameterSampler(HGB_SPACE, n_iter=N_TRIALS_SK, random_state=SEED))] + \
                 [("et", c, f"et{i}") for i, c in enumerate(
                     ParameterSampler(ET_SPACE, n_iter=N_TRIALS_SK, random_state=SEED))]

        todo = [t for t in trials if (h, t[2]) not in done]
        print(f"\n=== {h}: {len(trials)} configs ({len(todo)} to run) ===")
        for fam, cfg, tag in todo:
            t1 = time.time()
            pipe = build(fam, cfg, feats)
            pipe.fit(Xtr, ytr)
            dev = score_cells(pipe, d, masks, y_all, tr, feats, keep_pos)
            mean_pr = float(np.mean(list(dev.values())))
            v22 = float(dev.get("val_2022", float("nan")))
            append_row({"horizon": h, "family": fam, "tag": tag, "mean_pr_auc": mean_pr,
                        "val_2022": v22, "secs": round(time.time() - t1, 1), **cfg})
            print(f"  {tag:8s} {fam:4s} mean={mean_pr:.4f} val2022={v22:.4f} {time.time()-t1:.0f}s",
                  flush=True)
            del pipe
            gc.collect()

    write_outputs()
    print(f"\nDONE wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
