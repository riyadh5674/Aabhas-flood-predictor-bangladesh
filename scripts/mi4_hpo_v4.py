"""MI-4: Extended HPO on FULL_V4 (187 features).

Budget: 71 configs/horizon (vs MI-3b's 41) — per MI-8 design.
Families: xgb, lgbm, hgb, et — same as MI-3b.
Target: val_2022 PR-AUC (pre-registered decision cell).
Resumable, memory-light, validation-only.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "mi"
MODELS.mkdir(parents=True, exist_ok=True)

PARTIAL = REPORTS / "mi4_hpo_partial_v4.csv"
HOR = B.HORIZONS
DEV = ["val_2020", "val_2021", "val_2022",
       "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
N_TRIALS = 40        # 20 xgb + 20 lgbm
N_TRIALS_SK = 15     # 15 hgb + 15 et = 30 sklearn
N_JOBS = 4
SEED = B.SEED

XGB_SPACE = {
    "n_estimators": [300, 500, 800, 1000],
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
    "scale_pos_weight": [1.0, 3.0, 10.0],
}
LGBM_SPACE = {
    "n_estimators": [300, 500, 800, 1000],
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}
HGB_SPACE = {
    "max_iter": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 1.0, 3.0],
}
ET_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 12, 20, 30],
    "min_samples_leaf": [1, 3, 5],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.5, 0.7],
}

META_COLS = ["horizon", "family", "tag", "mean_pr_auc", "val_2022", "secs"]
CFG_COLS = ["n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "reg_lambda", "scale_pos_weight",
            "num_leaves", "min_child_samples",
            "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
            "min_samples_split", "max_features"]


def load_v4():
    """Load experimental_dataset + mi_v2_features + WorldCover + mi8_features + split assignments."""
    exp = pd.read_csv(PROC / "experimental_dataset.csv", parse_dates=["date"])
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    wc = pd.read_csv(PROC / "esa_worldcover_features.csv")
    mi8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
    splits = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])
    
    d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
    d = d.merge(wc, on="station_id", how="left", validate="many_to_one")
    d = d.merge(mi8, on=["station_id", "date"], validate="one_to_one")
    d = d.merge(splits[["station_id", "date", "split", "spatial_fold", "station_holdout"]],
                on=["station_id", "date"], validate="one_to_one")
    d = d.sort_values(["station_id", "date"]).drop_duplicates(
        ["station_id", "date"], keep="first").reset_index(drop=True)
    
    fsets = json.loads((PROC / "feature_sets_v4.json").read_text(encoding="utf-8"))["sets"]
    return d, fsets


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
        clf = HistGradientBoostingClassifier(**cfg, random_state=SEED)
    elif family == "et":
        clf = ExtraTreesClassifier(**cfg, n_jobs=N_JOBS, class_weight="balanced_subsample",
                                   random_state=SEED)
    else:
        raise ValueError(f"unknown family {family}")
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", clf)])


def score_cells(pipe, d, masks, y_all, train_mask, feats, keep_pos):
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
    try:
        return pd.read_csv(PARTIAL, on_bad_lines="skip")
    except TypeError:
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
    df = pd.DataFrame([row]).reindex(columns=META_COLS + CFG_COLS)
    header = (not PARTIAL.exists()) or PARTIAL.stat().st_size == 0
    df.to_csv(PARTIAL, mode="a", header=header, index=False)


def write_outputs():
    if not PARTIAL.exists():
        print("no partial ledger; nothing to write")
        return
    hpo = _read_partial().drop_duplicates(subset=["horizon", "tag"], keep="last")
    hpo = hpo.sort_values(["horizon", "mean_pr_auc"], ascending=[True, False])
    hpo.to_csv(REPORTS / "mi4_hpo_v4.csv", index=False)
    for h, g in hpo.groupby("horizon"):
        b = g.sort_values("val_2022", ascending=False).iloc[0]
        cfg = {k: v for k, v in b.items()
               if k not in ("horizon", "family", "tag", "mean_pr_auc", "val_2022", "secs")
               and not (isinstance(v, float) and np.isnan(v))}
        (MODELS / f"mi4_hpo_v4_{h}.json").write_text(
            json.dumps({"family": b.family, "tag": b.tag, "val_2022": float(b.val_2022),
                        "mean_pr_auc": float(b.mean_pr_auc), "cfg": cfg}, indent=2),
            encoding="utf-8")
    print(f"WROTE {REPORTS/'mi4_hpo_v4.csv'}  ({len(hpo)} rows over {hpo.horizon.nunique()} horizons)")


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    a, rest = ap.parse_known_args(argv)
    want = [h for h in rest if h in HOR] or HOR
    done = load_done()
    print(f"resume ledger: {len(done)} configs already done; running horizons {want}")

    d, fsets = load_v4()
    feats = fsets["FULL_V4"]
    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()

    dev_union = np.zeros(len(d), dtype=bool)
    for m in masks.values():
        dev_union |= m
    keep_pos = np.where(train | dev_union)[0]
    print(f"loaded {len(d):,} rows x {len(feats)+13} cols; "
          f"train {int(train.sum()):,} / keep {len(keep_pos):,}")
    print(f"FULL_V4 features: {len(feats)}")

    t0 = time.time()
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