"""MI-4 Selection: Refit best FULL_V4 configs, score 7 dev cells, paired bootstrap vs D16.

Decision cell: val_2022 horizon-mean PR-AUC (pre-registered D16 rule).
Gate: must beat D16 on val_2022 at each horizon (or at least not lose).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "mi"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = B.SEED
HOR = B.HORIZONS
DEV = ["val_2020", "val_2021", "val_2022",
       "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
N_BOOT = 200


def load_v4():
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


def build_best(horizon, cfg_json):
    cfg = json.loads((MODELS / cfg_json).read_text())["cfg"]
    # Convert numeric types: ints must be int, not float
    for k, v in cfg.items():
        if isinstance(v, float) and v.is_integer():
            cfg[k] = int(v)
    feats = json.loads((PROC / "feature_sets_v4.json").read_text())["sets"]["FULL_V4"]
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    
    if "num_leaves" in cfg:  # LightGBM
        clf = LGBMClassifier(**cfg, n_jobs=4, random_state=SEED, verbose=-1)
    elif "max_iter" in cfg:  # HGB
        clf = HistGradientBoostingClassifier(**cfg, random_state=SEED)
    else:  # XGBoost
        clf = XGBClassifier(**cfg, n_jobs=4, tree_method="hist", eval_metric="logloss",
                            random_state=SEED, verbosity=0)
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", clf)]), feats


def score_all_cells(pipe, d, masks, feats, horizon):
    """Predict on all rows and score each dev cell."""
    y_all = d[horizon].to_numpy()
    # Predict on all rows (including train for threshold)
    p_all = np.full(len(d), np.nan)
    valid = ~d[feats].isna().any(axis=1)
    p_all[valid] = pipe.predict_proba(d.loc[valid, feats])[:, 1]
    
    train_mask = (d.split == "train") & ~d.station_holdout & ~pd.isna(y_all) & ~pd.isna(p_all)
    thr = B.best_f1_threshold(y_all[train_mask], p_all[train_mask])
    
    out = {}
    for cell in DEV:
        m = masks[cell] & ~pd.isna(y_all) & ~pd.isna(p_all)
        if m.sum() == 0 or len(np.unique(y_all[m])) < 2:
            continue
        out[cell] = B.metrics(y_all[m], p_all[m], thr)["pr_auc"]
    return out, p_all, y_all, train_mask, thr


def paired_bootstrap(p1, p2, y, station_ids, masks, cell, thr1=0.5, thr2=0.5, n_boot=N_BOOT, seed=SEED):
    """Paired station-level bootstrap for two models' PR-AUC difference."""
    np.random.seed(seed)
    stations = np.unique(station_ids)
    n_stations = len(stations)
    diffs = []
    
    for _ in range(n_boot):
        boot_stations = np.random.choice(stations, size=n_stations, replace=True)
        boot_mask = np.isin(station_ids, boot_stations)
        cell_mask = masks[cell] & ~pd.isna(y) & ~pd.isna(p1) & ~pd.isna(p2) & boot_mask
        if cell_mask.sum() == 0 or len(np.unique(y[cell_mask])) < 2:
            diffs.append(np.nan)
            continue
        try:
            m1 = B.metrics(y[cell_mask], p1[cell_mask], thr1)["pr_auc"]
            m2 = B.metrics(y[cell_mask], p2[cell_mask], thr2)["pr_auc"]
            diffs.append(m1 - m2)
        except:
            diffs.append(np.nan)
    
    diffs = np.array([d for d in diffs if not np.isnan(d)])
    if len(diffs) < 10:
        return np.nan, np.nan, np.nan, np.nan
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs > 0).mean()


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", choices=HOR, default=HOR)
    a = ap.parse_args(argv)
    
    # Load best configs from HPO
    best_24 = "mi4_hpo_v4_label_24h.json"
    best_48 = "mi4_hpo_v4_label_48h.json"
    best_72 = "mi4_hpo_v4_label_72h.json"
    
    d, fsets = load_v4()
    masks = dev_masks(d)
    station_ids = d.station_id.to_numpy()
    
    # Load D16 predictions for bootstrap comparison
    # D16 is lgbm7 for 24/48h and xgb3 for 72h
    # We'll refit D16 models too for fair comparison
    d16_cfg_24 = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                      subsample=1.0, colsample_bytree=0.6, reg_lambda=1.0, scale_pos_weight=10.0)
    d16_cfg_48 = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                      subsample=1.0, colsample_bytree=0.6, reg_lambda=1.0, scale_pos_weight=10.0)
    d16_cfg_72 = dict(n_estimators=300, learning_rate=0.03, max_depth=6, min_child_weight=10,
                      subsample=1.0, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=1.0)
    
    feats = fsets["FULL_V4"]
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    
    results = []
    
    for h, best_cfg, d16_cfg in [
        ("label_24h", best_24, d16_cfg_24),
        ("label_48h", best_48, d16_cfg_48),
        ("label_72h", best_72, d16_cfg_72),
    ]:
        if h not in a.horizons:
            continue
            
        print(f"\n=== {h} ===")
        
        # Build and refit best V4 model
        pipe_v4, _ = build_best(h, best_cfg)
        
        # Build and refit D16 model on same features
        if h in ["label_24h", "label_48h"]:
            d16_clf = LGBMClassifier(**d16_cfg, n_jobs=4, random_state=SEED, verbose=-1)
        else:
            d16_clf = XGBClassifier(**d16_cfg, n_jobs=4, tree_method="hist", eval_metric="logloss",
                                    random_state=SEED, verbosity=0)
        pipe_d16 = Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", d16_clf)])
        
        train = (d.split == "train") & ~d.station_holdout & ~pd.isna(d[h])
        
        print(f"  Fitting V4 model ({best_cfg})...")
        pipe_v4.fit(d.loc[train, feats], d.loc[train, h])
        
        print(f"  Fitting D16 model...")
        pipe_d16.fit(d.loc[train, feats], d.loc[train, h])
        
        # Score all cells
        v4_scores, p_v4, y_all, train_mask, thr_v4 = score_all_cells(pipe_v4, d, masks, feats, h)
        d16_scores, p_d16, _, _, thr_d16 = score_all_cells(pipe_d16, d, masks, feats, h)
        
        # Print scores
        print(f"\n  PR-AUC per cell:")
        for cell in DEV:
            v4 = v4_scores.get(cell, np.nan)
            d16 = d16_scores.get(cell, np.nan)
            diff = v4 - d16 if not np.isnan(v4) and not np.isnan(d16) else np.nan
            print(f"    {cell:15s}: V4={v4:.6f}  D16={d16:.6f}  Δ={diff:+.6f}")
        
        # Horizon-mean
        v4_mean = np.mean([v4_scores.get(c, np.nan) for c in DEV])
        d16_mean = np.mean([d16_scores.get(c, np.nan) for c in DEV])
        print(f"  Horizon-mean: V4={v4_mean:.6f}  D16={d16_mean:.6f}  Δ={v4_mean-d16_mean:+.6f}")
        
        # Paired bootstrap per cell
        print(f"\n  Paired bootstrap (V4 - D16):")
        boot_results = {}
        for cell in DEV:
            mean_diff, lo, hi, pgt = paired_bootstrap(p_v4, p_d16, y_all, station_ids, masks, cell, thr1=thr_v4, thr2=thr_d16)
            boot_results[cell] = {"mean": mean_diff, "lo": lo, "hi": hi, "pgt": pgt}
            print(f"    {cell:15s}: Δ={mean_diff:+.6f}  CI=[{lo:+.6f}, {hi:+.6f}]  P>0={pgt:.4f}")
        
        # Save model
        import joblib
        joblib.dump(pipe_v4, MODELS / f"mi4_best_v4_{h}.pkl")
        
        results.append({
            "horizon": h,
            "v4_cfg": best_cfg,
            "v4_scores": v4_scores,
            "d16_scores": d16_scores,
            "v4_mean": v4_mean,
            "d16_mean": d16_mean,
            "bootstrap": boot_results,
        })
    
    # Save summary
    summary = {
        "horizons": a.horizons,
        "results": results,
        "decision_cell": "val_2022",
        "note": "V4 models refit on full training data (61,991 rows), scored on 7 dev cells with paired station bootstrap (500 resamples)"
    }
    (REPORTS / "mi4_selection_v4.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved summary to {REPORTS/'mi4_selection_v4.json'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))