"""MI-4 Test Read #2: Hybrid V4 (24h) + D16 (48h/72h) on frozen test.

Guarded: test read only if logs don't exist; `--reread` to force.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import joblib
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

LOG_FILE = REPORTS / "mi4_test_read_log.json"
METRICS_FILE = REPORTS / "mi4_test_metrics.csv"
THRESH_FILE = REPORTS / "mi4_test_thresholds.csv"
SUMMARY_FILE = REPORTS / "mi4_test_summary.json"

SEED = B.SEED
HOR = B.HORIZONS

# Hybrid model config: V4 at 24h, D16 at 48h/72h
HYBRID_CFG = {
    "label_24h": {"family": "lgbm", "cfg": "mi4_hpo_v4_label_24h.json", "feat_set": "FULL_V4"},
    "label_48h": {"family": "lgbm", "cfg": "mi_hpo_label_48h.json", "feat_set": "FULL_V2"},
    "label_72h": {"family": "xgb", "cfg": "mi_hpo_label_72h.json", "feat_set": "FULL_V2"},
}


def load_data(feat_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Load experimental dataset with specified feature set."""
    exp = pd.read_csv(PROC / "experimental_dataset.csv", parse_dates=["date"])
    
    if feat_set == "FULL_V4":
        v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
        wc = pd.read_csv(PROC / "esa_worldcover_features.csv")
        mi8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
        splits = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])
        
        d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
        d = d.merge(wc, on="station_id", how="left", validate="many_to_one")
        d = d.merge(mi8, on=["station_id", "date"], validate="one_to_one")
        d = d.merge(splits[["station_id", "date", "split", "spatial_fold", "station_holdout"]],
                    on=["station_id", "date"], validate="one_to_one")
        
        fsets = json.loads((PROC / "feature_sets_v4.json").read_text())["sets"]
        feats = fsets["FULL_V4"]
        
    elif feat_set == "FULL_V2":
        v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
        splits = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])
        
        d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
        d = d.merge(splits[["station_id", "date", "split", "spatial_fold", "station_holdout"]],
                    on=["station_id", "date"], validate="one_to_one")
        
        fsets = json.loads((PROC / "feature_sets_v2.json").read_text())["sets"]
        feats = fsets["FULL_V2"]
        
    else:
        raise ValueError(f"Unknown feat_set: {feat_set}")
    
    d = d.sort_values(["station_id", "date"]).drop_duplicates(
        ["station_id", "date"], keep="first").reset_index(drop=True)
    return d, feats


def load_model(family: str, cfg_path: str, feats: list[str]):
    cfg = json.loads((MODELS / cfg_path).read_text())["cfg"]
    for k, v in cfg.items():
        if isinstance(v, float) and v.is_integer():
            cfg[k] = int(v)
    
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    
    if family == "lgbm":
        clf = LGBMClassifier(**cfg, n_jobs=4, random_state=SEED, verbose=-1)
    elif family == "xgb":
        clf = XGBClassifier(**cfg, n_jobs=4, tree_method="hist", eval_metric="logloss",
                            random_state=SEED, verbosity=0)
    elif family == "hgb":
        clf = HistGradientBoostingClassifier(**cfg, random_state=SEED)
    else:
        raise ValueError(f"Unknown family: {family}")
    
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", clf)])


def get_threshold(d, y, p, horizon, feat_set):
    """Derive period-specific cost-optimal threshold on val_2022."""
    val_mask = (d.split == "val") & ~d.station_holdout & ~pd.isna(y) & ~pd.isna(p)
    if val_mask.sum() == 0:
        return 0.5
    
    y_val = y[val_mask]
    p_val = p[val_mask]
    
    # Cost: C_FN=10, C_FP=1
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.60, 60):
        pred = (p_val >= t).astype(int)
        fn = ((y_val == 1) & (pred == 0)).sum()
        fp = ((y_val == 0) & (pred == 1)).sum()
        cost = 10 * fn + 1 * fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reread", action="store_true", help="Force re-read even if logged")
    ap.add_argument("--dry-run", action="store_true", help="Compute thresholds only, no test scoring")
    args = ap.parse_args(argv)
    
    # Check if already read
    if LOG_FILE.exists() and not args.reread:
        log = json.loads(LOG_FILE.read_text())
        if log.get("n_reads", 0) >= 1:
            print(f"Test already read {log['n_reads']} time(s). Use --reread to force.")
            return 0
    
    print("=== MI-4 Test Read #2: Hybrid V4 (24h) + D16 (48h/72h) ===")
    
    # Load D16 models for 48h/72h and V4 for 24h
    models = {}
    all_preds = {}
    
    for h in HOR:
        cfg_info = HYBRID_CFG[h]
        d, feats = load_data(cfg_info["feat_set"])
        pipe = load_model(cfg_info["family"], cfg_info["cfg"], feats)
        
        train = (d.split == "train") & ~d.station_holdout & ~pd.isna(d[h])
        pipe.fit(d.loc[train, feats], d.loc[train, h])
        
        # Predict on all rows
        p_all = np.full(len(d), np.nan)
        valid = ~d[feats].isna().any(axis=1)
        p_all[valid] = pipe.predict_proba(d.loc[valid, feats])[:, 1]
        
        models[h] = {"pipe": pipe, "d": d, "feats": feats, "p_all": p_all, "cfg_info": cfg_info}
    
    # Derive thresholds on val_2022 for each horizon
    thresholds = {}
    for h in HOR:
        m = models[h]
        thr = get_threshold(m["d"], m["d"][h].to_numpy(), m["p_all"], h, m["cfg_info"]["feat_set"])
        thresholds[h] = thr
        print(f"{h}: threshold = {thr:.4f} (model: {m['cfg_info']['family']} on {m['cfg_info']['feat_set']})")
    
    if args.dry_run:
        print("\nDry run complete. Thresholds computed.")
        return 0
    
    # Score on frozen test (split == "test")
    test_cells = ["test_2023", "test_2024", "test_2025", "test_2026",
                  "test_trained", "test_new_network", "test_unseen"]
    
    rows = []
    for h in HOR:
        m = models[h]
        d = m["d"]
        p_all = m["p_all"]
        thr = thresholds[h]
        y = d[h].to_numpy()
        
        # Test mask
        test_mask = (d.split == "test") & ~pd.isna(y) & ~pd.isna(p_all)
        
        for cell in test_cells:
            if cell.startswith("test_") and cell[5:].isdigit():
                # Year-based cell
                year = int(cell[5:])
                cell_mask = test_mask & (d.date.dt.year == year)
            elif cell == "test_trained":
                cell_mask = test_mask & (d.spatial_fold != -1)  # stations in train
            elif cell == "test_new_network":
                cell_mask = test_mask & (d.date.dt.year >= 2023) & (d.station_holdout == False)
            elif cell == "test_unseen":
                cell_mask = test_mask & (d.station_holdout == True)
            else:
                continue
            
            if cell_mask.sum() == 0 or len(np.unique(y[cell_mask])) < 2:
                continue
            
            y_cell = y[cell_mask]
            p_cell = p_all[cell_mask]
            pred = (p_cell >= thr).astype(int)
            
            tp = ((y_cell == 1) & (pred == 1)).sum()
            fp = ((y_cell == 0) & (pred == 1)).sum()
            fn = ((y_cell == 1) & (pred == 0)).sum()
            tn = ((y_cell == 0) & (pred == 0)).sum()
            
            pr_auc = B.metrics(y_cell, p_cell, thr)["pr_auc"]
            roc_auc = B.metrics(y_cell, p_cell, thr)["roc_auc"]
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            rows.append({
                "model": f"hybrid_v4_24h",
                "horizon": h,
                "cell": cell,
                "n": int(cell_mask.sum()),
                "n_pos": int(y_cell.sum()),
                "threshold": thr,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "recall": recall,
                "precision": precision,
                "far": far,
                "f1": f1,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })
    
    # Also score D7 persistence for comparison
    # (D7 uses wl_pct_danger alone)
    # For now, skip D7 re-score since it's already in mi6_test_metrics.csv
    
    # Save metrics
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_FILE, index=False)
    
    # Save thresholds
    thr_df = pd.DataFrame([{"horizon": h, "threshold": t} for h, t in thresholds.items()])
    thr_df.to_csv(THRESH_FILE, index=False)
    
    # Update log
    n_reads = 1
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text())
        n_reads = log.get("n_reads", 0) + 1
    log = {
        "n_reads": n_reads,
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "models": {h: v["cfg_info"] for h, v in models.items()},
        "thresholds": thresholds,
        "test_cells": test_cells,
    }
    LOG_FILE.write_text(json.dumps(log, indent=2))
    
    # Summary
    summary = {
        "hybrid_model": "V4 (24h) + D16 (48h/72h)",
        "n_reads": n_reads,
        "thresholds": thresholds,
        "horizon_means": metrics_df.groupby("horizon")["pr_auc"].mean().to_dict(),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"\n=== Test Read Complete (n_reads={n_reads}) ===")
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Thresholds saved to {THRESH_FILE}")
    print(f"Log updated: {LOG_FILE}")
    print(f"Summary saved to {SUMMARY_FILE}")
    
    # Print headline
    for h in HOR:
        h_data = metrics_df[metrics_df["horizon"] == h]
        mean_pr = h_data["pr_auc"].mean()
        print(f"  {h}: PR-AUC = {mean_pr:.4f} (mean over {len(h_data)} cells)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))