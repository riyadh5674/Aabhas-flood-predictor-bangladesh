"""MI-3a: does the v2 feature block actually improve the frozen xgboost on the dev cells?

Same model (L2-03 hyper-parameters), same dev cells, same metric function as Phase 20/27.
Feature sets compared: FULL (v1, 62 cols) vs FULL_V2 (v1 + MI-2 block).
Writes reports/mi3_featureset_compare.csv + reports/mi3_featureset_summary.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B          # noqa: E402
from sklearn.pipeline import Pipeline          # noqa: E402
from xgboost import XGBClassifier              # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HOR = B.HORIZONS
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=6,
                  min_child_weight=10, subsample=1.0, colsample_bytree=0.8)
DEV = ["val_2020", "val_2021", "val_2022",
       "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]


def dev_masks(d):
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()
    m = {}
    m["val_2020"] = (year == 2020) & ~holdout
    m["val_2021"] = (year == 2021) & ~holdout
    m["val_2022"] = (split == "val") & ~holdout
    for k in (1, 2, 3, 4):
        m[f"spatial_fold_{k}"] = (split == "train") & (fold == k)
    return m


def make_xgb(features):
    num = [c for c in features if c not in B.TEXT_COLS]
    txt = [c for c in features if c in B.TEXT_COLS]
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)),
                     ("clf", XGBClassifier(**XGB_PARAMS, scale_pos_weight=1.0,
                                           n_jobs=8, tree_method="hist",
                                           eval_metric="logloss",
                                           random_state=B.SEED, verbosity=0))])


def load_v2():
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    fs2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]
    assert len(d) == 200665
    return d, fsets, fs2


def main() -> int:
    t0 = time.time()
    d, fsets, fs2 = load_v2()
    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()

    rows = []
    for fs_name, feats in [("FULL", fsets["FULL"]), ("FULL_V2", fs2["FULL_V2"])]:
        print(f"\n=== feature set {fs_name} ({len(feats)} cols) ===")
        for h in HOR:
            y_all = d[h].to_numpy()
            tr = train & ~pd.isna(y_all)
            pipe = make_xgb(feats)
            t1 = time.time()
            pipe.fit(d.loc[tr, feats], y_all[tr])
            p_all = pipe.predict_proba(d[feats])[:, 1]
            print(f"  {h}: fit {time.time()-t1:.1f}s")
            for cell in DEV:
                m = masks[cell] & ~pd.isna(y_all)
                y, p = y_all[m], p_all[m]
                if len(np.unique(y)) < 2:
                    continue
                thr = B.best_f1_threshold(y_all[tr], pipe.predict_proba(d.loc[tr, feats])[:, 1])
                mm = B.metrics(y, p, thr)
                rows.append({"featureset": fs_name, "horizon": h, "cell": cell,
                             "n": int(m.sum()), "n_pos": int(y.sum()),
                             "pr_auc": mm["pr_auc"], "roc_auc": mm["roc_auc"],
                             "brier": mm["brier"], "ece": mm["ece"],
                             "recall": mm["recall"], "fn": mm["fn"]})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "mi3_featureset_compare.csv", index=False)
    piv = out.pivot_table(index=["horizon", "cell"], columns="featureset",
                          values="pr_auc").round(4)
    piv["delta_v2"] = (piv["FULL_V2"] - piv["FULL"]).round(4)
    print("\n=== PR-AUC by cell (FULL vs FULL_V2) ===")
    print(piv.to_string())
    summ = {}
    for h in HOR:
        sub = out[out.horizon == h]
        summ[h] = {
            "FULL_mean": round(float(sub[sub.featureset == "FULL"].pr_auc.mean()), 5),
            "FULL_V2_mean": round(float(sub[sub.featureset == "FULL_V2"].pr_auc.mean()), 5),
        }
        summ[h]["delta"] = round(summ[h]["FULL_V2_mean"] - summ[h]["FULL_mean"], 5)
    summ["val_2022_horizon_mean"] = {
        "FULL": round(float(out[(out.featureset == "FULL") & (out.cell == "val_2022")].pr_auc.mean()), 5),
        "FULL_V2": round(float(out[(out.featureset == "FULL_V2") & (out.cell == "val_2022")].pr_auc.mean()), 5),
    }
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    (REPORTS / "mi3_featureset_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summ, indent=2))
    print(f"\nWROTE {REPORTS/'mi3_featureset_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
