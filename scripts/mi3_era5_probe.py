r"""MI-3a(ERA5): does the ERA5-Land block improve the frozen xgboost on the dev cells?

Same model (L2-03 hyper-parameters), same dev cells, same metric function as MI-3a.
Feature sets compared on the SAME rows of the ERA5-augmented table:
  * FULL          — v1, 62 cols
  * FULL_V2_pre   — MI-2 block WITHOUT ERA5 (110 cols, read from the frozen feature_sets_v2.json)
  * FULL_V2_era5  — MI-2 block WITH ERA5   (135 cols, read from feature_sets_v2_era5.json)

Non-destructive: reads mi_v2_features_era5.csv, writes reports/mi3_era5_probe.csv
+ mi3_era5_probe.json.  The untagged frozen v2 table/feature sets are not modified.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                      # noqa: E402
from mi3_compare import make_xgb, dev_masks, DEV  # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HOR = B.HORIZONS
ERA5_TABLE = PROC / "mi_v2_features_era5.csv"
ERA5_SETS = PROC / "feature_sets_v2_era5.json"
PRE_SETS = PROC / "feature_sets_v2.json"


def main() -> int:
    t0 = time.time()
    assert ERA5_TABLE.exists(), f"missing {ERA5_TABLE} (run build_mi_features.py --tag era5)"
    d, fsets = B.load()
    era = pd.read_csv(ERA5_TABLE, parse_dates=["date"])
    d = d.merge(era, on=["station_id", "date"], validate="one_to_one")
    assert len(d) == 200665, len(d)

    fs_pre = json.loads(PRE_SETS.read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    fs_new = json.loads(ERA5_SETS.read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    e5_cols = [c for c in fs_new if c not in fs_pre]
    print(f"FULL={len(fsets['FULL'])}  FULL_V2_pre={len(fs_pre)}  "
          f"FULL_V2_era5={len(fs_new)}  (+{len(e5_cols)} new cols)")
    print("new cols:", e5_cols)

    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()

    rows = []
    for fs_name, feats in [("FULL", fsets["FULL"]),
                           ("FULL_V2_pre", fs_pre),
                           ("FULL_V2_era5", fs_new)]:
        print(f"\n=== feature set {fs_name} ({len(feats)} cols) ===")
        for h in HOR:
            y_all = d[h].to_numpy()
            tr = train & ~pd.isna(y_all)
            pipe = make_xgb(feats)
            t1 = time.time()
            pipe.fit(d.loc[tr, feats], y_all[tr])
            p_all = pipe.predict_proba(d[feats])[:, 1]
            print(f"  {h}: fit {time.time()-t1:.1f}s")
            thr = B.best_f1_threshold(y_all[tr], pipe.predict_proba(d.loc[tr, feats])[:, 1])
            for cell in DEV:
                m = masks[cell] & ~pd.isna(y_all)
                y, p = y_all[m], p_all[m]
                if len(np.unique(y)) < 2:
                    continue
                mm = B.metrics(y, p, thr)
                rows.append({"featureset": fs_name, "horizon": h, "cell": cell,
                             "n": int(m.sum()), "n_pos": int(y.sum()),
                             "pr_auc": mm["pr_auc"], "roc_auc": mm["roc_auc"],
                             "brier": mm["brier"], "ece": mm["ece"],
                             "recall": mm["recall"], "fn": mm["fn"]})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "mi3_era5_probe.csv", index=False)

    piv = out.pivot_table(index=["horizon", "cell"], columns="featureset",
                          values="pr_auc").round(4)
    piv["d_era5_vs_pre"] = (piv["FULL_V2_era5"] - piv["FULL_V2_pre"]).round(4)
    print("\n=== PR-AUC by cell ===")
    print(piv.to_string())

    def hmean(sub, fs):
        return round(float(sub[sub.featureset == fs].pr_auc.mean()), 5)

    v22 = out[out.cell == "val_2022"]
    summ = {
        "era5_new_cols": e5_cols,
        "n_new_cols": len(e5_cols),
        "by_horizon": {},
        "val_2022_horizon_mean": {
            "FULL": hmean(v22, "FULL"),
            "FULL_V2_pre": hmean(v22, "FULL_V2_pre"),
            "FULL_V2_era5": hmean(v22, "FULL_V2_era5"),
        },
        "all_dev_horizon_mean": {
            "FULL": hmean(out, "FULL"),
            "FULL_V2_pre": hmean(out, "FULL_V2_pre"),
            "FULL_V2_era5": hmean(out, "FULL_V2_era5"),
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    summ["val_2022_horizon_mean"]["delta_era5_vs_pre"] = round(
        summ["val_2022_horizon_mean"]["FULL_V2_era5"]
        - summ["val_2022_horizon_mean"]["FULL_V2_pre"], 5)
    for h in HOR:
        sub = out[out.horizon == h]
        summ["by_horizon"][h] = {
            "FULL": hmean(sub, "FULL"),
            "FULL_V2_pre": hmean(sub, "FULL_V2_pre"),
            "FULL_V2_era5": hmean(sub, "FULL_V2_era5"),
            "delta_era5_vs_pre": round(hmean(sub, "FULL_V2_era5")
                                       - hmean(sub, "FULL_V2_pre"), 5),
        }
    (REPORTS / "mi3_era5_probe.json").write_text(json.dumps(summ, indent=2),
                                                 encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summ, indent=2))
    print(f"\nWROTE {REPORTS/'mi3_era5_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
