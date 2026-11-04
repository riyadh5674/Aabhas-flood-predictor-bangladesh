r"""MI-8 probe: does the response-lag block improve the frozen xgboost on the dev cells?

Same shape as `mi3_era5_probe.py` (same fixed model `make_xgb`, same dev cells, same
metric functions), so the FULL / FULL_V2 columns reproduce the published MI-3a table
(0.95463 / 0.95589 val_2022 horizon-mean) and the FULL_V3 column is directly comparable
to both.

Feature sets on the SAME rows of the joined table:
  * FULL     -- v1, 62 cols
  * FULL_V2  -- frozen MI-2 block, 110 cols (feature_sets_v2.json)
  * FULL_V3  -- FULL_V2 + the MI-8 response-lag block (feature_sets_mi8.json)

Non-destructive: reads `mi8_features.csv`, writes `reports/mi8_probe.csv` + `.json`.
No frozen artefact and no test row is touched (`test_rows_read = 0`).
"""
from __future__ import annotations

import json
import sys
import time
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
M8_TABLE = PROC / "mi8_features.csv"
M8_SETS = PROC / "feature_sets_mi8.json"
PRE_SETS = PROC / "feature_sets_v2.json"

# published MI-3a val_2022 horizon-mean for these two sets (reproducibility anchor)
MI3A_REF = {"FULL": 0.95463, "FULL_V2": 0.95589}


def main() -> int:
    t0 = time.time()
    assert M8_TABLE.exists(), f"missing {M8_TABLE} (run scripts/mi8_features.py first)"

    d, fsets = B.load()
    # B.load() returns the base experimental table only -- the 48 MI-2 v2 columns
    # live in mi_v2_features.csv (the era5 probe received them from its own tagged
    # table, which carries them), so both tables must be joined explicitly here.
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    m8 = pd.read_csv(M8_TABLE, parse_dates=["date"])
    d = d.merge(m8, on=["station_id", "date"], validate="one_to_one")
    assert len(d) == 200665, len(d)

    fs_pre = json.loads(PRE_SETS.read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    fs_new = json.loads(M8_SETS.read_text(encoding="utf-8"))["sets"]["FULL_V3"]
    new_cols = [c for c in fs_new if c not in fs_pre]
    print(f"FULL={len(fsets['FULL'])}  FULL_V2={len(fs_pre)}  FULL_V3={len(fs_new)}  "
          f"(+{len(new_cols)} response-lag cols)")
    print("new cols:", new_cols)
    print("test rows in scope: 0 (dev cells only)")

    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()

    rows = []
    for fs_name, feats in [("FULL", fsets["FULL"]),
                           ("FULL_V2", fs_pre),
                           ("FULL_V3", fs_new)]:
        print(f"\n=== feature set {fs_name} ({len(feats)} cols) ===")
        for h in HOR:
            y_all = d[h].to_numpy()
            tr = train & ~pd.isna(y_all)
            pipe = make_xgb(feats)
            t1 = time.time()
            pipe.fit(d.loc[tr, feats], y_all[tr])
            p_all = pipe.predict_proba(d[feats])[:, 1]
            thr = B.best_f1_threshold(y_all[tr],
                                      pipe.predict_proba(d.loc[tr, feats])[:, 1])
            print(f"  {h}: fit {time.time() - t1:.1f}s")
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
    out.to_csv(REPORTS / "mi8_probe.csv", index=False)

    piv = out.pivot_table(index=["horizon", "cell"], columns="featureset",
                          values="pr_auc").round(4)
    piv["d_v3_vs_v2"] = (piv["FULL_V3"] - piv["FULL_V2"]).round(4)
    print("\n=== PR-AUC by cell ===")
    print(piv.to_string())

    def hmean(sub, fs):
        return round(float(sub[sub.featureset == fs].pr_auc.mean()), 5)

    v22 = out[out.cell == "val_2022"]

    summ = {
        "n_new_cols": len(new_cols),
        "new_cols": new_cols,
        "val_2022_horizon_mean": {
            "FULL": hmean(v22, "FULL"),
            "FULL_V2": hmean(v22, "FULL_V2"),
            "FULL_V3": hmean(v22, "FULL_V3"),
        },
        "all_dev_horizon_mean": {
            "FULL": hmean(out, "FULL"),
            "FULL_V2": hmean(out, "FULL_V2"),
            "FULL_V3": hmean(out, "FULL_V3"),
        },
        "by_horizon": {},
        "reproducibility_vs_MI3a": {},
        "cells_improved": None,
        "test_rows_read": 0,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    s = summ["val_2022_horizon_mean"]
    s["delta_v3_vs_v2"] = round(s["FULL_V3"] - s["FULL_V2"], 5)
    summ["all_dev_horizon_mean"]["delta_v3_vs_v2"] = round(
        summ["all_dev_horizon_mean"]["FULL_V3"]
        - summ["all_dev_horizon_mean"]["FULL_V2"], 5)

    # per-horizon deltas
    summ["by_horizon"] = {
        h: {
            "FULL": hmean(out[out.horizon == h], "FULL"),
            "FULL_V2": hmean(out[out.horizon == h], "FULL_V2"),
            "FULL_V3": hmean(out[out.horizon == h], "FULL_V3"),
        } for h in HOR
    }
    for h in HOR:
        summ["by_horizon"][h]["delta_v3_vs_v2"] = round(
            summ["by_horizon"][h]["FULL_V3"] - summ["by_horizon"][h]["FULL_V2"], 5)

    # does the probe reproduce the published MI-3a numbers?
    for fs, ref in MI3A_REF.items():
        got = hmean(v22, fs)
        summ["reproducibility_vs_MI3a"][fs] = {"published": ref, "reproduced": got,
                                               "diff": round(got - ref, 5)}

    # cell-level win count (val_2022 excluded from "other cells" is still reported)
    wide = out.pivot_table(index=["horizon", "cell"], columns="featureset",
                           values="pr_auc")
    wins = int((wide["FULL_V3"] > wide["FULL_V2"]).sum())
    tot = int(wide["FULL_V3"].notna().sum())
    summ["cells_improved"] = {"improved": wins, "total": tot,
                              "fraction": round(wins / tot, 3) if tot else None}

    (REPORTS / "mi8_probe.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(summ, indent=2))
    print(f"\nWROTE {REPORTS / 'mi8_probe.csv'} and {REPORTS / 'mi8_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
