r"""MI-7 / Phase 35 auxiliary — frozen-ML row-level predictions on dev cells.

WHY
---
The Phase-20 metric tables carry only aggregate PR-AUC. Track 3 (head-to-head)
and the ML+DL stacker both need *row-level* classical-ML predictions so that a
Delta-PR-AUC can be bootstrapped in a PAIRED way on identical rows. This script
regenerates them by applying the saved Phase-20 P1-primary pipelines to the dev
rows, in the exact row order of data/processed/sequence_index.csv (the order the
MI-7 sequence tensors use).

The Phase-20 P1 pipelines were fit on train<2022 only, so the val_2022 rows are
genuinely out-of-sample for them: the paired comparison is leakage-free. For
P2/P3 there is no clean P1-pipeline route (the rolling/spatial folds retrain per
task in Phase 20), so only val_2022 is materialised here; P2/P3 are compared at
the aggregate level from reports/phase20_metrics.csv in mi7_evaluate.py.

Outputs
    reports/mi7_preds_ml/{family}_val_2022.npz  {idx, y (n,3), p (n,3), threshold (3,)}

Run:  python -u scripts/mi7_mlpreds.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import joblib                                                   # noqa: E402
from baseline_model import HORIZONS, best_f1_threshold, load    # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "mi7_preds_ml"

# the three strongest Phase-20 P1 families (FULL-xgboost is the best ML here)
P1_PICKLES = {"xgboost": "L2-03", "lightgbm": "L2-04",
              "xgboost_selected": "L2-05"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    d, _ = load()

    key = pd.MultiIndex.from_frame(ix[["station_id", "date"]])
    d2 = d.set_index(["station_id", "date"])
    miss = key.difference(d2.index)
    assert len(miss) == 0, f"{len(miss)} sequence_index rows absent from table"
    d2 = d2.reindex(key)
    assert len(d2) == len(ix)
    for h in HORIZONS:                       # labels must be identical row-for-row
        assert np.nansum(np.abs(ix[h].to_numpy() - d2[h].to_numpy())) == 0, h

    hold = ix.station_holdout.astype(bool).to_numpy()
    year = ix.date.dt.year.to_numpy()
    split = ix.split.to_numpy()
    idx = np.flatnonzero((split == "val") & ~hold & (year == 2022))
    print(f"val_2022 rows: {len(idx):,}")

    for fam, rid in P1_PICKLES.items():
        s = joblib.load(MODELS / f"phase20_{rid}.pkl")
        feats = list(s["features"])
        missing = [c for c in feats if c not in d2.columns]
        assert not missing, f"{fam}: features absent {missing[:5]}"
        n = len(idx)
        P = np.full((n, 3), np.nan, dtype=np.float32)
        thr = np.full(3, np.nan)
        for hi, h in enumerate(HORIZONS):
            y_all = ix[h].to_numpy()
            keep = ~np.isnan(y_all[idx])
            pipe = s["horizons"][h]
            P[keep, hi] = pipe.predict_proba(
                d2.iloc[idx[keep]][feats])[:, 1]
            tri = np.flatnonzero((split == "train") & ~hold
                                 & ~np.isnan(y_all))
            ptr = pipe.predict_proba(d2.iloc[tri][feats])[:, 1]
            thr[hi] = best_f1_threshold(y_all[tri].astype(int), ptr)
        y = np.column_stack([ix[h].to_numpy()[idx] for h in HORIZONS])
        np.savez_compressed(OUT / f"{fam}_val_2022.npz",
                            idx=idx.astype(np.int64), y=y.astype(np.float32),
                            p=P, threshold=thr)
        print(f"  {fam:20s} set={s['feature_set']:8s} n={n:,} "
              f"nonnan24={int(np.isfinite(P[:, 0]).sum()):,} "
              f"thr={np.round(thr, 4)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
