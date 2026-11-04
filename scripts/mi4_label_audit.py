"""MI-4: label-quality audit.

Answers three questions, all from frozen artefacts (no model is trained here):
  A. Inventory & class-imbalance drift  -- base rate by split / year / season / station.
  B. Label noise & fragility            -- near-threshold crossings, nesting, the
                                           "24 h label == next-day exceedance" identity,
                                           observation-density label drop-out.
  C. Empirical ceiling                  -- the base-rate cap, and an inter-model
                                           disagreement floor using the cached Phase-28
                                           val_2022 probabilities (24 models x 3 horizons).

Discipline: every *model/ceiling* number is validation-only (`val_2022`, from
`reports/phase28_predictions.npz`). The test-period base rate is a descriptive label
count (already published by Phases 4/5 and 33); no test prediction enters any decision.

Outputs: `reports/mi4_base_rates.csv`, `reports/mi4_label_audit.json`.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B          # noqa: E402

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
HOR = B.HORIZONS
MIN_R = 3
DELTAS = [0.01, 0.02, 0.05, 0.10]                 # metres of water-level margin
# A 12-model panel spanning every model family (L1 linear ... L4 graph/transformer).
SKILL = ["xgboost", "lightgbm", "xgboost_selected", "random_forest", "gradient_boosting",
         "fusion_L3", "transformer_bestL", "lstm_L3", "gcn", "hybrid_L3",
         "tcn_bestL", "logreg_linear"]


def load_labels() -> pd.DataFrame:
    d, _ = B.load()
    return d.sort_values(["station_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------- A. inventory
def base_rates(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    yr = d.date.dt.year
    for h in HOR:
        y = d[h]

        def add(group: str, value, mask: pd.Series) -> None:
            lab = y[mask]
            nn = int(lab.notna().sum())
            rows.append({"horizon": h, "group": group, "value": str(value),
                         "rows": int(mask.sum()), "labelled": nn,
                         "n_pos": int((lab == 1).sum()),
                         "base_rate": float(lab.mean()) if nn else np.nan})

        add("ALL", "all", pd.Series(True, index=d.index))
        for sp in ("train", "val", "test"):
            add("split", sp, d.split == sp)
        for yv in sorted(yr.unique()):
            add("year", int(yv), yr == yv)
        if "season" in d:
            for sv in sorted(d.season.dropna().unique()):
                add("season", sv, d.season == sv)
    return pd.DataFrame(rows)


def station_spread(d: pd.DataFrame) -> dict:
    out = {}
    for h in HOR:
        g = d.dropna(subset=[h]).groupby("station_id")[h].agg(["mean", "size"])
        g = g[g["size"] >= 30]
        out[h] = {"stations_ge30": int(len(g)),
                  "min": float(g["mean"].min()), "p25": float(g["mean"].quantile(.25)),
                  "median": float(g["mean"].median()), "p75": float(g["mean"].quantile(.75)),
                  "max": float(g["mean"].max())}
    return out


# ----------------------------------------------------------------- B. noise/fragility
def _shift(d: pd.DataFrame, col: str, k: int) -> np.ndarray:
    return d.groupby("station_id")[col].shift(-k).to_numpy(dtype=float)


def fragility(d: pd.DataFrame) -> dict:
    d = d.copy()
    obs = d.n_readings.ge(MIN_R) & d.danger_level.notna()
    d["_ex"] = np.where(obs, (d.wl_max >= d.danger_level).astype(float), np.nan)
    d["_mg"] = (d.wl_max - d.danger_level).to_numpy(dtype=float)

    # observed station-days that sit within delta of the danger level
    od = d.loc[obs, "_mg"].abs()
    boundary = {f"{dl:.2f}": float((od <= dl).mean()) for dl in DELTAS}

    out = {"observed_day_boundary_share": boundary,
           "observed_days": int(obs.sum())}

    for h in (1, 2, 3):
        col = f"label_{24 * h}h"
        E = np.vstack([_shift(d, "_ex", k) for k in range(1, h + 1)])
        M = np.vstack([_shift(d, "_mg", k) for k in range(1, h + 1)])
        pos = (E == 1).any(axis=0)
        with np.errstate(invalid="ignore"):
            # smallest |margin| among the exceeding days of the window
            marg = np.where(E == 1, np.abs(M), np.inf).min(axis=0)
        y = d[col].to_numpy(dtype=float)
        npos = int((y == 1).sum())
        info = {"n_pos": npos,
                "pos_vs_window_mismatch": int((pos & (y == 0)).sum()
                                              + (~pos & (y == 1)).sum())}
        for dl in DELTAS:
            info[f"pos_margin_le_{dl:.2f}"] = int(((y == 1) & (marg <= dl)).sum())
            info[f"pos_share_le_{dl:.2f}"] = (float(((y == 1) & (marg <= dl)).sum() / npos)
                                              if npos else np.nan)
        out[col] = info
    return out


def nesting_identity(d: pd.DataFrame) -> dict:
    m = d[HOR].notna().all(axis=1)
    l24, l48, l72 = (d.loc[m, "label_24h"], d.loc[m, "label_48h"], d.loc[m, "label_72h"])
    # 24 h label == day t+1 exceedance (LEADS maps 24h -> a 1-day window)
    ex = np.where(d.n_readings.ge(MIN_R) & d.danger_level.notna(),
                  (d.wl_max >= d.danger_level).astype(float), np.nan)
    d2 = d.assign(_ex=ex)
    exn = d2.groupby("station_id")["_ex"].shift(-1).to_numpy(dtype=float)
    lab = d["label_24h"].to_numpy(dtype=float)
    cmp_mask = ~np.isnan(exn) & ~np.isnan(lab)
    return {"rows_all3_defined": int(m.sum()),
            "viol_24_implies_48": int(((l24 == 1) & (l48 == 0)).sum()),
            "viol_48_implies_72": int(((l48 == 1) & (l72 == 0)).sum()),
            "viol_24_implies_72": int(((l24 == 1) & (l72 == 0)).sum()),
            "identity_24h_nextday_comparable": int(cmp_mask.sum()),
            "identity_24h_nextday_match_rate": float((exn[cmp_mask] == lab[cmp_mask]).mean())}


def obs_density(d: pd.DataFrame) -> dict:
    out = {"grid_rows": int(len(d)),
           "n_readings_lt3_share": float((d.n_readings < MIN_R).mean()),
           "danger_missing_stations": int(d.loc[d.danger_level.isna(), "station_id"].nunique())}
    for h in HOR:
        nan_mask = d[h].isna()
        out[h] = {"label_nan_rows": int(nan_mask.sum()),
                  "label_nan_share": float(nan_mask.mean())}
    return out


# ------------------------------------------------------------- C. empirical ceiling
def ceiling() -> dict:
    z = np.load(REPORTS / "phase28_predictions.npz", allow_pickle=True)
    models = [str(m) for m in z["models"]]
    panel = [m for m in SKILL if m in models]
    out = {"panel": panel, "cells": {}}
    for h in HOR:
        y = z[f"y::{h}"].astype(float)
        P = np.vstack([z[f"p::{m}::{h}"] for m in panel])
        maxp, minp = P.max(0), P.min(0)
        med = np.median(P, 0)
        base = float(y.mean())
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        win = ~np.isnan(y)
        # a row is "unanimous" when ALL panel members agree on the extreme
        um = int(((y == 1) & (maxp < 0.01)).sum())          # no model sees the flood
        ufa = int(((y == 0) & (minp > 0.99)).sum())         # every model is sure, label says no
        agree = float((((P >= 0.5).all(0)) | ((P < 0.5).all(0))).mean())
        pr_med = float(B.metrics(y[win], med[win], 0.5)["pr_auc"])
        pr_xgb = float(B.metrics(y[win], z[f"p::xgboost::{h}"][win], 0.5)["pr_auc"])
        out["cells"][h] = {
            "n": int(win.sum()), "base_rate": base, "n_pos": npos, "n_neg": nneg,
            "unanimous_miss_pos": um, "unanimous_miss_share_of_pos": um / npos if npos else None,
            "unanimous_false_alarm_neg": ufa,
            "unanimous_fa_share_of_neg": ufa / nneg if nneg else None,
            "all_panel_agree_rate": agree,
            "recall_ceiling_from_unanimous_miss": 1 - (um / npos) if npos else None,
            "pr_auc_median_panel": pr_med, "pr_auc_xgboost": pr_xgb,
            "lift_median_over_base": pr_med / base if base else None}
    return out


def main() -> int:
    t0 = time.time()
    d = load_labels()
    br = base_rates(d)
    br.to_csv(REPORTS / "mi4_base_rates.csv", index=False)

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": B.SEED,
        "note": ("A/B read the immutable dataset (labels only). C is validation-only "
                 "(val_2022, Phase-28 cached probabilities). Test base rate is a "
                 "descriptive count; no test prediction is used."),
        "station_spread": station_spread(d),
        "nesting_identity": nesting_identity(d),
        "fragility": fragility(d),
        "obs_density": obs_density(d),
        "ceiling": ceiling(),
    }
    (REPORTS / "mi4_label_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # ---- console report -------------------------------------------------------
    print("=== A. base rates (by split) ===")
    print(br[br.group == "split"].to_string(index=False))
    print("\n=== A. base rates (by year) ===")
    print(br[br.group == "year"].to_string(index=False))
    print("\n=== B. nesting & identity ===")
    print(json.dumps(summary["nesting_identity"], indent=2))
    print("\n=== B. fragility ===")
    print(json.dumps(summary["fragility"], indent=2))
    print("\n=== B. observation density ===")
    print(json.dumps(summary["obs_density"], indent=2))
    print("\n=== C. ceiling (val_2022) ===")
    print(json.dumps(summary["ceiling"], indent=2))
    print(f"\nWROTE {REPORTS/'mi4_base_rates.csv'} and {REPORTS/'mi4_label_audit.json'}"
          f"  wall={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
