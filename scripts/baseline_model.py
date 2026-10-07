"""Baseline models for the Bangladesh flood forecaster (plan.doc section on evaluation).

Protocol
  Data      data/processed/flood_dataset_model_ready.csv (rows with a knowable outcome)
  Features  everything EXCEPT labels, obs_days_* (future info), station_id, name, date, source
  Models    LogisticRegression (scaled + one-hot) | LightGBM | XGBoost
  Splits
    temporal folds   train on all days <= cutoff, test on the next calendar year
                     (no random shuffling: an early-warning system is scored on the future)
    station folds    5 shuffled station groups, train on 4 / test on 1
                     (leakage guard: can the model generalise to unseen rivers?)
  Metrics   ROC-AUC, PR-AUC (average precision), Brier, log loss, 10-bin ECE,
            confusion matrix at 0.5 and at the train-optimal-F1 threshold,
            per-station confusion (top stations by test rows)
  Output    reports/baseline_metrics.csv, reports/baseline_per_station.csv,
            reports/baseline_summary.md, figures/calibration_<horizon>.png

Run:  set PYTHONPATH=D:\Flood predictor\src
      python -u scripts/baseline_model.py
"""
from __future__ import annotations

import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
CATS = ["river", "basin", "division", "district", "upazilla"]
SEED = 20261007
N_ESTIMATORS = 400
ECE_BINS = 10


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(PROC / "flood_dataset_model_ready.csv", parse_dates=["date"])
    drop = {"station_id", "name", "date", "source", "obs_days_24h", "obs_days_48h",
            "obs_days_72h"} | set(HORIZONS)
    feats = [c for c in df.columns if c not in drop]
    num = [c for c in feats if c not in CATS]
    # categoricals must exist as columns for the one-hot path even if constant
    for c in CATS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    print(f"loaded {len(df):,} rows x {df.shape[1]} cols; {len(feats)} features "
          f"({len(num)} numeric, {len(CATS)} categorical)")
    return df, feats, num


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold on TRAIN predictions that maximises F1 (never looks at the test set)."""
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 60)))
    best, bt = 0.0, 0.5
    for t in cand:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.column_stack([1 - p, p])),
        "ece": ece(y, p),
        "base_rate": float(y.mean()),
        "threshold": thr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
    }


def build_models(num: list[str]):
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    pre_lr = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", ohe, CATS)], sparse_threshold=1.0)
    models = {
        "logreg": Pipeline([
            ("pre", pre_lr),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
            verbose=-1),
        "xgboost": xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.05, max_depth=7,
            min_child_weight=50, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric="logloss", random_state=SEED,
            n_jobs=-1, tree_method="hist"),
    }
    return models


def prepare_tree(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = df[feats].copy()
    for c in CATS:
        out[c] = out[c].astype("category")
    return out


def splits(df: pd.DataFrame):
    """Yield (split_name, train_idx, test_idx)."""
    # --- temporal: test = one calendar year at a time
    years = sorted(df["date"].dt.year.unique())
    test_years = [y for y in years if y >= 2022]
    for y in test_years:
        tr = df["date"].dt.year < y
        te = df["date"].dt.year == y
        if tr.sum() > 1000 and te.sum() > 500:
            yield f"temporal_{y}", np.flatnonzero(tr.to_numpy()), np.flatnonzero(te.to_numpy())
    # --- station groups: 5 shuffled folds over stations
    stations = sorted(df["station_id"].unique().tolist())
    random.Random(SEED).shuffle(stations)
    folds = [set(s) for s in np.array_split(np.array(stations), 5)]
    for k, held in enumerate(folds, 1):
        te = df["station_id"].isin(held).to_numpy()
        yield f"station_fold{k}", np.flatnonzero(~te), np.flatnonzero(te)


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df, feats, num = load()
    tree_df = prepare_tree(df, feats)
    models = build_models(num)

    rows = []
    per_station = []
    # calibration data for the figure: temporal fold nearest to "final" (latest test year)
    calib_store = {}

    for split_name, tr, te in splits(df):
        for hor in HORIZONS:
            y_all = df[hor].to_numpy()
            mask_tr = tr[~np.isnan(y_all[tr])]
            mask_te = te[~np.isnan(y_all[te])]
            ytr, yte = y_all[mask_tr], y_all[mask_te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                print(f"  skip {split_name}/{hor}: not both classes present")
                continue
            for mname, model in models.items():
                t1 = time.time()
                if mname == "logreg":
                    model.fit(df.iloc[mask_tr], ytr.astype(int))
                    ptr = model.predict_proba(df.iloc[mask_te])[:, 1]
                else:
                    model.fit(tree_df.iloc[mask_tr], ytr.astype(int),
                              eval_set=[(tree_df.iloc[mask_te], yte.astype(int))],
                              callbacks=[lgb.early_stopping(50, verbose=False)]
                              if mname == "lightgbm" else [xgb.EarlyStopping(rounds=50,
                                                                           save_best=True)])
                    ptr = model.predict_proba(tree_df.iloc[mask_te])[:, 1]
                # train-prediction threshold (approximated on the model's own train fit)
                if mname == "logreg":
                    ptrain = model.predict_proba(df.iloc[mask_tr])[:, 1]
                else:
                    ptrain = model.predict_proba(tree_df.iloc[mask_tr])[:, 1]
                thr = best_f1_threshold(ytr, ptrain)
                m = metrics(yte, ptr, thr)
                m.update(dict(split=split_name, horizon=hor, model=mname,
                              n_train=len(mask_tr), n_test=len(mask_te),
                              fit_s=round(time.time() - t1, 1)))
                rows.append(m)
                print(f"  {split_name:16s} {hor} {mname:10s} "
                      f"AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
                      f"Brier={m['brier']:.4f} ECE={m['ece']:.3f} "
                      f"recall={m['recall']:.3f} prec={m['precision']:.3f} "
                      f"({m['fit_s']}s)")

                # per-station confusion for the latest temporal fold only (feeds FN analysis)
                if split_name.startswith("temporal") and mname == "lightgbm":
                    sub = df.iloc[mask_te][["station_id", "name"]].copy()
                    sub["y"], sub["p"], sub["pred"] = yte, ptr, (ptr >= thr)
                    for (sid, nm), g in sub.groupby(["station_id", "name"]):
                        gy = g["y"].to_numpy().astype(bool)
                        gp = g["pred"].to_numpy()
                        per_station.append(dict(
                            split=split_name, horizon=hor, station_id=sid, name=nm,
                            n=len(g), positives=int(gy.sum()),
                            tp=int((gp & gy).sum()), fp=int((gp & ~gy).sum()),
                            fn=int((~gp & gy).sum()), tn=int((~gp & ~gy).sum())))
                    calib_store[hor] = (yte, ptr)

    met = pd.DataFrame(rows)
    met.to_csv(REPORTS / "baseline_metrics.csv", index=False)
    if per_station:
        ps = pd.DataFrame(per_station)
        ps.to_csv(REPORTS / "baseline_per_station.csv", index=False)

    write_summary(met, df)
    plot_calibration(calib_store)
    print(f"\nwrote reports/baseline_metrics.csv ({len(met)} rows), "
          f"reports/baseline_per_station.csv, reports/baseline_summary.md, "
          f"figures/calibration_*.png   [{time.time() - t0:.0f}s]")


def write_summary(met: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = ["# Baseline modelling results", ""]
    lines.append(f"Rows used: {len(df):,} model-ready station-days; "
                 f"{df.station_id.nunique()} stations; "
                 f"{df.date.min().date()} .. {df.date.max().date()}.")
    lines.append("")
    lines.append("Splits: `temporal_YYYY` = train on all earlier days, test on year YYYY "
                 "(chronological, no shuffle); `station_foldK` = 5 shuffled station groups, "
                 "train on 4 / test on held-out group (geographic generalisation guard).")
    lines.append("")
    for hor in HORIZONS:
        sub = met[met.horizon == hor]
        if sub.empty:
            continue
        lines.append(f"## {hor}")
        lines.append("")
        agg = (sub.groupby("model")
               .agg(roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
                    brier=("brier", "mean"), ece=("ece", "mean"),
                    recall=("recall", "mean"), precision=("precision", "mean"))
               .round(4).sort_values("pr_auc", ascending=False))
        lines.append("Mean over all folds:")
        lines.append("")
        lines.append(agg.to_markdown())
        lines.append("")
        worst = sub.sort_values("pr_auc").head(3)
        lines.append("Weakest folds (lowest PR-AUC):")
        lines.append("")
        lines.append(worst[["split", "model", "n_test", "roc_auc", "pr_auc",
                            "recall", "precision", "fn"]].round(4).to_markdown(index=False))
        lines.append("")
    lines.append("Caveats carried from the data layer: 2018 is a genuine FFWC archive gap; "
                 "MERRA-2 precipitation is reanalysis (modelled), not observed; tidal "
                 "stations (Mongla, Khulna, Daulatkhan) have legitimately high base rates "
                 "and must not be treated as labelling errors.")
    (REPORTS / "baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_calibration(calib_store: dict) -> None:
    if not calib_store:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for hor, (y, p) in calib_store.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        frac, mean = calibration_curve(y, p, n_bins=ECE_BINS, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(mean, frac, "o-", label=f"{hor} (ECE={ece(y, p):.3f})")
        ax.set_xlabel("predicted P(exceed)")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Calibration — latest temporal test fold, {hor}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"calibration_{hor}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
