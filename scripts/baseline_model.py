r"""Phase 16 — establish baseline models (workflow body step 28 / checklist stage 30).

Replaces the pre-Phase-15 version of this script, which had four defects recorded in
reports/phase15_split_design.md and reports/phase0_research_definition.md:
  1. LightGBM early-stopped ON THE TEST FOLD           -> no GBM in this phase at all
                                                         (candidate ML is Phase 17+), so
                                                         the flaw is structurally gone;
  2. trained on flood_dataset_model_ready.csv           -> now uses the Phase-14
                                                         experimental table (200,665 x 67);
  3. invented its own shuffled station folds            -> now uses Phase-15
                                                         split_assignments.csv (cohort-
                                                         balanced folds, holdout excluded);
  4. no persistence baseline                            -> D7 persistence (wl_pct_danger
                                                         alone) is fit and reported first.

Protocols (data/processed/splits.json):
  P1 temporal primary   train <2022 (holdout excluded) / val 2022 / test 2023+ (frozen)
  P2 rolling origin     train <=Y-1 -> val year Y for Y in {2020, 2021, 2022}
  P3 spatial CV         4 of 5 station folds, train-period rows, station-disjoint
  P5 season             every cell additionally split monsoon (JJAS) vs non-monsoon

Rules enforced here (phase0 decisions):
  D1  JJAS = monsoon
  D2  all three horizons reported equally
  D7  persistence = wl_pct_danger alone sits next to every model result
  D8  no pooled headline metric: per-year / per-bucket / per-season cells + station
      bootstrap CIs + macro (per-station) averages
  D10 body Phase order: this is Phase 16 only — no candidate ML, no tuning, no test-driven
      decision of any kind.

Test policy: the frozen test period is evaluated ONCE and written to its own file
(reports/phase16_test_metrics.csv). No feature, threshold, hyperparameter or model
decision in this phase reads it.

Run:  python -u scripts/baseline_model.py        (venv python, no PYTHONPATH needed —
                                                  this script imports no flood package)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):        # scripts have it; Jupyter's OutStream does not
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

HORIZONS = ["label_24h", "label_48h", "label_72h"]
TEXT_COLS = ["river", "basin", "division", "district", "upazilla"]
SEED = 20261008
ECE_BINS = 10
BOOT_REPS = 100
BOOT_MIN_ROWS = 2000
ROLLING_YEARS = [2020, 2021, 2022]
SPATIAL_FOLDS = [1, 2, 3, 4]
TEST_YEARS = [2023, 2024, 2025, 2026]
TEST_CUTOFF = 2023          # anything at or after this date is the frozen test period
HOLDOUT_CUTOFF = 2023       # first_year > this => "new-network" station


# --------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    exp = pd.read_csv(PROC / "experimental_dataset.csv", parse_dates=["date"])
    spl = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])
    d = exp.merge(spl, on=["station_id", "date"], validate="one_to_one")
    assert len(d) == len(exp) == len(spl), "merge changed the row count"

    # station names only for reporting (never a feature: feature matrices are built
    # from the explicit Phase-12 sets, never from "all remaining columns")
    st = pd.read_csv(PROC.parent / "raw" / "ffwc" / "ffwc_stations.csv",
                     usecols=["id", "name"]).rename(
        columns={"id": "station_id", "name": "station_name"})
    d = d.merge(st, on="station_id", how="left", validate="many_to_one")

    fsets = json.loads((PROC / "feature_sets.json").read_text(encoding="utf-8"))["sets"]
    banned = {"station_id", "station_name", "date", "label_24h", "label_48h",
              "label_72h", "split", "spatial_fold", "cohort", "first_year",
              "station_holdout", "season", "in_embargo_window"}
    assert not (set(fsets["FULL"]) & banned), "banned column in FULL"

    # structural guarantees of the split contract (phase15)
    tr = d.split == "train"
    va = d.split == "val"
    te = d.split == "test"
    assert not (set(d.loc[tr, "date"]) & set(d.loc[va, "date"])), "train/val share dates"
    assert not (set(d.loc[tr | va, "date"]) & set(d.loc[te, "date"])), "test not frozen"
    assert d.loc[te, "date"].min().year >= TEST_CUTOFF, "test period starts before 2023"
    assert d.loc[tr & ~d.station_holdout, "date"].max().year <= 2021, "train ends 2021"
    assert d.loc[va & ~d.station_holdout, "date"].dt.year.eq(2022).all(), "val is 2022"
    # phase15 stores holdout rows inside split train/val: every mask in build_tasks
    # must filter them, otherwise the frozen unseen-location claim is broken.
    # 21 of the 26 holdout stations have pre-2023 rows; 5 are new-network (2024+).
    assert d.loc[d.station_holdout & (tr | va), "station_id"].nunique() == 21

    print(f"loaded {len(d):,} rows x {d.shape[1]} cols "
          f"(train {(tr & ~d.station_holdout).sum():,} / val "
          f"{(va & ~d.station_holdout).sum():,} / test {te.sum():,}); "
          f"feature sets FULL {len(fsets['FULL'])} / LINEAR {len(fsets['LINEAR'])} / "
          f"SELECTED {len(fsets['SELECTED'])}")
    return d, fsets


# ------------------------------------------------------------------ preprocessing
class FreqEncoder(BaseEstimator, TransformerMixin):
    """Train-fitted frequency encoding of one text column (phase14 §6.4).

    Deliberately NOT target encoding (that would leak the label). Unseen or missing
    categories map to frequency 0.0 — the encoder never sees evaluation data.
    """

    def fit(self, X, y=None):
        col = X.columns[0]
        s = X[col].astype("object").where(X[col].notna(), "__NA__")
        vc = s.value_counts(normalize=True)
        self.map_ = vc.to_dict()
        self.default_ = 0.0
        return self

    def transform(self, X):
        col = X.columns[0]
        s = X[col].astype("object").where(X[col].notna(), "__NA__")
        return s.map(self.map_).fillna(self.default_).to_numpy().reshape(-1, 1)


def preprocessor(numeric: list[str], text: list[str], scale: bool) -> ColumnTransformer:
    steps: list[tuple] = []
    if numeric:
        num = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            num.append(("sc", StandardScaler()))
        steps.append(("num", Pipeline(num), numeric))
    for c in text:
        steps.append((f"freq_{c}", FreqEncoder(), [c]))
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=0.0)


def build_models(fsets: dict[str, list[str]]) -> dict[str, dict]:
    """model name -> {features, pipeline}.

    Phase 16 baselines only (workflow step 28): climatology, D7 persistence,
    logistic regression on the Phase-12 LINEAR set (the set phase12 §4 mandates for
    linear models), the same without the 67-70 %-NaN up_* columns (finding F14-2),
    a compact SELECTED cross-check, and one shallow decision tree.
    """
    full = fsets["FULL"]
    linear = fsets["LINEAR"]
    linear_no_up = [c for c in linear if not c.startswith("up_")]
    selected = fsets["SELECTED"]

    def num_of(cols):
        return [c for c in cols if c not in TEXT_COLS]

    def text_of(cols):
        return [c for c in cols if c in TEXT_COLS]

    def d7_pipeline():
        # D7 persistence: the ONE feature wl_pct_danger, logistic link.
        # A monotone link keeps the ranking identical to the raw score (phase5 D7
        # numbers), while still producing a valid probability for Brier/ECE.
        return Pipeline([
            ("pre", preprocessor(["wl_pct_danger"], [], scale=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
        ])

    return {
        # constant probability = training base rate; the "always predict climatology"
        # majority-class reference
        "climatology": {"features": [], "kind": "const"},
        # D7: one feature, wl_pct_danger alone
        "persistence": {"features": ["wl_pct_danger"], "kind": "pipe",
                        "pipe": d7_pipeline(), "d7": True},
        "logreg_linear": {
            "features": linear, "kind": "pipe",
            "pipe": Pipeline([
                ("pre", preprocessor(num_of(linear), text_of(linear), scale=True)),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
            ])},
        "logreg_linear_no_up": {
            "features": linear_no_up, "kind": "pipe",
            "pipe": Pipeline([
                ("pre", preprocessor(num_of(linear_no_up), text_of(linear_no_up), scale=True)),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
            ])},
        "logreg_selected": {
            "features": selected, "kind": "pipe",
            "pipe": Pipeline([
                ("pre", preprocessor(num_of(selected), text_of(selected), scale=True)),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
            ])},
        "tree_depth4": {
            "features": full, "kind": "pipe",
            "pipe": Pipeline([
                ("pre", preprocessor(num_of(full), text_of(full), scale=False)),
                ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=200,
                                               random_state=SEED)),
            ])},
    }


# ------------------------------------------------------------------------- metrics
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
    """Threshold from TRAIN predictions only (never the evaluation cell)."""
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 60)))
    best, bt = 0.0, float(cand[0]) if len(cand) else 0.5
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
    both = len(np.unique(y)) > 1
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": ece(y, p),
        "base_rate": float(y.mean()),
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
    }


def macro_pr_auc(stations: np.ndarray, y: np.ndarray, p: np.ndarray,
                 min_rows: int = 30) -> float:
    """Mean per-station PR-AUC (D8 macro companion to the pooled micro number)."""
    vals = []
    for s in pd.unique(stations):
        m = stations == s
        if m.sum() < min_rows:
            continue
        ys = y[m]
        if len(np.unique(ys)) < 2:
            continue
        vals.append(average_precision_score(ys, p[m]))
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_pr_auc(stations: np.ndarray, y: np.ndarray, p: np.ndarray,
                     reps: int = BOOT_REPS) -> tuple[float, float]:
    """Station-level bootstrap CI on PR-AUC (D8). Resamples stations, not rows."""
    if len(np.unique(stations)) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    uniq = np.unique(stations)
    idx_by_st = {s: np.flatnonzero(stations == s) for s in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_st[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    if len(vals) < reps // 2:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------------ protocols
def build_tasks(d: pd.DataFrame) -> list[dict]:
    """Each task = one training set + the evaluation cells scored with a model fit
    on it. Every mask is built from Phase-15 columns only."""
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()

    tasks = []

    # P1 temporal primary
    tr = (split == "train") & ~holdout
    evals = [("P1", "val_2022", (split == "val") & ~holdout)]
    for y in TEST_YEARS:
        evals.append(("P1", f"test_{y}", (split == "test") & (year == y)))
    is_test = split == "test"
    first_year = d.first_year.to_numpy()
    evals += [
        ("P1", "test_trained", is_test & ~holdout & (first_year <= HOLDOUT_CUTOFF)),
        ("P1", "test_new_network", is_test & ~holdout & (first_year > HOLDOUT_CUTOFF)),
        ("P1", "test_unseen", is_test & holdout),
    ]
    tasks.append({"protocol": "P1_temporal_primary", "name": "train_<2022",
                  "train": tr, "evals": evals})

    # P2 rolling origin
    for Y in ROLLING_YEARS:
        tr = (year <= Y - 1) & ~holdout
        tasks.append({"protocol": "P2_rolling_origin", "name": f"train_<={Y - 1}",
                      "train": tr,
                      "evals": [("P2", f"val_{Y}", (year == Y) & ~holdout)]})

    # P3 spatial CV (same train-period rows, station-disjoint)
    for k in SPATIAL_FOLDS:
        tr = (split == "train") & ~holdout & (fold != k)
        ev = (split == "train") & (fold == k)
        tasks.append({"protocol": "P3_spatial_cv", "name": f"excl_fold_{k}",
                      "train": tr, "evals": [("P3", f"spatial_fold_{k}", ev)]})

    # sanity: rolling 2022 train must equal P1 train (phase15 asserted this)
    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")["train"]
    r22 = next(t for t in tasks if t["protocol"] == "P2_rolling_origin"
               and t["name"] == "train_<=2021")["train"]
    assert np.array_equal(p1, r22), "rolling-2022 train != P1 train"
    return tasks


def score_cell(d: pd.DataFrame, mask: np.ndarray, season: str | None) -> np.ndarray:
    if season == "monsoon":
        return mask & (d.season.to_numpy() == "monsoon")
    if season == "non-monsoon":
        return mask & (d.season.to_numpy() == "non-monsoon")
    return mask


# ---------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    d, fsets = load()
    models = build_models(fsets)
    tasks = build_tasks(d)

    date = d.date.dt.year.to_numpy()
    stations = d.station_id.to_numpy()
    rows: list[dict] = []
    test_rows: list[dict] = []
    per_station: list[dict] = []
    calib: dict[str, tuple] = {}

    for task in tasks:
        tr_mask = task["train"]
        for mname, spec in models.items():
            for hor in HORIZONS:
                t1 = time.time()
                y_all = d[hor].to_numpy()
                tr_idx = np.flatnonzero(tr_mask & ~np.isnan(y_all))
                ytr = y_all[tr_idx].astype(int)
                if len(np.unique(ytr)) < 2:
                    continue

                # ---- fit on this task's training rows only
                base_rate = float(ytr.mean())
                if spec["kind"] == "const":
                    # constant probability = training base rate ("majority-class")
                    ptrain = np.full(len(tr_idx), base_rate)
                    thr = 0.5
                else:
                    pipe = spec["pipe"]
                    pipe.fit(d.iloc[tr_idx][spec["features"]], ytr)
                    ptrain = pipe.predict_proba(d.iloc[tr_idx][spec["features"]])[:, 1]
                    thr = best_f1_threshold(ytr, ptrain)
                    if spec.get("d7"):
                        # the D7 reference must rank exactly like the raw score
                        assert pipe.named_steps["clf"].coef_.ravel()[0] > 0, (
                            "D7 persistence fitted a negative coefficient — the link "
                            "is no longer monotone in wl_pct_danger")
                fit_s = time.time() - t1

                # ---- evaluation cells
                for protocol, cell, ev_mask in task["evals"]:
                    for season in (None, "monsoon", "non-monsoon"):
                        m = score_cell(d, ev_mask, season)
                        idx = np.flatnonzero(m & ~np.isnan(y_all))
                        if len(idx) == 0:
                            continue
                        y = y_all[idx].astype(int)
                        if spec["kind"] == "const":
                            p = np.full(len(idx), base_rate)
                        else:
                            p = spec["pipe"].predict_proba(
                                d.iloc[idx][spec["features"]])[:, 1]

                        met = metrics(y, p, thr)
                        is_test = cell.startswith("test")
                        if season is None and len(idx) >= 1000 and (
                                len(np.unique(stations[idx])) >= 10):
                            lo, hi = bootstrap_pr_auc(stations[idx], y, p)
                        else:
                            lo, hi = np.nan, np.nan
                        met.update(protocol=protocol, train=task["name"], cell=cell,
                                   season=season or "all", model=mname, horizon=hor,
                                   n=len(idx), train_rows=int(tr_idx.size),
                                   train_base_rate=base_rate,
                                   pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                                   macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                                   fit_s=round(fit_s, 2))
                        (test_rows if is_test else rows).append(met)

                        if (is_test and season is None and mname in
                                ("persistence", "logreg_linear")):
                            sub = pd.DataFrame({
                                "station_id": stations[idx],
                                "name": d["station_name"].to_numpy()[idx],
                                "y": y, "p": p})
                            sub["pred"] = (p >= thr).astype(int)
                            for (sid, nm), g in sub.groupby(["station_id", "name"]):
                                gy = g.y.to_numpy().astype(bool)
                                gp = g.pred.to_numpy().astype(bool)
                                per_station.append(dict(
                                    cell=cell, horizon=hor, model=mname,
                                    station_id=sid, name=nm, n=len(g),
                                    positives=int(gy.sum()),
                                    tp=int((gp & gy).sum()), fp=int((gp & ~gy).sum()),
                                    fn=int((~gp & gy).sum()), tn=int((~gp & ~gy).sum())))

                        if (is_test and season is None and cell == "test_2025" and
                                mname in ("persistence", "logreg_linear")):
                            calib[hor] = (mname, y, p, thr)

                print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                      f"{mname:19s} ({time.time() - t1:.1f}s)")

    met = pd.DataFrame(rows)
    met_test = pd.DataFrame(test_rows)
    met.to_csv(REPORTS / "phase16_baseline_metrics.csv", index=False)
    met_test.to_csv(REPORTS / "phase16_test_metrics.csv", index=False)
    if per_station:
        pd.DataFrame(per_station).to_csv(
            REPORTS / "phase16_test_per_station.csv", index=False)

    headline(met, met_test)
    plot_pr_skill(met, met_test)
    plot_calibration(calib)
    print(f"\nwrote reports/phase16_baseline_metrics.csv ({len(met)} rows), "
          f"phase16_test_metrics.csv ({len(met_test)} rows), "
          f"phase16_test_per_station.csv, figures/phase16_*.png   "
          f"[{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------------ reporting
def headline(met: pd.DataFrame, met_test: pd.DataFrame) -> None:
    """Console summary: val + spatial CV only (test stays in its own file)."""
    print("\n=== Phase 16 baselines — validation 2022 (selection cell, P1) ===")
    sub = met[(met.protocol == "P1") & (met.cell == "val_2022")
              & (met.season == "all")]
    if not sub.empty:
        piv = sub.pivot_table(index="model", columns="horizon",
                              values=["pr_auc", "recall", "brier"])
        print(piv.round(4).to_string())

    print("\n=== spatial CV (P3, unseen stations, train-period rows) ===")
    sub = met[(met.protocol == "P3") & (met.season == "all")]
    if not sub.empty:
        agg = (sub.groupby(["model", "horizon"])
               .agg(pr_auc=("pr_auc", "mean"), pr_lo=("pr_auc_ci_lo", "mean"),
                    recall=("recall", "mean"), brier=("brier", "mean"),
                    cells=("pr_auc", "size")).round(4))
        print(agg.to_string())

    print("\n=== rolling origin (P2, per year — D8: never pooled) ===")
    sub = met[(met.protocol == "P2") & (met.season == "all")]
    if not sub.empty:
        agg = (sub.groupby(["model", "cell", "horizon"])
               .agg(pr_auc=("pr_auc", "mean"), recall=("recall", "mean"),
                    base_rate=("base_rate", "mean")).round(4))
        print(agg.to_string())

    print("\n=== frozen test (2023+) — recorded once, no decision reads it ===")
    if not met_test.empty:
        agg = (met_test[met_test.season == "all"]
               .groupby(["model", "cell", "horizon"])
               .agg(pr_auc=("pr_auc", "mean"), lo=("pr_auc_ci_lo", "mean"),
                    hi=("pr_auc_ci_hi", "mean"), recall=("recall", "mean"),
                    precision=("precision", "mean"), base=("base_rate", "mean"),
                    n=("n", "sum")).round(4))
        print(agg.to_string())


def plot_pr_skill(met: pd.DataFrame, met_test: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = pd.concat([met, met_test])
    sub = sub[(sub.season == "all") & (sub.cell.str.startswith("test_20"))]
    if sub.empty:
        return
    cells = sorted(sub.cell.unique())
    models = [m for m in ["climatology", "persistence", "logreg_linear",
                          "logreg_linear_no_up", "logreg_selected", "tree_depth4"]
              if m in sub.model.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    x = np.arange(len(cells))
    for ax, hor in zip(axes, HORIZONS):
        s = sub[sub.horizon == hor]
        for i, m in enumerate(models):
            g = s[s.model == m].set_index("cell").reindex(cells)
            ax.plot(x, g.pr_auc, marker="o", label=m)
        ax.set_title(hor)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("test_", "") for c in cells], rotation=45)
        ax.set_ylabel("PR-AUC")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Phase 16 — PR-AUC per test year vs persistence bar (D7), season=all")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase16_pr_skill.png", dpi=130)
    plt.close(fig)


def plot_calibration(calib: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if not calib:
        return
    fig, axes = plt.subplots(1, len(calib), figsize=(4.6 * len(calib), 4.4))
    axes = np.atleast_1d(axes)
    for ax, hor in zip(axes, HORIZONS):
        if hor not in calib:
            continue
        mname, y, p, thr = calib[hor]
        frac, mean = calibration_curve(y, p, n_bins=ECE_BINS, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(mean, frac, "o-", label=f"{mname} (ECE={ece(y, p):.3f})")
        ax.set_xlabel("predicted P(flood)")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Calibration — test 2025, {hor}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase16_calibration_test2025.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
