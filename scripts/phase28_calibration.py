"""Phase 28 — probability calibration (stage 47).

Deliverables
------------
* reports/phase28_calibration.csv   calibrated-vs-raw table (24 models x 3 horizons x 3 seasons)
* reports/phase28_summary.json      decision: which models may proceed uncalibrated
* reports/phase28_predictions.npz   cached P1 val_2022 probabilities (evidence / fast re-runs)
* figures/phase28_reliability.png   reliability diagrams (raw vs recalibrated)

Method
------
For every registered model the **same** P1 predictor used in Phase 27
(`evaluate_all.predict_one`) produces the `val_2022` probabilities; calibration is then
assessed with the shared metric code (`baseline_model.ece`).  Two recalibrators are compared:

* **Platt** (sigmoid) — logistic regression on the logit of the model probability;
* **isotonic** — monotone piecewise-constant regression.

Both are fitted **on validation only** (never test), as the workflow requires.  Because
fitting and scoring on the same rows is optimistic, the headline calibrated numbers are
**cross-fitted**: the 84 `val_2022` stations are split into 5 folds and each fold is calibrated
by a calibrator fitted on the other four (out-of-fold, OOF).  In-sample numbers are also
reported as an optimistic upper bound.  Test is never read (`test_rows_read = 0`).
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
import evaluate_all as E                         # noqa: E402

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.isotonic import IsotonicRegression              # noqa: E402
from sklearn.metrics import brier_score_loss                 # noqa: E402
from sklearn.metrics import log_loss                         # noqa: E402

ROOT = E.ROOT
REPORTS = E.REPORTS
FIGURES = E.FIGURES
HOR = E.HOR
SEASONS = E.SEASONS
CATALOG = E.CATALOG
SEED = B.SEED
HARDWARE = E.HARDWARE

TOL_ECE = 0.02          # "good calibration" tolerance (a model may proceed uncalibrated)
K_FOLDS = 5             # station folds for cross-fitted calibration
RNG = np.random.default_rng(SEED)
_EPS = 1e-6


# ------------------------------------------------------------------- calibrators
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def platt_fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(x.reshape(-1, 1), y.astype(int))
    return clf


def platt_pred(clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x.reshape(-1, 1))[:, 1]


def iso_fit(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x, y.astype(float))
    return ir


def cal_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pc = np.clip(p, _EPS, 1 - _EPS)
    return {
        "brier": float(brier_score_loss(y, pc)),
        "ece": float(B.ece(y, pc)),
        "logloss": float(log_loss(y, np.column_stack([1 - pc, pc]), labels=[0, 1])),
    }


def oof_calibrate(p: np.ndarray, y: np.ndarray, station: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold Platt + isotonic predictions over station folds."""
    u = np.unique(station)
    perm = RNG.permutation(len(u))
    fold_of = {s: int(perm[i] % K_FOLDS) for i, s in enumerate(u)}
    f = np.array([fold_of[s] for s in station])
    pl = np.full_like(p, np.nan, dtype=float)
    iso = np.full_like(p, np.nan, dtype=float)
    for k in range(K_FOLDS):
        tr, te = f != k, f == k
        if tr.sum() < 20 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        pl[te] = platt_pred(platt_fit(logit(p[tr]), y[tr]), logit(p[te]))
        iso[te] = iso_fit(p[tr], y[tr]).predict(p[te])
    return pl, iso


# --------------------------------------------------------------- predictions
def load_or_build_predictions(data, p1_eval) -> dict:
    cache = REPORTS / "phase28_predictions.npz"
    names = [c["model"] for c in CATALOG]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["models"]) == names and list(z["hors"]) == list(HOR):
            print(f"[cache] reusing P1 val_2022 probabilities ({cache.name})")
            return {k: z[k] for k in z.files}
    store: dict[str, np.ndarray] = {}
    for hor in HOR:
        y = data.d[hor].to_numpy()
        evi = np.flatnonzero(p1_eval & ~np.isnan(y))
        store[f"y::{hor}"] = y[evi].astype(int)
        store[f"st::{hor}"] = data.station[evi]
        store[f"se::{hor}"] = data.season[evi].astype(str)
        store[f"dt::{hor}"] = data.date[evi]
    for ci, cat in enumerate(CATALOG):
        name = cat["model"]
        t1 = time.time()
        for hor in HOR:
            y = data.d[hor].to_numpy()
            tri = np.flatnonzero(data.p1_train & ~np.isnan(y))
            evi = np.flatnonzero(p1_eval & ~np.isnan(y))
            _, p_ev, thr = E.predict_one(data, cat, tri, evi, hor)
            store[f"p::{name}::{hor}"] = p_ev.astype(float)
            store[f"thr::{name}::{hor}"] = np.array([thr], float)
        print(f"  predicted {ci + 1:2d}/{len(CATALOG)} {name:20s} ({time.time() - t1:.1f}s)")
    store["models"] = np.array(names, dtype=object)
    store["hors"] = np.array(list(HOR), dtype=object)
    np.savez_compressed(cache, **store)
    print(f"wrote {cache.name} ({(cache.stat().st_size) / 1e6:.1f} MB)")
    return store


# --------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data = E.Data()

    split = data.d.split.to_numpy()
    hold = data.d.station_holdout.to_numpy()
    year = data.d.date.dt.year.to_numpy()
    data.p1_train = (split == "train") & ~hold
    p1_eval = (split == "val") & ~hold & (year == 2022)
    assert not (p1_eval & (split == "test")).any()

    print(f"Phase 28 — probability calibration | {len(CATALOG)} models | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')}")
    print(f"P1 train rows={int(data.p1_train.sum())}  val_2022 rows={int(p1_eval.sum())}  "
          f"hardware={HARDWARE}")

    store = load_or_build_predictions(data, p1_eval)

    rows: list[dict] = []
    for cat in CATALOG:
        name = cat["model"]
        for hor in HOR:
            y = store[f"y::{hor}"]
            p = store[f"p::{name}::{hor}"]
            st = store[f"st::{hor}"]
            se = store[f"se::{hor}"]
            for season in SEASONS:
                m = np.ones(len(y), bool) if season == "all" else (se == season)
                yy, pp, ss = y[m], p[m], st[m]
                if len(yy) == 0 or len(np.unique(yy)) < 2:
                    continue
                raw = cal_metrics(yy, pp)
                # cross-fitted + in-sample recalibration
                pl_oof, iso_oof = oof_calibrate(pp, yy, ss)
                pl_in = platt_pred(platt_fit(logit(pp), yy), logit(pp))
                iso_in = iso_fit(pp, yy).predict(pp)
                cm = {}
                for tag, arr in (("platt_oof", pl_oof), ("iso_oof", iso_oof),
                                 ("platt_in", pl_in), ("iso_in", iso_in)):
                    mm = ~np.isnan(arr)
                    cm[tag] = cal_metrics(yy[mm], arr[mm]) if mm.any() else \
                        {"brier": np.nan, "ece": np.nan, "logloss": np.nan}
                # best OOF method
                cand = {"platt": cm["platt_oof"], "isotonic": cm["iso_oof"]}
                best = min(cand, key=lambda k: (np.inf if np.isnan(cand[k]["brier"])
                                                else cand[k]["brier"]))
                rows.append(dict(
                    model=name, level=cat["level"], registry_id=cat["rid"],
                    horizon=hor, season=season, n=int(len(yy)), n_pos=int(yy.sum()),
                    base_rate=float(yy.mean()), mean_pred=float(pp.mean()),
                    brier_raw=raw["brier"], ece_raw=raw["ece"], logloss_raw=raw["logloss"],
                    ece_platt_insample=cm["platt_in"]["ece"],
                    ece_iso_insample=cm["iso_in"]["ece"],
                    brier_platt_oof=cm["platt_oof"]["brier"], ece_platt_oof=cm["platt_oof"]["ece"],
                    brier_iso_oof=cm["iso_oof"]["brier"], ece_iso_oof=cm["iso_oof"]["ece"],
                    best_method_oof=best, best_brier_oof=cand[best]["brier"],
                    best_ece_oof=cand[best]["ece"],
                    delta_brier_oof=raw["brier"] - cand[best]["brier"],
                    delta_ece_oof=raw["ece"] - cand[best]["ece"],
                ))
        print(f"  calibrated {name}")

    cal = pd.DataFrame(rows)

    # decision: cleared if raw ECE <= TOL_ECE on every horizon (all-season)
    allc = cal[cal.season == "all"]
    worst = allc.groupby("model").ece_raw.max()
    cleared = (worst <= TOL_ECE)
    cal["cleared_all"] = cal.model.map(cleared)

    cal.to_csv(REPORTS / "phase28_calibration.csv", index=False)
    print(f"wrote reports/phase28_calibration.csv ({len(cal)} rows)")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = ["logreg_linear", "xgboost", "fusion_L3",
               "transformer_bestL", "lstm_L3", "static_only"]
        bins = np.linspace(0, 1, 11)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for j, hor in enumerate(HOR):
            axr, axc = axes[0, j], axes[1, j]
            for name in sel:
                y = store[f"y::{hor}"]
                p = store[f"p::{name}::{hor}"]
                st = store[f"st::{hor}"]
                pl, iso = oof_calibrate(p, y, st)
                for ax, q in ((axr, p), (axc, iso)):
                    idx = np.clip(np.digitize(q, bins) - 1, 0, len(bins) - 2)
                    xb, yb = [], []
                    for b in range(len(bins) - 1):
                        mmn = idx == b
                        if mmn.any():
                            xb.append(q[mmn].mean()); yb.append(y[mmn].mean())
                    ax.plot(xb, yb, marker="o", ms=3, lw=1, label=name)
            for ax, t in ((axr, "raw"), (axc, "isotonic (OOF)")):
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3)
                ax.set_title(f"{hor[6:]} — {t}")
                ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
            axr.legend(fontsize=7, loc="upper left")
        fig.suptitle("Phase 28 — reliability diagrams (P1 val_2022, season=all)")
        fig.tight_layout()
        fig.savefig(FIGURES / "phase28_reliability.png", dpi=130)
        print("wrote figures/phase28_reliability.png")
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)

    # ------------------------------------------------------------------ summary
    summary = {
        "phase": 28, "stage": 47, "seed": SEED, "hardware": HARDWARE,
        "n_models": len(CATALOG), "n_rows": int(len(cal)),
        "tol_ece": TOL_ECE, "k_folds": K_FOLDS, "test_rows_read": 0,
        "cleared_uncalibrated": sorted(cleared[cleared].index.tolist()),
        "needs_recalibration": sorted(worst[~cleared].sort_values(ascending=False)
                                      .round(4).to_dict().items(),
                                      key=lambda kv: -kv[1]),
        "worst_raw_ece_all_season": worst.sort_values(ascending=False).round(4).to_dict(),
        "median_raw_ece_all_season": float(worst.median()),
        "median_best_oof_ece_all_season": float(
            allc.groupby("model").best_ece_oof.median().median()),
    }
    (REPORTS / "phase28_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Phase 28 calibration summary ({time.time() - t0:.0f}s) ===")
    print(f"tolerance ECE <= {TOL_ECE} on every horizon (all-season):")
    print(f"  cleared uncalibrated: {len(summary['cleared_uncalibrated'])}/{len(CATALOG)}")
    for name, v in summary["needs_recalibration"]:
        print(f"  needs recalibration: {name} (worst raw ECE {v})")
    print(f"median raw ECE = {summary['median_raw_ece_all_season']:.4f} | "
          f"median best OOF ECE = {summary['median_best_oof_ece_all_season']:.4f}")
    print(f"\nwrote reports/phase28_calibration.csv ({len(cal)} rows), "
          f"phase28_summary.json  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
