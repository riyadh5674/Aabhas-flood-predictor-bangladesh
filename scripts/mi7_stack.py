r"""MI-7 / Phase 35 — Track 5: does an ML+DL stacker beat the best single model?

The sprint's score play: if no single deep family clears the classical-ML
ceiling, a *blend* of the two families may, because they make different errors.

Data available (both genuinely out-of-sample on val_2022, both fit on train<2022):
  * DL  : reports/mi7_preds/<family>/<task>_val_2022.npz   (per-seed averaged)
  * ML  : reports/mi7_preds_ml/<family>_val_2022.npz       (Phase-20 P1 pipes)

Method (dev-only, honest caveat recorded):
  * Stack all available base probabilities per horizon into a design matrix.
  * Logistic regression on logit(p), fit with STATION-GROUPED 5-fold CV inside
    val_2022, so every stacked prediction comes from a model that never saw the
    row's station. This tests generalisation across locations, not calibration
    to the very rows used.
  * Report OOF PR-AUC of the stacker vs the best single base model, with a
    paired station-bootstrap of the delta.
  * CAVEAT: base hyper-parameters were selected on val_2022 (the P1 selection
    cell), so absolute numbers are mildly optimistic; the DELTA between stacker
    and best-single is the quantity this script is about, and it is measured on
    identical rows/folds.

Outputs
    reports/mi7_stack.csv        per-horizon OOF PR-AUC: stacker vs base models
    reports/mi7_stack_pairs.csv  paired bootstrap of stacker - best_single
    figures/mi7_stack.png

Run:  python -u scripts/mi7_stack.py
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
from baseline_model import HORIZONS, SEED                                   # noqa: E402
from sklearn.linear_model import LogisticRegression                          # noqa: E402
from sklearn.metrics import average_precision_score as _aps                  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures"
MLP = REPORTS / "mi7_preds_ml"
DLP = REPORTS / "mi7_preds"
N_SPLITS = 5


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _gather():
    """{horizon: DataFrame indexed by row idx, columns = model probabilities}."""
    cols = {}
    if MLP.exists():
        for p in sorted(MLP.glob("*_val_2022.npz")):
            d = np.load(p)
            cols[f"ml::{p.name.replace('_val_2022.npz','')}"] = (
                d["idx"].astype(np.int64), d["p"])
    if DLP.exists():
        for fd in sorted(DLP.iterdir()):
            if not fd.is_dir():
                continue
            fs = sorted(fd.glob("*val_2022.npz"))
            if fs:
                d = np.load(fs[0])
                cols[f"dl::{fd.name}"] = (d["ev_idx"].astype(np.int64), d["p"])
    return cols


def _stack_one(y, idxs, probs, names, stations):
    """GroupKFold-by-station OOF logistic stack; returns (oof, {name: auc})."""
    from sklearn.model_selection import GroupKFold
    X = np.column_stack([_logit(p) for p in probs])
    y = y.astype(int)
    oof = np.full(len(idxs), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(X, y, groups=stations):
        if len(np.unique(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    single = {name: float(_aps(y, p)) for name, p in zip(names, probs)}
    return oof, single


def main() -> None:
    cols = _gather()
    if not cols:
        print("no base predictions yet (need mi7_preds + mi7_preds_ml)"); return
    print("base models:", list(cols))
    stations_all = pd.read_csv(PROC / "sequence_index.csv",
                               usecols=["station_id"]).station_id.to_numpy()
    ix_full = pd.read_csv(PROC / "sequence_index.csv",
                          usecols=["station_id", "date"] + HORIZONS)
    rows, pairs = [], []
    pos = {name: pd.Series(np.arange(len(idx)), index=idx)
           for name, (idx, _) in cols.items()}
    for hi, h in enumerate(HORIZONS):
        idx_sets = [idx[np.isfinite(p[:, hi])] for idx, p in cols.values()]
        common = idx_sets[0]
        for s in idx_sets[1:]:
            common = np.intersect1d(common, s)
        if len(common) < 500:
            continue
        probs, names = [], []
        for name, (idx, p) in cols.items():
            a = pos[name].reindex(common).to_numpy()
            probs.append(p[a, hi]); names.append(name)
        ylab = ix_full[h].to_numpy()[common]
        st = stations_all[common]
        oof, single = _stack_one(ylab, common, probs, names, st)
        ok = np.isfinite(oof)
        stack_auc = float(_aps(ylab[ok].astype(int), oof[ok]))
        best_single = max(single, key=single.get)
        for name, a in single.items():
            rows.append(dict(horizon=h, kind="base", model=name, pr_auc=a,
                             n=int(ok.sum())))
        rows.append(dict(horizon=h, kind="stack_LR", model="ML+DL",
                         pr_auc=stack_auc, n=int(ok.sum())))
        ranks = np.column_stack([pd.Series(p[ok]).rank(pct=True).to_numpy()
                                 for p in probs])
        rows.append(dict(horizon=h, kind="mean_rank", model="all_bases",
                         pr_auc=float(_aps(ylab[ok].astype(int),
                                           ranks.mean(axis=1))),
                         n=int(ok.sum())))
        bi = names.index(best_single)
        r = paired_delta(st, ylab[ok], oof[ok], probs[bi][ok])
        pairs.append(dict(horizon=h, best_single=best_single,
                          single_auc=single[best_single], stack_auc=stack_auc,
                          delta=r["delta"], lo=r["lo"], hi=r["hi"],
                          frac_stack_gt=r["frac_a_gt_b"]))
        print(f"{h:10s} stack={stack_auc:.5f} best_single={best_single}="
              f"{single[best_single]:.5f} delta={r['delta']:+.5f} "
              f"[{r['lo']:+.5f},{r['hi']:+.5f}]")
    pd.DataFrame(rows).to_csv(REPORTS / "mi7_stack.csv", index=False)
    pd.DataFrame(pairs).to_csv(REPORTS / "mi7_stack_pairs.csv", index=False)
    _fig(pairs)
    print("wrote reports/mi7_stack.csv, reports/mi7_stack_pairs.csv, "
          "figures/mi7_stack.png")


def paired_delta(stations, y, pa, pb, reps=500, seed=SEED):
    from sklearn.metrics import average_precision_score as aps
    ok = np.isfinite(pa) & np.isfinite(pb) & ~np.isnan(y)
    y, pa, pb, st = y[ok].astype(int), pa[ok], pb[ok], stations[ok]
    order = np.argsort(st, kind="stable")
    _, start = np.unique(st[order], return_index=True)
    idxs = np.split(order, start[1:])
    if len(np.unique(y)) < 2:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, frac_a_gt_b=np.nan)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        ii = np.concatenate([idxs[j] for j in
                             rng.integers(0, len(idxs), len(idxs))])
        if len(np.unique(y[ii])) < 2:
            continue
        draws.append(aps(y[ii], pa[ii]) - aps(y[ii], pb[ii]))
    draws = np.array(draws)
    return dict(delta=float(aps(y, pa) - aps(y, pb)),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)),
                frac_a_gt_b=float((draws > 0).mean()))


def _fig(pairs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not pairs:
        return
    FIGS.mkdir(exist_ok=True)
    d = pd.DataFrame(pairs)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.single_auc, 0.4, label="best single")
    ax.bar(x + 0.2, d.stack_auc, 0.4, label="ML+DL stack")
    ax.set_xticks(x); ax.set_xticklabels(d.horizon)
    ax.set_ylabel("OOF PR-AUC (val_2022)")
    ax.set_title("Stacker vs best single model")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "mi7_stack.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
