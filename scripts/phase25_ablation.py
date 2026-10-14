r"""Phase 25.2 ablation - temporal Transformer vs the temporal baselines (paired).

Gate 25.1 pre-registered criterion: the Transformer (L4-03) must beat the temporal
baselines (LSTM L3-01, TCN L3-06) on val_2022 (primary) and must not lose to them on
P3 spatial CV (secondary), with a paired station-bootstrap CI on the difference.

Each model's search winner is re-fit on the *same* cells (identical batching/seed)
and the per-sample predictions are differenced under a station-resampling bootstrap.

Outputs:
    reports/phase25_ablation.csv        per cell: pr_auc, diff, CI (transformer - base)
    reports/phase25_ablation_preds.npz  per-sample y/pred/station for audit

Run:
    python -u scripts/phase25_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import train_sequence as ts                                    # noqa: E402
import train_transformer as T                                  # noqa: E402
from baseline_model import HORIZONS, SEED                      # noqa: E402

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ABL_CSV = REPORTS / "phase25_ablation.csv"
ABL_NPZ = REPORTS / "phase25_ablation_preds.npz"
N_BOOT = 1000


def paired_bootstrap(st, y, pa, pb, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(st)
    idx_by_st = {s: np.flatnonzero(st == s) for s in uniq}
    diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_st[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(average_precision_score(y[idx], pa[idx])
                     - average_precision_score(y[idx], pb[idx]))
    d = np.array(diffs)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def winner_cfg(path: Path, family: str) -> tuple[dict, dict, int]:
    s = pd.read_csv(path)
    sub = s[s.family == family].sort_values("mean_pr_auc", ascending=False)
    row = sub.iloc[0]
    cfg = json.loads(row["params"])
    cfg["_weight"] = "on" if row["weighting"].startswith("pos") else "off"
    ep = json.loads(row["best_epochs"])
    L = int(json.loads(row["params"]).get("L", 3)) if "L" in row["params"] else 3
    return cfg, ep, L


def fit(kind: str, cfg: dict, L: int, X, M, mu, sd, tr, ytr, ev, yev, fixed):
    if kind == "transformer":
        r = T.fit_transformer(cfg, X, M, mu, sd, tr, ytr, ev, yev, fixed_epochs=fixed)
        return r["pred"]
    r = ts.fit_model(kind, L, cfg, X, M, mu, sd, tr, ytr, ev, yev, fixed_epochs=fixed)
    return r["pred"]


def main() -> None:
    t0 = time.time()
    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    st_id = ix.station_id.to_numpy()
    season_v = ix.season.to_numpy()
    tasks = ts.build_tasks_index(ix)
    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    p3 = [t for t in tasks if t["protocol"] == "P3_spatial_cv"]

    tx_cfg, tx_ep, tx_L = winner_cfg(REPORTS / "phase25_search.csv", "transformer_bestL")
    lc_cfg, lc_ep, lc_L = winner_cfg(REPORTS / "phase21_search.csv", "lstm_L3")
    tc_cfg, tc_ep, tc_L = winner_cfg(REPORTS / "phase22_search.csv", "tcn_bestL")
    print(f"transformer winner L={tx_L} cfg={tx_cfg} epochs={tx_ep}")
    print(f"lstm_L3 winner cfg={lc_cfg} epochs={lc_ep}")
    print(f"tcn winner cfg={tc_cfg} epochs={tc_ep}")

    Xc: dict[int, tuple] = {}

    def data(L):
        if L not in Xc:
            Xc[L] = ts.prepare(L)
        return Xc[L]

    MODELS = [("transformer", "transformer", tx_cfg, tx_ep, tx_L),
              ("lstm_L3", "lstm", lc_cfg, lc_ep, lc_L),
              ("tcn_bestL", "tcn", tc_cfg, tc_ep, tc_L)]

    rows, store = [], {}
    for name, kind, cfg, ep, L in MODELS:
        X, M = data(L)
        # ---- P1 val_2022 ----
        for hor in HORIZONS:
            yv = labels[hor]
            tr = np.flatnonzero(p1["train"] & ~np.isnan(yv))
            ev = np.flatnonzero(next(m for _, c, m in p1["evals"] if c == "val_2022")
                                & ~np.isnan(yv))
            mu, sd = ts.fold_stats(X, M, tr)
            p = fit(kind, cfg, L, X, M, mu, sd, tr, yv[tr], ev, yv[ev],
                    int(ep[hor[6:]]))
            store[f"P1_{name}_{hor[6:]}"] = p
            print(f"  P1 {name} {hor[6:]} ap={average_precision_score(yv[ev].astype(int), p):.5f}")
        # ---- P3 folds ----
        for task in p3:
            _, cell, ev_mask = task["evals"][0]
            for hor in HORIZONS:
                yv = labels[hor]
                tr = np.flatnonzero(task["train"] & ~np.isnan(yv))
                ev = np.flatnonzero(ev_mask & ~np.isnan(yv))
                mu, sd = ts.fold_stats(X, M, tr)
                p = fit(kind, cfg, L, X, M, mu, sd, tr, yv[tr], ev, yv[ev],
                        int(ep[hor[6:]]))
                store[f"P3_{name}_{cell}_{hor[6:]}"] = p
        print(f"  P3 {name} done")

    # ---- pairwise (transformer vs each baseline) ----
    def collect(proto, name, cell, hor):
        if proto == "P1":
            yv = labels[hor]
            ev = next(m for _, c, m in p1["evals"] if c == "val_2022") & ~np.isnan(yv)
            return yv[ev].astype(int), st_id[ev], store[f"P1_{name}_{hor[6:]}"]
        yv = labels[hor]
        task = next(t for t in p3 if t["evals"][0][1] == cell)
        ev = task["evals"][0][2] & ~np.isnan(yv)
        return yv[ev].astype(int), st_id[ev], store[f"P3_{name}_{cell}_{hor[6:]}"]

    for base in ("lstm_L3", "tcn_bestL"):
        # P1 per horizon
        for hor in HORIZONS:
            y, st, pt = collect("P1", "transformer", "val_2022", hor)
            _, _, pb = collect("P1", base, "val_2022", hor)
            d, lo, hi = paired_bootstrap(st, y, pt, pb)
            rows.append({"comparison": f"transformer_vs_{base}", "protocol": "P1",
                         "cell": "val_2022", "horizon": hor[6:], "n": len(y),
                         "pr_auc_transformer": average_precision_score(y, pt),
                         "pr_auc_base": average_precision_score(y, pb),
                         "diff": d, "ci_lo": lo, "ci_hi": hi})
            print(f"  P1 {hor[6:]} tx-vs-{base}: {average_precision_score(y,pt):.5f} vs "
                  f"{average_precision_score(y,pb):.5f} diff={d:+.5f} [{lo:+.5f},{hi:+.5f}]")
        # P3 pooled over folds
        for hor in HORIZONS:
            ys, sts, ps, pbs = [], [], [], []
            for task in p3:
                _, cell, _ = task["evals"][0]
                y, st, pt = collect("P3", "transformer", cell, hor)
                _, _, pb = collect("P3", base, cell, hor)
                ys.append(y); sts.append(st); ps.append(pt); pbs.append(pb)
            y = np.concatenate(ys); st = np.concatenate(sts)
            pt = np.concatenate(ps); pb = np.concatenate(pbs)
            d, lo, hi = paired_bootstrap(st, y, pt, pb)
            rows.append({"comparison": f"transformer_vs_{base}", "protocol": "P3",
                         "cell": "pooled_folds", "horizon": hor[6:], "n": len(y),
                         "pr_auc_transformer": average_precision_score(y, pt),
                         "pr_auc_base": average_precision_score(y, pb),
                         "diff": d, "ci_lo": lo, "ci_hi": hi})
            print(f"  P3 {hor[6:]} tx-vs-{base}: {average_precision_score(y,pt):.5f} vs "
                  f"{average_precision_score(y,pb):.5f} diff={d:+.5f} [{lo:+.5f},{hi:+.5f}]")

    pd.DataFrame(rows).to_csv(ABL_CSV, index=False)
    np.savez_compressed(ABL_NPZ, **store)
    print(f"\nwrote {ABL_CSV.name}, {ABL_NPZ.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
