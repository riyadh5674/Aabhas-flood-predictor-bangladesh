r"""Phase 24.2 ablation - GNN vs temporal-only on P3 (unseen stations), paired.

The registered kill criterion (reports/phase24_graph_gate.md): the GCN (L4-01)
and an attention variant (L4-02) must beat the temporal-only LSTM (L3-01) on P3
spatial-CV PR-AUC with a paired station-level bootstrap CI.  This script trains
each family's search winner AND the L3-01 winner on the *same* P3 fold x horizon
cells (identical batching/seed for each), then differences the per-sample
predictions under a station-resampling bootstrap.

It also re-evaluates the GNN winner on the river-only and identity (self-loop)
graphs to separate "message passing" from "added depth".

Outputs:
    reports/phase24_ablation.csv        per cell: pr_auc gnn vs lstm, diff, CI
    reports/phase24_ablation_preds.npz  per-sample y/pred/station for audit

Run:
    python -u scripts\phase24_ablation.py gcn gat
    python -u scripts\phase24_ablation.py gcn gat --variants=full   (registered only)

The graph-independent L3-01 LSTM is fit once per (cell, horizon) and reused
across families.  --variants limits the graph variants (default full,river,id);
the registered kill criterion needs only "full".
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

import train_sequence as ts                              # noqa: E402
import train_gnn as g                                    # noqa: E402
from baseline_model import HORIZONS, SEED      # noqa: E402

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ABL_CSV = REPORTS / "phase24_ablation.csv"
ABL_NPZ = REPORTS / "phase24_ablation_preds.npz"
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
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), \
        float(np.percentile(diffs, 97.5))


def build_graph(sub, sp_index, n):
    adj = np.zeros((n, n), np.float32)
    for a, b in zip(sub.station_id_a, sub.station_id_b):
        i, j = sp_index[int(a)], sp_index[int(b)]
        adj[i, j] = adj[j, i] = 1.0
    tilde = adj + np.eye(n, dtype=np.float32)
    deg = tilde.sum(1)
    d = np.diag(1.0 / np.sqrt(deg))
    return (torch.tensor(d @ tilde @ d, dtype=torch.float32),
            torch.tensor((adj + np.eye(n)) > 0))


def winner_cfg(family, search):
    sub = search[search.family == family].sort_values("mean_pr_auc", ascending=False)
    row = sub.iloc[0]
    cfg = json.loads(row["params"])
    cfg["_weight"] = "on" if row["weighting"].startswith("pos") else "off"
    epochs = json.loads(row["best_epochs"])
    return cfg, epochs, row["config_id"]


def main() -> None:
    args = sys.argv[1:]
    fams, variants = [], None
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--variants="):
            variants = a.split("=", 1)[1].split(",")
        elif a == "--variants":
            i += 1
            variants = args[i].split(",")
        elif not a.startswith("--"):
            fams.append(a)
        i += 1
    fams = fams or ["gcn", "gat"]
    variants = variants or ["full", "river", "id"]
    t0 = time.time()
    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    dp, sp, row_of, sp_index = g.build_index_maps(ix)
    n_st = len(sp_index)
    a_hat, a_mask = g.load_graph(sp_index)
    graph = pd.read_csv(ROC / "station_graph.csv")
    VAR = {"full": (a_hat, a_mask),
           "river": build_graph(graph[graph.kind == "river"], sp_index, n_st),
           "id": build_graph(graph.iloc[0:0], sp_index, n_st)}
    L = int(json.loads((ROC / "phase21_lstar.json").read_text())["L_star"])
    X, M = ts.prepare(L)
    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    tasks = ts.build_tasks_index(ix)
    p3 = [t for t in tasks if t["protocol"] == "P3_spatial_cv"]
    st_id = ix.station_id.to_numpy()

    search24 = pd.read_csv(REPORTS / "phase24_search.csv")
    search21 = pd.read_csv(REPORTS / "phase21_search.csv")
    l3 = search21[search21.family == "lstm_L3"].sort_values(
        "mean_pr_auc", ascending=False).iloc[0]
    l3cfg = json.loads(l3["params"]); l3cfg["_weight"] = \
        "on" if l3["weighting"].startswith("pos") else "off"
    l3ep = json.loads(l3["best_epochs"])
    print(f"L3-01 winner {l3['config_id']} mean={l3['mean_pr_auc']:.5f}")
    print(f"families={fams} variants={variants}")

    rows, store = [], {}
    lstm_cache = {}
    for family in fams:
        cfg, ep, cid = winner_cfg(family, search24)
        print(f"\n{family} winner {cid} cfg={cfg} epochs={ep}")
        for variant in variants:
            ah, am = VAR[variant]
            for task in p3:
                for hor in HORIZONS:
                    yv = labels[hor]
                    vtr = g.make_view(task["train"], yv, sp, dp, X, M)
                    _, cell, ev_mask = task["evals"][0]
                    vev = g.make_view(ev_mask, yv, sp, dp, X, M)
                    fixed = int(ep[hor[6:]])
                    rg = g.fit_gnn({"agg": family, "reg": "abl"}, cfg, X, M,
                                   row_of, ah, am, vtr, vev, fixed_epochs=fixed)
                    pg = rg["pred"]
                    mu, sd = vtr["mu"], vtr["sd"]
                    tr_idx = vtr["idx"]; ev_idx = vev["idx"]
                    if variant == "full":
                        ck = (cell, hor[6:])
                        if ck not in lstm_cache:
                            rl = ts.fit_model("lstm", L, l3cfg, X, M, mu, sd,
                                              tr_idx, yv[tr_idx], ev_idx,
                                              yv[ev_idx],
                                              fixed_epochs=int(l3ep[hor[6:]]))
                            lstm_cache[ck] = rl["pred"]
                        pl = lstm_cache[ck]
                    y = vev["y"].astype(int)
                    ap_g = average_precision_score(y, pg)
                    if variant == "full":
                        ap_l = average_precision_score(y, pl)
                        d, lo, hi = paired_bootstrap(sp[ev_idx], y, pg, pl)
                        key = f"{family}_{cell}_{hor[6:]}"
                        store[key + "_y"] = y
                        store[key + "_st"] = st_id[ev_idx]
                        store[key + "_p_gnn"] = pg
                        store[key + "_p_lstm"] = pl
                    else:
                        ap_l, d, lo, hi = np.nan, ap_g - np.nan, np.nan, np.nan
                    rows.append({
                        "family": family, "config_id": cid, "graph": variant,
                        "cell": cell, "horizon": hor[6:], "n": len(y),
                        "pr_auc_gnn": ap_g, "pr_auc_lstm_L3_01": ap_l,
                        "diff_gnn_minus_lstm": d if variant == "full" else np.nan,
                        "diff_ci_lo": lo, "diff_ci_hi": hi,
                        "gnn_fit_s": round(rg["fit_s"], 1)})
                    print(f"  {variant:5s} {cell} {hor[6:]} gnn={ap_g:.5f} "
                          f"lstm={ap_l if variant=='full' else float('nan'):.5f} "
                          f"diff={d:.5f} [{lo:.5f},{hi:.5f}]")
    pd.DataFrame(rows).to_csv(ABL_CSV, index=False)
    np.savez_compressed(ABL_NPZ, **store)
    print(f"\nwrote {ABL_CSV.name}, {ABL_NPZ.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
