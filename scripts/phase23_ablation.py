r"""Phase 23 - matched-capacity fusion ablation (workflow Phase 23 / stage 38 DoD).

The Phase-23 tuning lets each variant pick its own config, so the fused winner is
a LARGER encoder than the standalone LSTM winner. To answer "does knowing WHERE
add skill", the encoder must be held fixed. This script:

  1. takes the tuned fusion_L3 winner config (and its per-horizon epoch counts),
  2. trains the SAME encoder with the static branch ON (fused) and OFF
     (temporal-only) - identical seed, splits, epochs, batch, everything,
  3. scores both on the full dev grid (8 tasks x 3 seasons x 3 horizons),
  4. reports the per-horizon gain fused - temporal-only with a PAIRED
     station-bootstrap CI (resample stations, difference of the two PR-AUCs).

This is the feature-group ablation the DoD asks for (fused vs temporal-only vs
static-only) at fixed capacity; the fully-tuned three-way comparison stays in
reports/phase23_metrics.csv.

Outputs:
    reports/phase23_ablation.csv        metrics rows for temporal_only_matched +
                                        fusion_matched (same schema as phase23_metrics)
    reports/phase23_ablation_boot.csv   per-horizon paired-bootstrap gain (val_2022)
    reports/phase23_ablation_preds.npz  per-cell y / p for both models (notebook)
Run:
    python -u scripts\phase23_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import train_fusion as tf                        # noqa: E402
from baseline_model import (HORIZONS, SEED, best_f1_threshold,  # noqa: E402
                            macro_pr_auc, metrics, bootstrap_pr_auc)
from sklearn.metrics import average_precision_score  # noqa: E402

REPORTS = ROOT / "reports"
ROC = ROOT / "data" / "processed"
SEARCH = REPORTS / "phase23_search.csv"
OUT_CSV = REPORTS / "phase23_ablation.csv"
OUT_BOOT = REPORTS / "phase23_ablation_boot.csv"
OUT_PRED = REPORTS / "phase23_ablation_preds.npz"
BOOT_REPS = 100


def paired_boot(stations, y, p_a, p_b, reps=BOOT_REPS):
    """Paired station bootstrap CI for PR-AUC(a) - PR-AUC(b)."""
    uniq = np.unique(stations)
    idx_by = {s: np.flatnonzero(stations == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(average_precision_score(y[idx], p_a[idx])
                 - average_precision_score(y[idx], p_b[idx]))
    if len(d) < reps // 2:
        return np.nan, np.nan, np.nan
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    t0 = time.time()
    fw = pd.read_csv(SEARCH).query("family == 'fusion_L3'").sort_values(
        "mean_pr_auc", ascending=False).iloc[0]
    cfg = json.loads(fw.params)
    cfg["_weight"] = "on" if fw.weighting == "pos_weight=ratio" else "off"
    cfg["_cell"] = -1
    epochs = json.loads(fw.best_epochs)
    L = tf.resolve_lstar()
    print(f"matched config = fusion_L3 winner {fw.config_id}: {fw.params} "
          f"| epochs {epochs} | L={L}")

    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    tf.STATION_OF = ix.station_id.to_numpy()
    S = pd.read_csv(ROC / "static_vectors.csv").set_index("station_id")
    d_lin = None
    X, M = tf.ts.prepare(L)
    tasks = tf.ts.build_tasks_index(ix)
    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    stations = ix.station_id.to_numpy()
    season_v = ix.season.to_numpy()

    SPECS = {
        "fusion_matched": dict(kind="lstm", use_seq=True, use_static=True, lin=False),
        "temporal_only_matched": dict(kind="lstm", use_seq=True, use_static=False,
                                      lin=False),
    }
    rows, store = [], {}
    for task in tasks:
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx = np.flatnonzero(task["train"] & ~np.isnan(yv))
            if len(np.unique(yv[tr_idx])) < 2:
                continue
            enc = tf.fit_encoders(SPECS["fusion_matched"], X, M, S, d_lin, tr_idx)
            tr_t = tf.encode(enc, SPECS["fusion_matched"], X, M, S, d_lin, tr_idx)
            fixed = int(epochs[hor.split("_")[-1]])
            for mname, spec in SPECS.items():
                r = tf.fit_net(spec, cfg, tr_t, yv[tr_idx], tr_t, yv[tr_idx],
                               fixed_epochs=fixed)
                net = r["net"]
                with torch.no_grad():
                    p_tr = torch.sigmoid(net(tr_t.get("seq"), tr_t.get("stat"),
                                             tr_t.get("lin"))).numpy()
                thr = best_f1_threshold(yv[tr_idx].astype(int), p_tr)
                for protocol, cell, ev_mask in task["evals"]:
                    for seas in (None, "monsoon", "non-monsoon"):
                        msk = ev_mask.copy()
                        if seas is not None:
                            msk = ev_mask & (season_v == seas)
                        idx = np.flatnonzero(msk & ~np.isnan(yv))
                        if len(idx) == 0:
                            continue
                        ev_t = tf.encode(enc, spec, X, M, S, d_lin, idx)
                        with torch.no_grad():
                            p = torch.sigmoid(net(ev_t.get("seq"), ev_t.get("stat"),
                                                  ev_t.get("lin"))).numpy()
                        y = yv[idx].astype(int)
                        row = metrics(y, p, thr)
                        lo, hi = (np.nan, np.nan)
                        if seas is None and len(idx) >= 1000 and \
                                len(np.unique(stations[idx])) >= 10:
                            lo, hi = bootstrap_pr_auc(stations[idx], y, p)
                        row.update(protocol=protocol, train=task["name"], cell=cell,
                                   season=seas or "all", model=mname, horizon=hor,
                                   n=len(idx), train_rows=int(tr_idx.size),
                                   train_base_rate=float(yv[tr_idx].mean()),
                                   pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                                   macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                                   config_id=fw.config_id, n_params=r["net"].n_params())
                        rows.append(row)
                        if protocol == "P1" and cell == "val_2022" and seas is None:
                            store[f"{mname}|{hor}|y"] = y
                            store[f"{mname}|{hor}|p"] = p
                            store[f"{mname}|{hor}|st"] = stations[idx]
                print(f"  {task['protocol']:18s} {task['name']:13s} {hor} {mname:22s} "
                      f"fit {r['fit_s']:.0f}s")

    met = pd.DataFrame(rows)
    met.to_csv(OUT_CSV, index=False)
    np.savez_compressed(OUT_PRED, **store)

    boot_rows = []
    print("\n=== fused - temporal-only gain (val_2022, season=all, paired station "
          "bootstrap) ===")
    for hor in HORIZONS:
        y = store[f"fusion_matched|{hor}|y"]
        st = store[f"fusion_matched|{hor}|st"]
        pf = store[f"fusion_matched|{hor}|p"]
        pt = store[f"temporal_only_matched|{hor}|p"]
        ap_f, ap_t = average_precision_score(y, pf), average_precision_score(y, pt)
        m, lo, hi = paired_boot(st, y, pf, pt)
        boot_rows.append({"horizon": hor, "pr_auc_fused": ap_f,
                          "pr_auc_temporal_only": ap_t, "gain": ap_f - ap_t,
                          "gain_mean_boot": m, "gain_ci_lo": lo, "gain_ci_hi": hi,
                          "n": len(y), "n_stations": len(np.unique(st))})
        print(f"  {hor}: fused {ap_f:.4f} vs temporal-only {ap_t:.4f}  "
              f"gain {ap_f - ap_t:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(boot_rows).to_csv(OUT_BOOT, index=False)
    print(f"\nwrote {OUT_CSV.name} ({len(met)} rows), {OUT_BOOT.name}, "
          f"{OUT_PRED.name}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
