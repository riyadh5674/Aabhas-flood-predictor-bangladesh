"""Phase 31 - robustness analysis (workflow Phase 31 / stage 53; phase0 section 3 evidence).

Question
--------
How does the *surviving* model set behave when the evaluation conditions change?
The phase is a pure analysis: it re-fits the frozen Phase-16/20 winning configs (no new
hyper-parameter search, no architecture change) on the Phase-15 dev protocols and reports
**every result per cell, never pooled** (D8).

Surviving models (Phase 27/28/29/30 leaders + references):
    leaders    lightgbm (24 h), xgboost (48 h), xgboost_selected (72 h)
    references logreg_linear, persistence, climatology

Every dimension below is reported per cell x season x horizon (D8):
    different years        P2 rolling origin   val_2020/2021/2022
    unseen stations        P3 spatial CV        spatial_fold_1..4
    monsoon / non-monsoon  season slices inside every cell
    high vs extreme rain   cell-relative `rain_7d` quantiles (low <= p50, high p50..p90, extreme > p90)
    low-data windows       input missingness buckets (NaN fraction over the model's own features)
    all three horizons     24 h / 48 h / 72 h
    stricter issue-time    features through `t-1` (phase0 section 3), a per-cell sensitivity check

Decision rule (used everywhere): row cost `C_FN*FN + C_FP*FP`, Phase-29 primary ratio
`C_FN : C_FP = 10 : 1`; the operating point of each slice is the exact cost argmin on that
slice (`phase29_thresholds.best_exact`).  A second column applies the **frozen Phase-29 P1
`val_2022` threshold** to every cell (`recall_frozen`) to separate ranking loss from
threshold-transfer loss; for cells other than `val_2022` that is a transfer application.

Outputs
-------
    reports/phase31_cells.csv          per cell x season x horizon x model
    reports/phase31_rainfall.csv       rain_7d buckets (low/high/extreme)
    reports/phase31_missingness.csv    input-missingness buckets
    reports/phase31_issuetime.csv      standard vs stricter issue-time per cell x horizon x model
    reports/phase31_reuse_check.csv    Phase-31 fresh vs Phase-27 generalisation deltas
    reports/phase31_summary.json       seed, SHA, test_rows_read=0, headline
    figures/phase31_robustness.png

Run:
    python -u scripts/phase31_robustness.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import baseline_model as B                      # noqa: E402
import phase20_classical_ml as P20              # noqa: E402
from phase29_thresholds import best_exact, metrics_at, CFN_PRIMARY   # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
PROC = ROOT / "data" / "processed"
DATA = PROC / "experimental_dataset.csv"
DATA_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"

SEED = B.SEED
HOR = list(B.HORIZONS)
HORD = {h: h.replace("label_", "") for h in HOR}
SEASONS = ("all", "monsoon", "non-monsoon")

CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
LEADERS = ["lightgbm", "xgboost", "xgboost_selected"]
REFS = ["logreg_linear", "persistence", "climatology"]
PANEL = LEADERS + REFS
L2_FAMILIES = {"lightgbm", "xgboost", "xgboost_selected"}

RAIN_BUCKETS = ["low", "high", "extreme"]
NAN_BUCKETS = ["none", "low", "high", "very-high"]
NAN_EDGES = (0.0, 0.05, 0.15)


# ------------------------------------------------------------------------------ utils
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def bucket_row(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pr, roc = safe_auc(y, p)
    m = metrics_at(y, p, thr)
    return dict(pr_auc=pr, roc_auc=roc, recall=m["recall"], precision=m["precision"],
                far=m["far"], missed=m["missed"])


def nan_bucket(nf: np.ndarray) -> np.ndarray:
    idx = np.digitize(nf, np.asarray(NAN_EDGES), right=True)      # 0,1,2,3
    return np.asarray(NAN_BUCKETS, dtype=object)[np.clip(idx, 0, 3)]


def dev_tasks(d: pd.DataFrame) -> list[dict]:
    """The seven Phase-15 dev cells, each scored once.

    `build_tasks` intentionally exposes `val_2022` twice (P1 primary and the P2
    rolling-origin `train_<=2021` task, whose training mask is asserted identical to P1's).
    Phase 31 keeps the first occurrence (P1) so every cell appears exactly once.
    """
    tasks: list[dict] = []
    seen: set[str] = set()
    for t in B.build_tasks(d):
        ev = []
        for pr, c, m in t["evals"]:
            if c in CELLS and c not in seen:
                ev.append((pr, c, m))
                seen.add(c)
        if ev:
            tasks.append({"protocol": t["protocol"], "name": t["name"],
                          "train": t["train"], "evals": ev})
    got = {c for t in tasks for _, c, _ in t["evals"]}
    assert got == set(CELLS), f"dev grid mismatch: {sorted(got)}"
    return tasks


def shift_index(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Within-station previous *calendar day* row index (stricter issue-time variant)."""
    key = d[["station_id", "date"]].copy()
    key["date"] = key["date"] + pd.Timedelta(days=1)
    key["prev_idx"] = np.arange(len(d))
    m = d[["station_id", "date"]].merge(key, on=["station_id", "date"], how="left")
    assert len(m) == len(d), "shift merge changed the row count"
    prev = m["prev_idx"].to_numpy(dtype=float)
    has_prev = ~np.isnan(prev)
    prev_idx = np.where(has_prev, prev, 0.0).astype(int)
    return prev_idx, has_prev


def build_shifted(d: pd.DataFrame, feats: list[str],
                  prev_idx: np.ndarray, has_prev: np.ndarray) -> pd.DataFrame:
    out = d.copy()
    src = d[feats].iloc[prev_idx].reset_index(drop=True)
    src.loc[~has_prev, :] = np.nan
    for c in feats:
        out[c] = src[c].to_numpy()
    return out


def fit_predict(model: str, feats: list[str], X: pd.DataFrame, y: np.ndarray,
                tr_idx: np.ndarray, ev_idx: np.ndarray, l1: dict,
                winners: dict) -> np.ndarray:
    ytr = y[tr_idx].astype(int)
    if model in L2_FAMILIES:
        ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        pipe, _w, sw = P20.make_pipe(model, winners[model]["cfg"], feats, ratio)
        P20.fit_pipe(pipe, X.iloc[tr_idx][feats], ytr, sw)
    else:
        spec = l1[model]
        if spec["kind"] == "const":
            return np.full(len(ev_idx), float(ytr.mean()))
        pipe = spec["pipe"]
        pipe.fit(X.iloc[tr_idx][feats], ytr)
    return pipe.predict_proba(X.iloc[ev_idx][feats])[:, 1]


# ------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    print(f"Phase 31 - robustness analysis (stage 53) | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')}")
    print(f"hardware: CPU {platform.machine()} {platform.system()} | "
          f"{platform.python_version()} | torch {_torch_ver()}")
    sha = sha256(DATA)
    print(f"experimental_dataset.csv SHA-256 {sha}")
    assert sha == DATA_SHA, "experimental table changed - stop"

    d, fsets = B.load()
    d = d.reset_index(drop=True)
    split = d.split.to_numpy()
    hold = d.station_holdout.to_numpy()
    test_mask = split == "test"
    season = d.season.to_numpy()
    rain7 = d.rain_7d.to_numpy(dtype=float)

    tasks = dev_tasks(d)
    l1 = B.build_models(fsets)

    winners = {}
    search = pd.read_csv(REPORTS / "phase20_search.csv")
    for fam in L2_FAMILIES:
        r = search[search.family == fam].sort_values("mean_pr_auc", ascending=False).iloc[0]
        winners[fam] = {"cfg": json.loads(r["params"]), "config_id": str(r["config_id"]),
                        "mean_pr_auc": float(r["mean_pr_auc"])}
        print(f"  winner {fam:17s} {r['config_id']:20s} mean_pr_auc={r['mean_pr_auc']:.5f}")

    feats_of = {m: (list(fsets[P20.FEATURE_SET[m]]) if m in L2_FAMILIES
                    else list(l1[m]["features"])) for m in PANEL}

    all_feats = sorted(set(fsets["FULL"]) | set(fsets["LINEAR"]) | set(fsets["SELECTED"]))
    prev_idx, has_prev = shift_index(d)
    d_shift = build_shifted(d, all_feats, prev_idx, has_prev)
    print(f"shifted features through t-1 available for {int(has_prev.sum()):,} / "
          f"{len(d):,} rows ({has_prev.mean():.3f})")

    # frozen Phase-29 P1 thresholds: (model, hord, period) -> t_warn
    chosen = pd.read_csv(REPORTS / "phase29_chosen_thresholds.csv")
    TW = {(r.model, r.horizon, r.period): float(r.t_warn) for r in chosen.itertuples()}

    cell_rows: list[dict] = []
    rain_rows: list[dict] = []
    miss_rows: list[dict] = []
    it_rows: list[dict] = []
    test_rows_read = 0

    for task in tasks:
        tr = task["train"]
        assert not (tr & test_mask).any(), "test row in a training mask"
        for protocol, cell, ev_mask in task["evals"]:
            assert not (ev_mask & test_mask).any(), "test row in an eval mask"
            for hor in HOR:
                y = d[hor].to_numpy()
                tr_idx = np.flatnonzero(tr & ~np.isnan(y))
                ev_idx = np.flatnonzero(ev_mask & ~np.isnan(y))
                for model in PANEL:
                    feats = feats_of[model]
                    nan_frac = d[feats].isna().mean(axis=1).to_numpy()
                    p = fit_predict(model, feats, d, y, tr_idx, ev_idx, l1, winners)
                    sev = season[ev_idx]
                    yv = y[ev_idx].astype(int)
                    stv = d.station_id.to_numpy()[ev_idx]
                    rv = rain7[ev_idx]
                    nv = nan_frac[ev_idx]

                    # ---- (a) cell x season x horizon
                    for s in SEASONS:
                        sm = np.ones(len(yv), bool) if s == "all" else (sev == s)
                        ys, ps = yv[sm], p[sm]
                        if len(ys) == 0:
                            continue
                        thr_s = best_exact(ys, ps, CFN_PRIMARY)[0]
                        mrow = metrics_at(ys, ps, thr_s)
                        pr, roc = safe_auc(ys, ps)
                        tf = TW.get((model, HORD[hor], s), np.nan)
                        frow = (metrics_at(ys, ps, tf) if np.isfinite(tf) else
                                {"recall": np.nan, "far": np.nan, "missed": np.nan})
                        cell_rows.append({
                            "model": model,
                            "role": "leader" if model in LEADERS else "reference",
                            "cell": cell, "protocol": protocol, "season": s, "horizon": HORD[hor],
                            "n": int(len(ys)), "n_pos": int(ys.sum()),
                            "base_rate": float(ys.mean()),
                            "pr_auc": pr, "roc_auc": roc,
                            "brier": float(np.mean((ps - ys) ** 2)),
                            "ece": float(B.ece(ys, ps)),
                            "macro_pr_auc": float(B.macro_pr_auc(stv[sm], ys, ps)),
                            "thr_cost": float(thr_s),
                            "recall": mrow["recall"], "precision": mrow["precision"],
                            "far": mrow["far"], "fpr": mrow["fpr"],
                            "missed": mrow["missed"],
                            "thr_frozen": tf, "recall_frozen": frow["recall"],
                            "far_frozen": frow["far"], "missed_frozen": frow["missed"],
                        })

                    # ---- (b) rainfall-intensity buckets (season all, threshold fixed)
                    thr_all = best_exact(yv, p, CFN_PRIMARY)[0]
                    fin = np.isfinite(rv)
                    if fin.sum() >= 50:
                        q50, q90 = np.nanpercentile(rv[fin], [50, 90])
                    else:
                        q50 = q90 = np.nan
                    for b in RAIN_BUCKETS:
                        if b == "low":
                            bm = fin & (rv <= q50)
                        elif b == "high":
                            bm = fin & (rv > q50) & (rv <= q90)
                        else:
                            bm = fin & (rv > q90)
                        if bm.sum() == 0:
                            continue
                        br = bucket_row(yv[bm], p[bm], thr_all)
                        rain_rows.append({"model": model, "cell": cell, "horizon": HORD[hor],
                                          "bucket": b, "q50": q50, "q90": q90,
                                          "n": int(bm.sum()), "n_pos": int(yv[bm].sum()),
                                          "base_rate": float(yv[bm].mean()),
                                          "thr": float(thr_all), **br})

                    # ---- (c) missingness buckets (season all, threshold fixed)
                    nb = nan_bucket(nv)
                    for b in NAN_BUCKETS:
                        bm = nb == b
                        if bm.sum() == 0:
                            continue
                        br = bucket_row(yv[bm], p[bm], thr_all)
                        miss_rows.append({"model": model, "cell": cell, "horizon": HORD[hor],
                                          "bucket": b, "n": int(bm.sum()),
                                          "n_pos": int(yv[bm].sum()),
                                          "base_rate": float(yv[bm].mean()),
                                          "thr": float(thr_all), **br})

                    # ---- (d) stricter issue-time variant (features through t-1)
                    # ---- (d) stricter issue-time variant (features through t-1)
                    keep = has_prev[ev_idx]
                    if keep.sum() >= 50 and len(np.unique(yv[keep])) >= 2:
                        tr_s = tr_idx[has_prev[tr_idx]]
                        p_str = fit_predict(model, feats, d_shift, y, tr_s,
                                            ev_idx[keep], l1, winners)
                        ys, ps_std, ps_str = yv[keep], p[keep], p_str
                        thr_s = best_exact(ys, ps_std, CFN_PRIMARY)[0]
                        ms = metrics_at(ys, ps_std, thr_s)
                        pr_s, roc_s = safe_auc(ys, ps_std)
                        thr_r = best_exact(ys, ps_str, CFN_PRIMARY)[0]
                        mr = metrics_at(ys, ps_str, thr_r)
                        pr_r, roc_r = safe_auc(ys, ps_str)
                        base = {"model": model, "cell": cell, "horizon": HORD[hor],
                                "n": int(keep.sum()), "n_pos": int(ys.sum()),
                                "base_rate": float(ys.mean())}
                        it_rows.append({**base, "variant": "standard",
                                        "pr_auc": pr_s, "roc_auc": roc_s, "thr": float(thr_s),
                                        "recall": ms["recall"], "far": ms["far"]})
                        it_rows.append({**base, "variant": "stricter_t-1",
                                        "pr_auc": pr_r, "roc_auc": roc_r, "thr": float(thr_r),
                                        "recall": mr["recall"], "far": mr["far"]})
            print(f"  {protocol:20s} {cell:15s} done ({time.time() - t0:.0f}s)")

    cells_df = pd.DataFrame(cell_rows)
    rain_df = pd.DataFrame(rain_rows)
    miss_df = pd.DataFrame(miss_rows)
    it_df = pd.DataFrame(it_rows)
    cells_df.to_csv(REPORTS / "phase31_cells.csv", index=False)
    rain_df.to_csv(REPORTS / "phase31_rainfall.csv", index=False)
    miss_df.to_csv(REPORTS / "phase31_missingness.csv", index=False)
    it_df.to_csv(REPORTS / "phase31_issuetime.csv", index=False)

    # --------------------------------------------------------- reuse check vs Phase 27
    # Phase 27 already scored the same dev cells for all 24 models with the same metric
    # function; the table below proves the Phase-31 fresh re-fit agrees (it must).
    gen = pd.read_csv(REPORTS / "phase27_generalisation.csv")
    gen["horizon"] = gen["horizon"].str.replace("label_", "", regex=False)
    gcell = gen[gen.block.isin(["temporal_year", "spatial_fold"])].rename(
        columns={"bucket": "cell", "pr_auc": "pr_auc_p27"})[
        ["model", "horizon", "cell", "pr_auc_p27"]]
    reuse = cells_df[cells_df.season == "all"].merge(
        gcell, on=["model", "horizon", "cell"], how="inner")
    reuse["delta_pr_auc"] = reuse["pr_auc"] - reuse["pr_auc_p27"]
    reuse.to_csv(REPORTS / "phase31_reuse_check.csv", index=False)
    max_delta = (float(np.nanmax(np.abs(reuse["delta_pr_auc"].to_numpy())))
                 if len(reuse) else float("nan"))

    # ------------------------------------------------------------------- headline
    leaders = cells_df[(cells_df.role == "leader") & (cells_df.season == "all")]
    head = {}
    for h in ["24h", "48h", "72h"]:
        sub = leaders[leaders.horizon == h]
        for m in LEADERS:
            v = sub[sub.model == m].sort_values("cell")["pr_auc"]
            if len(v) == 0:
                continue
            head[f"{m}::{h}"] = {"min": float(v.min()), "median": float(v.median()),
                                 "max": float(v.max()),
                                 "cells": sub[sub.model == m].set_index("cell")["pr_auc"].round(5).to_dict()}

    summary = {
        "phase": 31, "stage": 53, "seed": SEED, "test_rows_read": test_rows_read,
        "dataset_sha256": sha,
        "hardware": f"CPU {platform.machine()} {platform.system()}",
        "n_cells": len(CELLS), "n_models": len(PANEL),
        "leaders": LEADERS, "references": REFS,
        "thresholds": {"objective": "C_FN*FN + C_FP*FP", "primary_ratio": CFN_PRIMARY},
        "rain_buckets": RAIN_BUCKETS, "nan_buckets": NAN_BUCKETS,
        "reuse_vs_phase27": {"rows": int(len(reuse)),
                             "max_abs_delta_pr_auc": max_delta},
        "headline_leaders_P1_all_season": head,
    }
    (REPORTS / "phase31_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nreuse check vs Phase 27: {len(reuse)} rows, max |delta PR-AUC| = {max_delta:.2e}")

    make_figure(cells_df, rain_df, miss_df, it_df)

    print("\nphase31_cells.csv head:")
    print(cells_df.head(6).to_string(index=False))
    print(f"\nTOTAL {time.time() - t0:.0f}s - exit=0")


def make_figure(cells_df: pd.DataFrame, rain_df: pd.DataFrame,
                miss_df: pd.DataFrame, it_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                              # pragma: no cover
        print(f"[figure] skipped: {exc}")
        return
    FIGS = ROOT / "figures"
    FIGS.mkdir(exist_ok=True)
    order = CELLS
    colors = {"lightgbm": "#1f77b4", "xgboost": "#ff7f0e", "xgboost_selected": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # row 1: PR-AUC per cell (leaders), one panel per horizon
    for ax, h in zip(axes[0], ["24h", "48h", "72h"]):
        sub = cells_df[(cells_df.role == "leader") & (cells_df.season == "all")
                       & (cells_df.horizon == h)]
        for m in LEADERS:
            dd = sub[sub.model == m].groupby("cell")["pr_auc"].first()
            ax.plot(order, dd.reindex(order).to_numpy(), "o-", color=colors[m], label=m)
        ax.set_title(f"PR-AUC per cell - {h}")
        ax.set_ylim(0.85, 1.0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # row 2: sensitivity - one marker per cell (never pooled)
    ax = axes[1][0]
    sub = rain_df[(rain_df.horizon == "24h") & (rain_df.model.isin(LEADERS))]
    for m in LEADERS:
        for b in RAIN_BUCKETS:
            v = sub[(sub.model == m) & (sub.bucket == b)]["pr_auc"]
            ax.scatter([RAIN_BUCKETS.index(b)] * len(v), v, color=colors[m], alpha=0.6, s=18)
    ax.set_xticks(range(len(RAIN_BUCKETS)))
    ax.set_xticklabels(RAIN_BUCKETS)
    ax.set_title("PR-AUC by rain_7d bucket (24h, per cell)")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    sub = miss_df[(miss_df.horizon == "24h") & (miss_df.model.isin(LEADERS))]
    for m in LEADERS:
        for b in NAN_BUCKETS:
            v = sub[(sub.model == m) & (sub.bucket == b)]["pr_auc"]
            ax.scatter([NAN_BUCKETS.index(b)] * len(v), v, color=colors[m], alpha=0.6, s=18)
    ax.set_xticks(range(len(NAN_BUCKETS)))
    ax.set_xticklabels(NAN_BUCKETS, rotation=20)
    ax.set_title("PR-AUC by missingness bucket (24h, per cell)")
    ax.grid(alpha=0.3)

    ax = axes[1][2]
    dfp = it_df[(it_df.horizon == "24h") & (it_df.model.isin(LEADERS))].pivot_table(
        index=["model", "cell"], columns="variant", values="pr_auc")
    for m in LEADERS:
        if m not in dfp.index.get_level_values(0):
            continue
        v = dfp.loc[m]
        ax.scatter(v["standard"], v["stricter_t-1"], color=colors[m], alpha=0.7, s=22,
                   label=m)
    lim = [0.85, 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("standard issue-time PR-AUC")
    ax.set_ylabel("stricter (t-1) PR-AUC")
    ax.set_title("issue-time sensitivity (24h, per cell)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Phase 31 - robustness analysis (stage 53): surviving leaders, per cell, never pooled")
    fig.tight_layout()
    out = FIGS / "phase31_robustness.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[figure] wrote {out}")


def _torch_ver() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:                                    # pragma: no cover
        return "n/a"


if __name__ == "__main__":
    main()
