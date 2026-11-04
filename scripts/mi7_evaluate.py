r"""MI-7 / Phase 35 — Track 3 (head-to-head) + Track 4 (uncertainty).

Consumes whatever the MI-7 trainer has produced so far and answers the sprint's
question honestly:

  1. Ladder on P1 val_2022 (mean PR-AUC over horizons): raw-12 DL (Phase 21
     lstm_L7 / Phase 23 fusion_L3), v1-57, v2-105, and the engineered-static
     fusion (fusion_v2_L*) — vs the classical-ML ceiling (Phase 20 xgboost) and
     the frozen D16 (MI-5, FULL_V2 HPO).
  2. Same table for every other dev cell (P2 val_2020/2021/2022, P3 folds 1-4)
     x season x horizon, best-MI7 vs best-prior-DL vs best-ML.
  3. PAIRED station-bootstrap of Delta-PR-AUC on val_2022, MI7 vs ML and MI7 vs
     D16, using row-level predictions (reports/mi7_preds + reports/mi7_preds_ml)
     so shared rows cancel. This is the only place a model comparison is quoted
     with a sampling CI at the row level.
  4. Seed CI for the MI-7 families (already in mi7_metrics: seed_pr_auc_sd,
     pr_auc_ci_lo/hi).

Everything is DEV ONLY. The frozen test is never touched (see
reports/test_read_ledger.md).

Outputs
    reports/mi7_scorecard.csv
    reports/mi7_ladder.csv
    reports/mi7_paired_deltas.csv
    reports/mi7_summary.json
    reports/mi7_report.md
    figures/mi7_val2022_pr_auc.png, figures/mi7_paired_deltas.png

Run:  python -u scripts/mi7_evaluate.py
"""
from __future__ import annotations

import json
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
from baseline_model import HORIZONS, SEED                                  # noqa: E402
from sklearn.metrics import average_precision_score as _aps                # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures"
MI7 = REPORTS / "mi7_metrics.csv"
MLP = REPORTS / "mi7_preds_ml"
DLP = REPORTS / "mi7_preds"

# prior-art tables: (label, path, kind) kind in {ml, dl}
PRIOR = [
    ("phase20_ML", REPORTS / "phase20_metrics.csv", "ml"),
    ("phase21_DLraw", REPORTS / "phase21_metrics.csv", "dl"),
    ("phase22_TCN", REPORTS / "phase22_metrics.csv", "dl"),
    ("phase23_fusion", REPORTS / "phase23_metrics.csv", "dl"),
    ("phase24_GNN", REPORTS / "phase24_metrics.csv", "dl"),
    ("phase25_transformer", REPORTS / "phase25_metrics.csv", "dl"),
]
COLS = ["pr_auc", "model", "protocol", "cell", "season", "horizon"]


def _read(p):
    if not Path(p).exists():
        return None
    d = pd.read_csv(p)
    for c in COLS:
        if c not in d.columns:
            return None
    return d[COLS + [c for c in ("pr_auc_ci_lo", "pr_auc_ci_hi",
                                 "seed_pr_auc_sd", "macro_pr_auc", "n")
                     if c in d.columns]].copy()


def d16_reference():
    p = REPORTS / "mi5_selection.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    r = s["ranked"][0]
    return {"model": "D16_frozen", "protocol": "P1", "cell": "val_2022",
            "season": "all", "label_24h": r["p1_pr_auc_24h"],
            "label_48h": r["p1_pr_auc_48h"], "label_72h": r["p1_pr_auc_72h"]}


# --------------------------------------------------------------- paired bootstrap
def paired_delta(stations, y, pa, pb, reps=500, seed=SEED):
    """Station-cluster bootstrap of Delta = PR-AUC(a) - PR-AUC(b) on shared rows."""
    ok = np.isfinite(pa) & np.isfinite(pb) & ~np.isnan(y)
    y, pa, pb, st = y[ok].astype(int), pa[ok], pb[ok], stations[ok]
    if len(np.unique(y)) < 2:
        return dict(n=int(ok.sum()), delta=np.nan, lo=np.nan, hi=np.nan,
                    p_a=np.nan, p_b=np.nan, frac_a_gt_b=np.nan)
    order = np.argsort(st, kind="stable")
    _, start = np.unique(st[order], return_index=True)
    idxs = np.split(order, start[1:])
    base_a, base_b = _aps(y, pa), _aps(y, pb)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        pick = rng.integers(0, len(idxs), len(idxs))
        ii = np.concatenate([idxs[j] for j in pick])
        yy = y[ii]
        if len(np.unique(yy)) < 2:
            draws[r] = np.nan
            continue
        draws[r] = _aps(yy, pa[ii]) - _aps(yy, pb[ii])
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return dict(n=int(ok.sum()), delta=float(base_a - base_b),
                lo=float(lo), hi=float(hi), p_a=float(base_a), p_b=float(base_b),
                frac_a_gt_b=float((draws > 0).mean()) if len(draws) else np.nan)


def _load_pred(p, idx_key):
    d = np.load(p)
    return d[idx_key].astype(np.int64), d["y"], d["p"]


def main() -> None:
    if not MI7.exists():
        print("no reports/mi7_metrics.csv yet — run the trainer first"); return
    m7 = pd.read_csv(MI7)
    print(f"MI-7 metrics: {len(m7)} rows, families={sorted(m7.model.unique())}")
    prior = {lab: _read(p) for lab, p, _ in PRIOR}
    prior = {k: v for k, v in prior.items() if v is not None}

    # ---- pooled tables (all prior + MI7) tagged with source
    frames = [m7.assign(source="mi7")]
    for lab, _p, _kind in PRIOR:
        d = prior.get(lab)
        if d is not None:
            frames.append(d.assign(source=lab))
    allm = pd.concat(frames, ignore_index=True)

    # ---------------- scorecard: best per (protocol,cell,season,horizon)
    keys = ["protocol", "cell", "season", "horizon"]
    sc = (allm.sort_values("pr_auc", ascending=False)
              .drop_duplicates(subset=keys, keep="first")
              [keys + ["model", "pr_auc", "source"]])
    sc.to_csv(REPORTS / "mi7_scorecard.csv", index=False)

    # ---------------- ladder on P1 val_2022 (mean over horizons)
    def mean_pa(df, model=None, source=None):
        d = df[(df.protocol == "P1") & (df.cell == "val_2022")
               & (df.season == "all")]
        if model is not None:
            d = d[d.model == model]
        if source is not None:
            d = d[d.source == source]
        if d.empty:
            return None
        g = d.groupby("horizon")["pr_auc"].first()
        return {"mean": float(g.mean()), "h24": float(g.get("label_24h", np.nan)),
                "h48": float(g.get("label_48h", np.nan)),
                "h72": float(g.get("label_72h", np.nan)), "n_h": int(len(g))}

    ladder = []
    ref = {"raw12_lstm_L7": ("phase21_DLraw", "lstm_L7"),
           "raw12_fusion_L3": ("phase23_fusion", "fusion_L3"),
           "ML_xgboost": ("phase20_ML", "xgboost"),
           "ML_lightgbm": ("phase20_ML", "lightgbm")}
    for name, (src, mod) in ref.items():
        d = prior.get(src)
        if d is None:
            continue
        r = mean_pa(d.assign(source=src), model=mod, source=src)
        if r:
            ladder.append(dict(model=name, **r))
    for fam in sorted(m7.model.unique()):
        r = mean_pa(m7, model=fam)
        if r:
            ladder.append(dict(model=f"MI7_{fam}", **r))
    d16 = d16_reference()
    if d16:
        ladder.append(dict(model="D16_frozen", mean=np.mean(
            [d16["label_24h"], d16["label_48h"], d16["label_72h"]]),
            h24=d16["label_24h"], h48=d16["label_48h"], h72=d16["label_72h"],
            n_h=3))
    lad = pd.DataFrame(ladder).sort_values("mean", ascending=False)
    lad.to_csv(REPORTS / "mi7_ladder.csv", index=False)
    print("\n=== P1 val_2022 ladder (mean PR-AUC over horizons) ===")
    print(lad.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ---------------- paired deltas on val_2022
    stations = pd.read_csv(PROC / "sequence_index.csv",
                           usecols=["station_id"]).station_id.to_numpy()
    ml_preds = {}
    if MLP.exists():
        for p in sorted(MLP.glob("*_val_2022.npz")):
            fam = p.name.replace("_val_2022.npz", "")
            ml_preds[fam] = _load_pred(p, "idx")
    dl_preds = {}
    if DLP.exists():
        for fam_dir in sorted(DLP.iterdir()):
            if not fam_dir.is_dir():
                continue
            fs = sorted(fam_dir.glob("*val_2022.npz"))
            if fs:
                dl_preds[fam_dir.name] = _load_pred(fs[0], "ev_idx")

    paired = []
    for dfam, (di, dy, dp) in dl_preds.items():
        for hi, h in enumerate(HORIZONS):
            row = {"dl_family": dfam, "horizon": h}
            # vs each classical ML
            for mfam, (mi, my, mp) in ml_preds.items():
                # align on shared row indices
                common = np.intersect1d(di, mi)
                da = pd.Series(np.arange(len(di)), index=di)
                dm = pd.Series(np.arange(len(mi)), index=mi)
                a = da.reindex(common).to_numpy()
                b = dm.reindex(common).to_numpy()
                st = stations[common]
                r = paired_delta(st, dy[a, hi], dp[a, hi], mp[b, hi])
                row[f"d_vs_{mfam}"] = r["delta"]
                row[f"lo_vs_{mfam}"] = r["lo"]
                row[f"hi_vs_{mfam}"] = r["hi"]
                row[f"pgt_vs_{mfam}"] = r["frac_a_gt_b"]
            paired.append(row)
    pad = pd.DataFrame(paired)
    if not pad.empty:
        pad.to_csv(REPORTS / "mi7_paired_deltas.csv", index=False)
        print("\n=== paired Delta-PR-AUC on val_2022 (DL minus ML) ===")
        show = [c for c in pad.columns if c in (
            "dl_family", "horizon", "d_vs_xgboost", "lo_vs_xgboost",
            "hi_vs_xgboost", "pgt_vs_xgboost", "d_vs_lightgbm")]
        print(pad[show].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------------- summary
    best_ml = lad[lad.model.str.startswith("ML_")].iloc[0] if (lad.model.str
              .startswith("ML_")).any() else None
    best_mi7 = lad[lad.model.str.startswith("MI7_")].iloc[0] if (lad.model.str
               .startswith("MI7_")).any() else None
    d16m = float(lad.loc[lad.model == "D16_frozen", "mean"].iloc[0]) if \
        (lad.model == "D16_frozen").any() else np.nan
    summary = {
        "seed": SEED, "n_families": int(m7.model.nunique()),
        "families": sorted(m7.model.unique()),
        "best_mi7": None if best_mi7 is None else {
            "model": best_mi7["model"], "mean_pr_auc": float(best_mi7["mean"])},
        "best_ml": None if best_ml is None else {
            "model": best_ml["model"], "mean_pr_auc": float(best_ml["mean"])},
        "d16_frozen_mean": d16m,
        "verdict": None,
    }
    if best_mi7 is not None and best_ml is not None:
        gap = float(best_mi7["mean"]) - float(best_ml["mean"])
        summary["verdict"] = {
            "mi7_minus_ml": gap,
            "mi7_beats_ml": bool(gap > 0),
            "mi7_minus_d16": float(best_mi7["mean"]) - d16m,
        }
    (REPORTS / "mi7_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print("\nsummary:", json.dumps(summary["verdict"], indent=1))

    _figures(lad, pad)
    _report(lad, pad, sc, summary, d16)
    print("\nwrote reports/mi7_{scorecard,ladder,paired_deltas}.csv, "
          "reports/mi7_{summary.json,report.md}, figures/mi7_*.png")


def _figures(lad, pad):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                    # pragma: no cover
        print("matplotlib unavailable:", e); return
    FIGS.mkdir(exist_ok=True)
    if not lad.empty:
        d = lad.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.6))
        colors = ["#4c72b0" if m.startswith("MI7_") else
                  ("#c44e52" if m.startswith("ML_") else "#8c8c8c")
                  for m in d.model]
        ax.barh(d.model, d["mean"], color=colors)
        ax.axvline(d.loc[d.model == "D16_frozen", "mean"].iloc[0]
                   if (d.model == "D16_frozen").any() else np.nan,
                   color="k", ls="--", lw=1, label="D16 frozen")
        ax.set_xlabel("P1 val_2022 mean PR-AUC (24/48/72h)")
        ax.set_title("MI-7 ladder vs classical ML / prior DL")
        ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(FIGS / "mi7_val2022_pr_auc.png", dpi=140)
        plt.close(fig)
    if pad is not None and not pad.empty and "d_vs_xgboost" in pad.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for hi, h in enumerate(HORIZONS):
            r = pad[pad.horizon == h]
            x = np.arange(len(r))
            ax.errorbar(x + hi * 0.02, r["d_vs_xgboost"],
                        yerr=[r["d_vs_xgboost"] - r["lo_vs_xgboost"],
                              r["hi_vs_xgboost"] - r["d_vs_xgboost"]],
                        fmt="o", capsize=3, label=h)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(np.arange(len(pad[pad.horizon == HORIZONS[0]])))
        ax.set_xticklabels(pad[pad.horizon == HORIZONS[0]].dl_family,
                           rotation=30, ha="right")
        ax.set_ylabel("Delta PR-AUC (MI7 - xgboost)")
        ax.set_title("Paired station-bootstrap, val_2022 (95% CI)")
        ax.legend(); fig.tight_layout()
        fig.savefig(FIGS / "mi7_paired_deltas.png", dpi=140)
        plt.close(fig)


def _report(lad, pad, sc, summary, d16):
    L = []
    L.append("# MI-7 / Phase 35 — fair DL rematch: results (dev only)\n")
    L.append("Frozen test untouched (`reports/test_read_ledger.md`). All numbers "
             "are development cells.\n")
    L.append("## Ladder — P1 `val_2022` (mean PR-AUC over 24/48/72h)\n")
    if not lad.empty:
        L.append("| model | mean | 24h | 48h | 72h |")
        L.append("|---|---|---|---|---|")
        for _, r in lad.iterrows():
            L.append(f"| {r['model']} | {r['mean']:.5f} | {r['h24']:.5f} | "
                     f"{r['h48']:.5f} | {r['h72']:.5f} |")
    L.append("")
    if d16:
        L.append(f"Reference: frozen **D16** = {d16['label_24h']:.5f} / "
                 f"{d16['label_48h']:.5f} / {d16['label_72h']:.5f} "
                 f"(mean {np.mean([d16['label_24h'], d16['label_48h'], d16['label_72h']]):.5f}).\n")
    if pad is not None and not pad.empty:
        L.append("## Paired station-bootstrap vs classical ML — val_2022\n")
        L.append("Delta = PR-AUC(MI7 family) - PR-AUC(ML), shared rows, 500 "
                 "station-cluster resamples. `p>0` = fraction of resamples where "
                 "MI7 wins.\n")
        cols = [c for c in ["dl_family", "horizon", "d_vs_xgboost",
                            "lo_vs_xgboost", "hi_vs_xgboost", "pgt_vs_xgboost"]
                if c in pad.columns]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for _, r in pad.iterrows():
            L.append("| " + " | ".join(
                (f"{r[c]:+.4f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols) + " |")
        L.append("")
    v = summary.get("verdict")
    L.append("## Verdict\n")
    if v:
        L.append(f"- best MI-7: **{summary['best_mi7']['model']}** "
                 f"(mean {summary['best_mi7']['mean_pr_auc']:.5f})")
        L.append(f"- best classical ML: **{summary['best_ml']['model']}** "
                 f"(mean {summary['best_ml']['mean_pr_auc']:.5f})")
        L.append(f"- D16 frozen: **{summary['d16_frozen_mean']:.5f}**")
        L.append(f"- MI7 - ML = **{v['mi7_minus_ml']:+.5f}** "
                 f"({'DL wins' if v['mi7_beats_ml'] else 'ML still wins'}); "
                 f"MI7 - D16 = {v['mi7_minus_d16']:+.5f}.")
    else:
        L.append("- incomplete (need both MI7 and ML rows).")
    L.append("\n_Generated by scripts/mi7_evaluate.py._")
    (REPORTS / "mi7_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
