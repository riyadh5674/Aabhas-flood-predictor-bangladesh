"""Phase 29 — risk levels & threshold optimization (stages 48, 49).

Deliverables
------------
* reports/phase29_threshold_sweep.csv     full sweep (model x horizon x period x threshold)
* reports/phase29_chosen_thresholds.csv   cost-optimal + F1-optimal threshold per cell
* reports/phase29_transfer.csv            period-transfer penalty (raw + calibrated)
* reports/phase29_risk_bands.csv          low/moderate/high/very-high, cost-derived vs quartile
* reports/phase29_pr_auc.csv              P1 val_2022 PR-AUC per model/horizon + flagged
* reports/phase29_summary.json            cost statement, frozen thresholds/bands, test_rows_read
* figures/phase29_threshold_cost.png      expected cost vs threshold (one panel per horizon)
* figures/phase29_bands_transfer.png      P-R curves, period transfer, band flood rates

Design
------
Thresholds and risk bands are chosen **on the P1 primary validation cell (`val_2022`) only**,
from the exact probabilities Phase 28 produced (`reports/phase28_predictions.npz`): a single
score path, no retraining, test never read.

The operating objective is stated explicitly: **a missed flood row is `C_FN` times as costly as
a false-alarm row** (`C_FP = 1`).  Primary `C_FN = 10`; sensitivity at 5 and 20.  Cost is the
row-level proxy `C_FN*FN + C_FP*FP` (normalised per 1 000 rows).  Thresholds are derived on
**train-free validation** and, because the Phase-16 lesson is binding, **re-derived per period**;
a second option — carry the decision on a **calibrated** probability — is evaluated with a
deterministic 5-fold station cross-fit of isotonic calibration.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
NPZ = REPORTS / "phase28_predictions.npz"
REG = REPORTS / "phase17_experiment_registry.csv"
SUMMARY28 = REPORTS / "phase28_summary.json"

SEED = 20261008
HOR = ["label_24h", "label_48h", "label_72h"]
HOR_DISP = {h: h.replace("label_", "") for h in HOR}
PERIODS = ["all", "monsoon", "non-monsoon"]
CFN_PRIMARY = 10.0
COST_RATIOS = {"fn5": 5.0, "fn10": 10.0, "fn20": 20.0}
GRID = sorted({round(x, 4) for x in
               list(np.arange(0.01, 0.06, 0.005)) + list(np.arange(0.05, 0.61, 0.01))})


# ------------------------------------------------------------------ fast counts
def suffix_counts(y: np.ndarray, p: np.ndarray):
    """Sort ascending once; return (ps, suf_tp, suf_fp, P, n) for O(n log n) sweeps."""
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ys = y[order].astype(np.int64)
    suf_tp = np.cumsum(ys[::-1])[::-1]
    suf_fp = np.cumsum((1 - ys)[::-1])[::-1]
    return ps, suf_tp, suf_fp, int(ys.sum()), len(ys)


def _row(t: float, tp: int, fp: int, n: int, P: int) -> dict:
    fn = P - tp
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and (prec + rec)) else 0.0
    out = dict(threshold=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
               precision=prec, recall=rec, f1=float(f1),
               far=fp / (fp + tp) if (fp + tp) else np.nan,
               fpr=fp / (fp + tn) if (fp + tn) else np.nan,
               alarm_rate=(tp + fp) / n, missed=fn)
    for tag, c in COST_RATIOS.items():
        out[f"cost_{tag}_per1k"] = (c * fn + fp) * 1000.0 / n
    return out


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    tp = int(np.sum((p >= t) & (y == 1)))
    fp = int(np.sum((p >= t) & (y == 0)))
    return _row(t, tp, fp, len(y), int(y.sum()))


def grid_sweep(y: np.ndarray, p: np.ndarray, grid) -> list[dict]:
    ps, stp, sfp, P, n = suffix_counts(y, p)
    g = np.asarray(grid, dtype=float)
    idx = np.searchsorted(ps, g, side="left")
    out = []
    for k, t in enumerate(g):
        i = int(idx[k])
        tp, fp = (int(stp[i]), int(sfp[i])) if i < n else (0, 0)
        out.append(_row(t, tp, fp, n, P))
    return out


def best_exact(y: np.ndarray, p: np.ndarray, cfn: float) -> tuple[float, dict]:
    """Exact argmin over every distinct observed score, plus the 'never alarm' interval
    (a threshold strictly above the max score), which the bare score set misses."""
    ps, stp, sfp, P, n = suffix_counts(y, p)
    ts = np.unique(ps)
    idx = np.searchsorted(ps, ts, side="left")
    tp = stp[idx]
    fp = sfp[idx]
    cost = cfn * (P - tp) + fp
    j = int(np.argmin(cost))
    if cfn * P < cost[j]:                      # predict none (t > max score)
        t = float(np.nextafter(ps[-1], np.inf))
        return t, _row(t, 0, 0, n, P)
    return float(ts[j]), _row(float(ts[j]), int(tp[j]), int(fp[j]), n, P)


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ps, stp, sfp, P, n = suffix_counts(y, p)
    ts = np.unique(ps)
    idx = np.searchsorted(ps, ts, side="left")
    tp = stp[idx].astype(float)
    fp = sfp[idx].astype(float)
    fn = P - tp
    denom = 2 * tp + fp + fn
    f1 = np.where((tp > 0) & (denom > 0), 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    j = int(np.argmax(f1))
    return float(ts[j]), float(f1[j])


# ---------------------------------------------------------------- calibration
def sfold(s, k: int = 5) -> int:
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16) % k


def iso_oof(p: np.ndarray, y: np.ndarray, st: np.ndarray, k: int = 5) -> np.ndarray:
    """Deterministic out-of-fold isotonic (station-hash folds; RNG-order independent)."""
    f = np.array([sfold(s, k) for s in st])
    out = np.full_like(p, np.nan, dtype=float)
    for i in range(k):
        tr, te = f != i, f == i
        if tr.sum() < 20 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p[tr], y[tr])
        out[te] = ir.predict(p[te])
    return out


# --------------------------------------------------------------- risk bands
def choose_cuts(y: np.ndarray, p: np.ndarray, t_warn: float) -> tuple[float, float, float]:
    """Cost-derived cuts: watch = t_warn/2, warning = t_warn, high = first t with precision>=0.90."""
    t_high = float(p.max())
    for t in np.sort(np.unique(p[p >= t_warn])):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        if tp + fp and tp / (tp + fp) >= 0.90:
            t_high = float(t)
            break
    return 0.5 * t_warn, float(t_warn), t_high


def band_table(y: np.ndarray, p: np.ndarray, cuts: tuple, model: str, hor: str,
               level: str, scheme: str) -> list[dict]:
    lo, mid, hi = cuts
    edges = [(-np.inf, lo, "low"), (lo, mid, "moderate"), (mid, hi, "high"),
             (hi, np.inf, "very high")]
    action = {"low": "routine monitoring", "moderate": "watch",
              "high": "warning", "very high": "high-risk warning / activate"}
    base = float(y.mean())
    rows = []
    for a, b, name in edges:
        mm = np.ones(len(p), bool)
        if np.isfinite(a):
            mm &= (p >= a)
        if np.isfinite(b):
            mm &= (p < b)
        n = int(mm.sum())
        floods = int(y[mm].sum())
        rate = floods / n if n else np.nan
        rows.append(dict(model=model, level=level, horizon=hor, scheme=scheme, band=name,
                         p_lo=(None if not np.isfinite(a) else round(float(a), 4)),
                         p_hi=(None if not np.isfinite(b) else round(float(b), 4)),
                         n=n, freq=n / len(y), floods=floods, flood_rate=rate,
                         lift=(rate / base if (base and n) else np.nan),
                         action=action[name]))
    return rows


# ------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    z = np.load(NPZ, allow_pickle=True)
    models = list(z["models"])
    reg = pd.read_csv(REG)
    lvl = dict(zip(reg.model, reg.level))
    lvl.setdefault("transformer_bestL", "L4")   # registry L4-03 is named transformer_temporal
    s28 = json.loads(SUMMARY28.read_text(encoding="utf-8"))
    flagged = {pair[0] for pair in s28["needs_recalibration"]}

    print(f"Phase 29 — risk levels & thresholds | {len(models)} models | "
          f"CFN={CFN_PRIMARY:.0f} (FP=1) | grid {GRID[0]}..{GRID[-1]} ({len(GRID)}) | seed={SEED}")

    # ---------------------------------------------------------------- sweep
    sweep, chosen, legend = [], [], []
    for name in models:
        for hor in HOR:
            y_all = z[f"y::{hor}"]
            se = z[f"se::{hor}"]
            p_all = z[f"p::{name}::{hor}"]
            ap = float(average_precision_score(y_all, p_all))
            legend.append(dict(model=name, level=lvl.get(name, "?"), horizon=HOR_DISP[hor],
                               pr_auc=ap, flagged=name in flagged))
            for period in PERIODS:
                m = np.ones(len(y_all), bool) if period == "all" else (se == period)
                y, p = y_all[m], p_all[m]
                for row in grid_sweep(y, p, GRID):
                    sweep.append(dict(model=name, level=lvl.get(name, "?"),
                                      horizon=HOR_DISP[hor], period=period, **row))
                tc, mc = best_exact(y, p, CFN_PRIMARY)
                tf, f1 = best_f1(y, p)
                t5, _ = best_exact(y, p, 5.0)
                t20, _ = best_exact(y, p, 20.0)
                chosen.append(dict(model=name, level=lvl.get(name, "?"),
                                   horizon=HOR_DISP[hor], period=period,
                                   n=len(y), base_rate=float(y.mean()), pr_auc=ap,
                                   t_warn=tc, recall=mc["recall"], precision=mc["precision"],
                                   f1=mc["f1"], far=mc["far"], fpr=mc["fpr"],
                                   alarm_rate=mc["alarm_rate"], missed=mc["missed"],
                                   cost_per1k=mc[f"cost_fn{int(CFN_PRIMARY)}_per1k"],
                                   t_f1=tf, f1_opt=f1, t_fn5=t5, t_fn20=t20))
    sweep = pd.DataFrame(sweep)
    chosen = pd.DataFrame(chosen)
    legend = pd.DataFrame(legend)

    # --------------------------------------------------------- period transfer
    transfer = []
    for name in models:
        for hor in HOR:
            y_all = z[f"y::{hor}"]
            se = z[f"se::{hor}"]
            masks = {"all": np.ones(len(y_all), bool),
                     "monsoon": (se == "monsoon"), "non-monsoon": (se == "non-monsoon")}
            variants = {"raw": z[f"p::{name}::{hor}"]}
            if name in flagged:
                pc = iso_oof(z[f"p::{name}::{hor}"], y_all, z[f"st::{hor}"])
                if not np.isnan(pc).any():
                    variants["iso"] = pc
            for vname, pv in variants.items():
                own = {per: best_exact(y_all[masks[per]], pv[masks[per]], CFN_PRIMARY)[0]
                       for per in PERIODS}
                for src in PERIODS:
                    for tgt in PERIODS:
                        mt = masks[tgt]
                        mm = metrics_at(y_all[mt], pv[mt], own[src])
                        self_rec = metrics_at(y_all[mt], pv[mt], own[tgt])["recall"]
                        transfer.append(dict(
                            model=name, level=lvl.get(name, "?"), horizon=HOR_DISP[hor],
                            variant=vname, source=src, target=tgt, t_source=own[src],
                            t_target_own=own[tgt], self_recall=self_rec,
                            target_recall=mm["recall"], target_precision=mm["precision"],
                            target_far=mm["far"], target_cost_per1k=mm["cost_fn10_per1k"],
                            recall_loss=self_rec - mm["recall"]))
    transfer = pd.DataFrame(transfer)

    # ------------------------------------------------------------- risk bands
    allc = chosen[chosen.period == "all"]
    bands = []
    leaders = {}
    for hor in HOR:
        d = allc[allc.horizon == HOR_DISP[hor]]
        lead = d.sort_values("pr_auc", ascending=False).iloc[0]["model"]
        leaders[HOR_DISP[hor]] = lead
        for name in dict.fromkeys([lead, "fusion_L3", "transformer_bestL"]):
            y = z[f"y::{hor}"]
            p = z[f"p::{name}::{hor}"]
            tw = float(d[d.model == name].iloc[0]["t_warn"]) if (d.model == name).any() \
                else float(allc[(allc.model == name) & (allc.horizon == HOR_DISP[hor])]
                           .iloc[0]["t_warn"])
            bands += band_table(y, p, choose_cuts(y, p, tw), name, HOR_DISP[hor],
                                lvl.get(name, "?"), "cost-derived")
            bands += band_table(y, p, tuple(np.quantile(p, [0.25, 0.5, 0.75])), name,
                                HOR_DISP[hor], lvl.get(name, "?"), "quartile")
    bands = pd.DataFrame(bands)

    # ---------------------------------------------------------------- write
    sweep.to_csv(REPORTS / "phase29_threshold_sweep.csv", index=False)
    chosen.to_csv(REPORTS / "phase29_chosen_thresholds.csv", index=False)
    transfer.to_csv(REPORTS / "phase29_transfer.csv", index=False)
    bands.to_csv(REPORTS / "phase29_risk_bands.csv", index=False)
    legend.to_csv(REPORTS / "phase29_pr_auc.csv", index=False)
    print(f"wrote 5 csvs: sweep {len(sweep)}, chosen {len(chosen)}, "
          f"transfer {len(transfer)}, bands {len(bands)}, legend {len(legend)}")

    # --------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"L1": "#888888", "L2": "#1f77b4", "L3": "#2ca02c", "L4": "#d62728"}
        leadset = list(dict.fromkeys(leaders.values()))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, hor in zip(axes, HOR):
            d = sweep[(sweep.horizon == HOR_DISP[hor]) & (sweep.period == "all")]
            for name in leadset:
                dd = d[d.model == name]
                if dd.empty:
                    continue
                ax.plot(dd.threshold, dd.cost_fn10_per1k,
                        color=cols.get(lvl.get(name, "?"), "k"), lw=1.6, label=name)
            ax.set_title(f"{HOR_DISP[hor]} — expected cost (FN:FP = 10:1)")
            ax.set_xlabel("threshold"); ax.set_ylabel("cost per 1,000 rows")
            ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.suptitle("Phase 29 — cost-optimal threshold search (P1 val_2022, all-season)")
        fig.tight_layout()
        fig.savefig(FIGURES / "phase29_threshold_cost.png", dpi=130)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax = axes[0]
        for name in leadset:
            p = z[f"p::{name}::label_24h"]
            y = z["y::label_24h"]
            o = np.argsort(-p)
            yy = y[o]
            tp = np.cumsum(yy)
            rec = tp / tp[-1]
            prec = tp / np.arange(1, len(yy) + 1)
            ax.plot(rec, prec, lw=1.6, label=name)
        ax.set_title("24 h precision–recall"); ax.set_xlabel("recall")
        ax.set_ylabel("precision"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax = axes[1]
        d = transfer[(transfer.variant == "raw") & (transfer.source == "non-monsoon")
                     & (transfer.target == "monsoon") & (transfer.horizon == "48h")]
        d = d.sort_values("recall_loss", ascending=False).head(8)
        ax.barh(range(len(d)), d.recall_loss, color="#d62728")
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.model, fontsize=7)
        ax.set_title("non-monsoon → monsoon recall loss (48 h)")
        ax.set_xlabel("recall lost when the threshold is carried"); ax.grid(alpha=.3, axis="x")
        ax = axes[2]
        bd = bands[bands.scheme == "cost-derived"]
        order = ["low", "moderate", "high", "very high"]
        w = 0.35
        names24 = list(dict.fromkeys(bd[bd.horizon == "24h"].model))
        for i, name in enumerate(names24):
            dd = bd[(bd.horizon == "24h") & (bd.model == name)].set_index("band").reindex(order)
            ax.bar(np.arange(4) + (i - 0.5) * w, dd.flood_rate, w, label=name)
        ax.set_xticks(range(4)); ax.set_xticklabels(order)
        ax.set_title("24 h observed flood rate per band")
        ax.set_ylabel("observed flood rate"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
        fig.suptitle("Phase 29 — discrimination, period transfer and risk bands")
        fig.tight_layout()
        fig.savefig(FIGURES / "phase29_bands_transfer.png", dpi=130)
        print("wrote 2 figures")
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)

    # --------------------------------------------------------------- summary
    lead_thr = {}
    for hor in HOR:
        lead = leaders[HOR_DISP[hor]]
        r = allc[(allc.model == lead) & (allc.horizon == HOR_DISP[hor])].iloc[0]
        lead_thr[HOR_DISP[hor]] = dict(
            model=lead, t_warn=round(float(r.t_warn), 4), recall=round(float(r.recall), 4),
            precision=round(float(r.precision), 4), far=round(float(r.far), 4),
            missed_rows=int(r.missed), alarm_rate=round(float(r.alarm_rate), 4),
            cost_per1k=round(float(r.cost_per1k), 3))
    tr_lead = {}
    for hor in HOR:
        lead = leaders[HOR_DISP[hor]]
        r = transfer[(transfer.model == lead) & (transfer.variant == "raw")
                     & (transfer.source == "non-monsoon") & (transfer.target == "monsoon")
                     & (transfer.horizon == HOR_DISP[hor])]
        if len(r):
            r = r.iloc[0]
            tr_lead[HOR_DISP[hor]] = dict(recall_loss=round(float(r.recall_loss), 4),
                                          self_recall=round(float(r.self_recall), 4),
                                          carried_recall=round(float(r.target_recall), 4))
    band_sep = {}
    for hor in HOR:
        lead = leaders[HOR_DISP[hor]]
        dd = bands[(bands.scheme == "cost-derived") & (bands.model == lead)
                   & (bands.horizon == HOR_DISP[hor])]
        rates = dd.flood_rate.tolist()
        band_sep[HOR_DISP[hor]] = bool(len(rates) == 4 and rates[-1] > rates[0]
                                       and rates[-2] > rates[0])
    lead_thr_period = {}
    for hor in HOR:
        lead = leaders[HOR_DISP[hor]]
        d = chosen[(chosen.model == lead) & (chosen.horizon == HOR_DISP[hor])]
        lead_thr_period[HOR_DISP[hor]] = {r.period: round(float(r.t_warn), 4)
                                          for _, r in d.iterrows()}
    w = transfer[(transfer.variant == "raw") & (transfer.source == "non-monsoon")
                 & (transfer.target == "monsoon")]
    worst = w.sort_values("recall_loss", ascending=False).head(6)[
        ["model", "horizon", "recall_loss"]].round(4).to_dict("records")
    summary = {
        "phase": 29, "stages": [48, 49], "seed": SEED,
        "cost_statement": {"C_FN": CFN_PRIMARY, "C_FP": 1,
                           "sensitivity": sorted(COST_RATIOS.values()),
                           "note": "row-level proxy; missing a flood is far worse than a false alarm"},
        "threshold_grid": [GRID[0], GRID[-1], len(GRID)],
        "leaders": leaders,
        "frozen_thresholds_all_season": lead_thr,
        "frozen_thresholds_per_period": lead_thr_period,
        "period_transfer_dry_to_wet_leaders": tr_lead,
        "worst_dry_to_wet_transfer_raw": worst,
        "band_separation_ok_leaders": band_sep,
        "n_flagged_needing_calibrated_decision": sorted(flagged),
        "test_rows_read": 0,
    }
    (REPORTS / "phase29_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")

    print(f"\n=== Phase 29 summary ({time.time() - t0:.0f}s) ===")
    print(f"cost: FN:FP = {int(CFN_PRIMARY)}:1 (sensitivity 5, 20); grid {GRID[0]}..{GRID[-1]}")
    for h in HOR:
        r = lead_thr[HOR_DISP[h]]
        print(f"  {HOR_DISP[h]:>4} leader {r['model']:18s} t={r['t_warn']:.3f} "
              f"recall={r['recall']:.3f} prec={r['precision']:.3f} FAR={r['far']:.3f} "
              f"missed={r['missed_rows']}")
        t = tr_lead.get(HOR_DISP[h], {})
        print(f"        dry→wet carried recall {t.get('carried_recall')} vs own "
              f"{t.get('self_recall')} (loss {t.get('recall_loss')}); bands separate "
              f"{band_sep[HOR_DISP[h]]}")
    print(f"\nwrote reports/phase29_*.csv|json + figures/phase29_*.png  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
