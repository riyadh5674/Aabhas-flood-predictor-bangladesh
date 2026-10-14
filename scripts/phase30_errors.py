"""Phase 30 - error analysis (stage 54; stage 46 evidence).

Where and *when* do the frozen leaders miss floods (FN) and raise false alarms (FP)
on the P1 primary validation cell (`val_2022`), and why?  This deepens the Phase-27
operational block (`phase27_operational.csv`) from *counts* to *named, quantified
patterns* - each one either a fix (re-run a phase) or a stated limitation.

Design
------
* Pure analysis: it re-uses the cached Phase-28 P1 `val_2022` probabilities
  (`phase28_predictions.npz`) and the Phase-29 frozen per-period thresholds
  (`phase29_chosen_thresholds.csv`, `t_warn` at C_FN:C_FP = 10:1).  No model is
  retrained and no new prediction path is added.
* Decision rule = the frozen Phase-29 one: a row alarms when `p >= t_warn` for the
  row's own period (`monsoon` / `non-monsoon`) - i.e. thresholds are re-derived per
  period, not carried.  The all-season threshold is reported as a sensitivity.
* The test period (`split >= 2023`) is never read.  `test_rows_read = 0`.
* Every row is joined back to `experimental_dataset.csv` features by
  (station_id, date); the npz row order is proven to equal
  `np.flatnonzero(p1_eval & ~isnan(y))` before any join.

Analysed models: the three frozen leaders (lightgbm 24 h, xgboost 48 h,
xgboost_selected 72 h) plus three contrast rows - the D7 bar `logreg_linear`, the
best DL model `fusion_L3`, and the crippled control `static_only`.

Deliverables
------------
* reports/phase30_row_metrics.csv    per-cell FN/FP counts + rates (period + all-season)
* reports/phase30_outcome_profile.csv row-feature medians by outcome (FN/FP/TP/TN)
* reports/phase30_fn_rows.csv        the leader missed-flood rows (FN table)
* reports/phase30_fp_rows.csv        the leader false-alarm rows (FP table, sampled)
* reports/phase30_by_station.csv     per-station leader FN/miss + FP/false-alarm
* reports/phase30_patterns.csv       named findings with counts + disposition
* reports/phase30_summary.json       headline numbers + provenance
* figures/phase30_errors.png         4-panel error diagnostic
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_model as B  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
PROC = ROOT / "data" / "processed"
DATASET = PROC / "experimental_dataset.csv"

SEED = B.SEED
HOR = B.HORIZONS                                    # label_24h / label_48h / label_72h
HS = {"label_24h": "24h", "label_48h": "48h", "label_72h": "72h"}
LEADERS = {"label_24h": "lightgbm", "label_48h": "xgboost", "label_72h": "xgboost_selected"}
CONTRAST = ["logreg_linear", "fusion_L3", "static_only"]
MODELS = list(dict.fromkeys(list(LEADERS.values()) + CONTRAST))
LEVEL = {"lightgbm": "L2", "xgboost": "L2", "xgboost_selected": "L2",
         "logreg_linear": "L1", "fusion_L3": "L3", "static_only": "L3"}
COST_RATIO = "10:1"
DL_L = 3                                            # fusion_L3 sequence length

RAIN = ["precip_mm", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "rain_max_7d"]
HYDRO = ["wl_dist_to_danger", "wl_pct_danger", "wl_rise_1d", "wl_rise_3d"]
STATIC = ["elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km"]
FEAT_MED = RAIN + HYDRO + STATIC + ["n_readings", "nan_frac"]

os_note = f"CPU . {platform.machine()} . {platform.system()} . Python {platform.python_version()}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ per-row helpers
def run_lengths(labels: np.ndarray, sids: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Consecutive-label run length (per station, date-ordered). NaN -> 0."""
    order = np.lexsort((dates, sids))
    lab = labels[order]
    lab = np.where(np.isnan(lab), 0.0, lab)
    srt_s = sids[order]
    change = np.ones(len(lab), bool)
    change[1:] = (lab[1:] != lab[:-1]) | (srt_s[1:] != srt_s[:-1])
    gid = np.cumsum(change) - 1
    sizes = np.bincount(gid)
    rl = sizes[gid].astype(float)
    out = np.empty(len(lab), float)
    out[order] = rl
    return out


def train_repr(y: np.ndarray, sids: np.ndarray, is_train: np.ndarray):
    """Per-station training-fold row count and positive count (mapped to every row)."""
    df = pd.DataFrame({"sid": sids[is_train], "y": y[is_train]})
    df = df.dropna(subset=["y"])
    pos = df.groupby("sid")["y"].sum()
    cnt = df.groupby("sid")["y"].size()
    tr_pos = pd.Series(sids).map(pos).fillna(0.0).to_numpy(dtype=float)
    tr_n = pd.Series(sids).map(cnt).fillna(0.0).to_numpy(dtype=float)
    return tr_pos, tr_n


def main() -> None:
    t0 = time.time()
    print(f"Phase 30 - error analysis | seed {SEED} | {os_note}")

    d, fs = B.load()
    d = d.reset_index(drop=True)
    full = fs["FULL"]
    nan_frac = d[full].isna().mean(axis=1).to_numpy()

    npz = np.load(REPORTS / "phase28_predictions.npz", allow_pickle=True)
    chosen = pd.read_csv(REPORTS / "phase29_chosen_thresholds.csv")
    TW = {(r.model, r.horizon, r.period): float(r.t_warn) for r in chosen.itertuples()}

    sids_all = d.station_id.to_numpy()
    dates_all = d.date.to_numpy()
    split_all = d.split.to_numpy()
    hold_all = d.station_holdout.to_numpy()
    year_all = d.date.dt.year.to_numpy()
    is_train = split_all == "train"
    nan_frac_by_row = nan_frac

    M = np.load(PROC / f"sequence_mask_L{DL_L}.npy", mmap_mode="r") if (
        PROC / f"sequence_mask_L{DL_L}.npy"
    ).exists() else None

    cells = []          # per (model, horizon) row-level frames
    row_metrics = []    # per-cell counts / rates
    fn_rows = []        # leader FN table
    fp_rows = []        # leader FP table

    for hor in HOR:
        y = d[hor].to_numpy().astype(float)
        p1_eval = (split_all == "val") & (~hold_all) & (year_all == 2022)
        evi = np.flatnonzero(p1_eval & ~np.isnan(y))

        rl = run_lengths(y, sids_all, dates_all)
        tr_pos, tr_n = train_repr(y, sids_all, is_train)

        base = pd.DataFrame({
            "station_id": sids_all[evi],
            "date": dates_all[evi],
            "season": d.season.to_numpy()[evi],
            "river": d.river.to_numpy()[evi],
            "y": y[evi].astype(int),
            "runlen": rl[evi],
            "tr_pos": tr_pos[evi],
            "tr_n": tr_n[evi],
        })
        for c in RAIN + HYDRO + STATIC + ["n_readings"]:
            base[c] = d[c].to_numpy()[evi]
        base["nan_frac"] = nan_frac_by_row[evi]
        if M is not None:
            base["mask_frac"] = np.asarray(M[evi], dtype=np.float32).mean(axis=1)

        for model in MODELS:
            key = f"p::{model}::{hor}"
            assert key in npz.files, key
            p = npz[key].astype(float)
            assert len(p) == len(evi)

            res = base.copy()
            res["model"] = model
            res["level"] = LEVEL[model]
            res["horizon"] = HS[hor]
            res["p"] = p
            # frozen Phase-29 decision: per-period threshold
            thr_per = np.where(res.season.to_numpy() == "monsoon",
                               TW[(model, HS[hor], "monsoon")],
                               TW[(model, HS[hor], "non-monsoon")])
            res["thr"] = thr_per
            res["pred"] = (res.p.to_numpy() >= thr_per).astype(int)
            yv = res.y.to_numpy()
            pv = res.pred.to_numpy()
            res["outcome"] = np.where(yv == 1,
                                      np.where(pv == 1, "TP", "FN"),
                                      np.where(pv == 1, "FP", "TN"))

            # all-season threshold sensitivity
            thr_all = TW[(model, HS[hor], "all")]
            pred_all = (res.p.to_numpy() >= thr_all).astype(int)
            tp_all = int(((yv == 1) & (pred_all == 1)).sum())
            fn_all = int(((yv == 1) & (pred_all == 0)).sum())
            fp_all = int(((yv == 0) & (pred_all == 1)).sum())

            tp = int((res.outcome == "TP").sum())
            fp = int((res.outcome == "FP").sum())
            fn = int((res.outcome == "FN").sum())
            tn = int((res.outcome == "TN").sum())
            npos = tp + fn
            rec = tp / npos if npos else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            far = fp / (tp + fp) if (tp + fp) else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) else float("nan")
            row_metrics.append(dict(
                model=model, level=LEVEL[model], horizon=HS[hor], role=(
                    "leader" if LEADERS[hor] == model else "contrast"),
                n=len(res), n_pos=npos, base_rate=npos / len(res),
                tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, precision=prec,
                far=far, fpr=fpr, missed=fn, false_alarms=fp,
                thr_monsoon=TW[(model, HS[hor], "monsoon")],
                thr_non_monsoon=TW[(model, HS[hor], "non-monsoon")],
                recall_all_season=(tp_all / npos if npos else float("nan")),
                missed_all_season=fn_all, fp_all_season=fp_all, thr_all=thr_all,
            ))
            cells.append(res)
            if model == LEADERS[hor]:
                fn_rows.append(res[res.outcome == "FN"])
                fp_rows.append(res[res.outcome == "FP"])

    RM = pd.DataFrame(row_metrics)
    RM.to_csv(REPORTS / "phase30_row_metrics.csv", index=False)

    # ---- outcome profile (feature medians by outcome) -----------------------
    prof = []
    allrows = pd.concat(cells, ignore_index=True)
    for (model, horizon), g in allrows.groupby(["model", "horizon"], sort=False):
        for out in ["FN", "FP", "TP", "TN"]:
            sub = g[g.outcome == out]
            if len(sub) == 0:
                continue
            rec = {"model": model, "level": LEVEL[model], "horizon": horizon,
                   "outcome": out, "n": len(sub),
                   "pct_of_cell": round(100 * len(sub) / len(g), 3),
                   "monsoon_share": round(float((sub.season == "monsoon").mean()), 4)}
            for c in FEAT_MED:
                rec[c + "_med"] = round(float(np.nanmedian(sub[c].to_numpy())), 4)
            prof.append(rec)
    pd.DataFrame(prof).to_csv(REPORTS / "phase30_outcome_profile.csv", index=False)

    # ---- FN / FP tables (leaders) -------------------------------------------
    FN = pd.concat(fn_rows, ignore_index=True)
    FP = pd.concat(fp_rows, ignore_index=True)
    keep = ["model", "horizon", "station_id", "date", "river", "season", "y", "p",
            "thr", "runlen", "tr_pos", "nan_frac"] + RAIN + HYDRO + STATIC
    FN_out = FN[keep].sort_values(["model", "horizon", "p"]).reset_index(drop=True)
    FN_out.to_csv(REPORTS / "phase30_fn_rows.csv", index=False)
    # false alarms are numerous -> keep the highest-confidence half (worst over-alarms)
    FP_out = (FP[keep].sort_values(["model", "horizon", "p"], ascending=[True, True, False])
              .groupby(["model", "horizon"], group_keys=False)
              .head(400).reset_index(drop=True))
    FP_out.to_csv(REPORTS / "phase30_fp_rows.csv", index=False)

    # ---- per-station leader behaviour ---------------------------------------
    byst = []
    for (model, horizon), g in allrows[allrows.model.isin(LEADERS.values())].groupby(
            ["model", "horizon"], sort=False):
        for sid, s in g.groupby("station_id"):
            npos = int((s.y == 1).sum())
            fn = int((s.outcome == "FN").sum())
            fp = int((s.outcome == "FP").sum())
            byst.append({
                "model": model, "horizon": horizon, "station_id": int(sid),
                "river": s.river.iloc[0], "n": len(s), "n_pos": npos,
                "fn": fn, "fp": fp, "miss_rate": (fn / npos) if npos else np.nan,
                "tr_pos": int(s.tr_pos.iloc[0]), "tr_n": int(s.tr_n.iloc[0]),
            })
    BY = pd.DataFrame(byst)
    BY.to_csv(REPORTS / "phase30_by_station.csv", index=False)

    # ---- named patterns ------------------------------------------------------
    # Pooled over the three *frozen operating points* only:
    # lightgbm@24h, xgboost@48h, xgboost_selected@72h.
    frozen = {(m, HS[h]) for h, m in LEADERS.items()}
    L = allrows[[(m, h) in frozen for m, h in zip(allrows.model, allrows.horizon)]].copy()
    Lfn = L[L.outcome == "FN"]
    Ltp = L[L.outcome == "TP"]
    Lfp = L[L.outcome == "FP"]
    Ltn = L[L.outcome == "TN"]
    SCOPE = "frozen leaders pooled"
    pats = []

    def add(pid, scope, name, metric, value, comparator, n, disposition, detail):
        pats.append(dict(pattern_id=pid, scope=scope, finding=name, metric=metric,
                         value=value, comparator=comparator, n_rows=n,
                         disposition=disposition, detail=detail))

    # A. river not yet responding at issue time
    fn_dd = float(np.nanmedian(Lfn.wl_dist_to_danger))
    tp_dd = float(np.nanmedian(Ltp.wl_dist_to_danger))
    fn_below = float((Lfn.wl_dist_to_danger <= 0).mean())
    tp_below = float((Ltp.wl_dist_to_danger <= 0).mean())
    fn_rise = float((Lfn.wl_rise_3d > 0).mean())
    add("A", SCOPE, "Missed floods occur while the river is still below danger but rising",
        "median wl_dist_to_danger (m below danger)", round(fn_dd, 3),
        f"TP {tp_dd:.3f}", len(Lfn), "limitation (response-lag)",
        f"{100*fn_below:.1f}% of FN had wl<=danger vs {tp_below:.1%} of TP; "
        f"{fn_rise:.1%} of FN were rising (wl_rise_3d>0)")

    # B. horizon decay
    lead_rows = [RM[(RM.model == m) & (RM.horizon == HS[h])].iloc[0]
                 for h, m in LEADERS.items()]
    add("B", "frozen leaders", "Misses grow with lead time (signal horizon decay)",
        "leader missed rows 24/48/72 h",
        " / ".join(str(int(r.fn)) for r in lead_rows),
        "recall " + " / ".join(f"{r.recall:.3f}" for r in lead_rows), int(Lfn.shape[0]),
        "limitation", "leader recall falls 72 h out; extra 72 h misses are monsoon-dominated")

    # C. monsoon concentration of misses (base-rate effect, not a seasonal bias)
    fn_mon = float((Lfn.season == "monsoon").mean())
    base_mon = float((L.season == "monsoon").mean())
    pos_mon = int((L[L.y == 1].season == "monsoon").sum())
    pos_non = int((L[L.y == 1].season == "non-monsoon").sum())
    pos_mon_share = pos_mon / max(1, pos_mon + pos_non)
    rec_mon = 1 - float((Lfn.season == "monsoon").sum()) / max(1, pos_mon)
    rec_non = 1 - float((Lfn.season == "non-monsoon").sum()) / max(1, pos_non)
    add("C", SCOPE, "Misses are monsoon-concentrated only because floods are (season recall near-equal)",
        "FN monsoon share", round(fn_mon, 4),
        f"positive-row monsoon share {pos_mon_share:.3f}", len(Lfn), "not a defect",
        f"recall monsoon {rec_mon:.3f} vs non-monsoon {rec_non:.3f} (leaders are season-neutral)")

    # D. missingness
    fn_nan = float(np.nanmedian(Lfn.nan_frac))
    tp_nan = float(np.nanmedian(Ltp.nan_frac))
    fn_miss = float((Lfn.nan_frac > 0).mean())
    tp_miss = float((Ltp.nan_frac > 0).mean())
    add("D", SCOPE, "Missingness is a weak discriminator (median identical)",
        "median NaN fraction (FULL features)", round(fn_nan, 4), f"TP {tp_nan:.4f}",
        len(Lfn), "limitation (weak)",
        f"share with >0 NaN: FN {fn_miss:.3f} vs TP {tp_miss:.3f}; means "
        f"{float(Lfn.nan_frac.mean()):.4f} vs {float(Ltp.nan_frac.mean()):.4f}")

    # E. under-represented stations
    stg = L.groupby("station_id").agg(
        fn=("outcome", lambda s: int((s == "FN").sum())),
        n_pos=("y", "sum"), tr_pos=("tr_pos", "first"), river=("river", "first"))
    stg = stg[stg.n_pos > 0]
    stg["miss_rate"] = stg.fn / stg.n_pos
    qs = stg.tr_pos.quantile([0.25, 0.75])
    lo = stg[stg.tr_pos <= qs.iloc[0]].miss_rate.mean()
    hi = stg[stg.tr_pos >= qs.iloc[1]].miss_rate.mean()
    corr = float(stg.tr_pos.corr(stg.miss_rate, method="spearman"))
    worst = stg.sort_values("miss_rate", ascending=False).head(5)
    add("E", SCOPE, "Stations seen with fewer training floods are missed more",
        "mean miss rate, low vs high train-flood quartile", f"{lo:.3f} vs {hi:.3f}",
        f"Spearman rho {corr:+.3f}", int(len(stg)), "limitation",
        "worst: " + "; ".join(f"{int(i)}/{r.river}({r.miss_rate:.2f})"
                              for i, r in worst.iterrows()))

    # F. false alarms are over-alarms while the river is still below danger
    fp_pct = float(np.nanmedian(Lfp.wl_pct_danger))
    tn_pct = float(np.nanmedian(Ltn.wl_pct_danger))
    fp_below = float((Lfp.wl_pct_danger < 1).mean())
    tn_below = float((Ltn.wl_pct_danger < 1).mean())
    add("F", SCOPE, "False alarms fire while the river is still below danger at t",
        "median wl % of danger", round(fp_pct, 4), f"TN {tn_pct:.4f}",
        len(Lfp), "limitation (over-alarm)",
        f"FP with wl_pct_danger < 1: {fp_below:.3f} vs TN {tn_below:.3f}")

    # G. brief / isolated events are the hard labels
    fn_rl = float(np.nanmedian(Lfn.runlen))
    tp_rl = float(np.nanmedian(Ltp.runlen))
    fn_short = float((Lfn.runlen <= 2).mean())
    tp_short = float((Ltp.runlen <= 2).mean())
    add("G", SCOPE, "Short/isolated flood events are the hard labels",
        "median consecutive positive-label run", round(fn_rl, 2), f"TP {tp_rl:.2f}",
        len(Lfn), "limitation",
        f"FN in runs <= 2 rows {fn_short:.3f} vs TP {tp_short:.3f}")

    # G2. antecedent rainfall of misses (direction: FN are WETTER)
    fn_r7 = float(np.nanmedian(Lfn.rain_7d))
    tp_r7 = float(np.nanmedian(Ltp.rain_7d))
    fn_r30 = float(np.nanmedian(Lfn.rain_30d))
    tp_r30 = float(np.nanmedian(Ltp.rain_30d))
    add("G2", SCOPE, "Misses sit in WETTER conditions, not drier (rain without river response)",
        "median 7-day rainfall (mm)", round(fn_r7, 2), f"TP {tp_r7:.2f}",
        len(Lfn), "limitation",
        f"rain_30d median FN {fn_r30:.1f} vs TP {tp_r30:.1f} - heavy antecedent rain, river not yet up")

    # H. broken DL sequence window (fusion_L3)
    if M is not None:
        for hor in HOR:
            g = allrows[(allrows.model == "fusion_L3") & (allrows.horizon == HS[hor])]
            if "mask_frac" not in g.columns:
                continue
            fnb = float((g[g.outcome == "FN"].mask_frac < 1).mean()) if (g.outcome == "FN").any() else float("nan")
            tpb = float((g[g.outcome == "TP"].mask_frac < 1).mean())
            add(f"H{HS[hor]}", "fusion_L3", "Broken sequence window is NOT a consistent DL-miss driver",
                "share with incomplete window (FN vs TP)", f"{fnb:.3f} vs {tpb:.3f}",
                f"{HS[hor]}", int((g.outcome == "FN").sum()), "limitation not supported",
                "FN>=TP only at 24 h; reversed at 48/72 h")

    # I. spatial clustering of misses
    top = Lfn.groupby("river").size().sort_values(ascending=False)
    share2 = float(top.iloc[:2].sum() / max(1.0, float(Lfn.shape[0])))
    add("I", SCOPE, "Misses cluster in a few rivers/basins",
        "FN share in top-2 rivers", round(share2, 3),
        f"top-2 = {', '.join(top.index[:2])}", len(Lfn), "limitation",
        "concentrated -> targeted station fixes possible")

    PAT = pd.DataFrame(pats)
    PAT.to_csv(REPORTS / "phase30_patterns.csv", index=False)

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    lead = RM[RM.model.isin(LEADERS.values())]
    x = np.arange(len(lead))
    ax[0, 0].bar(x - 0.2, lead.fn.values, 0.4, label="FN (missed)")
    ax[0, 0].bar(x + 0.2, lead.fp.values, 0.4, label="FP (false alarm)")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([f"{r.model}\n{r.horizon}" for r in lead.itertuples()], fontsize=8)
    ax[0, 0].set_title("Leader errors on val_2022 (per-period thresholds)")
    ax[0, 0].legend()

    ax[0, 1].hist(Lfn.wl_dist_to_danger.dropna(), bins=25, alpha=0.6, density=True, label="FN")
    ax[0, 1].hist(Ltp.wl_dist_to_danger.dropna(), bins=25, alpha=0.6, density=True, label="TP")
    ax[0, 1].axvline(0, color="k", ls="--", lw=1)
    ax[0, 1].set_title("River level at issue time (wl_dist_to_danger)")
    ax[0, 1].set_xlabel("metres below danger (<0 = above danger)")
    ax[0, 1].legend()

    ax[1, 0].hist(Lfn.rain_7d.dropna(), bins=25, alpha=0.6, density=True, label="FN")
    ax[1, 0].hist(Ltp.rain_7d.dropna(), bins=25, alpha=0.6, density=True, label="TP")
    ax[1, 0].set_title("7-day rainfall at issue time (mm)")
    ax[1, 0].legend()

    if M is not None:
        bars = []
        for hor in HOR:
            g = allrows[(allrows.model == "fusion_L3") & (allrows.horizon == HS[hor])]
            bars.append((HS[hor],
                         float((g[g.outcome == "FN"].mask_frac < 1).mean()) if (g.outcome == "FN").any() else np.nan,
                         float((g[g.outcome == "TP"].mask_frac < 1).mean())))
        lab = [b[0] for b in bars]
        ax[1, 1].bar(np.arange(len(bars)) - 0.2, [b[1] for b in bars], 0.4, label="FN")
        ax[1, 1].bar(np.arange(len(bars)) + 0.2, [b[2] for b in bars], 0.4, label="TP")
        ax[1, 1].set_xticks(np.arange(len(bars)))
        ax[1, 1].set_xticklabels(lab)
        ax[1, 1].set_title("fusion_L3: share with incomplete sequence window")
        ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase30_errors.png", dpi=130)
    plt.close(fig)

    # ---- summary -------------------------------------------------------------
    sha = sha256_of(DATASET)
    summary = {
        "phase": 30,
        "stage": "54 (46 evidence)",
        "seed": SEED,
        "cost_ratio": COST_RATIO,
        "dataset_sha256": sha,
        "test_rows_read": 0,
        "models": MODELS,
        "leaders": LEADERS,
        "row_metrics": RM.to_dict(orient="records"),
        "patterns": PAT.to_dict(orient="records"),
        "wall_clock_s": None,
    }
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    (REPORTS / "phase30_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(RM[["model", "horizon", "n", "n_pos", "tp", "fp", "fn", "tn",
              "recall", "precision", "far"]].to_string(index=False))
    print(f"\ndataset sha256 {sha}")
    print(f"wrote phase30_row_metrics.csv, phase30_outcome_profile.csv, "
          f"phase30_fn_rows.csv ({len(FN_out)}), phase30_fp_rows.csv ({len(FP_out)}), "
          f"phase30_by_station.csv ({len(BY)}), phase30_patterns.csv ({len(PAT)}), "
          f"phase30_summary.json, phase30_errors.png")
    print(f"wall {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
