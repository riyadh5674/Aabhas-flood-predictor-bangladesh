"""Phase 32 - ablation & comparative experiment matrix (workflow Phase 32 / stages 55, 56).

Question
--------
Which information actually contributes to flood prediction, and how do the model
families compare once the input is the only thing that changes?  The phase runs one
planned matrix (E0-E16) once and reports it as a research story - every result per
cell, never pooled (D8).

Design
------
* **E1-E5, E11 (new run).**  A *nested feature-group ablation* holding the learner
  fixed and varying only the inputs.  The FULL 62-feature set is partitioned into five
  disjoint, cumulative groups:

      G1 weather                     (5)  tmax/tmin/precip + seasonal sin/cos
      G2 + rainfall history          (7)  rain_1d..30d, rain_max_7d, wet_days_7
      G3 + geography                (14)  lat/lon/elevation/slope/distances + 5 admin text
      G4 + hydrology                (24)  danger_level + local water-level family
      G5 + station network          (12)  nb3_/nb5_ neighbourhood + up_* upstream

  G5 == FULL (asserted).  **E11** repeats G5 with the four `up_*` columns removed
  (F14-2) - the G5 vs G5_no_up contrast.
  Two fixed learners (the Phase-20 winners `xgboost_c01` and `lightgbm_c00`) are fit
  per group; the *same* hyper-parameters are used for every group so the feature
  contribution is isolated from the tuning.
* **E0/E6/E7/E8-E10** fall out of the same run (references; full model on spatial and
  monsoon cells; full model per horizon).
* **E12-E16 (reuse)** - already-run evidence is integrated from the frozen Phase-16/
  21/23/24/25/26/27 artefacts; E12 is explicitly *blocked* by R8 (no soil/landcover
  groups exist in the dataset).

Reporting: PR-AUC is primary (micro = pooled, macro = mean per-station);
station-bootstrap 95% CI; paired station-bootstrap CI on the *increments* between
consecutive groups; ROC-AUC/Brier/ECE companions.  The frozen test split is never
read (`test_rows_read = 0`).

Run:  .venv\\Scripts\\python.exe -u scripts/phase32_ablation.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_model as B          # noqa: E402
# NOTE: phase20_classical_ml calls sys.stdout.reconfigure() at import time, which fails
# under Jupyter's OutStream, so it is imported lazily inside main() (notebooks import this
# module only for its constants).

REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
PROC = ROOT / "data" / "processed"
DATA = PROC / "experimental_dataset.csv"
DATA_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"

SEED = B.SEED
BOOT_REPS = int(getattr(B, "BOOT_REPS", 100))
HOR = list(B.HORIZONS)
HORD = {h: h.replace("label_", "") for h in HOR}
SEASONS = ("all", "monsoon", "non-monsoon")

CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
TEMPORAL_CELLS = ["val_2020", "val_2021", "val_2022"]
SPATIAL_CELLS = ["spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
LEARNERS = ["xgboost", "lightgbm"]

# ---------------------------------------------------------------- feature partition
WEATHER = ["tmax_c", "tmin_c", "precip_mm", "doy_sin", "doy_cos"]
RAIN_HIST = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
             "rain_max_7d", "wet_days_7"]
GEO = ["river", "basin", "division", "district", "upazilla",
       "lat", "lon", "elevation_m", "slope_deg", "dist_to_coast_km",
       "dist_river_km", "dist_major_river_km", "river_density_10km_km_per_km2",
       "n_river_lines_10km"]
HYDRO = ["danger_level", "n_readings", "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last",
         "wl_last_lag1", "wl_last_lag2", "wl_last_lag3", "wl_last_lag4",
         "wl_last_lag5", "wl_last_lag6", "wl_last_lag7",
         "wl_max_lag1", "wl_max_lag2", "wl_max_lag3",
         "wl_rise_1d", "wl_rise_3d", "wl_dist_to_danger", "wl_pct_danger",
         "wl_max_roll7", "wl_max_roll30", "wl_exceed_days_30"]
UP = ["up_rain7d_lag1", "up_rain7d_lag2", "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2"]
NETWORK = ["nb3_rain_7d", "nb3_rain_14d", "nb3_wl_pct_danger",
           "nb5_rain_7d", "nb5_rain_14d", "nb5_wl_pct_danger",
           "nb3_n_wl_danger", "nb5_n_wl_danger"] + UP

GROUP_DEF: list[tuple[str, list[str]]] = [
    ("G1_weather", WEATHER),
    ("G2_weather_rainhist", WEATHER + RAIN_HIST),
    ("G3_plus_geography", WEATHER + RAIN_HIST + GEO),
    ("G4_plus_hydrology", WEATHER + RAIN_HIST + GEO + HYDRO),
    ("G5_full", WEATHER + RAIN_HIST + GEO + HYDRO + NETWORK),
    ("G5_no_up", WEATHER + RAIN_HIST + GEO + HYDRO + [c for c in NETWORK if c not in UP]),
]
_INCREMENTS = [("G2_weather_rainhist", "G1_weather"),
        ("G3_plus_geography", "G2_weather_rainhist"),
        ("G4_plus_hydrology", "G3_plus_geography"),
        ("G5_full", "G4_plus_hydrology"),
        ("G5_no_up", "G5_full")]      # negative increment => up_* unhelpful
REFS = ["persistence", "climatology"]


# ------------------------------------------------------------------------------ utils
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def paired_boot(stations: np.ndarray, y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                reps: int = BOOT_REPS) -> tuple[float, float]:
    """Paired station bootstrap 95% CI on PR-AUC(a) - PR-AUC(b) (D8)."""
    from sklearn.metrics import average_precision_score
    uniq = np.unique(stations)
    if len(uniq) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx_by = {s: np.flatnonzero(stations == s) for s in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]))
    if len(vals) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def dev_tasks(d: pd.DataFrame) -> list[dict]:
    """The seven Phase-15 dev cells; val_2022 kept once (P1 first occurrence)."""
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


def ref_predict(spec: dict, X: pd.DataFrame, feats: list[str],
                tr_idx: np.ndarray, ev_idx: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    if spec.get("kind") == "const":
        return np.full(len(ev_idx), float(ytr.mean()))
    pipe = spec["pipe"]
    pipe.fit(X.iloc[tr_idx][feats], ytr)
    return pipe.predict_proba(X.iloc[ev_idx][feats])[:, 1]


def winner_cfg(fam: str, search: pd.DataFrame) -> dict:
    r = search[search.family == fam].sort_values("mean_pr_auc", ascending=False).iloc[0]
    return {"cfg": json.loads(r["params"]), "config_id": str(r["config_id"]),
            "mean_pr_auc": float(r["mean_pr_auc"])}


def make_figure(groups: pd.DataFrame, refs: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = ["G1_weather", "G2_weather_rainhist", "G3_plus_geography",
            "G4_plus_hydrology", "G5_full"]
    labels = ["weather", "+rain\nhistory", "+geography", "+hydrology", "+network\n(FULL)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col in zip(axes, ("pr_auc", "macro_pr_auc")):
        for h, mk in zip(HORD.values(), ("o-", "s-", "^-")):
            ys = [float(groups[(groups.learner == "xgboost") & (groups.group == g)
                               & (groups.horizon == h) & (groups.season == "all")
                               & (groups.cell.isin(TEMPORAL_CELLS))][col].mean()) for g in fams]
            ax.plot(range(len(fams)), ys, mk, label=f"xgboost {h}")
        refv = [float(refs[(refs.model == "persistence") & (refs.horizon == "24h")
                           & (refs.season == "all") & (refs.cell.isin(TEMPORAL_CELLS))][col].mean()),
                float(refs[(refs.model == "climatology") & (refs.horizon == "24h")
                           & (refs.season == "all") & (refs.cell.isin(TEMPORAL_CELLS))][col].mean())]
        ax.axhline(refv[0], ls="--", c="gray", lw=1, label="D7 persistence (24h)")
        ax.axhline(refv[1], ls=":", c="black", lw=1, label="climatology (24h)")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("PR-AUC" if col == "pr_auc" else "macro PR-AUC")
        ax.set_title("micro (pooled)" if col == "pr_auc" else "macro (per-station mean)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 32 - nested feature-group ablation (temporal cells mean)")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase32_ablation.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    print(f"Phase 32 - ablation & comparative experiment matrix (stages 55, 56) | "
          f"seed={SEED} | {pd.Timestamp.now('UTC')}")
    print(f"hardware: CPU {platform.machine()} {platform.system()} | "
          f"{platform.python_version()}")
    sha = sha256(DATA)
    print(f"experimental_dataset.csv SHA-256 {sha}")
    assert sha == DATA_SHA, "experimental table changed - stop"

    d, fsets = B.load()
    d = d.reset_index(drop=True)
    full = set(fsets["FULL"])

    # the five LAYERS must tile FULL exactly; the cumulative groups are their prefix sums
    layers = WEATHER + RAIN_HIST + GEO + HYDRO + NETWORK
    assert len(layers) == len(set(layers)) == len(full) == 62, "layer partition broken"
    assert set(layers) == full, "layers do not tile FULL"
    assert set(GROUP_DEF[5][1]) == full - set(UP), "G5_no_up != FULL - up_*"
    print("feature partition OK: " + " | ".join(f"{n}={len(c)}" for n, c in GROUP_DEF))

    test_mask = d.split.to_numpy() == "test"
    season = d.season.to_numpy()

    tasks = dev_tasks(d)
    l1 = B.build_models(fsets)
    import phase20_classical_ml as P20  # noqa: E402  (lazy: reconfigure unsafe in Jupyter)
    search = pd.read_csv(REPORTS / "phase20_search.csv")
    winners = {fam: winner_cfg(fam, search) for fam in LEARNERS}
    for fam, w in winners.items():
        print(f"  winner {fam:10s} {w['config_id']:16s} mean_pr_auc={w['mean_pr_auc']:.5f}")
    ref_feats = {r: list(l1[r]["features"]) for r in REFS}

    group_rows: list[dict] = []
    ref_rows: list[dict] = []
    inc_rows: list[dict] = []
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
                ytr = y[tr_idx].astype(int)
                yv = y[ev_idx].astype(int)
                stv = d.station_id.to_numpy()[ev_idx]
                sev = season[ev_idx]
                ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

                # --- references (E0)
                for ref in REFS:
                    feats = ref_feats[ref]
                    p = ref_predict(l1[ref], d, feats, tr_idx, ev_idx, ytr)
                    for s in SEASONS:
                        sm = np.ones(len(yv), bool) if s == "all" else (sev == s)
                        ys, ps = yv[sm], p[sm]
                        if len(ys) == 0:
                            continue
                        pr, roc = safe_auc(ys, ps)
                        lo, hi = (B.bootstrap_pr_auc(stv[sm], ys, ps, BOOT_REPS)
                                  if s == "all" else (np.nan, np.nan))
                        ref_rows.append({
                            "model": ref, "cell": cell, "protocol": protocol, "season": s,
                            "horizon": HORD[hor], "n": int(sm.sum()),
                            "n_stations": int(pd.unique(stv[sm]).size),
                            "base_rate": float(ys.mean()), "pr_auc": pr, "roc_auc": roc,
                            "macro_pr_auc": B.macro_pr_auc(stv[sm], ys, ps),
                            "pr_auc_ci_lo": lo, "pr_auc_ci_hi": hi,
                            "brier": float(np.mean((ps - ys) ** 2)),
                            "ece": B.ece(ys, ps)})

                # --- nested groups (E1-E5, E11)
                preds: dict[tuple[str, str], np.ndarray] = {}
                for gname, feats in GROUP_DEF:
                    for fam in LEARNERS:
                        t1 = time.time()
                        pipe, _w, sw = P20.make_pipe(fam, winners[fam]["cfg"], feats, ratio)
                        P20.fit_pipe(pipe, d.iloc[tr_idx][feats], ytr, sw)
                        p = pipe.predict_proba(d.iloc[ev_idx][feats])[:, 1]
                        preds[(gname, fam)] = p
                        fit_s = time.time() - t1
                        for s in SEASONS:
                            sm = np.ones(len(yv), bool) if s == "all" else (sev == s)
                            ys, ps = yv[sm], p[sm]
                            if len(ys) == 0:
                                continue
                            pr, roc = safe_auc(ys, ps)
                            lo, hi = (B.bootstrap_pr_auc(stv[sm], ys, ps, BOOT_REPS)
                                      if s == "all" else (np.nan, np.nan))
                            group_rows.append({
                                "learner": fam, "group": gname, "n_features": len(feats),
                                "cell": cell, "protocol": protocol, "season": s,
                                "horizon": HORD[hor], "n": int(sm.sum()),
                                "n_stations": int(pd.unique(stv[sm]).size),
                                "base_rate": float(ys.mean()), "pr_auc": pr, "roc_auc": roc,
                                "macro_pr_auc": B.macro_pr_auc(stv[sm], ys, ps),
                                "pr_auc_ci_lo": lo, "pr_auc_ci_hi": hi,
                                "brier": float(np.mean((ps - ys) ** 2)),
                                "ece": B.ece(ys, ps), "fit_s": round(fit_s, 2)})

                # --- paired increments G1 -> ... -> G5 -> G5_no_up
                for fam in LEARNERS:
                    for a, b in _INCREMENTS:
                        pa, pb = preds[(a, fam)], preds[(b, fam)]
                        pr_a, _ = safe_auc(yv, pa)
                        pr_b, _ = safe_auc(yv, pb)
                        lo, hi = paired_boot(stv, yv, pa, pb, BOOT_REPS)
                        inc_rows.append({
                            "learner": fam, "group_a": a, "group_b": b,
                            "cell": cell, "protocol": protocol, "horizon": HORD[hor],
                            "n": len(yv), "pr_auc_a": pr_a, "pr_auc_b": pr_b,
                            "diff": pr_a - pr_b, "ci_lo": lo, "ci_hi": hi,
                            "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
                print(f"  {cell:14s} {HORD[hor]:4s} done ({time.time() - t0:6.0f}s)")

    groups = pd.DataFrame(group_rows)
    refs = pd.DataFrame(ref_rows)
    incs = pd.DataFrame(inc_rows)

    # ---------------------------------------------------------------- aggregation
    def macro(df: pd.DataFrame, cells: list[str], learner: str, group: str, hor: str,
              col: str = "pr_auc", season_: str = "all") -> float:
        sub = df[(df.learner == learner) & (df.group == group) & (df.horizon == hor)
                 & (df.season == season_) & (df.cell.isin(cells))]
        return float(sub[col].mean()) if len(sub) else float("nan")

    def ref_macro(cells: list[str], model: str, hor: str,
                  col: str = "pr_auc", season_: str = "all") -> float:
        sub = refs[(refs.model == model) & (refs.horizon == hor)
                   & (refs.season == season_) & (refs.cell.isin(cells))]
        return float(sub[col].mean()) if len(sub) else float("nan")

    # ---------------------------------------------------------------- E0-E16 matrix
    def g_row(exp: str, features: str, validation: str, status: str, value: float,
              unit: str, source: str, cis: tuple = (np.nan, np.nan),
              detail: str = "") -> dict:
        return {"experiment": exp, "features": features, "validation": validation,
                "status": status, "metric": unit, "value": value,
                "ci_lo": cis[0], "ci_hi": cis[1], "source": source, "detail": detail}

    matrix: list[dict] = []
    for h in HORD.values():
        matrix.append(g_row(
            "E0", "D7 persistence bar (wl_pct_danger)", f"temporal, {h}", "new",
            ref_macro(TEMPORAL_CELLS, "persistence", h), "PR-AUC (micro)",
            "phase32_reference.csv",
            (ref_macro(TEMPORAL_CELLS, "persistence", h, "pr_auc_ci_lo"),
             ref_macro(TEMPORAL_CELLS, "persistence", h, "pr_auc_ci_hi")),
            f"climatology={ref_macro(TEMPORAL_CELLS, 'climatology', h):.4f}; "
            f"persistence_on_cells=" + " / ".join(
                f"{c}={float(refs[(refs.model=='persistence')&(refs.horizon==h)&(refs.season=='all')&(refs.cell==c)]['pr_auc'].iloc[0]):.4f}"
                for c in TEMPORAL_CELLS)))

    names = ["E1", "E2", "E3", "E4", "E5"]
    for e, (gname, feats) in zip(names, GROUP_DEF[:5]):
        matrix.append(g_row(
            e, f"{gname} ({len(feats)} features)", "temporal (mean 2020/21/22)", "new",
            macro(groups, TEMPORAL_CELLS, "xgboost", gname, "24h"), "PR-AUC (micro)",
            "phase32_groups.csv",
            detail=("xgboost per-horizon " + " / ".join(
                f"{h}={macro(groups, TEMPORAL_CELLS, 'xgboost', gname, h):.4f}" for h in HORD.values())
                + f" | lightgbm 24h={macro(groups, TEMPORAL_CELLS, 'lightgbm', gname, '24h'):.4f}"
                + f" | macro-per-station 24h={macro(groups, TEMPORAL_CELLS, 'xgboost', gname, '24h', 'macro_pr_auc'):.4f}")))

    matrix.append(g_row("E6", "full (FULL 62)", "spatial (mean 4 folds)", "new",
                        macro(groups, SPATIAL_CELLS, "xgboost", "G5_full", "24h"),
                        "PR-AUC (micro)", "phase32_groups.csv",
                        detail="xgboost per-horizon " + " / ".join(
                            f"{h}={macro(groups, SPATIAL_CELLS, 'xgboost', 'G5_full', h):.4f}"
                            for h in HORD.values())))
    matrix.append(g_row("E7", "full (FULL 62)", "temporal, monsoon", "new",
                        macro(groups, TEMPORAL_CELLS, "xgboost", "G5_full", "24h", season_="monsoon"),
                        "PR-AUC (micro)", "phase32_groups.csv",
                        detail="season slices 24h " + " / ".join(
                            f"{s}={macro(groups, TEMPORAL_CELLS, 'xgboost', 'G5_full', '24h', season_=s):.4f}"
                            for s in ("monsoon", "non-monsoon"))))
    for h, e in zip(HORD.values(), ("E8", "E9", "E10")):
        matrix.append(g_row(e, "full (FULL 62)", f"temporal, {h}", "new",
                            macro(groups, TEMPORAL_CELLS, "xgboost", "G5_full", h),
                            "PR-AUC (micro)", "phase32_groups.csv",
                            (macro(groups, TEMPORAL_CELLS, "xgboost", "G5_full", h, "pr_auc_ci_lo"),
                             macro(groups, TEMPORAL_CELLS, "xgboost", "G5_full", h, "pr_auc_ci_hi")),
                            f"lightgbm={macro(groups, TEMPORAL_CELLS, 'lightgbm', 'G5_full', h):.4f}"))

    d11 = incs[(incs.group_a == "G5_no_up") & (incs.group_b == "G5_full")
               & (incs.learner == "xgboost") & (incs.cell.isin(TEMPORAL_CELLS))]
    matrix.append(g_row("E11", "with vs without up_* (F14-2)", "temporal (mean)", "new",
                        float(d11["diff"].mean()), "ΔPR-AUC (no_up - full)",
                        "phase32_increments.csv",
                        (float(d11["ci_lo"].min()), float(d11["ci_hi"].max())),
                        "negative => up_* unhelpful; " + "; ".join(
                            f"{r.cell}/{r.horizon} {r['diff']:+.4f}[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
                            for _, r in d11.sort_values(["cell", "horizon"]).iterrows())))
    matrix.append(g_row("E12", "with vs without soil/landcover groups", "temporal",
                        "blocked", np.nan, "-", "R8 (Phase-13 register)",
                        detail="no soil/landcover groups exist in the processed table; "
                               "nothing to ablate - blocker recorded, not silently dropped."))

    # ------------------------------------------------------------- reused E13-E16
    p27 = pd.read_csv(REPORTS / "phase27_metrics.csv")
    d27 = p27[p27.season == "all"]

    def m27(model: str, hor: str, cells: list[str], col: str = "pr_auc") -> float:
        sub = d27[(d27.model == model) & (d27.horizon == f"label_{hor}")
                  & (d27.cell.isin(cells))]
        return float(sub[col].mean()) if len(sub) else float("nan")

    reuse_rows: list[dict] = []

    def add_reuse(exp: str, model: str, cells_label: str, cells: list[str], source: str) -> None:
        for h in HORD.values():
            reuse_rows.append({
                "experiment": exp, "model": model, "cells": cells_label, "horizon": h,
                "pr_auc": m27(model, h, cells), "macro_pr_auc": m27(model, h, cells, "macro_pr_auc"),
                "ci_lo": m27(model, h, cells, "pr_auc_ci_lo"),
                "ci_hi": m27(model, h, cells, "pr_auc_ci_hi"), "source": source})

    inter = json.loads((REPORTS / "phase26_interaction.json").read_text(encoding="utf-8"))
    e13_lines = [f"{proto}: best engineered {inter['interaction'][proto]['best_engineered']} "
                 f"Δ{inter['interaction'][proto]['best_engineered_drop']:+.4f} vs best sequence "
                 f"{inter['interaction'][proto]['best_sequence']} "
                 f"Δ{inter['interaction'][proto]['best_sequence_drop']:+.4f}/step"
                 for proto in ("P1", "P2", "P3")]
    for m in ("xgboost", "fusion_L3"):
        add_reuse("E13", m, "temporal", TEMPORAL_CELLS, "reports/phase27_metrics.csv")
        add_reuse("E13", m, "spatial", SPATIAL_CELLS, "reports/phase27_metrics.csv")
    matrix.append(g_row("E13", "raw sequence -> DL vs engineered -> XGBoost", "all", "reused",
                        m27("xgboost", "24h", TEMPORAL_CELLS) - m27("fusion_L3", "24h", TEMPORAL_CELLS),
                        "ΔPR-AUC (xgboost - fusion_L3), 24h temporal",
                        "reports/phase27_metrics.csv + phase26_interaction.json",
                        detail=" | ".join(e13_lines)))

    for L in (3, 7, 14, 30):
        model = f"lstm_L{L}"
        add_reuse("E14", model, "temporal", TEMPORAL_CELLS, "reports/phase27_metrics.csv")
        matrix.append(g_row(f"E14_L{L}", f"sequence length L={L}", "temporal (mean)", "reused",
                            m27(model, "24h", TEMPORAL_CELLS), "PR-AUC (micro)",
                            "reports/phase27_metrics.csv",
                            detail="24/48/72h=" + " / ".join(
                                f"{h}={m27(model, h, TEMPORAL_CELLS):.4f}" for h in HORD.values())))

    a23 = pd.read_csv(REPORTS / "phase23_ablation.csv")

    def a23m(model: str, hor: str, cells: list[str] | None = None,
             col: str = "pr_auc") -> float:
        sub = a23[(a23.model == model) & (a23.horizon == f"label_{hor}") & (a23.season == "all")]
        if cells is not None:
            sub = sub[sub.cell.isin(cells)]
        return float(sub[col].mean()) if len(sub) else float("nan")

    a24 = pd.read_csv(REPORTS / "phase24_ablation.csv")
    gnn = float(pd.concat([a24[a24.family == "gcn"].pr_auc_gnn,
                           a24[a24.family == "gat"].pr_auc_gnn]).mean())
    lstm24 = float(a24[a24.family == "gcn"].pr_auc_lstm_L3_01.mean())
    # (matched-capacity phase23 rows have no phase27 twin; their numbers are quoted
    #  in the E15 detail string below, sourced directly from phase23_ablation.csv)
    for m in ("static_only", "fusion_L3", "gcn", "gat", "gnn_lstm"):
        if m in ("gcn", "gat", "gnn_lstm"):
            add_reuse("E15/E16", m, "spatial", SPATIAL_CELLS, "reports/phase27_metrics.csv")
        else:
            add_reuse("E15/E16", m, "temporal", TEMPORAL_CELLS, "reports/phase27_metrics.csv")
    matrix.append(g_row(
        "E15", "temporal only / +location / +river-basin / +graph edges", "temporal + spatial",
        "reused", m27("fusion_L3", "24h", TEMPORAL_CELLS) - m27("lstm_L3", "24h", TEMPORAL_CELLS),
        "ΔPR-AUC (fusion_L3 - lstm_L3), 24h temporal",
        "phase27_metrics.csv; phase23_ablation.csv; phase24_ablation.csv",
        detail=(f"temporal-only lstm_L3 {m27('lstm_L3','24h',TEMPORAL_CELLS):.4f} -> "
                f"+location fusion_L3 {m27('fusion_L3','24h',TEMPORAL_CELLS):.4f} "
                f"(matched-capacity P1 Δ24h={a23m('fusion_matched','24h') - a23m('temporal_only_matched','24h'):+.4f}); "
                f"+graph edges (river-first graph) vs temporal-only on P3 folds: "
                f"GNN {gnn:.4f} vs LSTM {lstm24:.4f} Δ{gnn - lstm24:+.4f}; "
                f"the '+river-basin' variant is the river-adjacency graph itself (52 river edges ∪ kNN fallback)")))
    matrix.append(g_row(
        "E16", "LSTM / +static / +attention; TCN / +static; GNN / +temporal", "temporal + spatial",
        "reused", m27("transformer_bestL", "24h", TEMPORAL_CELLS) - m27("lstm_L3", "24h", TEMPORAL_CELLS),
        "ΔPR-AUC (transformer - lstm_L3), 24h temporal",
        "phase27_metrics.csv; phase25_ablation.csv",
        detail=("LSTM(lstm_L3) %.4f | +static(fusion_L3) %.4f | +attention(transformer) %.4f | "
                "TCN(tcn_bestL) %.4f | GRU(gru_bestL) %.4f | TCN+static = not-run (no gated budget) | "
                "GNN temporal(gcn) %.4f" % (
                    m27('lstm_L3','24h',TEMPORAL_CELLS), m27('fusion_L3','24h',TEMPORAL_CELLS),
                    m27('transformer_bestL','24h',TEMPORAL_CELLS), m27('tcn_bestL','24h',TEMPORAL_CELLS),
                    m27('gru_bestL','24h',TEMPORAL_CELLS), m27('gcn','24h',TEMPORAL_CELLS)))))

    matrix_df = pd.DataFrame(matrix)
    reuse_df = pd.DataFrame(reuse_rows)

    # ------------------------------------------------------------------- write out
    groups.to_csv(REPORTS / "phase32_groups.csv", index=False)
    refs.to_csv(REPORTS / "phase32_reference.csv", index=False)
    incs.to_csv(REPORTS / "phase32_increments.csv", index=False)
    matrix_df.to_csv(REPORTS / "phase32_matrix.csv", index=False)
    reuse_df.to_csv(REPORTS / "phase32_reuse_detail.csv", index=False)
    make_figure(groups, refs)

    # increments summary (xgboost temporal, per contrast)
    inc_summary = {}
    for a, b in _INCREMENTS:
        sub = incs[(incs.group_a == a) & (incs.group_b == b) & (incs.learner == "xgboost")
                   & (incs.cell.isin(TEMPORAL_CELLS))]
        sub = sub.sort_values(["cell", "horizon"])
        inc_summary[f"{a}_minus_{b}"] = {
            "mean_diff": float(sub["diff"].mean()),
            "diffs": [round(float(v), 5) for v in sub["diff"]],
            "n_significant": int(sub["significant"].sum()), "n_cells": int(len(sub)),
            "ci_lo_min": float(sub["ci_lo"].min()), "ci_hi_max": float(sub["ci_hi"].max())}

    summary = {
        "phase": 32, "seed": SEED, "sha256": sha,
        "test_rows_read": test_rows_read,
        "cells": CELLS, "horizons": list(HORD.values()), "seasons": list(SEASONS),
        "learners": LEARNERS,
        "groups": {n: len(c) for n, c in GROUP_DEF},
        "n_group_rows": int(len(groups)), "n_reference_rows": int(len(refs)),
        "n_increment_rows": int(len(incs)), "n_matrix_rows": int(len(matrix_df)),
        "temporal_macro_pr_auc_xgboost": {
            g: {h: round(macro(groups, TEMPORAL_CELLS, "xgboost", g, h), 5) for h in HORD.values()}
            for g, _ in GROUP_DEF},
        "spatial_macro_pr_auc_xgboost": {
            g: {h: round(macro(groups, SPATIAL_CELLS, "xgboost", g, h), 5) for h in HORD.values()}
            for g, _ in GROUP_DEF},
        "increments": inc_summary,
        "wall_s": None,
    }
    summary["wall_s"] = round(time.time() - t0, 1)
    (REPORTS / "phase32_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n-- temporal macro PR-AUC (xgboost, mean of val_2020/21/22) --")
    for g, _ in GROUP_DEF:
        print(f"   {g:22s} " + " ".join(
            f"{h}={macro(groups, TEMPORAL_CELLS, 'xgboost', g, h):.4f}" for h in HORD.values()))
    print("   persistence           " + " ".join(
        f"{h}={ref_macro(TEMPORAL_CELLS, 'persistence', h):.4f}" for h in HORD.values()))
    print("\n-- xgboost temporal increments (a - b) --")
    for a, b in _INCREMENTS:
        print(f"   {a:22s} - {b:22s} mean {inc_summary[f'{a}_minus_{b}']['mean_diff']:+.4f} "
              f"sig {inc_summary[f'{a}_minus_{b}']['n_significant']}/{inc_summary[f'{a}_minus_{b}']['n_cells']}")
    print(f"\nrows: groups={len(groups)} refs={len(refs)} increments={len(incs)} "
          f"matrix={len(matrix_df)} reuse={len(reuse_df)}")
    print(f"TOTAL {time.time() - t0:.0f}s - exit=0")


if __name__ == "__main__":
    main()
