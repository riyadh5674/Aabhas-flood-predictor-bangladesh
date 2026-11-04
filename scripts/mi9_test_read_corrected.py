r"""MI-9 — corrected frozen-test read (**test read #3**).

Why this read exists
--------------------
`reports/d17_comparability_audit.md` shows that the MI-4 test read
(`scripts/mi4_test_read.py`) is not comparable with MI-6: it applied a
*complete-case* filter on the 187-feature FULL_V4 matrix (retaining ~30 % of
test rows, with systematically higher base rates) and used a
`test_new_network` mask that was not the Phase-15 definition.  The D17 decision
("hybrid V4/D16 beats D16 by +0.05 .. +0.08") is therefore unsupported.

This script re-runs the comparison **under the MI-6 protocol exactly**:

* every row with a non-NaN label is scored; missing features go through the
  pipeline's median imputer (no complete-case filtering);
* the Phase-15 test cells (`test_2023 .. test_2026`, `test_trained`,
  `test_new_network`, `test_unseen`) with `first_year > 2023` defining the new
  network;
* `evaluate_all.score` — the single scoring path used by MI-6;
* thresholds from `phase29_thresholds.best_exact` fitted on **val_2022 only**
  (C_FN = 10) for the two learned models, and the Phase-29 frozen D7 thresholds
  for the persistence bar;
* seasons `all` / `monsoon` / `non-monsoon`;
* a **hard sanity check**: re-scoring the frozen D16 model must reproduce
  `reports/mi6_test_metrics.csv` to < 1e-9, otherwise the read aborts before
  writing anything.

Pre-registered decision rule (written before the read)
------------------------------------------------------
1. sanity check on D16 passes;
2. the hybrid beats D16 on the 7-cell horizon-mean PR-AUC at **every** horizon;
3. the hybrid beats D7 on >= 20 of 21 (cell x horizon) points;
4. the paired station bootstrap on `test_2024` does not exclude 0 against the
   hybrid.

All four -> D17 is reinstated.  Otherwise **D16 remains champion**.

Run
---
    python -u scripts/mi9_test_read_corrected.py --dry-run
    python -u scripts/mi9_test_read_corrected.py
    python -u scripts/mi9_test_read_corrected.py --reread
"""
from __future__ import annotations

import argparse
import getpass
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_model as B          # noqa: E402
import evaluate_all as E            # noqa: E402
import phase29_thresholds as P29    # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

HOR = B.HORIZONS
H2D = dict(zip(HOR, ["24h", "48h", "72h"]))
SEASONS = E.SEASONS
TEST_YEARS = B.TEST_YEARS
CFN = 10.0
D7 = "persistence"
D16 = "hpo_perhorizon_FULL_V2"
HYB = "hybrid_d17_v4_24h"

LOG = REPORTS / "mi9_test_read_log.json"
METRICS = REPORTS / "mi9_test_metrics.csv"
THRESH = REPORTS / "mi9_test_thresholds.csv"
AUDIT = REPORTS / "mi9_sanity.json"
SUMMARY = REPORTS / "mi9_summary.json"


# --------------------------------------------------------------------------- data
def load_all() -> pd.DataFrame:
    """Same load as MI-6 (B.load + mi_v2) plus the V4-only blocks.

    Extra columns are inert for the D16 and D7 models, which only read their own
    frozen feature lists, so this cannot change their predictions.
    """
    d, _ = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    wc = pd.read_csv(PROC / "esa_worldcover_features.csv")
    mi8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    d = d.merge(wc, on="station_id", how="left", validate="many_to_one")
    d = d.merge(mi8, on=["station_id", "date"], validate="one_to_one")
    print(f"loaded {len(d):,} rows x {d.shape[1]} cols")
    return d


def test_cells(data: pd.DataFrame) -> dict:
    """Identical to `mi6_test_read.test_cells` (Phase-15 definitions)."""
    split = data.split.to_numpy()
    hold = data.station_holdout.to_numpy()
    year = data.date.dt.year.to_numpy()
    first_year = data.first_year.to_numpy()
    is_test = split == "test"
    cells = {f"test_{y}": is_test & (year == y) for y in TEST_YEARS}
    cells.update({
        "test_trained": is_test & ~hold & (first_year <= B.HOLDOUT_CUTOFF),
        "test_new_network": is_test & ~hold & (first_year > B.HOLDOUT_CUTOFF),
        "test_unseen": is_test & hold,
    })
    return cells


# -------------------------------------------------------------------- thresholds
def d7_thresholds() -> dict:
    p29 = pd.read_csv(REPORTS / "phase29_chosen_thresholds.csv")
    thr = {}
    for hor in HOR:
        for s in SEASONS:
            q = p29[(p29.model == D7) & (p29.horizon == H2D[hor]) & (p29.period == s)]
            assert len(q) == 1, (hor, s, len(q))
            thr[(D7, hor, s)] = float(q.t_warn.iloc[0])
    return thr


def derive_thresholds(name: str, pipes: dict, feats_of, data: pd.DataFrame) -> dict:
    """Cost-optimal period thresholds on val_2022 only (best_exact, C_FN=10)."""
    m = (data.split == "val").to_numpy() & ~data.station_holdout.to_numpy()
    seas = data.season.to_numpy()
    thr = {}
    for hor in HOR:
        feats = feats_of(hor)
        y = data[hor].to_numpy()
        valid = m & ~pd.isna(y)
        p = pipes[hor].predict_proba(data.loc[valid, feats])[:, 1]
        yv, sv = y[valid].astype(int), seas[valid]
        for s in SEASONS:
            sm = np.ones(len(yv), bool) if s == "all" else (sv == s)
            thr[(name, hor, s)] = float(P29.best_exact(yv[sm], p[sm], CFN)[0])
    return thr


# ------------------------------------------------------------------------ scoring
def score_one(name, pipes, feats_of, data, cells, thresholds) -> tuple[list[dict], int]:
    split = data.split.to_numpy()
    hold = data.station_holdout.to_numpy()
    station = data.station_id.to_numpy()
    dates = data.date.to_numpy()
    season = data.season.to_numpy()
    rows, n_read = [], 0

    for hor in HOR:
        feats = feats_of(hor)
        pipe = pipes[hor]
        y = data[hor].to_numpy()
        for cname, mask in cells.items():
            evi = np.flatnonzero(mask & ~np.isnan(y))
            if len(evi) == 0 or len(np.unique(y[evi])) < 2:
                continue
            p = pipe.predict_proba(data.iloc[evi][feats])[:, 1]
            yv, st, dt, sev = y[evi].astype(int), station[evi], dates[evi], season[evi]
            n_read += len(evi)
            for s in SEASONS:
                sm = np.ones(len(evi), bool) if s == "all" else (sev == s)
                if sm.sum() == 0 or len(np.unique(yv[sm])) < 2:
                    continue
                t = thresholds[(name, hor, s)]
                met, op = E.score(yv[sm], p[sm], t, st[sm], dt[sm], hor,
                                  do_ci=(sm.sum() >= 1000))
                rows.append({**met, **op, "model": name, "horizon": hor,
                             "season": s, "cell": cname, "protocol": "P1",
                             "n": int(sm.sum()), "n_pos": int(yv[sm].sum()),
                             "base_rate": float(yv[sm].mean()),
                             "thr_warn": float(t), "source": "mi9_test_read"})
        print(f"  scored {name} {hor}", flush=True)
    return rows, n_read


def fit_d7(data: pd.DataFrame) -> dict:
    split = data.split.to_numpy()
    hold = data.station_holdout.to_numpy()
    tri_any = (split == "train") & ~hold
    feats = ["wl_pct_danger"]
    pipes = {}
    for hor in HOR:
        y = data[hor].to_numpy()
        tri = np.flatnonzero(tri_any & ~np.isnan(y))
        pipe = B.build_models({"FULL": [], "LINEAR": [], "SELECTED": []})["persistence"]["pipe"]
        pipe.fit(data.iloc[tri][feats], y[tri].astype(int))
        pipes[hor] = pipe
    return pipes


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if LOG.exists() and not args.reread:
        log = json.loads(LOG.read_text(encoding="utf-8"))
        if log.get("n_reads", 0) >= 1:
            print(f"[guard] corrected read already done (n_reads={log['n_reads']}); "
                  f"pass --reread to force")
            return 0

    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    data = load_all()
    cells = test_cells(data)
    print("test cells:", {k: int(v.sum()) for k, v in cells.items()})

    d16 = joblib.load(MODELS / "final_v2" / "selected_model_v2.pkl")
    hyb24 = joblib.load(MODELS / "final_v3" / "hybrid_24h.pkl")
    v4_feats = list(hyb24.named_steps["pre"].feature_names_in_)
    v2_feats = list(d16["features"])
    print(f"D16 frozen: {d16['model']} / {len(v2_feats)} features")
    print(f"V4 24h    : {type(hyb24.named_steps['clf']).__name__} / {len(v4_feats)} features")

    d16_pipes = {h: d16["pipelines"][h] for h in HOR}
    hyb_pipes = {
        "label_24h": hyb24,
        "label_48h": d16["pipelines"]["label_48h"],
        "label_72h": d16["pipelines"]["label_72h"],
    }
    feats_d16 = lambda hor: v2_feats
    feats_hyb = lambda hor: v4_feats if hor == "label_24h" else v2_feats

    print("\nderiving thresholds on val_2022 (cost-optimal, C_FN=10) ...")
    thresholds = d7_thresholds()
    thresholds.update(derive_thresholds(D16, d16_pipes, feats_d16, data))
    thresholds.update(derive_thresholds(HYB, hyb_pipes, feats_hyb, data))

    tdf = pd.DataFrame([{"model": k[0], "horizon": k[1], "season": k[2],
                         "threshold": round(v, 6)} for k, v in thresholds.items()])
    print(tdf.to_string(index=False))

    if args.dry_run:
        tdf.to_csv(REPORTS / "mi9_thresholds_dryrun.csv", index=False)
        print("\ndry run: thresholds computed, no test scoring")
        return 0

    # ---- 1. D16 sanity check against MI-6 ------------------------------------
    print("\n=== scoring D16 (sanity check against MI-6) ===")
    d16_rows, _ = score_one(D16, d16_pipes, feats_d16, data, cells, thresholds)
    d16_df = pd.DataFrame(d16_rows)

    mi6 = pd.read_csv(REPORTS / "mi6_test_metrics.csv")
    mi6 = mi6[mi6.model == D16]
    key = ["cell", "horizon", "season"]
    j = d16_df.merge(mi6[key + ["n", "n_pos", "pr_auc", "brier"]], on=key,
                     suffixes=("", "_mi6"))
    dpr = float(np.nanmax(np.abs(j.pr_auc - j.pr_auc_mi6)))
    dbr = float(np.nanmax(np.abs(j.brier - j.brier_mi6)))
    dn = int((j.n - j.n_mi6).abs().max())
    audit = {"rows_compared": int(len(j)),
             "max_abs_delta_pr_auc": dpr, "max_abs_delta_brier": dbr,
             "max_abs_delta_n": dn, "tolerance": 1e-9,
             "passed": bool(dpr < 1e-9 and dbr < 1e-9 and dn == 0)}
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"  max |dPR-AUC| = {dpr:.3e}   max |dBrier| = {dbr:.3e}   max |dn| = {dn}")
    print(f"  sanity: {'PASS' if audit['passed'] else 'FAIL'}  ({len(j)} shared rows)")
    if not audit["passed"]:
        print(f"\n[ABORT] D16 sanity check failed. See {AUDIT}")
        return 2

    # ---- 2. hybrid + D7 ------------------------------------------------------
    print("\n=== scoring hybrid D17 (V4 @ 24 h, D16 @ 48/72 h) ===")
    hyb_rows, _ = score_one(HYB, hyb_pipes, feats_hyb, data, cells, thresholds)

    print("\n=== scoring D7 persistence ===")
    d7_pipes = fit_d7(data)
    d7_rows, _ = score_one(D7, d7_pipes, lambda hor: ["wl_pct_danger"],
                           data, cells, thresholds)

    met = pd.concat([d16_df, pd.DataFrame(hyb_rows), pd.DataFrame(d7_rows)],
                    ignore_index=True)
    met.to_csv(METRICS, index=False)
    tdf.to_csv(THRESH, index=False)

    # ---- 3. pre-registered decision rule ------------------------------------
    allm = met[met.season == "all"]
    piv = allm.pivot_table(index=["horizon", "cell"], columns="model",
                           values="pr_auc")
    piv = piv[[D16, HYB, D7]]
    piv["delta_hyb_minus_d16"] = piv[HYB] - piv[D16]
    piv["delta_hyb_minus_d7"] = piv[HYB] - piv[D7]
    piv.to_csv(REPORTS / "mi9_headline.csv")

    rule = {}
    for hor in HOR:
        sub = piv.loc[hor]
        rule[f"R2_{hor}_hyb_ge_d16"] = bool((sub[HYB] >= sub[D16]).all())
        rule[f"R2_{hor}_mean_delta"] = float(sub["delta_hyb_minus_d16"].mean())
    rule["R2_all_horizons"] = bool(all(rule[f"R2_{h}_hyb_ge_d16"] for h in HOR))
    rule["R3_hyb_beats_d7_count"] = int((piv["delta_hyb_minus_d7"] > 0).sum())
    rule["R3_pass"] = bool(rule["R3_hyb_beats_d7_count"] >= 20)

    # paired station bootstrap on test_2024 (largest base rate)
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(B.SEED)
    boot_2024 = {}
    for hor in HOR:
        y = data[hor].to_numpy()
        mask = cells["test_2024"] & ~np.isnan(y)
        idx = np.flatnonzero(mask)
        fh = feats_hyb(hor)
        ph = hyb_pipes[hor].predict_proba(data.iloc[idx][fh])[:, 1]
        p16 = d16_pipes[hor].predict_proba(data.iloc[idx][v2_feats])[:, 1]
        yv = y[idx].astype(int)
        st = data.station_id.to_numpy()[idx]
        uniq = np.unique(st)
        by = {s: np.flatnonzero(st == s) for s in uniq}
        diffs = []
        for _ in range(400):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([by[s] for s in pick])
            if len(np.unique(yv[sel])) < 2:
                continue
            diffs.append(average_precision_score(yv[sel], ph[sel])
                         - average_precision_score(yv[sel], p16[sel]))
        diffs = np.asarray(diffs)
        boot_2024[hor] = {"mean": float(diffs.mean()),
                          "lo": float(np.percentile(diffs, 2.5)),
                          "hi": float(np.percentile(diffs, 97.5)),
                          "p_gt_0": float((diffs > 0).mean()), "n": int(len(diffs))}
    rule["R4_test_2024_bootstrap"] = boot_2024
    rule["R4_pass"] = bool(all(v["p_gt_0"] >= 0.975 for v in boot_2024.values()))

    rule["D17_reinstated"] = bool(rule["R2_all_horizons"] and rule["R3_pass"]
                                  and rule["R4_pass"])
    SUMMARY.write_text(json.dumps({"rules": rule, "sanity": audit}, indent=2,
                                  default=str), encoding="utf-8")

    log = {
        "n_reads": 1, "project_test_reads_total": 3,
        "reads": [{
            "utc": datetime.now(timezone.utc).isoformat(),
            "user": getpass.getuser(), "host": platform.node(),
            "phase": "MI-9", "kind": "corrected_read",
            "supersedes": "mi4_test_read (invalid - complete-case filter)",
            "models": [D16, HYB, D7],
            "horizons": HOR, "test_years": TEST_YEARS,
            "cells": list(cells), "seasons": list(SEASONS),
            "distinct_test_rows": int((data.split == "test").sum()),
            "test_rows_read": int(len(met) and met.n.sum()),
            "thresholds": "D7 = Phase-29 frozen; D16/hybrid = best_exact on val_2022, C_FN=10",
            "sanity_d16_vs_mi6": audit,
            "decision_rule": {k: v for k, v in rule.items()
                              if not isinstance(v, dict)},
            "bootstrap_2024": boot_2024,
            "wall_secs": round(time.time() - t0, 1),
        }],
    }
    LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    print("\n=== TEST PR-AUC, season=all ===")
    print(piv.round(4).to_string())
    print("\n=== pre-registered rule ===")
    for k, v in rule.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"\nD17 reinstated: {rule['D17_reinstated']}")
    print(f"\nwrote {METRICS}\n      {THRESH}\n      {REPORTS / 'mi9_headline.csv'}\n"
          f"      {AUDIT}\n      {SUMMARY}\n      {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
