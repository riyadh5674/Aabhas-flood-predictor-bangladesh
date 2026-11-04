r"""MI-6 — the single, logged frozen-test read for the MI-5/D16 model.

Contract (mirrors `scripts/phase33_selection.py` part C)
-------------------------------------------------------
* The frozen test period (`split == "test"`, `>= 2023`) is read **exactly once**, for the
  MI-5 selected model (`models/final_v2/selected_model_v2.pkl`, decision D16) and the
  **D7 persistence bar** only.
* It is guarded: if `reports/mi6_test_metrics.csv` + `reports/mi6_test_read_log.json`
  already exist, the read is reused and not repeated (pass ``--reread`` to force).
* Thresholds: D7 is carried from the Phase-29 cost-optimal table
  (`phase29_chosen_thresholds.csv`, period-specific); the D16 model has no Phase-29 entry,
  so its period-specific cost-optimal thresholds are derived on **val_2022 only** with the
  exact same method (`phase29_thresholds.best_exact`, C_FN=10, C_FP=1) and frozen to
  `reports/mi6_thresholds.csv`.  No test label is used to pick a threshold.
* A cross-check reproduces the D7 test PR-AUC / Brier against the historical Phase-16 read
  (`phase16_test_metrics.csv`): they must match to < 1e-6, proving the read is trustworthy.

Outputs: reports/mi6_test_metrics.csv · reports/mi6_test_read_log.json ·
         reports/mi6_thresholds.csv · reports/mi6_summary.json
"""
from __future__ import annotations

import getpass
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_model as B          # noqa: E402
import evaluate_all as E            # noqa: E402  (THE one scoring path)
import phase29_thresholds as P29    # noqa: E402  (best_exact, the Phase-29 threshold method)

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

HOR = B.HORIZONS
HOR_DISP = ["24h", "48h", "72h"]
H2D = dict(zip(HOR, HOR_DISP))
SEASONS = E.SEASONS
TEST_YEARS = B.TEST_YEARS
SEED = B.SEED
D7 = "persistence"
TBL = PROC / "experimental_dataset.csv"
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
FROZEN = MODELS / "final_v2" / "selected_model_v2.pkl"
CFN = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_data() -> pd.DataFrame:
    d, _ = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    return d


def test_cells(data: pd.DataFrame) -> dict:
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


def derive_thresholds(frozen: dict, data: pd.DataFrame) -> dict:
    """Cost-optimal (C_FN=10) period thresholds, val_2022 only — the Phase-29 method."""
    thr = {}

    # D7 bar: carry the Phase-29 frozen thresholds verbatim.
    p29 = pd.read_csv(REPORTS / "phase29_chosen_thresholds.csv")
    for hor in HOR:
        for seas in SEASONS:
            q = p29[(p29.model == D7) & (p29.horizon == H2D[hor]) & (p29.period == seas)]
            assert len(q) == 1, (hor, seas, len(q))
            thr[(D7, hor, seas)] = float(q.t_warn.iloc[0])

    # D16 model: no Phase-29 entry -> derive on val_2022 with the same exact sweep.
    m = (data.split == "val").to_numpy() & ~data.station_holdout.to_numpy()
    seas = data.season.to_numpy()
    for hor in HOR:
        y = data[hor].to_numpy()
        valid = m & ~pd.isna(y)                      # labels can be NaN on val rows
        p = frozen["pipelines"][hor].predict_proba(
            data.loc[valid, frozen["features"]])[:, 1]
        yv, sv = y[valid].astype(int), seas[valid]
        for s in SEASONS:
            sm = np.ones(len(yv), bool) if s == "all" else (sv == s)
            t, _ = P29.best_exact(yv[sm], p[sm], CFN)
            thr[(frozen["model"], hor, s)] = float(t)
    return thr


def read_test_once(frozen: dict, data: pd.DataFrame, thresholds: dict,
                   reread: bool) -> tuple[pd.DataFrame, dict, bool]:
    log_path = REPORTS / "mi6_test_read_log.json"
    metrics_path = REPORTS / "mi6_test_metrics.csv"
    if log_path.exists() and metrics_path.exists() and not reread:
        log = json.loads(log_path.read_text(encoding="utf-8"))
        print(f"[guard] test already read {log['n_reads']}x "
              f"(last {log['reads'][-1]['utc']}) — reusing cached metrics; "
              f"pass --reread to force")
        return pd.read_csv(metrics_path), log, False

    split = data.split.to_numpy()
    hold = data.station_holdout.to_numpy()
    station = data.station_id.to_numpy()
    dates = data.date.to_numpy()
    season = data.season.to_numpy()
    is_test = split == "test"
    cells = test_cells(data)

    import joblib
    sel_pipes = frozen["pipelines"]
    sel_feats = frozen["features"]
    d7_pipe = B.build_models({"FULL": [], "LINEAR": [], "SELECTED": []})["persistence"]["pipe"]
    tri_any = (split == "train") & ~hold

    rows, n_read = [], 0
    for model in [frozen["model"], D7]:
        for hor in HOR:
            y = data[hor].to_numpy()
            tri = np.flatnonzero(tri_any & ~np.isnan(y))
            if model == D7:
                feats = ["wl_pct_danger"]
                pipe = d7_pipe
                pipe.fit(data.iloc[tri][feats], y[tri].astype(int))
            else:
                feats = sel_feats
                pipe = sel_pipes[hor]
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
                    t = thresholds[(model, hor, s)]
                    met, op = E.score(yv[sm], p[sm], t, st[sm], dt[sm], hor,
                                      do_ci=(sm.sum() >= 1000))
                    rows.append({**met, **op, "model": model, "horizon": hor,
                                 "season": s, "cell": cname, "protocol": "P1",
                                 "n": int(sm.sum()), "n_pos": int(yv[sm].sum()),
                                 "base_rate": float(yv[sm].mean()),
                                 "thr_warn": float(t), "source": "mi6_test_read"})
    tm = pd.DataFrame(rows)
    tm.to_csv(metrics_path, index=False)

    # cross-check the D7 bar against the historical Phase-16 read
    p16 = pd.read_csv(REPORTS / "phase16_test_metrics.csv")
    p16 = p16[p16.model == D7]
    j = tm[tm.model == D7].merge(p16[["cell", "horizon", "season", "pr_auc", "brier"]],
                                 on=["cell", "horizon", "season"], suffixes=("", "_p16"))
    dpr = float(np.nanmax(np.abs(j.pr_auc - j.pr_auc_p16)))
    dbr = float(np.nanmax(np.abs(j.brier - j.brier_p16)))
    print(f"[cross-check] D7 test PR-AUC/Brier vs Phase-16: max|d| = "
          f"{dpr:.2e}/{dbr:.2e} over {len(j)} shared rows")

    log = {"n_reads": 1, "reads": [{
        "utc": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser(), "host": platform.node(),
        "phase": "MI-6", "decision": "D16",
        "models": [frozen["model"], D7],
        "horizons": HOR, "test_years": TEST_YEARS,
        "cells": list(cells), "seasons": list(SEASONS),
        "test_rows_read": int(n_read),
        "test_rows_read_definition": "sum over (model x horizon x cell) of scored rows",
        "distinct_test_rows": int(is_test.sum()),
        "selected_model_rows": int(n_read // 2),
        "thresholds": "D7 = Phase-29 frozen; D16 = cost-optimal on val_2022 (best_exact, C_FN=10)",
        "dataset_sha256": TBL_SHA,
        "dataset_sha256_v2": frozen["dataset_sha256_v2"],
        "model_frozen_at_utc": frozen.get("frozen_at_utc"),
        "d7_phase16_crosscheck_max_abs_delta_pr_auc": dpr,
        "d7_phase16_crosscheck_max_abs_delta_brier": dbr,
        "reason": "the single frozen-test read mandated by the MI-v2 programme (MI-6)",
    }]}
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return tm, log, True


def main() -> int:
    t0 = time.time()
    dry = "--dry-run" in sys.argv
    reread = "--reread" in sys.argv
    print(f"MI-6 — single frozen-test read | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')} | user={getpass.getuser()}")

    sha_v1 = sha256(TBL)
    assert sha_v1 == TBL_SHA, f"experimental table changed: {sha_v1}"
    import joblib
    frozen = joblib.load(FROZEN)
    assert frozen["test_rows_read"] == 0, "frozen model has already seen test"
    sha_v2 = sha256(PROC / "mi_v2_features.csv")
    assert sha_v2 == frozen["dataset_sha256_v2"], "v2 table changed since freeze"
    print(f"SHA verified  v1={TBL_SHA[:12]}..  v2={sha_v2[:12]}..")
    print(f"D16 = {frozen['model']}  ({frozen['feature_set']}, {len(frozen['features'])} feats, "
          f"families {frozen['per_horizon_family']})")

    data = load_data()
    thresholds = derive_thresholds(frozen, data)
    pd.DataFrame(
        [{"model": m, "horizon": H2D[h], "season": s, "t_warn": round(t, 5)}
         for (m, h, s), t in thresholds.items()]
    ).to_csv(REPORTS / "mi6_thresholds.csv", index=False)
    print("\nperiod cost-optimal thresholds (all-season):")
    for m in [frozen["model"], D7]:
        print(f"  {m:24s} " + "  ".join(
            f"{H2D[h]} t={thresholds[(m, h, 'all')]:.3f}" for h in HOR))

    if dry:
        print("\n[dry-run] thresholds derived; frozen test NOT read.")
        return 0

    tm, log, fresh = read_test_once(frozen, data, thresholds, reread)
    last = log["reads"][-1]
    print(f"\ntest read: {log['n_reads']}x, {last['test_rows_read']:,} scored rows "
          f"({last['distinct_test_rows']:,} distinct test rows), models={last['models']}")

    sel = tm[(tm.model == frozen["model"]) & (tm.season == "all")]
    d7t = tm[(tm.model == D7) & (tm.season == "all")]
    headline = []
    print("\n=== frozen test, all-season (D16 vs D7) ===")
    for c in ["test_2023", "test_2024", "test_2025", "test_2026",
              "test_trained", "test_new_network", "test_unseen"]:
        line, rec = [], {}
        for h, d in zip(HOR, HOR_DISP):
            s = sel[(sel.cell == c) & (sel.horizon == h)]
            b = d7t[(d7t.cell == c) & (d7t.horizon == h)]
            if len(s) and len(b):
                line.append(f"{d} {s.pr_auc.iloc[0]:.4f}/{b.pr_auc.iloc[0]:.4f}")
                rec[d] = {"d16": round(float(s.pr_auc.iloc[0]), 5),
                          "d7": round(float(b.pr_auc.iloc[0]), 5)}
        if line:
            print(f"  {c:16s} " + "  ".join(line))
            headline.append({"cell": c, **rec})

    if not fresh:
        print("\n[guard] summary not rewritten on a guarded reuse "
              "(the fresh-read summary is preserved).")
        return 0

    summary = {
        "phase": "MI-6", "decision": "D16", "seed": SEED, "model": frozen["model"],
        "feature_set": frozen["feature_set"], "horizons": HOR,
        "test_rows_read": int(last["test_rows_read"]),
        "test_reads": int(log["n_reads"]),
        "distinct_test_rows": int(last["distinct_test_rows"]),
        "d7_phase16_crosscheck_max_abs_delta_pr_auc":
            last.get("d7_phase16_crosscheck_max_abs_delta_pr_auc"),
        "d7_phase16_crosscheck_max_abs_delta_brier":
            last.get("d7_phase16_crosscheck_max_abs_delta_brier"),
        "headline_all_season": headline,
        "wall_s": round(time.time() - t0, 1), "exit_code": 0,
    }
    (REPORTS / "mi6_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\nwrote reports/mi6_test_metrics.csv ({len(tm)}), mi6_test_read_log.json, "
          f"mi6_thresholds.csv, mi6_summary.json  [{time.time() - t0:.0f}s]")
    print(f"MI-6 COMPLETE — test read {log['n_reads']}x; models {last['models']}; exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
