r"""MI-8 search: extended-budget per-horizon HPO on the MI-8 feature block (dev cells only).

Why this exists
---------------
MI-3b (the search that produced D16) was **budget-trimmed**: `N_TRIALS = 24` split across
xgboost + lightgbm, plus 8 each of hgb/et, i.e. roughly 25 configs per horizon over a
648-cell space.  The registry permits 40 per family, and the winning families (lgbm, xgb)
are exactly the ones this run concentrates budget on.  The probe (`mi8_probe.py`) measures
the *feature* contribution under a fixed model; this script measures what a **larger search
on the improved feature set** adds on top.

Reuse, not reinvention
----------------------
The fitting/scoring contract is imported from `mi3_search` -- `build()` (identical
preprocessor + family construction), `score_cells()` (predict only on train+dev rows, score
the 7 dev cells), `dev_masks()`, and the four search spaces -- so the numbers are directly
comparable to MI-3b and D16.  Only the ledger, the feature source and the budget are new.
Nothing frozen is written: outputs are arm-suffixed (`mi8_hpo_<arm>.csv`,
`models/mi/mi8_hpo_<arm>_<horizon>.json`, separate resume ledgers).

Test policy
-----------
Only the seven development cells exist in `dev_masks()`; there is no code path that can
score `>= 2023`.  Selection is on `val_2022` (season=all), the pre-registered D16 rule.

Run
---
    .venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v3
    .venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v2      # equal-budget control
    .venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v3 --quick
Budget knobs (env, defaults shown): MI8_XGB=30 MI8_LGBM=30 MI8_HGB=10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                                    # noqa: E402
import mi3_search as S                                         # noqa: E402
from sklearn.model_selection import ParameterSampler           # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "mi"
MODELS.mkdir(parents=True, exist_ok=True)
HOR = B.HORIZONS
SEED = B.SEED

META_COLS = S.META_COLS
CFG_COLS = S.CFG_COLS

RESULTS: list[tuple[str, bool, str]] = []


def check(cid: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((cid, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {detail}")
    return bool(ok)


# ------------------------------------------------------------------- loading
def load_arm(arm: str) -> tuple[pd.DataFrame, list[str]]:
    """B.load() + the frozen MI-2 block, plus the MI-8 block when the arm needs it."""
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    if arm == "full_v3":
        m8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
        d = d.merge(m8, on=["station_id", "date"], validate="one_to_one")
        feats = json.loads((PROC / "feature_sets_mi8.json").read_text(encoding="utf-8"))["sets"]["FULL_V3"]
    elif arm == "full_v2":
        feats = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    else:
        raise ValueError(f"unknown arm {arm}")
    check(f"L_{arm}", len(d) == 200665, f"rows={len(d)} feats={len(feats)}")
    return d, feats


# ------------------------------------------------------------------- ledger
def _read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(p, engine="python")
    except Exception:
        return pd.DataFrame()


def append_row(p: Path, row: dict) -> None:
    df = pd.DataFrame([row]).reindex(columns=META_COLS + CFG_COLS)
    header = (not p.exists()) or p.stat().st_size == 0
    df.to_csv(p, mode="a", header=header, index=False)


def _native(v):
    """Coerce numpy scalars to plain Python for json.dumps.

    np.float64 subclasses float, so json.dumps accepts it silently; np.int64 does NOT
    subclass int and raises "Object of type int64 is not JSON serializable".  Whether a
    config column survives as int64 depends on whether *any* executed family left it NaN
    (which forces float64) -- mi3_search never tripped this only because hgb/et rows
    NaN-out n_estimators.  Caught by the MI-8 smoke test before the full budget ran.
    """
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def make_trials(n_xgb: int, n_lgbm: int, n_hgb: int) -> list[tuple]:
    base = ("xgb", dict(n_estimators=300, learning_rate=0.03, max_depth=6,
                        min_child_weight=10, subsample=1.0, colsample_bytree=0.8,
                        scale_pos_weight=1.0), "frozen")
    out = [base]
    out += [("xgb", c, f"xgb{i}") for i, c in enumerate(
        ParameterSampler(S.XGB_SPACE, n_iter=n_xgb, random_state=SEED))]
    out += [("lgbm", c, f"lgbm{i}") for i, c in enumerate(
        ParameterSampler(S.LGBM_SPACE, n_iter=n_lgbm, random_state=SEED))]
    if n_hgb:
        out += [("hgb", c, f"hgb{i}") for i, c in enumerate(
            ParameterSampler(S.HGB_SPACE, n_iter=n_hgb, random_state=SEED))]
    return out


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["full_v3", "full_v2"], default="full_v3")
    ap.add_argument("--quick", action="store_true", help="2 configs/horizon smoke test")
    ap.add_argument("horizons", nargs="*", help="subset of label_24h/48h/72h")
    a = ap.parse_args(argv)

    arm = a.arm
    partial = REPORTS / f"mi8_hpo_partial_{arm}.csv"
    out_csv = REPORTS / f"mi8_hpo_{arm}.csv"
    n_xgb, n_lgbm, n_hgb = (int(os.environ.get("MI8_XGB", 30)),
                            int(os.environ.get("MI8_LGBM", 30)),
                            int(os.environ.get("MI8_HGB", 10)))
    if a.quick:
        n_xgb, n_lgbm, n_hgb = 1, 1, 0

    print(f"=== MI-8 search  arm={arm}  budget xgb={n_xgb} lgbm={n_lgbm} hgb={n_hgb} "
          f"per horizon  seed={SEED} ===")
    print("selection cell: val_2022 (season=all), pre-registered D16 rule")
    print("test_rows_read = 0 (dev cells only; no >=2023 row is scored)")

    d, feats = load_arm(arm)
    masks = S.dev_masks(d)
    check("M1", set(masks) == {"val_2020", "val_2021", "val_2022",
                               "spatial_fold_1", "spatial_fold_2",
                               "spatial_fold_3", "spatial_fold_4"},
          f"masks={sorted(masks)}")
    check("M2", not any("test" in k for k in masks), "no test cell in the scoring grid")

    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()
    dev_union = np.zeros(len(d), dtype=bool)
    for m in masks.values():
        dev_union |= m
    keep_pos = np.where(train | dev_union)[0]
    check("M3", not (dev_union & (d.split.to_numpy() == "test")).any(),
          f"dev union {int(dev_union.sum()):,} rows excludes split=='test'")

    done = set()
    if partial.exists() and partial.stat().st_size > 0:
        prev = _read(partial)
        if {"horizon", "tag"}.issubset(prev.columns):
            done = {(r.horizon, r.tag) for r in prev.itertuples()}
    print(f"resume ledger: {len(done)} configs already done")

    want = [h for h in a.horizons if h in HOR] or list(HOR)
    trials = make_trials(n_xgb, n_lgbm, n_hgb)
    t_start = time.time()

    for h in want:
        y_all = d[h].to_numpy()
        tr = train & ~pd.isna(y_all)
        Xtr, ytr = d.loc[tr, feats], y_all[tr]
        todo = [t for t in trials if (h, t[2]) not in done]
        print(f"\n=== {h}: {len(trials)} configs ({len(todo)} to run) ===", flush=True)
        for fam, cfg, tag in todo:
            t1 = time.time()
            pipe = S.build(fam, cfg, feats)
            pipe.fit(Xtr, ytr)
            dev = S.score_cells(pipe, d, masks, y_all, tr, feats, keep_pos)
            mean_pr = float(np.mean(list(dev.values())))
            v22 = float(dev.get("val_2022", float("nan")))
            append_row(partial, {"horizon": h, "family": fam, "tag": tag,
                                 "mean_pr_auc": mean_pr, "val_2022": v22,
                                 "secs": round(time.time() - t1, 1), **cfg})
            print(f"  {tag:9s} {fam:4s} mean={mean_pr:.5f} val2022={v22:.5f} "
                  f"{time.time() - t1:.0f}s", flush=True)
            del pipe

    # ---- write arm outputs -------------------------------------------------
    hpo = _read(partial).drop_duplicates(subset=["horizon", "tag"], keep="last")
    hpo = hpo.sort_values(["horizon", "val_2022"], ascending=[True, False])
    hpo.to_csv(out_csv, index=False)
    winners = {}
    for h, g in hpo.groupby("horizon"):
        b = g.sort_values("val_2022", ascending=False).iloc[0]
        cfg = {k: _native(v) for k, v in b.items()
               if k not in ("horizon", "family", "tag", "mean_pr_auc", "val_2022", "secs")
               and not (isinstance(v, float) and np.isnan(v))}
        rec = {"family": b.family, "tag": b.tag, "val_2022": float(b.val_2022),
               "mean_pr_auc": float(b.mean_pr_auc), "arm": arm, "cfg": cfg}
        (MODELS / f"mi8_hpo_{arm}_{h}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
        winners[h] = rec
    hm = float(np.mean([w["val_2022"] for w in winners.values()])) if winners else float("nan")

    print("\n=== winners (selection cell val_2022) ===")
    for h in sorted(winners):
        w = winners[h]
        print(f"  {h}: {w['family']:5s} {w['tag']:8s} val_2022={w['val_2022']:.5f} "
              f"mean7={w['mean_pr_auc']:.5f}")
    print(f"  horizon-mean = {hm:.5f}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    print(f"WROTE {out_csv}  ({len(hpo)} rows)  wall={time.time() - t_start:.0f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
