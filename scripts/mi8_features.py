"""MI-8 (score push): the response-lag feature block, keyed on (station_id, date).

WHY (Phase-30 pattern A, the dominant actionable finding)
---------------------------------------------------------
0.0 % of the frozen leaders' 83 false negatives had the river at/above danger at issue
time -- median 0.76 m BELOW danger -- while 68.7 % were already rising (`wl_rise_3d > 0`)
and median 7-day rain was 102 mm vs 86 mm for true positives.  The model sees a *state*
(level, antecedent rain) but not the *shape of the approach*: slope, acceleration,
proximity persistence, and the rain-without-river-response decoupling that names that
pattern directly.  This block adds exactly those, plus upstream/neighbour *propagation*
(the wave coming from above), which Phase 30 listed as the fix candidate.

Output (NON-DESTRUCTIVE -- same convention as `build_mi_features.py --tag era5`)
---------------------------------------------------------------------------------
  data/processed/mi8_features.csv        200,665 rows x (m8_* columns + keys)
  data/processed/feature_sets_mi8.json   sets.FULL_V3 = FULL_V2 (110) + m8_*

Nothing frozen is touched: the v1/v2 tables, `feature_sets_v2.json`, MI-5/MI-6 artefacts
and the frozen test are left exactly as they were.

LEAKAGE RULE (Phase-13) -- ENFORCED, not asserted
-------------------------------------------------
Every column is a backward shift or a trailing window over rows already observed at issue
time; there are no forward shifts and no whole-sample statistics.  `--audit` proves it by
rebuilding the whole block on data truncated to `< 2022-01-01` and requiring every 2017-2021
value to be *identical* to the full build.  A forward shift, or any statistic computed over
the full sample, would change the pre-2022 values and fail the audit.

Conventions (verified against the shipped table before writing this)
--------------------------------------------------------------------
  * `wl_pct_danger == wl_max / danger_level` and `wl_dist_to_danger == danger - wl_max`
    (exact, max|diff| = 0.0), so proximity features are expressed on the wl_max ratio.
  * `wl_last_lag1..7` are ROW lags of the previous *observed* reading and
    `wl_rise_1d == wl_last - wl_last_lag1` (exact), so this block follows the row-lag
    convention for level features.  The frame carries 148,670 missing calendar dates, so
    calendar-based rolling would be a different (and wrongly spaced) statistic; the shipped
    `rain_*d` aggregates are already calendar-built upstream and are therefore only
    differenced/lagged here, never re-accumulated from `precip_mm`.

Run
---
    .venv\\Scripts\\python.exe -u scripts\\mi8_features.py          # build + full self-checks
    .venv\\Scripts\\python.exe -u scripts\\mi8_features.py --audit  # leakage proof (truncation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

OUT_CSV = PROC / "mi8_features.csv"
OUT_JSON = PROC / "feature_sets_mi8.json"
PREFIX = "m8_"
TRAIN_END = "2022-01-01"

RESULTS: list[tuple[str, bool, str]] = []


def check(cid: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((cid, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {detail}")
    return bool(ok)


# --------------------------------------------------------------------- frame
def load_frame() -> pd.DataFrame:
    """experimental_dataset (base) FULL_JOINed with the frozen MI-2 v2 block."""
    exp = pd.read_csv(PROC / "experimental_dataset.csv", parse_dates=["date"])
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
    check("F1", len(d) == len(exp) == len(v2) == 200665,
          f"rows base={len(exp)} v2={len(v2)} merged={len(d)}")
    d = d.sort_values(["station_id", "date"]).drop_duplicates(
        ["station_id", "date"], keep="first").reset_index(drop=True)
    check("F2", len(d) == 200665, f"after sort/dedupe {len(d)}")
    return d


def v2_sets() -> dict:
    return json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]


# ------------------------------------------------------------------- helpers
def _segments(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end positional offsets of each contiguous station block."""
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(codes)]
    return starts, ends


def prefix_sum(col: pd.Series, lam: float, codes: np.ndarray) -> pd.Series:
    """api_t = lam * api_{t-1} + precip_t, recursed row-wise within a station (causal)."""
    vals = col.to_numpy(dtype=float, copy=True)
    np.nan_to_num(vals, copy=False, nan=0.0)
    out = np.empty(len(vals), dtype=float)
    starts, ends = _segments(codes)
    for a, b in zip(starts, ends):
        acc = 0.0
        for i in range(a, b):
            acc = lam * acc + vals[i]
            out[i] = acc
    return pd.Series(out, index=col.index)


def build(d: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of the m8_* block, aligned to d.index."""
    codes = pd.factorize(d["station_id"], sort=True)[0]
    o = pd.DataFrame(index=d.index)

    def lag(col: str, k: int) -> pd.Series:
        return d.groupby("station_id")[col].shift(k)

    def roll(col: str, w: int, how: str = "mean", minp: int = 1) -> pd.Series:
        return d.groupby("station_id")[col].transform(
            lambda s: getattr(s.rolling(w, min_periods=minp), how)())

    def roll_on(s: pd.Series, w: int, how: str = "mean", minp: int = 1) -> pd.Series:
        return s.groupby(d["station_id"]).transform(
            lambda x: getattr(x.rolling(w, min_periods=minp), how)())

    wl = d["wl_last"]
    wmax = d["wl_max"]
    pct = d["wl_pct_danger"]              # == wl_max / danger_level
    dist = d["wl_dist_to_danger"]         # == danger_level - wl_max

    # ---------------------------------------------------- A. level trajectory
    o["m8_wl_slope_7"] = wl - lag("wl_last", 7)
    o["m8_wl_slope_14"] = wl - lag("wl_last", 14)
    o["m8_wl_slope_30"] = wl - lag("wl_last", 30)
    o["m8_wl_accel_3"] = (wl - lag("wl_last", 3)) - (lag("wl_last", 3) - lag("wl_last", 7))
    o["m8_wl_std_7"] = roll("wl_last", 7, "std", 3)
    o["m8_wl_std_14"] = roll("wl_last", 14, "std", 3)
    o["m8_wl_min_7"] = roll("wl_last", 7, "min")
    o["m8_wl_range_7"] = roll("wl_last", 7, "max") - roll("wl_last", 7, "min")
    o["m8_wl_range_30"] = roll("wl_last", 30, "max") - roll("wl_last", 30, "min")
    rising = wl.groupby(d["station_id"]).diff().gt(0).astype(float)
    o["m8_frac_rising_7"] = roll_on(rising, 7, "mean")
    o["m8_wlmax_slope_3"] = wmax - lag("wl_max", 3)
    o["m8_wlmax_min_7"] = roll("wl_max", 7, "min")
    o["m8_wlmax_range_7"] = roll("wl_max", 7, "max") - roll("wl_max", 7, "min")
    o["m8_wlmax_std_14"] = roll("wl_max", 14, "std", 3)

    # ---------------------------------- B. proximity / danger-approach (A & F)
    o["m8_pct_lag1"] = lag("wl_pct_danger", 1)
    o["m8_pct_lag3"] = lag("wl_pct_danger", 3)
    o["m8_pct_slope3"] = pct - lag("wl_pct_danger", 3)
    o["m8_pct_max7"] = roll("wl_pct_danger", 7, "max")
    o["m8_pct_min7"] = roll("wl_pct_danger", 7, "min")
    near90 = (pct >= 0.90).astype(float)
    near95 = (pct >= 0.95).astype(float)
    o["m8_days_near090_7"] = roll_on(near90, 7, "sum")
    o["m8_days_near090_14"] = roll_on(near90, 14, "sum")
    o["m8_days_near095_7"] = roll_on(near95, 7, "sum")
    o["m8_dist_min_7"] = roll("wl_dist_to_danger", 7, "min")   # closest approach
    o["m8_dist_lag1"] = lag("wl_dist_to_danger", 1)
    o["m8_dist_slope3"] = dist - lag("wl_dist_to_danger", 3)
    mu180 = roll("wl_pct_danger", 180, "mean", 20)
    sd180 = roll("wl_pct_danger", 180, "std", 20)
    o["m8_pct_z180"] = (pct - mu180) / (sd180 + 1e-6)   # station-relative severity

    # -------------------------------------------- C. antecedent rain (G2, A)
    o["m8_rain7_lag1"] = lag("rain_7d", 1)
    o["m8_rain7_lag3"] = lag("rain_7d", 3)
    o["m8_rain14_prev"] = d["rain_14d"] - d["rain_7d"]          # the prior week
    o["m8_rain30_prev"] = d["rain_30d"] - d["rain_14d"]         # the prior fortnight
    o["m8_rain7_frac30"] = d["rain_7d"] / (d["rain_30d"] + 1.0)  # recency concentration
    o["m8_rain_int7"] = d["rain_7d"] / (d["wet_days_7"] + 1.0)   # mm per wet day
    o["m8_rain3_lag3"] = lag("rain_3d", 3)
    o["m8_rain30_lag1"] = lag("rain_30d", 1)
    o["m8_api095"] = prefix_sum(d["precip_mm"], 0.95, codes)
    o["m8_api085"] = prefix_sum(d["precip_mm"], 0.85, codes)

    # -------------------------------- D. rain/river decoupling (pattern A core)
    below = (pct < 1.0).astype(float)
    o["m8_ix_rain7_below"] = d["rain_7d"] * below
    o["m8_ix_rain30_pct"] = d["rain_30d"] * pct
    o["m8_rain_per_rise"] = d["rain_7d"] / (d["wl_rise_3d"].abs() + 0.1)
    o["m8_ix_rain7_wet"] = d["rain_7d"] * d["ps_wet_top"]

    # --------------------------------------- E. upstream propagation (fix A)
    up1 = d["up_wl_pct_danger_lag1"]
    up1_2 = lag("up_wl_pct_danger_lag1", 2)
    up1_4 = lag("up_wl_pct_danger_lag1", 4)
    o["m8_up_pct_lag3"] = up1_2
    rise3 = up1 - up1_2
    o["m8_up_rise3"] = rise3
    o["m8_up_accel"] = rise3 - (up1_2 - up1_4)
    o["m8_up_minus_self"] = up1 - pct
    o["m8_up_max3"] = d[["up_wl_pct_danger_lag1",
                         "up_wl_pct_danger_lag2"]].max(axis=1)
    o["m8_up_rain7_lag3"] = lag("up_rain7d_lag1", 2)

    # ---------------------------------------- F. neighbour propagation (fix I)
    nb5 = d["nb5_wl_pct_danger"]
    o["m8_nb5_pct_lag1"] = lag("nb5_wl_pct_danger", 1)
    o["m8_nb5_pct_lag3"] = lag("nb5_wl_pct_danger", 3)
    o["m8_nb5_rise3"] = nb5 - lag("nb5_wl_pct_danger", 3)
    o["m8_nb5_minus_self"] = nb5 - pct
    o["m8_nb3_minus_self"] = d["nb3_wl_pct_danger"] - pct
    o["m8_nb5_n_danger_lag1"] = lag("nb5_n_wl_danger", 1)

    # --------------------------------------------- G. GloFAS upstream (v2 col)
    o["m8_gf_up_slope7"] = d["gf_up_ratio"] - d["gf_up_ratio_lag7"]
    o["m8_gf_up_accel"] = ((d["gf_up_ratio_lag1"] - d["gf_up_ratio_lag5"])
                           - (d["gf_up_ratio_lag5"] - d["gf_up_ratio_lag14"]))

    # ------------------------------------------------- H. reading context
    o["m8_obs_gap_days"] = d.groupby("station_id")["date"].transform(
        lambda s: s.diff().dt.days)

    return o


# ----------------------------------------------------------------- leakage
def audit() -> int:
    """Rebuild on data truncated to <2022-01-01; pre-2022 values must be identical."""
    print("=== MI-8 leakage audit (truncation test) ===")
    full = load_frame()
    b_full = build(full)
    truncated = full[full["date"] < TRAIN_END].copy()
    b_trunc = build(truncated)

    keep = full["date"] < TRAIN_END
    a = b_full[keep].reset_index(drop=True)
    c = b_trunc.reset_index(drop=True)
    check("L1", len(a) == len(c), f"full-block rows pre-2022={len(a)} truncated={len(c)}")

    bad = []
    for col in b_full.columns:
        x, y = a[col].to_numpy(), c[col].to_numpy()
        if x.shape != y.shape:
            bad.append((col, "shape"))
            continue
        same = (np.isclose(x, y, equal_nan=True, rtol=0, atol=1e-12).all())
        if not same:
            bad.append((col, "values"))
    check("L2", not bad, f"columns changed under truncation: {bad}")
    n = len(b_full.columns)
    print(f"  audited {n} columns; a forward shift or a whole-sample statistic "
          f"would have failed L2.")
    return 0 if (not bad) else 1


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true",
                    help="truncation leakage proof only; writes nothing")
    args = ap.parse_args(argv)

    if args.audit:
        return audit()

    print("=== MI-8 build: response-lag feature block ===")
    d = load_frame()
    block = build(d)

    # ---- self-checks -------------------------------------------------------
    print("--- checks ---")
    check("N1", len(block) == len(d), f"{len(block)} rows")
    check("N2", block.index.equals(d.index), "index aligned to the base frame")
    check("N3", all(c.startswith(PREFIX) for c in block.columns),
          f"all {len(block.columns)} columns carry the m8_ prefix")
    check("N4", len(set(block.columns)) == len(block.columns), "no duplicate column names")

    existing = set()
    for p in ("feature_sets.json", "feature_sets_v2.json", "feature_sets_v2_era5.json"):
        for s in json.loads((PROC / p).read_text(encoding="utf-8"))["sets"].values():
            existing |= set(s)
    clash = sorted(set(block.columns) & existing)
    check("N5", not clash, f"name collisions with frozen sets: {clash}")

    const = [c for c in block.columns if block[c].nunique(dropna=True) <= 1]
    check("N6", not const, f"constant columns: {const}")

    # Missingness is judged against the table this block JOINS, not against zero:
    # FULL_V2 already ships `up_wl_pct_danger_lag1` at 70.4 % NaN because upstream
    # coverage exists for only 50 of 133 stations, so the inherited upstream subgroup
    # is expected to sit at the same level.  A column materially worse than the base
    # table's own worst feature means this build is broken, so that is what fails.
    nanrate = {c: float(block[c].isna().mean()) for c in block.columns}
    base_cols = [c for c in v2_sets()["FULL_V2"] if c in d.columns]
    base_worst = max(float(d[c].isna().mean()) for c in base_cols)
    worst = sorted(nanrate.items(), key=lambda kv: -kv[1])[:3]
    check("N7", max(nanrate.values()) <= base_worst + 0.05,
          f"max NaN {max(nanrate.values()):.3f} vs base-table worst {base_worst:.3f} "
          f"-> {worst}")
    inherited = sorted(c for c, r in nanrate.items() if r > 0.30)
    if inherited:
        print(f"       note: {len(inherited)} columns exceed 0.30 NaN, all inherited "
              f"from sparse upstream sources: {inherited}")

    # every lag must be non-positive in its correlation with the raw series lead
    # (a *forward* shift would make a column lead its own driver); spot-check
    # three lag features against their drivers at the row level.
    lead_ok = True
    for col, drv in (("m8_pct_lag1", "wl_pct_danger"),
                     ("m8_nb5_pct_lag1", "nb5_wl_pct_danger"),
                     ("m8_dist_lag1", "wl_dist_to_danger")):
        x, y = block[col].to_numpy(), d[drv].to_numpy()
        # lagged column must equal the driver shifted by one row within station
        expect = d.groupby("station_id")[drv].shift(1).to_numpy()
        if not np.allclose(x, expect, equal_nan=True):
            lead_ok = False
    check("N8", lead_ok, "lag columns equal their driver shifted BACKWARD by one row")

    # forward-dependence proof (same rule as --audit, on the full build)
    keep = d["date"] < TRAIN_END
    b_trunc = build(d[keep].copy())
    a = block[keep].reset_index(drop=True)
    c = b_trunc.reset_index(drop=True)
    changed = [col for col in block.columns
               if not np.allclose(a[col].to_numpy(), c[col].to_numpy(),
                                  equal_nan=True, rtol=0, atol=1e-12)]
    check("N9", not changed, f"columns sensitive to post-2021 rows: {changed}")

    check("N10", block.notna().sum().sum() > 0, "block carries values")
    finite = np.isfinite(block.to_numpy(dtype=float, na_value=np.nan)).sum()
    check("N11", finite > 0, f"{finite} finite cells")
    check("N12", str(OUT_CSV.parent).endswith("processed"), f"writes under {OUT_CSV.parent}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)

    # ---- write -------------------------------------------------------------
    if n_fail == 0:
        out = d[["station_id", "date"]].join(block)
        out.to_csv(OUT_CSV, index=False)
        sets = v2_sets()
        full_v3 = list(sets["FULL_V2"]) + list(block.columns)
        payload = {
            "created": pd.Timestamp.now(tz="UTC").isoformat(),
            "phase": "MI-8 (score push: response-lag block)",
            "source_base": "data/processed/experimental_dataset.csv",
            "source_v2": "data/processed/mi_v2_features.csv",
            "rows": int(len(out)),
            "leakage_rule": ("backward shifts / trailing windows only; proven by the "
                             "truncation audit (--audit): pre-2022 values are identical "
                             "to the full build"),
            "train_end": TRAIN_END,
            "prefix": PREFIX,
            "sets": {**sets, "FULL_V3": full_v3},
            "counts": {k: len(v) for k, v in {**sets, "FULL_V3": full_v3}.items()
                       if isinstance(v, list)},
            "m8_columns": list(block.columns),
            "m8_n": int(len(block.columns)),
            "reason": ("Phase-30 pattern A: missed floods are early-stage (below danger, "
                       "rising, wet catchment); this block adds approach shape, proximity "
                       "persistence, antecedent rain and upstream/neighbour propagation."),
        }
        OUT_JSON.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {OUT_CSV.name} ({out.shape[0]} x {out.shape[1]}) "
              f"and {OUT_JSON.name} (FULL_V3 = {len(full_v3)} features)")
    else:
        print(f"\n{n_fail} check(s) FAILED -- nothing written")

    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
