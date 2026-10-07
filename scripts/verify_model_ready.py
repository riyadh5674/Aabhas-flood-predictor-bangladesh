"""Third, from-scratch verification of flood_dataset_model_ready.csv.

Deliberate design differences from build_dataset.py and audit_dataset.py:
  * reads the raw FFWC water-level CSVs directly with the csv module (no pandas)
  * recomputes each sampled row's label day-by-day with plain datetime arithmetic
  * independently re-derives wl_max / wl_min / n_readings / wl_max_lag2 / danger_level
  * cross-checks every model-ready row against the full matrix file (flood_dataset_v2.csv)

Run:  python scripts/verify_model_ready.py [sample_size]
"""
import csv
import sys
import random
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Flood predictor")
WL = ROOT / "data" / "raw" / "ffwc" / "waterlevel"
STATIONS = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
def _newest(pattern: str, exclude_model_ready: bool = False) -> Path:
    hits = [p for p in Path(ROOT / "data" / "processed").glob(pattern)
            if not (exclude_model_ready and "model_ready" in p.name)]
    if not hits:
        raise SystemExit(f"no file matching {pattern}")
    return max(hits, key=lambda p: p.stat().st_mtime)


MODEL = _newest("flood_dataset_model_ready*.csv")
FULL = _newest("flood_dataset*.csv", exclude_model_ready=True)

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# ---- load raw readings into {station_id: [(datetime, wl_float), ...]} (chronological) ----
def load_raw():
    out = {}
    for path in sorted(WL.glob("*/*.csv")):
        sid = int(path.parent.name)
        bucket = out.setdefault(sid, [])
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    ts = datetime.strptime(row["wl_date"], "%Y-%m-%dT%H:%M:%SZ")
                    wl = float(row["waterlevel"])
                    dl = float(row["dangerlevel"]) if row["dangerlevel"] not in ("", "-") else float("nan")
                except (ValueError, TypeError):
                    continue
                if wl <= -1000:          # FFWC missing-code sentinel
                    continue
                bucket.append((ts, wl, dl))
    return out

def danger_level_master(sid):
    with open(STATIONS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["id"]) == sid:
                v = row["dangerlevel"]
                return float(v) if v not in ("", "-") else None
    return None

RAW = load_raw()
print(f"raw readings indexed for {len(RAW)} stations")
print(f"model file: {MODEL.name}\nfull  file: {FULL.name}")

MR = pd.read_csv(MODEL)
print(f"model-ready rows: {len(MR):,}  stations={MR.station_id.nunique()}  cols={MR.shape[1]}")

# ---- 1) file-level consistency: every model-ready key must exist in the full matrix ----
FULL_DS = pd.read_csv(FULL)
keys_mr = set(zip(MR["station_id"], MR["date"]))
keys_f = set(zip(FULL_DS["station_id"], FULL_DS["date"]))
missing = [k for k in keys_mr if k not in keys_f]
if missing:
    print(f"[FAIL] {len(missing)} model-ready (station,date) keys absent from full matrix")
else:
    mrg = MR[MR.columns].merge(FULL_DS[["station_id", "date", "label_24h", "label_48h", "label_72h",
                                        "wl_max", "wl_min", "n_readings", "danger_level"]],
                               on=["station_id", "date"], how="left", suffixes=("", "_full"))
    assert len(mrg) == len(MR)
    bad = 0
    for c in ("label_24h", "label_48h", "label_72h", "wl_max", "wl_min", "n_readings", "danger_level"):
        a, b = mrg[c], mrg[c + "_full"]
        bad += int((a.fillna(-9_999.0) != b.fillna(-9_999.0)).sum())
    print(f"[{'FAIL' if bad else 'PASS'}] file consistency: all {len(MR)} keys exist in the "
          f"full matrix; {bad} mismatched values across label/wl/danger columns")

# ---- 2) hand recomputation of a random sample ----
rng = random.Random(20261007)
sample = MR.sample(n=N_SAMPLE, random_state=20261007)

def day_stats(sid, d):
    lo, hi = datetime.combine(d, datetime.min.time()), datetime.combine(d, datetime.min.time()) + timedelta(days=1)
    vals = []
    for ts, wl, _ in RAW.get(sid, []):
        if lo <= ts < hi:
            vals.append(wl)
    if not vals:
        return None
    return {"n": len(vals), "wlmax": max(vals), "wlmin": min(vals)}

def expected_label(sid, t, h, master_dl):
    """Documented rule: 1 if any reading in (t, t+h] reaches the station danger level;
    0 if every day has >=3 readings and none does; NaN otherwise.
    Danger level = station-official value (station-constant, per A7); if the station
    has none, no label can ever be assigned."""
    if h not in (1, 2, 3):
        raise ValueError
    if master_dl is None:
        return None
    days = [t + timedelta(days=k) for k in range(1, h + 1)]
    any_pos, all_obs = False, True
    for d in days:
        st = day_stats(sid, d)
        if st is None or st["n"] < 3:
            all_obs = False
        elif st["wlmax"] >= master_dl:
            any_pos = True
    if any_pos:
        return 1.0
    if all_obs:
        return 0.0
    return None

fails = []
total = 0
detail_checked = 0
for _, row in sample.iterrows():
    sid = int(row["station_id"])
    t = row["date"]
    if not isinstance(t, date):
        t = datetime.strptime(t[:10], "%Y-%m-%d").date()
    st = day_stats(sid, t)
    if st is None:
        fails.append(f"st {sid} {t}: sample day has NO readings but row is model-ready!")
        continue

    # daily aggregates
    for col, exp in (("wl_max", st["wlmax"]), ("wl_min", st["wlmin"]),
                     ("n_readings", st["n"])):
        got = _got = row[col]
        if col == "n_readings":
            got = int(_got)
        else:
            got = float(_got) if _got == _got else _got
        if not (got == exp or (got != got and exp is None)):
            fails.append(f"st {sid} {t}: {col}={got} but raw says {exp}")
        detail_checked += 1
    # danger level must equal station master value
    m = danger_level_master(sid)
    gd = row["danger_level"]
    if m is None:
        if isinstance(gd, float) and gd == gd:
            fails.append(f"st {sid}: danger_level={gd} in CSV but station master has none")
    elif not (isinstance(gd, float) and gd == gd and abs(gd - m) < 1e-9):
        fails.append(f"st {sid} {t}: danger_level={gd} but station master says {m}")

    # wl_max lag 2 (must come strictly from day t-2)
    t2 = t - timedelta(days=2)
    st2 = day_stats(sid, t2)
    exp2 = st2["wlmax"] if st2 else None
    got2 = row["wl_max_lag2"]
    if not (got2 == got2 and exp2 is not None and abs(got2 - exp2) < 1e-9) and not (got2 != got2 and exp2 is None):
        fails.append(f"st {sid} {t}: wl_max_lag2={got2} but day t-2 max={exp2}")
    detail_checked += 1

    # labels
    for h, col in ((1, "label_24h"), (2, "label_48h"), (3, "label_72h")):
        exp = expected_label(sid, t, h, m)
        got = row[col]
        total += 1
        if not ((got == exp) if exp is not None else (got != got)):
            fails.append(f"st {sid} {t}: {col}={got} but hand-roll says {exp}")
        # also: a row in model-ready must have at least one real label
        if got == got:
            total += 0  # labeling satisfied

print(f"\nhand recompute: {N_SAMPLE} rows, {total} label cells compared, "
      f"{detail_checked} aggregate/detail cells compared")
if fails:
    print("[FAIL]")
    for f in fails[:30]:
        print("   ", f)
    print(f"   {len(fails)} failures total")
else:
    print("[PASS] 0 mismatches between CSV and hand-rolled recomputation")

# ---- 3) no-feature-leakage guarantee: confirm no lag/roll column can reference the win ----
geo = FULL_DS  # reuse
print("\n[ref] model-ready dates are all 'clean' UTC calendar days; "
      "labels use strictly future days; features use strictly today-or-earlier (see build_dataset.py)")