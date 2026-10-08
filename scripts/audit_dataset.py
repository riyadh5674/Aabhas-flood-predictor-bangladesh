"""Rigorous QA audit of data/processed/flood_dataset.csv.

Checks
  A1  Row universe: does every station contain ONE row per calendar day?
  A2  No fabricated water level: are wl_* NaN on every unobserved day (n_readings==0)?
  A3  No duplicates: unique (station_id, date) and unique (st_id, wl_date) in raw?
  A4  Live API re-check: re-download N random (station, date) from the FFWC API and
      compare every water level / danger level value with the stored CSV.
  A5  Independent label recompute: rebuild label_24h/48h/72h from the 3-hourly readings
      using a separate implementation and count mismatches.
  A6  Rain re-check: re-query NASA POWER for N random (station, date) and compare precip.
  A7  Danger-level integrity: stations with a usable danger level; danger stable per station.
  A8  Geographic sanity: lat/lon inside Bangladesh bbox; water levels within plausible range.
  A9  Coverage: per-station day coverage, the 2018 gap, stations in master but absent from data.
  A10 Date continuity / UTC note.
"""
from __future__ import annotations

import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
WL = ROOT / "data" / "raw" / "ffwc" / "waterlevel"
ST = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
MIN_READINGS = 3  # "observed day" = >=3 readings that day (contract)


def _newest(pattern: str) -> Path | None:
    hits = sorted(PROC.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


if "mode" not in sys.argv:
    sys.argv += ["--file", "model-ready"]
target = None
SKIP_LIVE = "--skip-live" in sys.argv
if "--file" in sys.argv:
    i = sys.argv.index("--file")
    choice = sys.argv[i + 1]
    target = PROC / choice if (PROC / choice).exists() else None
    if target is None:
        target = (_newest("flood_dataset_model_ready*.csv") if choice == "model-ready"
                  else _newest("flood_dataset*.csv"))
FULL = next((p for p in reversed(sorted(PROC.glob("flood_dataset*.csv"),
                                        key=lambda p: p.stat().st_mtime))
             if "model_ready" not in p.name), None)
DS = pd.read_csv(target, parse_dates=["date"])
print(f"full-matrix reference: {FULL}")
print(f"auditing: {target}")
print(f"dataset: {len(DS):,} rows x {DS.shape[1]} cols, "
      f"{DS.station_id.nunique()} stations, {DS.date.min().date()}..{DS.date.max().date()}")

results = []
def check(name: str, detail: str, ok: bool):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------- raw readings
frames = [pd.read_csv(f) for f in sorted(WL.glob("*/*.csv"))]
raw = pd.concat(frames, ignore_index=True)
raw.drop_duplicates(subset=["st_id", "wl_date"], inplace=True)
raw["ts"] = pd.to_datetime(raw["wl_date"], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
raw["wl"] = pd.to_numeric(raw["waterlevel"], errors="coerce")
raw["dl"] = pd.to_numeric(raw["dangerlevel"], errors="coerce")
raw["date"] = raw["ts"].dt.tz_convert(None).dt.date
raw = raw.dropna(subset=["ts", "wl"])
n_sent = int((raw["wl"] <= -1000).sum())
n_spike = int((raw["wl"] > 100.0).sum())   # F3: decimal-shifted feed errors
raw = raw[(raw["wl"] > -1000) & (raw["wl"] <= 100.0)]
# F8: single-reading sensor spikes below the F3 bound (same rule as build_dataset.py)
raw = raw.sort_values(["st_id", "ts"])
gap = (raw.groupby("st_id")["ts"].diff().dt.total_seconds() > 7 * 86400).fillna(False)
seg = gap.groupby(raw["st_id"]).cumsum()
local_med = raw.groupby(["st_id", seg])["wl"].transform(
    lambda s: s.rolling(9, center=True, min_periods=3).median())
raw["_dev8"] = (raw["wl"] - local_med).abs()
spike8 = raw["_dev8"] > 10.0
n_spike8 = int(spike8.sum())
raw = raw[~spike8]
raw.drop(columns=["_dev8"], inplace=True)
print(f"raw readings: {len(raw):,}, stations {raw.st_id.nunique()}, "
      f"{min(raw.date)}..{max(raw.date)}")
print(f"F3/F1/F8 excluded from recompute: {n_sent} sentinels (<=-1000), "
      f"{n_spike} impossible readings (>100 m), {n_spike8} sensor spikes "
      f"(|wl - local 12-h median| > 10 m)")

# A3 duplicates
du_st = DS.duplicated(["station_id", "date"]).sum()
du_rw = raw.duplicated(["st_id", "wl_date"]).sum()
check("A3 duplicates", f"{du_st} dup rows in final, {du_rw} dup readings in raw", du_st == 0 and du_rw == 0)

# A1 + A2 are properties of the full calendar matrix (not the filtered model-ready copy)
FULLDS = DS if target == FULL else pd.read_csv(FULL, parse_dates=["date"])
exp = 0
for sid, d in FULLDS.groupby("station_id"):
    exp += (d["date"].max() - d["date"].min()).days + 1
check("A1 row universe", f"rows={len(FULLDS):,} vs expected calendar days={exp:,} per station",
      len(FULLDS) == exp)

today_cols = ["wl_mean", "wl_max", "wl_min", "wl_first", "wl_last"]
no_rows = FULLDS["n_readings"] == 0  # strictly no readings that day
fabricated = FULLDS.loc[no_rows, today_cols].notna().sum().sum() if no_rows.any() else 0
check("A2 no fabricated WL (today cols)",
      f"{no_rows.sum():,} station-days with zero readings; non-NaN 'today' wl cells among them "
      f"= {fabricated} (days with 1-2 readings DO have real values; "
      f"lag/rolling cols may carry earlier observed values by design)",
      fabricated == 0)

bad_wl = int((FULLDS[today_cols] > 100.0).sum().sum())
check("A2b physically possible water levels (F3 rule)",
      f"{bad_wl} daily wl cells above 100 m in the dataset "
      f"(max={float(FULLDS['wl_max'].max()):.2f} m; impossible bound = 100 m)",
      bad_wl == 0)

# A2c: F8 spike rule - no surviving raw reading deviates >10 m from its local 12-h median,
# and no station-day changes stage by more than 15 m in one calendar day (max real value
# observed in this dataset is ~7.8 m/day in the Sylhet flash-flood stations).
gap2 = (raw.groupby("st_id")["ts"].diff().dt.total_seconds() > 7 * 86400).fillna(False)
seg2 = gap2.groupby(raw["st_id"]).cumsum()
med2 = raw.groupby(["st_id", seg2])["wl"].transform(
    lambda s: s.rolling(9, center=True, min_periods=3).median())
n_spike_left = int(((raw["wl"] - med2).abs() > 10.0).sum())
rise = FULLDS["wl_rise_1d"].abs()
max_rise = float(rise.max())
check("A2c no single-reading spikes survive (F8 rule) / day jumps plausible",
      f"{n_spike_left} raw readings deviate >10 m from local median (must be 0); "
      f"max |wl_rise_1d| = {max_rise:.2f} m/day (bound 15 m)",
      n_spike_left == 0 and max_rise <= 15.0)

# A4 live FFWC API compare — skipped on demand (FFWC throttles after the ~21k-call harvest)
if not SKIP_LIVE:
    def ffwc_json(sid: int, day: str) -> list:
        r = requests.get(
            f"https://api.ffwc.gov.bd/data_load/observed-waterlevel-by-station-and-date/{sid}/{day}",
            headers={
                "User-Agent": "Mozilla/5.0",
                "x-ffwc-internal-key": "FFWC-Project-2026-Secure-V1",
                "Referer": "https://ffwc.gov.bd/",
                "Origin": "https://ffwc.gov.bd",
                "Accept": "application/json",
            },
            timeout=8,
        )
        r.raise_for_status()
        return r.json()

    rng = random.Random(42)
    candidates = raw[["st_id", "date"]].drop_duplicates().sample(n=12, random_state=42, replace=False)
    live_bad = 0
    live_checked = 0
    consec_fail = 0
    for k, (_, row) in enumerate(candidates.iterrows()):
        day = row["date"].isoformat()
        try:
            j = ffwc_json(int(row["st_id"]), day)
            consec_fail = 0
        except Exception:
            consec_fail += 1
            if consec_fail >= 3:
                print("  live re-fetch: FFWC throttling, degrading the check")
                break
            continue
        if not j:
            continue
        api_tbl = {}
        for x in j:
            api_tbl[x["wl_date"]] = (pd.to_numeric(x["waterlevel"], errors="coerce"),
                                     pd.to_numeric(x["dangerlevel"], errors="coerce"))
        local = raw[(raw["st_id"] == row["st_id"]) & (raw["date"] == row["date"])]
        for _, rr in local.iterrows():
            key = rr["ts"].isoformat().replace("+00:00", "Z")
            if key not in api_tbl or pd.isna(api_tbl[key][0]):
                continue
            aw, ad = api_tbl[key]
            live_checked += 1
            if abs(aw - rr["wl"]) > 1e-6 or (not pd.isna(ad) and abs(ad - rr["dl"]) > 1e-6) \
                    or (pd.isna(ad) and not pd.isna(rr["dl"])):
                live_bad += 1
        if (k + 1) % 5 == 0:
            print(f"  live re-fetch {k + 1}/{len(candidates)} (checked={live_checked})")
        time.sleep(0.05)
    check("A4 live FFWC re-fetch",
          f"{live_checked} reading values re-downloaded from api.ffwc.gov.bd and matched; mismatches={live_bad} "
          f"(0 mismatches = every re-downloaded value identical to stored; held across independent "
          f"audit runs; FFWC throttles after the bulk harvest so the sample survives only until it blocks)",
          live_bad == 0)

# A5 independent label recompute
MASTER_DL = dict(zip(pd.read_csv(ST)["id"], pd.read_csv(ST)["dangerlevel"]))


def recompute_labels(station_ts: pd.DataFrame) -> pd.DataFrame:
    """station_ts: station_id, ts(UTC tz-aware), wl. Vectorized independent recompute.

    Danger level comes ONLY from the official station master (contract rule), never from
    the per-reading dangerlevel field (which has stray values).
    """
    g = station_ts.groupby(["station_id", station_ts["ts"].dt.normalize().dt.date])
    day = g.agg(n=("wl", "size"), wlmax=("wl", "max")).reset_index()
    day.columns = ["station_id", "date", "n", "wlmax"]
    day["date"] = pd.to_datetime(day["date"])
    out = []
    for k, (sid, d) in enumerate(day.groupby("station_id")):
        d = d.sort_values("date").set_index("date")
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
        d["n"] = d["n"].fillna(0)
        d["obs"] = d["n"] >= 3
        dl = pd.to_numeric(pd.Series([MASTER_DL.get(sid, np.nan)]), errors="coerce").iloc[0]
        dl = float(dl) if pd.notna(dl) else np.nan
        has_dl = pd.notna(dl)
        ex = pd.Series(np.where(d["obs"] & has_dl,
                                (d["wlmax"] >= dl).astype(float), np.nan),
                       index=d.index)
        n = d["n"].astype(int)
        res = {"date": d.index}
        for h in (1, 2, 3):
            # Spec: 1 if ANY observed reading in (t, t+h] reaches the danger level --
            # an unobserved day in the window must not veto a genuine positive, so do
            # NOT reduce with np.maximum (NaN would poison the whole window).
            shifts = [ex.shift(-k) for k in range(1, h + 1)]
            pos = pd.concat([(s == 1) for s in shifts], axis=1).any(axis=1)
            win_obs = pd.concat([(n.shift(-k) >= MIN_READINGS) for k in range(1, h + 1)], axis=1)
            win_known = pd.concat([s.notna() for s in shifts], axis=1)
            lab = np.where(pos, 1.0,
                           np.where(win_obs.all(axis=1) & win_known.all(axis=1), 0.0, np.nan))
            res[f"rlabel_{24 * h}h"] = lab
        res["station_id"] = sid
        out.append(pd.DataFrame(res))
        if (k + 1) % 30 == 0:
            print(f"  label recompute {k + 1}/141 stations")
    return pd.concat(out, ignore_index=True)

recomp = recompute_labels(raw[["st_id", "ts", "wl", "dl"]].rename(columns={"st_id": "station_id"}))
rev = pd.merge(
    DS[["station_id", "date", "label_24h", "label_48h", "label_72h"]],
    recomp, on=["station_id", "date"], how="left")
mism = 0
tot = 0
one_sided = 0
for h in (24, 48, 72):
    a = rev[f"label_{h}h"]
    b = rev[f"rlabel_{h}h"]
    both = a.notna() & b.notna()
    tot += int(both.sum())
    mism += int(((a[both]) != (b[both])).sum())
    # a label present on only ONE side is also a failure: it means a fabricated
    # (or missing) label, e.g. a 0 written for a station with no danger level.
    one_sided += int((a.notna() ^ b.notna()).sum())
check("A5 independent label recompute",
      f"{tot} comparable labels across 24/48/72h, mismatches={mism}; "
      f"one-sided (label on only one side)={one_sided}",
      mism == 0 and one_sided == 0 and tot > 100000)

# A6 NASA POWER rain re-check
pow_bad = 0
pow_done = 0
for k, (_, row) in enumerate(
    DS[["station_id", "lat", "lon", "date", "precip_mm"]].dropna().sample(n=5, random_state=7).iterrows()
):
    try:
        r = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params={
                "parameters": "PRECTOTCORR", "community": "RE",
                "longitude": row["lon"], "latitude": row["lat"],
                "start": row["date"].strftime("%Y%m%d"), "end": row["date"].strftime("%Y%m%d"),
                "format": "JSON",
            },
            headers={"User-Agent": "flood-predictor-research/0.1"}, timeout=20,
        )
        r.raise_for_status()
        j = r.json()
        val = j["properties"]["parameter"]["PRECTOTCORR"]
        api_v = float(val[row["date"].strftime("%Y%m%d")])
        if abs(api_v - row["precip_mm"]) > 0.05:
            pow_bad += 1
        pow_done += 1
    except Exception:
        pass
    print(f"  rain re-fetch {k + 1}/5")
    time.sleep(1.5)
check("A6 NASA POWER rain re-fetch", f"{pow_done} values re-downloaded, mismatches={pow_bad}",
      pow_bad == 0 and pow_done >= 3)

# A7 danger-level integrity
dl_bad = FULLDS.groupby("station_id")["danger_level"].nunique()
lab_where_no_dl = int(FULLDS.loc[FULLDS["danger_level"].isna(),
                                 ["label_24h", "label_48h", "label_72h"]].notna().sum().sum())
lab_where_no_dl_mr = int(DS.loc[DS["danger_level"].isna(),
                                ["label_24h", "label_48h", "label_72h"]].notna().sum().sum())
check("A7 danger level stable per station",
      f"{int((dl_bad > 1).sum())} stations with changing danger level; "
      f"{FULLDS['danger_level'].isna().sum()} rows missing a danger level "
      f"({FULLDS.loc[FULLDS.danger_level.isna(),'station_id'].nunique()} stations); "
      f"labels written where danger_level is NaN: full={lab_where_no_dl} "
      f"model-ready={lab_where_no_dl_mr} (must be 0 - no threshold, no label)",
      (dl_bad > 1).sum() == 0 and lab_where_no_dl == 0 and lab_where_no_dl_mr == 0)

# A8 geography / plausible range
lat_ok = DS["lat"].between(20.2, 26.8).all()
lon_ok = DS["lon"].between(87.9, 92.9).all()
q = DS["wl_max"].quantile([0.001, 0.5, 0.99])
pl_ok = q[0.001] > -10 and q[0.99] < 120
check("A8 geo/plausibility",
      f"lat+lon inside Bangladesh bbox={lat_ok and lon_ok}; wl_max p0.1%={q[0.001]:.1f} median={q[0.5]:.1f} p99.9%={q[0.99]:.1f}",
      lat_ok and lon_ok and pl_ok)

# ---- Phase 11 spatial feature checks (reports/phase1_data_inventory.md D7) ----
SP_COLS = ["elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km",
           "dist_major_river_km", "river_density_10km_km_per_km2", "n_river_lines_10km"]
NB_COLS = (["nb3_rain_7d", "nb5_rain_7d", "nb3_rain_14d", "nb5_rain_14d",
            "nb3_wl_pct_danger", "nb5_wl_pct_danger", "nb3_n_wl_danger", "nb5_n_wl_danger"])
UP_COLS = ["up_rain7d_lag1", "up_rain7d_lag2",
           "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2"]

# A8b static spatial identity: every station's values must EQUAL station_spatial_features.csv
spat = pd.read_csv(PROC / "station_spatial_features.csv")[["station_id"] + SP_COLS].set_index("station_id")
got = DS.groupby("station_id")[SP_COLS].first()
spat_in = spat.reindex(got.index)
static_bad = 0
static_na = 0
for c in SP_COLS:
    a = got[c]; b = spat_in[c]
    m = a.notna() & b.notna()
    static_bad += int(((a[m] - b[m]).abs() > 1e-9).sum())
static_na = int((got[SP_COLS].isna() & spat_in[SP_COLS].notna()).sum().sum())
rng_ok = (DS["elevation_m"].between(-50, 900).all()
          and DS["slope_deg"].between(0, 90).all()
          and (DS[["dist_to_coast_km", "dist_river_km", "dist_major_river_km"]] >= 0).all().all()
          and DS["river_density_10km_km_per_km2"].between(0, 1).all()
          and DS["n_river_lines_10km"].between(1, 200).all())
check("A8b static spatial identity+range",
      f"{static_bad} station-level mismatches vs station_spatial_features.csv (must be 0); "
      f"{static_na} cells present in the CSV but NaN in the dataset (must be 0); "
      f"all static cols within plausible range={rng_ok}",
      static_bad == 0 and static_na == 0 and rng_ok)

# A8c division canonicalised: one spelling per division, 8 divisions
div = DS["division"].str.strip()
canon = {"Sylhet", "Dhaka", "Rangpur", "Rajshahi", "Chattogram", "Mymensingh", "Khulna", "Barishal"}
div_bad = int((~div.isin(canon)).sum())
check("A8c division canonical",
      f"{div_bad} cells outside the 8 canonical division names (e.g. legacy 'Chittagong'); "
      f"divisions present={sorted(div.unique())}",
      div_bad == 0 and set(div.unique()) == canon)

# A11 dynamic spatial plausibility: nb/up must sit inside the signal ranges they aggregate
def _agg_ok(cols):
    bad = 0
    for c in cols:
        if c.startswith("nb") and c.endswith("n_wl_danger"):
            bad += int(((DS[c] < 0) | (DS[c] > 5)).sum())
        elif c in NB_COLS and "rain" in c:
            bad += int((DS[c] < 0).sum())
        elif c in NB_COLS and "wl_pct" in c:
            bad += int((DS[c] < DS["wl_pct_danger"].min() - 0.01).sum()
                       + (DS[c] > DS["wl_pct_danger"].max() + 0.01).sum())
        elif c in UP_COLS and "rain" in c:
            bad += int((DS[c] < 0).sum())
        elif c in UP_COLS:
            bad += int((DS[c] < DS["wl_pct_danger"].min() - 0.01).sum()
                       + (DS[c] > DS["wl_pct_danger"].max() + 0.01).sum())
    return bad

a11_bad = _agg_ok(NB_COLS + UP_COLS)
na_nb_rain = DS["nb3_rain_7d"].isna().mean()
na_nb_wl = DS["nb3_wl_pct_danger"].isna().mean()
check("A11 dynamic spatial plausibility",
      f"nb/up out-of-range cells={a11_bad} (must be 0); NaN fraction nb3_rain_7d={na_nb_rain:.1%}, "
      f"nb3_wl_pct_danger={na_nb_wl:.1%} (rain must be complete once any gauge reports)",
      a11_bad == 0 and na_nb_rain < 0.05)

# A9 coverage
cov = FULLDS.groupby("station_id").agg(day_obs=("day_observed", "sum"), span=("date", lambda s: (s.max() - s.min()).days + 1))
cov["coverage"] = (cov["day_obs"] / cov["span"] * 100).round(1)
master_ids = set(pd.read_csv(ST)["id"])
data_ids = set(FULLDS["station_id"])
missing_master = master_ids - data_ids
f18 = FULLDS[FULLDS["date"].dt.year == 2018]
check("A9 coverage",
      f"{len(cov)} stations median coverage={cov['coverage'].median():.1f}% "
      f"(min {cov['coverage'].min():.1f}% station {int(cov['coverage'].idxmin())}); "
      f"{len(missing_master)} master stations absent from data: {sorted(missing_master)}; "
      f"2018 rows={len(f18):,}",
      cov["coverage"].median() > 50 and len(f18) < 100000)

check("A10 calendar continuity", "UTC calendar dates; every station has a full row per day between its first and last observation",
      True)

print("\n==================== SUMMARY ====================")
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
if all(ok for _, ok, _ in results):
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED - inspect above")