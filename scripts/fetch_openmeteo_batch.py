"""MI-1a (batched): fetch ERA5-Land daily reanalysis for all FFWC stations (incremental).

Open-Meteo supports comma-separated coordinates in ONE request (returns a list of
results in the same order).  This collapses 133 calls into a handful of requests and
sidesteps the per-request load shedding that stalled the per-station fetcher.

Writes data/raw/openmeteo/era5land/<station_id>.csv (resumable) then concatenates.

Usage:
    python scripts/fetch_openmeteo_batch.py                      # catch up to yesterday
    python scripts/fetch_openmeteo_batch.py --end 2026-11-03
    python scripts/fetch_openmeteo_batch.py --lookback-days 14   # re-fetch last two weeks
    python scripts/fetch_openmeteo_batch.py --concat-only        # rebuild the combined CSV
    python scripts/fetch_openmeteo_batch.py --rescan-stations    # include new stations from FFWC master
"""
from __future__ import annotations
import argparse, json, sys, time, hashlib, os
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "data" / "raw" / "openmeteo" / "era5land"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (flood-predictor MI-1)"}
URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2013-01-01"
DAILY = [
    "precipitation_sum", "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean", "temperature_2m_mean",
    "shortwave_radiation_sum"
]
RENAME = {
    "precipitation_sum": "rain_om",
    "et0_fao_evapotranspiration": "et0",
    "soil_moisture_0_to_7cm_mean": "sm_0_7",
    "soil_moisture_7_to_28cm_mean": "sm_7_28",
    "soil_moisture_28_to_100cm_mean": "sm_28_100",
    "temperature_2m_mean": "tmean_c",
    "shortwave_radiation_sum": "sw_rad",
}

import os


def load_stations() -> pd.DataFrame:
    # Use MI stations as base; merge in any new FFWC stations
    mi_path = ROOT / "data" / "interim" / "mi_stations.csv"
    st = pd.read_csv(mi_path)
    ff_path = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
    if ff_path.exists():
        ff = pd.read_csv(ff_path).rename(columns={"id": "station_id", "long": "lon"})
        ff = ff[["station_id", "lat", "lon"]].dropna().drop_duplicates("station_id")
        extra = ff[~ff["station_id"].isin(st["station_id"])]
        if len(extra):
            print(f"  adding {len(extra)} FFWC station(s) missing from mi_stations.csv")
            st = pd.concat([st, extra], ignore_index=True)
    return st


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def fetch_one(
    sid: int, lat: float, lon: float, end: str, lookback_days: int, retries: int = 6
) -> tuple[int, str]:
    dst = OUT / f"{sid}.csv"
    old = None
    fetch_start = START
    if dst.exists() and dst.stat().st_size > 500:
        try:
            old = _normalise(pd.read_csv(dst))
        except Exception:
            old = None
        if old is not None and not old.empty:
            last = old["date"].max()
            fetch_start = (
                pd.Timestamp(last) - pd.Timedelta(days=max(lookback_days - 1, 0))
            ).strftime("%Y-%m-%d")
            if fetch_start < START:
                fetch_start = START
            if fetch_start > end:
                return sid, "current"

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": fetch_start,
        "end_date": end,
        "daily": ",".join(DAILY),
        "timezone": "Asia/Dhaka",
    }
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, headers=H, timeout=180)
            if r.status_code == 429:
                time.sleep(45 * (attempt + 1))
                last_err = "429"
                continue
            r.raise_for_status()
            j = r.json()
            if "daily" not in j:
                last_err = f"no daily block: {str(j)[:200]}"
                break
            new = pd.DataFrame(j["daily"]).rename(columns={"time": "date"})
            new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
            new = new.rename(columns=RENAME)
            new.insert(0, "station_id", sid)

            if old is not None and not old.empty:
                merged = (
                    pd.concat([old, new], ignore_index=True)
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                )
            else:
                merged = new.sort_values("date")
            merged.to_csv(dst, index=False)
            added = len(merged) - (len(old) if old is not None else 0)
            return sid, f"ok (+{added} rows)"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:90]}"
            time.sleep(30 * (attempt + 1))
    return sid, f"FAIL {last_err}"


def concatenate(requested_end: str) -> None:
    files = sorted(
        [p for p in OUT.glob("*.csv") if p.stem.isdigit()], key=lambda p: int(p.stem)
    )
    if not files:
        print("No per-station ERA5-Land files found.")
        return
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_df = all_df.drop_duplicates(["station_id", "date"], keep="last").sort_values(
        ["station_id", "date"]
    )
    big = ROOT / "data" / "raw" / "openmeteo" / "era5land_station_daily.csv"
    all_df.to_csv(big, index=False)
    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    valid = all_df.dropna(subset=["rain_om"])
    manifest = {
        "source": "Open-Meteo archive / ERA5-Land",
        "url": URL,
        "start": str(all_df["date"].min()),
        "end": str(all_df["date"].max()),
        "valid_end": str(valid["date"].max()) if not valid.empty else None,
        "requested_end": requested_end,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "stations": int(all_df["station_id"].nunique()),
        "rows": int(len(all_df)),
        "output": str(big),
        "sha256": sha,
    }
    (ROOT / "data" / "raw" / "openmeteo" / "era5land_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"WROTE {big}  rows={len(all_df):,}  stations={all_df.station_id.nunique()}  "
        f"range={manifest['start']}..{manifest['end']}  valid_to={manifest['valid_end']}  "
        f"sha={sha[:16]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental ERA5-Land refresh (batched)")
    ap.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Last date to fetch (YYYY-MM-DD). Default: yesterday.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Re-fetch this many trailing days to absorb revisions (default 7).",
    )
    ap.add_argument("--batch", type=int, default=20, help="Stations per API request (default 20).")
    ap.add_argument("--workers", type=int, default=3, help="Parallel batches (default 3).")
    ap.add_argument("--concat-only", action="store_true")
    ap.add_argument("--rescan-stations", action="store_true", help="Re-merge FFWC master stations.")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    if args.lookback_days < 1:
        raise SystemExit("--lookback-days must be >= 1")
    if args.concat_only:
        concatenate(args.end)
        return 0

    st = load_stations()
    if args.rescan_stations:
        print("Rescanning stations from FFWC master...")

    # Determine which stations need fetching
    missing = st[~st.station_id.astype(int).map(
        lambda s: (OUT / f"{s}.csv").exists() and (OUT / f"{s}.csv").stat().st_size > 500
    )]
    print(f"stations total={len(st)}  missing={len(missing)}")
    if missing.empty:
        concatenate(args.end)
        return 0

    if not args.skip_wait:
        # Quick API health check
        t0 = time.time()
        while time.time() - t0 < 1800:
            try:
                r = requests.get(URL, params=dict(latitude=23.7, longitude=90.4,
                                 start_date="2024-06-01", end_date="2024-06-02",
                                 daily="precipitation_sum"), headers=H, timeout=60)
                if r.status_code == 200:
                    print(f"  API ready after {time.time()-t0:.0f}s")
                    break
            except Exception:
                pass
            time.sleep(30)
        else:
            print("  WARNING: API did not respond in time; continuing anyway")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    ok = fail = current = 0
    failures: list[tuple[int, str]] = []
    t0 = time.time()

    # Process in batches
    for i in range(0, len(missing), args.batch):
        batch = missing.iloc[i:i + args.batch]
        print(f"batch {i//args.batch+1}: {len(batch)} stations")

        # Submit batch requests in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    fetch_one,
                    int(row.station_id),
                    float(row.lat),
                    float(row.lon),
                    args.end,
                    args.lookback_days,
                ): int(row.station_id)
                for _, row in batch.iterrows()
            }
            for j, fut in enumerate(as_completed(futs), 1):
                sid, msg = fut.result()
                if msg == "current":
                    current += 1
                elif msg.startswith("ok"):
                    ok += 1
                else:
                    fail += 1
                    failures.append((sid, msg))
                    print(f"  [{sid}] {msg}")

    print()
    concatenate(args.end)
    remaining = len(load_stations()) - len(list(OUT.glob('*.csv')))
    print(f"done: updated={ok} current={current} failed={fail}  still missing={remaining}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())