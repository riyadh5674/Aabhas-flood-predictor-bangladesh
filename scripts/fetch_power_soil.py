"""MI-1a: NASA POWER soil wetness for all FFWC stations (incremental).

Provides GWETTOP / GWETROOT / GWETPROF (surface / root-zone / profile soil wetness,
MERRA-2).  Writes data/raw/power_soil/<station_id>.csv (append-only, de-duplicated)
then concatenates into power_soil_daily.csv.

Usage:
    python scripts/fetch_power_soil.py                      # catch up to yesterday
    python scripts/fetch_power_soil.py --end 2026-11-03
    python scripts/fetch_power_soil.py --lookback-days 14   # re-fetch last two weeks
    python scripts/fetch_power_soil.py --concat-only        # rebuild the combined CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "data" / "raw" / "power_soil"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (flood-predictor MI-1)"}
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START = "20130101"
PARAMS = "GWETTOP,GWETROOT,GWETPROF"
RENAME = {"GWETTOP": "ps_top", "GWETROOT": "ps_root", "GWETPROF": "ps_prof"}
VALUE_COLS = list(RENAME.values())


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def fetch_one(
    sid: int, lat: float, lon: float, end: str, lookback_days: int, retries: int = 5
) -> tuple[int, str]:
    dst = OUT / f"{sid}.csv"
    old = None
    fetch_start = START
    if dst.exists() and dst.stat().st_size > 300:
        try:
            old = _normalise(pd.read_csv(dst))
        except Exception:
            old = None
        if old is not None and not old.empty:
            last = old["date"].max()
            fetch_start = (
                pd.Timestamp(last) - pd.Timedelta(days=max(lookback_days - 1, 0))
            ).strftime("%Y%m%d")
            if fetch_start < START:
                fetch_start = START
            if fetch_start > end:
                return sid, "current"

    params = {
        "parameters": PARAMS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": fetch_start,
        "end": end,
        "format": "JSON",
    }
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, headers=H, timeout=120)
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                last_err = "429"
                continue
            r.raise_for_status()
            payload = r.json()["properties"]["parameter"]
            new = pd.DataFrame(payload)
            new.index.name = "ymd"
            new = new.reset_index()
            new["date"] = pd.to_datetime(
                new["ymd"].astype(str), format="%Y%m%d"
            ).dt.strftime("%Y-%m-%d")
            new = new.drop(columns=["ymd"]).rename(columns=RENAME)
            for col in VALUE_COLS:
                new.loc[new[col] <= -998, col] = pd.NA
            # NASA POWER returns -999 fill values for days it has not published yet.
            # Drop those rows so a refresh never overwrites valid history with NaNs.
            before = len(new)
            new = new.dropna(subset=VALUE_COLS, how="all")
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
            dropped = before - len(new)
            suffix = f" (+{len(new)} rows, {dropped} unpublished)" if dropped else f" (+{len(new)} rows)"
            return sid, f"ok{suffix}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:70]}"
            time.sleep(5 * (attempt + 1))
    return sid, f"FAIL {last_err}"


def concatenate(requested_end: str) -> None:
    files = sorted(
        [p for p in OUT.glob("*.csv") if p.stem.isdigit()], key=lambda p: int(p.stem)
    )
    if not files:
        print("No per-station soil files found.")
        return
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_df = all_df.drop_duplicates(["station_id", "date"], keep="last").sort_values(
        ["station_id", "date"]
    )
    big = OUT / "power_soil_daily.csv"
    all_df.to_csv(big, index=False)
    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    valid = all_df.dropna(subset=["ps_top"])
    manifest = {
        "source": "NASA POWER / MERRA-2",
        "url": URL,
        "params": PARAMS,
        "start": str(all_df["date"].min()),
        "end": str(all_df["date"].max()),
        "valid_end": str(valid["date"].max()) if not valid.empty else None,
        "requested_end": requested_end,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "stations": int(all_df["station_id"].nunique()),
        "rows": int(len(all_df)),
        "null_top": int(all_df["ps_top"].isna().sum()),
        "output": str(big),
        "sha256": sha,
    }
    (OUT / "power_soil_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"WROTE {big}  rows={len(all_df):,}  stations={all_df.station_id.nunique()}  "
        f"range={manifest['start']}..{manifest['end']}  valid_to={manifest['valid_end']}  "
        f"sha={sha[:16]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental NASA POWER soil refresh")
    ap.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).strftime("%Y%m%d"),
        help="Last date to fetch (YYYYMMDD). Default: yesterday.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Re-fetch this many trailing days to absorb revisions (default 7).",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--concat-only", action="store_true")
    args = ap.parse_args()

    if args.lookback_days < 1:
        raise SystemExit("--lookback-days must be >= 1")
    if args.concat_only:
        concatenate(args.end)
        return 0

    st = pd.read_csv(ROOT / "data" / "interim" / "mi_stations.csv")
    # New FFWC stations are sometimes absent from the MI station list; include the
    # official master so a station never loses its soil features just because the
    # interim list is stale.
    ff_path = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
    if ff_path.exists():
        ff = pd.read_csv(ff_path).rename(columns={"id": "station_id", "long": "lon"})
        ff = (
            ff[["station_id", "lat", "lon"]]
            .dropna()
            .drop_duplicates("station_id")
        )
        extra = ff[~ff["station_id"].isin(st["station_id"])]
        if len(extra):
            print(f"adding {len(extra)} FFWC station(s) missing from mi_stations.csv")
            st = pd.concat([st, extra], ignore_index=True)
    if args.limit:
        st = st.head(args.limit)
    print(f"NASA POWER soil: {len(st)} stations  end={args.end}  lookback={args.lookback_days}d")

    ok = fail = current = 0
    failures: list[tuple[int, str]] = []
    t0 = time.time()
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
            for row in st.itertuples()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sid, msg = fut.result()
            if msg == "current":
                current += 1
            elif msg.startswith("ok"):
                ok += 1
            else:
                fail += 1
                failures.append((sid, msg))
                print(f"  [{sid}] {msg}")
            if i % 20 == 0 or i == len(futs):
                print(
                    f"  {i}/{len(futs)}  updated={ok} current={current} failed={fail}  "
                    f"{time.time()-t0:.0f}s"
                )

    print()
    concatenate(args.end)
    if failures:
        print(f"WARN: {len(failures)} station(s) failed; existing rows were kept.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
