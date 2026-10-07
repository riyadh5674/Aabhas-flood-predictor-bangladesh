"""Daily weather at every FFWC station from NASA POWER (MERRA-2 reanalysis).

        - precipitation: PRECTOTCORR (mm/day)
        - temperature : T2M, T2M_MAX, T2M_MIN (deg C)
One call per station (no API key needed).  Writes
    data/raw/rainfall/weather_station_daily.csv
Usage: python -m flood.ingest.rainfall [--start 2016-12-01] [--end 2026-10-06]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
HEADERS = {"User-Agent": "flood-predictor-research/0.1 (contact: local research project)"}
RAW = Path(r"D:\Flood predictor\data\raw\rainfall")
STATIONS = Path(r"D:\Flood predictor\data\raw\ffwc\ffwc_stations.csv")
OUT = RAW / "weather_station_daily.csv"
PART = RAW / "weather_station_daily.partial.csv"
PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN"


def _get_point(lat: float, lon: float, start: str, end: str, tries: int = 5) -> pd.DataFrame:
    params = {
        "parameters": PARAMS,
        "community": "RE",
        "longitude": round(lon, 5),
        "latitude": round(lat, 5),
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    delay = 3.0
    for i in range(tries):
        r = requests.get(API, params=params, headers=HEADERS, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        j = r.json()
        param = j["properties"]["parameter"]
        if not param:
            return pd.DataFrame()
        dates = sorted(param["PRECTOTCORR"].keys())
        return pd.DataFrame(
            {
                "date": pd.to_datetime([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates]),
                "precip_mm": [float(param["PRECTOTCORR"].get(d) or 0) for d in dates],
                "tmax_c": [float(param["T2M_MAX"].get(d) or float("nan")) for d in dates],
                "tmin_c": [float(param["T2M_MIN"].get(d) or float("nan")) for d in dates],
                "tmean_c": [float(param["T2M"].get(d) or float("nan")) for d in dates],
            }
        )
    raise RuntimeError(f"too many failures for ({lat}, {lon})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-12-01")
    ap.add_argument("--end", default="2026-10-06")
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    st = pd.read_csv(STATIONS).dropna(subset=["lat", "long"]).drop_duplicates("id")
    expected = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days + 1

    done_frames: list[pd.DataFrame] = []
    if PART.exists():
        done = pd.read_csv(PART, parse_dates=["date"])
        complete = done.groupby("station_id")["date"].nunique()
        keep = complete[complete >= expected].index.tolist()
        if keep:
            done_frames.append(done[done["station_id"].isin(keep)])
            print(f"resuming: {len(keep)} stations already complete")
        st = st[~st["id"].isin(keep)]
    print(f"fetching weather for {len(st)} stations from NASA POWER "
          f"({args.start}..{args.end})")

    t0 = time.time()
    for i, (_, row) in enumerate(st.iterrows(), 1):
        sid = int(row["id"])
        df = _get_point(row["lat"], row["long"], args.start, args.end)
        if not df.empty:
            df["station_id"] = sid
            done_frames.append(df)
        if i % 10 == 0:
            pd.concat(done_frames, ignore_index=True).to_csv(PART, index=False)
            print(f"  {i}/{len(st)} stations ({time.time() - t0:.0f}s)")
        time.sleep(args.sleep)

    RAW.mkdir(parents=True, exist_ok=True)
    allw = pd.concat(done_frames, ignore_index=True)
    allw["source"] = "NASA POWER / MERRA-2"
    allw.to_csv(OUT, index=False)
    PART.unlink(missing_ok=True)
    print(f"{OUT}: {allw.shape[0]:,} rows, {allw.station_id.nunique()} stations, "
          f"{allw.date.min().date()}..{allw.date.max().date()}")
    print(allw.head(3).to_string(index=False))


if __name__ == "__main__":
    main()