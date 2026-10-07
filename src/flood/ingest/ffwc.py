"""FFWC (Bangladesh Flood Forecasting & Warning Centre) API ingestion.

Endpoints (require an internal key + browser Origin headers):
    GET https://api.ffwc.gov.bd/data_load/stations
        -> station master: id, name, river, basin, district, upazilla,
           lat, long, dangerlevel, riverhighestwaterlevel, ...
    GET https://api.ffwc.gov.bd/data_load/observed-waterlevel-by-station-and-date/{id}/{YYYY-MM-DD}
        -> 3-hourly observed water level records for one station-day.

The API only answers single-day queries, so harvesting history is a
station x day loop.  Output is written as one CSV per station-year under
    data/raw/ffwc/waterlevel/{station_id}/{year}.csv
which makes the harvest resumable (existing dates are skipped).

Usage:
    python -m flood.ingest.ffwc stations
    python -m flood.ingest.ffwc harvest --start 2020-06-01 --end 2020-10-31 --stations 1 4 8 --workers 8
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

API_BASE = "https://api.ffwc.gov.bd/data_load"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "x-ffwc-internal-key": "FFWC-Project-2026-Secure-V1",
    "Referer": "https://ffwc.gov.bd/",
    "Origin": "https://ffwc.gov.bd",
    "Accept": "application/json",
}
RAW_DIR = Path(r"D:\Flood predictor\data\raw\ffwc")
WL_DIR = RAW_DIR / "waterlevel"

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def _get(path: str, retries: int = 4, timeout: int = 30):
    last = None
    for attempt in range(retries):
        try:
            r = _session().get(f"{API_BASE}/{path}", timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} for {path}")
        except Exception as exc:  # network hiccups are expected in a bulk crawl
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise last


def fetch_stations(out_csv: Path | None = None) -> pd.DataFrame:
    """Download the station master (153 stations) and save it as CSV."""
    df = pd.DataFrame(_get("stations"))
    out_csv = out_csv or (RAW_DIR / "ffwc_stations.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def fetch_station_day(station_id: int, day: date) -> pd.DataFrame:
    """Return the 3-hourly readings of one station on one day (empty if none)."""
    rows = _get(f"observed-waterlevel-by-station-and-date/{station_id}/{day:%Y-%m-%d}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["query_date"] = day.isoformat()
    return df


MANIFEST = RAW_DIR / "harvest_manifest.csv"


def _manifest() -> set[tuple[int, str]]:
    if not MANIFEST.exists():
        return set()
    m = pd.read_csv(MANIFEST, dtype={"query_date": str})
    return set(zip(m["station_id"].astype(int), m["query_date"]))


def _record_manifest(pairs: list[tuple[int, str]]) -> None:
    if not pairs:
        return
    new = pd.DataFrame(pairs, columns=["station_id", "query_date"])
    if MANIFEST.exists():
        new = pd.concat([pd.read_csv(MANIFEST, dtype={"query_date": str}), new], ignore_index=True)
    new.drop_duplicates(inplace=True)
    new.to_csv(MANIFEST, index=False)


def _write_station_year(station_id: int, year: int, frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    f = WL_DIR / str(station_id) / f"{year}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.drop_duplicates(subset=["st_id", "wl_date"], keep="last", inplace=True)
    df.sort_values("wl_date", inplace=True, ignore_index=True)
    df.to_csv(f, index=False)


def harvest(
    start: date,
    end: date,
    station_ids: list[int],
    workers: int = 8,
    stride: int = 35,
    verbose: int = 500,
) -> pd.DataFrame:
    """Crawl station history in [start, end].

    One API call answers with a ~40.5-day trailing window of readings, so the
    crawl only needs a query date every `stride` days; overlapping rows are
    deduplicated on (st_id, wl_date) when the station-year CSV is written.
    A manifest of already-fetched (station, query date) pairs makes the run
    resumable.
    """
    WL_DIR.mkdir(parents=True, exist_ok=True)
    done_set = _manifest()

    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=stride)
    if not dates or dates[-1] != end:
        dates.append(end)

    jobs: list[tuple[int, date]] = [
        (sid, day)
        for sid in station_ids
        for day in dates
        if (sid, day.isoformat()) not in done_set
    ]

    print(
        f"harvest {start}..{end} stride={stride}d: {len(station_ids)} stations x "
        f"{len(dates)} query dates = {len(station_ids) * len(dates)} calls, "
        f"{len(jobs)} to fetch ({len(station_ids) * len(dates) - len(jobs)} already done), "
        f"workers={workers}"
    )

    buffers: dict[tuple[int, int], list[pd.DataFrame]] = {}
    fetched: list[tuple[int, str]] = []
    done = 0
    t0 = time.time()

    def work(sid: int, day: date):
        return sid, day, fetch_station_day(sid, day)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, sid, day) for sid, day in jobs]
        for fut in as_completed(futures):
            sid, day, df = fut.result()
            if not df.empty:
                buffers.setdefault((sid, day.year), []).append(df)
            fetched.append((sid, day.isoformat()))
            done += 1
            if verbose and done % verbose == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} fetched ({rate:.1f} req/s)")
            if len(buffers) >= 64:
                _flush(buffers)
                _record_manifest(fetched)
                fetched.clear()

    _flush(buffers)
    _record_manifest(fetched)
    print(f"done: {done} calls in {time.time() - t0:.0f}s")
    return summary()


def _flush(buffers: dict[tuple[int, int], list[pd.DataFrame]]) -> None:
    for (sid, year), frames in buffers.items():
        _write_station_year(sid, year, frames)
    buffers.clear()


def summary() -> pd.DataFrame:
    rows = []
    if WL_DIR.exists():
        for f in sorted(WL_DIR.glob("*/*.csv")):
            rows.append({"station_id": f.parent.name, "year": f.stem, "path": str(f), "bytes": f.stat().st_size})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False) if not df.empty else "no water-level files yet")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="FFWC API ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stations")
    h = sub.add_parser("harvest")
    h.add_argument("--start", required=True)
    h.add_argument("--end", required=True)
    h.add_argument("--stations", nargs="+", type=int, default=None)
    h.add_argument("--workers", type=int, default=8)
    h.add_argument("--stride", type=int, default=35)
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.cmd == "stations":
        df = fetch_stations()
        print(f"{len(df)} stations -> data/raw/ffwc/ffwc_stations.csv")
        print(df[["id", "name", "river", "basin", "district", "dangerlevel", "lat", "long"]].head(10).to_string(index=False))
    elif args.cmd == "harvest":
        stations = args.stations
        if stations is None:
            master = RAW_DIR / "ffwc_stations.csv"
            if not master.exists():
                raise SystemExit("run 'stations' first")
            stations = pd.read_csv(master)["id"].astype(int).tolist()
        harvest(datetime.strptime(args.start, "%Y-%m-%d").date(),
                datetime.strptime(args.end, "%Y-%m-%d").date(),
                stations, workers=args.workers, stride=args.stride)
    else:
        summary()


if __name__ == "__main__":
    main()
