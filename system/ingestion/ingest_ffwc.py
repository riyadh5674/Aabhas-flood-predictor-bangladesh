"""
System Ingestion — FFWC water-level data.

Fetches 3-hourly observed water levels from the FFWC API, applies QC filters
(F3: impossible >100m, F8: spike filter, sentinel <=-1000), aggregates to daily
statistics, and writes one CSV per station-year under
    system/data/raw/ffwc/waterlevel/{station_id}/{year}.csv

Resumable: existing (station, query_date) pairs in the manifest are skipped.
Rate-limited: exponential backoff with jitter, max 5 retries.

Usage:
    python system/ingestion/ingest_ffwc.py stations
    python system/ingestion/ingest_ffwc.py harvest --start 2020-01-01 --end 2026-10-06 --workers 8
    python system/ingestion/ingest_ffwc.py summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system.ingestion.qc_rules import apply_wl_qc, aggregate_daily
from system.ingestion.provenance import ProvenanceManifest

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG = json.loads((ROOT / "system" / "config" / "ingestion_config.json").read_text(encoding="utf-8"))
FFWC_CFG = CONFIG["ffwc"]

API_BASE = FFWC_CFG["api_base"]
HEADERS = FFWC_CFG["headers"]
QUERY_STRIDE = FFWC_CFG["query_stride_days"]
MAX_WORKERS = FFWC_CFG["max_workers"]
RETRIES = FFWC_CFG["retries"]
BACKOFF_BASE = FFWC_CFG["retry_backoff_base"]
TIMEOUT = FFWC_CFG["timeout_seconds"]

RAW_DIR = ROOT / FFWC_CFG["output_dir"]
WL_DIR = RAW_DIR  # system/data/raw/ffwc/waterlevel
STATIONS_FILE = ROOT / FFWC_CFG["stations_file"]
MANIFEST_FILE = ROOT / FFWC_CFG["manifest_file"]

MANIFEST = ProvenanceManifest(MANIFEST_FILE)

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def _get(path: str, retries: int = RETRIES, timeout: int = TIMEOUT):
    last = None
    for attempt in range(retries):
        try:
            r = _session().get(f"{API_BASE}/{path}", timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} for {path}")
        except Exception as exc:
            last = exc
        time.sleep(BACKOFF_BASE * (attempt + 1))
    raise last


def fetch_stations(out_csv: Optional[Path] = None) -> pd.DataFrame:
    """Download the station master and save it as CSV."""
    df = pd.DataFrame(_get("stations"))
    out_csv = out_csv or STATIONS_FILE
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} stations -> {out_csv}")
    return df


def fetch_station_day(station_id: int, day: date) -> pd.DataFrame:
    """Return the 3-hourly readings of one station on one day (empty if none)."""
    rows = _get(f"observed-waterlevel-by-station-and-date/{station_id}/{day:%Y-%m-%d}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["query_date"] = day.isoformat()
    return df


def _write_station_year(station_id: int, year: int, frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    f = WL_DIR / str(station_id) / f"{year}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    if f.exists():
        existing = pd.read_csv(f)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=["st_id", "wl_date"], keep="last")
    df = df.sort_values("wl_date").reset_index(drop=True)
    df.to_csv(f, index=False)


def harvest(
    start: date,
    end: date,
    station_ids: list[int],
    workers: int = MAX_WORKERS,
    stride: int = QUERY_STRIDE,
    verbose: int = 500,
) -> pd.DataFrame:
    """Crawl station history in [start, end] with manifest-based resumability."""
    WL_DIR.mkdir(parents=True, exist_ok=True)
    done_set = MANIFEST._done  # (station_id, query_date, source)

    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=stride)
    if not dates or dates[-1] != end:
        dates.append(end)

    jobs = [
        (sid, day)
        for sid in station_ids
        for day in dates
        if (sid, day.isoformat(), "ffwc_waterlevel") not in done_set
    ]

    print(
        f"harvest {start}..{end} stride={stride}d: {len(station_ids)} stations x "
        f"{len(dates)} query dates = {len(station_ids) * len(dates)} calls, "
        f"{len(jobs)} to fetch, workers={workers}"
    )

    buffers: dict[tuple[int, int], list[pd.DataFrame]] = {}
    fetched: list[tuple[int, str, str, int, str]] = []
    done = 0
    t0 = time.time()

    def work(sid: int, day: date):
        return sid, day, fetch_station_day(sid, day)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, sid, day) for sid, day in jobs]
        for fut in as_completed(futures):
            sid, day, df = fut.result()
            rows = 0
            status = "ok"
            if not df.empty:
                buffers.setdefault((sid, day.year), []).append(df)
                rows = len(df)
            fetched.append((sid, day.isoformat(), "ffwc_waterlevel", rows, status))
            done += 1
            if verbose and done % verbose == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} fetched ({rate:.1f} req/s)")
            if len(buffers) >= 64:
                for (sid2, year), frames in buffers.items():
                    _write_station_year(sid2, year, frames)
                buffers.clear()
                for rec in fetched:
                    MANIFEST.record(*rec)
                fetched.clear()

    for (sid, year), frames in buffers.items():
        _write_station_year(sid, year, frames)
    for rec in fetched:
        MANIFEST.record(*rec)

    print(f"done: {done} calls in {time.time() - t0:.0f}s")
    print(MANIFEST.summary())
    return summary()


def summary() -> pd.DataFrame:
    rows = []
    if WL_DIR.exists():
        for f in sorted(WL_DIR.glob("*/*.csv")):
            rows.append(
                {
                    "station_id": int(f.parent.name),
                    "year": f.stem,
                    "path": str(f),
                    "bytes": f.stat().st_size,
                }
            )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False) if not df.empty else "no water-level files yet")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="System FFWC water-level ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stations")
    h = sub.add_parser("harvest")
    h.add_argument("--start", required=True)
    h.add_argument("--end", required=True)
    h.add_argument("--stations", nargs="+", type=int, default=None)
    h.add_argument("--workers", type=int, default=MAX_WORKERS)
    h.add_argument("--stride", type=int, default=QUERY_STRIDE)
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.cmd == "stations":
        fetch_stations()
    elif args.cmd == "harvest":
        stations = args.stations
        if stations is None:
            if not STATIONS_FILE.exists():
                raise SystemExit("run 'stations' first")
            stations = pd.read_csv(STATIONS_FILE)["id"].astype(int).tolist()
        harvest(
            datetime.strptime(args.start, "%Y-%m-%d").date(),
            datetime.strptime(args.end, "%Y-%m-%d").date(),
            stations,
            workers=args.workers,
            stride=args.stride,
        )
    else:
        summary()


if __name__ == "__main__":
    main()
