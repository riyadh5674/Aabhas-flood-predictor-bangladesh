"""
System Ingestion — Orchestration script.

Runs the full ingestion pipeline:
  1. FFWC station master
  2. FFWC water-level harvest (3-hourly -> daily aggregation with QC)
  3. Rainfall fetch (NASA POWER primary, Open-Meteo fallback)

All steps are resumable via the provenance manifest.

Usage:
    python system/ingestion/run_ingestion.py --all
    python system/ingestion/run_ingestion.py --stations-only
    python system/ingestion/run_ingestion.py --ffwc --start 2020-01-01 --end 2026-10-06 --workers 8
    python system/ingestion/run_ingestion.py --rainfall --source nasa_power --start 2016-12-01 --end 2026-10-06
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.ingestion.ingest_ffwc import main as ffc_main, fetch_stations
from system.ingestion.ingest_rainfall import main as rain_main


def run_stations() -> None:
    print("=" * 60)
    print("STEP 1: FFWC station master")
    print("=" * 60)
    fetch_stations()


def run_ffwc(start: str, end: str, stations: list[int] | None, workers: int, stride: int) -> None:
    print("=" * 60)
    print("STEP 2: FFWC water-level harvest")
    print("=" * 60)
    sys.argv = [
        "ingest_ffwc.py", "harvest",
        "--start", start, "--end", end,
        "--workers", str(workers), "--stride", str(stride),
    ]
    if stations is not None:
        sys.argv += ["--stations"] + [str(s) for s in stations]
    ffc_main()


def run_rainfall(start: str, end: str, source: str, stations: list[int] | None, sleep: float) -> None:
    print("=" * 60)
    print(f"STEP 3: Rainfall fetch ({source})")
    print("=" * 60)
    sys.argv = [
        "ingest_rainfall.py",
        "--start", start, "--end", end,
        "--source", source,
        "--sleep", str(sleep),
    ]
    if stations is not None:
        sys.argv += ["--stations"] + [str(s) for s in stations]
    rain_main()


def main() -> None:
    ap = argparse.ArgumentParser(description="System ingestion orchestration")
    ap.add_argument("--all", action="store_true", help="run all steps")
    ap.add_argument("--stations-only", action="store_true", help="fetch station master only")
    ap.add_argument("--ffwc", action="store_true", help="run FFWC water-level harvest")
    ap.add_argument("--rainfall", action="store_true", help="run rainfall fetch")
    ap.add_argument("--start", default="2016-12-01", help="start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-10-06", help="end date (YYYY-MM-DD)")
    ap.add_argument("--stations", nargs="+", type=int, default=None, help="station IDs (default: all)")
    ap.add_argument("--workers", type=int, default=8, help="FFWC harvest workers")
    ap.add_argument("--stride", type=int, default=35, help="FFWC query stride days")
    ap.add_argument("--source", choices=["nasa_power", "open_meteo"], default="nasa_power", help="rainfall source")
    ap.add_argument("--sleep", type=float, default=0.6, help="sleep between NASA POWER station requests")
    args = ap.parse_args()

    if args.all or args.stations_only:
        run_stations()
    if args.all or args.ffwc:
        run_ffwc(args.start, args.end, args.stations, args.workers, args.stride)
    if args.all or args.rainfall:
        run_rainfall(args.start, args.end, args.source, args.stations, args.sleep)

    if not (args.all or args.stations_only or args.ffwc or args.rainfall):
        ap.print_help()


if __name__ == "__main__":
    main()
