"""
Daily pipeline — automated data acquisition for FloodGuard Bangladesh.

Runs ingestion (FFWC water levels + rainfall) and feature computation for a
target date window, keeping the model-ready feature store current so the API
and UI can serve today's / tomorrow's flood risk.

Default behavior: process yesterday's date (FFWC posts 3-hourly readings with
a short lag, so yesterday is the latest complete day).

Usage:
    python system/ops/daily_pipeline.py              # process yesterday
    python system/ops/daily_pipeline.py --date 2026-10-29   # specific date
    python system/ops/daily_pipeline.py --catchup            # backfill missing dates
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def run(cmd: list[str], label: str) -> bool:
    """Run a command, print output, return success/failure."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {label}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, check=False
        )
        if result.stdout:
            lines = result.stdout.strip().splitlines()
            for line in lines[-50:]:
                print(line)
        if result.returncode != 0:
            print(f"\nFAILED with exit code {result.returncode}")
            if result.stderr:
                err_lines = result.stderr.strip().splitlines()
                for line in err_lines[-30:]:
                    print(f"  {line}")
            return False
        print(f"\nOK: {label}")
        return True
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily flood data pipeline")
    ap.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD). Default: yesterday.",
    )
    ap.add_argument(
        "--catchup",
        action="store_true",
        help="Backfill all dates missing from the feature store.",
    )
    args = ap.parse_args()

    # Determine target date(s)
    if args.catchup:
        import pandas as pd

        prq_dir = ROOT / "system" / "data" / "processed" / "model_ready"
        if prq_dir.exists():
            existing = {
                d.name.split("=")[-1]
                for d in prq_dir.iterdir()
                if d.is_dir() and d.name.startswith("date=")
            }
        else:
            existing = set()
        start = date.today() - timedelta(days=30)
        end = date.today() - timedelta(days=1)
        missing = []
        d = start
        while d <= end:
            if d.isoformat() not in existing:
                missing.append(d)
            d += timedelta(days=1)
        if not missing:
            print("No missing dates to backfill.")
            return 0
        dates = missing
        print(f"Catch-up: {len(dates)} missing dates ({dates[0]} .. {dates[-1]})")
    else:
        target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
        dates = [target]
        print(f"Target date: {target.isoformat()}")

    failures = []
    for d in dates:
        dstr = d.isoformat()
        print(f"\n{'#' * 60}")
        print(f"Processing {dstr}")
        print(f"{'#' * 60}")

        # Step 1: FFWC water level + NASA POWER rainfall ingestion
        ok = run(
            [
                str(VENV_PY),
                "system/ingestion/run_ingestion.py",
                "--all",
                "--start",
                dstr,
                "--end",
                dstr,
                "--workers",
                "8",
            ],
            f"Ingestion FFWC+NASA ({dstr})",
        )
        if not ok:
            failures.append((dstr, "ingestion"))
            continue

        # Step 2: Open-Meteo rainfall for recent dates (NASA POWER has a
        # multi-day publication lag; Open-Meteo ERA5-Land is near-real-time)
        ok = run(
            [
                str(VENV_PY),
                "system/ingestion/ingest_rainfall.py",
                "--source",
                "open_meteo",
                "--start",
                dstr,
                "--end",
                dstr,
            ],
            f"Ingestion Open-Meteo ({dstr})",
        )
        if not ok:
            failures.append((dstr, "open_meteo_ingestion"))
            continue

        # Step 3: Merge Open-Meteo into main rainfall CSV (takes precedence)
        ok = run(
            [
                str(VENV_PY),
                "system/ops/merge_rainfall.py",
                "--date",
                dstr,
            ],
            f"Merge rainfall ({dstr})",
        )
        if not ok:
            failures.append((dstr, "rainfall_merge"))
            continue

        # Step 4: Feature computation
        ok = run(
            [
                str(VENV_PY),
                "-m",
                "system.features.run_build",
                "--start",
                dstr,
                "--end",
                dstr,
            ],
            f"Features ({dstr})",
        )
        if not ok:
            failures.append((dstr, "features"))
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for d, step in failures:
            print(f"  {d}: {step}")
        return 1
    else:
        print("All dates processed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
