"""
Daily pipeline â€” automated data acquisition for FloodGuard Bangladesh.

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
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

# Resolve paths from this file so the pipeline runs unchanged on Windows,
# Linux, GitHub Actions, and containers. AABHAS_ROOT is only needed when the
# repository is checked out somewhere other than the parent of `system/`.
ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
PYTHON = Path(os.getenv("AABHAS_PYTHON", sys.executable)).resolve()


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
    ap.add_argument(
        "--skip-source-refresh",
        action="store_true",
        help="Do not refresh GloFAS / soil before building features.",
    )
    ap.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Allow feature build to succeed even when quality gate fails (explicit opt-in).",
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
    degraded = False

# Step 0: refresh every external source the model depends on. This runs once
    # for the whole window, not per date: GloFAS and soil are daily snapshots, so
    # refreshing to the newest target date covers every date in a catch-up run.
    if not args.skip_source_refresh:
        refresh_end = max(dates).isoformat()
        cmd = [
            str(PYTHON),
            "system/ingestion/refresh_sources.py",
            "--end",
            refresh_end,
            "--lookback-days",
            "7",
        ]
        if args.allow_degraded:
            cmd.append("--allow-stale")
        ok = run(cmd, f"Refresh external sources (to {refresh_end})")
        if not ok:
            if args.allow_degraded:
                degraded = True
                print("WARNING: continuing with a stale source (degraded run).")
            else:
                print("ERROR: a critical source is stale; aborting before feature build.")
                return 1

    for d in dates:
        dstr = d.isoformat()
        print(f"\n{'#' * 60}")
        print(f"Processing {dstr}")
        print(f"{'#' * 60}")

        # Step 1: FFWC water level + NASA POWER rainfall ingestion
        ok = run(
            [
                str(PYTHON),
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
                str(PYTHON),
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
                str(PYTHON),
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
        feature_cmd = [
            str(PYTHON),
            "-m",
            "system.features.run_build",
            "--start",
            dstr,
            "--end",
            dstr,
        ]
        if args.allow_degraded:
            feature_cmd.append("--allow-degraded")
        ok = run(feature_cmd, f"Features ({dstr})")
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
    elif args.allow_degraded:
        print("All dates processed with --allow-degraded (quality gate bypassed).")
        return 0
    else:
        print("All dates processed successfully (quality gate passed).")
        return 0


if __name__ == "__main__":
    sys.exit(main())

