"""Refresh every external source the live model depends on, then verify freshness.

This is the single entry point used by the daily pipeline. Each source refreshes
incrementally (only the missing tail), and the result is written to
``system/data/processed/source_status/<date>.json`` so the feature build, the
publisher and monitoring can see whether a run is safe to publish.

Usage:
    python system/ingestion/refresh_sources.py                 # catch up to yesterday
    python system/ingestion/refresh_sources.py --end 2026-11-03
    python system/ingestion/refresh_sources.py --only glofas,power_soil
    python system/ingestion/refresh_sources.py --allow-stale    # warn, do not fail
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
STATUS_DIR = ROOT / "system" / "data" / "processed" / "source_status"

# Each source: the fetcher, how it expects the end date, the manifest that proves
# freshness, and the maximum acceptable lag. `valid_end` is used for sources that
# publish with a lag and therefore have a separate "last real observation" date.
SOURCES = [
    {
        "name": "glofas",
        "script": "scripts/fetch_glofas.py",
        "date_format": "%Y-%m-%d",
        "manifest": "data/raw/glofas/glofas_manifest.json",
        "manifest_date_key": "end",
        "max_lag_days": 2,
        "critical": True,
        "extra_args": [],
    },
    {
        "name": "power_soil",
        "script": "scripts/fetch_power_soil.py",
        "date_format": "%Y%m%d",
        "manifest": "data/raw/power_soil/power_soil_manifest.json",
        "manifest_date_key": "valid_end",
        "max_lag_days": 7,
        "critical": True,
        "extra_args": ["--workers", "3"],
    },
    {
        "name": "era5",
        "script": "scripts/fetch_openmeteo_batch.py",
        "date_format": "%Y-%m-%d",
        "manifest": "data/raw/openmeteo/era5land_manifest.json",
        "manifest_date_key": "valid_end",
        "max_lag_days": 7,
        "critical": True,
        "extra_args": ["--workers", "3", "--batch", "20"],
    },
]


def _read_manifest_date(path: Path, key: str) -> date | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get(key)
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def classify_freshness(last: date | None, end: date, max_lag_days: int) -> tuple[str, int | None]:
    """Return ``(status, lag_days)`` for a source's last available date.

    ``status`` is ``"ok"`` within the allowed lag, ``"stale"`` beyond it, and
    ``"failed"`` when no usable date could be read.
    """
    if last is None:
        return "failed", None
    lag = max((end - last).days, 0)
    return ("ok" if lag <= max_lag_days else "stale"), lag


def refresh_source(source: dict, end: date, lookback_days: int) -> dict:
    script = ROOT / source["script"]
    if not script.exists():
        return {"status": "missing", "detail": f"{script} not found"}

    cmd = [
        sys.executable,
        str(script),
        "--end",
        end.strftime(source["date_format"]),
        "--lookback-days",
        str(lookback_days),
    ]
    cmd.extend(source.get("extra_args", []))

    print(f"\n--- refreshing {source['name']} ---")
    proc = subprocess.run(cmd, cwd=ROOT)
    manifest = ROOT / source["manifest"]
    last = _read_manifest_date(manifest, source["manifest_date_key"])

    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "last_date": last.isoformat() if last else None,
        }
    if last is None:
        return {"status": "failed", "detail": f"no usable date in {manifest}"}

    status, lag = classify_freshness(last, end, source["max_lag_days"])
    return {
        "status": status,
        "last_date": last.isoformat(),
        "lag_days": lag,
        "max_lag_days": source["max_lag_days"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh external data sources")
    ap.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Last date to refresh (YYYY-MM-DD). Default: yesterday.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Trailing days re-fetched to absorb revisions (default 7).",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated subset, e.g. glofas,power_soil (default: all).",
    )
    ap.add_argument(
        "--allow-stale",
        action="store_true",
        help="Exit 0 even when a critical source is stale.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    args = ap.parse_args(argv)

    end = date.fromisoformat(args.end)
    selected = [s for s in SOURCES if not args.only or s["name"] in args.only.split(",")]

    if args.dry_run:
        for source in selected:
            print(f"would refresh {source['name']} -> {source['script']} (end {end})")
        return 0

    results: dict[str, dict] = {}
    for source in selected:
        results[source["name"]] = refresh_source(source, end, args.lookback_days)
        results[source["name"]]["critical"] = source["critical"]

    ok = True
    print("\n=== source freshness ===")
    for name, result in results.items():
        print(
            f"  {name:12s} {result['status']:7s} last={result.get('last_date')} "
            f"lag={result.get('lag_days')}d max={result.get('max_lag_days', '-')}d"
        )
        if result["status"] != "ok" and result.get("critical"):
            ok = False

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_end": end.isoformat(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sources": results,
        "ok": ok,
    }
    status_file = STATUS_DIR / f"{end.isoformat()}.json"
    status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {status_file}")

    if not ok and not args.allow_stale:
        print("ERROR: a critical source is stale; downstream features will be degraded.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
