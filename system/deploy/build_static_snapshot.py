"""Build or incrementally update a precomputed forecast snapshot for static hosting.

The snapshot stores derived station forecasts only; it never copies the feature
store or the model. Dates are processed in chunks so each horizon's model runs
on a concatenated frame instead of once per date.

Two modes:

* full (default)      - wipe the snapshot and rebuild every available date.
* incremental          - only score dates that are not already in
                         ``dates.json``; everything else is left untouched.

Trend files are rebuilt from the prediction files on disk for a bounded window
(``--trend-days``), which keeps daily runs cheap. The UI only plots the last 30
days, so a 90-day window leaves comfortable headroom.

Run from the repository root with the project's Python environment::

    python system/deploy/build_static_snapshot.py --incremental --trend-days 90
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from system.api.inference import get_model, thresholds_info
from system.config.model_config import FEATURE_STORE, HORIZONS

DEFAULT_OUT = ROOT / "system" / "web" / "public" / "data" / "snapshot"
STATIONS_FILE = ROOT / "system" / "web" / "public" / "data" / "stations.json"
CHUNK_SIZE = 128

NOTICE = (
    "Precomputed model guidance, refreshed daily. "
    "Not real-time and not an official flood warning."
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def local_dates() -> list[str]:
    """Sorted dates present in the local partitioned feature store.

    The snapshot build intentionally reads the local store even when
    ``HF_FEATURE_DATASET`` is configured, so a daily local run never depends on
    a remote download.
    """
    if not FEATURE_STORE.is_dir():
        return []
    return sorted(
        entry.name.split("=", 1)[1]
        for entry in FEATURE_STORE.iterdir()
        if entry.is_dir() and entry.name.startswith("date=")
    )


def load_local(date: str) -> pd.DataFrame:
    path = FEATURE_STORE / f"date={date}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_predictions(out: Path, dates: list[str], model, generated_at: str) -> None:
    for start in range(0, len(dates), CHUNK_SIZE):
        chunk = dates[start : start + CHUNK_SIZE]
        frames = []
        for date in chunk:
            frame = load_local(date).copy()
            frame["_snapshot_date"] = date
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)

        by_date: dict[str, dict[str, list[dict]]] = {date: {} for date in chunk}
        for horizon in HORIZONS:
            predicted = model.predict(combined, horizon)
            # predict() returns rows in the same order as `combined`, but only
            # its own columns; reattach the source date for grouping.
            predicted["_snapshot_date"] = combined["_snapshot_date"].to_numpy()
            predicted["computed_at"] = generated_at
            for record in predicted.to_dict("records"):
                date = str(record.pop("_snapshot_date"))
                record["date"] = date
                by_date.setdefault(date, {}).setdefault(horizon, []).append(record)

        for date in chunk:
            dump(out / "predictions" / f"{date}.json", by_date.get(date, {}))

        done = min(start + CHUNK_SIZE, len(dates))
        if done % 100 < CHUNK_SIZE or done == len(dates):
            print(f"Generated {done:,}/{len(dates):,} dates", flush=True)


def write_trends(out: Path, dates: list[str], stations: list[dict]) -> set[int]:
    """Rebuild per-station trend files from prediction files on disk.

    Returns the set of station ids that had at least one prediction in the
    window, which is used for the health summary.
    """
    buckets: dict[str, dict[int, list[tuple]]] = {
        horizon: {int(station["id"]): [] for station in stations} for horizon in HORIZONS
    }
    active: set[int] = set()
    missing = 0
    for date in dates:
        path = out / "predictions" / f"{date}.json"
        if not path.exists():
            missing += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for horizon in HORIZONS:
            for record in payload.get(horizon, []):
                sid = int(record["station_id"])
                buckets[horizon].setdefault(sid, []).append(
                    (
                        date,
                        record["probability"],
                        record["band"],
                        record["action"],
                        record["threshold"],
                    )
                )
                active.add(sid)

    for horizon, per_station in buckets.items():
        for station_id, points in per_station.items():
            points.sort(key=lambda point: point[0])
            dump(
                out / "trends" / horizon / f"{station_id}.json",
                {
                    "points": [
                        {
                            "date": date,
                            "probability": probability,
                            "band": band,
                            "action": action,
                            "threshold": threshold,
                        }
                        for date, probability, band, action, threshold in points
                    ]
                },
            )
    if missing:
        print(f"Trend rebuild skipped {missing} date(s) with no prediction file")
    return active


def model_meta(model) -> dict:
    return {
        "version": model.version,
        "decision": "D16",
        "seed": model.seed,
        "feature_set": model._artefact.get("feature_set") if model._artefact else None,
        "n_features": model.n_features,
        "frozen_at_utc": model.frozen_at_utc,
        "dataset_sha256_v2": model.dataset_sha256_v2,
        "per_horizon_family": model._artefact.get("per_horizon_family") if model._artefact else None,
        "horizons": list(HORIZONS),
        "loaded_at": model._loaded_at.isoformat() if model._loaded_at else None,
    }


def latest_station_count(out: Path, latest_date: str | None) -> int:
    """Number of stations actually forecast on the newest snapshot date."""
    if not latest_date:
        return 0
    path = out / "predictions" / f"{latest_date}.json"
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return len(payload.get(HORIZONS[0], []))


def source_freshness() -> tuple[dict, bool]:
    """Latest non-future per-source status and whether a critical source is stale."""
    status_dir = ROOT / "system" / "data" / "processed" / "source_status"
    if not status_dir.exists():
        return {}, False
    today = datetime.now(timezone.utc).date()
    for path in sorted(status_dir.glob("*.json"), reverse=True):
        try:
            if date_cls.fromisoformat(path.stem) > today:
                continue
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sources: dict = {}
        degraded = False
        for name, info in payload.get("sources", {}).items():
            sources[name] = {
                "status": info.get("status"),
                "last_date": info.get("last_date"),
                "lag_days": info.get("lag_days"),
                "max_lag_days": info.get("max_lag_days"),
                "critical": bool(info.get("critical", False)),
            }
            if info.get("critical") and info.get("status") != "ok":
                degraded = True
        return sources, degraded
    return {}, False


def write_metadata(
    out: Path,
    dates: list[str],
    model,
    generated_at: str,
    station_count: int,
) -> None:
    latest = dates[-1] if dates else None
    latest_count = latest_station_count(out, latest)
    sources, degraded = source_freshness()
    dump(
        out / "health.json",
        {
            "status": "snapshot",
            "model_loaded": True,
            "model_version": model.version,
            "feature_store_dates": len(dates),
            "latest_feature_date": latest,
            "timestamp": generated_at,
            "generated_at": generated_at,
            # Stations with a prediction on the latest date. `trend_station_count`
            # is the wider union over the trend window and can be larger when a
            # station stops reporting.
            "station_count": latest_count,
            "trend_station_count": station_count,
            "station_count_scope": "latest_feature_date",
            "sources": sources,
            "degraded": degraded,
            "notice": NOTICE,
        },
    )
    dump(out / "dates.json", {"dates": dates})
    dump(out / "thresholds.json", thresholds_info())
    dump(out / "model.json", model_meta(model))


def existing_dates(out: Path) -> set[str]:
    path = out / "dates.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("dates", []))
    except Exception:
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or incrementally update the static snapshot")
    parser.add_argument("--incremental", action="store_true", help="Only score dates missing from dates.json.")
    parser.add_argument(
        "--trend-days",
        type=int,
        default=0,
        help="Keep only the last N dates in trend files (0 = full history).",
    )
    parser.add_argument(
        "--rebuild-trends",
        action="store_true",
        help="Rewrite trend files even when no new dates were scored.",
    )
    parser.add_argument(
        "--rebuild-from",
        help="Re-score every date on/after this date (YYYY-MM-DD), even if already published.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Snapshot output directory.")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    dates = local_dates()
    if not dates:
        print("No local feature partitions found; refusing to publish an empty snapshot.")
        return 1

    stations = json.loads(STATIONS_FILE.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).isoformat()

    if args.incremental:
        known = existing_dates(out)
        new_dates = [date for date in dates if date not in known]
        out.mkdir(parents=True, exist_ok=True)
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        new_dates = list(dates)

    if args.rebuild_from:
        forced = [date for date in dates if date >= args.rebuild_from]
        new_dates = sorted(set(new_dates) | set(forced))

    if new_dates:
        model = get_model()
        build_predictions(out, new_dates, model, generated_at)
    elif not args.rebuild_trends:
        print("No new dates; snapshot is already current.")
        return 0
    else:
        model = get_model()

    if args.trend_days > 0:
        trend_dates = dates[-args.trend_days :]
    else:
        trend_dates = dates
    active = write_trends(out, trend_dates, stations)
    write_metadata(out, dates, model, generated_at, station_count=len(active))
    latest_count = latest_station_count(out, dates[-1])

    print(
        f"Snapshot {'updated' if args.incremental else 'complete'}: "
        f"{len(dates):,} dates, {latest_count:,} stations on {dates[-1]} "
        f"({len(active):,} in the trend window), D16."
    )
    if new_dates:
        print(f"  scored {len(new_dates):,} new date(s): {new_dates[0]} .. {new_dates[-1]}")
    print(f"  trends cover {len(trend_dates):,} date(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
