"""Build an all-real, precomputed forecast snapshot for Aabhas static hosting.

Run from the repository root with the project's Python environment. This stores
derived station forecasts only; it never copies the feature store or model.

Dates are processed in chunks so each horizon's model runs on a concatenated
frame instead of once per date. Trend points are accumulated as compact tuples
and only expanded to dicts when written.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from system.api.inference import list_available_dates, load_features_for_date, get_model, thresholds_info
from system.config.model_config import HORIZONS

CHUNK_SIZE = 128


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def main() -> None:
    out = ROOT / "system" / "web" / "public" / "data" / "snapshot"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dates = list_available_dates()
    if not dates:
        raise SystemExit("No feature store dates found; refusing to publish an empty snapshot.")
    model = get_model()
    generated_at = datetime.now(timezone.utc).isoformat()
    station_ids: set[int] = set()

    # horizon -> station_id -> list[(date, probability, band, action, threshold)]
    trends: dict[str, dict[int, list[tuple]]] = {h: {} for h in HORIZONS}

    for start in range(0, len(dates), CHUNK_SIZE):
        chunk = dates[start : start + CHUNK_SIZE]
        frames = []
        for date in chunk:
            frame = load_features_for_date(date)
            frame = frame.copy()
            frame["_snapshot_date"] = date
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        station_ids.update(int(sid) for sid in combined["station_id"].dropna().unique())

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
                bucket = by_date.setdefault(date, {})
                bucket.setdefault(horizon, []).append(record)
                sid = int(record["station_id"])
                trends[horizon].setdefault(sid, []).append(
                    (
                        date,
                        record["probability"],
                        record["band"],
                        record["action"],
                        record["threshold"],
                    )
                )

        for date in chunk:
            dump(out / "predictions" / f"{date}.json", by_date[date])

        done = min(start + CHUNK_SIZE, len(dates))
        if done % 100 < CHUNK_SIZE or done == len(dates):
            print(f"Generated {done:,}/{len(dates):,} dates", flush=True)

    for horizon, stations in trends.items():
        station_file = ROOT / "system" / "web" / "public" / "data" / "stations.json"
        master = json.loads(station_file.read_text(encoding="utf-8"))
        for station in master:
            stations.setdefault(int(station["id"]), [])
        for station_id, points in stations.items():
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

    model_meta = {
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
    dump(out / "health.json", {
        "status": "snapshot", "model_loaded": True, "model_version": model.version,
        "feature_store_dates": len(dates), "latest_feature_date": dates[-1], "timestamp": generated_at,
        "generated_at": generated_at, "station_count": len(station_ids),
        "notice": "Precomputed model guidance, refreshed manually. Not real-time and not an official flood warning.",
    })
    dump(out / "dates.json", {"dates": dates})
    dump(out / "thresholds.json", thresholds_info())
    dump(out / "model.json", model_meta)
    print(f"Snapshot complete: {len(dates):,} dates, {len(station_ids):,} stations, D16 through {dates[-1]}.")


if __name__ == "__main__":
    main()
