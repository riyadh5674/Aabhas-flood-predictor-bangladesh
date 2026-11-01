"""
System Map — Choropleth builder.

Joins per-station D16 predictions to admin boundaries (ADM1 division, ADM2
district) and produces risk-band choropleth GeoJSON. For each polygon, the
worst (highest-severity) risk band among its stations is used, plus max
probability and station count.

Usage:
    python system/map/run_map.py --date 2024-07-15 --horizon 24h --level adm2
    python system/map/run_map.py --date 2024-07-15 --all-levels
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from system.map.config import (
    ADM1_GEOJSON, ADM2_GEOJSON, STATIONS_FILE,
    MAP_OUT_DIR, CHOROPLETH_ADM1, CHOROPLETH_ADM2,
    BAND_SEVERITY, DIVISION_NORM, DISTRICT_NORM,
)
from system.api.inference import predict_all_stations

# District name normalization (imported from system.map.config.DISTRICT_NORM)


def norm_name(name: str, norm_map: dict) -> str:
    """Normalize a division/district name using the mapping, else return as-is."""
    if not name:
        return name
    return norm_map.get(str(name).strip(), str(name).strip())


def load_station_admin() -> pd.DataFrame:
    """Load station master and return station_id -> normalized division/district."""
    st = pd.read_csv(STATIONS_FILE)
    st = st.rename(columns={"id": "station_id", "long": "lon"})
    st["division_norm"] = st["division"].apply(lambda x: norm_name(x, DIVISION_NORM))
    st["district_norm"] = st["district"].apply(lambda x: norm_name(x, DISTRICT_NORM))
    return st[["station_id", "division_norm", "district_norm", "lat", "lon", "division", "district"]]


def worst_band(bands: pd.Series) -> str:
    """Return the highest-severity risk band in a series."""
    if len(bands) == 0:
        return "unknown"
    sev = bands.map(BAND_SEVERITY).fillna(-1)
    worst_idx = sev.idxmax()
    return bands.loc[worst_idx]


def build_choropleth(
    predictions: pd.DataFrame,
    stations: pd.DataFrame,
    level: str,
    geojson_path: Path,
) -> dict:
    """Build a choropleth GeoJSON for the given admin level.

    Joins predictions to stations, aggregates worst band per polygon, and
    attaches risk info to each GeoJSON feature.
    """
    if level == "adm1":
        admin_col = "division_norm"
        geojson_path = ADM1_GEOJSON
        out_path = CHOROPLETH_ADM1
    elif level == "adm2":
        admin_col = "district_norm"
        geojson_path = ADM2_GEOJSON
        out_path = CHOROPLETH_ADM2
    else:
        raise ValueError(f"Unknown admin level: {level}")

    # Join predictions to station admin info
    df = predictions.merge(stations, on="station_id", how="inner")
    if df.empty:
        raise ValueError(f"No stations matched for level {level}")

    # Aggregate per admin polygon
    agg = (
        df.groupby(admin_col)
        .agg(
            worst_band=("band", worst_band),
            max_probability=("probability", "max"),
            min_probability=("probability", "min"),
            station_count=("station_id", "nunique"),
            avg_probability=("probability", "mean"),
        )
        .reset_index()
    )

    # Load GeoJSON
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    # Build lookup: normalized shapeName -> agg row
    agg_lookup = {}
    for _, r in agg.iterrows():
        agg_lookup[r[admin_col]] = r.to_dict()

    # Attach risk info to each feature
    matched = 0
    unmatched = []
    for feature in gj["features"]:
        shape_name = feature["properties"].get("shapeName", "")
        norm = norm_name(shape_name, {})  # shapeName already canonical
        if norm in agg_lookup:
            info = agg_lookup[norm]
            feature["properties"]["risk_band"] = info["worst_band"]
            feature["properties"]["max_probability"] = round(info["max_probability"], 4)
            feature["properties"]["min_probability"] = round(info["min_probability"], 4)
            feature["properties"]["avg_probability"] = round(info["avg_probability"], 4)
            feature["properties"]["station_count"] = int(info["station_count"])
            feature["properties"]["alert_level"] = info["worst_band"]
            matched += 1
        else:
            unmatched.append(shape_name)
            feature["properties"]["risk_band"] = "unknown"
            feature["properties"]["max_probability"] = None
            feature["properties"]["min_probability"] = None
            feature["properties"]["avg_probability"] = None
            feature["properties"]["station_count"] = 0
            feature["properties"]["alert_level"] = "unknown"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, indent=2)

    print(
        f"Choropleth {level}: {matched} matched / {len(gj['features'])} polygons, "
        f"{len(unmatched)} unmatched"
    )
    if unmatched:
        print(f"  Unmatched: {sorted(set(unmatched))}")

    return gj


def run_map(target_date: date | str, horizons: list[str] | None = None, levels: list[str] | None = None):
    """Run choropleth build for given date/horizons/levels."""
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    levels = levels or ["adm1", "adm2"]
    horizons = horizons or ["24h"]

    stations = load_station_admin()

    for horizon in horizons:
        for level in levels:
            try:
                preds = predict_all_stations(target_date, horizon)
            except FileNotFoundError as exc:
                print(f"WARN: no features for {target_date} {horizon}: {exc}")
                continue

            out = CHOROPLETH_ADM2 if level == "adm2" else CHOROPLETH_ADM1
            print(f"\n=== {target_date} {horizon} {level} ===")
            build_choropleth(preds, stations, level, out)

    return


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build flood risk choropleth GeoJSON")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    ap.add_argument(
        "--horizon",
        choices=["24h", "48h", "72h"],
        default=None,
        help="Single horizon (default: 24h)",
    )
    ap.add_argument(
        "--all-horizons",
        action="store_true",
        help="Evaluate all horizons",
    )
    ap.add_argument(
        "--level",
        choices=["adm1", "adm2"],
        default=None,
        help="Admin level (default: both)",
    )
    ap.add_argument(
        "--all-levels",
        action="store_true",
        help="Evaluate both admin levels",
    )
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date)
    horizons = ["24h", "48h", "72h"] if args.all_horizons else [args.horizon or "24h"]
    levels = ["adm1", "adm2"] if args.all_levels else [args.level or "adm2"]

    run_map(target_date, horizons=horizons, levels=levels)


if __name__ == "__main__":
    main()
