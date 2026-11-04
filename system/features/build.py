"""
System Features — Build orchestration.

End-to-end feature computation pipeline:

  1. Load QC'd water levels (system/data/raw/ffwc/)
  2. Load rainfall (system/data/raw/rainfall/)
  3. Compute daily aggregates + hydrology features
  4. Join static spatial features (parent repo, read-only)
  5. Join soil moisture + GloFAS (parent repo, read-only)
  6. Compute neighbour + upstream features
  7. Compute derived + interaction features
  8. Merge on (station_id, date), select exactly FULL_V2 columns
  9. Write model-ready parquet partitioned by date

Usage:
    python system/features/run_build.py --start 2017-01-01 --end 2026-10-06
    python system/features/run_build.py --dates 2024-07-01 2024-07-02 2024-07-03
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.features.config import (
    ROOT, SYSTEM, RAW_WL_DIR, RAW_RAIN, STATIONS_FILE,
    PROC_DIR, MODEL_READY_DIR, FEATURES_PARQUET,
    FULL_V2, MIN_READINGS_PER_DAY,
)
from system.features.hydrology import (
    load_and_qc_waterlevels, build_daily, add_hydrology_features,
)
from system.features.rainfall import load_rainfall, add_rainfall_features, add_temporal_features
from system.features.static import add_static_spatial, load_soil_moisture, add_soil_moisture_features, load_glofas
from system.features.validate import validate_features
from system.features.spatial import add_neighbourhood, add_upstream
from system.features.derived import add_derived_features, add_interactions


def load_station_master() -> pd.DataFrame:
    """Load FFWC station master (parent repo, read-only).

    Returns columns: station_id, name, river, basin, division, district,
    upazilla, lat, lon, danger_level
    """
    st = pd.read_csv(STATIONS_FILE)
    st = st.rename(columns={"id": "station_id", "long": "lon"})
    st["division"] = st["division"].str.strip().replace({"Chittagong": "Chattogram"})
    master_dl = st[["station_id", "dangerlevel"]].rename(
        columns={"dangerlevel": "danger_level"})
    master_dl["danger_level"] = pd.to_numeric(master_dl["danger_level"], errors="coerce")
    st = st[["station_id", "name", "river", "basin", "division", "district",
             "upazilla", "lat", "lon"]]
    return st, master_dl


def build_feature_matrix(start: date, end: date,
                         output_dates: set[date] | None = None,
                         allow_degraded: bool = False) -> pd.DataFrame:
    """Build the full FULL_V2 feature matrix for [start, end].

    Features are computed over the *entire* available history, then squeezed to
    the requested output dates. Lag/rolling/recursive features must see the days
    before ``start``; computing them on a pre-filtered frame leaves recent
    partitions with all-null lags (the daily pipeline hit exactly that).

    ``output_dates`` optionally restricts the written rows further, which lets a
    catch-up run recompute a few broken partitions without rewriting neighbours.

    Returns DataFrame with columns: station_id, date, + all 110 FULL_V2 features.
    """
    print("=" * 60)
    print("STEP 1: Load station master")
    print("=" * 60)
    st, master_dl = load_station_master()
    print(f"{len(st)} stations loaded")

    print()
    print("=" * 60)
    print("STEP 2: Load QC'd water levels + build daily aggregates")
    print("=" * 60)
    raw = load_and_qc_waterlevels()
    daily = build_daily(raw, master_dl)

    print()
    print("=" * 60)
    print("STEP 3: Add hydrology features")
    print("=" * 60)
    daily = add_hydrology_features(daily)

    print()
    print("=" * 60)
    print("STEP 4: Load rainfall + add rainfall features")
    print("=" * 60)
    weather = load_rainfall()
    daily = add_rainfall_features(daily, weather)
    daily = add_temporal_features(daily)

    print()
    print("=" * 60)
    print("STEP 5: Join station static info + spatial features")
    print("=" * 60)
    df = st.merge(daily, on="station_id", how="inner")
    df = add_static_spatial(df)

    print()
    print("=" * 60)
    print("STEP 6: Add neighbourhood features (nb3/nb5)")
    print("=" * 60)
    df = add_neighbourhood(df)

    print()
    print("=" * 60)
    print("STEP 7: Add upstream features")
    print("=" * 60)
    df = add_upstream(df)

    print()
    print("=" * 60)
    print("STEP 8: Join soil moisture features")
    print("=" * 60)
    sm = load_soil_moisture()
    df = add_soil_moisture_features(df, sm)

    print()
    print("=" * 60)
    print("STEP 9: Add GloFAS upstream discharge features")
    print("=" * 60)
    gf = load_glofas(df)
    if not gf.empty:
        df = df.merge(gf, on=["station_id", "date"], how="left")

    print()
    print("=" * 60)
    print("STEP 10: Add derived features (wl_last long lags, api_09/07)")
    print("=" * 60)
    dv = add_derived_features(df)
    # drop long lags from df to avoid _x/_y suffix collision on merge;
    # add_derived_features re-provides them cleanly
    for c in ("wl_last_lag10", "wl_last_lag14", "wl_last_lag21", "wl_last_lag30"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(dv, on=["station_id", "date"], how="left")

    print()
    print("=" * 60)
    print("STEP 11: Add interaction features (ix_*)")
    print("=" * 60)
    df = add_interactions(df)

    # Keep the full history for feature computation, but only write the dates
    # that were actually requested. ``output_dates`` is used by catch-up runs.
    in_range = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    if output_dates is not None:
        in_range &= df["date"].dt.date.isin(output_dates)
    df = df[in_range].copy()

    print()
    print("=" * 60)
    print("STEP 12: Select FULL_V2 columns + write output")
    print("=" * 60)

    # Verify all FULL_V2 columns present
    missing = [c for c in FULL_V2 if c not in df.columns]
    if missing:
        raise KeyError(f"Feature build: declared columns never produced: {missing}")

    # Select exactly FULL_V2 + keys
    out_cols = ["station_id", "date"] + FULL_V2
    df = df[out_cols].sort_values(["station_id", "date"]).reset_index(drop=True)

    print(f"Final matrix: {len(df):,} rows × {len(out_cols)} cols")
    print(f"Stations: {df.station_id.nunique()}")
    print(f"Date range: {df.date.min().date()}..{df.date.max().date()}")

    # Null rate report
    null_rates = df[FULL_V2].isna().mean().sort_values(ascending=False)
    print("\nTop null rates:")
    for c, r in null_rates.head(10).items():
        print(f"  {c}: {r:.1%}")

    # Quality gate: refuse to hand a degraded matrix to the model unless the
    # caller explicitly accepts it. Writes a per-date report either way.
    validate_features(df, allow_degraded=allow_degraded)

    return df


def write_model_ready(df: pd.DataFrame, partition_by_date: bool = True) -> None:
    """Write feature matrix as parquet.

    If partition_by_date: one parquet per date under MODEL_READY_DIR/date=YYYY-MM-DD/
    Else: single parquet at FEATURES_PARQUET
    """
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    if partition_by_date:
        for d, group in df.groupby(df["date"].dt.date):
            part_dir = MODEL_READY_DIR / f"date={d.isoformat()}"
            part_dir.mkdir(parents=True, exist_ok=True)
            group[["station_id", "date"] + FULL_V2].to_parquet(
                part_dir / "data.parquet", index=False)
        print(f"Wrote partitioned parquet to {MODEL_READY_DIR} "
              f"({df['date'].dt.date.nunique()} dates)")
    else:
        df.to_parquet(FEATURES_PARQUET, index=False)
        print(f"Wrote single parquet to {FEATURES_PARQUET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="System feature computation")
    ap.add_argument("--start", default="2017-01-01", help="start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-10-06", help="end date (YYYY-MM-DD)")
    ap.add_argument("--dates", nargs="+", default=None,
                    help="specific dates (YYYY-MM-DD) instead of range")
    ap.add_argument("--no-partition", action="store_true",
                    help="write single parquet instead of per-date partition")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="write output even when the feature quality gate fails")
    args = ap.parse_args()

    if args.dates:
        dates = [date.fromisoformat(d) for d in args.dates]
        start, end = min(dates), max(dates)
        output_dates = set(dates)
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        output_dates = None

    df = build_feature_matrix(
        start, end, output_dates=output_dates, allow_degraded=args.allow_degraded
    )
    write_model_ready(df, partition_by_date=not args.no_partition)


if __name__ == "__main__":
    main()
