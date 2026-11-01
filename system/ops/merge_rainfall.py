"""
Merge Open-Meteo recent rainfall into the main NASA POWER CSV for a specific date.
Open-Meteo takes precedence (NASA POWER MERRA-2 has a publication lag).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Flood predictor")
NASA_FILE = ROOT / "system" / "data" / "raw" / "rainfall" / "weather_station_daily.csv"
OM_FILE = ROOT / "system" / "data" / "raw" / "openmeteo" / "era5land_station_daily.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge Open-Meteo rainfall into main CSV")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    args = ap.parse_args()

    target = date.fromisoformat(args.date).isoformat()

    nasa = pd.read_csv(NASA_FILE, parse_dates=["date"])
    om = pd.read_csv(OM_FILE, parse_dates=["date"])

    # Filter Open-Meteo to the target date
    om_target = om[om["date"] == target].copy()
    if om_target.empty:
        print(f"No Open-Meteo data for {target}, skipping merge")
        return 0

    print(f"Open-Meteo rows for {target}: {len(om_target)}")
    print(f"  Precip non-null: {om_target['precip_mm'].notna().sum()} / {len(om_target)}")

    # Remove NASA POWER rows for the target date
    mask = nasa["date"] == target
    removed = int(mask.sum())
    nasa_clean = nasa[~mask]
    print(f"Removed {removed} NASA POWER rows for {target}")

    # Standardize Open-Meteo columns
    om_clean = om_target[["station_id", "date", "precip_mm"]].copy()
    om_clean["source"] = "Open-Meteo ERA5-Land"

    # Combine
    merged = pd.concat([nasa_clean, om_clean], ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "date"], keep="last")
    merged = merged.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Verify
    verify = merged[merged["date"] == target]
    print(f"Merged precip for {target}:")
    print(f"  Non-null: {verify['precip_mm'].notna().sum()} / {len(verify)}")
    print(f"  Sum: {verify['precip_mm'].sum():.2f} mm")

    merged.to_csv(NASA_FILE, index=False)
    print(f"Written {NASA_FILE}: {len(merged):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
