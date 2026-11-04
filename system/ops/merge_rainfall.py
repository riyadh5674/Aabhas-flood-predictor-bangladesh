"""
Merge Open-Meteo recent rainfall into the main NASA POWER CSV for a specific date.

Open-Meteo precip takes precedence for the target date (NASA POWER / MERRA-2 has a
publication lag), but the NASA temperature columns are preserved. The previous
implementation deleted the NASA rows and replaced them with precip-only rows, which
silently nulled ``tmax_c``/``tmin_c`` for every merged date.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
NASA_FILE = ROOT / "system" / "data" / "raw" / "rainfall" / "weather_station_daily.csv"
OM_FILE = ROOT / "system" / "data" / "raw" / "openmeteo" / "era5land_station_daily.csv"

VALUE_COLS = ["precip_mm", "tmax_c", "tmin_c", "tmean_c"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge Open-Meteo rainfall into main CSV")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    args = ap.parse_args()

    target = date.fromisoformat(args.date).isoformat()

    nasa = pd.read_csv(NASA_FILE, parse_dates=["date"])
    om = pd.read_csv(OM_FILE, parse_dates=["date"])

    om_target = om.loc[om["date"] == target, ["station_id", "date", "precip_mm"]].copy()
    if om_target.empty:
        print(f"No Open-Meteo data for {target}, skipping merge")
        return 0
    om_target = om_target.rename(columns={"precip_mm": "om_precip_mm"})

    print(f"Open-Meteo rows for {target}: {len(om_target)}")
    print(f"  Precip non-null: {om_target['om_precip_mm'].notna().sum()} / {len(om_target)}")

    nasa = nasa.drop_duplicates(subset=["station_id", "date"], keep="last")

    merged = nasa.merge(om_target, on=["station_id", "date"], how="outer")
    used_om = merged["om_precip_mm"].notna()
    merged.loc[used_om, "precip_mm"] = merged.loc[used_om, "om_precip_mm"]
    merged.loc[used_om, "source"] = "Open-Meteo ERA5-Land (precip) + NASA POWER (temp)"
    merged.loc[~used_om & merged["source"].isna(), "source"] = "Open-Meteo ERA5-Land"
    merged = merged.drop(columns=["om_precip_mm"])
    merged = merged.sort_values(["station_id", "date"]).reset_index(drop=True)

    # NASA POWER publishes temperature with a ~3-day lag. When the target date has
    # no temperature yet, carry each station's last observed value forward so the
    # live feature row keeps defined tmax/tmin instead of nulls. This is an
    # explicit "as-of" fallback; the quality report records the staleness.
    target_rows = merged["date"] == target
    filled = 0
    if target_rows.any():
        history = merged.loc[merged["date"] < target]
        for col in ("tmax_c", "tmin_c", "tmean_c"):
            missing = target_rows & merged[col].isna()
            if not missing.any():
                continue
            last_values = (
                history.dropna(subset=[col]).groupby("station_id")[col].last()
            )
            merged.loc[missing, col] = merged.loc[missing, "station_id"].map(last_values)
            filled = int(missing.sum())
        if filled:
            print(f"  carried temperature forward for {filled} station-column(s)")

    verify = merged[merged["date"] == target]
    print(f"Merged {target}:")
    print(f"  Precip non-null: {verify['precip_mm'].notna().sum()} / {len(verify)}")
    print(f"  Precip sum: {verify['precip_mm'].sum():.2f} mm")
    print(f"  tmax non-null: {verify['tmax_c'].notna().sum()} / {len(verify)}")

    merged.to_csv(NASA_FILE, index=False)
    print(f"Written {NASA_FILE}: {len(merged):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
