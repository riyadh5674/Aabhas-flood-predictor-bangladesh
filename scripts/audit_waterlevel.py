"""Phase 0 audit of harvested FFWC water-level CSVs: coverage + label base rate."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
WL = Path(r"D:\Flood predictor\data\raw\ffwc\waterlevel")
ST = Path(r"D:\Flood predictor\data\raw\ffwc\ffwc_stations.csv")

files = sorted(WL.glob("*/*.csv"))
print(f"{len(files)} station-year files")

rows, recs = 0, []
for f in files:
    d = pd.read_csv(f, usecols=["st_id", "wl_date", "waterlevel", "dangerlevel", "query_date"])
    dl = pd.to_numeric(d["dangerlevel"], errors="coerce")
    wl = pd.to_numeric(d["waterlevel"], errors="coerce")
    ok = wl.notna() & dl.notna()
    exceed = (wl[ok] >= dl[ok]).mean() if ok.any() else float("nan")
    d["ts"] = pd.to_datetime(d["wl_date"], errors="coerce", utc=True)
    dates = pd.to_datetime(d["query_date"]).dt.date
    recs.append(
        dict(
            station=int(d["st_id"].iloc[0]),
            year=int(f.stem),
            rows=len(d),
            readings=d["ts"].nunique(),
            days=dates.nunique(),
            first=d["ts"].min(),
            last=d["ts"].max(),
            pct_with_danger=100 * ok.mean(),
            exceed_rate=exceed,
        )
    )
    rows += len(d)

r = pd.DataFrame(recs)
print(f"total rows: {rows:,}")
print("\n--- readings per station-year (data density)")
print(r.groupby("year")[["readings", "days"]].sum().to_string())
print("\n--- stations with usable record (>=500 days of readings)")
per = r.groupby("station").agg(years=("year", "nunique"), readings=("readings", "sum"),
                               first=("first", "min"), last=("last", "max"),
                               exceed=("exceed_rate", "mean"), dl=("pct_with_danger", "mean"))
per = per.sort_values("readings", ascending=False)
print(per.head(20).to_string())
print(f"\nstations total: {len(per)}")
print(f"stations with >=1000 readings: {(per.readings >= 1000).sum()}")
print(f"stations with >=5 years: {(per.years >= 5).sum()}")
print(f"mean exceedance (danger-level) rate: {100*per.exceed.mean():.1f}%  "
      f"(median {100*per.exceed.median():.1f}%)")
print(f"stations with a danger level recorded: {(per.dl > 95).sum()}")
r.to_csv(r"D:\Flood predictor\data\interim\ffwc_waterlevel_audit.csv", index=False)
per.to_csv(r"D:\Flood predictor\data\interim\ffwc_station_audit.csv")
print("\nwrote data/interim/ffwc_waterlevel_audit.csv, ffwc_station_audit.csv")
