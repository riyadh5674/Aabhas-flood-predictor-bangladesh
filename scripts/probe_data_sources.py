"""Connectivity probe for the external data sources the advanced experiments need.

Run:  .venv\\Scripts\\python.exe scripts/probe_data_sources.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "domain_adaptation" / "data_source_probe.json"

URLS = {
    # --- Experiment 2: SAR inundation -------------------------------------
    "ASF DAAC search (Sentinel-1)":
        "https://api.daac.asf.alaska.edu/services/search/granules?dataset=Sentinel-1&maxResults=1",
    "Copernicus DataSpace (S1 catalogue)":
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$top=1",
    "Copernicus DataSpace (S1 download)":
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products",
    # --- Experiment 3: NWP ensembles --------------------------------------
    "NOAA NOMADS GFS (single run)":
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.0p25deg/atmos/2025102100/gfs.t00z.pgrb2.0-120.0-90.0",
    "NOAA NOMADS GFS ensemble (GEFS)":
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gefs/prod/gefs.0p25deg/atmos/2025102100/gep01.t00z.pgrb2.0-120.0-90.0",
    "ECMWF open data (ENS, licence needed)":
        "https://storage.ecmwf.int/european/",
    # --- rainfall / soil moisture ------------------------------------------
    "CHIRPS daily netCDF (UCSB)":
        "https://data.chc.ucsb.edu/chirps/chirps-v2.0.0/daily/netcdf/bd/chirps-v2.0.0.2026-10-20.nc",
    "NASA POWER soil moisture":
        "https://power.larc.nasa.gov/api/temporal/regional/point?parameters=T2M&community=AG&longitude=90&latitude=23&start=20260101&end=20260102&format=JSON",
    "SMAP (NSIDC auth required)":
        "https://nsidc.org/api/hub/authenticate",
    "ESA WorldCover (already used)":
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/2000_2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7/",
    # --- DEM / ancillary ---------------------------------------------------
    "SRTM / NASADEM via earthdata (auth)":
        "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/",
}


def probe(name: str, url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "flood-predictor-probe/1.0",
                                          "Range": "bytes=0-512"})
    rec = {"name": name, "url": url}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["status"] = r.status
            rec["ok"] = r.status < 400
            rec["content_type"] = r.headers.get("Content-Type", "")
            rec["length"] = r.headers.get("Content-Length", "")
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["ok"] = False
        rec["error"] = f"HTTPError {e.code}"
    except Exception as e:
        rec["status"] = None
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return rec


def main() -> int:
    results = []
    for name, url in URLS.items():
        r = probe(name, url)
        results.append(r)
        flag = "OK " if r["ok"] else "-- "
        print(f'{flag}{name:42s} {r.get("status")}  {r.get("error", "")}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
