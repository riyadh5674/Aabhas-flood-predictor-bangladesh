"""Connectivity probe v2 — the sources that Experiment 2-5 depend on.

Run:  .venv\\Scripts\\python.exe scripts\\probe_data_sources2.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "domain_adaptation" / "data_source_probe2.json"

CASES = [
    # (name, url, extra headers)
    ("Copernicus S1 catalogue (Bangladesh AOI)",
     "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
     "$filter=Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;"
     "POLYGON((88.0,20.7,89.9,20.7,89.9,24.2,88.0,24.2,88.0,20.7))') and "
     "ContentDate/Start gt 2023-01-01T00:00:00.000Z and ContentDate/Start lt 2023-02-01T00:00:00.000Z"
     "&$top=2&$orderby=ContentDate/Start asc&$count=true", {}),
    ("Copernicus S2 catalogue (Bangladesh AOI)",
     "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
     "$filter=Collection/Name eq 'SENTINEL-2' and OData.CSC.Intersects(area=geography'SRID=4326;"
     "POLYGON((88.0,20.7,89.9,20.7,89.9,24.2,88.0,24.2,88.0,20.7))')"
     "&$top=2&$count=true", {}),
    ("Copernicus DataSpace token endpoint",
     "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
     {"Content-Type": "application/x-www-form-urlencoded"}),
    ("NOMADS GEFS (no Range header)",
     "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gefs/prod/gefs.0p25deg/atmos/2025102100/"
     "gep01.t00z.pgrb2.0-120.0-90.0", {}),
    ("NOMADS GFS index",
     "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.0p25deg/atmos/2025102100/", {}),
    ("NOAA GFS AWS open data (S3, no key needed)",
     "https://noaa-gfs-bdp-pub.s3.amazonaws.com/index.html", {}),
    ("AWS open-data GFS 0.25 (gfs.t00z.pgrb2.0-120.0-90.0)",
     "https://noaa-gfs-bdp-pub.s3.amazonaws.com/gfs.2025102100/gfs.t00z.pgrb2.0-120.0-90.0", {}),
    ("NASA POWER point API (T2M, 2 days)",
     "https://power.larc.nasa.gov/api/temporal/point?parameters=T2M&community=AG&longitude=90.4"
     "&latitude=23.7&start=20260101&end=20260102&format=JSON", {}),
    ("NASA POWER monthly climatology (multi-year)",
     "https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,PREP2M&community=AG"
     "&longitude=90.4&latitude=23.7&start=2001&end=2020&format=JSON", {}),
    ("CHIRPS daily netCDF (bd, older date)",
     "https://data.chc.ucsb.edu/chirps/chirps-v2.0.0/daily/netcdf/bd/chirps-v2.0.0.2023-06-15.nc", {}),
    ("CHIRPS global daily netCDF",
     "https://data.chc.ucsb.edu/chirps/chirps-v2.0.0/daily/netcdf/global/chirps-v2.0.0.2023-06-15.nc", {}),
    ("CHIRPS v2.0.1 directory listing",
     "https://data.chc.ucsb.edu/chirps/chirps-v2.0.1/daily/netcdf/bd/", {}),
    ("ESA WorldCover S3 bucket (known-good prefix)",
     "https://esa-worldcover.s3.eu-central-1.amazonaws.com/2000_2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7_ESE_WGS84_2020_N30E090_Map.tif", {}),
    ("USGS/NASADEM elevation (ASTER) via earthdata mirror",
     "https://dwtkns.com/srtm30m/", {}),
    ("OpenTopography global DEM API (requires key)",
     "https://portal.opentopography.org/ot/rest/v1/globaldem", {}),
]


def probe(name: str, url: str, headers: dict) -> dict:
    h = {"User-Agent": "flood-predictor-probe/2.0"}
    h.update(headers)
    rec = {"name": name, "url": url}
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=40) as r:
            body = r.read(2048)
            rec["status"] = r.status
            rec["ok"] = r.status < 400
            rec["content_type"] = r.headers.get("Content-Type", "")
            rec["length"] = r.headers.get("Content-Length", "")
            rec["head"] = body[:300].decode("utf-8", "replace").replace("\n", " ")
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["ok"] = False
        rec["error"] = f"HTTPError {e.code} {e.reason}"
    except Exception as e:
        rec["status"] = None
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec


def main() -> int:
    out = []
    for name, url, hdr in CASES:
        r = probe(name, url, hdr)
        out.append(r)
        flag = "OK " if r["ok"] else "-- "
        extra = r.get("error", "") or (r.get("head", "")[:90])
        print(f'{flag}{name:52s} {r.get("status")}  {extra}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
