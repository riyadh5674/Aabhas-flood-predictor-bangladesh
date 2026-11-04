"""Connectivity probe v3 — the three sources that decide Experiment 2/3 feasibility.

Run:  .venv\\Scripts\\python.exe scripts\\probe_data_sources3.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "domain_adaptation" / "data_source_probe3.json"

BD_AOI = "POLYGON((88.0,20.7,89.9,20.7,89.9,24.2,88.0,24.2,88.0,20.7))"


def odata_product_url(collection: str, start: str, end: str, top: int = 3) -> str:
    flt = (f"Collection/Name eq '{collection}' and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{BD_AOI}') and "
           f"ContentDate/Start gt {start}T00:00:00.000Z and "
           f"ContentDate/Start lt {end}T00:00:00.000Z")
    params = {"$filter": flt, "$top": str(top), "$orderby": "ContentDate/Start asc",
              "$count": "true"}
    return ("https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            + urllib.parse.urlencode(params, quote_via=urllib.parse.quote))


CASES = []


def add(name, url, headers=None):
    CASES.append((name, url, headers or {}))


add("Copernicus S1 products over Bangladesh (2023-06)",
    odata_product_url("SENTINEL-1", "2023-06-01", "2023-07-01"))
add("Copernicus S2 products over Bangladesh (2023-06)",
    odata_product_url("SENTINEL-2", "2023-06-01", "2023-07-01"))
add("Copernicus S1 products 2025-08 (near test period)",
    odata_product_url("SENTINEL-1", "2025-08-01", "2025-09-01"))
add("NOAA GFS bucket listing (prefix)",
    "https://noaa-gfs-bdp-pub.s3.amazonaws.com/?list-type=2&max-keys=5&prefix=gfs.20251021")
add("NOAA GFS latest run index",
    "https://noaa-gfs-bdp-pub.s3.amazonaws.com/index.html")
add("NASA POWER point (correct query)",
    "https://power.larc.nasa.gov/api/temporal/point?parameters=T2M,PRECTOTCORR&community=AG"
    "&longitude=90.4&latitude=23.7&start=20260101&end=20260110&format=JSON")
add("NASA POWER daily climatology",
    "https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M&community=AG"
    "&longitude=90.4&latitude=23.7&start=2001&end=2020&format=JSON")
add("Copernicus DEM (GLO-30) collection",
    odata_product_url("COP-DEM", "2020-01-01", "2020-02-01", top=1))
add("Copernicus WorldCover (esa-worldcover 2020)",
    odata_product_url("ESA_WORLDCOVER", "2020-01-01", "2020-02-01", top=1))


def probe(name, url, headers):
    h = {"User-Agent": "Mozilla/5.0 (flood-predictor-probe/3.0)"}
    h.update(headers)
    rec = {"name": name, "url": url}
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read(4000)
            rec["status"] = r.status
            rec["ok"] = r.status < 400
            rec["content_type"] = r.headers.get("Content-Type", "")
            rec["length"] = r.headers.get("Content-Length", "")
            rec["head"] = body[:600].decode("utf-8", "replace").replace("\n", " ")
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
        extra = r.get("error", "") or r.get("head", "")[:140]
        print(f'{flag}{name:52s} {r.get("status")}  {extra}\n')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
