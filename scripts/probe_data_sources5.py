"""Connectivity probe v5 — final syntax check for the Copernicus DataSpace catalogue.

Run:  .venv\\Scripts\\python.exe scripts\\probe_data_sources5.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "domain_adaptation" / "data_source_probe5.json"

BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# OData v4 geography literal: coordinates separated by SPACES, not commas.
BD_SPACE = ("POLYGON((88.0 20.7,89.9 20.7,89.9 24.2,88.0 24.2,88.0 20.7))")
BD_COMMA = ("POLYGON((88.0,20.7,89.9,20.7,89.9,24.2,88.0,24.2,88.0,20.7))")

VARIANTS = [
    ("intersects + space-separated coords",
     {"$top": "2", "$filter":
      f"OData.CSC.Intersects(area=geography'SRID=4326;{BD_SPACE}')"}),
    ("collection + intersects (space coords)",
     {"$top": "2", "$filter":
      "Collection/Name eq 'SENTINEL-1' and "
      f"OData.CSC.Intersects(area=geography'SRID=4326;{BD_SPACE}')"}),
    ("collection + intersects (comma coords) + date",
     {"$top": "2", "$filter":
      "Collection/Name eq 'SENTINEL-1' and "
      f"OData.CSC.Intersects(area=geography'SRID=4326;{BD_COMMA}') and "
      "ContentDate/Start gt 2023-06-01T00:00:00.000Z and "
      "ContentDate/Start lt 2023-07-01T00:00:00.000Z"}),
    ("startswith(Name,'S1A_GRD')",
     {"$top": "3", "$filter": "startswith(Name,'S1A_GRD')",
      "$orderby": "ContentDate/Start desc"}),
    ("attributes metadata",
     {"$top": "5", "$filter": "startswith(Name,'S1') and "
                             "ContentDate/Start gt 2025-08-01T00:00:00.000Z",
      "$orderby": "ContentDate/Start desc"}),
    ("S1 GRD in 2025-08 + intersects (space)",
     {"$top": "5", "$filter":
      "startswith(Name,'S1') and "
      f"OData.CSC.Intersects(area=geography'SRID=4326;{BD_SPACE}') and "
      "ContentDate/Start gt 2025-08-01T00:00:00.000Z and "
      "ContentDate/Start lt 2025-09-01T00:00:00.000Z",
      "$orderby": "ContentDate/Start desc"}),
]


def probe(url):
    h = {"User-Agent": "Mozilla/5.0"}
    rec = {"url": url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
            body = r.read(6000).decode("utf-8", "replace")
            rec["status"] = r.status
            rec["ok"] = True
            rec["head"] = body[:500].replace("\n", " ")
            rec["len"] = len(body)
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["ok"] = False
        rec["error"] = f"HTTPError {e.code}: {e.read(300).decode('utf-8','replace')}"
    except Exception as e:
        rec["status"] = None
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    return rec


def main() -> int:
    out = []
    for name, params in VARIANTS:
        url = BASE + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        r = probe(url)
        r["name"] = name
        out.append(r)
        print(f'{"OK " if r["ok"] else "-- "}{name:44s} {r.get("status")}\n'
              f'    {r.get("error", r.get("head", ""))[:260]}\n')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
