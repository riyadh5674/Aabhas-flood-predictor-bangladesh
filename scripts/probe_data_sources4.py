"""Connectivity probe v4 — narrow down the Copernicus DataSpace OData filter syntax.

Run:  .venv\\Scripts\\python.exe scripts\\probe_data_sources4.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "domain_adaptation" / "data_source_probe4.json"

BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
BD = "POLYGON((88.0,20.7,89.9,20.7,89.9,24.2,88.0,24.2,88.0,20.7))"

VARIANTS = [
    ("a: $top only", {"$top": "2"}),
    ("b: + collection filter",
     {"$top": "2", "$filter": "Collection/Name eq 'SENTINEL-1'"}),
    ("c: + intersects",
     {"$top": "2", "$filter": "Collection/Name eq 'SENTINEL-1' and "
                             f"OData.CSC.Intersects(area=geography'SRID=4326;{BD}')"}),
    ("d: + date range",
     {"$top": "2", "$filter": "Collection/Name eq 'SENTINEL-1' and "
                             f"OData.CSC.Intersects(area=geography'SRID=4326;{BD}') and "
                             "ContentDate/Start gt 2023-06-01T00:00:00.000Z and "
                             "ContentDate/Start lt 2023-07-01T00:00:00.000Z"}),
    ("e: orderby only", {"$top": "2", "$orderby": "ContentDate/Start desc"}),
    ("f: count only", {"$count": "true", "$top": "1"}),
    ("g: product type", {"$top": "2", "$filter": "ProductType eq 'S1GRD'"}),
    ("h: intersects only (no collection)",
     {"$top": "2", "$filter": f"OData.CSC.Intersects(area=geography'SRID=4326;{BD}')"}),
]


def probe(url: str):
    h = {"User-Agent": "Mozilla/5.0"}
    rec = {"url": url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=45) as r:
            body = r.read(2500).decode("utf-8", "replace")
            rec["status"] = r.status
            rec["ok"] = True
            rec["head"] = body[:400].replace("\n", " ")
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
        print(f'{"OK " if r["ok"] else "-- "}{name:34s} {r.get("status")}  '
              f'{r.get("error", r.get("head", ""))[:170]}\n')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
