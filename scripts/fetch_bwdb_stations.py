"""Phase 0a: harvest public BWDB/FFWC station metadata + availability tables."""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(r"D:\Flood predictor\data\raw\ffwc")
RAW.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}

TARGETS = {
    "bwdb_water_level_availability.html": "http://www.hydrology.bwdb.gov.bd/includes/water_level_data_available_print.php?dist=&river=",
    "bwdb_discharge_availability.html": "http://www.hydrology.bwdb.gov.bd/includes/discharge_data_available_print.php?dist=&river=",
    "ffwc_observed_waterlevel_today.html": "http://old.ffwc.gov.bd/dev/ffwc_charts/waterlevelobs.php",
    "ffwc_home.html": "http://old.ffwc.gov.bd/index.php",
}

for name, url in TARGETS.items():
    try:
        r = requests.get(url, headers=H, timeout=60)
        print(r.status_code, len(r.content), name)
        (RAW / name).write_bytes(r.content)
    except Exception as e:
        print("ERR", name, e)

# parse the two availability tables into CSV
for kind in ["water_level", "discharge"]:
    p = RAW / f"bwdb_{kind}_availability.html"
    if not p.exists() or p.stat().st_size == 0:
        continue
    try:
        tables = pd.read_html(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print("read_html failed", kind, e)
        continue
    print(kind, "tables found:", len(tables), [t.shape for t in tables])
    for i, t in enumerate(tables):
        print("  cols:", list(t.columns)[:14])
        if t.shape[0] > 10:
            out = RAW / f"bwdb_{kind}_availability.csv"
            t.to_csv(out, index=False)
            print("  saved", out, t.shape)
            break
