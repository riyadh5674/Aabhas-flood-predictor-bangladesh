"""Phase 0b: FFWC API - station master + historical water-level depth scan."""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(r"D:\Flood predictor\data\raw\ffwc")
RAW.mkdir(parents=True, exist_ok=True)

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "x-ffwc-internal-key": "FFWC-Project-2026-Secure-V1",
    "Referer": "https://ffwc.gov.bd/",
    "Origin": "https://ffwc.gov.bd",
    "Accept": "application/json",
}
API = "https://api.ffwc.gov.bd/data_load"


def get(path, timeout=30):
    r = requests.get(f"{API}/{path}", headers=H, timeout=timeout)
    r.raise_for_status()
    return r.json()


# 1. station master
st = get("stations")
df = pd.DataFrame(st)
df.to_csv(RAW / "ffwc_stations.csv", index=False)
print("stations:", df.shape)
print(df.columns.tolist())
print(df.head(3).to_string())

# 2. depth scan: earliest available date per sample station
sample = df.head(5)["id"].tolist() + df.tail(5)["id"].tolist()
years = list(range(2025, 2013, -1))
rows = []
for sid in sample:
    line = []
    for y in years:
        try:
            j = get(f"observed-waterlevel-by-station-and-date/{sid}/{y}-07-15")
            line.append("Y" if j else ".")
        except Exception:
            line.append("?")
        time.sleep(0.15)
    rows.append((sid, df.loc[df["id"] == sid, "name"].iloc[0], "".join(line)))
    print(rows[-1])

print("\nyears:", years)
