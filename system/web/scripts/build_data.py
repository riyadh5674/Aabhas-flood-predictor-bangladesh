"""One-off generator for the Aabhas frontend static data.

Reads the real FFWC station master and the geoBoundaries admin layers, then
writes normalised JSON into system/web/public/data so the frontend can run
without any backend.

Run:
    .\\.venv\\Scripts\\python.exe system\\web\\scripts\\build_data.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "system" / "web"
OUT = WEB / "public" / "data"

# station master -> geoBoundaries shapeName (ADM1)
DIVISION_KEY = {
    "Barishal": "Barisal",
    "Chattogram": "Chittagong",
    "Chittagong": "Chittagong",
    "Dhaka": "Dhaka",
    "Khulna": "Khulna",
    "Mymensingh": "Mymensingh",
    "Rajshahi": "Rajshani",  # geoBoundaries spells it this way
    "Rangpur": "Rangpur",
    "Sylhet": "Sylhet",
}

# station master -> geoBoundaries shapeName (ADM2)
DISTRICT_KEY = {
    "Barishal": "Barisal",
    "Bogura": "Bogra",
    "Brahmmonbaria": "Brahamanbaria",
    "Chapai Nawabganj": "Nawabganj",
    "Chattogram": "Chittagong",
    "Cumilla": "Comilla",
    "Kishorganj": "Kishoreganj",
    "Laksmipur": "Lakshmipur",
    "Moulvi Bazar": "Maulvibazar",
    "Netrokona": "Netrakona",
}

# Display names used in the UI (modern spelling, what people actually search for)
DISTRICT_DISPLAY = {
    "Bogra": "Bogura",
    "Chattogram": "Chattogram",
    "Comilla": "Cumilla",
    "Jessore": "Jashore",
    "Barishal": "Barishal",
    "Chapai Nawabganj": "Chapai Nawabganj",
    "Maulvibazar": "Moulvibazar",
    "Netrokona": "Netrokona",
    "Brahmmonbaria": "Brahmanbaria",
    "Kishorganj": "Kishoreganj",
}

DIVISION_DISPLAY = {"Chattogram": "Chattogram", "Chittagong": "Chattogram"}

# Fallback for station rows with an empty division field (newer stations).
DIVISION_BY_DISTRICT = {
    "Bogura": "Rajshahi",
    "Chattogram": "Chattogram",
    "Cumilla": "Chattogram",
    "Feni": "Chattogram",
    "Laksmipur": "Chattogram",
    "Sirajganj": "Rajshahi",
}


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geo").mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv")
    df = df.dropna(subset=["id", "name"])
    df["id"] = df["id"].astype(int)

    stations = []
    for _, r in df.iterrows():
        district = str(r["district"]).strip()
        division = str(r["division"]).strip()
        if division in ("nan", "None", ""):
            division = DIVISION_BY_DISTRICT.get(district, "")
        stations.append(
            {
                "id": int(r["id"]),
                "name": str(r["name"]).strip(),
                "river": str(r["river"]).strip() if pd.notna(r["river"]) else "",
                "basin": str(r["basin"]).strip() if pd.notna(r["basin"]) else "",
                "division": DIVISION_DISPLAY.get(division, division),
                "divisionKey": DIVISION_KEY.get(division, division),
                "district": DISTRICT_DISPLAY.get(district, district),
                "districtKey": DISTRICT_KEY.get(district, district),
                "upazilla": str(r["upazilla"]).strip() if pd.notna(r["upazilla"]) else "",
                "union": str(r["union"]).strip() if pd.notna(r["union"]) else "",
                "lat": round(float(r["lat"]), 5),
                "lon": round(float(r["long"]), 5),
                "dangerLevel": _num(r["dangerlevel"]),
                "recordHigh": _num(r["rhwl"]),
            }
        )

    stations.sort(key=lambda s: (s["division"], s["district"], s["name"]))
    (OUT / "stations.json").write_text(
        json.dumps(stations, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    for level, src in {
        "adm1": ROOT / "data/raw/geo/admin/BGD_ADM1/geoBoundaries-BGD-ADM1_simplified.geojson",
        "adm2": ROOT / "data/raw/geo/admin/BGD_ADM2/geoBoundaries-BGD-ADM2_simplified.geojson",
    }.items():
        shutil.copyfile(src, OUT / "geo" / f"{level}.geojson")

    print(f"stations: {len(stations)}")
    print(f"districts covered: {len({s['districtKey'] for s in stations})}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
