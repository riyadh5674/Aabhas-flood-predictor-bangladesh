"""
System Map — Geographic / map layer configuration.

Paths and normalization for joining station risk bands to admin boundaries.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()

# Admin boundary GeoJSON (geoBoundaries)
ADM1_GEOJSON = ROOT / "data" / "raw" / "geo" / "admin" / "BGD_ADM1" / "geoBoundaries-BGD-ADM1_simplified.geojson"
ADM2_GEOJSON = ROOT / "data" / "raw" / "geo" / "admin" / "BGD_ADM2" / "geoBoundaries-BGD-ADM2_simplified.geojson"

# Station master (ingested by Step 2)
STATIONS_FILE = ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv"

# Output
MAP_OUT_DIR = ROOT / "system" / "data" / "processed" / "map"
CHOROPLETH_ADM1 = MAP_OUT_DIR / "choropleth_adm1.geojson"
CHOROPLETH_ADM2 = MAP_OUT_DIR / "choropleth_adm2.geojson"

# Risk band severity (worst = highest)
BAND_SEVERITY = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "very_high": 3,
    "unknown": -1,
}

SEVERITY_TO_BAND = {v: k for k, v in BAND_SEVERITY.items()}

# Division name normalization: station master -> GeoJSON shapeName
DIVISION_NORM = {
    "Chattogram": "Chittagong",
    "Chittagong": "Chittagong",
    "Barishal": "Barisal",
    "Sylhet": "Sylhet",
    "Dhaka": "Dhaka",
    "Rajshahi": "Rajshani",
    "Khulna": "Khulna",
    "Rangpur": "Rangpur",
    "Mymensingh": "Mymensingh",
    "Rajshani": "Rajshahi",
}

# District name normalization: station master -> GeoJSON shapeName
DISTRICT_NORM = {
    "Barishal": "Barisal",
    "Brahmmonbaria": "Brahamanbaria",
    "Chapai Nawabganj": "Nawabganj",
    "Chattogram": "Chittagong",
    "Chittagong": "Chittagong",
    "Kishorganj": "Kishoreganj",
    "Netrokona": "Netrakona",
}

ADM_LEVELS = ["adm1", "adm2"]

__all__ = [
    "ROOT", "ADM1_GEOJSON", "ADM2_GEOJSON", "STATIONS_FILE",
    "MAP_OUT_DIR", "CHOROPLETH_ADM1", "CHOROPLETH_ADM2",
    "BAND_SEVERITY", "SEVERITY_TO_BAND", "DIVISION_NORM", "DISTRICT_NORM",
    "ADM_LEVELS",
]
