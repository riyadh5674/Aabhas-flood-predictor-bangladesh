"""Fetch ESA WorldCover 10m land cover for Bangladesh tiles.

Tiles covering Bangladesh (lat 20-27, lon 88-93):
- N21E087, N21E090, N24E087, N24E090, N27E087, N27E090
- Plus N30E087, N30E090 for northern margin

Each tile ~3°x3°, ~10-30 MB compressed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
OUT_DIR = ROOT / "data" / "raw" / "esa_worldcover"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/"

# Bangladesh tiles (3°x3° grid)
BANGLADESH_TILES = [
    "N21E087", "N21E090", "N21E093",
    "N24E087", "N24E090", "N24E093",
    "N27E087", "N27E090", "N27E093",
    "N30E087", "N30E090", "N30E093",
]

TILE_PATTERN = "ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"


def download_tile(tile: str) -> Path:
    """Download a single WorldCover tile."""
    fname = TILE_PATTERN.format(tile=tile)
    url = BASE_URL + fname
    out_path = OUT_DIR / fname

    if out_path.exists():
        print(f"  [{tile}] Already exists ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    print(f"  [{tile}] Downloading from {url}...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"    {tile}", leave=False
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"    [{tile}] Done ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"    [{tile}] FAILED: {e}")
        if out_path.exists():
            out_path.unlink()
        raise


def main() -> int:
    print(f"Fetching ESA WorldCover tiles for Bangladesh...")
    print(f"Output directory: {OUT_DIR}")
    print(f"Tiles to download: {len(BANGLADESH_TILES)}")

    failed = []
    for tile in BANGLADESH_TILES:
        try:
            download_tile(tile)
        except Exception as e:
            failed.append((tile, str(e)))

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for tile, err in failed:
            print(f"  {tile}: {err}")
        return 1

    print(f"\nAll {len(BANGLADESH_TILES)} tiles downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())