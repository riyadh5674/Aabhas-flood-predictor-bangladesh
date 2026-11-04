"""Extract ESA WorldCover land cover features for FFWC stations.

WorldCover classes (v100):
  10: Tree cover
  20: Shrubland
  30: Grassland
  40: Cropland
  50: Built-up
  60: Bare / sparse vegetation
  70: Snow and ice
  80: Permanent water bodies
  90: Herbaceous wetland
  95: Mangroves
  100: Moss and lichen

For each station, compute fractional cover in 1km and 5km buffers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import Point, box, mapping
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
WC_DIR = ROOT / "data" / "raw" / "esa_worldcover"
STATIONS_CSV = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
OUT_CSV = ROOT / "data" / "processed" / "esa_worldcover_features.csv"

# Buffer radii in meters
BUFFERS = [1000, 5000]  # 1km, 5km

# WorldCover class mapping
WC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def load_stations() -> gpd.GeoDataFrame:
    """Load FFWC stations as GeoDataFrame."""
    df = pd.read_csv(STATIONS_CSV)
    df = df.rename(columns={"id": "station_id", "long": "lon"})
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf[["station_id", "name", "geometry"]].copy()


def find_tile_for_point(lon: float, lat: float) -> str:
    """Find WorldCover tile name for a lat/lon point.
    
    Tiles are 3°x3° named N{lat}E{lon} where lat/lon are the SW corner
    rounded down to nearest 3°.
    """
    # Latitude bands: N00, N03, N06, ..., N27, N30, N33...
    tile_lat = int(np.floor(lat / 3.0) * 3)
    # Longitude bands: E000, E003, E006, ..., E087, E090, E093...
    tile_lon = int(np.floor(lon / 3.0) * 3)
    
    # Format: N{lat:02d}E{lon:03d}
    hemi_lat = "N" if tile_lat >= 0 else "S"
    hemi_lon = "E" if tile_lon >= 0 else "W"
    return f"{hemi_lat}{abs(tile_lat):02d}{hemi_lon}{abs(tile_lon):03d}"


def extract_features_for_station(
    station_id: int,
    point: Point,
    tile_cache: dict,
) -> dict:
    """Extract land cover fractions for a station at multiple buffer radii."""
    lon, lat = point.x, point.y
    tile_name = find_tile_for_point(lon, lat)
    tile_path = WC_DIR / f"ESA_WorldCover_10m_2020_v100_{tile_name}_Map.tif"
    
    if not tile_path.exists():
        return {f"wc_{buf}m_{name}": np.nan for buf in BUFFERS for name in WC_CLASSES.values()}
    
    # Open tile (cached)
    if tile_name not in tile_cache:
        tile_cache[tile_name] = rasterio.open(tile_path)
    src = tile_cache[tile_name]
    
    # Create buffer polygons in WGS84, then transform to tile CRS
    results = {}
    for buf_m in BUFFERS:
        # Create buffer in UTM for accurate distance
        # Use UTM zone 46N for Bangladesh (EPSG:32646)
        point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
        point_utm = point_gdf.to_crs("EPSG:32646")
        buffer_utm = point_utm.buffer(buf_m)
        buffer_wgs = buffer_utm.to_crs("EPSG:4326")
        
        # Transform buffer to tile CRS
        buffer_tile = transform_geom("EPSG:4326", src.crs, mapping(buffer_wgs.geometry.iloc[0]))
        
        try:
            # Mask raster with buffer
            out_image, _ = mask(src, [buffer_tile], crop=True, nodata=255)
            data = out_image[0]
            
            # Count pixels per class (exclude nodata=255)
            valid = data != 255
            total = valid.sum()
            
            if total == 0:
                for name in WC_CLASSES.values():
                    results[f"wc_{buf_m}m_{name}"] = np.nan
                continue
            
            for code, name in WC_CLASSES.items():
                count = (data == code).sum()
                results[f"wc_{buf_m}m_{name}"] = count / total
                
        except Exception as e:
            # If buffer extends beyond tile, return NaN
            for name in WC_CLASSES.values():
                results[f"wc_{buf_m}m_{name}"] = np.nan
    
    return results


def main() -> int:
    print("Extracting ESA WorldCover features for FFWC stations...")
    
    stations = load_stations()
    print(f"Loaded {len(stations)} stations")
    
    tile_cache = {}
    all_results = []
    
    for _, row in tqdm(stations.iterrows(), total=len(stations), desc="Stations"):
        station_id = row["station_id"]
        point = row["geometry"]
        
        features = extract_features_for_station(station_id, point, tile_cache)
        features["station_id"] = station_id
        all_results.append(features)
    
    # Close cached rasters
    for src in tile_cache.values():
        src.close()
    
    # Create output DataFrame
    out_df = pd.DataFrame(all_results)
    
    # Ensure column order
    cols = ["station_id"] + sorted([c for c in out_df.columns if c != "station_id"])
    out_df = out_df[cols]
    
    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")
    print(f"Shape: {out_df.shape}")
    print(f"Columns: {list(out_df.columns)}")
    
    # Print summary stats
    for buf in BUFFERS:
        buf_cols = [c for c in out_df.columns if c.startswith(f"wc_{buf}m_")]
        print(f"\n{buf}m buffer coverage:")
        for c in buf_cols:
            valid = out_df[c].notna().sum()
            mean_val = out_df[c].mean()
            print(f"  {c}: {valid}/{len(out_df)} stations, mean={mean_val:.3f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())