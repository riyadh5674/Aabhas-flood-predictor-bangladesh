"""Phase 11 - static spatial features for the 133 FFWC stations.

Sources
    data/raw/geo/srtm/N{lat}/N{lat}E{lon}.hgt.gz     SRTM 1" elevation tiles (Mapzen/AWS skadi)
    data/raw/geo/osm/rivers_bangladesh.geojson       OSM waterway=river centerlines (GeoFabrik extract)
    data/raw/ffwc/ffwc_stations.csv                  station catalog (lat/lon)

Output
    data/processed/station_spatial_features.csv      133 rows, one per station

Notes
    - elevation / slope: SRTM 1-arc-sec (1201x1201), nearest-neighbour sample, slope from a
      3x3 neighbourhood (meters via 110540 / 111320*cos(lat) per degree).
    - dist_to_coast: nearest SRTM land/ocean boundary cell (ocean = nodata/-1); equirectangular
      approximation, documented to ~1 km.
    - dist_river_km / dist_major_river_km: nearest OSM river line; "major" = segments longer
      than MAJOR_KM (trunk flood-carriers), a structural proxy for the Ganges/Meghna/Jamuna
      trunks (OSM names are mostly Bengali; no reliable transliteration join to the catalog).
    - river_density_10km: total river line length within a 10 km radius / area of the disc.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
SRTM = ROOT / "data" / "raw" / "geo" / "srtm"
RIVERS = ROOT / "data" / "raw" / "geo" / "osm" / "rivers_bangladesh.geojson"
GEO = ROOT / "data" / "processed" / "station_spatial_features.csv"
MAJOR_KM = 20.0
DEN_KM = 10.0
UTM = "EPSG:32646"


def load_tile(lat: int, lon: int) -> np.ndarray | None:
    p = SRTM / f"N{lat:02d}" / f"N{lat:02d}E{lon:03d}.hgt.gz"
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rb") as fh:
            raw = np.frombuffer(fh.read(), dtype=">i2").reshape(3601, 3601).astype(np.float32)
    except Exception:
        return None
    raw[raw == -32768] = np.nan
    return raw


def sample_elev(tile_lat: int, tile_lon: int, tile: np.ndarray, lat: float, lon: float):
    row = int(round((tile_lat + 1.0 - lat) * 3600.0))
    col = int(round((lon - tile_lon) * 3600.0))
    return tile[row, col]


def main() -> None:
    st = pd.read_csv(ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv")
    st = st.rename(columns={"id": "station_id"})
    st = st[["station_id", "name", "river", "basin", "division", "lat", "long"]].copy()
    st = st.rename(columns={"long": "lon"})

    elev, slope = np.full(len(st), np.nan), np.full(len(st), np.nan)
    coast_all = []  # accumulated coastline lon/lat from every tile
    for la in range(20, 28):
        for lo in range(88, 94):
            tile = load_tile(la, lo)
            if tile is None:
                continue
            ocean = np.isnan(tile) | (tile < 0.0)
            if la in range(21, 28) and lo in range(88, 94):
                land = ~ocean
                boundary = np.zeros_like(land)
                if land.any():
                    neigh = np.zeros_like(land)
                    for d1 in (-1, 0, 1):
                        for d2 in (-1, 0, 1):
                            if d1 == 0 and d2 == 0:
                                continue
                            sh = np.roll(np.roll(ocean, d1, axis=0), d2, axis=1)
                            neigh |= sh
                    boundary = land & neigh
                    if boundary.any():
                        ys, xs = np.where(boundary)
                        step = 5
                        coast_lat = (la + 1.0) - ys[::step] / 3600.0
                        coast_lon = lo + xs[::step] / 3600.0
                        coast_all.append(np.column_stack([coast_lon, coast_lat]))
            # station elevation + slope from the tile that covers each station
            if la in range(21, 28) and lo in range(88, 94):
                for i, (sla, slo) in enumerate(zip(st["lat"], st["lon"])):
                    tl = int(np.floor(sla))
                    tn = int(np.floor(slo))
                    if tl != la or tn != lo:
                        continue
                    r = int(round((la + 1.0 - sla) * 3600.0))
                    c = int(round((slo - lo) * 3600.0))
                    if r < 0 or c < 0 or r > 3600 or c > 3600:
                        continue
                    tilep = np.pad(tile, 1, mode="edge")
                    elev[i] = tile[r, c]
                    win = tilep[r:r + 3, c:c + 3]
                    if np.isnan(win).any():
                        slope[i] = np.nan
                    else:
                        m_per_d_lat = 110540.0
                        m_per_d_lon = 111320.0 * np.cos(np.radians(sla))
                        dlat = m_per_d_lat / 3600.0
                        dlon = m_per_d_lon / 3600.0
                        zx = (win[2, 1] - win[0, 1]) / 2.0
                        zy = (win[1, 2] - win[1, 0]) / 2.0
                        slope[i] = np.degrees(np.arctan(np.hypot(zx / dlat, zy / dlon)))

    coast = np.vstack(coast_all) if coast_all else np.zeros((1, 2))

    # distance to coast (equirectangular)
    dist_coast = np.full(len(st), np.nan)
    c_lon, c_lat = coast[:, 0], coast[:, 1]
    for i, (sla, slo) in enumerate(zip(st["lat"], st["lon"])):
        dx = (c_lon - slo) * 111.32 * np.cos(np.radians(sla))
        dy = (c_lat - sla) * 110.57
        dist_coast[i] = np.sqrt((dx ** 2 + dy ** 2).min())

    # river distance
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs="EPSG:4326").to_crs(UTM)
    riv = gpd.read_file(RIVERS).to_crs(UTM)
    riv["len_km"] = riv.geometry.length / 1000.0

    nn = gpd.sjoin_nearest(pts, riv[["geometry"]], how="left", distance_col="dist_river_m")
    dist_river = nn["dist_river_m"].fillna(np.nan).to_numpy() / 1000.0

    major = riv[riv["len_km"] >= MAJOR_KM]
    nn2 = gpd.sjoin_nearest(pts, major[["geometry"]], how="left", distance_col="dist_maj_m")
    dist_major = nn2["dist_maj_m"].fillna(np.nan).to_numpy() / 1000.0

    den, nlin = np.full(len(st), np.nan), np.zeros(len(st), dtype=int)
    riv_u = riv.geometry.union_all()
    for i, point in enumerate(pts.geometry):
        buf = point.buffer(DEN_KM * 1000.0)
        inter = gpd.clip(riv, buf)
        length_km = inter.geometry.length.sum() / 1000.0
        den[i] = length_km / (np.pi * DEN_KM ** 2)
        nlin[i] = len(inter)

    out = st.copy()
    out["elevation_m"] = elev
    out["slope_deg"] = slope
    out["dist_to_coast_km"] = dist_coast
    out["dist_river_km"] = dist_river
    out["dist_major_river_km"] = dist_major
    out["river_density_10km_km_per_km2"] = den
    out["n_river_lines_10km"] = nlin
    out.to_csv(GEO, index=False)
    pd.set_option("display.width", 200)
    print(out[["station_id", "name", "elevation_m", "slope_deg", "dist_to_coast_km",
               "dist_river_km", "dist_major_river_km", "river_density_10km_km_per_km2"]].describe().round(2).to_string())
    print("\n-- known-sanity stations --")
    sanity = ["Mongla", "Khulna", "Daulatkhan", "Bandarban", "Lama", "Chiringa",
              "Panchagarh", "Sylhet", "Dhaka", "Kurigram", "goalundo", "Tarapur", "Mymensingh"]
    mask = out["name"].astype(str).str.lower().isin([s.lower() for s in sanity])
    print(out[mask][["station_id", "name", "elevation_m", "slope_deg", "dist_to_coast_km",
                     "dist_river_km", "dist_major_river_km"]].round(2).to_string())
    print("\ncoast points:", len(coast), "| major lines kept:", len(major))


if __name__ == "__main__":
    main()