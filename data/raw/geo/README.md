# data/raw/geo - provenance manifest

Downloaded 2026-10-08 into `data/raw/geo`. Bulk data is git-ignored; this README is the record.

| path | source | licence | files | bytes |
| --- | --- | --- | --- | --- |
| `srtm/` | NASA SRTM 1-arc-sec (30 m) via Mapzen AWS `elevation-tiles-prod/skadi` | public domain (NASA/USGS) | 40 tiles N21-N27 x E88-E93 | 379593029 |
| `osm/bangladesh.osm.pbf` | https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf | ODbL (c) OpenStreetMap contributors | 1 | 355789515 |
| `osm/rivers_bangladesh.geojson` | derived: GDAL OSM lines layer, waterway=river, bbox 88,20.4,92.7,26.7 | ODbL (c) OpenStreetMap contributors | 1 | 12821461 |

Notes
- Two west-edge tiles were initially truncated by the AWS mirror download; N24E088 redownloaded OK, N27E088 persistently truncates. No station sits in N27E088, so elevation coverage is 133/133 anyway.
- `riverbanks_bangladesh.geojson` is deferred: the OSM multipolygons layer is ~12M features; the 6,792 river centerlines fully cover the distance/density features (Phase 11).
- `dist_to_coast_km` = nearest SRTM land/ocean boundary cell (equirectangular approx, ~1 km).
- `dist_major_river_km` = nearest OSM river segment longer than 20 km (structural proxy for the trunk flood carriers; OSM names are mostly Bengali so no reliable name join to the FFWC catalog).
- Derived artefacts live in `data/processed/station_spatial_features.csv` (133+ stations).