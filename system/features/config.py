"""
System Features — Configuration.

Defines the 110-column FULL_V2 feature set used by the frozen D16 model,
plus feature-group metadata and paths for the system ingestion pipeline.

All feature names must match models/final_v2/final_config_v2.json exactly.
"""
from __future__ import annotations

import os
from pathlib import Path

# Keep this module portable: the same code runs in the Windows development
# checkout, GitHub Actions, and the Render container.
ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
SYSTEM = ROOT / "system"

# ---- Input paths (raw data ingested by Step 2) ----
RAW_WL_DIR = SYSTEM / "data" / "raw" / "ffwc" / "waterlevel"
RAW_WL_MANIFEST = SYSTEM / "data" / "raw" / "ffwc" / "harvest_manifest.csv"
RAW_RAIN = SYSTEM / "data" / "raw" / "rainfall" / "weather_station_daily.csv"
RAW_RAIN_MANIFEST = SYSTEM / "data" / "raw" / "rainfall" / "rainfall_manifest.csv"
STATIONS_FILE = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"  # parent repo, read-only

# ---- Reference data (parent repo, read-only) ----
SPATIAL_FILE = ROOT / "data" / "processed" / "station_spatial_features.csv"
ERA5_FILE = ROOT / "data" / "raw" / "openmeteo" / "era5land_station_daily.csv"
GLOFAS_DIR = ROOT / "data" / "raw" / "glofas"
POWER_SOIL_FILE = ROOT / "data" / "raw" / "power_soil" / "power_soil_daily.csv"

# ---- Output paths ----
PROC_DIR = SYSTEM / "data" / "processed"
MODEL_READY_DIR = PROC_DIR / "model_ready"
FEATURES_PARQUET = PROC_DIR / "features.parquet"

# ---- QC thresholds (same as ingestion layer) ----
MIN_READINGS_PER_DAY = 3
WL_MAX_PHYSICAL_M = 100.0
MISSING_CODE_SENTINEL = -1000.0
SPIKE_JUMP_M = 10.0
SPIKE_WINDOW_READINGS = 4
SPIKE_GAP_S = 7 * 86400

# ---- Feature group constants (from build_dataset.py / build_mi_features.py) ----
NEIGHB_KS = (3, 5)
NEIGHB_SIGS = ("rain_7d", "rain_14d", "wl_pct_danger")
NEIGHB_POOL = 15
UPSTREAM_MAX_KM = 250.0
UPSTREAM_ELEV_DIFF_M = 0.5
UPSTREAM_LAGS = (1, 2)

# GloFAS basin anchor mapping (from build_mi_features.py)
BASIN_ANCHORS = {
    "Ganges":      ["patna_ganges", "farakka_ganges", "chisapani_karnali", "bhimnagar_kosi", "hardinge_ganges"],
    "Brahmaputra": ["dibrugarh_brah", "tezpur_brah", "teesta_in", "bahadurabad_brah"],
    "Meghna":      ["silchar_barak", "bhairab_meghna"],
}
DEFAULT_ANCHORS = ["bahadurabad_brah", "hardinge_ganges", "bhairab_meghna"]

SPATIAL_COLS = ["elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km",
                "dist_major_river_km", "river_density_10km_km_per_km2", "n_river_lines_10km"]

DIVISION_NORM = {"Chittagong": "Chattogram"}

# ---- FULL_V2 = 110 columns (exact order from feature_sets_v2.json) ----
FULL_V2 = [
    "river", "basin", "division", "district", "upazilla",
    "lat", "lon", "danger_level",
    "n_readings",
    "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last",
    "elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km",
    "dist_major_river_km", "river_density_10km_km_per_km2", "n_river_lines_10km",
    "wl_last_lag1", "wl_last_lag2", "wl_last_lag3", "wl_last_lag4",
    "wl_last_lag5", "wl_last_lag6", "wl_last_lag7",
    "wl_max_lag1", "wl_max_lag2", "wl_max_lag3",
    "wl_rise_1d", "wl_rise_3d",
    "wl_dist_to_danger", "wl_pct_danger",
    "wl_max_roll7", "wl_max_roll30",
    "wl_exceed_days_30",
    "precip_mm", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "rain_max_7d", "wet_days_7",
    "tmax_c", "tmin_c",
    "doy_sin", "doy_cos",
    "nb3_rain_7d", "nb3_rain_14d", "nb3_wl_pct_danger",
    "nb5_rain_7d", "nb5_rain_14d", "nb5_wl_pct_danger",
    "nb3_n_wl_danger", "nb5_n_wl_danger",
    "up_rain7d_lag1", "up_rain7d_lag2",
    "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2",
    "ps_wet_top", "ps_wet_root", "ps_wet_prof",
    "ps_wet_top_lag1", "ps_wet_top_lag7",
    "ps_wet_root_lag1", "ps_wet_root_lag7",
    "ps_wet_prof_lag1", "ps_wet_prof_lag7",
    "ps_wet_top_chg1d", "ps_wet_top_chg7d",
    "ps_wet_top_7d", "ps_wet_top_14d", "ps_wet_top_30d",
    "ps_wet_root_7d",
    "ps_wet_top_clim", "ps_wet_top_anom",
    "ps_wet_root_clim", "ps_wet_root_anom",
    "gf_up_ratio", "gf_up_rise3d",
    "gf_up_ratio_lag1", "gf_up_ratio_lag2", "gf_up_ratio_lag3",
    "gf_up_ratio_lag5", "gf_up_ratio_lag7", "gf_up_ratio_lag10", "gf_up_ratio_lag14",
    "gf_up_ratio_3d", "gf_up_ratio_7d", "gf_up_ratio_14d",
    "gf_up_rise3d_lag1", "gf_up_rise3d_lag3",
    "gf_up_accel",
    "gf_near_dis", "gf_near_dis_lag1", "gf_near_dis_lag3",
    "wl_last_lag10", "wl_last_lag14", "wl_last_lag21", "wl_last_lag30",
    "api_09", "api_07",
    "ix_wet_top_x_slope", "ix_wet_top_x_updis",
    "ix_updis_x_wlrise", "ix_api09_x_wet_top",
    "ix_updis14_x_api",
]

assert len(FULL_V2) == 110, f"FULL_V2 has {len(FULL_V2)} columns, expected 110"

# ---- Feature groups for reporting ----
FEATURE_GROUPS = {
    "static": ["river", "basin", "division", "district", "upazilla", "lat", "lon",
               "danger_level", "elevation_m", "slope_deg", "dist_to_coast_km",
               "dist_river_km", "dist_major_river_km",
               "river_density_10km_km_per_km2", "n_river_lines_10km"],
    "hydrology_base": ["n_readings", "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last"],
    "hydrology_lags": ["wl_last_lag1", "wl_last_lag2", "wl_last_lag3", "wl_last_lag4",
                       "wl_last_lag5", "wl_last_lag6", "wl_last_lag7",
                       "wl_max_lag1", "wl_max_lag2", "wl_max_lag3",
                       "wl_last_lag10", "wl_last_lag14", "wl_last_lag21", "wl_last_lag30"],
    "hydrology_derived": ["wl_rise_1d", "wl_rise_3d", "wl_dist_to_danger", "wl_pct_danger",
                          "wl_max_roll7", "wl_max_roll30", "wl_exceed_days_30",
                          "api_09", "api_07"],
    "rainfall": ["precip_mm", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
                 "rain_max_7d", "wet_days_7", "tmax_c", "tmin_c"],
    "temporal": ["doy_sin", "doy_cos"],
    "neighbour": ["nb3_rain_7d", "nb3_rain_14d", "nb3_wl_pct_danger",
                  "nb5_rain_7d", "nb5_rain_14d", "nb5_wl_pct_danger",
                  "nb3_n_wl_danger", "nb5_n_wl_danger"],
    "upstream": ["up_rain7d_lag1", "up_rain7d_lag2",
                 "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2"],
    "soil_moisture": ["ps_wet_top", "ps_wet_root", "ps_wet_prof",
                      "ps_wet_top_lag1", "ps_wet_top_lag7",
                      "ps_wet_root_lag1", "ps_wet_root_lag7",
                      "ps_wet_prof_lag1", "ps_wet_prof_lag7",
                      "ps_wet_top_chg1d", "ps_wet_top_chg7d",
                      "ps_wet_top_7d", "ps_wet_top_14d", "ps_wet_top_30d",
                      "ps_wet_root_7d",
                      "ps_wet_top_clim", "ps_wet_top_anom",
                      "ps_wet_root_clim", "ps_wet_root_anom"],
    "glofas": ["gf_up_ratio", "gf_up_rise3d",
               "gf_up_ratio_lag1", "gf_up_ratio_lag2", "gf_up_ratio_lag3",
               "gf_up_ratio_lag5", "gf_up_ratio_lag7", "gf_up_ratio_lag10", "gf_up_ratio_lag14",
               "gf_up_ratio_3d", "gf_up_ratio_7d", "gf_up_ratio_14d",
               "gf_up_rise3d_lag1", "gf_up_rise3d_lag3",
               "gf_up_accel",
               "gf_near_dis", "gf_near_dis_lag1", "gf_near_dis_lag3"],
    "interactions": ["ix_wet_top_x_slope", "ix_wet_top_x_updis",
                     "ix_updis_x_wlrise", "ix_api09_x_wet_top",
                     "ix_updis14_x_api"],
}

__all__ = ["FULL_V2", "FEATURE_GROUPS", "ROOT", "SYSTEM",
           "RAW_WL_DIR", "RAW_RAIN", "STATIONS_FILE",
           "SPATIAL_FILE", "ERA5_FILE", "GLOFAS_DIR", "POWER_SOIL_FILE",
           "PROC_DIR", "MODEL_READY_DIR", "FEATURES_PARQUET",
           "MIN_READINGS_PER_DAY", "WL_MAX_PHYSICAL_M",
           "NEIGHB_KS", "NEIGHB_POOL", "UPSTREAM_MAX_KM", "UPSTREAM_LAGS",
           "BASIN_ANCHORS", "DEFAULT_ANCHORS"]
