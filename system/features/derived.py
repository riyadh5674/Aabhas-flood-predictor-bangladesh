"""
System Features — Derived block.

Computes V2_NEW derived features from existing columns:
  wl_last_lag10/14/21/30, api_09/api_07 (recursive precipitation index),
  interaction features (ix_*)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from system.features.config import FULL_V2


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wl_last long lags and recursive precipitation index.

    Input must be sorted by [station_id, date].
    Adds: wl_last_lag10/14/21/30, api_09/api_07
    """
    k = df.sort_values(["station_id", "date"]).copy()
    g = k.groupby("station_id")

    for L in (10, 14, 21, 30):
        k[f"wl_last_lag{L}"] = g["wl_last"].shift(L)

    for decay, name in ((0.9, "api_09"), (0.7, "api_07")):
        k[name] = g["precip_mm"].transform(
            lambda s: lfilter([1.0], [1.0, -decay],
                              np.nan_to_num(s.to_numpy(dtype="float64"))))

    keep = ["station_id", "date", "wl_last_lag10", "wl_last_lag14",
            "wl_last_lag21", "wl_last_lag30", "api_09", "api_07"]
    return k[keep]


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features (V2_NEW ix_*).

    Adds: ix_wet_top_x_slope, ix_wet_top_x_updis,
          ix_updis_x_wlrise, ix_api09_x_wet_top, ix_updis14_x_api
    """
    out = df.copy()
    out["ix_wet_top_x_slope"] = out["ps_wet_top"] * out["slope_deg"]
    out["ix_wet_top_x_updis"] = out["ps_wet_top"] * out["gf_up_ratio_lag1"]
    out["ix_updis_x_wlrise"] = out["gf_up_ratio_lag1"] * (out["wl_last"] - out["wl_max"])
    out["ix_api09_x_wet_top"] = out["api_09"] * out["ps_wet_top"]
    out["ix_updis14_x_api"] = out["gf_up_ratio_14d"] * out["api_09"]
    return out


__all__ = ["add_derived_features", "add_interactions"]
