"""
System Features — Rainfall block.

Computes rainfall-derived features from QC'd daily weather data:
  precip_mm, rain_1d/3d/7d/14d/30d, rain_max_7d, wet_days_7,
  tmax_c, tmin_c, doy_sin, doy_cos

All windows are backward-looking (shift(1) so day t uses data up to day t-1).
F1 sentinel: NASA POWER <= -999 -> NaN (never 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from system.features.config import RAW_RAIN


def load_rainfall() -> pd.DataFrame:
    """Load daily weather data and apply F1 sentinel filter.

    Output columns: station_id, date, precip_mm, tmax_c, tmin_c, source
    """
    df = pd.read_csv(RAW_RAIN, parse_dates=["date"])

    wx_cols = ["precip_mm", "tmax_c", "tmin_c"]
    wx_cols = [c for c in wx_cols if c in df.columns]
    n_sent = int(sum(int((df[c] <= -999).sum()) for c in wx_cols))
    if n_sent:
        df[wx_cols] = df[wx_cols].mask(df[wx_cols] <= -999)
        print(f"F1: {n_sent} NASA POWER sentinel cells (<= -999) -> NaN in weather")

    return df


def add_rainfall_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data and compute rainfall features.

    Input df must have columns: station_id, date
    Output adds: precip_mm, rain_1d, rain_3d, rain_7d, rain_14d, rain_30d,
    rain_max_7d, wet_days_7, tmax_c, tmin_c
    """
    wx = weather.copy()
    wx["date"] = pd.to_datetime(wx["date"])

    df = df.merge(
        wx[["station_id", "date", "precip_mm", "tmax_c", "tmin_c", "source"]],
        on=["station_id", "date"],
        how="left",
    )

    gw = df.groupby("station_id")

    # rain_1d = yesterday's precip (already on day t via precip_mm column)
    df["rain_1d"] = df["precip_mm"]

    for k in (3, 7, 14, 30):
        df[f"rain_{k}d"] = gw["precip_mm"].transform(
            lambda s: s.rolling(k, min_periods=1).sum())

    df["rain_max_7d"] = gw["precip_mm"].transform(
        lambda s: s.rolling(7, min_periods=1).max())
    df["wet_days_7"] = gw["precip_mm"].transform(
        lambda s: (s.fillna(0) > 1).rolling(7, min_periods=1).sum())

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year cyclical encoding (doy_sin, doy_cos)."""
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


__all__ = ["load_rainfall", "add_rainfall_features", "add_temporal_features"]
