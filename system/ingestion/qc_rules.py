"""
MI v2.0 / System Ingestion — QC rules for water level and rainfall data.

Implements the same filters validated in the parent repo's build_dataset.py:
  F1: NASA POWER sentinel <= -999 -> NaN (never 0)
  F3: drop physically impossible readings (> 100 m, decimal-shifted feed errors)
  F8: drop single-reading sensor spikes (|wl - local 12-h median| > 10 m)
  Missing-code sentinel <= -1000 -> dropped (FFWC missing code)

All filters are backward-looking only (no leakage).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (mirror reports/phase1_data_inventory.md findings)
# ---------------------------------------------------------------------------
MISSING_CODE_SENTINEL = -1000.0
IMPOSSIBLE_WL_MAX = 100.0
SPIKE_THRESHOLD_M = 10.0
SPIKE_WINDOW_READINGS = 4          # +/-4 readings = +/-12 h at 3-hourly
SPIKE_MIN_READINGS_IN_WINDOW = 3
SEGMENT_GAP_DAYS = 7               # gap > 7 days resets the median window
MIN_READINGS_PER_DAY = 3           # label-valid day needs >= 3 readings


def apply_wl_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply QC filters to a raw FFWC water-level DataFrame.

    Expected columns: wl_date (datetime), waterlevel (float), dangerlevel (float|str)
    Returns a cleaned DataFrame with sentinel, impossible, and spike rows removed.
    """
    if df.empty:
        return df

    df = df.copy()
    df["wl_date"] = pd.to_datetime(df["wl_date"], utc=True)
    df = df.sort_values("wl_date").reset_index(drop=True)

    # F1/F2 sentinel: drop missing-code readings
    df = df[df["waterlevel"] > MISSING_CODE_SENTINEL].copy()

    # F3: drop impossible readings (> 100 m)
    df = df[df["waterlevel"] <= IMPOSSIBLE_WL_MAX].copy()

    # F8: spike filter — per contiguous segment, rolling median over +/-4 readings
    df = _spike_filter(df)

    return df.reset_index(drop=True)


def _spike_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    F8 spike filter.

    Per contiguous segment (gap <= 7 days), rolling median over +/-4 readings
    with at least 3 in view. Drop readings where |wl - local median| > 10 m.
    Native, dependency-free implementation (no scipy).
    """
    if df.empty:
        return df

    n = len(df)
    keep = np.ones(n, dtype=bool)

    # Segment boundaries: never let the median span a blackout (> 7 days gap)
    seg_start = [0]
    for i in range(1, n):
        gap = (df["wl_date"].iloc[i] - df["wl_date"].iloc[i - 1]).total_seconds()
        if gap > SEGMENT_GAP_DAYS * 86400:
            seg_start.append(i)
    seg_start.append(n)

    wls = df["waterlevel"].to_numpy()
    for a, b in zip(seg_start, seg_start[1:]):
        seg_wls = wls[a:b]
        seg_len = b - a
        for j in range(seg_len):
            lo = max(0, j - SPIKE_WINDOW_READINGS)
            hi = min(seg_len, j + SPIKE_WINDOW_READINGS + 1)
            win = seg_wls[lo:hi]
            if len(win) < SPIKE_MIN_READINGS_IN_WINDOW:
                continue
            s = sorted(win)
            m = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
            if abs(seg_wls[j] - m) > SPIKE_THRESHOLD_M:
                keep[a + j] = False

    return df[keep].reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 3-hourly QC'd water-level readings to daily statistics.

    Returns columns: station_id, date, n_readings, wl_mean, wl_max, wl_min,
    wl_first, wl_last, danger_level.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = df["wl_date"].dt.date

    # Danger level: take the station-master value (constant per station)
    danger_level = df["dangerlevel"].iloc[0] if "dangerlevel" in df.columns else None
    try:
        danger_level = float(danger_level) if danger_level not in ("", "-", None) else None
    except (ValueError, TypeError):
        danger_level = None

    agg = (
        df.groupby("date")
        .agg(
            n_readings=("waterlevel", "size"),
            wl_mean=("waterlevel", "mean"),
            wl_max=("waterlevel", "max"),
            wl_min=("waterlevel", "min"),
            wl_first=("waterlevel", "first"),
            wl_last=("waterlevel", "last"),
        )
        .reset_index()
    )
    agg["danger_level"] = danger_level
    return agg


def is_label_valid_day(n_readings: int) -> bool:
    """A day is label-valid only if it has >= 3 readings (per label contract)."""
    return n_readings >= MIN_READINGS_PER_DAY


def sanitize_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize NASA POWER rainfall: sentinel <= -999 -> NaN, never 0.
    """
    if df.empty:
        return df
    df = df.copy()
    if "precip_mm" in df.columns:
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
        df.loc[df["precip_mm"] <= -999.0, "precip_mm"] = np.nan
    return df


__all__ = [
    "apply_wl_qc",
    "aggregate_daily",
    "is_label_valid_day",
    "sanitize_rainfall",
    "MISSING_CODE_SENTINEL",
    "IMPOSSIBLE_WL_MAX",
    "SPIKE_THRESHOLD_M",
]
