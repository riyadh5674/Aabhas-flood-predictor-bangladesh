"""
System Features — Hydrology block.

Computes water-level derived features from QC'd daily water-level data:
  n_readings, wl_mean/max/min/first/last, wl_last_lag1-7/10/14/21/30,
  wl_max_lag1-3, wl_rise_1d/3d, wl_dist_to_danger, wl_pct_danger,
  wl_max_roll7/30, wl_exceed_days_30

All shifts are backward-looking (day t uses only data up to day t).
No leakage.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from system.features.config import (
    MIN_READINGS_PER_DAY, WL_MAX_PHYSICAL_M, MISSING_CODE_SENTINEL,
    SPIKE_JUMP_M, SPIKE_WINDOW_READINGS, SPIKE_GAP_S,
)


def load_and_qc_waterlevels() -> pd.DataFrame:
    """Load all station-year CSVs, apply QC (sentinel, impossible, spike),
    return long-format 3-hourly readings.

    Output columns: station_id, wl_date, waterlevel, danger_level
    """
    from pathlib import Path

    from system.features.config import RAW_WL_DIR

    frames = []
    for f in sorted(RAW_WL_DIR.glob("*/*.csv")):
        d = pd.read_csv(f, usecols=["st_id", "wl_date", "waterlevel", "dangerlevel"])
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["station_id", "wl_date", "waterlevel", "danger_level"])

    raw = pd.concat(frames, ignore_index=True)
    raw.drop_duplicates(subset=["st_id", "wl_date"], inplace=True)

    raw["ts"] = pd.to_datetime(raw["wl_date"], format="%Y-%m-%dT%H:%M:%SZ",
                               utc=True, errors="coerce")
    raw["wl"] = pd.to_numeric(raw["waterlevel"], errors="coerce")
    raw["danger_level"] = pd.to_numeric(raw["dangerlevel"], errors="coerce")
    raw = raw.dropna(subset=["ts", "wl"])

    n0 = len(raw)
    sentinel = int((raw["wl"] <= MISSING_CODE_SENTINEL).sum())
    raw = raw[raw["wl"] > MISSING_CODE_SENTINEL]

    spikes = raw["wl"] > WL_MAX_PHYSICAL_M
    n_spikes = int(spikes.sum())
    if n_spikes:
        print(f"F3: dropped {n_spikes} physically impossible readings (> {WL_MAX_PHYSICAL_M} m)")
    raw = raw[raw["wl"] <= WL_MAX_PHYSICAL_M]

    raw.sort_values(["st_id", "ts"], inplace=True)

    # Spike filter (F8): per segment, rolling median over +/-4 readings
    gap = (raw.groupby("st_id")["ts"].diff().dt.total_seconds() > SPIKE_GAP_S).fillna(False)
    seg = gap.groupby(raw["st_id"]).cumsum()
    local_med = raw.groupby(["st_id", seg])["wl"].transform(
        lambda s: s.rolling(SPIKE_WINDOW_READINGS * 2 + 1, center=True, min_periods=3).median())
    raw["_dev8"] = (raw["wl"] - local_med).abs()
    spike8 = raw["_dev8"] > SPIKE_JUMP_M
    n_spike8 = int(spike8.sum())
    if n_spike8:
        print(f"F8: dropped {n_spike8} single-reading sensor spikes "
              f"(|wl - local median| > {SPIKE_JUMP_M} m)")
    raw = raw[~spike8].drop(columns=["_dev8"])

    raw["date"] = raw["ts"].dt.tz_convert(None).dt.date
    raw.rename(columns={"st_id": "station_id"}, inplace=True)

    print(f"water level readings: {len(raw):,} (from {n0:,})  "
          f"stations={raw.station_id.nunique()}  "
          f"{raw['date'].min()}..{raw['date'].max()}")

    if sentinel:
        print(f"  dropped {sentinel} sentinel water levels (<= {MISSING_CODE_SENTINEL}, missing-code)")

    return raw[["station_id", "wl_date", "waterlevel", "danger_level", "date"]]


def build_daily(raw: pd.DataFrame, master_dl: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 3-hourly readings to daily stats per station.

    Output columns: station_id, date, n_readings, wl_mean, wl_max, wl_min,
    wl_first, wl_last, danger_level, day_observed, exceed
    """
    if raw.empty:
        return pd.DataFrame()

    g = raw.groupby(["station_id", "date"])
    daily = g.agg(
        n_readings=("waterlevel", "size"),
        wl_mean=("waterlevel", "mean"),
        wl_max=("waterlevel", "max"),
        wl_min=("waterlevel", "min"),
        wl_first=("waterlevel", "first"),
        wl_last=("waterlevel", "last"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_observed"] = daily["n_readings"] >= MIN_READINGS_PER_DAY

    # Override per-reading dangerlevel with authoritative master value
    daily = daily.merge(master_dl, on="station_id", how="left")

    daily["exceed"] = np.where(
        daily["day_observed"] & daily["danger_level"].notna(),
        (daily["wl_max"] >= daily["danger_level"]).astype("float"),
        np.nan,
    )

    print(f"station-days: {len(daily):,}  observed (>=3 readings): {daily.day_observed.mean():.1%}")
    return daily


def add_hydrology_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add water-level derived features (lags, rolls, rise, pct_danger, etc.).

    Input must be sorted by [station_id, date].
    Output adds columns: wl_last_lag1-7/10/14/21/30, wl_max_lag1-3,
    wl_rise_1d/3d, wl_dist_to_danger, wl_pct_danger, wl_max_roll7/30,
    wl_exceed_days_30
    """
    df = daily.sort_values(["station_id", "date"]).copy()

    g = df.groupby("station_id")

    # lags 1-7
    for k in range(1, 8):
        df[f"wl_last_lag{k}"] = g["wl_last"].shift(k)
    # lags 10, 14, 21, 30 (V2_NEW)
    for k in (10, 14, 21, 30):
        df[f"wl_last_lag{k}"] = g["wl_last"].shift(k)
    # wl_max lags 1-3
    for k in (1, 2, 3):
        df[f"wl_max_lag{k}"] = g["wl_max"].shift(k)

    # rise
    df["wl_rise_1d"] = df["wl_last"] - g["wl_last"].shift(1)
    df["wl_rise_3d"] = df["wl_last"] - g["wl_last"].shift(3)

    # distance / percent to danger
    df["wl_dist_to_danger"] = df["danger_level"] - df["wl_max"]
    df["wl_pct_danger"] = df["wl_max"] / df["danger_level"]

    # rolling windows
    df["wl_max_roll7"] = g["wl_max"].transform(lambda s: s.rolling(7, min_periods=4).mean())
    df["wl_max_roll30"] = g["wl_max"].transform(lambda s: s.rolling(30, min_periods=10).mean())
    df["wl_exceed_days_30"] = g["exceed"].transform(
        lambda s: s.fillna(0).rolling(30, min_periods=1).sum())

    return df


__all__ = ["load_and_qc_waterlevels", "build_daily", "add_hydrology_features"]
