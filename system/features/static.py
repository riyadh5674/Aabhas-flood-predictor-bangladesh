"""
System Features — Static / reference block.

Joins per-station static geographic features, soil moisture (ERA5-Land),
and GloFAS transboundary discharge features. All reference data is read-only
from the parent repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from system.features.config import (
    SPATIAL_FILE, ERA5_FILE, GLOFAS_DIR, POWER_SOIL_FILE,
    SPATIAL_COLS, UPSTREAM_MAX_KM, UPSTREAM_ELEV_DIFF_M,
    UPSTREAM_LAGS, BASIN_ANCHORS as _BASIN_ANCHORS, DEFAULT_ANCHORS,
)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_static_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Join per-station static geographic features (station_spatial_features.csv).

    Every modelling station must resolve; missing stations raise a warning.
    """
    static = pd.read_csv(SPATIAL_FILE)[["station_id"] + SPATIAL_COLS]
    n0 = df["station_id"].nunique()
    out = df.merge(static, on="station_id", how="left")
    missing = n0 - out["station_id"].nunique()
    na = {c: int(out[c].isna().sum()) for c in SPATIAL_COLS}
    if missing:
        print(f"WARN add_static_spatial: {missing} stations absent from spatial features")
    print(f"static spatial joined for {out['station_id'].nunique()} stations; "
          f"NaN cells per column = {na}")
    return out


def _haversine_series(lat1, lon1, lat2, lon2) -> pd.Series:
    return pd.Series(_haversine_km(lat1.values, lon1.values, lat2.values, lon2.values),
                     index=lat1.index)


def load_soil_moisture() -> pd.DataFrame:
    """Load NASA POWER soil wetness (power_soil_daily.csv).

    Output columns: station_id, date, ps_wet_top, ps_wet_root, ps_wet_prof
    """
    p = POWER_SOIL_FILE
    if not p.exists():
        print(f"WARN: {p} not found — soil moisture features will be NaN")
        return pd.DataFrame(columns=["station_id", "date",
                                     "ps_wet_top", "ps_wet_root", "ps_wet_prof"])

    ps = pd.read_csv(p, parse_dates=["date"])
    ps = ps.rename(columns={"ps_top": "ps_wet_top", "ps_root": "ps_wet_root",
                            "ps_prof": "ps_wet_prof"})
    return ps[["station_id", "date", "ps_wet_top", "ps_wet_root", "ps_wet_prof"]]


def add_soil_moisture_features(df: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """Join soil moisture and compute derived features (lags, changes, rolls,
    climatology, anomaly).

    Climatology built from <=2021 rows only (no leakage).
    """
    if sm.empty:
        for c in ("ps_wet_top", "ps_wet_root", "ps_wet_prof",
                  "ps_wet_top_lag1", "ps_wet_top_lag7",
                  "ps_wet_root_lag1", "ps_wet_root_lag7",
                  "ps_wet_prof_lag1", "ps_wet_prof_lag7",
                  "ps_wet_top_chg1d", "ps_wet_top_chg7d",
                  "ps_wet_top_7d", "ps_wet_top_14d", "ps_wet_top_30d",
                  "ps_wet_root_7d",
                  "ps_wet_top_clim", "ps_wet_top_anom",
                  "ps_wet_root_clim", "ps_wet_root_anom"):
            df[c] = np.nan
        return df

    sm = sm.sort_values(["station_id", "date"]).copy()

    # NASA POWER publishes with a ~3-day lag. Carry each station's last observed
    # value forward to the feature date so the model receives a defined "latest
    # available soil" reading instead of nulls. Interior gaps are not filled; only
    # the tail is extended. The source date is recorded in the quality report so
    # consumers can see how stale the reading is.
    target_max = pd.to_datetime(df["date"]).max()
    tail_rows = []
    carried = 0
    for sid, grp in sm.groupby("station_id"):
        valid = grp.dropna(
            subset=["ps_wet_top", "ps_wet_root", "ps_wet_prof"], how="all"
        )
        if valid.empty:
            continue
        last = valid.sort_values("date").iloc[-1]
        if last["date"] >= target_max:
            continue
        dates = pd.date_range(
            last["date"] + pd.Timedelta(days=1), target_max, freq="D"
        )
        if len(dates) == 0:
            continue
        block = pd.DataFrame({"station_id": sid, "date": dates})
        for c in ("ps_wet_top", "ps_wet_root", "ps_wet_prof"):
            block[c] = last[c]
        tail_rows.append(block)
        carried += 1
    if tail_rows:
        sm = pd.concat([sm, *tail_rows], ignore_index=True).sort_values(
            ["station_id", "date"]
        )
        print(
            f"soil: carried last observed value forward for {carried} station(s) "
            f"to {target_max.date()}"
        )

    g = sm.groupby("station_id")

    for c in ("ps_wet_top", "ps_wet_root", "ps_wet_prof"):
        sm[f"{c}_lag1"] = g[c].shift(1)
        sm[f"{c}_lag7"] = g[c].shift(7)

    sm["ps_wet_top_chg1d"] = sm["ps_wet_top"] - sm["ps_wet_top_lag1"]
    sm["ps_wet_top_chg7d"] = sm["ps_wet_top"] - sm["ps_wet_top_lag7"]
    sm["ps_wet_top_7d"] = g["ps_wet_top"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    sm["ps_wet_top_14d"] = g["ps_wet_top"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    sm["ps_wet_top_30d"] = g["ps_wet_top"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    sm["ps_wet_root_7d"] = g["ps_wet_root"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    # climatology + anomaly (<=2021 only)
    sm["doy"] = sm["date"].dt.dayofyear
    tr = sm[sm["date"] < "2022-01-01"]
    for c in ("ps_wet_top", "ps_wet_root"):
        clim = tr.groupby(["station_id", "doy"])[c].mean().rename(f"{c}_clim")
        sm = sm.merge(clim, on=["station_id", "doy"], how="left")
        sm[f"{c}_anom"] = sm[c] - sm[f"{c}_clim"]

    sm = sm.drop(columns=["doy"])

    keep = ["station_id", "date",
            "ps_wet_top", "ps_wet_root", "ps_wet_prof",
            "ps_wet_top_lag1", "ps_wet_top_lag7",
            "ps_wet_root_lag1", "ps_wet_root_lag7",
            "ps_wet_prof_lag1", "ps_wet_prof_lag7",
            "ps_wet_top_chg1d", "ps_wet_top_chg7d",
            "ps_wet_top_7d", "ps_wet_top_14d", "ps_wet_top_30d",
            "ps_wet_root_7d",
            "ps_wet_top_clim", "ps_wet_top_anom",
            "ps_wet_root_clim", "ps_wet_root_anom"]
    sm = sm[keep]

    out = df.merge(sm, on=["station_id", "date"], how="left")
    return out


def load_glofas(st_meta: pd.DataFrame) -> pd.DataFrame:
    """Load GloFAS anchor discharge and compute per-basin features.

    Output columns: station_id, date, gf_* features
    """
    gl_files = list(GLOFAS_DIR.glob("discharge_anchors_daily.csv"))
    if not gl_files:
        print(f"WARN: no GloFAS discharge file in {GLOFAS_DIR}")
        return pd.DataFrame(columns=["station_id", "date"])
    gl = pd.read_csv(gl_files[0], parse_dates=["date"]).sort_values(["anchor", "date"])

    # climatology per anchor (<=2021 only)
    gl["doy"] = gl["date"].dt.dayofyear
    clim = (gl[gl["date"] < "2022-01-01"]
            .groupby(["anchor", "doy"])["river_discharge"]
            .mean().rename("clim"))
    gl = gl.merge(clim, on=["anchor", "doy"], how="left")
    gl["dis_ratio"] = gl["river_discharge"] / gl["clim"].replace(0, np.nan)
    gl["rise3d"] = gl.groupby("anchor")["river_discharge"].transform(lambda s: s - s.shift(3))
    gl = gl.sort_values(["anchor", "date"])

    present = set(gl.anchor.unique())

    def basin_idx(anchors):
        sub = gl[gl.anchor.isin([a for a in anchors if a in present])]
        return sub.groupby("date")[["dis_ratio", "rise3d"]].mean().rename(
            columns={"dis_ratio": "gf_up_ratio", "rise3d": "gf_up_rise3d"})

    B = {b: basin_idx(a) for b, a in _BASIN_ANCHORS.items()}
    B["_default"] = basin_idx(DEFAULT_ANCHORS)
    Bdf = pd.concat(B, names=["bkey", "date"]).reset_index().sort_values(["bkey", "date"])

    gb = Bdf.groupby("bkey")
    for k in (1, 2, 3, 5, 7, 10, 14):
        Bdf[f"gf_up_ratio_lag{k}"] = gb["gf_up_ratio"].shift(k)
    for w in (3, 7, 14):
        Bdf[f"gf_up_ratio_{w}d"] = gb["gf_up_ratio"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    Bdf["gf_up_rise3d_lag1"] = gb["gf_up_rise3d"].shift(1)
    Bdf["gf_up_rise3d_lag3"] = gb["gf_up_rise3d"].shift(3)
    Bdf["gf_up_accel"] = Bdf["gf_up_rise3d"] - Bdf["gf_up_rise3d_lag3"]

    # chosen cells for near-anchor distance
    cells_file = GLOFAS_DIR / "chosen_cells.json"
    if cells_file.exists():
        cells = json.loads(cells_file.read_text(encoding="utf-8"))
        anch = pd.DataFrame([{"anchor": k, "lat": v["lat"], "lon": v["lon"]}
                             for k, v in cells.items()])
    else:
        anch = pd.DataFrame(columns=["anchor", "lat", "lon"])

    # station → basin mapping
    st = st_meta.drop_duplicates("station_id")[["station_id", "lat", "lon", "basin"]].copy()
    st["bkey"] = np.where(st["basin"].isin(_BASIN_ANCHORS), st["basin"], "_default")
    if not anch.empty:
        st["near_anchor"] = [
            anch.iloc[int(np.argmin(
                _haversine_km(r.lat, r.lon, anch.lat.to_numpy(), anch.lon.to_numpy()
            )))].anchor
            for r in st.itertuples()]
    else:
        st["near_anchor"] = np.nan

    near_df = gl[["anchor", "date", "river_discharge"]].copy()
    near_df = near_df.rename(columns={"anchor": "near_anchor",
                                      "river_discharge": "gf_near_dis"})
    near_df["gf_near_dis_lag1"] = near_df.groupby("near_anchor")["gf_near_dis"].shift(1)
    near_df["gf_near_dis_lag3"] = near_df.groupby("near_anchor")["gf_near_dis"].shift(3)

    out = st.merge(Bdf, on="bkey", how="left").merge(near_df, on=["near_anchor", "date"], how="left")

    keep = ["station_id", "date", "gf_up_ratio", "gf_up_rise3d",
            "gf_up_ratio_lag1", "gf_up_ratio_lag2", "gf_up_ratio_lag3",
            "gf_up_ratio_lag5", "gf_up_ratio_lag7", "gf_up_ratio_lag10", "gf_up_ratio_lag14",
            "gf_up_ratio_3d", "gf_up_ratio_7d", "gf_up_ratio_14d",
            "gf_up_rise3d_lag1", "gf_up_rise3d_lag3",
            "gf_up_accel",
            "gf_near_dis", "gf_near_dis_lag1", "gf_near_dis_lag3"]
    return out[keep]


__all__ = ["add_static_spatial", "load_soil_moisture", "add_soil_moisture_features",
           "load_glofas"]
