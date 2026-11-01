"""
System Features — Spatial block.

Computes nearest-neighbour (nb3/nb5) and upstream features from QC'd daily data.
All neighbour features use same-day (day t) values of nearby gauges — known by the
issue instant, so no leakage. Upstream features use lag1/lag2 of upstream gauge state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from system.features.config import (
    NEIGHB_KS, NEIGHB_SIGS, NEIGHB_POOL, UPSTREAM_MAX_KM, UPSTREAM_ELEV_DIFF_M,
    UPSTREAM_LAGS,
)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest-k neighbour features (nb3/nb5).

    For each station-day: mean over the nearest-k gauges that actually reported a
    value that day. Resolved per day — if fewer than k reported, use those that did.
    Same-day values are known by the issue instant -> no leakage.

    Adds: nb3_rain_7d, nb3_rain_14d, nb3_wl_pct_danger,
          nb5_rain_7d, nb5_rain_14d, nb5_wl_pct_danger,
          nb3_n_wl_danger, nb5_n_wl_danger
    """
    stcol = df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    latv = stcol["lat"].to_numpy()
    lonv = stcol["lon"].to_numpy()
    ids = stcol["station_id"].to_numpy().astype("int64")

    # SxS distance -> per-gauge ranked pool (nearest-first, self=inf)
    D = np.empty((len(ids), len(ids)), dtype="float")
    for i in range(len(ids)):
        D[i] = _haversine_km(latv[i], lonv[i], latv, lonv)
    np.fill_diagonal(D, np.inf)
    pools = np.argsort(D, axis=1)[:, :NEIGHB_POOL]

    def resolve(pivot: np.ndarray, has: np.ndarray):
        """For each station-day: mean over nearest-k observed values.

        pivot: (T, S) matrix, has: (T, S) boolean mask
        Returns dict k -> (T, S) means matrix
        """
        means = {}
        for k in NEIGHB_KS:
            means[k] = np.full(pivot.shape, np.nan)

        for i in range(pivot.shape[1]):
            cols = pools[i]
            sub = pivot[:, cols]
            sub_has = has[:, cols]
            cum = sub_has.cumsum(axis=1)
            for k in NEIGHB_KS:
                ok_everywhere = cum[:, -1] >= k
                kth = np.full(pivot.shape[0], cols.shape[0], dtype="int64")
                kth[ok_everywhere] = np.argmax(cum[ok_everywhere] >= k, axis=1)
                sel = np.arange(cols.shape[0])[None, :] <= kth[:, None]
                use = sel & sub_has
                cnt = use.sum(axis=1)
                wsum = np.nansum(np.where(use, sub, 0.0), axis=1)
                means[k][:, i] = np.where(cnt > 0, wsum / np.maximum(cnt, 1), np.nan)
        return means

    def stack_means(means_dict, colname, index, columns):
        long_df = pd.DataFrame(means_dict[NEIGHB_KS[0]], index=index, columns=columns)
        # Actually we need to stack for each k separately
        frames = []
        for k in NEIGHB_KS:
            long = pd.DataFrame(means_dict[k], index=index, columns=columns).stack()
            long.name = f"nb{k}_{colname}"
            frames.append(long.reset_index())
        return frames

    # Rain neighbour signals (from weather record)
    wxframe = df[["station_id", "date", "rain_7d", "rain_14d"]].copy()
    pivot7 = wxframe.pivot_table(index="date", columns="station_id", values="rain_7d", dropna=False)
    pivot7 = pivot7.reindex(columns=ids)
    pivot14 = wxframe.pivot_table(index="date", columns="station_id", values="rain_14d", dropna=False)
    pivot14 = pivot14.reindex(columns=ids)

    means7 = resolve(pivot7.to_numpy(dtype="float"), ~pivot7.isna().to_numpy())
    means14 = resolve(pivot14.to_numpy(dtype="float"), ~pivot14.isna().to_numpy())

    for k in NEIGHB_KS:
        long = pd.DataFrame(means7[k], index=pivot7.index, columns=pivot7.columns).stack()
        long.name = f"nb{k}_rain_7d"
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left",
                      validate="one_to_one")
        long = pd.DataFrame(means14[k], index=pivot14.index, columns=pivot14.columns).stack()
        long.name = f"nb{k}_rain_14d"
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left",
                      validate="one_to_one")

    # Neighbour water-level state (wl_pct_danger)
    pv = df.pivot_table(index="date", columns="station_id", values="wl_pct_danger", dropna=False)
    pv = pv.reindex(columns=ids)
    pvv = pv.to_numpy(dtype="float")
    means_wl = resolve(pvv, ~np.isnan(pvv))
    for k in NEIGHB_KS:
        long = pd.DataFrame(means_wl[k], index=pv.index, columns=pv.columns).stack()
        long.name = f"nb{k}_wl_pct_danger"
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left",
                      validate="one_to_one")

    # Count of neighbours at/above danger level
    dng = (pvv >= 1.0).astype("float")
    dng[np.isnan(pvv)] = np.nan
    has_dng = ~np.isnan(dng)
    for k in NEIGHB_KS:
        res_k = np.zeros(pvv.shape, dtype="int64")
        for i in range(pvv.shape[1]):
            cols = pools[i]
            sub_has = has_dng[:, cols]
            cum = sub_has.cumsum(axis=1)
            ok_everywhere = cum[:, -1] >= k
            kth = np.full(pvv.shape[0], cols.shape[0], dtype="int64")
            kth[ok_everywhere] = np.argmax(cum[ok_everywhere] >= k, axis=1)
            sel = np.arange(cols.shape[0])[None, :] <= kth[:, None]
            use = sel & sub_has
            res_k[:, i] = np.nansum(np.where(use, dng[:, cols], 0.0), axis=1)
        long = pd.DataFrame(res_k, index=pv.index, columns=pv.columns).stack()
        long.name = f"nb{k}_n_wl_danger"
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left",
                      validate="one_to_one")

    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    return df


def add_upstream(df: pd.DataFrame) -> pd.DataFrame:
    """Add upstream dwell-time features.

    Upstream = nearest same-river gauge with clearly higher elevation within
    UPSTREAM_MAX_KM. Its rain_7d / wl_pct_danger shifted by 1 and 2 days.

    Adds: up_rain7d_lag1/2, up_wl_pct_danger_lag1/2
    """
    topo = df[["station_id", "river", "lat", "lon", "elevation_m"]].drop_duplicates("station_id")
    up_map: dict[int, int] = {}

    for _, g in topo.groupby("river"):
        g = g.dropna(subset=["elevation_m"])
        if len(g) < 2:
            continue
        for _, s in g.iterrows():
            higher = g[g["elevation_m"] > s["elevation_m"] + UPSTREAM_ELEV_DIFF_M]
            if higher.empty:
                continue
            d = _haversine_km(s["lat"], s["lon"], higher["lat"].to_numpy(), higher["lon"].to_numpy())
            j = int(np.argmin(d))
            if d[j] <= UPSTREAM_MAX_KM:
                up_map[int(s["station_id"])] = int(higher["station_id"].iloc[j])

    n_mapped = len(up_map)
    if not up_map:
        print("WARN add_upstream: no upstream relations found")
        for k in UPSTREAM_LAGS:
            df[f"up_rain7d_lag{k}"] = np.nan
            df[f"up_wl_pct_danger_lag{k}"] = np.nan
        return df

    usrc = df[["station_id", "date", "rain_7d", "wl_pct_danger"]].copy()
    for k in UPSTREAM_LAGS:
        usrc[f"up_rain7d_lag{k}"] = usrc.groupby("station_id")["rain_7d"].shift(k)
        usrc[f"up_wl_pct_danger_lag{k}"] = usrc.groupby("station_id")["wl_pct_danger"].shift(k)

    usrc = usrc[["station_id", "date",
                 *[f"up_rain7d_lag{k}" for k in UPSTREAM_LAGS],
                 *[f"up_wl_pct_danger_lag{k}" for k in UPSTREAM_LAGS]]]

    df["_up_id"] = df["station_id"].map(up_map)
    df = df.merge(usrc, left_on=["_up_id", "date"], right_on=["station_id", "date"],
                  how="left", suffixes=("", "_usrc"))

    usrc_cols = [f"up_rain7d_lag{k}" for k in UPSTREAM_LAGS] + \
                [f"up_wl_pct_danger_lag{k}" for k in UPSTREAM_LAGS]
    for c in usrc_cols:
        if c + "_usrc" in df.columns:
            df[c] = df[c + "_usrc"]

    keepcol = [c for c in df.columns if c.endswith("_usrc")]
    df.drop(columns=["_up_id", *keepcol], inplace=True)

    print(f"upstream map: {n_mapped} downstream stations with upstream gauge "
          f"within {UPSTREAM_MAX_KM:g} km")
    for c in usrc_cols:
        print(f"    {c}: {int(df[c].isna().sum())} NaN")

    return df


__all__ = ["add_neighbourhood", "add_upstream"]
