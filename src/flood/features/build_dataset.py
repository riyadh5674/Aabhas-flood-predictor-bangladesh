"""Build the final modelling table: one row per station-day with 24/48/72 h labels.

Inputs
    data/raw/ffwc/waterlevel/{station_id}/{year}.csv   (3-hourly observed water level)
    data/raw/ffwc/ffwc_stations.csv                    (station master, danger levels, geo)
    data/raw/rainfall/weather_station_daily.csv        (daily precipitation / temperature)
    data/processed/station_spatial_features.csv        (Phase 11 static geo: elevation, slope, distances)

Output
    data/processed/flood_dataset.csv        the dataset to model on
    data/processed/data_dictionary.csv      column -> meaning

Label rule (no assumed zeros):
    y_h = 1 if any observed reading in (day t, day t+h] reaches the danger level
    y_h = 0 if every day in that window has >=3 readings and none reaches it
    y_h = NaN otherwise (window not fully observed -> dropped at training time)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
WL = ROOT / "data" / "raw" / "ffwc" / "waterlevel"
STATIONS = ROOT / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
WEATHER = ROOT / "data" / "raw" / "rainfall" / "weather_station_daily.csv"
SPATIAL = ROOT / "data" / "processed" / "station_spatial_features.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT = OUT_DIR / "flood_dataset.csv"
MIN_READINGS_PER_DAY = 3
LEADS = {"label_24h": 1, "label_48h": 2, "label_72h": 3}
# Phase 11 spatial feature engineering (reports/phase1_data_inventory.md D7):
#  * static per-station geo from SRTM + OSM + geoBoundaries (station_spatial_features.csv)
#  * nearest-neighbour regional context: mean of the k nearest matrix gauges' rain and
#    water-level state on the SAME day t (all known by the issue instant -> no leakage)
#  * upstream propagation: on multi-station rivers the nearest same-river gauge at HIGHER
#    elevation (lat ordering is wrong for the west-flowing Ganges/Padma systems) within
#    UPSTREAM_MAX_KM, lagged 1-2 days (strictly <= day t-1)
NEIGHB_KS = (3, 5)
NEIGHB_SIGS = ("rain_7d", "rain_14d", "wl_pct_danger")
NEIGHB_POOL = 15  # each gauge's candidate pool = nearest POOL gauges; the nearest-k that actually reported are used
UPSTREAM_MAX_KM = 250.0
UPSTREAM_ELEV_DIFF_M = 0.5  # must be clearly higher, else the "nearest upstream" is ambiguous
UPSTREAM_LAGS = (1, 2)
SPATIAL_COLS = ["elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km",
                "dist_major_river_km", "river_density_10km_km_per_km2", "n_river_lines_10km"]
# FFWC master spells the same division two ways (api_master vs legacy catalog)
DIVISION_NORM = {"Chittagong": "Chattogram"}
# F3 (reports/phase2_3_data_quality.md): the FFWC feed contains decimal-shifted readings
# (1213.00 m, 1315.13 m).  No gauge in Bangladesh can physically exceed the highest
# published danger level (70.3 m); 100 m is a generous impossible-value bound.
WL_MAX_PHYSICAL_M = 100.0
# F8 (reports/phase2_3_data_quality.md): single-reading sensor spikes that sit BELOW the F3
# bound (Chuadanga 38.84 m beside 3.85 m -> false-positive labels; Panchagarh 0.65 m beside
# 64.75 m; 0.00 m year-boundary codes).  No river stage in Bangladesh changes by more than
# SPIKE_JUMP_M within a +/-12 h neighbourhood, so a reading that deviates from its local
# median is a sensor artefact, not water.
SPIKE_JUMP_M = 10.0
SPIKE_READINGS = 9          # +/- 4 readings = +/- 12 h at the nominal 3-hourly cadence
SPIKE_GAP_S = 7 * 86400     # never let the local median span a data blackout
_MASTER_DL: pd.DataFrame = None


def load_waterlevels() -> pd.DataFrame:
    frames = []
    for f in sorted(WL.glob("*/*.csv")):
        d = pd.read_csv(f, usecols=["st_id", "wl_date", "waterlevel", "dangerlevel"])
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw.drop_duplicates(subset=["st_id", "wl_date"], inplace=True)
    n0 = len(raw)
    raw["ts"] = pd.to_datetime(raw["wl_date"], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
    raw["wl"] = pd.to_numeric(raw["waterlevel"], errors="coerce")
    raw["danger_level"] = pd.to_numeric(raw["dangerlevel"], errors="coerce")
    raw = raw.dropna(subset=["ts", "wl"])
    sentinel = (raw["wl"] <= -1000).sum()
    spikes = raw["wl"] > WL_MAX_PHYSICAL_M
    n_spikes = int(spikes.sum())
    if n_spikes:
        hit = raw.loc[spikes].groupby("st_id").agg(
            n=("wl", "size"), worst=("wl", "max"))
        print(f"F3: dropped {n_spikes} physically impossible readings "
              f"(> {WL_MAX_PHYSICAL_M} m, decimal-shifted feed errors):")
        print(hit.sort_values("worst", ascending=False).to_string())
    raw = raw[(raw["wl"] > -1000) & (raw["wl"] <= WL_MAX_PHYSICAL_M)]
    raw.sort_values(["st_id", "ts"], inplace=True)
    gap = (raw.groupby("st_id")["ts"].diff().dt.total_seconds() > SPIKE_GAP_S).fillna(False)
    seg = gap.groupby(raw["st_id"]).cumsum()
    local_med = raw.groupby(["st_id", seg])["wl"].transform(
        lambda s: s.rolling(SPIKE_READINGS, center=True, min_periods=3).median())
    raw["_dev8"] = (raw["wl"] - local_med).abs()
    spike8 = raw["_dev8"] > SPIKE_JUMP_M
    n_spike8 = int(spike8.sum())
    if n_spike8:
        hit = raw.loc[spike8].groupby("st_id").agg(n=("wl", "size"), worst_dev=("_dev8", "max"))
        print(f"F8: dropped {n_spike8} single-reading sensor spikes "
              f"(|wl - local 12-h median| > {SPIKE_JUMP_M} m):")
        print(hit.sort_values("worst_dev", ascending=False).to_string())
        raw = raw[~spike8]
    raw.drop(columns=["_dev8"], inplace=True)
    raw["date"] = raw["ts"].dt.tz_convert(None).dt.date
    if sentinel:
        print(f"  dropped {sentinel} sentinel water levels (<= -1000, FFWC missing-code)")
    print(f"water level readings: {len(raw):,} (from {n0:,})  stations={raw.st_id.nunique()}  "
          f"{raw['date'].min()}..{raw['date'].max()}")
    return raw


def build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["st_id", "date"])
    daily = g.agg(
        n_readings=("wl", "size"),
        wl_mean=("wl", "mean"),
        wl_max=("wl", "max"),
        wl_min=("wl", "min"),
        wl_first=("wl", "first"),
        wl_last=("wl", "last"),
        danger_level=("danger_level", "first"),
    ).reset_index()
    daily.rename(columns={"st_id": "station_id"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_observed"] = daily["n_readings"] >= MIN_READINGS_PER_DAY
    # The published danger level is the ONLY valid threshold: it is station-constant
    # (A7 verified 0 stations change it) and it comes from the official station master.
    # Per-reading 'dangerlevel' fields in the raw CSVs are unreliable (often '-') and can
    # contain stray values, so we overwrite them with the authoritative master value.
    daily = daily.drop(columns=["danger_level"])
    daily = daily.merge(
        _MASTER_DL, on="station_id", how="left"
    )  # stations without an official DL keep NaN -> labels NaN (never 0)
    daily["exceed"] = np.where(
        daily["day_observed"] & daily["danger_level"].notna(),
        (daily["wl_max"] >= daily["danger_level"]).astype("float"),
        np.nan,
    )
    print(f"station-days: {len(daily):,}  observed (>=3 readings): {daily.day_observed.mean():.1%}")
    return daily


def add_labels(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sid, d in daily.groupby("station_id"):
        d = d.sort_values("date").set_index("date")
        full = pd.date_range(d.index.min(), d.index.max(), freq="D")
        d = d.reindex(full)
        d["danger_level"] = d["danger_level"].ffill().bfill()  # station-constant
        d.index.name = "date"
        n_obs = d["n_readings"].fillna(0).astype(int)
        ex = d["exceed"]
        for col, h in LEADS.items():
            win_ex = pd.DataFrame(
                {k: ex.shift(-k) for k in range(1, h + 1)}
            )
            win_obs = pd.DataFrame(
                {k: (n_obs.shift(-k) >= MIN_READINGS_PER_DAY) for k in range(1, h + 1)}
            )
            pos = (win_ex == 1).any(axis=1)
            all_obs = win_obs.all(axis=1)
            # A 0 is only knowable if every day in the window has a DEFINED exceed value:
            # that needs both a fully observed day AND an official danger level. Stations
            # without one have exceed=NaN for every day -> their labels must stay NaN,
            # never 0 (contract rule: never assume "no flood" when the outcome is unknown).
            all_known = win_ex.notna().all(axis=1)
            lab = np.where(pos, 1.0, np.where(all_obs & all_known, 0.0, np.nan))
            d[col] = lab
            d[f"obs_days_{col[-3:]}"] = win_obs.sum(axis=1)
        d["station_id"] = sid
        out.append(d.reset_index())
    lab = pd.concat(out, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["date"])
    for col in LEADS:
        print(f"{col}: n={lab[col].notna().sum():,}  base rate={lab[col].mean():.2%}")
    return lab


def add_features(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["station_id", "date"]).copy()

    num = ["wl_mean", "wl_max", "wl_min", "wl_first", "wl_last", "n_readings"]
    for k in range(1, 8):
        df[f"wl_last_lag{k}"] = df.groupby("station_id")["wl_last"].shift(k)
    for k in (1, 2, 3):
        df[f"wl_max_lag{k}"] = df.groupby("station_id")["wl_max"].shift(k)
    df["wl_rise_1d"] = df["wl_last"] - df.groupby("station_id")["wl_last"].shift(1)
    df["wl_rise_3d"] = df["wl_last"] - df.groupby("station_id")["wl_last"].shift(3)
    df["wl_dist_to_danger"] = df["danger_level"] - df["wl_max"]
    df["wl_pct_danger"] = df["wl_max"] / df["danger_level"]

    g = df.groupby("station_id")
    df["wl_max_roll7"] = g["wl_max"].transform(lambda s: s.rolling(7, min_periods=4).mean())
    df["wl_max_roll30"] = g["wl_max"].transform(lambda s: s.rolling(30, min_periods=10).mean())
    df["wl_exceed_days_30"] = g["exceed"].transform(
        lambda s: s.fillna(0).rolling(30, min_periods=1).sum()
    )

    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    df = df.merge(
        weather[["station_id", "date", "precip_mm", "tmax_c", "tmin_c", "source"]],
        on=["station_id", "date"],
        how="left",
    )
    gw = df.groupby("station_id")
    df["rain_1d"] = df["precip_mm"]
    for k in (3, 7, 14, 30):
        df[f"rain_{k}d"] = gw["precip_mm"].transform(lambda s: s.rolling(k, min_periods=1).sum())
    df["rain_max_7d"] = gw["precip_mm"].transform(lambda s: s.rolling(7, min_periods=1).max())
    df["wet_days_7"] = gw["precip_mm"].transform(lambda s: (s.fillna(0) > 1).rolling(7, min_periods=1).sum())

    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df, num


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_static_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 11: join per-station SRTM/OSM-derived static features (station_spatial_features.csv).

    Kept as a separate audit-able group; every modelling station must resolve."""
    static = pd.read_csv(SPATIAL)[["station_id"] + SPATIAL_COLS]
    n0 = df["station_id"].nunique()
    out = df.merge(static, on="station_id", how="left")
    missing = n0 - out["station_id"].nunique()
    na = {c: int(out[c].isna().sum()) for c in SPATIAL_COLS}
    if missing:
        print(f"WARN add_static_spatial: {missing} stations absent from station_spatial_features.csv")
    print(f"static spatial joined for {out['station_id'].nunique()} stations; "
          f"NaN cells per column = {na}")
    return out


def add_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 11: nearest-k matrix gauges' regional context on day t (never future).

    Newly-installed gauges only exist from their first report, so the neighbour set is
    RESOLVED PER DAY: of the nearest NEIGHB_POOL gauges, use the nearest k that actually
    reported a value that day; if fewer than k reported, use those that did.  Their rain /
    water-level state on the SAME day is known by the issue instant -> no leakage.
    """
    stcol = df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    latv = stcol["lat"].to_numpy()
    lonv = stcol["lon"].to_numpy()
    ids = stcol["station_id"].to_numpy().astype("int64")
    # SxS distance -> per-gauge ranked pool (nearest-first, self-last)
    D = np.empty((len(ids), len(ids)), dtype="float")
    for i in range(len(ids)):
        D[i] = _haversine_km(latv[i], lonv[i], latv, lonv)
    np.fill_diagonal(D, np.inf)
    pools = np.argsort(D, axis=1)[:, :NEIGHB_POOL]          # position ranks at distance order

    def resolve(pivot):
        """for each station-day: mean over the nearest-k SIG observed that day.

        Returns (mean_matrix, count_matrix) with columns = pivot columns (station order)."""
        vals = pivot.to_numpy(dtype="float")                # (T, S)
        has = ~np.isnan(vals)
        means = {k: np.full(vals.shape, np.nan) for k in NEIGHB_KS}
        counts = {k: np.zeros(vals.shape, dtype="int64") for k in NEIGHB_KS}
        for i in range(vals.shape[1]):
            cols = pools[i]                                  # ranked candidate positions
            sub = vals[:, cols]
            sub_has = has[:, cols]
            cum = sub_has.cumsum(axis=1)
            for k in NEIGHB_KS:
                ok_everywhere = cum[:, -1] >= k
                kth = np.full(vals.shape[0], cols.shape[0], dtype="int64")
                kth[ok_everywhere] = np.argmax(cum[ok_everywhere] >= k, axis=1)
                sel = np.arange(cols.shape[0])[None, :] <= kth[:, None]
                use = sel & sub_has
                cnt = use.sum(axis=1)
                wsum = np.nansum(np.where(use, sub, 0.0), axis=1)
                means[k][:, i] = np.where(cnt > 0, wsum / np.maximum(cnt, 1), np.nan)
                counts[k][:, i] = cnt
        return means, counts

    # RAIN neighbour signals are built from the weather record, not the WL-active matrix:
    # rainfall is knowable at any gauge location on any day (NASA POWER), independent of
    # whether that gauge reported a river stage.  A weather-only day therefore still
    # carries the areal signal.  Same F1 sentinel rule as main().
    wxframe = pd.read_csv(WEATHER, usecols=["station_id", "date", "precip_mm"])
    wxframe = wxframe[wxframe["station_id"].isin(ids)]
    n_sent_wx = int((wxframe["precip_mm"] <= -999).sum())
    wxframe.loc[wxframe["precip_mm"] <= -999, "precip_mm"] = np.nan
    wxframe["date"] = pd.to_datetime(wxframe["date"])
    wxframe["rain_7d"] = wxframe.groupby("station_id")["precip_mm"].transform(
        lambda s: s.rolling(7, min_periods=1).sum())
    wxframe["rain_14d"] = wxframe.groupby("station_id")["precip_mm"].transform(
        lambda s: s.rolling(14, min_periods=1).sum())
    for sig in ("rain_7d", "rain_14d"):
        pivot = wxframe.pivot_table(index="date", columns="station_id",
                                    values=sig, dropna=False).reindex(columns=ids)
        assert (np.asarray(pivot.columns, dtype="int64") == ids).all()
        means, _ = resolve(pivot)
        for k in NEIGHB_KS:
            colname = f"nb{k}_{sig}"
            long = pd.DataFrame(means[k], index=pivot.index, columns=pivot.columns).stack()
            long.name = colname
            df = df.merge(long.reset_index(), on=["station_id", "date"], how="left", validate="one_to_one")
    if n_sent_wx:
        print(f"  nb: {n_sent_wx} weather sentinel cells skipped in the neighbour rain window (F1)")

    # neighbour water-level state: mean wl/danger ratio of the nearest k gauges that
    # actually reported a stage on day t (same issue instant -> no leakage).  This is
    # the NEIGHB_SIGS 'wl_pct_danger' entry; without it the column is silently dropped
    # by the `keep` filter below and audit check A11 fails.
    pv = df.pivot_table(index="date", columns="station_id", values="wl_pct_danger", dropna=False)
    assert (np.asarray(pv.columns, dtype="int64") == ids).all()
    means, _ = resolve(pv)
    for k in NEIGHB_KS:
        colname = f"nb{k}_wl_pct_danger"
        long = pd.DataFrame(means[k], index=pv.index, columns=pv.columns).stack()
        long.name = colname
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left", validate="one_to_one")

    # count of nearest neighbours currently at/above their danger level (same day, no leak)
    pvv = pv.to_numpy(dtype="float")
    dng = (pvv >= 1.0).astype("float")
    dng[np.isnan(pvv)] = np.nan
    has_dng = ~np.isnan(dng)
    res = {k: np.zeros(pv.shape, dtype="int64") for k in NEIGHB_KS}
    for i in range(pvv.shape[1]):
        cols = pools[i]
        sub_has = has_dng[:, cols]
        cum = sub_has.cumsum(axis=1)
        for k in NEIGHB_KS:
            ok_everywhere = cum[:, -1] >= k
            kth = np.full(pvv.shape[0], cols.shape[0], dtype="int64")
            kth[ok_everywhere] = np.argmax(cum[ok_everywhere] >= k, axis=1)
            sel = np.arange(cols.shape[0])[None, :] <= kth[:, None]
            use = sel & sub_has
            res[k][:, i] = np.nansum(np.where(use, dng[:, cols], 0.0), axis=1)
    for k in NEIGHB_KS:
        colname = f"nb{k}_n_wl_danger"
        long = pd.DataFrame(res[k], index=pv.index, columns=pv.columns).stack()
        long.name = colname
        df = df.merge(long.reset_index(), on=["station_id", "date"], how="left", validate="one_to_one")
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    return df


def add_upstream(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 11: upstream dwell-time features on multi-station rivers.

    Lat ordering is wrong for the west-flowing Ganges/Padma lower systems, so upstream =
    nearest same-river gauge with clearly HIGHER elevation within UPSTREAM_MAX_KM.  Its
    rain / WL state shifted 1 and 2 days (strictly before the label window) precedes
    arrivals at the downstream gauge.
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
        raise RuntimeError("add_upstream: no upstream relations found - elevation unreliable?")
    usrc = df[["station_id", "date", "rain_7d", "wl_pct_danger"]].copy()
    for k in UPSTREAM_LAGS:
        usrc[f"up_rain7d_lag{k}"] = usrc.groupby("station_id")["rain_7d"].shift(k)
        usrc[f"up_wl_pct_danger_lag{k}"] = usrc.groupby("station_id")["wl_pct_danger"].shift(k)
    usrc = usrc[["station_id", "date", *[f"up_rain7d_lag{k}" for k in UPSTREAM_LAGS],
                 *[f"up_wl_pct_danger_lag{k}" for k in UPSTREAM_LAGS]]]
    df["_up_id"] = df["station_id"].map(up_map)
    df = df.merge(usrc, left_on=["_up_id", "date"], right_on=["station_id", "date"],
                  how="left", suffixes=("", "_usrc"))
    usrc_cols = [f"up_rain7d_lag{k}" for k in UPSTREAM_LAGS] + \
                [f"up_wl_pct_danger_lag{k}" for k in UPSTREAM_LAGS]
    for c in usrc_cols:
        df[c] = df[c + "_usrc"] if c + "_usrc" in df.columns else df[c]
    keepcol = [c for c in df.columns if c.endswith("_usrc")]
    df.drop(columns=["_up_id", *keepcol], inplace=True)
    print(f"upstream map: {n_mapped} downstream stations with a same-river upstream gauge "
          f"within {UPSTREAM_MAX_KM:g} km; NaN cells per feature = ")
    for c in usrc_cols:
        print(f"    {c}: {int(df[c].isna().sum())}")
    return df


def _save(df: pd.DataFrame, path: Path, what: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            df.to_csv(path, index=False)
            break
        except OSError:
            path = path.with_name(f"{path.stem}_v2.csv")
    else:
        raise PermissionError(f"could not write any variant of {path.name} (open in Excel?)")
    print(f"{what}: {path}")
    return path


def main() -> None:
    global OUT, _MASTER_DL
    st = pd.read_csv(STATIONS)
    _MASTER_DL = st[["id", "dangerlevel"]].rename(
        columns={"id": "station_id", "dangerlevel": "danger_level"}
    )
    _MASTER_DL["danger_level"] = pd.to_numeric(_MASTER_DL["danger_level"], errors="coerce")
    st = st.rename(columns={"id": "station_id"})[
        ["station_id", "name", "river", "basin", "division", "district",
         "upazilla", "lat", "long", "riverhighestwaterlevel"]
    ]
    # The FFWC master calls the same division two different ways; a feature must not
    # split a 21-station division on a spelling.  Normalise to the standard, audible.
    st["division"] = st["division"].str.strip().replace(DIVISION_NORM)
    weather = pd.read_csv(WEATHER)
    # F1 (reports/phase1_data_inventory.md): NASA POWER uses -999.0 for "no data".
    # Those cells are not NaN, so every isna() check reports "0 missing" while the
    # rolling rain sums pick up -999 mm.  Unknown stays unknown -> NaN, never 0.
    wx_cols = ["precip_mm", "tmax_c", "tmin_c", "tmean_c"]
    wx_cols = [c for c in wx_cols if c in weather.columns]
    n_sent = int(sum(int((weather[c] <= -999).sum()) for c in wx_cols))
    if n_sent:
        weather[wx_cols] = weather[wx_cols].mask(weather[wx_cols] <= -999)
        print(f"F1: {n_sent} NASA POWER sentinel cells (<= -999) -> NaN in weather")

    raw = load_waterlevels()
    daily = build_daily(raw)
    daily = add_labels(daily)
    daily, _ = add_features(daily, weather)

    df = st.merge(daily, on="station_id", how="inner")
    df.rename(columns={"long": "lon"}, inplace=True)

    df["n_readings"] = df["n_readings"].fillna(0).astype("int64")
    df["day_observed"] = df["day_observed"].fillna(False).astype("bool")
    df = add_static_spatial(df)
    df = add_neighbourhood(df)
    df = add_upstream(df)
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    lead_cols = list(LEADS)
    neighb_cols = ([f"nb{k}_{sig}" for k in NEIGHB_KS for sig in NEIGHB_SIGS]
                   + [f"nb{k}_n_wl_danger" for k in NEIGHB_KS])
    up_cols = ([f"up_rain7d_lag{k}" for k in UPSTREAM_LAGS]
               + [f"up_wl_pct_danger_lag{k}" for k in UPSTREAM_LAGS])
    keep = [
        "station_id", "name", "river", "basin", "division", "district", "upazilla",
        "lat", "lon", "danger_level", "date", "n_readings", "day_observed",
        "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last",
    ] + SPATIAL_COLS + [f"wl_last_lag{k}" for k in range(1, 8)] + [f"wl_max_lag{k}" for k in (1, 2, 3)] + [
        "wl_rise_1d", "wl_rise_3d", "wl_dist_to_danger", "wl_pct_danger",
        "wl_max_roll7", "wl_max_roll30", "wl_exceed_days_30",
        "precip_mm", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
        "rain_max_7d", "wet_days_7", "tmax_c", "tmin_c", "source",
        "doy_sin", "doy_cos",
    ] + neighb_cols + up_cols + lead_cols + [f"obs_days_{c[-3:]}" for c in lead_cols]
    # a column that was declared but never produced must FAIL the build, not vanish:
    # that silent drop is exactly how nb3/nb5_wl_pct_danger disappeared (Phase 11).
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"build_dataset: declared columns were never produced: {missing}")
    df = df[keep]
    df.sort_values(["station_id", "date"], inplace=True, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT = _save(df, OUT, "FINAL (full calendar matrix)")

    # Model-ready copy: only days that were actually observed AND have at least one
    # usable (non-NaN) label. This is the table to train/evaluate on.
    obs = df[df["day_observed"] & df[lead_cols].notna().any(axis=1)].copy()
    OBS = _save(obs, OUT_DIR / "flood_dataset_model_ready.csv",
                f"MODEL-READY (rows with observed day + a usable label, {len(obs) / len(df):.1%} of matrix)")

    for col in lead_cols:
        v = obs[col]
        print(f"  {col}: n={v.notna().sum():,}  positives={int(v.sum()):,}  "
              f"base rate={v.mean():.2%}")

    print(f"\ncolumns={len(keep)}  stations={df.station_id.nunique()}  "
        f"{df.date.min().date()}..{df.date.max().date()}")
    for c in lead_cols:
        v = df[c]
        print(f"  {c}: n={v.notna().sum():,} ({v.notna().mean():.1%} of rows)  "
              f"positives={int(v.sum()):,}  base rate={v.mean():.2%}")
    print("  per-station label_72h base rate (model-ready, top 10):")
    b = obs.groupby("station_id").agg(n=("label_72h", "size"), pos=("label_72h", "sum"),
                                      rate=("label_72h", "mean"), name=("name", "first"))
    b["rate"] = (b["rate"] * 100).round(1)
    print(b.sort_values("rate", ascending=False).head(10).to_string())

    dict_rows = [
        ("station_id,name,river,basin,division,district,upazilla,lat,lon", "station identity and geography (from FFWC /stations)"),
        ("danger_level", "FFWC published danger level in m; the threshold that defines a flood label"),
        ("date", "issue day UTC; all features use only information up to end of this day"),
        ("n_readings,day_observed", "observed 3-hourly readings that day; day_observed = n_readings >= 3"),
        ("wl_mean..wl_last", "daily aggregate observed water level (m)"),
        ("wl_last_lag1..7,wl_max_lag1..3", "water level of previous days (shifted, never future)"),
        ("wl_rise_1d,wl_rise_3d", "water level rise over 1 and 3 days"),
        ("wl_dist_to_danger,wl_pct_danger", "how far today's max is below/at the danger level"),
        ("wl_max_roll7,wl_max_roll30,wl_exceed_days_30", "antecedent hydrology windows"),
        ("precip_mm,rain_1d..rain_30d,rain_max_7d,wet_days_7", "NASA POWER (MERRA-2) precipitation at the station (mm)"),
        ("tmax_c,tmin_c", "daily temperature at the station (deg C)"),
        ("doy_sin,doy_cos", "day-of-year harmonics (deliberately narrow seasonality)"),
        ("elevation_m,slope_deg,dist_to_coast_km,dist_river_km,dist_major_river_km,"
         "river_density_10km_km_per_km2,n_river_lines_10km",
         "Phase 11 static geo (SRTM/OSM): station elevation, terrain slope, km to coast, "
         "km to nearest river line, km to nearest major river line, river density in a 10 km disc"),
        ("nb3_rain_7d,nb5_rain_7d,nb3_rain_14d,nb5_rain_14d",
         "mean of the 3/5 nearest gauges' precipitation sum over 7/14 d *at day t* (known by issue)"),
        ("nb3_wl_pct_danger,nb5_wl_pct_danger", "mean of the 3/5 nearest gauges' wl/danger ratio at day t"),
        ("nb3_n_wl_danger,nb5_n_wl_danger", "count of the 3/5 nearest gauges currently above danger level at day t"),
        ("up_rain7d_lag1,up_rain7d_lag2", "nearest same-river HIGHER-elevation gauge's rainfall: 7 d sum 1 and 2 days earlier"),
        ("up_wl_pct_danger_lag1,up_wl_pct_danger_lag2", "that upstream gauge's wl/danger ratio 1 and 2 days earlier"),
        ("label_24h,label_48h,label_72h", "1 = danger level reached in the next 1/2/3 days; "
         "0 = fully observed window with no reach; NaN = window not fully observed"),
        ("obs_days_24h,obs_days_48h,obs_days_72h", "how many days of the label window were observed"),
    ]
    pd.DataFrame(dict_rows, columns=["columns", "meaning"]).to_csv(
        OUT_DIR / "data_dictionary.csv", index=False
    )
    print("  data/processed/data_dictionary.csv written")


if __name__ == "__main__":
    main()
