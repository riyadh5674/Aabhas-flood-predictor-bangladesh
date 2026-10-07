"""Build the final modelling table: one row per station-day with 24/48/72 h labels.

Inputs
    data/raw/ffwc/waterlevel/{station_id}/{year}.csv   (3-hourly observed water level)
    data/raw/ffwc/ffwc_stations.csv                    (station master, danger levels, geo)
    data/raw/rainfall/weather_station_daily.csv        (daily precipitation / temperature)

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
OUT_DIR = ROOT / "data" / "processed"
OUT = OUT_DIR / "flood_dataset.csv"
MIN_READINGS_PER_DAY = 3
LEADS = {"label_24h": 1, "label_48h": 2, "label_72h": 3}
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
    raw = raw[raw["wl"] > -1000]
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
    pd.set_option("future.no_silent_downcasting", True)
    st = pd.read_csv(STATIONS)
    _MASTER_DL = st[["id", "dangerlevel"]].rename(
        columns={"id": "station_id", "dangerlevel": "danger_level"}
    )
    _MASTER_DL["danger_level"] = pd.to_numeric(_MASTER_DL["danger_level"], errors="coerce")
    st = st.rename(columns={"id": "station_id"})[
        ["station_id", "name", "river", "basin", "division", "district",
         "upazilla", "lat", "long", "riverhighestwaterlevel"]
    ]
    weather = pd.read_csv(WEATHER)

    raw = load_waterlevels()
    daily = build_daily(raw)
    daily = add_labels(daily)
    daily, _ = add_features(daily, weather)

    df = st.merge(daily, on="station_id", how="inner")
    df.rename(columns={"long": "lon"}, inplace=True)

    df["n_readings"] = df["n_readings"].fillna(0).astype("int64")
    df["day_observed"] = df["day_observed"].fillna(False).astype("bool")
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    lead_cols = list(LEADS)
    keep = [
        "station_id", "name", "river", "basin", "division", "district", "upazilla",
        "lat", "lon", "danger_level", "date", "n_readings", "day_observed",
        "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last",
    ] + [f"wl_last_lag{k}" for k in range(1, 8)] + [f"wl_max_lag{k}" for k in (1, 2, 3)] + [
        "wl_rise_1d", "wl_rise_3d", "wl_dist_to_danger", "wl_pct_danger",
        "wl_max_roll7", "wl_max_roll30", "wl_exceed_days_30",
        "precip_mm", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
        "rain_max_7d", "wet_days_7", "tmax_c", "tmin_c", "source",
        "doy_sin", "doy_cos",
    ] + lead_cols + [f"obs_days_{c[-3:]}" for c in lead_cols]
    df = df[[c for c in keep if c in df.columns]]
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
