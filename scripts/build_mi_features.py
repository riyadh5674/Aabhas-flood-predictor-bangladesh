"""MI-2: build the v2 feature table keyed on (station_id, date).

Sources (all joined on the immutable experimental_dataset.csv key):
  * data/raw/power_soil/power_soil_daily.csv      NASA POWER soil wetness (133/133)  -> ps_*
  * data/raw/glofas/discharge_anchors_daily.csv   GloFAS transboundary discharge      -> gf_*
  * data/raw/openmeteo/era5land_station_daily.csv ERA5-Land soil+rain (used only if   -> e5_*
                                                   full 133-station coverage exists)

Output: data/processed/mi_v2_features.csv  (+ feature_sets_v2.json)

LEAKAGE RULE (Phase-13): every new column is a day-t observation or a BACKWARD
lag/window.  No forward shifts.  Climatologies are built from <=2021 rows only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

BASIN_ANCHORS = {
    "Ganges":      ["patna_ganges", "farakka_ganges", "chisapani_karnali", "bhimnagar_kosi", "hardinge_ganges"],
    "Brahmaputra": ["dibrugarh_brah", "tezpur_brah", "teesta_in", "bahadurabad_brah"],
    "Meghna":      ["silchar_barak", "bhairab_meghna"],
}
DEFAULT_ANCHORS = ["bahadurabad_brah", "hardinge_ganges", "bhairab_meghna"]


def load_key() -> pd.DataFrame:
    cols = ["station_id", "date", "lat", "lon", "basin", "danger_level",
            "slope_deg", "dist_major_river_km", "wl_last", "wl_max", "precip_mm"]
    return pd.read_csv(PROC / "experimental_dataset.csv", usecols=cols, parse_dates=["date"])


def _doy_anomaly(df: pd.DataFrame, cols, train_end="2022-01-01") -> pd.DataFrame:
    """Add <col>_anom = col - per-station day-of-year climatology (<=2021 only)."""
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    tr = df[df["date"] < train_end]
    for c in cols:
        clim = tr.groupby(["station_id", "doy"])[c].mean().rename(f"{c}_clim")
        df = df.merge(clim, on=["station_id", "doy"], how="left")
        df[f"{c}_anom"] = df[c] - df[f"{c}_clim"]
    return df.drop(columns=["doy"])


def era5_block() -> pd.DataFrame | None:
    p = RAW / "openmeteo" / "era5land_station_daily.csv"
    e = pd.read_csv(p, parse_dates=["time"]).rename(columns={"time": "date"})
    if e.station_id.nunique() < 130:
        return None
    e = e.rename(columns={
        "precipitation_sum": "e5_rain", "et0_fao_evapotranspiration": "e5_et0",
        "soil_moisture_0_to_7cm_mean": "e5_sm0_7", "soil_moisture_7_to_28cm_mean": "e5_sm7_28",
        "soil_moisture_28_to_100cm_mean": "e5_sm28_100",
        "temperature_2m_mean": "e5_temp", "shortwave_radiation_sum": "e5_rad"})
    e = e.sort_values(["station_id", "date"])
    g = e.groupby("station_id")
    for k in (3, 7, 14, 30):
        e[f"e5_rain_{k}d"] = g["e5_rain"].transform(lambda s: s.rolling(k, min_periods=1).sum())
    e["e5_rain_max7d"] = g["e5_rain"].transform(lambda s: s.rolling(7, min_periods=1).max())
    e["e5_et0_7d"] = g["e5_et0"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    e["e5_water_bal_7d"] = e["e5_rain_7d"] - e["e5_et0_7d"]
    for c in ("e5_sm0_7", "e5_sm7_28", "e5_sm28_100"):
        e[f"{c}_lag1"] = g[c].shift(1)
        e[f"{c}_lag3"] = g[c].shift(3)
    e["e5_sm0_7_chg1d"] = e["e5_sm0_7"] - e["e5_sm0_7_lag1"]
    e["e5_sm0_7_chg3d"] = e["e5_sm0_7"] - e["e5_sm0_7_lag3"]
    return _doy_anomaly(e, ["e5_sm0_7"])


def power_block() -> pd.DataFrame:
    ps = pd.read_csv(RAW / "power_soil" / "power_soil_daily.csv", parse_dates=["date"])
    ps = ps.rename(columns={"ps_top": "ps_wet_top", "ps_root": "ps_wet_root",
                            "ps_prof": "ps_wet_prof"})
    ps = ps.sort_values(["station_id", "date"])
    g = ps.groupby("station_id")
    for c in ("ps_wet_top", "ps_wet_root", "ps_wet_prof"):
        ps[f"{c}_lag1"] = g[c].shift(1)
        ps[f"{c}_lag7"] = g[c].shift(7)
    ps["ps_wet_top_chg1d"] = ps["ps_wet_top"] - ps["ps_wet_top_lag1"]
    ps["ps_wet_top_chg7d"] = ps["ps_wet_top"] - ps["ps_wet_top_lag7"]
    ps["ps_wet_top_7d"] = g["ps_wet_top"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    ps["ps_wet_top_14d"] = g["ps_wet_top"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    ps["ps_wet_top_30d"] = g["ps_wet_top"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    ps["ps_wet_root_7d"] = g["ps_wet_root"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return _doy_anomaly(ps, ["ps_wet_top", "ps_wet_root"])


def _hav(lat1, lon1, la, lo):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(la)
    dp, dl = np.radians(la - lat1), np.radians(lo - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def glofas_block(key: pd.DataFrame) -> pd.DataFrame:
    gl = pd.read_csv(RAW / "glofas" / "discharge_anchors_daily.csv",
                     parse_dates=["date"]).sort_values(["anchor", "date"])
    gl["doy"] = gl["date"].dt.dayofyear
    clim = (gl[gl["date"] < "2022-01-01"].groupby(["anchor", "doy"])["river_discharge"]
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

    B = {b: basin_idx(a) for b, a in BASIN_ANCHORS.items()}
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

    gl["dis_lag1"] = gl.groupby("anchor")["river_discharge"].shift(1)
    gl["dis_lag3"] = gl.groupby("anchor")["river_discharge"].shift(3)
    cells = json.loads((RAW / "glofas" / "chosen_cells.json").read_text(encoding="utf-8"))
    anch = pd.DataFrame([{"anchor": k, "lat": v["lat"], "lon": v["lon"]} for k, v in cells.items()])

    st = key.drop_duplicates("station_id")[["station_id", "lat", "lon", "basin"]].copy()
    st["bkey"] = np.where(st["basin"].isin(BASIN_ANCHORS), st["basin"], "_default")
    st["near_anchor"] = [
        anch.iloc[int(np.argmin(_hav(r.lat, r.lon, anch.lat.to_numpy(), anch.lon.to_numpy())))].anchor
        for r in st.itertuples()]
    near_df = gl[["anchor", "date", "river_discharge", "dis_lag1", "dis_lag3"]].rename(columns={
        "anchor": "near_anchor", "river_discharge": "gf_near_dis",
        "dis_lag1": "gf_near_dis_lag1", "dis_lag3": "gf_near_dis_lag3"})
    out = st.merge(Bdf, on="bkey", how="left").merge(near_df, on=["near_anchor", "date"], how="left")
    keep = ["station_id", "date", "gf_up_ratio", "gf_up_rise3d",
            "gf_up_ratio_lag1", "gf_up_ratio_lag2", "gf_up_ratio_lag3",
            "gf_up_ratio_lag5", "gf_up_ratio_lag7", "gf_up_ratio_lag10", "gf_up_ratio_lag14",
            "gf_up_ratio_3d", "gf_up_ratio_7d", "gf_up_ratio_14d",
            "gf_up_rise3d_lag1", "gf_up_rise3d_lag3", "gf_up_accel",
            "gf_near_dis", "gf_near_dis_lag1", "gf_near_dis_lag3"]
    return out[keep]


def derived_block(key: pd.DataFrame) -> pd.DataFrame:
    """Derived features from the existing water-level + rainfall columns."""
    from scipy.signal import lfilter
    k = key.sort_values(["station_id", "date"]).copy()
    g = k.groupby("station_id")
    for L in (10, 14, 21, 30):
        k[f"wl_last_lag{L}"] = g["wl_last"].shift(L)
    for decay, name in ((0.9, "api_09"), (0.7, "api_07")):
        k[name] = g["precip_mm"].transform(
            lambda s: lfilter([1.0], [1.0, -decay], np.nan_to_num(s.to_numpy(dtype="float64"))))
    keep = ["station_id", "date", "wl_last_lag10", "wl_last_lag14",
            "wl_last_lag21", "wl_last_lag30", "api_09", "api_07"]
    return k[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="optional suffix for the outputs, e.g. 'era5' -> "
                         "mi_v2_features_era5.csv + feature_sets_v2_era5.json "
                         "(leaves the frozen untagged v2 table untouched)")
    a = ap.parse_args()
    suffix = f"_{a.tag}" if a.tag else ""
    key = load_key()
    print(f"key rows={len(key):,} stations={key.station_id.nunique()}"
          f"{'  (tag=' + a.tag + ')' if a.tag else ''}")

    blocks, newcols = [], []
    e5 = era5_block()
    if e5 is not None:
        e5_new = [c for c in e5.columns if c.startswith("e5_")]
        blocks.append(e5[["station_id", "date"] + e5_new]); newcols += e5_new
        print(f"era5 block:   {len(e5_new):2d} cols (full coverage)")
    else:
        print("era5 block:   SKIPPED (coverage < 130 stations)")

    ps = power_block()
    ps_new = [c for c in ps.columns if c.startswith("ps_")]
    blocks.append(ps[["station_id", "date"] + ps_new]); newcols += ps_new
    print(f"power block:  {len(ps_new):2d} cols")

    gf = glofas_block(key)
    gf_new = [c for c in gf.columns if c.startswith("gf_")]
    blocks.append(gf[["station_id", "date"] + gf_new]); newcols += gf_new
    print(f"glofas block: {len(gf_new):2d} cols")

    dv = derived_block(key)
    dv_new = [c for c in dv.columns if c not in ("station_id", "date")]
    blocks.append(dv[["station_id", "date"] + dv_new]); newcols += dv_new
    print(f"derived block:{len(dv_new):2d} cols")

    out = key.copy()
    for b in blocks:
        out = out.merge(b, on=["station_id", "date"], how="left", validate="one_to_one")

    out["ix_wet_top_x_slope"] = out["ps_wet_top"] * out["slope_deg"]
    out["ix_wet_top_x_updis"] = out["ps_wet_top"] * out["gf_up_ratio_lag1"]
    out["ix_updis_x_wlrise"] = out["gf_up_ratio_lag1"] * (out["wl_last"] - out["wl_max"])
    out["ix_api09_x_wet_top"] = out["api_09"] * out["ps_wet_top"]
    out["ix_updis14_x_api"] = out["gf_up_ratio_14d"] * out["api_09"]
    if e5 is not None:
        out["ix_rain7_x_sm0_7"] = out["e5_rain_7d"] * out["e5_sm0_7"]
        out["ix_sm0_7_x_slope"] = out["e5_sm0_7"] * out["slope_deg"]
    ix = ["ix_wet_top_x_slope", "ix_wet_top_x_updis", "ix_updis_x_wlrise",
          "ix_api09_x_wet_top", "ix_updis14_x_api"] + \
         (["ix_rain7_x_sm0_7", "ix_sm0_7_x_slope"] if e5 is not None else [])
    newcols += ix

    report = {c: round(float(out[c].isna().mean()), 4) for c in newcols}
    dst = PROC / f"mi_v2_features{suffix}.csv"
    out[["station_id", "date"] + newcols].to_csv(dst, index=False)
    print(f"\nWROTE {dst}  rows={len(out):,}  new cols={len(newcols)}")
    print("worst null rates:", dict(sorted(report.items(), key=lambda kv: -kv[1])[:6]))

    fs = json.loads((PROC / "feature_sets.json").read_text(encoding="utf-8"))
    fs["created"] = "2026-10-15"
    fs["source_v2"] = f"data/processed/mi_v2_features{suffix}.csv (MI-2)"
    if a.tag:
        fs["tag"] = a.tag
        fs["era5_included"] = e5 is not None
    fs["sets"]["FULL_V2"] = fs["sets"]["FULL"] + newcols
    fs.setdefault("counts", {})["FULL_V2"] = len(fs["sets"]["FULL_V2"])
    fs["counts"]["V2_NEW"] = len(newcols)
    fs["v2_new_columns"] = newcols
    fs_dst = PROC / f"feature_sets_v2{suffix}.json"
    fs_dst.write_text(json.dumps(fs, indent=2), encoding="utf-8")
    print(f"WROTE {fs_dst}  FULL_V2={fs['counts']['FULL_V2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
