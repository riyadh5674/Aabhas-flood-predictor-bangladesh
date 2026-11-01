"""
System Ingestion — Rainfall data.

Fetches daily precipitation at every FFWC station from NASA POWER (MERRA-2
reanalysis) with optional Open-Meteo fallback. Applies QC: sentinel <= -999 ->
NaN (never 0). Writes to
    system/data/raw/rainfall/weather_station_daily.csv

Resumable: stations already complete in the partial file are skipped.
Rate-limited: sleep between station requests, exponential backoff on 429/5xx.

Usage:
    python system/ingestion/ingest_rainfall.py --start 2016-12-01 --end 2026-10-06 --source nasa_power
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system.ingestion.qc_rules import sanitize_rainfall
from system.ingestion.provenance import ProvenanceManifest

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
CONFIG = json.loads((ROOT / "system" / "config" / "ingestion_config.json").read_text(encoding="utf-8"))
RAIN_CFG = CONFIG["rainfall"]
NASA_CFG = RAIN_CFG["nasa_power"]
OM_CFG = RAIN_CFG["open_meteo"]

API = NASA_CFG["api"]
HEADERS = {"User-Agent": "flood-predictor-system/1.0 (contact: local research project)"}
RAW_DIR = ROOT / "system" / "data" / "raw" / "rainfall"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "weather_station_daily.csv"
PART = RAW_DIR / "weather_station_daily.partial.csv"
PARAMS = NASA_CFG["parameters"]
SLEEP = NASA_CFG["sleep_seconds"]
RETRIES = NASA_CFG["retries"]
DELAY_BASE = NASA_CFG["retry_delay_base"]
TIMEOUT = NASA_CFG["timeout_seconds"]

MANIFEST = ProvenanceManifest(RAW_DIR / "rainfall_manifest.csv")


def _series(param: dict, key: str, dates: list[str]) -> list[float]:
    """NASA POWER returns -999.0 for 'no data' and may return null.
    Both mean UNKNOWN -> NaN. Never 0 (that would assert 'no rain').
    Finding F1 (reports/phase1_data_inventory.md)."""
    out = []
    for d in dates:
        v = param.get(key, {}).get(d)
        if v is None or v == "":
            out.append(float("nan"))
        else:
            v = float(v)
            out.append(float("nan") if v <= -999.0 else v)
    return out


def _get_point(lat: float, lon: float, start: str, end: str, tries: int = RETRIES) -> pd.DataFrame:
    params = {
        "parameters": PARAMS,
        "community": NASA_CFG["community"],
        "longitude": round(lon, 5),
        "latitude": round(lat, 5),
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    delay = DELAY_BASE
    for i in range(tries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            j = r.json()
            param = j["properties"]["parameter"]
            if not param:
                return pd.DataFrame()
            dates = sorted(param["PRECTOTCORR"].keys())
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates]
                    ),
                    "precip_mm": _series(param, "PRECTOTCORR", dates),
                    "tmax_c": _series(param, "T2M_MAX", dates),
                    "tmin_c": _series(param, "T2M_MIN", dates),
                    "tmean_c": _series(param, "T2M", dates),
                }
            )
        except Exception as exc:
            if i == tries - 1:
                raise RuntimeError(f"too many failures for ({lat}, {lon}): {exc}")
            time.sleep(delay)
            delay *= 2
    return pd.DataFrame()


def fetch_nasa_power(
    start: str, end: str, station_ids: Optional[list[int]] = None, sleep: float = SLEEP
) -> pd.DataFrame:
    """Fetch NASA POWER daily weather for all (or selected) stations."""
    st_path = ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
    st = pd.read_csv(st_path).dropna(subset=["lat", "long"]).drop_duplicates("id")
    if station_ids is not None:
        st = st[st["id"].isin(station_ids)]

    expected = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1

    done_frames: list[pd.DataFrame] = []
    if PART.exists():
        done = pd.read_csv(PART, parse_dates=["date"])
        complete = done.groupby("station_id")["date"].nunique()
        keep = complete[complete >= expected].index.tolist()
        if keep:
            done_frames.append(done[done["station_id"].isin(keep)])
            print(f"resuming: {len(keep)} stations already complete")
        st = st[~st["id"].isin(keep)]

    print(f"fetching NASA POWER weather for {len(st)} stations ({start}..{end})")

    t0 = time.time()
    for i, (_, row) in enumerate(st.iterrows(), 1):
        sid = int(row["id"])
        try:
            df = _get_point(row["lat"], row["long"], start, end)
        except Exception as exc:
            print(f"  station {sid}: FAILED {exc}")
            MANIFEST.record(sid, end, "nasa_power", 0, "failed")
            continue
        if not df.empty:
            df = sanitize_rainfall(df)
            df["station_id"] = sid
            df["source"] = "NASA POWER / MERRA-2"
            done_frames.append(df)
            MANIFEST.record(sid, end, "nasa_power", len(df), "ok")
        if i % 10 == 0:
            pd.concat(done_frames, ignore_index=True).to_csv(PART, index=False)
            print(f"  {i}/{len(st)} stations ({time.time() - t0:.0f}s)")
        time.sleep(sleep)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    allw = pd.concat(done_frames, ignore_index=True)
    allw["source"] = "NASA POWER / MERRA-2"

    # Append to existing file instead of overwriting (bug fix)
    if OUT.exists():
        existing = pd.read_csv(OUT, parse_dates=["date"])
        existing["source"] = existing.get("source", "NASA POWER / MERRA-2")
        allw = pd.concat([existing, allw], ignore_index=True)
        allw = allw.drop_duplicates(subset=["station_id", "date"], keep="last")
        allw = allw.sort_values(["station_id", "date"]).reset_index(drop=True)

    allw.to_csv(OUT, index=False)
    PART.unlink(missing_ok=True)
    print(
        f"{OUT}: {allw.shape[0]:,} rows, {allw.station_id.nunique()} stations, "
        f"{allw.date.min().date()}..{allw.date.max().date()}"
    )
    return allw


def fetch_openmeteo_batch(
    start: str, end: str, station_ids: Optional[list[int]] = None
) -> pd.DataFrame:
    """Fetch Open-Meteo ERA5-Land daily data in batches (collapses 133 calls into a few requests)."""
    st_path = ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
    st = pd.read_csv(st_path).dropna(subset=["lat", "long"]).drop_duplicates("id")
    if station_ids is not None:
        st = st[st["id"].isin(station_ids)]

    OUT_DIR = ROOT / "system" / "data" / "raw" / "openmeteo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DAILY = OM_CFG["daily_variables"]
    H = {"User-Agent": "Mozilla/5.0 (flood-predictor-system/1.0)"}
    URL = OM_CFG["api"]
    BATCH_SIZE = OM_CFG["batch_size"]
    RETRIES = OM_CFG["retries"]
    BACKOFF_429 = OM_CFG["retry_backoff_429"]
    TIMEOUT = OM_CFG["timeout_seconds"]

    frames = []
    t0 = time.time()

    for batch_start in range(0, len(st), BATCH_SIZE):
        batch = st.iloc[batch_start : batch_start + BATCH_SIZE]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in batch.lat),
            "longitude": ",".join(f"{v:.4f}" for v in batch.long),
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY),
            "timezone": OM_CFG["timezone"],
        }
        for a in range(RETRIES):
            try:
                r = requests.get(URL, params=params, headers=H, timeout=TIMEOUT)
                if r.status_code == 429:
                    time.sleep(BACKOFF_429 * (a + 1))
                    continue
                r.raise_for_status()
                j = r.json()
                if isinstance(j, list) and all("daily" in x for x in j):
                    for idx, loc in enumerate(j):
                        sid = int(batch.iloc[idx]["id"])
                        daily = loc["daily"]
                        loc_df = pd.DataFrame(
                            {
                                "date": pd.to_datetime(daily["time"]),
                                "station_id": sid,
                                "precip_mm": daily.get("precipitation_sum", [np.nan] * len(daily["time"])),
                                "et0_fao_evapotranspiration": daily.get("et0_fao_evapotranspiration", [np.nan] * len(daily["time"])),
                                "soil_moisture_0_to_7cm_mean": daily.get("soil_moisture_0_to_7cm_mean", [np.nan] * len(daily["time"])),
                                "soil_moisture_7_to_28cm_mean": daily.get("soil_moisture_7_to_28cm_mean", [np.nan] * len(daily["time"])),
                                "soil_moisture_28_to_100cm_mean": daily.get("soil_moisture_28_to_100cm_mean", [np.nan] * len(daily["time"])),
                                "temperature_2m_mean": daily.get("temperature_2m_mean", [np.nan] * len(daily["time"])),
                                "shortwave_radiation_sum": daily.get("shortwave_radiation_sum", [np.nan] * len(daily["time"])),
                            }
                        )
                        loc_df["source"] = "Open-Meteo ERA5-Land"
                        frames.append(loc_df)
                        MANIFEST.record(sid, end, "open_meteo", len(loc_df), "ok")
                    break
                else:
                    print(f"  batch {batch_start//BATCH_SIZE + 1}: unexpected response")
                    time.sleep(30 * (a + 1))
            except Exception as exc:
                print(f"  batch {batch_start//BATCH_SIZE + 1} try {a+1}: {type(exc).__name__}: {str(exc)[:90]}")
                time.sleep(30 * (a + 1))

    if not frames:
        print("no Open-Meteo data fetched")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"])
    out_file = OUT_DIR / "era5land_station_daily.csv"
    all_df.to_csv(out_file, index=False)
    import hashlib

    sha = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(
        f"{out_file}: {all_df.shape[0]:,} rows, {all_df.station_id.nunique()} stations, "
        f"sha256={sha[:16]}..."
    )
    return all_df


def main() -> None:
    ap = argparse.ArgumentParser(description="System rainfall ingestion")
    ap.add_argument("--start", default="2016-12-01")
    ap.add_argument("--end", default="2026-10-06")
    ap.add_argument(
        "--source",
        choices=["nasa_power", "open_meteo"],
        default="nasa_power",
    )
    ap.add_argument("--stations", nargs="+", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=SLEEP)
    args = ap.parse_args()

    if args.source == "nasa_power":
        fetch_nasa_power(args.start, args.end, args.stations, args.sleep)
    else:
        fetch_openmeteo_batch(args.start, args.end, args.stations)


if __name__ == "__main__":
    main()
