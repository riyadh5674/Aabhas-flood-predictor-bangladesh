"""MI-1a: fetch ERA5-Land daily reanalysis for every FFWC station (Open-Meteo archive).

Writes one CSV per station under data/raw/openmeteo/era5land/ (resumable), plus a manifest.
No authentication required. Cache-first: an existing per-station file is not re-fetched.
"""
from __future__ import annotations
import argparse, json, sys, time, hashlib
from pathlib import Path
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "data" / "raw" / "openmeteo" / "era5land"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (flood-predictor MI-1)"}

START = "2013-01-01"
END = "2026-10-12"
DAILY = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "temperature_2m_mean",
    "shortwave_radiation_sum",
]
URL = "https://archive-api.open-meteo.com/v1/archive"


def load_stations() -> pd.DataFrame:
    p = ROOT / "data" / "interim" / "mi_stations.csv"
    if p.exists():
        return pd.read_csv(p)
    d = pd.read_csv(ROOT / "data" / "processed" / "experimental_dataset.csv",
                    usecols=["station_id", "lat", "lon"])
    u = d.drop_duplicates("station_id").sort_values("station_id")
    p.parent.mkdir(parents=True, exist_ok=True)
    u.to_csv(p, index=False)
    return u


def fetch_one(sid: int, lat: float, lon: float, retries: int = 8) -> tuple[int, str]:
    dst = OUT / f"{sid}.csv"
    if dst.exists() and dst.stat().st_size > 500:
        return sid, "cached"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "daily": ",".join(DAILY),
        "timezone": "Asia/Dhaka",
    }
    last = ""
    for a in range(retries):
        try:
            r = requests.get(URL, params=params, headers=H, timeout=120)
            if r.status_code == 429:
                time.sleep(30 * (a + 1)); last = "429"; continue
            r.raise_for_status()
            j = r.json()
            d = j.get("daily")
            if not d or "time" not in d:
                last = f"no-daily:{str(j)[:120]}"; time.sleep(3); continue
            df = pd.DataFrame(d)
            df.insert(0, "station_id", sid)
            df.to_csv(dst, index=False)
            return sid, f"ok rows={len(df)}"
        except Exception as e:
            last = f"{type(e).__name__}:{str(e)[:80]}"
            time.sleep(4 * (a + 1))
    return sid, f"FAIL {last}"


def concatenate() -> None:
    files = sorted(OUT.glob("*.csv"), key=lambda p: int(p.stem))
    if not files:
        print("no per-station files"); return
    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True).sort_values(["station_id", "time"])
    big = ROOT / "data" / "raw" / "openmeteo" / "era5land_station_daily.csv"
    all_df.to_csv(big, index=False)
    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    man = {
        "source": "Open-Meteo archive / ERA5-Land",
        "url": URL, "start": START, "end": END, "variables": DAILY,
        "stations": int(all_df.station_id.nunique()), "rows": int(len(all_df)),
        "output": str(big), "sha256": sha,
    }
    (ROOT / "data" / "raw" / "openmeteo" / "era5land_manifest.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8")
    print(f"WROTE {big}  rows={len(all_df):,}  stations={all_df.station_id.nunique()}  sha256={sha[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--concat-only", action="store_true")
    a = ap.parse_args()

    if a.concat_only:
        concatenate(); return 0

    st = load_stations()
    if a.limit:
        st = st.head(a.limit)
    print(f"ERA5-Land fetch: {len(st)} stations  {START}..{END}  vars={len(DAILY)}")

    ok = fail = cache = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, int(r.station_id), float(r.lat), float(r.lon)): r.station_id
                for r in st.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            sid, msg = f.result()
            if msg == "cached":
                cache += 1
            elif msg.startswith("ok"):
                ok += 1
            else:
                fail += 1
                print(f"  [{sid}] {msg}")
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  ok={ok} cached={cache} fail={fail}  {time.time()-t0:.0f}s")

    print()
    concatenate()
    print(f"done in {time.time()-t0:.0f}s  ok={ok} cached={cache} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
