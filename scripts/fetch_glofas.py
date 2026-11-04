"""MI-1b: fetch GloFAS river discharge (Open-Meteo flood API) at transboundary/upstream
and in-country anchor points on the Ganges-Brahmaputra-Meghna system.

Incremental refresher:
- first run finds a representative cell per anchor and downloads full history;
- later runs request only the missing tail (plus a short lookback to absorb revisions)
  and merge it into the existing per-anchor CSV. Raw history is append-only; rows are
  de-duplicated on date with the newest value winning.

Usage:
    python scripts/fetch_glofas.py                     # catch up to yesterday
    python scripts/fetch_glofas.py --end 2026-11-03
    python scripts/fetch_glofas.py --lookback-days 14  # re-fetch last two weeks
    python scripts/fetch_glofas.py --rescan-cells      # re-run the cell search
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "data" / "raw" / "glofas"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (flood-predictor MI-1)"}
URL = "https://flood-api.open-meteo.com/v1/flood"

START = "2013-01-01"  # earliest history to keep
SCAN_START, SCAN_END = "2024-07-01", "2024-08-31"

# name: (lat, lon, role, basin)  role: 'upstream' (transboundary) | 'in_country'
ANCHORS = {
    # --- transboundary / upstream ---
    "patna_ganges": (25.59, 85.14, "upstream", "Ganges"),
    "farakka_ganges": (24.80, 87.93, "upstream", "Ganges"),
    "chisapani_karnali": (28.65, 81.25, "upstream", "Ganges"),
    "bhimnagar_kosi": (26.48, 87.27, "upstream", "Ganges"),
    "dibrugarh_brah": (27.48, 94.90, "upstream", "Brahmaputra"),
    "tezpur_brah": (26.63, 92.80, "upstream", "Brahmaputra"),
    "silchar_barak": (24.82, 92.79, "upstream", "Meghna"),
    "teesta_in": (26.30, 88.70, "upstream", "Brahmaputra"),
    # --- in-country entry / junction ---
    "bahadurabad_brah": (25.18, 89.67, "in_country", "Brahmaputra"),
    "hardinge_ganges": (24.07, 89.03, "in_country", "Ganges"),
    "bhairab_meghna": (24.05, 90.98, "in_country", "Meghna"),
}


def q(lat, lon, s, e):
    r = requests.get(
        URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge,river_discharge_max",
            "start_date": s,
            "end_date": e,
            "timezone": "Asia/Dhaka",
        },
        headers=H,
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def find_cell(lat, lon, step=0.1, span=3):
    best = None
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            la, lo = round(lat + i * step, 3), round(lon + j * step, 3)
            try:
                j2 = q(la, lo, SCAN_START, SCAN_END)
                v = [x for x in j2.get("daily", {}).get("river_discharge", []) if x is not None]
                mx = max(v) if v else -1
            except Exception:
                mx = -1
            if mx is not None and (best is None or mx > best[2]):
                best = (la, lo, mx)
            time.sleep(0.15)
    return best


def read_anchor(path: Path) -> pd.DataFrame | None:
    """Existing per-anchor history, normalised to a `date` string column."""
    if not path.exists() or path.stat().st_size <= 500:
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  WARN could not read {path.name}: {exc}")
        return None
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" not in df.columns or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=[c for c in ("anchor", "role", "basin") if c in df.columns])
    return df


def fetch_anchor(name: str, cell: dict, end: str, lookback_days: int):
    """Fetch the missing tail and merge it into the per-anchor CSV."""
    dst = OUT / "anchors" / f"{name}.csv"
    old = read_anchor(dst)
    if old is not None and not old.empty:
        last = old["date"].max()
        fetch_start = (
            pd.Timestamp(last) - pd.Timedelta(days=max(lookback_days - 1, 0))
        ).strftime("%Y-%m-%d")
        if fetch_start < START:
            fetch_start = START
        if fetch_start > end:
            print(f"  {name:20s} current (last={last})")
            merged = old
        else:
            merged = None
    else:
        fetch_start = START
        merged = None

    if merged is None:
        new = None
        last_err = ""
        for attempt in range(4):
            try:
                payload = q(cell["lat"], cell["lon"], fetch_start, end)
                daily = payload.get("daily", {})
                if "time" not in daily:
                    last_err = "no daily block"
                    break
                new = pd.DataFrame(daily).rename(columns={"time": "date"})
                new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
                break
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                print(f"  {name:20s} fetch error try {attempt + 1}: {last_err}")
                time.sleep(15 * (attempt + 1))
        if new is None:
            print(f"  {name:20s} FAILED ({last_err}); keeping existing rows")
            if old is None or old.empty:
                return None, "failed"
            merged = old
        elif old is not None and not old.empty:
            merged = (
                pd.concat([old, new], ignore_index=True)
                .drop_duplicates("date", keep="last")
                .sort_values("date")
            )
            print(f"  {name:20s} ok +{len(new):,} rows ({fetch_start}..{end})")
        else:
            merged = new.sort_values("date")
            print(f"  {name:20s} ok {len(new):,} rows ({fetch_start}..{end})")

    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged.insert(0, "anchor", name)
    merged["role"] = cell.get("role", "")
    merged["basin"] = cell.get("basin", "")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Per-anchor raw files hold observations only; forecast rows are returned in
    # memory and written to the separate forecast file by main().
    cutoff = (date.today() - timedelta(days=1)).isoformat()
    observations = merged[merged["date"] <= cutoff]
    if not observations.empty:
        observations.to_csv(dst, index=False)
    return merged, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental GloFAS discharge refresh")
    ap.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Last date to fetch (YYYY-MM-DD). Default: yesterday.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=5,
        help="Re-fetch this many trailing days to absorb source revisions (default 5).",
    )
    ap.add_argument("--rescan-cells", action="store_true", help="Re-run the cell search.")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--span", type=int, default=3)
    args = ap.parse_args()

    if args.lookback_days < 1:
        raise SystemExit("--lookback-days must be >= 1")

    chosen: dict = {}
    cells_file = OUT / "chosen_cells.json"
    if cells_file.exists() and not args.rescan_cells:
        chosen = json.loads(cells_file.read_text(encoding="utf-8"))
        print(f"loaded {len(chosen)} chosen cells")

    for name, (la, lo, role, basin) in ANCHORS.items():
        if name in chosen:
            continue
        t = time.time()
        b = find_cell(la, lo, args.step, args.span)
        if b:
            chosen[name] = {
                "lat": b[0],
                "lon": b[1],
                "max_scan_discharge": b[2],
                "role": role,
                "basin": basin,
                "nominal": [la, lo],
            }
            print(f"  {name:20s} -> cell ({b[0]},{b[1]}) maxScan={b[2]:.1f}  {time.time()-t:.0f}s")
        else:
            print(f"  {name:20s} -> NO CELL FOUND")
        cells_file.write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    frames = []
    for name, cell in chosen.items():
        df, _status = fetch_anchor(name, cell, args.end, args.lookback_days)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        print("No anchor data available; nothing written.")
        return 1

    all_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["anchor", "date"], keep="last")
        .sort_values(["anchor", "date"])
    )

    # The flood API returns forecast discharge when the requested end date is in
    # the future. Forecasts are not observations: keep them out of the raw
    # observation store so freshness checks and features only see measured data.
    # They are written to a separate forecast file for later operational use.
    observation_cutoff = (date.today() - timedelta(days=1)).isoformat()
    forecast = all_df[all_df["date"] > observation_cutoff].copy()
    all_df = all_df[all_df["date"] <= observation_cutoff].copy()

    if forecast.empty and all_df.empty:
        print("No observation or forecast rows available; nothing written.")
        return 1

    fc_path = OUT / "forecast_anchors_daily.csv"
    if not forecast.empty:
        forecast.insert(0, "issued_at", pd.Timestamp.now(tz="UTC").isoformat())
        forecast.sort_values(["anchor", "date"]).to_csv(fc_path, index=False)
        print(
            f"forecast: {len(forecast):,} rows "
            f"({forecast['date'].min()}..{forecast['date'].max()}) -> {fc_path.name}"
        )

    if all_df.empty:
        print("No observation rows available; observation file left unchanged.")
        return 1

    big = OUT / "discharge_anchors_daily.csv"
    all_df.to_csv(big, index=False)
    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    manifest = {
        "source": "Open-Meteo flood API / GloFAS",
        "url": URL,
        "start": str(all_df["date"].min()),
        "end": str(all_df["date"].max()),
        "requested_end": args.end,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "anchors": sorted(all_df["anchor"].unique().tolist()),
        "rows": int(len(all_df)),
        "null_discharge": int(all_df["river_discharge"].isna().sum())
        if "river_discharge" in all_df.columns
        else None,
        "output": str(big),
        "sha256": sha,
        "forecast_file": str(fc_path) if fc_path.exists() else None,
        "forecast_rows": int(len(forecast)),
        "forecast_end": str(forecast["date"].max()) if not forecast.empty else None,
        "cells": chosen,
    }
    (OUT / "glofas_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nWROTE {big} rows={len(all_df):,} anchors={all_df.anchor.nunique()} "
        f"range={manifest['start']}..{manifest['end']} sha={sha[:16]}"
    )
    if manifest["end"] < args.end:
        print(f"WARN: source only reaches {manifest['end']} (requested {args.end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
