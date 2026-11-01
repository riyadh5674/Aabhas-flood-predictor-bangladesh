"""
System Backtest — validate D16 predictions against actual water-level exceedances.

Picks a recent monsoon date with known flooding, runs predictions through the
API for all stations, and compares predicted risk bands vs actual danger-level
exceedances.

Usage:
    python system/backtest.py --date 2024-07-15 --horizon 24h
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import requests
import urllib.request

API_BASE = "http://localhost:8000"
ROOT = Path(r"D:\Flood predictor")


def api_get(path, params=None, timeout=30):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        print(f"API error ({path}): {exc}")
    return None


def load_danger_levels() -> dict:
    """station_id -> danger_level (float) from station master."""
    st = pd.read_csv(ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv")
    st = st.rename(columns={"id": "station_id", "dangerlevel": "danger_level"})
    st["danger_level"] = pd.to_numeric(st["danger_level"], errors="coerce")
    return dict(zip(st.station_id, st.danger_level))


def load_waterlevels(target_date: str) -> dict:
    """station_id -> max water level on target_date from raw QC'd data."""
    target = pd.to_datetime(target_date).date()
    year = pd.to_datetime(target_date).year
    result = {}
    wl_dir = ROOT / "system" / "data" / "raw" / "ffwc" / "waterlevel"
    for station_dir in sorted(wl_dir.iterdir()):
        if not station_dir.is_dir():
            continue
        f = station_dir / f"{year}.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, usecols=["st_id", "wl_date", "waterlevel"])
        except Exception:
            continue
        df["wl_date"] = pd.to_datetime(df["wl_date"], errors="coerce", utc=True)
        day = df[df["wl_date"].dt.date == target]
        if day.empty:
            continue
        sid = int(df["st_id"].iloc[0])
        result[sid] = float(day["waterlevel"].max())
    return result


def run_backtest(target_date: str, horizon: str = "24h"):
    print(f"Backtest: {target_date} horizon={horizon}")
    print("=" * 60)

    preds = api_get(
        f"/predict/all?date={target_date}&horizon={horizon}"
    )
    if preds is None:
        print("Failed to get predictions from API")
        return
    pred_df = pd.DataFrame(preds)
    valid = pred_df[pred_df["probability"].notna()].copy()
    print(f"Predictions available: {len(valid)} stations")

    # Actual: did water level exceed danger?
    danger = load_danger_levels()
    actual_wl = load_waterlevels(target_date)
    print(f"Actual water-level readings: {len(actual_wl)} stations")

    # Merge
    rows = []
    for _, r in valid.iterrows():
        sid = int(r["station_id"])
        dl = danger.get(sid)
        wl = actual_wl.get(sid)
        exceeded = (pd.notna(dl) and pd.notna(wl) and wl >= dl)
        rows.append({
            "station_id": sid,
            "probability": float(r["probability"]),
            "band": r["band"],
            "danger_level": dl,
            "wl_max": wl,
            "exceeded": exceeded,
        })
    out = pd.DataFrame(rows)

    # Overall metrics
    exceeded_mask = out["exceeded"].fillna(False)
    high_risk_mask = out["band"].isin(["high", "very_high"])

    n_exceed = int(exceeded_mask.sum())
    n_high_risk = int(high_risk_mask.sum())
    n_caught = int((exceeded_mask & high_risk_mask).sum())
    n_false_alarm = int((~exceeded_mask & high_risk_mask).sum())
    n_missed = int((exceeded_mask & ~high_risk_mask).sum())

    print()
    print(f"Actual exceedances (wl >= danger): {n_exceed}")
    print(f"High/very_high risk predictions: {n_high_risk}")
    print(f"Correctly caught (high risk + exceeded): {n_caught}")
    print(f"False alarms (high risk + no exceed): {n_false_alarm}")
    print(f"Missed (low risk + exceeded): {n_missed}")

    if n_exceed > 0:
        recall = n_caught / n_exceed
        print(f"Recall (floods caught): {recall:.1%}")
    if n_high_risk > 0:
        precision = n_caught / n_high_risk
        print(f"Precision (high-risk real): {precision:.1%}")

    # Show caught and missed
    caught = out[exceeded_mask & high_risk_mask]
    if len(caught):
        print()
        print("Caught floods (high risk + exceeded):")
        for _, r in caught.iterrows():
            wl = r['wl_max']
            dl = r['danger_level']
            wl_str = f"{wl:.2f}" if pd.notna(wl) else "N/A"
            dl_str = f"{dl:.2f}" if pd.notna(dl) else "N/A"
            print(f"  Station {r['station_id']}: p={r['probability']:.3f} band={r['band']} "
                  f"wl={wl_str} danger={dl_str}")

    missed = out[exceeded_mask & ~high_risk_mask]
    if len(missed):
        print()
        print("Missed floods (low/moderate risk + exceeded):")
        for _, r in missed.iterrows():
            wl = r['wl_max']
            dl = r['danger_level']
            wl_str = f"{wl:.2f}" if pd.notna(wl) else "N/A"
            dl_str = f"{dl:.2f}" if pd.notna(dl) else "N/A"
            print(f"  Station {r['station_id']}: p={r['probability']:.3f} band={r['band']} "
                  f"wl={wl_str} danger={dl_str}")

    false = out[~exceeded_mask & high_risk_mask]
    if len(false):
        print()
        print("False alarms (high risk + no exceed):")
        for _, r in false.head(5).iterrows():
            wl = r['wl_max']
            dl = r['danger_level']
            wl_str = f"{wl:.2f}" if pd.notna(wl) else "N/A"
            dl_str = f"{dl:.2f}" if pd.notna(dl) else "N/A"
            print(f"  Station {r['station_id']}: p={r['probability']:.3f} band={r['band']} "
                  f"wl={wl_str} danger={dl_str}")

    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Backtest flood predictions")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    ap.add_argument("--horizon", choices=["24h", "48h", "72h"], default="24h")
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date).isoformat()
    run_backtest(target_date, args.horizon)


if __name__ == "__main__":
    main()
