"""
System Alerts — Early-warning engine.

Evaluates D16 inference output per (station, horizon) and emits alert events on
band escalation. Implements per-station state persistence and cooldown so the
same station isn't re-alerted within the cooldown window unless the band
escalates further.

Alert levels (from risk band):
  low        -> no alert
  moderate   -> watch
  high       -> warning
  very_high  -> high

Cooldown: default 24 h; bypassed on escalation to a worse band.

Usage:
    python system/alerts/run_alerts.py --date 2024-07-15 --horizon 24h
    python system/alerts/run_alerts.py --date 2024-07-15 --all-horizons
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from system.api.inference import predict_all_stations, model_info
from system.config.model_config import RISK_BANDS, THRESHOLDS, HORIZONS

ROOT = Path(r"D:\Flood predictor")
SYSTEM = ROOT / "system"


def load_alert_config() -> dict:
    path = SYSTEM / "config" / "alert_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Alert level mapping (risk band -> alert level)
# ---------------------------------------------------------------------------

_BAND_TO_LEVEL = {
    "low": None,
    "moderate": "watch",
    "high": "warning",
    "very_high": "high",
}

_LEVEL_SEVERITY = {
    None: 0,
    "watch": 1,
    "warning": 2,
    "high": 3,
}


def band_to_alert_level(band: str) -> Optional[str]:
    return _BAND_TO_LEVEL.get(band)


def is_worse(level_a: Optional[str], level_b: Optional[str]) -> bool:
    """True if level_a is a worse (higher-severity) alert than level_b."""
    return _LEVEL_SEVERITY.get(level_a, 0) > _LEVEL_SEVERITY.get(level_b, 0)


# ---------------------------------------------------------------------------
# Station state persistence
# ---------------------------------------------------------------------------

class AlertState:
    """Per-station alert state: last band, last alert level, last alert date."""

    def __init__(self, path: Path):
        self.path = path
        self._state: dict = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._state = {}
            return
        try:
            self._state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._state = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def get(self, station_id: int, horizon: str) -> dict:
        key = str(station_id)
        return self._state.get(key, {}).get(horizon, {})

    def update(
        self,
        station_id: int,
        horizon: str,
        band: str,
        alert_level: Optional[str],
        alerted_at: str,
    ) -> None:
        key = str(station_id)
        if key not in self._state:
            self._state[key] = {}
        self._state[key][horizon] = {
            "band": band,
            "alert_level": alert_level,
            "last_alert_at": alerted_at if alert_level else None,
        }


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------

def evaluate_alerts(
    predictions: pd.DataFrame,
    state: AlertState,
    config: dict,
    target_date: date,
) -> list[dict]:
    """Compare each prediction to previous state and emit alert events.

    Returns a list of alert event dicts.
    """
    cooldown_hours = config["cooldown"]["cooldown_hours"]
    escalation_bypass = config["cooldown"]["escalation_bypass_cooldown"]
    now_utc = datetime.now(timezone.utc).isoformat()

    alerts: list[dict] = []

    for _, row in predictions.iterrows():
        sid = int(row["station_id"])
        horizon = row["horizon"]
        band = row["band"]
        prob = float(row["probability"])
        alert_level = band_to_alert_level(band)

        prev = state.get(sid, horizon)
        prev_band = prev.get("band")
        prev_level = band_to_alert_level(prev_band) if prev_band else None
        last_alert_at = prev.get("last_alert_at")

        should_alert = False
        reason = ""

        if alert_level is None:
            # Band resolved to low — no alert, just update state
            state.update(sid, horizon, band, None, now_utc)
            continue

        if prev_level is None:
            # First-ever evaluation for this station+horizon
            should_alert = True
            reason = "initial_detection"
        elif is_worse(alert_level, prev_level):
            # Escalation to a worse band
            should_alert = True
            reason = "escalation"
        elif escalation_bypass is False and last_alert_at:
            # Check cooldown
            last_dt = datetime.fromisoformat(last_alert_at)
            if datetime.now(timezone.utc) - last_dt > timedelta(hours=cooldown_hours):
                should_alert = True
                reason = "cooldown_expired"

        if should_alert:
            alert_event = {
                "station_id": sid,
                "date": target_date.isoformat(),
                "horizon": horizon,
                "risk_band": band,
                "probability": prob,
                "previous_band": prev_band,
                "alert_level": alert_level,
                "triggered_at": now_utc,
                "reason": reason,
            }
            alerts.append(alert_event)
            state.update(sid, horizon, band, alert_level, now_utc)
        else:
            # No alert — still update band state
            state.update(sid, horizon, band, alert_level, last_alert_at or now_utc)

    return alerts


def write_alert_log(alerts: list[dict], log_path: Path) -> None:
    """Append alert events to the CSV log."""
    if not alerts:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    fieldnames = [
        "station_id", "date", "horizon", "risk_band", "probability",
        "previous_band", "alert_level", "triggered_at", "reason",
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for a in alerts:
            writer.writerow(a)


def notify_stub(alerts: list[dict], config: dict) -> None:
    """Stub notification interface — replace with real dispatch later."""
    if not config["notification"]["enabled"]:
        return
    for a in alerts:
        print(f"[NOTIFY] {a['alert_level'].upper()} station={a['station_id']} "
              f"horizon={a['horizon']} band={a['risk_band']} p={a['probability']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_alerts(
    target_date: date | str,
    horizons: list[str] | None = None,
    write_log: bool = True,
) -> list[dict]:
    """Run alert evaluation for a given date and horizon(s)."""
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    config = load_alert_config()
    horizons = horizons or config["horizons"]

    state_path = ROOT / config["state"]["state_file"]
    log_path = ROOT / config["state"]["alert_log"]
    state = AlertState(state_path)

    all_alerts: list[dict] = []

    for horizon in horizons:
        if horizon not in HORIZONS:
            print(f"WARN: unknown horizon {horizon}, skipping")
            continue
        try:
            preds = predict_all_stations(target_date, horizon)
        except FileNotFoundError as exc:
            print(f"WARN: no features for {target_date} horizon {horizon}: {exc}")
            continue

        alerts = evaluate_alerts(preds, state, config, target_date)
        all_alerts.extend(alerts)

    state.save()

    if write_log:
        write_alert_log(all_alerts, log_path)

    notify_stub(all_alerts, config)

    return all_alerts


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run flood alert evaluation")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    ap.add_argument(
        "--horizon",
        choices=HORIZONS,
        default=None,
        help="Single horizon (default: all)",
    )
    ap.add_argument(
        "--all-horizons",
        action="store_true",
        help="Evaluate all horizons",
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write alert log CSV",
    )
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date)
    horizons = [args.horizon] if args.horizon else None

    alerts = run_alerts(target_date, horizons=horizons, write_log=not args.no_log)

    print(f"Alert evaluation complete for {args.date}")
    print(f"  Horizons: {horizons or 'all'}")
    print(f"  Alerts emitted: {len(alerts)}")
    for a in alerts:
        print(
            f"  {a['alert_level'].upper():9s} "
            f"station={a['station_id']:>3} "
            f"horizon={a['horizon']:>3} "
            f"band={a['risk_band']:>10} "
            f"p={a['probability']:.4f} "
            f"reason={a['reason']}"
        )


if __name__ == "__main__":
    main()
