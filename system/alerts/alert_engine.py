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

Approval workflow:
  - All alerts create a DRAFT record
  - watch: auto-approved if configured
  - warning/high: require duty officer approval
  - Approved alerts are dispatched via configured channels
  - All actions logged to immutable audit log

Usage:
    python system/alerts/run_alerts.py --date 2024-07-15 --horizon 24h
    python system/alerts/run_alerts.py --date 2024-07-15 --all-horizons
"""
from __future__ import annotations

import csv
import logging

from system.utils.logging_config import get_logger

logger = get_logger("alerts")
import hashlib
import json
import os
import smtplib
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from system.api.inference import predict_all_stations, model_info
from system.config.model_config import RISK_BANDS, THRESHOLDS, HORIZONS

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
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


# ---------------------------------------------------------------------------
# Audit log (immutable append-only)
# ---------------------------------------------------------------------------

def write_audit_log(event: dict, audit_path: Path) -> None:
    """Write an immutable audit log entry (JSONL)."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # Add integrity hash
    entry = event.copy()
    entry["logged_at"] = datetime.now(timezone.utc).isoformat()
    # Create hash chain: each entry hashes the previous entry's hash + current content
    prev_hash = ""
    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    prev_hash = last_entry.get("hash", "")
        except Exception:
            pass
    content = json.dumps(entry, sort_keys=True) + prev_hash
    entry["hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]
    entry["prev_hash"] = prev_hash[:16] if prev_hash else None
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Delivery adapters
# ---------------------------------------------------------------------------

def _get_secret(secret_name: str) -> str:
    """Retrieve secret from environment or Secret Manager (GCP)."""
    # First try environment variable
    env_val = os.getenv(secret_name.upper().replace("-", "_"))
    if env_val:
        return env_val
    # Try GCP Secret Manager
    try:
        from google.cloud import secretmanager
        project_id = os.getenv("GCP_PROJECT_ID")
        if project_id:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
    except Exception:
        pass
    return ""


# Use centralized secrets module
try:
    from system.utils.secrets import get_secret as _get_secret_module
    def _get_secret(secret_name: str) -> str:
        return _get_secret_module(secret_name) or ""
except ImportError:
    pass  # Use local implementation above


def send_email(alert: dict, config: dict) -> tuple[bool, str]:
    """Send alert via SMTP email."""
    email_cfg = config["notification"].get("email", {})
    if not email_cfg.get("enabled", True):
        return False, "email disabled"
    
    smtp_host = email_cfg.get("smtp_host")
    smtp_port = email_cfg.get("smtp_port", 587)
    username = email_cfg.get("username")
    password = _get_secret(email_cfg.get("password_secret", "smtp-password"))
    from_addr = email_cfg.get("from_address")
    to_addrs = email_cfg.get("to_addresses", [])
    use_tls = email_cfg.get("use_tls", True)
    
    if not all([smtp_host, username, password, from_addr, to_addrs]):
        return False, "email config incomplete"
    
    try:
        msg = EmailMessage()
        msg["Subject"] = f"[Aabhas Flood Alert] {alert['alert_level'].upper()} - Station {alert['station_id']} ({alert['horizon']})"
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        
        body = f"""Aabhas Flood Early-Warning System

Alert Level: {alert['alert_level'].upper()}
Station ID: {alert['station_id']}
Forecast Horizon: {alert['horizon']}
Risk Band: {alert['risk_band']}
Probability: {alert['probability']:.4f}
Previous Band: {alert['previous_band'] or 'N/A'}
Trigger Reason: {alert['reason']}
Triggered At: {alert['triggered_at']}
Date: {alert['date']}

This is an automated alert from the Aabhas flood prediction system.
Research prototype — not an official warning service. FFWC advisories take precedence.

https://aabhas-flood-predictor.onrender.com
"""
        msg.set_content(body)
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, f"email error: {e}"


def send_webhook(alert: dict, config: dict) -> tuple[bool, str]:
    """Send alert via HTTP webhook."""
    webhook_cfg = config["notification"].get("webhook", {})
    url = webhook_cfg.get("url")
    secret = _get_secret(webhook_cfg.get("secret", "webhook-secret"))
    timeout = webhook_cfg.get("timeout_seconds", 10)
    
    if not url:
        return False, "webhook url not configured"
    
    try:
        payload = {
            "alert": alert,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "aabhas-flood-predictor"
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-Aabhas-Signature": hashlib.sha256((secret + data.decode()).encode()).hexdigest() if secret else ""
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"webhook delivered (HTTP {resp.status})"
            return False, f"webhook HTTP {resp.status}"
    except Exception as e:
        return False, f"webhook error: {e}"


def dispatch_alert(alert: dict, config: dict) -> dict:
    """Dispatch alert via all enabled channels. Returns delivery results."""
    results = {}
    channels = config["notification"].get("channels", [])
    
    if "email" in channels:
        ok, msg = send_email(alert, config)
        results["email"] = {"success": ok, "message": msg}
    
    if "webhook" in channels:
        ok, msg = send_webhook(alert, config)
        results["webhook"] = {"success": ok, "message": msg}
    
    # Dashboard is handled via API polling / websocket (not implemented here)
    if "dashboard" in channels:
        results["dashboard"] = {"success": True, "message": "available via /alerts API"}
    
    return results


# ---------------------------------------------------------------------------
# Approval workflow
# ---------------------------------------------------------------------------

DRAFT_DIR = ROOT / "system" / "data" / "processed" / "alerts_draft"
APPROVED_DIR = ROOT / "system" / "data" / "processed" / "alerts_approved"

def create_draft(alert: dict, config: dict) -> str:
    """Create a draft alert record requiring approval."""
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    draft_id = f"{alert['station_id']}_{alert['horizon']}_{alert['date']}_{int(time.time())}"
    draft_path = DRAFT_DIR / f"{draft_id}.json"
    
    auto_approve = config["approval"].get("auto_approve_watch", False)
    requires_approval = config["approval"].get("required", True)
    
    if not requires_approval or (auto_approve and alert["alert_level"] == "watch"):
        # Auto-approve
        alert["status"] = "approved"
        alert["approved_at"] = datetime.now(timezone.utc).isoformat()
        alert["approved_by"] = "auto"
    else:
        alert["status"] = "draft"
        alert["created_at"] = datetime.now(timezone.utc).isoformat()
    
    alert["draft_id"] = draft_id
    draft_path.write_text(json.dumps(alert, indent=2), encoding="utf-8")
    return draft_id


def approve_alert(draft_id: str, approver: str) -> tuple[bool, str]:
    """Approve a draft alert and move to approved directory."""
    draft_path = DRAFT_DIR / f"{draft_id}.json"
    if not draft_path.exists():
        return False, "draft not found"
    
    alert = json.loads(draft_path.read_text(encoding="utf-8"))
    if alert.get("status") != "draft":
        return False, f"alert not in draft state: {alert.get('status')}"
    
    alert["status"] = "approved"
    alert["approved_at"] = datetime.now(timezone.utc).isoformat()
    alert["approved_by"] = approver
    
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    approved_path = APPROVED_DIR / f"{draft_id}.json"
    approved_path.write_text(json.dumps(alert, indent=2), encoding="utf-8")
    draft_path.unlink()
    
    return True, "approved"


def get_pending_approvals() -> list[dict]:
    """Get all alerts awaiting approval."""
    if not DRAFT_DIR.exists():
        return []
    drafts = []
    for f in DRAFT_DIR.glob("*.json"):
        try:
            alert = json.loads(f.read_text(encoding="utf-8"))
            if alert.get("status") == "draft":
                drafts.append(alert)
        except Exception:
            pass
    return sorted(drafts, key=lambda x: x.get("created_at", ""))


# ---------------------------------------------------------------------------
# Notification dispatch (replaces stub)
# ---------------------------------------------------------------------------

def notify_alerts(alerts: list[dict], config: dict) -> None:
    """Process alerts through approval workflow and dispatch."""
    if not config["notification"].get("enabled", False):
        # Still create drafts for audit trail
        for alert in alerts:
            create_draft(alert, config)
        return
    
    for alert in alerts:
        draft_id = create_draft(alert, config)
        
        # If auto-approved, dispatch immediately
        if alert.get("status") == "approved":
            delivery = dispatch_alert(alert, config)
            
            # Write audit log
            audit_entry = {
                "action": "alert_dispatched",
                "draft_id": draft_id,
                "alert": alert,
                "delivery": delivery,
                "dispatched_at": datetime.now(timezone.utc).isoformat(),
            }
            audit_path = ROOT / config["state"].get("audit_log", "system/logs/alert_audit.jsonl")
            write_audit_log(audit_entry, audit_path)
            
            # Move to approved
            approve_alert(draft_id, "auto")


def notify_stub(alerts: list[dict], config: dict) -> None:
    """Legacy stub - kept for backward compatibility."""
    if not config["notification"]["enabled"]:
        return
    for a in alerts:
        logger.info("Alert notification", extra={
            "alert_level": a['alert_level'].upper(),
            "station_id": a['station_id'],
            "horizon": a['horizon'],
            "risk_band": a['risk_band'],
            "probability": a['probability'],
        })


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
    audit_path = ROOT / config["state"].get("audit_log", "system/logs/alert_audit.jsonl")
    state = AlertState(state_path)

    all_alerts: list[dict] = []

    for horizon in horizons:
        if horizon not in HORIZONS:
            logger.warning("Unknown horizon", extra={"horizon": horizon})
            continue
        try:
            preds = predict_all_stations(target_date, horizon)
        except FileNotFoundError as exc:
            logger.warning("No features for date/horizon", extra={"date": str(target_date), "horizon": horizon, "error": str(exc)})
            continue

        alerts = evaluate_alerts(preds, state, config, target_date)
        all_alerts.extend(alerts)

    state.save()

    if write_log:
        write_alert_log(all_alerts, log_path)

    # Process through approval workflow and dispatch
    notify_alerts(all_alerts, config)

    # Write audit log for evaluation
    for alert in all_alerts:
        audit_entry = {
            "action": "alert_evaluated",
            "alert": alert,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_audit_log(audit_entry, audit_path)

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

    logger.info("Alert evaluation complete", extra={
        "date": args.date,
        "horizons": horizons or "all",
        "alerts_emitted": len(alerts),
    })
    for a in alerts:
        logger.info("Alert emitted", extra={
            "alert_level": a['alert_level'].upper(),
            "station_id": a['station_id'],
            "horizon": a['horizon'],
            "risk_band": a['risk_band'],
            "probability": round(a['probability'], 4),
            "reason": a['reason'],
        })


if __name__ == "__main__":
    main()
