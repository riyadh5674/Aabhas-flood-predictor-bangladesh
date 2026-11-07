"""Alert engine unit tests (Phase 45 / PRODUCTION_PLAN WP-4).

Covers band -> level mapping, cooldown re-fire, escalation bypass, state
persistence, draft/approval workflow, audit hash chaining and disabled
delivery. No network calls and no real SMTP/webhook is touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.alerts import alert_engine as ae


def _config(escalation_bypass: bool = True, cooldown_hours: int = 24) -> dict:
    return {
        "cooldown": {
            "cooldown_hours": cooldown_hours,
            "escalation_bypass_cooldown": escalation_bypass,
        },
        "notification": {"enabled": False, "channels": []},
        "approval": {"required": True, "auto_approve_watch": False},
        "state": {"audit_log": "system/logs/alert_audit.jsonl"},
    }


def _prediction(band: str, station_id: int = 7, horizon: str = "24h") -> pd.DataFrame:
    return pd.DataFrame(
        [{"station_id": station_id, "horizon": horizon, "band": band, "probability": 0.9}]
    )


class AlertMappingTests(unittest.TestCase):
    def test_band_to_level(self):
        self.assertIsNone(ae.band_to_alert_level("low"))
        self.assertEqual(ae.band_to_alert_level("moderate"), "watch")
        self.assertEqual(ae.band_to_alert_level("high"), "warning")
        self.assertEqual(ae.band_to_alert_level("very_high"), "high")
        self.assertIsNone(ae.band_to_alert_level("unknown"))

    def test_severity_order(self):
        self.assertTrue(ae.is_worse("high", "warning"))
        self.assertTrue(ae.is_worse("warning", "watch"))
        self.assertFalse(ae.is_worse("watch", "warning"))
        self.assertFalse(ae.is_worse(None, "watch"))


class AlertEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = ae.AlertState(Path(self.tmp.name) / "state.json")
        self.config = _config()
        self.day = date(2026, 7, 15)
        self.t0 = datetime(2026, 7, 15, 6, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_initial_detection_alerts(self):
        alerts = ae.evaluate_alerts(
            _prediction("high"), self.state, self.config, self.day, now=self.t0
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["reason"], "initial_detection")
        self.assertEqual(alerts[0]["alert_level"], "warning")

    def test_low_does_not_alert(self):
        alerts = ae.evaluate_alerts(
            _prediction("low"), self.state, self.config, self.day, now=self.t0
        )
        self.assertEqual(alerts, [])
        self.assertIsNone(self.state.get(7, "24h")["alert_level"])

    def test_same_band_within_cooldown_is_suppressed(self):
        ae.evaluate_alerts(_prediction("high"), self.state, self.config, self.day, now=self.t0)
        alerts = ae.evaluate_alerts(
            _prediction("high"),
            self.state,
            self.config,
            self.day,
            now=self.t0 + timedelta(hours=1),
        )
        self.assertEqual(alerts, [])

    def test_same_band_refires_after_cooldown(self):
        ae.evaluate_alerts(_prediction("high"), self.state, self.config, self.day, now=self.t0)
        alerts = ae.evaluate_alerts(
            _prediction("high"),
            self.state,
            self.config,
            self.day,
            now=self.t0 + timedelta(hours=25),
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["reason"], "cooldown_expired")

    def test_escalation_bypasses_cooldown_when_enabled(self):
        ae.evaluate_alerts(_prediction("moderate"), self.state, self.config, self.day, now=self.t0)
        alerts = ae.evaluate_alerts(
            _prediction("very_high"),
            self.state,
            self.config,
            self.day,
            now=self.t0 + timedelta(minutes=5),
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["reason"], "escalation")
        self.assertEqual(alerts[0]["alert_level"], "high")

    def test_escalation_waits_for_cooldown_when_bypass_disabled(self):
        config = _config(escalation_bypass=False)
        ae.evaluate_alerts(_prediction("moderate"), self.state, config, self.day, now=self.t0)
        alerts = ae.evaluate_alerts(
            _prediction("very_high"),
            self.state,
            config,
            self.day,
            now=self.t0 + timedelta(minutes=5),
        )
        self.assertEqual(alerts, [])
        alerts = ae.evaluate_alerts(
            _prediction("very_high"),
            self.state,
            config,
            self.day,
            now=self.t0 + timedelta(hours=25),
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["reason"], "escalation")

    def test_return_to_low_clears_state_and_allows_fresh_detection(self):
        ae.evaluate_alerts(_prediction("high"), self.state, self.config, self.day, now=self.t0)
        ae.evaluate_alerts(
            _prediction("low"),
            self.state,
            self.config,
            self.day,
            now=self.t0 + timedelta(hours=1),
        )
        alerts = ae.evaluate_alerts(
            _prediction("high"),
            self.state,
            self.config,
            self.day,
            now=self.t0 + timedelta(hours=2),
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["reason"], "initial_detection")

    def test_state_persists_across_instances(self):
        path = Path(self.tmp.name) / "state.json"
        ae.evaluate_alerts(_prediction("high"), self.state, self.config, self.day, now=self.t0)
        self.state.save()
        reloaded = ae.AlertState(path)
        self.assertEqual(reloaded.get(7, "24h")["band"], "high")
        self.assertEqual(reloaded.get(7, "24h")["alert_level"], "warning")


class AlertWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self._old_draft = ae.DRAFT_DIR
        self._old_approved = ae.APPROVED_DIR
        ae.DRAFT_DIR = self.base / "drafts"
        ae.APPROVED_DIR = self.base / "approved"
        self.config = _config()

    def tearDown(self):
        ae.DRAFT_DIR = self._old_draft
        ae.APPROVED_DIR = self._old_approved
        self.tmp.cleanup()

    def _alert(self):
        return {
            "station_id": 7,
            "date": "2026-07-15",
            "horizon": "24h",
            "risk_band": "high",
            "probability": 0.9,
            "previous_band": "moderate",
            "alert_level": "warning",
            "triggered_at": "2026-07-15T06:00:00+00:00",
            "reason": "escalation",
        }

    def test_disabled_delivery_still_writes_draft(self):
        ae.notify_alerts([self._alert()], self.config)
        pending = ae.get_pending_approvals()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"], "draft")

    def test_approve_moves_draft_to_approved(self):
        draft_id = ae.create_draft(self._alert(), self.config)
        ok, message = ae.approve_alert(draft_id, "duty_officer")
        self.assertTrue(ok, message)
        self.assertFalse((ae.DRAFT_DIR / f"{draft_id}.json").exists())
        approved = json.loads((ae.APPROVED_DIR / f"{draft_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["approved_by"], "duty_officer")

    def test_approving_twice_fails(self):
        draft_id = ae.create_draft(self._alert(), self.config)
        ae.approve_alert(draft_id, "duty_officer")
        ok, _ = ae.approve_alert(draft_id, "duty_officer")
        self.assertFalse(ok)

    def test_auto_approve_disabled_requires_manual(self):
        draft_id = ae.create_draft(self._alert(), self.config)
        record = json.loads((ae.DRAFT_DIR / f"{draft_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "draft")

    def test_audit_log_chains_hashes(self):
        audit = self.base / "audit.jsonl"
        ae.write_audit_log({"action": "one"}, audit)
        ae.write_audit_log({"action": "two"}, audit)
        lines = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["prev_hash"], None)
        self.assertEqual(lines[1]["prev_hash"], lines[0]["hash"])
        self.assertNotEqual(lines[0]["hash"], lines[1]["hash"])

    def test_delivery_adapters_disabled_by_default(self):
        config = {
            "notification": {
                "enabled": False,
                "channels": ["email", "webhook"],
                "email": {"enabled": False},
                "webhook": {"url": ""},
            }
        }
        results = ae.dispatch_alert(self._alert(), config)
        self.assertFalse(results["email"]["success"])
        self.assertFalse(results["webhook"]["success"])


class AlertLogTests(unittest.TestCase):
    def test_write_log_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.csv"
            alert = {
                "station_id": 7,
                "date": "2026-07-15",
                "horizon": "24h",
                "risk_band": "high",
                "probability": 0.9,
                "previous_band": None,
                "alert_level": "warning",
                "triggered_at": "2026-07-15T06:00:00+00:00",
                "reason": "initial_detection",
            }
            ae.write_alert_log([alert], path)
            ae.write_alert_log([alert], path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("station_id"), 1)
            self.assertEqual(len(text.strip().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
