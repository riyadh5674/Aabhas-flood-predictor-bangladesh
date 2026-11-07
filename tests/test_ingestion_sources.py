"""Ingestion + feature-quality-gate tests (PRODUCTION_PLAN WP-1..WP-3).

These tests do not hit the network. They verify manifest parsing, freshness
classification, source-table integrity, missing-script handling and the
feature quality gate's pass/fail behaviour.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.features import validate as fv
from system.ingestion import refresh_sources as rs


class ManifestTests(unittest.TestCase):
    def test_reads_valid_manifest_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"end": "2026-11-05"}), encoding="utf-8")
            self.assertEqual(rs._read_manifest_date(path, "end"), date(2026, 11, 5))

    def test_truncates_timestamp_to_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"valid_end": "2026-11-02T00:00:00"}), encoding="utf-8")
            self.assertEqual(rs._read_manifest_date(path, "valid_end"), date(2026, 11, 2))

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(rs._read_manifest_date(Path("does-not-exist.json"), "end"))

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"other": "2026-11-05"}), encoding="utf-8")
            self.assertIsNone(rs._read_manifest_date(path, "end"))

    def test_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(rs._read_manifest_date(path, "end"))


class FreshnessTests(unittest.TestCase):
    def test_ok_within_lag(self):
        status, lag = rs.classify_freshness(date(2026, 11, 4), date(2026, 11, 5), 2)
        self.assertEqual((status, lag), ("ok", 1))

    def test_ok_on_boundary(self):
        status, lag = rs.classify_freshness(date(2026, 11, 3), date(2026, 11, 5), 2)
        self.assertEqual((status, lag), ("ok", 2))

    def test_stale_beyond_lag(self):
        status, lag = rs.classify_freshness(date(2026, 11, 2), date(2026, 11, 5), 2)
        self.assertEqual((status, lag), ("stale", 3))

    def test_future_date_is_not_negative(self):
        status, lag = rs.classify_freshness(date(2026, 11, 6), date(2026, 11, 5), 2)
        self.assertEqual((status, lag), ("ok", 0))

    def test_none_is_failed(self):
        self.assertEqual(rs.classify_freshness(None, date(2026, 11, 5), 2), ("failed", None))

    def test_source_table_is_well_formed(self):
        names = {s["name"] for s in rs.SOURCES}
        self.assertEqual(names, {"glofas", "power_soil", "era5"})
        for source in rs.SOURCES:
            for key in (
                "name",
                "script",
                "date_format",
                "manifest",
                "manifest_date_key",
                "max_lag_days",
                "critical",
            ):
                self.assertIn(key, source, f"{source.get('name')} missing {key}")
            self.assertTrue((ROOT / source["script"]).exists(), source["script"])
            self.assertGreaterEqual(source["max_lag_days"], 0)

    def test_missing_script_is_reported(self):
        result = rs.refresh_source(
            {"name": "ghost", "script": "scripts/definitely_missing.py"}, date(2026, 11, 5), 7
        )
        self.assertEqual(result["status"], "missing")

    def test_dry_run_does_not_write_status(self):
        status_dir = rs.STATUS_DIR
        before = set(status_dir.glob("*.json")) if status_dir.exists() else set()
        code = rs.main(["--dry-run", "--only", "glofas", "--end", "2026-11-05"])
        after = set(status_dir.glob("*.json")) if status_dir.exists() else set()
        self.assertEqual(code, 0)
        self.assertEqual(before, after)


def _frame(rows: int, null_share: float = 0.0) -> pd.DataFrame:
    """Synthetic feature frame with the critical group prefixes."""
    frame = pd.DataFrame(
        {
            "station_id": range(rows),
            "date": pd.Timestamp("2026-11-05"),
            "gf_discharge": [1.0] * rows,
            "ps_soil": [0.2] * rows,
            "ix_interaction": [0.1] * rows,
            "tmax_c": [30.0] * rows,
            "wl_level": [4.0] * rows,
            "rain_24h": [0.0] * rows,
        }
    )
    if null_share:
        n = max(1, int(rows * null_share))
        for col in ("gf_discharge", "ps_soil", "ix_interaction", "tmax_c"):
            frame.loc[: n - 1, col] = None
    return frame


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_dir = fv.QUALITY_DIR
        fv.QUALITY_DIR = Path(self.tmp.name)

    def tearDown(self):
        fv.QUALITY_DIR = self._old_dir
        self.tmp.cleanup()

    def test_clean_frame_passes(self):
        payload = fv.validate_features(_frame(100))
        self.assertTrue(payload["ok"])
        self.assertTrue((Path(self.tmp.name) / "latest.json").exists())
        self.assertTrue((Path(self.tmp.name) / "2026-11-05.json").exists())

    def test_sparse_critical_group_fails(self):
        with self.assertRaises(SystemExit):
            fv.validate_features(_frame(100, null_share=0.50))

    def test_allow_degraded_does_not_raise(self):
        payload = fv.validate_features(_frame(100, null_share=0.50), allow_degraded=True)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["dates"][0]["failures"] != [], True)

    def test_empty_frame_fails(self):
        with self.assertRaises(SystemExit):
            fv.validate_features(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
