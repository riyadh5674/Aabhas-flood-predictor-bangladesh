"""API contract tests (PRODUCTION_PLAN WP-5 / stage 71).

Exercises the FastAPI app in-process with Starlette's TestClient. Verifies
response schemas, validation errors, the /api alias, security headers and
admin auth. No external network calls are made.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the contract tests deterministic and below the rate limits.
os.environ.setdefault("AABHAS_RATE_LIMIT_DISABLED", "true")
os.environ.setdefault("AABHAS_PREVIEW_MODE", "")

from fastapi.testclient import TestClient  # noqa: E402

from system.api.app import app  # noqa: E402

PREDICT_FIELDS = {
    "station_id",
    "date",
    "horizon",
    "probability",
    "band",
    "action",
    "threshold",
    "model_version",
    "computed_at",
}


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()
        dates = cls.client.get("/dates")
        assert dates.status_code == 200, dates.text
        cls.dates = dates.json()["dates"]
        assert cls.dates, "feature store has no dates"
        cls.latest = cls.dates[-1]
        all_resp = cls.client.get("/predict/all", params={"date": cls.latest, "horizon": "24h"})
        assert all_resp.status_code == 200, all_resp.text
        cls.all_rows = all_resp.json()
        assert cls.all_rows, "predict/all returned no rows"
        cls.station_id = int(cls.all_rows[0]["station_id"])

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    # ---- system / model ------------------------------------------------
    def test_health_schema(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in (
            "status",
            "model_loaded",
            "model_version",
            "feature_store_dates",
            "latest_feature_date",
            "latest_feature_age_days",
            "timestamp",
            "api_version",
            "sources",
            "degraded",
        ):
            self.assertIn(key, body)
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["model_version"], "D16")
        self.assertIsInstance(body["degraded"], bool)
        self.assertEqual(resp.headers.get("Cache-Control"), "no-store")
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")

    def test_model_info(self):
        resp = self.client.get("/model/info")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body.get("version"), "D16")

    def test_thresholds_schema(self):
        resp = self.client.get("/thresholds")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("thresholds", body)
        self.assertIn("risk_bands", body)
        for horizon in ("24h", "48h", "72h"):
            self.assertIn(horizon, body["thresholds"])
            self.assertIn(horizon, body["risk_bands"])

    def test_dates_schema(self):
        resp = self.client.get("/dates")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsInstance(resp.json()["dates"], list)
        self.assertEqual(self.dates, sorted(self.dates))

    # ---- inference -----------------------------------------------------
    def test_predict_single_schema(self):
        resp = self.client.get(
            "/predict",
            params={"station_id": self.station_id, "date": self.latest, "horizon": "24h"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(PREDICT_FIELDS.issubset(body), body)
        self.assertEqual(body["station_id"], self.station_id)
        self.assertEqual(body["date"], self.latest)
        self.assertEqual(body["horizon"], "24h")
        self.assertIn(body["band"], {"low", "moderate", "high", "very_high"})

    def test_predict_all_schema(self):
        for horizon in ("24h", "48h", "72h"):
            resp = self.client.get("/predict/all", params={"date": self.latest, "horizon": horizon})
            self.assertEqual(resp.status_code, 200, resp.text)
            rows = resp.json()
            self.assertGreater(len(rows), 0)
            for row in rows[:5]:
                self.assertTrue(PREDICT_FIELDS.issubset(row), row)
                self.assertEqual(row["horizon"], horizon)

    def test_predict_trend_schema(self):
        start = self.dates[max(0, len(self.dates) - 2)]
        resp = self.client.get(
            "/predict/trend",
            params={
                "station_id": self.station_id,
                "dates": f"{start},{self.latest}",
                "horizon": "24h",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["station_id"], self.station_id)
        self.assertEqual(body["horizon"], "24h")
        self.assertIsInstance(body["points"], list)
        self.assertGreaterEqual(len(body["points"]), 1)
        self.assertTrue(PREDICT_FIELDS.issubset(body["points"][0]))

    def test_batch_prediction_schema(self):
        resp = self.client.post(
            "/predict/batch",
            json={"station_ids": [self.station_id], "date": self.latest, "horizon": "48h"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("results", body)
        self.assertEqual(len(body["results"]), 1)
        self.assertTrue(PREDICT_FIELDS.issubset(body["results"][0]))

    # ---- validation ----------------------------------------------------
    def test_invalid_date_rejected(self):
        resp = self.client.get(
            "/predict", params={"station_id": self.station_id, "date": "not-a-date"}
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_invalid_horizon_rejected(self):
        resp = self.client.get(
            "/predict",
            params={"station_id": self.station_id, "date": self.latest, "horizon": "12h"},
        )
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_missing_required_params(self):
        self.assertEqual(self.client.get("/predict").status_code, 422)

    def test_unknown_date_returns_404(self):
        resp = self.client.get("/predict/all", params={"date": "1990-01-01", "horizon": "24h"})
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_batch_rejects_empty_station_list(self):
        resp = self.client.post(
            "/predict/batch",
            json={"station_ids": [], "date": self.latest, "horizon": "24h"},
        )
        self.assertEqual(resp.status_code, 422, resp.text)

    # ---- alerts / monitoring ------------------------------------------
    def test_alerts_feed_schema(self):
        resp = self.client.get("/alerts/feed")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in ("alerts", "pending_count", "total_count"):
            self.assertIn(key, body)
        self.assertIsInstance(body["alerts"], list)
        self.assertGreaterEqual(body["pending_count"], 0)
        self.assertGreaterEqual(body["total_count"], 0)

    def test_alerts_feed_rejects_bad_limit(self):
        resp = self.client.get("/alerts/feed", params={"limit": 0})
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_monitoring_schema(self):
        resp = self.client.get("/monitoring")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in (
            "status",
            "model_loaded",
            "latest_feature_date",
            "sources",
            "degraded",
            "alerts_pending",
        ):
            self.assertIn(key, body)
        self.assertIsInstance(body["degraded"], bool)

    # ---- gateway ------------------------------------------------------
    def test_api_prefix_alias(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["model_version"], "D16")

    def test_admin_requires_key(self):
        resp = self.client.get("/admin/status")
        self.assertIn(resp.status_code, (401, 503), resp.text)
        if resp.status_code == 401:
            self.assertIn("admin", resp.json()["detail"].lower())


class BundledDemoStoreTests(unittest.TestCase):
    """The tracked demo slice keeps the hosted API self-contained."""

    def test_manifest_and_latest_partition_exist(self):
        import json

        from system.config.model_config import BUNDLED_FEATURE_STORE

        manifest_path = BUNDLED_FEATURE_STORE / "manifest.json"
        self.assertTrue(manifest_path.is_file(), manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["partition_count"], 30)
        latest = BUNDLED_FEATURE_STORE / f"date={manifest['last_date']}" / "data.parquet"
        self.assertTrue(latest.is_file(), latest)

    def test_prediction_date_falls_back_when_partition_omits_it(self):
        import pandas as pd

        from system.api.model_loader import D16Model

        model = D16Model()
        model._artefact = {"decision": "D16", "pipelines": {}, "features": []}
        model._features = []
        frame = pd.DataFrame({"station_id": [1, 2]})
        # Exercise the date fallback without unpickling the real model.
        model.predict_proba = lambda _df, _horizon: [0.1, 0.1]  # type: ignore[assignment]
        result = model.predict(frame, "24h", target_date="2026-09-11")
        self.assertEqual(result["date"].tolist(), ["2026-09-11", "2026-09-11"])


if __name__ == "__main__":
    unittest.main()
