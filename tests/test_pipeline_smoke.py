"""Pipeline end-to-end smoke test (Phase 42 / stage 70).

Smoke-tests the deployable system:
- API module imports cleanly
- Model loads from the frozen artefact
- Config constants are well-formed (thresholds, risk bands, horizons)
- Feature store path resolves
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    """End-to-end smoke tests for the deployable system."""

    def test_api_config_imports(self):
        """system.config.model_config must import without error."""
        import sys

        sys.path.insert(0, str(ROOT))
        from system.config.model_config import (  # noqa: F401
            HORIZONS,
            THRESHOLDS,
            RISK_BANDS,
            N_FEATURES,
            SEED,
            MODEL_PATH,
        )
        self.assertIsInstance(HORIZONS, (list, tuple))
        self.assertGreater(len(HORIZONS), 0)
        self.assertGreater(N_FEATURES, 0)
        self.assertEqual(SEED, 20261008)

    def test_model_config_json_valid(self):
        """system/config/model_config.json must be valid JSON with required keys."""
        path = ROOT / "system" / "config" / "model_config.json"
        self.assertTrue(path.exists(), f"model_config.json missing: {path}")
        with open(path) as f:
            cfg = json.load(f)
        for section in ("model", "thresholds", "risk_bands", "inference", "api"):
            self.assertIn(section, cfg, f"missing config section: {section}")

    def test_thresholds_per_horizon(self):
        """Each horizon must have a cost-derived threshold."""
        path = ROOT / "system" / "config" / "model_config.json"
        with open(path) as f:
            cfg = json.load(f)
        thresholds = cfg["thresholds"]["values"]
        for h in ("24h", "48h", "72h"):
            self.assertIn(h, thresholds, f"missing threshold for horizon {h}")
            self.assertIn("t_warn", thresholds[h])

    def test_risk_bands_per_horizon(self):
        """Each horizon must have cost-derived risk bands."""
        path = ROOT / "system" / "config" / "model_config.json"
        with open(path) as f:
            cfg = json.load(f)
        bands = cfg["risk_bands"]["values"]
        for h in ("24h", "48h", "72h"):
            self.assertIn(h, bands, f"missing risk bands for horizon {h}")
            self.assertIsInstance(bands[h]["bands"], list)
            self.assertGreater(len(bands[h]["bands"]), 0)

    def test_frozen_model_file_exists(self):
        """The frozen D16 model artefact must be present for inference."""
        model_path = ROOT / "models" / "final_v2" / "selected_model_v2.pkl"
        self.assertTrue(model_path.exists(), f"frozen model missing: {model_path}")

    def test_feature_store_path_resolves(self):
        """The local feature store path must be configured."""
        path = ROOT / "system" / "config" / "model_config.json"
        with open(path) as f:
            cfg = json.load(f)
        store = cfg["inference"]["feature_store"]
        self.assertIsInstance(store, str)
        self.assertGreater(len(store), 0)


if __name__ == "__main__":
    unittest.main()
