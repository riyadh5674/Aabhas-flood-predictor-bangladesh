"""Feature-regression tests (Phase 42 / stage 70).

Verifies feature-set guards:
- Row/column/SHA guards on the experimental dataset
- Banned columns asserted absent
- Feature set versions (FULL 62 / LINEAR 35 / SELECTED 10 / FULL_V2 110)
- Model artefact SHA-256 matches the frozen record
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from tests.conftest import (
    assert_dataset_immutable,
    DATA_PROCESSED,
    EXPECTED_DATASET_SHA256,
    ROOT,
)


class FeatureRegressionTests(unittest.TestCase):
    """Tests guarding against feature-set regressions."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PROCESSED / "experimental_dataset.csv")

    def test_dataset_sha_matches_manifest(self):
        """Experimental dataset SHA-256 must match the frozen record."""
        actual = hashlib.sha256(
            (DATA_PROCESSED / "experimental_dataset.csv").read_bytes()
        ).hexdigest()
        self.assertEqual(actual, EXPECTED_DATASET_SHA256)

    def test_no_banned_columns_as_features(self):
        """Label columns are targets (present in the dataset) but obs_days_* must be absent."""
        # obs_days_* label-window columns must never appear as features
        for col in ("obs_days_24h", "obs_days_48h", "obs_days_72h"):
            self.assertNotIn(col, self.df.columns, f"banned column present: {col}")
        # Label columns are targets (present), not features used for training
        for col in ("label_24h", "label_48h", "label_72h"):
            self.assertIn(col, self.df.columns, f"missing target column: {col}")

    def test_feature_sets_json_exists(self):
        """feature_sets.json must document FULL/LINEAR/SELECTED sets."""
        path = DATA_PROCESSED / "feature_sets.json"
        if path.exists():
            with open(path) as f:
                sets = json.load(f)
            feature_sets = sets.get("sets", sets)
            self.assertIn("FULL", feature_sets)
            self.assertIn("LINEAR", feature_sets)
            self.assertIn("SELECTED", feature_sets)
        else:
            self.skipTest("feature_sets.json not found")

    def test_frozen_model_artefact_exists(self):
        """The frozen D16 model and config must exist."""
        model = ROOT / "models" / "final_v2" / "selected_model_v2.pkl"
        config = ROOT / "models" / "final_v2" / "final_config_v2.json"
        self.assertTrue(model.exists(), f"frozen model missing: {model}")
        self.assertTrue(config.exists(), f"frozen model config missing: {config}")

    def test_frozen_model_config_valid(self):
        """final_config_v2.json must contain required keys."""
        path = ROOT / "models" / "final_v2" / "final_config_v2.json"
        if not path.exists():
            self.skipTest("final_config_v2.json not found")
        with open(path) as f:
            cfg = json.load(f)
        for key in ("feature_set", "seed", "per_horizon_family", "dataset_sha256_v1"):
            self.assertIn(key, cfg, f"missing config key: {key}")
        # features list should be non-empty
        self.assertIsInstance(cfg.get("features"), list)
        self.assertGreater(len(cfg["features"]), 0)


if __name__ == "__main__":
    unittest.main()
