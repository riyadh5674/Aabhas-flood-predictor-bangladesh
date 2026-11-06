r"""Test runner for the flood-predictor test suite (Phase 42 / stage 70).

Runs all test modules and reports a summary. Usage:

    .venv\Scripts\python.exe tests\run_tests.py

Works from any working directory: the repository root is prepended to
``sys.path`` so the ``tests`` package and ``system`` package resolve.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_alert_engine import (
    AlertEvaluationTests,
    AlertLogTests,
    AlertMappingTests,
    AlertWorkflowTests,
)
from tests.test_api_contract import APIContractTests
from tests.test_feature_regression import FeatureRegressionTests
from tests.test_ingestion_sources import (
    FreshnessTests,
    ManifestTests,
    QualityGateTests,
)
from tests.test_label_contract import LabelContractTests
from tests.test_pipeline_smoke import PipelineSmokeTests
from tests.test_sequence_integrity import SequenceIntegrityTests
from tests.test_split_integrity import SplitIntegrityTests


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in (
        LabelContractTests,
        SplitIntegrityTests,
        SequenceIntegrityTests,
        FeatureRegressionTests,
        PipelineSmokeTests,
        AlertMappingTests,
        AlertEvaluationTests,
        AlertWorkflowTests,
        AlertLogTests,
        ManifestTests,
        FreshnessTests,
        QualityGateTests,
        APIContractTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit non-zero if any test failed or errored
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
