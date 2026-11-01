"""Test runner for the flood-predictor test suite (Phase 42 / stage 70).

Runs all test modules and reports a summary. Usage:

    .venv\Scripts\python.exe tests\run_tests.py
"""

from __future__ import annotations

import unittest

from tests.test_feature_regression import FeatureRegressionTests
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
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit non-zero if any test failed or errored
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
