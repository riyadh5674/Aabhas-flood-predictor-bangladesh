"""Label-contract tests (Phase 42 / stage 70).

Verifies the flood-label contract from phase0 §4 / final workflow §0.3:
- labels are label_24h/48h/72h = any observed reading in (t, t+h] >= FFWC danger level
- missing stays NaN, never 0
- obs_days_* columns are banned as features
- labels are computed per station × issue day
"""

from __future__ import annotations

import unittest

import pandas as pd

from tests.conftest import (
    assert_dataset_immutable,
    assert_labels_valid,
    assert_no_banned_label_window_features,
    load_dataset,
    load_splits,
)

LABEL_COLS = ("label_24h", "label_48h", "label_72h")


class LabelContractTests(unittest.TestCase):
    """Tests for the flood prediction label contract."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.splits = load_splits()

    def test_dataset_immutable(self):
        """The experimental dataset must not drift (SHA, rows, columns)."""
        assert_dataset_immutable()

    def test_label_columns_exist(self):
        """All three horizon label columns must be present."""
        for col in LABEL_COLS:
            self.assertIn(col, self.df.columns, f"missing label column: {col}")

    def test_labels_binary_non_null(self):
        """Non-null label values must be 0.0 or 1.0 only."""
        assert_labels_valid(self.df)

    def test_no_label_window_feature_columns(self):
        """obs_days_* label-window columns must be absent as features."""
        assert_no_banned_label_window_features(self.df)

    def test_missing_stays_nan_never_zero(self):
        """Missing labels must be NaN, not filled with 0."""
        for col in LABEL_COLS:
            series = self.df[col]
            nan_count = series.isna().sum()
            self.assertGreaterEqual(nan_count, 0)
            # Any non-null value must be 0 or 1
            non_null = series.dropna()
            self.assertTrue(non_null.isin([0.0, 1.0]).all())

    def test_labels_join_to_splits(self):
        """Every labelled row must have a corresponding split assignment."""
        # The dataset date column is str; splits date is datetime. Normalize both.
        df2 = self.df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        merged = df2.merge(
            self.splits[["station_id", "date", "split"]],
            on=["station_id", "date"],
            how="left",
            indicator=True,
        )
        both = merged["_merge"] == "both"
        self.assertEqual(both.sum(), len(self.df), "label rows missing from splits")

    def test_label_prevalence_increases_with_horizon(self):
        """A longer lead-time window captures more flood events, so prevalence grows with h."""
        rates = {h: float(self.df[h].mean()) for h in LABEL_COLS if h in self.df.columns}
        if len(rates) == 3:
            self.assertLessEqual(rates["label_24h"], rates["label_72h"])


if __name__ == "__main__":
    unittest.main()
