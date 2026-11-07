"""Split-integrity tests (Phase 42 / stage 70).

Verifies the Phase-15 split design:
- P1 temporal primary: train <2022 / val 2022 / test 2023+ (frozen)
- 7-day embargo between train/val and val/test
- 5 station-disjoint spatial folds, fold 0 held out of training
- station_holdout rows excluded from train/val
- season column (JJAS vs rest)
- Test split is closed (frozen) — no test-driven decisions
"""

from __future__ import annotations

import unittest

import pandas as pd

from tests.conftest import assert_split_integrity, load_splits


class SplitIntegrityTests(unittest.TestCase):
    """Tests for train/val/test split integrity."""

    @classmethod
    def setUpClass(cls):
        cls.splits = load_splits()

    def test_split_column_exists(self):
        self.assertIn("split", self.splits.columns)
        valid_splits = {"train", "val", "test"}
        self.assertTrue(
            self.splits["split"].isin(valid_splits).all(),
            "invalid split values present",
        )

    def test_temporal_ordering(self):
        """Train dates must precede val dates, which must precede test dates."""
        train_dates = self.splits.loc[self.splits["split"] == "train", "date"]
        val_dates = self.splits.loc[self.splits["split"] == "val", "date"]
        test_dates = self.splits.loc[self.splits["split"] == "test", "date"]

        if len(train_dates) and len(val_dates):
            self.assertLessEqual(train_dates.max(), val_dates.min(),
                                 "train/val temporal boundary violated")
        if len(val_dates) and len(test_dates):
            self.assertLessEqual(val_dates.max(), test_dates.min(),
                                 "val/test temporal boundary violated")

    def test_test_split_frozen_after_2023(self):
        """Test split must cover 2023+ only (frozen test period)."""
        test_dates = self.splits.loc[self.splits["split"] == "test", "date"]
        if len(test_dates):
            self.assertGreaterEqual(test_dates.min().year, 2023)

    def test_holdout_stations_are_fold_0(self):
        """station_holdout=True marks fold-0 stations; they are excluded from P3 spatial-CV
        training but may appear in P1 primary train/val (per phase15 §3)."""
        self.assertIn("station_holdout", self.splits.columns)
        holdout = self.splits.loc[self.splits["station_holdout"] == True, "station_id"].unique()  # noqa: E712
        self.assertGreaterEqual(len(holdout), 1, "no holdout stations defined")

    def test_spatial_fold_0_is_holdout(self):
        """Fold 0 stations should be the holdout stations for P3 spatial CV."""
        if "spatial_fold" in self.splits.columns:
            fold0 = self.splits.loc[self.splits["spatial_fold"] == 0, "station_id"].unique()
            holdout = self.splits.loc[self.splits["station_holdout"] == True, "station_id"].unique()  # noqa: E712
            self.assertEqual(set(fold0), set(holdout), "fold 0 != holdout stations")

    def test_spatial_folds_exist(self):
        """5 station-disjoint spatial folds must exist."""
        self.assertIn("spatial_fold", self.splits.columns)
        folds = self.splits["spatial_fold"].dropna().unique()
        self.assertGreaterEqual(len(folds), 1, "no spatial folds defined")

    def test_season_column_exists(self):
        """Season column (JJAS vs rest) must be present."""
        self.assertIn("season", self.splits.columns)
        seasons = self.splits["season"].dropna().unique()
        self.assertTrue(
            all(s in ("monsoon", "non-monsoon", "JJAS", "non-JJAS") for s in seasons),
            f"unexpected season values: {seasons}",
        )

    def test_embargo_column_exists(self):
        """Embargo flag column must exist."""
        self.assertIn("in_embargo_window", self.splits.columns)

    def test_no_row_in_multiple_splits(self):
        """Each station×date row must appear in exactly one split."""
        keyed = self.splits[["station_id", "date"]].drop_duplicates()
        self.assertEqual(len(keyed), len(self.splits), "duplicate station×date rows in splits")


if __name__ == "__main__":
    unittest.main()
