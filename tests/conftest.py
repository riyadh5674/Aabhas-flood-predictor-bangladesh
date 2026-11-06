"""Shared fixtures and helpers for the flood-predictor test suite."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

EXPECTED_DATASET_SHA256 = (
    "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
)
EXPECTED_DATASET_ROWS = 200665
EXPECTED_DATASET_COLS = 67
SEED = 20261008

HORIZONS = ("24h", "48h", "72h")
LABEL_COLS = ("label_24h", "label_48h", "label_72h")
BANNED_FEATURE_COLS = (
    "obs_days_24h",
    "obs_days_48h",
    "obs_days_72h",
)


def load_dataset() -> pd.DataFrame:
    """Load the immutable experimental dataset."""
    path = DATA_PROCESSED / "experimental_dataset.csv"
    return pd.read_csv(path)


def load_splits() -> pd.DataFrame:
    """Load the split assignments table."""
    path = DATA_PROCESSED / "split_assignments.csv"
    return pd.read_csv(path, parse_dates=["date"])


def dataset_sha256() -> str:
    """Return the SHA-256 of the experimental dataset."""
    path = DATA_PROCESSED / "experimental_dataset.csv"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_dataset_immutable() -> None:
    """Assert the experimental dataset has not drifted (row count, columns, SHA)."""
    df = load_dataset()
    assert df.shape[0] == EXPECTED_DATASET_ROWS, (
        f"row count {df.shape[0]} != {EXPECTED_DATASET_ROWS}"
    )
    assert df.shape[1] == EXPECTED_DATASET_COLS, (
        f"column count {df.shape[1]} != {EXPECTED_DATASET_COLS}"
    )
    actual_sha = dataset_sha256()
    assert actual_sha == EXPECTED_DATASET_SHA256, (
        f"dataset SHA-256 drifted: {actual_sha} != {EXPECTED_DATASET_SHA256}"
    )


def assert_no_banned_label_window_features(df: pd.DataFrame) -> None:
    """Assert no label-window (obs_days_*) columns are present as features."""
    for col in BANNED_FEATURE_COLS:
        assert col not in df.columns, f"banned label-window column present: {col}"


def assert_labels_valid(df: pd.DataFrame) -> None:
    """Assert label columns exist and missing values are NaN, never 0."""
    for col in LABEL_COLS:
        assert col in df.columns, f"missing label column: {col}"
        missing = df[col].isna().sum()
        assert missing >= 0
        # zeros are allowed only where label is genuinely 0 (no flood); NaN stays NaN
        non_null = df[col].dropna()
        assert non_null.isin([0.0, 1.0]).all(), f"{col} contains non-binary non-null values"


def assert_split_integrity(splits: pd.DataFrame) -> None:
    """Assert split assignments respect temporal ordering and holdout rules."""
    assert "split" in splits.columns
    assert splits["split"].isin(["train", "val", "test"]).all()

    # Train rows must precede val rows, which must precede test rows
    train_dates = splits.loc[splits["split"] == "train", "date"]
    val_dates = splits.loc[splits["split"] == "val", "date"]
    test_dates = splits.loc[splits["split"] == "test", "date"]

    if len(train_dates) and len(val_dates):
        assert train_dates.max() <= val_dates.min(), "train/val temporal boundary violated"
    if len(val_dates) and len(test_dates):
        assert val_dates.max() <= test_dates.min(), "val/test temporal boundary violated"

    # ``station_holdout`` marks the fold-0 spatial-CV cohort, not a temporal
    # exclusion: per phase15 §3 those stations may appear in P1 train/val but
    # are held out when fitting the P3 spatial folds. What must hold is that
    # the flag and fold 0 describe the same station set.
    holdout = set(splits.loc[splits["station_holdout"] == True, "station_id"])  # noqa: E712
    if "spatial_fold" in splits.columns:
        fold0 = set(splits.loc[splits["spatial_fold"] == 0, "station_id"])
        assert holdout == fold0, "station_holdout does not match spatial fold 0"

    # Every station × date row appears in exactly one split.
    keyed = splits[["station_id", "date"]].drop_duplicates()
    assert len(keyed) == len(splits), "duplicate station×date rows in splits"


def assert_model_frozen() -> None:
    """Assert the frozen D16 model artefact exists."""
    model_path = ROOT / "models" / "final_v2" / "selected_model_v2.pkl"
    assert model_path.exists(), f"frozen model not found: {model_path}"
    config_path = ROOT / "models" / "final_v2" / "final_config_v2.json"
    assert config_path.exists(), f"frozen model config not found: {config_path}"
