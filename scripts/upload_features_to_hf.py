#!/usr/bin/env python
"""Upload the partitioned feature store to Hugging Face Datasets.

The API expects the production layout::

    data/train/date=YYYY-MM-DD/data.parquet

Uploading the existing partitions (instead of concatenating them into one
file) keeps per-request downloads small: the API fetches only the asked-for
date.

Usage:
    # Set credentials first:
    export HF_TOKEN="hf_xxxxxxxxxxxxx"
    export HF_FEATURE_DATASET="username/aabhas-flood-features"

    # Run upload:
    python scripts/upload_features_to_hf.py

    # Dry run (scan and report only):
    python scripts/upload_features_to_hf.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_PREFIX = "data/train"


def resolve_token(value: str | None) -> str | None:
    """Return an explicit token, else the cached Hugging Face login token."""
    if value:
        return value
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def find_feature_partitions(feature_store: Path) -> list[Path]:
    """Find all date=YYYY-MM-DD/data.parquet files."""
    return sorted(feature_store.glob("date=*/data.parquet"))


def upload_partitions(
    feature_store: Path,
    dataset_id: str,
    token: str,
    prefix: str = DEFAULT_PREFIX,
    revision: str = "main",
    private: bool = False,
) -> None:
    """Upload partitions preserving the date=YYYY-MM-DD folder layout."""
    prefix = prefix.strip("/")
    api = HfApi(token=token)

    print(f"Creating/updating dataset: {dataset_id}")
    create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        token=token,
        private=private,
        exist_ok=True,
    )

    print(f"Uploading {feature_store} -> {dataset_id}:{prefix}/ ...")
    api.upload_folder(
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path=str(feature_store),
        path_in_repo=prefix,
        allow_patterns=["date=*/data.parquet"],
        revision=revision,
        commit_message="Update partitioned feature store (date=YYYY-MM-DD/data.parquet)",
    )

    print(f"✅ Uploaded to https://huggingface.co/datasets/{dataset_id}")


def verify_upload(dataset_id: str, token: str, prefix: str, revision: str) -> int:
    """Return the number of partitions visible in the dataset repo."""
    api = HfApi(token=token)
    files = api.list_repo_files(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
    )
    marker = f"{prefix.strip('/')}/date="
    partitions = [f for f in files if f.startswith(marker) and f.endswith("/data.parquet")]
    dates = sorted(f.split("/date=")[1].split("/")[0] for f in partitions)
    print(f"Verified: {len(partitions)} date partitions")
    if dates:
        print(f"  first: {dates[0]}  last: {dates[-1]}")
    return len(partitions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload feature store to HF Datasets")
    parser.add_argument(
        "--feature-store",
        default="system/data/processed/model_ready",
        help="Path to partitioned feature store (default: system/data/processed/model_ready)",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("HF_FEATURE_DATASET"),
        help="HF Dataset repo ID (e.g., username/aabhas-flood-features). Can use HF_FEATURE_DATASET env var.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF token with write access. Can use HF_TOKEN env var.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Folder inside the dataset repo (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision/branch (default: main)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private (default: public; existing repos keep their visibility)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only, don't upload",
    )
    args = parser.parse_args()

    # Validate inputs
    feature_store = Path(args.feature_store)
    if not feature_store.exists():
        print(f"❌ Feature store not found: {feature_store}")
        return 1

    if not args.dataset:
        print("❌ Dataset ID required. Set --dataset or HF_FEATURE_DATASET env var.")
        return 1

    args.token = resolve_token(args.token)
    if not args.token:
        print("❌ HF token required. Set --token, HF_TOKEN, or run `hf auth login`.")
        return 1

    # Find partitions
    print(f"Scanning feature store: {feature_store}")
    partitions = find_feature_partitions(feature_store)
    print(f"Found {len(partitions)} date partitions")

    if not partitions:
        print("❌ No date=*/data.parquet partitions found.")
        return 1

    if args.dry_run:
        print("\nDry run - would upload these partitions:")
        for p in partitions[:10]:
            print(f"  {p}")
        if len(partitions) > 10:
            print(f"  ... and {len(partitions) - 10} more")
        return 0

    try:
        upload_partitions(
            feature_store=feature_store,
            dataset_id=args.dataset,
            token=args.token,
            prefix=args.prefix,
            revision=args.revision,
            private=args.private,
        )
        verify_upload(args.dataset, args.token, args.prefix, args.revision)
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
