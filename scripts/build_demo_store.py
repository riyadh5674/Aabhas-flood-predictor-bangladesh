"""Bundle a recent slice of the local feature store for self-contained demos.

The full runtime feature store (``system/data/processed/model_ready``) is
gitignored because it is large and reproducible. A deployed demo still needs a
small, tracked slice so the API can serve real model predictions without a
Hugging Face dataset or external credentials.

This script copies the most recent *N* date partitions into
``system/data/demo/model_ready`` and writes a manifest with provenance. The API
automatically falls back to that directory when the full store is absent.

Usage:
    python scripts/build_demo_store.py            # last 120 dates
    python scripts/build_demo_store.py --days 90
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "system" / "data" / "processed" / "model_ready"
TARGET = ROOT / "system" / "data" / "demo" / "model_ready"


def _partitions(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.glob("date=*")
        if (path / "data.parquet").is_file()
    )


def build(days: int) -> dict:
    if days < 1:
        raise SystemExit("--days must be at least 1")
    available = _partitions(SOURCE)
    if not available:
        raise SystemExit(
            f"No feature partitions found under {SOURCE}. "
            "Run the feature pipeline first or download the store from Hugging Face."
        )

    selected = available[-days:]
    TARGET.mkdir(parents=True, exist_ok=True)

    # Remove stale date partitions from previous builds, keep non-date files.
    for child in TARGET.iterdir():
        if child.is_dir() and child.name.startswith("date="):
            shutil.rmtree(child)

    total_bytes = 0
    for part in selected:
        destination = TARGET / part.name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(part / "data.parquet", destination / "data.parquet")
        total_bytes += (destination / "data.parquet").stat().st_size

    manifest = {
        "purpose": "Self-contained demo feature store for the hosted Aabhas API.",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "partition_count": len(selected),
        "first_date": selected[0].name.replace("date=", ""),
        "last_date": selected[-1].name.replace("date=", ""),
        "total_bytes": total_bytes,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "refresh_command": "python scripts/build_demo_store.py --days 120",
        "notes": (
            "This is a recent slice of the model-ready feature store, not the "
            "full research dataset. It exists so the public demo can serve real "
            "D16 predictions without external credentials."
        ),
    }
    (TARGET / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="Number of most recent date partitions to bundle (default: 120)",
    )
    args = parser.parse_args()
    manifest = build(args.days)
    print(
        f"Bundled {manifest['partition_count']} partitions "
        f"({manifest['first_date']} -> {manifest['last_date']}, "
        f"{manifest['total_bytes'] / 1_048_576:.1f} MiB) into "
        f"{TARGET.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
