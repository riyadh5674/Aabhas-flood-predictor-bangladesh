#!/usr/bin/env python
"""Publish the built static frontend (``system/web/dist``) to the HF Static Space.

The first deployment needed several commits: a single commit carrying ~3,000
build files is slow and frequently times out. This script reproduces that
chunked upload deterministically and uploads ``index.html`` last, so the Space
keeps serving the previous build until the new data is fully in place.

Usage (PowerShell)::

    $env:HF_TOKEN = "hf_xxxxxxxxxxxxx"
    .\\.venv\\Scripts\\python.exe scripts\\deploy_static_space.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\deploy_static_space.py

Requires ``huggingface_hub`` (already in ``requirements-api.txt``).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

DEFAULT_SPACE = "riyadhuggy/aabhas-flood-outlook"
DEFAULT_DIST = "system/web/dist"
DEFAULT_README = "system/deploy/static_space_README.md"
DEFAULT_PREDICTION_PARTS = 6


def files_under(root: Path, sub: str) -> list[Path]:
    base = root / sub
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def build_plan(
    dist: Path,
    readme: Path | None,
    prediction_parts: int,
) -> list[tuple[str, list[Path]]]:
    """Return ordered ``(label, files)`` upload chunks."""
    index_html = dist / "index.html"
    predictions = files_under(dist, "data/snapshot/predictions")
    trends = files_under(dist, "data/snapshot/trends")

    missing = [
        str(path)
        for path in (index_html, dist / "data/snapshot/dates.json")
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Refusing to publish an incomplete build; missing: " + ", ".join(missing)
        )
    if not predictions:
        raise SystemExit(
            "No prediction files in the build. Run system/deploy/build_static_snapshot.py "
            "and rebuild the frontend before deploying."
        )

    # Everything that is not a per-date prediction, a trend series, or the
    # entrypoint: hashed assets, geo data, stations, snapshot metadata.
    reserved = {"index.html"}
    static_chunk: list[Path] = []
    for path in sorted(p for p in dist.rglob("*") if p.is_file()):
        rel = path.relative_to(dist).as_posix()
        if rel in reserved:
            continue
        if rel.startswith("data/snapshot/predictions/") or rel.startswith(
            "data/snapshot/trends/"
        ):
            continue
        static_chunk.append(path)

    per_part = max(1, math.ceil(len(predictions) / prediction_parts))
    prediction_chunks = [
        predictions[i : i + per_part] for i in range(0, len(predictions), per_part)
    ]

    plan: list[tuple[str, list[Path]]] = [("static assets + snapshot metadata", static_chunk)]
    plan += [
        (f"precomputed predictions ({i}/{len(prediction_chunks)})", chunk)
        for i, chunk in enumerate(prediction_chunks, start=1)
    ]
    if trends:
        plan.append(("station trend series", trends))
    # Entrypoint last: the Space keeps serving the previous build until here.
    plan.append(("entrypoint", [index_html]))
    if readme is not None:
        plan.append(("Space README", [readme]))
    return plan


def describe(files: list[Path], dist: Path) -> str:
    total = sum(p.stat().st_size for p in files)
    return f"{len(files):>5} files  {total / 1_048_576:7.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("HF_SPACE", DEFAULT_SPACE))
    parser.add_argument("--dist", default=DEFAULT_DIST)
    parser.add_argument("--readme", default=DEFAULT_README)
    parser.add_argument(
        "--parts",
        type=int,
        default=DEFAULT_PREDICTION_PARTS,
        help=f"Number of prediction commits (default {DEFAULT_PREDICTION_PARTS})",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--message-prefix", default="Deploy Aabhas static build")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Leave the Space README untouched",
    )
    args = parser.parse_args()

    dist = Path(args.dist).resolve()
    readme = None if args.skip_readme else Path(args.readme).resolve()
    if readme is not None and not readme.exists():
        raise SystemExit(f"README not found: {readme}")
    if not (1 <= args.parts <= 20):
        raise SystemExit("--parts must be between 1 and 20")

    plan = build_plan(dist, readme, args.parts)
    print(f"Target Space : {args.repo}")
    print(f"Build folder : {dist}")
    print(f"Upload plan  : {len(plan)} commits\n")
    for i, (label, files) in enumerate(plan, start=1):
        print(f"  {i:>2}. {label:<40} {describe(files, dist)}")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return 0

    if not args.token:
        raise SystemExit("HF token required: set HF_TOKEN or pass --token")

    api = HfApi(token=args.token)
    for i, (label, files) in enumerate(plan, start=1):
        operations = [
            CommitOperationAdd(
                path_in_repo=(
                    "README.md"
                    if label == "Space README"
                    else path.relative_to(dist).as_posix()
                ),
                path_or_fileobj=str(path),
            )
            for path in files
        ]
        message = f"{args.message_prefix}: {label} ({i}/{len(plan)})"
        print(f"\n[{i}/{len(plan)}] {message} — {describe(files, dist)}", flush=True)
        api.create_commit(
            repo_id=args.repo,
            repo_type="space",
            operations=operations,
            commit_message=message,
        )

    remote = set(
        api.list_repo_files(repo_id=args.repo, repo_type="space", revision="main")
    )
    local = {
        p.relative_to(dist).as_posix() for p in dist.rglob("*") if p.is_file()
    }
    missing = sorted(local - remote)
    if missing:
        print(f"\nWARNING: {len(missing)} build files are not visible in the Space repo:")
        for rel in missing[:10]:
            print(f"  - {rel}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return 1

    print(f"\nDone: {len(local)} build files published to https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
