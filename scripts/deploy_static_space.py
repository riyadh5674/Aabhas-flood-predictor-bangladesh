#!/usr/bin/env python
"""Publish the built static frontend (``system/web/dist``) to the HF Static Space.

Two modes:

* full (default) — chunked, deterministic upload of every build file with
  ``index.html`` last. Used for bootstrap and for code/asset changes.
* ``--changed-only`` — compare each file against a local state manifest
  (SHA-256) and upload only new/changed files. Used by the daily data refresh,
  where a few hundred small snapshot files change and everything else does not.

Both modes keep the Space serving the previous build until the data files are
in place: prediction/trend/metadata files are committed before ``index.html``.

Usage (PowerShell)::

    .\\.venv\\Scripts\\python.exe scripts\\deploy_static_space.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\deploy_static_space.py
    .\\.venv\\Scripts\\python.exe scripts\\deploy_static_space.py --changed-only

Requires ``huggingface_hub`` (already in ``requirements-api.txt``). The token is
read from ``HF_TOKEN`` or the local Hugging Face token cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = "riyadhuggy/aabhas-flood-outlook"
DEFAULT_DIST = ROOT / "system" / "web" / "dist"
DEFAULT_README = ROOT / "system" / "deploy" / "static_space_README.md"
DEFAULT_STATE = ROOT / "system" / "deploy" / ".deploy_state.json"
DEFAULT_PREDICTION_PARTS = 6
DEFAULT_BATCH_FILES = 300


def resolve_token(value: str | None) -> str | None:
    if value:
        return value
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def files_under(root: Path, sub: str) -> list[Path]:
    base = root / sub
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path, repo: str) -> dict:
    if not path.exists():
        return {"repo": repo, "files": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"repo": repo, "files": {}}
    if state.get("repo") != repo:
        return {"repo": repo, "files": {}}
    state.setdefault("files", {})
    return state


def save_state(path: Path, repo: str, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"repo": repo, "files": files}, separators=(",", ":")),
        encoding="utf-8",
    )


def build_plan(
    dist: Path,
    readme: Path | None,
    prediction_parts: int,
) -> list[tuple[str, list[Path]]]:
    """Return ordered ``(label, files)`` upload chunks for a full deploy."""
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


def describe(files: list[Path], base: Path) -> str:
    total = sum(p.stat().st_size for p in files)
    return f"{len(files):>5} files  {total / 1_048_576:7.1f} MB"


def relative_for(path: Path, dist: Path, readme: Path | None) -> str:
    if readme is not None and path == readme:
        return "README.md"
    return path.relative_to(dist).as_posix()


def commit_files(
    api: HfApi,
    repo: str,
    items: list[tuple[str, Path]],
    message: str,
) -> None:
    operations = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
        for rel, path in items
    ]
    api.create_commit(
        repo_id=repo,
        repo_type="space",
        operations=operations,
        commit_message=message,
    )


def changed_priority(rel: str) -> int:
    if rel.startswith("data/snapshot/predictions/"):
        return 0
    if rel.startswith("data/snapshot/trends/"):
        return 1
    if rel.startswith("data/snapshot/") and rel.endswith(".json"):
        return 2  # dates/health/model/thresholds after the data they describe
    if rel == "index.html":
        return 4
    return 3


def publish_changed(
    api: HfApi,
    repo: str,
    dist: Path,
    readme: Path | None,
    state_path: Path,
    batch_files: int,
    dry_run: bool,
) -> int:
    state = load_state(state_path, repo)
    files = state.get("files", {})

    entries: list[tuple[int, str, Path, str]] = []
    for path in sorted(p for p in dist.rglob("*") if p.is_file()):
        rel = path.relative_to(dist).as_posix()
        digest = sha256_file(path)
        if files.get(rel) != digest:
            entries.append((changed_priority(rel), rel, path, digest))
    readme_entry: tuple[int, str, Path, str] | None = None
    if readme is not None and readme.exists():
        digest = sha256_file(readme)
        if files.get("README.md") != digest:
            readme_entry = (5, "README.md", readme, digest)

    if not entries and readme_entry is None:
        print("No changed files; nothing to upload.")
        return 0

    entries.sort(key=lambda item: (item[0], item[1]))
    index_items = [item for item in entries if item[1] == "index.html"]
    data_items = [item for item in entries if item[1] != "index.html"]

    total = len(entries) + (1 if readme_entry else 0)
    if dry_run:
        print(f"Changed-only dry run: {total} file(s) would be uploaded")
        for _, rel, path, _ in entries[:20]:
            print(f"  {rel}")
        if len(entries) > 20:
            print(f"  ... and {len(entries) - 20} more")
        return 0

    batches = [
        data_items[i : i + batch_files] for i in range(0, len(data_items), batch_files)
    ]
    uploaded = 0
    for index, batch in enumerate(batches, start=1):
        items = [(rel, path) for _, rel, path, _ in batch]
        label = f"data refresh ({index}/{len(batches)})"
        print(f"[{index}/{len(batches)}] {label} — {len(items)} files", flush=True)
        commit_files(api, repo, items, f"Deploy Aabhas static build: {label}")
        for _, rel, _, digest in batch:
            files[rel] = digest
        save_state(state_path, repo, files)
        uploaded += len(batch)

    if readme_entry is not None:
        _, rel, path, digest = readme_entry
        print("[README] Space README", flush=True)
        commit_files(api, repo, [(rel, path)], "Deploy Aabhas static build: README")
        files[rel] = digest
        save_state(state_path, repo, files)
        uploaded += 1

    if index_items:
        items = [(rel, path) for _, rel, path, _ in index_items]
        print("[entrypoint] index.html last", flush=True)
        commit_files(api, repo, items, "Deploy Aabhas static build: entrypoint")
        for _, rel, _, digest in index_items:
            files[rel] = digest
        save_state(state_path, repo, files)
        uploaded += len(index_items)

    print(f"\nDone: {uploaded} changed file(s) published to https://huggingface.co/spaces/{repo}")
    return 0


def record_full_state(repo: str, dist: Path, readme: Path | None, state_path: Path) -> int:
    files: dict[str, str] = {}
    for path in sorted(p for p in dist.rglob("*") if p.is_file()):
        files[path.relative_to(dist).as_posix()] = sha256_file(path)
    if readme is not None and readme.exists():
        files["README.md"] = sha256_file(readme)
    save_state(state_path, repo, files)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("HF_SPACE", DEFAULT_SPACE))
    parser.add_argument("--dist", default=str(DEFAULT_DIST))
    parser.add_argument("--readme", default=str(DEFAULT_README))
    parser.add_argument(
        "--parts",
        type=int,
        default=DEFAULT_PREDICTION_PARTS,
        help=f"Number of prediction commits (default {DEFAULT_PREDICTION_PARTS})",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--message-prefix", default="Deploy Aabhas static build")
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Upload only files whose SHA-256 differs from the local state manifest.",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE),
        help="Manifest of files already published (changed-only mode).",
    )
    parser.add_argument(
        "--batch-files",
        type=int,
        default=DEFAULT_BATCH_FILES,
        help=f"Files per commit in changed-only mode (default {DEFAULT_BATCH_FILES}).",
    )
    parser.add_argument("--verify", action="store_true", help="List remote files and check completeness.")
    parser.add_argument(
        "--seed-state",
        action="store_true",
        help="Record the current build as already published (no upload); use once when the remote is known to match.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Leave the Space README untouched",
    )
    args = parser.parse_args()

    dist = Path(args.dist).resolve()
    readme = None if args.skip_readme else Path(args.readme).resolve()
    state_path = Path(args.state_file).resolve()
    if readme is not None and not readme.exists():
        raise SystemExit(f"README not found: {readme}")
    if not (1 <= args.parts <= 20):
        raise SystemExit("--parts must be between 1 and 20")
    if args.batch_files < 1:
        raise SystemExit("--batch-files must be positive")

    if args.seed_state:
        if not dist.is_dir():
            raise SystemExit(f"Build folder not found: {dist}")
        count = record_full_state(args.repo, dist, readme, state_path)
        print(f"Recorded {count} file hashes in {state_path}")
        return 0

    print(f"Target Space : {args.repo}")
    print(f"Build folder : {dist}")
    print(f"Mode         : {'changed-only' if args.changed_only else 'full'}")

    token = resolve_token(args.token)
    if not token and not args.dry_run:
        raise SystemExit("HF token required: set HF_TOKEN, run `hf auth login`, or pass --token")

    api = HfApi(token=token) if token else None

    if args.changed_only:
        if not (dist / "index.html").exists():
            raise SystemExit(f"Incomplete build; missing {dist / 'index.html'}")
        return publish_changed(
            api,
            args.repo,
            dist,
            readme,
            state_path,
            args.batch_files,
            args.dry_run,
        )

    plan = build_plan(dist, readme, args.parts)
    print(f"Upload plan  : {len(plan)} commits\n")
    for i, (label, files) in enumerate(plan, start=1):
        print(f"  {i:>2}. {label:<40} {describe(files, dist)}")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return 0

    assert api is not None
    for i, (label, files) in enumerate(plan, start=1):
        items = [(relative_for(path, dist, readme), path) for path in files]
        message = f"{args.message_prefix}: {label} ({i}/{len(plan)})"
        print(f"\n[{i}/{len(plan)}] {message} — {describe(files, dist)}", flush=True)
        commit_files(api, args.repo, items, message)

    record_full_state(args.repo, dist, readme, state_path)

    if args.verify:
        remote = set(
            api.list_repo_files(repo_id=args.repo, repo_type="space", revision="main")
        )
        local = {p.relative_to(dist).as_posix() for p in dist.rglob("*") if p.is_file()}
        missing = sorted(local - remote)
        if missing:
            print(f"\nWARNING: {len(missing)} build files are not visible in the Space repo:")
            for rel in missing[:10]:
                print(f"  - {rel}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            return 1

    print(f"\nDone: published to https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
