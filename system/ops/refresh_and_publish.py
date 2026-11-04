"""
Daily refresh and publish orchestrator.

Runs the data pipeline, incrementally updates the precomputed static snapshot,
and publishes only the changed files to the Hugging Face Static Space.

The public site is a Static Space that serves precomputed forecasts. A new date
becomes visible when its prediction file, the trend series, and ``dates.json``
are uploaded; no container rebuild is involved.

Typical local run (Windows Task Scheduler uses this via run_daily.bat):

    .venv\\Scripts\\python.exe system\\ops\\refresh_and_publish.py --catchup

Environment:
    HF_TOKEN               optional; falls back to the Hugging Face token cache
    HF_SPACE               Space repo id (default riyadhuggy/aabhas-flood-outlook)
    HF_FEATURE_DATASET     only needed with --push-features
    HF_RAW_DATASET         only needed with --push-raw
    RENDER_DEPLOY_HOOK     optional; only used with --deploy-hook

Steps that lack their required credentials are skipped with a clear log line
instead of failing silently.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
PYTHON = Path(os.getenv("AABHAS_PYTHON", sys.executable)).resolve()
FEATURE_STORE = ROOT / "system" / "data" / "processed" / "model_ready"
SNAPSHOT_PUBLIC = ROOT / "system" / "web" / "public" / "data" / "snapshot"
FRONTEND_DIST = ROOT / "system" / "web" / "dist"
DEFAULT_SPACE = "riyadhuggy/aabhas-flood-outlook"
DEFAULT_HEALTH_URL = os.getenv("AABHAS_HEALTH_URL", "")

# Paths mirrored to the private raw-input dataset. Each tuple is
# (source path relative to the repo root, destination prefix in the dataset).
RAW_BUNDLE = [
    ("system/data/raw", "system/data/raw"),
    ("data/raw/ffwc", "data/raw/ffwc"),
    ("data/raw/openmeteo", "data/raw/openmeteo"),
    ("data/raw/glofas", "data/raw/glofas"),
    ("data/raw/power_soil", "data/raw/power_soil"),
    ("data/processed/station_spatial_features.csv", "data/processed/station_spatial_features.csv"),
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], label: str, cwd: Path | None = None) -> bool:
    log(f"STEP: {label}")
    log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), check=False)
    if result.returncode != 0:
        log(f"FAILED: {label} (exit {result.returncode})")
        return False
    log(f"OK: {label}")
    return True


def resolve_token() -> str:
    token = os.getenv("HF_TOKEN", "")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token() or ""
    except Exception:
        return ""


def latest_partition() -> str | None:
    if not FEATURE_STORE.is_dir():
        return None
    dates = sorted(
        entry.name.split("=", 1)[1]
        for entry in FEATURE_STORE.iterdir()
        if entry.is_dir() and entry.name.startswith("date=")
    )
    return dates[-1] if dates else None


def _hub():
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: F401
    except ImportError:
        log("huggingface_hub is not installed; run: pip install -r requirements-api.txt")
        return None
    return HfApi, snapshot_download


def pull_raw(dataset: str, token: str, dry_run: bool) -> bool:
    hub = _hub()
    if hub is None:
        return False
    _, snapshot_download = hub
    if dry_run:
        log(f"DRY RUN: would download raw bundle from {dataset} into {ROOT}")
        return True
    log(f"Downloading raw bundle from {dataset} ...")
    try:
        snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            token=token,
            local_dir=str(ROOT),
            allow_patterns=[src for src, _ in RAW_BUNDLE]
            + [f"{src.rstrip('/')}/**" for src, _ in RAW_BUNDLE],
        )
    except Exception as exc:  # network/auth/not-found
        log(f"Raw bundle download failed: {exc}")
        return False
    log("Raw bundle downloaded.")
    return True


def push_raw(dataset: str, token: str, dry_run: bool) -> bool:
    hub = _hub()
    if hub is None:
        return False
    HfApi, _ = hub
    if dry_run:
        log(f"DRY RUN: would upload raw bundle to {dataset}")
        return True
    api = HfApi(token=token)
    try:
        for src, dest in RAW_BUNDLE:
            path = ROOT / src
            if not path.exists():
                log(f"SKIP missing raw path: {src}")
                continue
            log(f"Uploading raw path {src} -> {dataset}:{dest}")
            if path.is_dir():
                api.upload_folder(
                    repo_id=dataset,
                    repo_type="dataset",
                    folder_path=str(path),
                    path_in_repo=dest,
                    token=token,
                    commit_message=f"Update raw inputs from {src}",
                )
            else:
                api.upload_file(
                    repo_id=dataset,
                    repo_type="dataset",
                    path_or_fileobj=str(path),
                    path_in_repo=dest,
                    token=token,
                    commit_message=f"Update {src}",
                )
    except Exception as exc:
        log(f"Raw bundle upload failed: {exc}")
        return False
    log("Raw bundle uploaded.")
    return True


def upload_features(dataset: str, dry_run: bool) -> bool:
    # The uploader resolves the token from HF_TOKEN or the local token cache, so
    # it never has to appear on the command line (and in the log).
    cmd = [
        str(PYTHON),
        "scripts/upload_features_to_hf.py",
        "--feature-store",
        str(FEATURE_STORE),
        "--dataset",
        dataset,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run(cmd, "Upload feature store to HF")


def update_snapshot(trend_days: int, incremental: bool, dry_run: bool) -> bool:
    cmd = [
        str(PYTHON),
        "system/deploy/build_static_snapshot.py",
        "--trend-days",
        str(trend_days),
    ]
    if incremental:
        cmd.append("--incremental")
    if dry_run:
        log("DRY RUN: would run " + " ".join(cmd))
        return True
    return run(cmd, "Update static snapshot" + (" (incremental)" if incremental else " (full)"))


def sync_snapshot_to_dist(dry_run: bool) -> bool:
    """Mirror the snapshot from public/ into the already-built dist/ folder."""
    if not (FRONTEND_DIST / "index.html").exists():
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm:
            log(f"Frontend build missing ({FRONTEND_DIST}) and npm is not available.")
            return False
        if dry_run:
            log(f"DRY RUN: would rebuild frontend in {ROOT / 'system' / 'web'}")
            return True
        return run([npm, "run", "build"], "Build frontend", cwd=ROOT / "system" / "web")

    if not SNAPSHOT_PUBLIC.is_dir():
        log(f"Snapshot folder missing: {SNAPSHOT_PUBLIC}")
        return False

    destination = FRONTEND_DIST / "data" / "snapshot"
    copied = 0
    for path in SNAPSHOT_PUBLIC.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(SNAPSHOT_PUBLIC)
        if (
            not target.exists()
            or target.stat().st_size != path.stat().st_size
            or int(target.stat().st_mtime) < int(path.stat().st_mtime)
        ):
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            copied += 1
    log(f"Synced {copied} changed snapshot file(s) into {destination}")
    return True


def deploy_static(space: str, full: bool, dry_run: bool) -> bool:
    cmd = [
        str(PYTHON),
        "scripts/deploy_static_space.py",
        "--repo",
        space,
    ]
    if not full:
        cmd.append("--changed-only")
    if dry_run:
        cmd.append("--dry-run")
    label = "Publish static snapshot to HF (full)" if full else "Publish static snapshot to HF (changed only)"
    return run(cmd, label)


def trigger_deploy(hook: str, health_url: str, dry_run: bool) -> bool:
    if dry_run:
        log(f"DRY RUN: would POST deploy hook and verify {health_url or '(no health URL)'}")
        return True
    try:
        request = urllib.request.Request(hook, method="POST", data=b"")
        with urllib.request.urlopen(request, timeout=30) as response:
            log(f"Deploy hook responded HTTP {response.status}")
    except Exception as exc:
        log(f"Deploy hook failed: {exc}")
        return False
    return verify_health(health_url) if health_url else True


def verify_health(health_url: str, attempts: int = 20, delay: float = 15.0) -> bool:
    """Poll /health until the service is up, then print its freshness fields."""
    if not health_url:
        log("No health URL configured; skipping health verification.")
        return True
    last_error = "no response"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            log(
                "Health OK: status={status} model={model_version} "
                "latest_feature_date={latest_feature_date} "
                "age_days={latest_feature_age_days}".format(**payload)
            )
            if payload.get("latest_feature_date") is None:
                log("Warning: health did not report latest_feature_date")
            return True
        except Exception as exc:
            last_error = str(exc)
            log(f"Health attempt {attempt}/{attempts} failed: {last_error}")
            time.sleep(delay)
    log(f"Health check failed after {attempts} attempts: {last_error}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh data and publish the static site")
    parser.add_argument("--date", help="Process one date (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--catchup", action="store_true", help="Backfill missing feature dates.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only publish already-built data.")
    parser.add_argument("--skip-static", action="store_true", help="Do not update/publish the static site.")
    parser.add_argument("--static-full", action="store_true", help="Rebuild every date and upload every file.")
    parser.add_argument("--trend-days", type=int, default=90, help="Trend history kept in the snapshot (0 = all).")
    parser.add_argument("--space", default=os.getenv("HF_SPACE", DEFAULT_SPACE))
    parser.add_argument("--push-features", action="store_true", help="Also upload feature partitions to HF.")
    parser.add_argument("--skip-features-upload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pull-raw", action="store_true", help="Download raw inputs from HF first.")
    parser.add_argument("--push-raw", action="store_true", help="Upload changed raw inputs after the run.")
    parser.add_argument("--deploy-hook", action="store_true", help="POST RENDER_DEPLOY_HOOK and verify /health.")
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Publish even if feature quality gate fails (explicit opt-in).",
    )
    args = parser.parse_args()

    token = resolve_token()
    feature_dataset = os.getenv("HF_FEATURE_DATASET", "")
    raw_dataset = os.getenv("HF_RAW_DATASET", "")
    hook = os.getenv("RENDER_DEPLOY_HOOK", "")

    failures: list[str] = []

    if args.pull_raw:
        if not raw_dataset or not token:
            log("SKIP pull-raw: set HF_RAW_DATASET and a Hugging Face token")
        elif not pull_raw(raw_dataset, token, args.dry_run):
            failures.append("pull-raw")

    if not args.skip_pipeline:
        cmd = [str(PYTHON), "system/ops/daily_pipeline.py"]
        if args.catchup:
            cmd.append("--catchup")
        elif args.date:
            cmd += ["--date", args.date]
        if args.allow_degraded:
            cmd.append("--allow-degraded")
        if not run(cmd, "Daily ingestion + feature build"):
            failures.append("pipeline")

    newest = latest_partition()
    if newest is None:
        log("No local feature partitions found; refusing to publish.")
        failures.append("feature-store")
        newest = None
    else:
        log(f"Latest feature partition: {newest}")

    # Quality gate check: refuse to publish degraded snapshots unless explicitly allowed
    if not args.allow_degraded and newest is not None:
        quality_file = ROOT / "system" / "data" / "processed" / "quality" / "latest.json"
        if quality_file.exists():
            import json
            with open(quality_file, "r", encoding="utf-8") as f:
                quality = json.load(f)
            if not quality.get("ok", False):
                log(f"Quality gate FAILED for {newest}: {quality}")
                failures.append("quality-gate")
                log("Refusing to publish degraded snapshot. Use --allow-degraded to override.")
        else:
            log("WARNING: No quality report found; proceeding without quality check")

    if not args.skip_static and newest is not None:
        if not update_snapshot(args.trend_days, incremental=not args.static_full, dry_run=args.dry_run):
            failures.append("snapshot")
        elif not sync_snapshot_to_dist(args.dry_run):
            failures.append("snapshot-sync")
        elif not deploy_static(args.space, full=args.static_full, dry_run=args.dry_run):
            failures.append("static-deploy")

    if args.push_features and not args.skip_features_upload and newest is not None:
        if not feature_dataset:
            log("SKIP feature upload: set HF_FEATURE_DATASET")
        elif not upload_features(feature_dataset, args.dry_run):
            failures.append("feature-upload")

    if args.push_raw:
        if not raw_dataset or not token:
            log("SKIP push-raw: set HF_RAW_DATASET and a Hugging Face token")
        elif not push_raw(raw_dataset, token, args.dry_run):
            failures.append("push-raw")

    if args.deploy_hook:
        if not hook:
            log("SKIP deploy hook: RENDER_DEPLOY_HOOK is not set")
        elif not trigger_deploy(hook, args.health_url, args.dry_run):
            failures.append("deploy")

    if failures:
        log("REFRESH FAILED: " + ", ".join(failures))
        return 1
    log("REFRESH COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
