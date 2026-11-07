# Changelog

All notable changes to Aabhas are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [1.1.0] — 2026-11-07

### Added

- **Self-contained live API on Render** — the Docker image now bundles a tracked
  120-day, 112-station feature-store slice (`system/data/demo/model_ready/`), so the
  frozen D16 model serves real predictions with no external credentials.
- `scripts/build_demo_store.py` to refresh the bundled slice from the full local store.
- `AABHAS_FEATURE_STORE` override plus automatic fallback to the bundled store.
- Bundled-store and prediction-date regression tests.

### Changed

- Render is no longer dependent on Hugging Face secrets; the full HF dataset remains
  an optional override for deployment with the complete 2,617-date history.
- The Docker image now includes `system/features` and `system/alerts` so `/monitoring`
  and the alert feed work in production.
- The Python test suite is tracked in git (ad-hoc script ignores are root-anchored).

### Fixed

- Prediction rows now carry the requested date even when an older feature partition
  omits the `date` column.
- Default-branch drift: `main` and `master` now point at the same commit so Render's
  connected branch receives the same code.

### Verified

- 86/86 Python tests pass.
- `/health` reports `model_loaded: true`, `feature_store_dates: 120`.

## [1.0.0] — 2026-11-07

### Added

- **Public static demo on GitHub Pages** — precomputed snapshot, no server required:
  https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/
- Investor brief (`docs/PITCH.md`) and 5-minute demo script (`docs/DEMO.md`).
- Architecture document with data contracts, API surface and deployment topologies
  (`docs/ARCHITECTURE.md`).
- Repository documentation index (`docs/README.md`), contribution guide, security
  policy, code of conduct and machine-readable citation.
- One-command local demo launchers (`scripts/run_demo.ps1`, `scripts/run_demo.sh`).
- GitHub issue and pull-request templates.
- Screenshots of the live application in `docs/screenshots/`.

### Changed

- README restructured for stakeholders: problem, product, evidence, architecture,
  quick start, documentation and roadmap.
- Base-path awareness in the frontend (`AABHAS_BASE`): station/geography fetches,
  service worker scope, manifest and PWA registration now work from a sub-path.
- Production builds can omit source maps with `AABHAS_SOURCEMAP=false`.

### Fixed

- Service-worker shell and data paths were root-relative, which broke project-site
  deployments; paths now derive from the registration scope.

### Verified

- 84/84 Python tests pass.
- 21/21 frontend tests pass; `typecheck` and `build` pass.
- Frozen evidence check 13/13; 0 changed files.
- Feature store and quality gate green through 2026-11-05.

## [0.9.0] — 2026-11-06

- Alert engine with draft → approval → delivery lifecycle and hash-chained audit.
- Automated source refresh (GloFAS, POWER soil, ERA5-Land) with freshness telemetry.
- Mandatory feature quality gate wired into the daily pipeline.
- Freshness banner, error boundary, PWA offline shell, trend batching.
- Thesis report and final scientific evaluation.

## [0.8.0] — 2026-11-01

- D16 frozen production model; test split closed after a single read.
- Cost-derived thresholds and four risk bands per horizon.
- Calibration, robustness, ablation, error analysis and explainability reports.
- FastAPI inference service and React dashboard.

## [0.1.0] — 2026-10-01

- Initial leakage-aware research pipeline: ingestion, features, splits, baselines.
