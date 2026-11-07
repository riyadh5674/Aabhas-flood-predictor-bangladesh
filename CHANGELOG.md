# Changelog

All notable changes to Aabhas are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

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
