# Aabhas Flood Predictor — Flaw Tracker (Updated)

**Generated:** 2026-11-01  
**Updated:** 2026-11-10  
**Status:** Research/prototype complete — D16 frozen, test split closed for this paper (`test_rows_read = 0`; `reports/test_read_ledger.md` discloses the programme-level history), automated source refresh + quality gate wired into the daily pipeline, alert engine implemented (external delivery disabled by config), 89 Python + 21 frontend tests passing. Public static site is live on **GitHub Pages** (snapshot through 2026-11-09); the **Render** free API is live with a 120-day demo slice through 2026-11-05. Both verified 2026-11-10. Remaining P0s are operational: managed always-on scheduling, external sign-off, delivery credentials.  
**Total open items:** 17 (tracked below; all remaining P0s need external accounts/credentials or a deploy)

> This tracker reflects the current repo state. Items marked **Resolved** were fixed in this working tree (API middleware, security headers, rate limiting, freshness banner, ErrorBoundary, PWA/offline shell, trend endpoint, demo gating, tests, path portability).

---

## Legend
| Priority | Meaning | SLA |
|----------|---------|-----|
| **P0** | Blocks production / security / data integrity | Fix immediately |
| **P1** | Major UX/ops gap, degrades reliability | Fix this sprint |
| **P2** | Technical debt, maintainability, performance | Fix next sprint |
| **P3** | Polish, nice-to-have, future-proofing | Fix when capacity |

---

## Resolved (in this checkout)
| ID | Area | Resolution |
|---|------|------------|
| R-01 | API hardening | Security headers on all responses (incl. 401/429), CSP (non-docs), HSTS behind `X-Forwarded-Proto=https`, request IDs, Server-Timing, gzip, CORS allow-list via `AABHAS_ALLOWED_ORIGINS`. |
| R-02 | Rate limiting / DoS | Sliding-window limiter (general + `/predict`), `/health` excluded, `X-RateLimit-*`, `Retry-After` on 429, bounded buckets, toggle via `AABHAS_RATE_LIMIT_DISABLED`. |
| R-03 | Path normalization | `/api/*` stripped before routing (same-origin proxy/ingress friendly). |
| R-04 | Crash safety | `ErrorBoundary` wrapping app, chunk-load auto-reload, safe DatePicker, safe date parsing/formatting. |
| R-05 | Freshness visibility | `DataFreshnessBanner`, `freshness.ts` policy (current/aging/stale/expired/unknown), health polling centralized via `useHealth()`, production/demo config. |
| R-06 | Demo leakage prevention | `DEMO_ENABLED` build-time flag (dev or `VITE_AABHAS_ALLOW_DEMO=true`), stale localStorage `demo` mode reset to `auto` in prod, demo banner unmissable. |
| R-07 | Trend performance | Batched `/predict/trend` (dates list, max 60), legacy per-date fallback, caching in inference. |
| R-08 | PWA/offline | Service worker (offline shell, cache-first assets, network-first data/shell), manifest, PWA registration (prod only). |
| R-09 | Portability | Hardcoded `D:\Flood predictor` roots replaced with `AABHAS_ROOT`/file-relative paths across ingestion/features/ops/alerts/map/ui/backtest; `sys.executable` used in pipeline. |
| R-10 | Tests | `vitest` + 21 tests (`freshness`, `risk`, `format`, `aggregate`). `npm run typecheck` and `npm run build` pass. Python: **89 unittest tests** via `tests/run_tests.py` (label/split/sequence/feature regression, pipeline smoke, alert engine, ingestion + quality gate, API contract). |
| R-11 | Stations resilience | Safe loading, error states in StationDetail; snapshot/static builds use precomputed data. |
| R-12 | Query tuning | Backoff on retries, reasonable intervals, cache keys include horizon/date/mode. |
| R-13 | Automated source refresh | GloFAS, NASA POWER soil and ERA5-Land fetchers are idempotent incremental refreshers; `system/ingestion/refresh_sources.py` validates freshness and writes `system/data/processed/source_status/<date>.json`; `system/ops/daily_pipeline.py` runs it as Step 0. `/health` and the freshness banner surface per-source status and a `degraded` flag. |
| R-14 | Feature quality gate | `system/features/validate.py` checks critical-group null shares, required columns/types and active-station row count; writes `system/data/processed/quality/<date>.json` and fails the build unless `--allow-degraded`. Latest 2026-11-09 partition: 81 rows, critical null share 0.0, `passed: true`. |
| R-15 | Snapshot metadata | `station_count` now scopes to the latest feature date (112) and `trend_station_count` reports the 90-day union (127), with `station_count_scope`; health metadata regenerated. |
| R-16 | Alert engine correctness | Same-band re-fire after cooldown, escalation bypass, suppressed-escalation pending state, draft→approval→audit lifecycle, `/alerts/feed` + approve endpoint; no fabricated `last_alert_at`. |
| R-17 | Freeze portability | `scripts/freeze_results.py` resolves `AABHAS_ROOT` / file-relative root and records manifest revision history; `--check` → `13/13`, 0 changed files. |
| R-18 | Thesis | `reports/phase45_thesis.md` written with frozen numbers, architecture, incident analysis, limitations and reproducibility commands. |
| R-19 | Public demo | Static build published to GitHub Pages (`gh-pages`) with a sub-path-aware base, service worker and manifest; investor pitch, demo script, architecture and community docs added. |
| R-20 | Hosted live API | Render image bundles the frozen D16 model plus a tracked 120-day feature store; `/health` reports `model_loaded: true`, 120 dates and latest `2026-11-05`; `/predict/all`, `/predict/trend`, `/monitoring` and `/alerts/feed` verified live. |
| R-21 | Risk-band consistency | 48 h and 72 h now use the documented four-level cost-derived scheme (`very_high` included) derived from actual D16 validation predictions: 48 h cuts `0.0434 / 0.0868 / 0.563`, 72 h cuts `0.0482 / 0.0963 / 0.5277`. Config, frontend fallback, tests and the precomputed snapshot agree. |
| R-22 | Alert lifecycle | Draft → approve → dispatch is now a single, consistent path: auto-approved alerts are persisted, manual approval dispatches when notifications are enabled, final status is `approved` / `dispatched` / `dispatch_failed`, and every step is hash-chained in the audit log. `/alerts/feed` reads both directories, deduplicates and reports `pending_count` independently of the status filter. |
| R-23 | Inference concurrency | Model loading is guarded by a lock so concurrent cold requests cannot load or replace the singleton twice; `/model/info` canonicalises artefact names (`lgbm` → `lightgbm`, `xgb` → `xgboost`). |
| R-24 | Empty partitions | A valid but zero-row feature partition now returns an empty prediction frame (HTTP 200) instead of raising inside `predict_proba` and returning 500. |
| R-25 | Monitoring correctness | Prometheus `aabhas_uptime_seconds` reports real process uptime (monotonic clock) instead of seconds since the Unix epoch; the admin-or-preview dependency is awaited instead of returning an un-awaited coroutine. |
| R-26 | Static snapshot band refresh | `build_static_snapshot.py --remap-bands` recomputes band/action fields from probabilities and the current config without re-scoring 2,621 dates; snapshot `thresholds.json` and `model.json` are regenerated consistently. |

---

## Open Items

### P0 — Still requires external action/config
| ID | Area | Issue | Fix |
|---|------|-------|-----|
| O-01 | Data freshness (live) | **Local** sources and feature store are current through `2026-11-09` (GloFAS `2026-11-09`, POWER soil `2026-11-06`, ERA5-Land `2026-11-03`; quality gate green, 81 rows, critical null share 0.0). The static GitHub Pages snapshot was rebuilt on 2026-11-10 from the full 2,621-date local store through `2026-11-09`. The Render service bundles the frozen D16 model plus a tracked **120-day** feature slice (`2026-07-09` → `2026-11-05`), so `/health` reports `model_loaded: true` and `feature_store_dates: 120` and the live API serves real predictions. It is a recently-current demo store, not the full 2,621-date history. | Refresh the bundled slice after each publish (`python scripts/build_demo_store.py --days 120`) or attach the full HF dataset; rebuild/upload the static snapshot and redeploy. Needs a live host for always-on freshness. |
| O-02 | Always-on scheduling | The refresh itself is implemented and wired (`refresh_sources.py` Step 0 → features → quality gate → publish). The **free-tier GitHub Actions daily workflow exists** (`.github/workflows/refresh.yml`, 04:30 UTC) but is opt-in: scheduled runs are skipped until `AABHAS_REFRESH_ENABLED=true` and the HF/Render secrets are configured. The current fallback is one Windows Task Scheduler machine; no managed cloud scheduler is configured. | Set the repository variable and secrets, then enable the workflow; or add a Cloud Run Job + Scheduler / agency VM with retry, lock and run-summary. See `docs/FREE_TIER_OPERATIONS.md` and `DEPLOYMENT.md` §3. |
| O-03 | AuthZ (optional) | Public API has no auth; mitigated by rate limiting + single-instance deployment. | If exposed publicly beyond same-origin, add API key/JWT and admin auth. Not required for same-origin static+API behind proxy. |

### P1
| ID | Area | Issue | Fix |
|---|------|-------|-----|
| P1-01 | Map performance | Large SVG, re-renders on interaction. | Memoize `MapRegion`/`MapStation`, throttle pan/zoom, consider virtualization for station list. |
| P1-02 | Bundle size | ~180 KB gz (motion/d3). | Lazy-load map, trim deps or replace heavy utilities where possible. Target < 100 KB gz on 3G. |
| P1-03 | Accessibility | Map roles/aria, chart summaries, focus management. | Add ARIA labels/pressed states, keyboard nav improvements, pattern/icon for colorblind users. |
| P1-04 | StationDetail UX | Some empty states could be clearer; trend gaps now handled. | Polish copy and ensure all error paths show retry actions. |

### P2
| ID | Area | Issue | Fix |
|---|------|-------|-----|
| P2-01 | State complexity | `AppProvider` is large. | Split contexts (Theme/I18n/DateHorizon/DataMode). |
| P2-02 | Component decomposition | `BangladeshMap`, `StationDetail`, `AppShell` do too much. | Extract hooks/components, add `React.memo`. |
| P2-03 | List virtualization | 141 stations rendered. | Virtualize sidebar/list on low-end devices. |
| P2-04 | Design tokens | Incomplete CSS variables. | Complete token system and dark mode map tokens. |
| P2-05 | Component docs | UI primitives undocumented. | Add Storybook or inline docs. |

### P3
| ID | Area | Issue | Fix |
|---|------|-------|-----|
| P3-01 | i18n polish | Bengali typography/line height. | Tune font stack and line-height for `lang="bn"`. |
| P3-02 | Keyboard discoverability | Shortcuts exist but undiscoverable. | Add shortcut hint/tooltip. |
| P3-03 | Analytics/telemetry | None. | Privacy-friendly opt-in telemetry if needed. |
| P3-04 | Visual regression | No screenshot tests. | Add Playwright snapshots for key pages. |
| P3-05 | PWA icons | Basic icons only. | Generate maskable icons and complete manifest. |

---

## Quick Verification
```powershell
# Python suite (from repo root)
.\.venv\Scripts\python.exe tests\run_tests.py        # 89 tests, exit 0

# Frozen evidence
.\.venv\Scripts\python.exe scripts\freeze_results.py --check   # 13/13; changed files 0

# Source freshness + feature quality (latest date)
Get-Content system\data\processed\source_status\2026-11-09.json
Get-Content system\data\processed\quality\latest.json

# API
.\.venv\Scripts\python.exe system\api\run_api.py --port 8000
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/predict/trend?station_id=1&dates=2026-10-25,2026-11-09"
curl -i -X GET "http://127.0.0.1:8000/predict/all?date=2026-11-09"  # check 429 after burst

# Frontend
cd system/web
npm run typecheck
npm run test
npm run build
```

**Note:** The app is explicitly labeled as research/prototype and not an official warning service. Disclaimers are preserved in UI and docs.
