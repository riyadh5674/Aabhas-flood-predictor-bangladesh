# Aabhas Flood Predictor — Flaw Tracker (Updated)

**Generated:** 2026-11-01  
**Updated:** 2026-11-06  
**Status:** Research/prototype complete — D16 frozen, test split closed (`test_rows_read = 0`), automated source refresh + quality gate wired into the daily pipeline, alert engine implemented (external delivery disabled by config), 84 Python + 21 frontend tests passing. Static HF Space remains the published demo; the FastAPI deployment path is ready for a Docker-capable host.  
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
| R-10 | Tests | `vitest` + 21 tests (`freshness`, `risk`, `format`, `aggregate`). `npm run typecheck` and `npm run build` pass. Python: **84 unittest tests** via `tests/run_tests.py` (label/split/sequence/feature regression, pipeline smoke, alert engine, ingestion + quality gate, API contract). |
| R-11 | Stations resilience | Safe loading, error states in StationDetail; snapshot/static builds use precomputed data. |
| R-12 | Query tuning | Backoff on retries, reasonable intervals, cache keys include horizon/date/mode. |
| R-13 | Automated source refresh | GloFAS, NASA POWER soil and ERA5-Land fetchers are idempotent incremental refreshers; `system/ingestion/refresh_sources.py` validates freshness and writes `system/data/processed/source_status/<date>.json`; `system/ops/daily_pipeline.py` runs it as Step 0. `/health` and the freshness banner surface per-source status and a `degraded` flag. |
| R-14 | Feature quality gate | `system/features/validate.py` checks critical-group null shares, required columns/types and active-station row count; writes `system/data/processed/quality/<date>.json` and fails the build unless `--allow-degraded`. Latest 2026-11-05 partition: 112 rows, critical null share 0.0, `passed: true`. |
| R-15 | Snapshot metadata | `station_count` now scopes to the latest feature date (112) and `trend_station_count` reports the 90-day union (127), with `station_count_scope`; health metadata regenerated. |
| R-16 | Alert engine correctness | Same-band re-fire after cooldown, escalation bypass, suppressed-escalation pending state, draft→approval→audit lifecycle, `/alerts/feed` + approve endpoint; no fabricated `last_alert_at`. |
| R-17 | Freeze portability | `scripts/freeze_results.py` resolves `AABHAS_ROOT` / file-relative root and records manifest revision history; `--check` → `13/13`, 0 changed files. |
| R-18 | Thesis | `reports/phase45_thesis.md` written with frozen numbers, architecture, incident analysis, limitations and reproducibility commands. |
| R-19 | Public demo | Static build published to GitHub Pages (`gh-pages`) with a sub-path-aware base, service worker and manifest; investor pitch, demo script, architecture and community docs added. |

---

## Open Items

### P0 — Still requires external action/config
| ID | Area | Issue | Fix |
|---|------|-------|-----|
| O-01 | Data freshness (live) | **Local** sources and feature store are current through `2026-11-05` (GloFAS `2026-11-05`, POWER soil `2026-11-02`, ERA5-Land `2026-11-03`; quality gate green). The public GitHub Pages demo serves the snapshot built on 2026-11-06. The Render service bundles the frozen D16 model plus a tracked **120-day** feature slice (`2026-07-09` → `2026-11-05`), so `/health` reports `model_loaded: true` and `feature_store_dates: 120` and the live API serves real predictions. It is a recently-current demo store, not the full 2,617-date history. | Refresh the bundled slice after each publish (`python scripts/build_demo_store.py --days 120`) or attach the full HF dataset; rebuild/upload the static snapshot and redeploy. Needs a live host for always-on freshness. |
| O-02 | Always-on scheduling | The refresh itself is implemented and wired (`refresh_sources.py` Step 0 → features → quality gate → publish), but execution still depends on one Windows Task Scheduler machine; no managed cloud scheduler is configured. | Add a cloud runner (Cloud Run Job + Scheduler, GitHub Actions cron with secrets, or an agency VM) with retry/lock/run-summary. Documented in `DEPLOYMENT.md` §3. |
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
.\.venv\Scripts\python.exe tests\run_tests.py        # 86 tests, exit 0

# Frozen evidence
.\.venv\Scripts\python.exe scripts\freeze_results.py --check   # 13/13; changed files 0

# Source freshness + feature quality (latest date)
Get-Content system\data\processed\source_status\2026-11-05.json
Get-Content system\data\processed\quality\latest.json

# API
curl http://127.0.0.1:8765/health
curl "http://127.0.0.1:8765/predict/trend?station_id=1&dates=2026-10-25,2026-11-05"
curl -i -X GET "http://127.0.0.1:8765/predict/all?date=2026-11-05"  # check 429 after burst

# Frontend
cd system/web
npm run typecheck
npm run test
npm run build
```

**Note:** The app is explicitly labeled as research/prototype and not an official warning service. Disclaimers are preserved in UI and docs.
