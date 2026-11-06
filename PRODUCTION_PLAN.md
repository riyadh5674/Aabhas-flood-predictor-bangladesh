# Aabhas Flood Predictor — Completion & Production Plan

**Status date:** 2026-11-04
**References:** `final workflow.md` (Phases 35–45 / stages 62–73), `FLAWS.md`, current repo, live static Space.
**Purpose:** define exactly what remains to make the project fully dynamic and fit for a government organisation.

> **Status update (2026-11-06):** the P0 data defect described in §0 is fixed. WP-1…WP-5 are
> implemented — the three v2 sources are idempotent incremental refreshers,
> `system/ingestion/refresh_sources.py` runs as **Step 0** of `daily_pipeline.py`, the feature
> quality gate writes `system/data/processed/quality/<date>.json`, and the local feature store is
> current through **2026-11-05** with critical null share **0.0**. The alert engine now has
> approval, audit and `/alerts` endpoints with tests (external delivery still disabled), and
> `reports/phase45_thesis.md` completes stage 73. Remaining work is **operational**: paid
> always-on hosting / redeploy, formal threshold sign-off plus delivery credentials, and
> republishing the public snapshot. The diagnosis below is retained as the 2026-11-04 history.

---

## 0. Bottom line

The final workflow lists 12 system/closure stages (62–73). Ten are marked DONE, but two of
those "DONE" labels are only structurally true, and one stage is unfinished:

| Reality | Impact |
|---|---|
| **37 of the model's 110 features are 100% null** on every recent date (all `gf_*` and `ps_*`), and the 5 `ix_*` interactions are built from them. | 42 of 110 inputs (38%) are null or derived from null. The frozen D16 model is running degraded right now. |
| The three v2 raw feeds (**GloFAS, NASA POWER soil, ERA5-Land**) all stop at **2026-10-12**. | The model has been running blind on hydrology/soil for ~3 weeks. |
| The daily pipeline only refreshes **FFWC water levels + rainfall**. | Nothing will fix the above automatically; it will get worse. |
| Production automation is a **single Windows PC + Task Scheduler**. | One powered-off machine stops the service. Not government-grade. |
| The **live FastAPI service is down** (Render free, HTTP 000). Only the static snapshot works. | "Real-time" is really "yesterday's precomputed snapshot". |
| The alert engine only writes **CSV + a stub**. Deliveries are disabled. | Stage 66 is not operational. |
| There was **no thesis** (`reports/` Phase 45 missing). | Stage 73 was TODO (now written). |

The workflow also states explicitly: *"Everything here is a research prototype, never an
official warning service."* If a government body will use this operationally, that label can
only be removed after formal validation and sign-off. That is a governance workstream, not a
code change. See §7.

---

## 1. What "complete" means per the workflow

Phases 35–45 map to stages 62–73:

| Stage | Requirement | Current |
|---|---|---|
| 62 | Location-based prediction | DONE |
| 63 | Bangladesh flood-risk map | DONE |
| 64 | **Automatic data acquisition** — source → ingestion → validation → preprocessing → features → model; raw append-only; QC on read | **DONE** — incremental source refresh + freshness telemetry + quality gate |
| 65 | Real-time prediction pipeline | DONE (local/system); public live redeploy pending |
| 66 | **Early-warning engine** — probability → threshold → decision → alert (no-warning/watch/warning/high-risk), research-prototype label | **DONE (prototype)** — engine + approval/audit/API; delivery disabled by config |
| 67 | Citizen interface | DONE (static mode only in production) |
| 68 | Administrator dashboard | DONE (static mode only in production) |
| 69 | System architecture documented as built | DONE (updated 2026-11-06) |
| 70 | System testing — label/split/sequence/feature/pipeline | DONE (**84 tests**, incl. ingestion + alert + API contract) |
| 71 | Final scientific evaluation | DONE (re-run; freeze 13/13, test split closed) |
| 72 | Reproducibility & documentation | DONE (updated 2026-11-06) |
| 73 | **Research report / thesis** | **DONE** — `reports/phase45_thesis.md` (draft; PDF/typesetting later) |

The literal remaining work is **operational, not scientific**: paid always-on hosting and a
public redeploy, formal threshold sign-off plus SMTP/webhook credentials, and the
government-grade workstreams in §6–§7. Stages 64, 66 and 73 are now complete at the
prototype level.

---

## 2. P0 — Make data acquisition genuinely automatic (stage 64)

### 2.1 Measured problem

```
data/raw/glofas/discharge_anchors_daily.csv     2013-01-01 .. 2026-10-12
data/raw/power_soil/power_soil_daily.csv        2013-01-01 .. 2026-10-12
data/raw/openmeteo/era5land_station_daily.csv   2013-01-01 .. 2026-10-12
```

Every feature partition from roughly 2026-10-13 onward (including 2026-11-02 and 2026-11-03)
has **100% null** for the 18 `gf_*` and 19 `ps_*` columns. The frozen model
(`models/final_v2/final_config_v2.json`) lists all 37 as required inputs, and its 5 `ix_*`
interactions are computed from them (`ix_wet_top_x_slope`, `ix_wet_top_x_updis`,
`ix_updis_x_wlrise`, `ix_api09_x_wet_top`, `ix_updis14_x_api`). That is 42 of 110 model
features compromised.

Root causes, confirmed in code:

1. `scripts/fetch_glofas.py`, `fetch_power_soil.py`, `fetch_openmeteo_batch.py` have a
   **hard-coded `END = 2026-10-12`**.
2. They **skip any existing per-station/per-anchor file**, so re-running them never extends
   history — they just report "cached".
3. `system/ops/daily_pipeline.py` calls only FFWC + NASA POWER rainfall + Open-Meteo
   rainfall + feature build. It never calls the three v2 refreshers.
4. There is **no quality gate**: the build succeeds and publishes a snapshot even when a
   third of the inputs are null.

### 2.2 Work packages

**WP-1 — Rewrite the three source fetchers as incremental refreshers**

- `scripts/fetch_glofas.py`
  - Replace module constants with `--start`, `--end`, `--lookback-days` (default end =
    today − 1).
  - Reuse `chosen_cells.json`; only re-scan cells when asked (`--rescan-cells`).
  - Read the existing anchor CSV, fetch only the missing tail (or a rolling lookback to
    absorb revisions), merge, de-duplicate on `(anchor, date)`, sort.
  - Rewrite `discharge_anchors_daily.csv` + `glofas_manifest.json` with row count, min/max
    date and SHA-256.
- `scripts/fetch_power_soil.py`
  - Same incremental tail logic per station (NASA POWER has a multi-day publication lag;
    keep a 7–10 day lookback and merge).
  - Stop treating "file exists" as "file is current".
  - Rebuild `power_soil_daily.csv` + manifest.
- `scripts/fetch_openmeteo_batch.py`
  - Same incremental tail logic; ERA5-Land archive lags ~5 days, so either accept the lag
    explicitly or add a forecast-API fallback for the last few days and mark the source as
    "provisional".
  - Rebuild `era5land_station_daily.csv` + manifest.

**Acceptance:** running each fetcher twice in a row is idempotent; after the second run the
max date equals the requested end date (minus documented source lag), and the manifest
records it.

**WP-2 — One source-refresh entrypoint wired into the daily pipeline**

- New `system/ingestion/refresh_sources.py`:
  - `--start`, `--end`, `--lookback-days`, `--only glofas,power_soil,era5`, `--strict`.
  - Calls each fetcher, then validates: max date, row count, duplicate keys, null share.
  - Emits a machine-readable `system/data/processed/source_status/<date>.json`.
  - Exits non-zero when a **critical** source is stale beyond its SLA (unless `--allow-stale`).
- `system/ops/daily_pipeline.py`: run it as **Step 0**, before feature build, for the same
  date window. A degraded source must either fail the date or mark the run degraded — never
  silently continue.
- `system/ops/refresh_and_publish.py`: propagate the degraded flag so the publisher can
  refuse to publish a degraded snapshot unless explicitly allowed.

**Acceptance:** `daily_pipeline.py --date YYYY-MM-DD` refreshes all five source families
and the resulting partition has `gf_*`/`ps_*` null share < 5%.

**WP-3 — Feature quality gate**

- New `system/features/validate.py` (or a function in `build.py`) that checks:
  - per-group null share for `FULL_V2` (`weather`, `rainfall`, `geography`, `hydrology`,
    `network`, `soil`, `glofas`);
  - required feature columns present and correctly typed;
  - probability inputs in range;
  - row count equals active stations.
- Write `system/data/processed/quality/<date>.json`.
- Fail the build (or require `--allow-degraded`) when critical groups exceed thresholds.
- Add a regression test asserting the live feature row has **no all-null v2 groups**.

**Acceptance:** a deliberately stale source cannot produce a "green" build.

**WP-4 — Backfill and verify**

- One-off backfill 2026-10-13 → today for GloFAS, POWER soil and ERA5-Land.
- Rebuild feature partitions for that window (the model needs the lags/rolls to refill).
- Rebuild the static snapshot for the affected dates.
- Re-run Phase 43 evaluation on the repaired features and record the delta.

**Acceptance:** the last 30 feature partitions have `gf_*`/`ps_*` null share < 5%, and the
live snapshot's latest date matches the feature store.

**WP-5 — Publish freshness and quality**

- Extend `health.json` (and `/health`) with per-source `last_date`, `lag_hours`, `null_rate`,
  and an overall `degraded` flag.
- Surface it in the existing frontend freshness banner.
- Document the SLAs: FFWC ≤ 24 h, rainfall ≤ 24 h, GloFAS ≤ 48 h, POWER soil ≤ 72 h,
  ERA5-Land ≤ 7 d (or forecast fallback).

**Acceptance:** a stale or degraded source is visible to users and to monitoring, not hidden.

---

## 3. P0 — Always-on execution (not a laptop)

The current scheduled task runs on one Windows machine. If it is off, asleep, rebooting, or
logged out, no data is produced. That is the single largest reliability risk.

Options, in order of fit for a government client:

| Option | Pros | Cons |
|---|---|---|
| **GitHub Actions cron** | Free/cheap, versioned, logs, secrets, no server | 6 h job limit, no persistent disk, data download each run, not an agency-controlled environment |
| **Cloud Run Job + Cloud Scheduler** (recommended) | Managed, retries, IAM, region `asia-south1`, audit logs, scales to zero | Some setup; needs GCP project |
| **Azure Container Apps Job / AWS ECS task** | Fits an agency cloud tenant | Tenant-specific setup |
| **Oracle Always Free VM / agency VM + cron** | Full control, persistent disk | You own patching, monitoring, uptime |
| Windows Task Scheduler (current) | Already works | Single point of failure; unsuitable for SLA |

Work:

- Containerise the pipeline (reuse `Dockerfile` patterns; add a `pipeline` target or a
  separate image).
- Add a scheduler entrypoint with retry/backoff and a lock to prevent overlapping runs.
- Emit a run summary (dates processed, source freshness, null rates, publish result).
- Send failure notifications (email / Teams / Slack webhook).
- Write a runbook: how to re-run a date, how to backfill, how to roll back a publish.

**Acceptance:** 7 consecutive days complete with zero manual action; a failed run notifies
within 15 minutes; re-running is idempotent.

---

## 4. P0 — Live API and hosting (stages 65, 67–69)

The repo already contains `render.yaml`, a multi-stage `Dockerfile`, `docker-compose.yml` and
a hardened FastAPI app (`system/api/app.py`). What is missing is a working host and the
production wiring.

**Render free is not suitable:** it sleeps, cold-starts, has no SLA, and the current service
is unreachable (HTTP 000). Choose one:

- **Render paid** — fastest path; still a third-party PaaS.
- **Google Cloud Run** — recommended managed option; region, IAM, uptime checks, audit logs.
- **Azure Container Apps / AWS ECS** — if the agency already has a tenant.
- **Agency VM / on-prem behind a reverse proxy** — if data residency requires it.

Work:

1. Build and push the image; deploy with `PORT`, `HF_FEATURE_DATASET`, `HF_TOKEN` (or an
   object-store mount), `AABHAS_PREVIEW_MODE=false`, `AABHAS_ENABLE_DOCS=false`.
2. Decide where the API reads features from:
   - current design: HF dataset (simple, but external dependency), or
   - agency object storage / persistent volume (better for government).
3. Point the frontend at the API (`VITE_AABHAS_STATIC_SNAPSHOT=false` + API base URL) and
   keep the static Space as a read-only fallback.
4. Add `/version`, `/freshness`, `/sources`; keep `/health` for the platform probe.
5. Create a **staging** service + staging Space; promote only after checks.
6. Add uptime monitoring and an on-call contact.

**Acceptance:** public URL serves the SPA and API over HTTPS; `/health` reports the current
feature date and source freshness; a staging deploy is promoted to production by a repeatable
procedure.

---

## 5. P1 — Early-warning engine (stage 66)

Current state: `system/alerts/alert_engine.py` maps bands → alert levels, applies per-station
state and a 24 h cooldown, writes `system/logs/alerts.csv`, and calls `notify_stub()`.
`system/config/alert_config.json` has `notification.enabled = false`.

Missing:

1. **Threshold sign-off.** The Phase-29 thresholds are research-derived. The agency must
   review and formally adopt them (or supply official danger levels).
2. **Delivery adapters.** Email (SMTP/SES), SMS gateway, Teams/Slack/webhook, dashboard feed.
   Retries, dedupe, rate limits, per-channel test mode.
3. **Human-in-the-loop.** Draft alert → duty officer review/approve → send → acknowledgement;
   escalation if unacknowledged; immutable audit log (who, when, what, why).
4. **Alert lifecycle and API.** Expose current alerts via `/alerts` and the dashboard; keep
   the "research prototype — not an official warning" label until formal approval.
5. **Tests.** Band→level mapping, cooldown, escalation, approval, audit, delivery failure.

**Acceptance:** a simulated high-risk event creates a draft, cannot send without approval,
sends to a test channel, records an audit trail, and appears on the dashboard.

---

## 6. P1 — Government hardening (beyond the workflow)

These are not in `final workflow.md`, but a government deployment needs them:

- **AuthN/AuthZ:** public read endpoints may stay open with rate limits; admin actions
  (alert approval, manual refresh, downloads) behind SSO/OIDC or scoped API keys.
- **Secrets:** secret manager, rotation, no long-lived tokens on a personal PC. The current
  Hugging Face token should be rotated before handover.
- **Observability:** structured logs, error tracking, uptime checks, pipeline success and
  freshness alerts, dashboards, runbook, on-call.
- **Availability:** target SLA (e.g. 99.5%), health checks, graceful degradation to the last
  known-good snapshot, restore drill.
- **Data governance:** append-only raw store, provenance manifest, retention policy,
  backup/restore, license/attribution for each source, data-sharing agreement.
- **Security:** dependency and container scanning, HTTPS/HSTS, CSP, rate limiting (partly
  present), WAF, vulnerability disclosure.
- **Accessibility & language:** WCAG checks, Bangla/English, mobile-first (partly present).
- **Change management:** versioned releases, staging → production, rollback, schema/model
  version pinning, release notes.

---

## 7. P1 — Official-use governance (the workflow's own caveat)

`final workflow.md` says the system is a **research prototype, never an official warning
service**. A government organisation cannot treat it as an official warning without:

1. A documented validation study on independent seasons/events (Phase 43 is a start; it
   must be re-run after the feature repair).
2. A written agreement on thresholds, false-alarm tolerance and lead times.
3. A named duty-officer process with approval and audit.
4. Legal/communications review of the disclaimer and public messaging.
5. A decision on whether the agency accepts the residual risk and labels it
   "decision support" vs "official warning".

Until then, every UI surface and API response should carry the research-prototype label.

---

## 8. P2 — Research report / thesis (stage 73)

Write `reports/phase45_thesis.md` (and PDF) following the document's outline:

1. Definition and data (including the new automatic-acquisition design and QC).
2. Method: leakage controls, splits, label contract.
3. Baselines.
4. Classical ML.
5. Deep learning: framework, sequences, models, gates.
6. Fair comparison and ablation (E0–E16).
7. Selection, calibration, explanation.
8. System (stages 62–70) as built.
9. Limitations (R1–R8, W-R2/W-R3, blocked stages).
10. The null-feature incident: detection, fix, and evaluation delta — a genuinely useful
    production lesson.

**Acceptance:** stage 73 flipped to DONE with a run record, report, and updated ledger.

---

## 9. Recommended order

| Order | Work | Why |
|---|---|---|
| 1 | WP-1…WP-5: source refreshers, pipeline wiring, quality gate, backfill, freshness | Fixes the live accuracy defect; no external accounts needed |
| 2 | Re-run evaluation + snapshot publish | Proves the fix and updates the public site |
| 3 | Always-on runner (CI or cloud job) + failure alerts | Removes the laptop single point of failure |
| 4 | Choose host and deploy live API + staging | Delivers stages 65/67–69 in production form |
| 5 | Alert delivery + approval + audit | Completes stage 66 |
| 6 | Hardening, tests, docs | Government readiness |
| 7 | Thesis | Completes stage 73 |

---

## 10. Decisions and access needed

1. **Host:** Render paid / Google Cloud Run (recommended) / Oracle Always Free /
   agency cloud tenant / on-prem. Which one, and who can create the project?
2. **Runner:** GitHub Actions, cloud cron, or an agency server?
3. **Alert channels:** email, SMS, Teams/Slack? Who is the approving officer?
4. **Status of the deployment:** research decision-support, or intended official warning?
5. **Budget:** paid hosting and SMS are required for any real SLA. Free tiers cannot
   provide one.
6. **Data residency:** must data stay inside a government tenant/region?

---

## 11. What can start immediately without new accounts

- WP-1…WP-5 (three incremental fetchers, refresh entrypoint, quality gate, backfill,
  freshness publishing).
- Re-run Phase 43 evaluation and rebuild the snapshot.
- Add ingestion/alert/API contract tests.
- Draft the thesis skeleton and the operational runbook.

These remove the live accuracy defect and make the daily update complete before any hosting
decision is finalised.
