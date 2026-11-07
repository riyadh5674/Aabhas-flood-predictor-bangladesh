# Security Policy

Aabhas is a research/decision-support prototype. It is **not** an official flood
warning service and must not be relied on as the sole source for life-safety
decisions.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private vulnerability reporting:

1. Go to the repository **Security** tab.
2. Choose **Report a vulnerability**.
3. Include: affected component, reproduction steps, impact, and any suggested fix.

If private reporting is unavailable, contact the maintainer through their GitHub
profile and ask for a private channel. Please allow a reasonable window for a response
before disclosure.

## Scope

In scope:

- the FastAPI service (`system/api`),
- the React application (`system/web`),
- ingestion, feature and alert pipelines (`system/ingestion`, `system/features`,
  `system/alerts`, `system/ops`),
- deployment and CI configuration (`.github`, Dockerfiles, `render.yaml`).

Out of scope for this prototype:

- denial-of-service through unrealistic traffic volumes on a free tier,
- issues that require a compromised host or stolen credentials,
- the accuracy of public data sources,
- missing features that are documented as non-goals.

## Existing controls

| Control | Where |
|---|---|
| Security headers, CSP, HSTS behind TLS | API middleware |
| CORS allow-list | `AABHAS_ALLOWED_ORIGINS` |
| Sliding-window rate limiting | API middleware, `AABHAS_RATE_LIMIT_*` |
| Admin endpoints require a key | `AABHAS_ADMIN_KEY` |
| No secrets in the frontend bundle | build-time env separation |
| Alert delivery disabled by default | `system/config/alert_config.json` |
| Hash-chained audit log | alert approval lifecycle |
| Freshness and quality gates | `/health`, `system/features/validate.py` |
| Dependency scanning guidance | `docs/DEPENDENCY_SCANNING.md` |

## Data and privacy

The public demo ships a precomputed snapshot of station forecasts. It contains no
personal data. If you deploy the live API, enable HTTPS, restrict CORS and rotate the
admin key.

## Disclosure

We will credit reporters who wish to be named, once a fix is available.
