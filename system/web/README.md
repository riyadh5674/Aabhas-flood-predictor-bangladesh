# Aabhas — Bangladesh flood outlook

Vite + React 19 + TypeScript frontend for the Bangladesh flood risk system.

## Run

```bash
cd "D:\Flood predictor\system\web"
npm install
npm run dev          # starts the local API (if needed) and web app
```

The dev server proxies `/api/*` to the FastAPI service at `http://127.0.0.1:8000`.
The launcher reuses an API already responding there, otherwise it starts the
repository `.venv`. Set `AABHAS_API=http://host:port` to point the proxy and
launcher at another API, or `AABHAS_PYTHON` to choose a Python executable.
To start only the web server while managing the API yourself, run `npm run dev:web`.

Synthetic values are available only when **Demo** is explicitly selected.
In API mode, a failed request is shown as an error and is never presented as a
forecast. The API status checks model readiness and feature-store coverage.

## Build

```bash
npm run build        # -> dist/
npm run preview
```

For production, deploy the FastAPI model service and the built `dist/` frontend
as separately managed services behind a same-origin reverse proxy (`/api` to
FastAPI). Configure `VITE_API_BASE` to your public API route at build time when
the frontend and API use different origins, and configure CORS on the API.
The local launcher is a development convenience, not a production process
manager. Add HTTPS, service supervision, monitoring, and operational forecast
validation in the deployment environment.

## Layout

```
src/
  lib/          api client, demo generator, risk model, formatting, geo helpers
  state/        app context (theme, language, date, horizon, toasts, data mode)
  components/   ui primitives, charts, map, shell, command palette
  pages/        overview, map, stations, station detail, alerts, model, about
public/data/    station master + simplified admin boundaries (geoBoundaries)
```

## Notes

- The supplied Aabhas logo lockups and icon are used throughout the interface.
- Risk bands and decision thresholds mirror `system/config/model_config.json`.
  They are fetched from `/thresholds` at runtime and fall back to the bundled
  copy if the API is unavailable.
- The Streamlit UI at `system/ui/app.py` is untouched and remains the legacy
  interface.
