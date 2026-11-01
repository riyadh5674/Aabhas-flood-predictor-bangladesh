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

Production is a **static Hugging Face Space**: the app ships precomputed
forecasts, so no API process runs there. Build with
`VITE_AABHAS_STATIC_SNAPSHOT=true` after generating the snapshot:

```bash
cd "D:\Flood predictor"
.venv\Scripts\python.exe system\deploy\build_static_snapshot.py
cd system\web
npm run build          # -> dist/ with data/snapshot/
cd ..\..
.venv\Scripts\python.exe scripts\deploy_static_space.py
```

The static build uses hash routing (`/#/stations/1`) because HF static Spaces
return 404 for arbitrary deep links. Serving the FastAPI service and `dist/`
behind a same-origin reverse proxy remains an option for a live-API deployment;
in that case build without the snapshot flag and set `VITE_API_BASE` plus CORS
on the API. The local launcher is a development convenience, not a production
process manager.

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
