# Aabhas on Cloudflare Pages (free static snapshot)

This deployment serves a static, precomputed forecast snapshot. It does not run
the Python API in the cloud. Forecasts are generated from the repository's real
D16 model and feature store, and the site shows the date through which they are
available. They are guidance, not official FFWC warnings or real-time forecasts.

## Build a fresh snapshot

From the repository root, with the project virtual environment and Node/npm
available:

```powershell
.\.venv\Scripts\python.exe system\deploy\build_static_snapshot.py
Set-Location system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT = "true"
npm run build
```

The generator writes derived forecasts to `system/web/public/data/snapshot/`;
it does not copy raw feature data or the model file. Review the displayed
`latest_feature_date`, then commit the generated snapshot and built UI changes.

## Publish with Cloudflare Pages

1. Push the repository to a Git provider supported by Cloudflare Pages.
2. In Cloudflare, create a Pages project connected to that repository.
3. Set the root directory to `system/web`, the build command to `npm run build`,
   and the output directory to `dist`.
4. Add build environment variable `VITE_AABHAS_STATIC_SNAPSHOT=true` for the
   production environment, then deploy.
5. For each manual refresh, rebuild the snapshot from current local data/model,
   commit the generated JSON, and push. Pages then rebuilds the site.

Cloudflare Pages static asset hosting has a free tier and needs no paid compute.
The deployment uses a few thousand JSON files, within the documented 20,000
files-per-site limit; keep each generated asset below the 25 MiB per-file cap.
The public site exposes the derived forecasts to anyone with its URL. Do not put
personal or confidential data in the snapshot.

## Local preview

After the build, run `npm run preview` in `system/web`. The `Snapshot` status
pill and data-through date confirm that static data mode is active.
