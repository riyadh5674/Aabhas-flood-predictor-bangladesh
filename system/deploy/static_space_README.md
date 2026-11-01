---
title: Aabhas — Flood Outlook
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Station-level 24/48/72-hour flood outlook for Bangladesh
---

# Aabhas — Flood Outlook

A clear, local flood outlook for Bangladesh. Explore station-level flood risk
and practical guidance for the next 24, 48 and 72 hours.

This static build uses **precomputed forecasts** from the frozen D16 model
(LightGBM 24h/48h, XGBoost 72h) over 2,611 dates through **2026-10-30**.

- Not real-time and not an official flood warning.
- Model guidance only; always follow official FFWC advisories.
- Data: FFWC station water levels, Open-Meteo/ERA5 rainfall, static terrain
  and land-cover features.

Built with React + Vite. Model and inference code live in the private
research repository.
