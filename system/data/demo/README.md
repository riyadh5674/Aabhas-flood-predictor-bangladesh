# Bundled demo feature store

This directory holds a tracked, recent slice of the model-ready feature store so
the hosted demo can serve **real D16 predictions without external credentials**.

- Partitions: `model_ready/date=YYYY-MM-DD/data.parquet`
- Rows per partition: 112 FFWC stations
- Columns: the full `FULL_V2` feature set used by the frozen D16 model
- Default window: the 120 most recent dates at build time

The full research store (2,600+ dates) lives at
`system/data/processed/model_ready/` and is intentionally gitignored. The API
prefers `AABHAS_FEATURE_STORE` when set, then the full local store, and finally
falls back to this directory when the full store is absent (the normal case in
a slim Docker image).

Refresh the slice after the feature pipeline publishes new dates:

```bash
python scripts/build_demo_store.py --days 120
```

This is a demo cache, not the canonical dataset. See `manifest.json` for the
exact date range and provenance. The hosted service remains a research
prototype, not an official flood-warning product.
