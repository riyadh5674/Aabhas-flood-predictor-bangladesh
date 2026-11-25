# FFWC API Access Request

Saved: 2026-11-25  
Purpose: Obtain official read-only FFWC API access so Aabhas can ingest fresh observed water-level data.

## Status

- **FFWC ingestion currently blocked**: `api.ffwc.gov.bd/data_load` returns `403 Access Denied: Invalid Security Header` without a valid key.
- **Local model-ready data is stale at this boundary**: FFWC raw water levels currently reach `2026-11-10`; catch-up beyond that requires fresh FFWC data.
- **No public official key found**. Request official access from FFWC/BWDB.

## Who to contact

### Primary (API portal support)
- **Email:** `api.support@bwdb.gov.bd`
- **Portal:** https://api.ffwc.gov.bd/
- Portal note: “Developer Documentation (Link will be provided upon registration)”.

### FFWC office contacts (CC recommended)
- **Emails:** `ffwc@bwdb.gov.bd`, `ffwcbwdb@gmail.com`
- **Phone:** `+88-02-2222-30070`
- **Fax:** `+880-2-9557386`
- **Address:** Flood Forecasting and Warning Centre, BWDB, Pani Bhaban, Level 2, Block G, Office 214, 72 Green Road, Dhaka-1205, Bangladesh
- **Contact page:** https://ffwc.gov.bd/app/about/contact-us

## What to request

Request **read-only** API access for:

- Base URL: `https://api.ffwc.gov.bd/data_load`
- Endpoints used by Aabhas:
  - `/stations`
  - `/observed-waterlevel-by-station-and-date/{station_id}/{date}`

Ask them to provide:

- Official API access method/key (project expects `x-ffwc-internal-key`, configurable via `FFWC_INTERNAL_KEY` or `FFWC_API_KEY` env var)
- Developer documentation
- Allowed usage/rate limits
- Confirmation that modest daily automated read-only fetching is acceptable

## Expected usage (be transparent)

- ~153 stations checked once daily
- Small catch-up requests only when needed (e.g., missed days/outages)
- Data cached responsibly; no high-frequency scraping
- Clear attribution to FFWC/BWDB as data source
- Non-commercial research/flood early-warning dashboard use

## Copy-paste email draft

**Subject:** Request for read-only FFWC API access for Aabhas flood early-warning research

**To:** api.support@bwdb.gov.bd  
**CC:** ffwc@bwdb.gov.bd; ffwcbwdb@gmail.com

**Body:**

> Dear FFWC/BWDB Team,
> 
> I am developing **Aabhas**, a non-commercial flood early-warning/research dashboard for Bangladesh. The system uses observed river water-level data to generate 1–3 day flood-risk outlooks for research and preparedness purposes.
> 
> I would like to request **read-only API access** to the FFWC data API:
> - `https://api.ffwc.gov.bd/data_load/stations`
> - `https://api.ffwc.gov.bd/data_load/observed-waterlevel-by-station-and-date/{station_id}/{date}`
> 
> Expected usage is modest: approximately 153 stations checked once daily, with small catch-up requests only when needed. I will respect rate limits, cache data responsibly, and clearly attribute FFWC/BWDB as the data source.
> 
> Could you please share the official API access method/key, developer documentation, and any terms of use/rate limits?
> 
> Thank you for your time and support.
> 
> Regards,  
> [Your name]  
> [Affiliation/role]  
> [Project/organization, if any]  
> [Email]  
> [Phone, optional]

## After they reply

When access is granted:

1. Set key locally (do **not** commit):  
   PowerShell: `$env:FFWC_INTERNAL_KEY="..."`
2. Test station access:
   ```powershell
   & "D:\Flood predictor\.venv\Scripts\python.exe" system/ingestion/run_ingestion.py --stations-only
   ```
3. Run source refresh and catch-up:
   ```powershell
   & "D:\Flood predictor\.venv\Scripts\python.exe" system/ingestion/refresh_sources.py --end YYYY-MM-DD
   & "D:\Flood predictor\.venv\Scripts\python.exe" system/ops/daily_pipeline.py --catchup --skip-source-refresh
   ```
4. Verify model-ready partitions advance past `2026-11-10` and quality gates pass.
5. Add `FFWC_INTERNAL_KEY` as a GitHub Actions secret for automated refresh workflows.

## Security note

- Never commit FFWC keys to git.
- Use `FFWC_INTERNAL_KEY` or `FFWC_API_KEY` environment variables/secrets.
- `.env` should remain local and gitignored if used.
