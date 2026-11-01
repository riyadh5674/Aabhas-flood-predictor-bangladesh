"""
System API — FastAPI application.

Endpoints:
  GET  /health                              — service health + model loaded status
  GET  /model/info                          — D16 model metadata
  GET  /thresholds                          — Phase-29 thresholds + risk bands
  GET  /predict?station_id=X&date=YYYY-MM-DD&horizon=24h
  POST /predict/batch                       — batch predictions
  GET  /dates                               — available feature dates

Usage:
    uvicorn system.api.app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from datetime import date as date_cls
import os
from pathlib import Path
import secrets
from typing import Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from system.api.inference import (
    get_model,
    load_features_for_date,
    list_available_dates,
    predict_station,
    predict_batch,
    predict_all_stations,
    model_info,
    thresholds_info,
)
from system.config.model_config import HORIZONS

app = FastAPI(
    title="Aabhas — Flood Risk Inference API",
    description="Aabhas D16 frozen model (LightGBM 24h/48h, XGBoost 72h) with published decision thresholds",
    version="1.0.0",
)

PREVIEW_MODE = os.getenv("AABHAS_PREVIEW_MODE", "").lower() == "true"
PREVIEW_USER = os.getenv("AABHAS_PREVIEW_USER", "")
PREVIEW_PASSWORD = os.getenv("AABHAS_PREVIEW_PASSWORD", "")
if PREVIEW_MODE and (not PREVIEW_USER or not PREVIEW_PASSWORD):
    raise RuntimeError("Preview mode requires AABHAS_PREVIEW_USER and AABHAS_PREVIEW_PASSWORD")
_preview_auth = HTTPBasic(auto_error=False)


@app.middleware("http")
async def preview_gateway(request, call_next):
    """Protect the staging preview and accept the web client's /api prefix."""
    if PREVIEW_MODE:
        credentials: HTTPBasicCredentials | None = await _preview_auth(request)
        valid = bool(
            credentials
            and secrets.compare_digest(credentials.username, PREVIEW_USER)
            and secrets.compare_digest(credentials.password, PREVIEW_PASSWORD)
        )
        if not valid:
            return Response(
                "Aabhas staging preview — sign in to continue.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Aabhas staging"'},
            )

    path = request.scope["path"]
    if path == "/api" or path.startswith("/api/"):
        request.scope["path"] = path[4:] or "/"
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    station_id: int
    date: str
    horizon: str
    probability: Optional[float] = None
    band: str
    action: str
    threshold: Optional[float] = None
    model_version: str
    computed_at: str
    error: Optional[str] = None


class BatchPredictRequest(BaseModel):
    station_ids: list[int] = Field(..., min_length=1, max_length=133)
    date: str
    horizon: str = Field(..., pattern="^(24h|48h|72h)$")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    feature_store_dates: int
    latest_feature_date: Optional[str] = None
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Service health check. Returns model load status and feature store coverage."""
    try:
        model = get_model()
        model_loaded = model.is_loaded
        model_version = model.version if model_loaded else None
    except Exception:
        model_loaded = False
        model_version = None

    latest_feature_date = None
    try:
        dates = list_available_dates()
        n_dates = len(dates)
        latest_feature_date = dates[-1] if dates else None
    except Exception:
        n_dates = 0

    return HealthResponse(
        status="ok" if model_loaded and n_dates > 0 else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        feature_store_dates=n_dates,
        latest_feature_date=latest_feature_date,
        timestamp=date_cls.today().isoformat(),
    )


@app.get("/model/info", tags=["model"])
async def model_information():
    """Return D16 model metadata (version, seed, features, frozen date, etc.)."""
    try:
        return model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/thresholds", tags=["model"])
async def thresholds():
    """Return Phase-29 cost-derived thresholds and risk bands per horizon."""
    try:
        return thresholds_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dates", tags=["system"])
async def available_dates():
    """Return sorted list of dates with feature data available in the store."""
    try:
        return {"dates": list_available_dates()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict", response_model=PredictResponse, tags=["inference"])
async def predict_single(
    station_id: int = Query(..., description="FFWC station ID"),
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("24h", pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Predict flood probability for a single station on a given date and horizon."""
    try:
        d = date_cls.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}. Use YYYY-MM-DD.")

    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon: {horizon}. Must be one of {HORIZONS}.",
        )

    try:
        return predict_station(station_id, d, horizon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch_endpoint(req: BatchPredictRequest):
    """Predict flood probabilities for multiple stations on a given date/horizon."""
    try:
        d = date_cls.fromisoformat(req.date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {req.date}. Use YYYY-MM-DD.")

    if req.horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon: {req.horizon}. Must be one of {HORIZONS}.",
        )

    try:
        results = predict_batch(req.station_ids, d, req.horizon)
        return BatchPredictResponse(results=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/all", tags=["inference"])
async def predict_all(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("24h", pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Predict flood probabilities for all stations on a given date/horizon.

    Returns a list of predictions, one per station in the feature store.
    """
    try:
        d = date_cls.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}. Use YYYY-MM-DD.")

    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon: {horizon}. Must be one of {HORIZONS}.",
        )

    try:
        df = predict_all_stations(d, horizon)
        return df.to_dict("records")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Startup: preload model
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Preload the model artefact on startup so the first request isn't slow."""
    try:
        get_model()
    except Exception as exc:
        # Don't crash on startup — log and let /health report degraded
        print(f"WARNING: model not preloaded: {exc}")


# In the staging container, FastAPI serves the built SPA on the same origin as
# the API. Keeping the mount last lets all API routes take precedence.
class SPAStaticFiles(StaticFiles):
    """Serve index.html for client-side routes (React BrowserRouter)."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                return await super().get_response("index.html", scope)
            raise


WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if WEB_DIST.is_dir():
    app.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="web")


__all__ = ["app"]
