"""
System API — FastAPI application.

Endpoints:
  GET  /health                              — service health + model loaded status
  GET  /model/info                          — D16 model metadata
  GET  /thresholds                          — Phase-29 thresholds + risk bands
  GET  /dates                               — available feature dates
  GET  /predict?station_id=X&date=YYYY-MM-DD&horizon=24h
  GET  /predict/all?date=YYYY-MM-DD&horizon=24h
  GET  /predict/trend?station_id=X&dates=...&horizon=24h
  POST /predict/batch                       — batch predictions
  GET  /                                    — built SPA (when system/web/dist exists)

Production behaviour:
  * per-IP sliding-window rate limiting with Retry-After
  * security headers + optional CORS allow-list
  * request IDs and structured access logs
  * bounded per-date prediction cache (see inference.py)
  * heavy inference endpoints are synchronous so FastAPI runs them in its
    thread pool instead of blocking the event loop

Usage:
    uvicorn system.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from datetime import date as date_cls, datetime, timezone
from typing import Optional
import uuid

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from system.api.inference import (
    get_model,
    list_available_dates,
    predict_station,
    predict_batch,
    predict_all_stations,
    predict_station_series,
    model_info,
    thresholds_info,
    prediction_cache_stats,
)
from system.config.model_config import HORIZONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_VERSION = "1.1.0"
logger = logging.getLogger("aabhas.api")

PREVIEW_MODE = os.getenv("AABHAS_PREVIEW_MODE", "").lower() == "true"
PREVIEW_USER = os.getenv("AABHAS_PREVIEW_USER", "")
PREVIEW_PASSWORD = os.getenv("AABHAS_PREVIEW_PASSWORD", "")
if PREVIEW_MODE and (not PREVIEW_USER or not PREVIEW_PASSWORD):
    raise RuntimeError("Preview mode requires AABHAS_PREVIEW_USER and AABHAS_PREVIEW_PASSWORD")
_preview_auth = HTTPBasic(auto_error=False)

RATE_LIMIT_PER_MINUTE = int(os.getenv("AABHAS_RATE_LIMIT_PER_MINUTE", "180"))
RATE_LIMIT_PREDICT_PER_MINUTE = int(
    os.getenv("AABHAS_RATE_LIMIT_PREDICT_PER_MINUTE", "60")
)
RATE_LIMIT_DISABLED = os.getenv("AABHAS_RATE_LIMIT_DISABLED", "").lower() == "true"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("AABHAS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
ENABLE_DOCS = os.getenv("AABHAS_ENABLE_DOCS", "true").lower() == "true"

# Admin authentication - API key based
ADMIN_API_KEYS = {
    key.strip() for key in os.getenv("AABHAS_ADMIN_API_KEYS", "").split(",") if key.strip()
}
ADMIN_API_KEY_HEADER = "X-Admin-API-Key"

# Paths served by the API. Used only for rate limiting and logging, because
# the web client calls the same process through its /api prefix.
_API_PREFIXES = ("/health", "/model", "/thresholds", "/dates", "/predict", "/alerts")


class SlidingWindowLimiter:
    """Small in-process limiter.

    A single free-tier instance is the deployment target. For multi-worker
    deployments each worker enforces its own budget, which is still enough to
    stop accidental request storms; a shared Redis limiter would be the next
    step if horizontal scaling is added.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = max(1, limit)
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window - (now - bucket[0])))
                return False, 0, retry_after
            bucket.append(now)
            remaining = self.limit - len(bucket)
            # Opportunistically drop empty buckets so a scan of many IPs does
            # not grow memory without bound.
            if len(self._hits) > 4096:
                for stale_key in [k for k, v in self._hits.items() if not v][:2048]:
                    self._hits.pop(stale_key, None)
            return True, remaining, 0


_general_limiter = SlidingWindowLimiter(RATE_LIMIT_PER_MINUTE)
_predict_limiter = SlidingWindowLimiter(RATE_LIMIT_PREDICT_PER_MINUTE)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_api_path(path: str) -> bool:
    return path.startswith(_API_PREFIXES)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Preload the model so the first user request is not a cold start."""
    try:
        get_model()
        logger.info("D16 model preloaded")
    except Exception as exc:  # do not crash; /health reports degraded
        logger.warning("Model not preloaded: %s", exc)
    yield


app = FastAPI(
    title="Aabhas — Flood Risk Inference API",
    description="Aabhas D16 frozen model (LightGBM 24h/48h, XGBoost 72h) with published decision thresholds",
    version=APP_VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
    lifespan=lifespan,
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        max_age=600,
    )
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def production_gateway(request: Request, call_next):
    """Auth, path normalisation, rate limiting, logging and security headers."""
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]

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
                headers={
                    "WWW-Authenticate": 'Basic realm="Aabhas staging"',
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                },
            )

    raw_path = request.scope["path"]
    if raw_path == "/api" or raw_path.startswith("/api/"):
        request.scope["path"] = raw_path[4:] or "/"
    path = request.scope["path"]

    # Render sends health probes frequently; never rate-limit them.
    limiter = None
    remaining: int | None = None
    if not RATE_LIMIT_DISABLED and _is_api_path(path) and path != "/health":
        limiter = _predict_limiter if path.startswith("/predict") else _general_limiter
        allowed, remaining, retry_after = limiter.check(_client_ip(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down and try again shortly.",
                    "request_id": request_id,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limiter.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
                },
            )

    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"app;dur={duration_ms:.1f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["X-RateLimit-Limit"] = str(
        limiter.limit if limiter is not None else _general_limiter.limit
    )
    if remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(remaining)

    is_docs = path.startswith(("/docs", "/redoc", "/openapi.json"))
    if not is_docs:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "worker-src 'self'; "
            "manifest-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if proto == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    if path == "/health":
        response.headers["Cache-Control"] = "no-store"
    elif _is_api_path(path) and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"

    if _is_api_path(path):
        logger.info(
            "%s %s %s -> %s %.1fms",
            request_id,
            request.method,
            path,
            response.status_code,
            duration_ms,
        )

    return response


# ---------------------------------------------------------------------------
# Admin authentication dependency
# ---------------------------------------------------------------------------

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

_admin_key_header = APIKeyHeader(name=ADMIN_API_KEY_HEADER, auto_error=False)

async def require_admin_key(api_key: str = Security(_admin_key_header)) -> str:
    """Require valid admin API key for admin endpoints."""
    if not ADMIN_API_KEYS:
        raise HTTPException(status_code=503, detail="Admin API keys not configured")
    if not api_key or api_key not in ADMIN_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
    return api_key


def require_admin_or_preview(api_key: str = Security(_admin_key_header)) -> str:
    """Allow either admin key or preview mode credentials."""
    if PREVIEW_MODE:
        return "preview"
    return require_admin_key(api_key)


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


class TrendResponse(BaseModel):
    station_id: int
    horizon: str
    points: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    feature_store_dates: int
    latest_feature_date: Optional[str] = None
    latest_feature_age_days: Optional[int] = None
    timestamp: str
    api_version: str
    prediction_cache: dict = Field(default_factory=dict)
    sources: dict = Field(default_factory=dict)
    degraded: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _health_payload() -> HealthResponse:
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

    now = datetime.now(timezone.utc)
    age_days = None
    if latest_feature_date:
        try:
            age_days = (now.date() - date_cls.fromisoformat(latest_feature_date)).days
        except ValueError:
            age_days = None

    # Read per-source freshness from latest source_status file (ignore future dates)
    sources_info = {}
    degraded = False
    try:
        status_dir = Path(__file__).resolve().parents[2] / "system" / "data" / "processed" / "source_status"
        if status_dir.exists():
            status_files = sorted(status_dir.glob("*.json"))
            today = datetime.now(timezone.utc).date()
            for status_file in reversed(status_files):
                # Skip future-dated files
                try:
                    file_date = date_cls.fromisoformat(status_file.stem)
                    if file_date > today:
                        continue
                except ValueError:
                    pass
                latest_status = json.loads(status_file.read_text(encoding="utf-8"))
                for name, info in latest_status.get("sources", {}).items():
                    sources_info[name] = {
                        "status": info.get("status"),
                        "last_date": info.get("last_date"),
                        "lag_days": info.get("lag_days"),
                        "max_lag_days": info.get("max_lag_days"),
                        "critical": info.get("critical", False),
                    }
                    if info.get("status") != "ok" and info.get("critical"):
                        degraded = True
                break  # Use the most recent valid status file
    except Exception:
        pass

    overall_status = "ok" if model_loaded and n_dates > 0 and not degraded else "degraded"

    return HealthResponse(
        status=overall_status,
        model_loaded=model_loaded,
        model_version=model_version,
        feature_store_dates=n_dates,
        latest_feature_date=latest_feature_date,
        latest_feature_age_days=age_days,
        timestamp=now.isoformat(),
        api_version=APP_VERSION,
        prediction_cache=prediction_cache_stats(),
        sources=sources_info,
        degraded=degraded,
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Service health check. Runs in the thread pool because HF listing may block."""
    return _health_payload()


@app.get("/model/info", tags=["model"])
def model_information():
    """Return D16 model metadata (version, seed, features, frozen date, etc.)."""
    try:
        return model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/thresholds", tags=["model"])
def thresholds():
    """Return Phase-29 cost-derived thresholds and risk bands per horizon."""
    try:
        return thresholds_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dates", tags=["system"])
def available_dates():
    """Return sorted list of dates with feature data available in the store."""
    try:
        dates = list_available_dates()
        return JSONResponse(
            {"dates": dates},
            headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=300"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _validate_date(value: str, parameter: str = "date") -> date_cls:
    try:
        return date_cls.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {parameter} format: {value}. Use YYYY-MM-DD.",
        )


def _validate_horizon(horizon: str) -> None:
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon: {horizon}. Must be one of {HORIZONS}.",
        )


@app.get("/predict", response_model=PredictResponse, tags=["inference"])
def predict_single(
    station_id: int = Query(..., description="FFWC station ID"),
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("24h", pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Predict flood probability for a single station on a given date and horizon."""
    d = _validate_date(date)
    _validate_horizon(horizon)
    try:
        return predict_station(station_id, d, horizon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch_endpoint(req: BatchPredictRequest):
    """Predict flood probabilities for multiple stations on a given date/horizon."""
    d = _validate_date(req.date)
    _validate_horizon(req.horizon)
    try:
        results = predict_batch(req.station_ids, d, req.horizon)
        return BatchPredictResponse(results=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/all", tags=["inference"])
def predict_all(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("24h", pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Predict flood probabilities for all stations on a given date/horizon."""
    d = _validate_date(date)
    _validate_horizon(horizon)
    try:
        df = predict_all_stations(d, horizon)
        return JSONResponse(
            df.to_dict("records"),
            headers={"Cache-Control": "public, max-age=120, stale-while-revalidate=600"},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/trend", response_model=TrendResponse, tags=["inference"])
def predict_trend(
    station_id: int = Query(..., description="FFWC station ID"),
    dates: str = Query(..., description="Comma-separated dates (max 60), oldest first"),
    horizon: str = Query("24h", pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Station history in one request.

    Replaces the web client's previous fan-out of up to 30 `/predict/all`
    calls. Dates are bounded so a single request cannot be used to scan the
    entire feature store.
    """
    _validate_horizon(horizon)
    requested = [value.strip() for value in dates.split(",") if value.strip()]
    if not requested:
        raise HTTPException(status_code=400, detail="At least one date is required.")
    if len(requested) > 60:
        raise HTTPException(status_code=400, detail="At most 60 dates may be requested.")
    parsed = sorted({_validate_date(value) for value in requested})

    try:
        points = predict_station_series(station_id, parsed, horizon)
        return JSONResponse(
            {
                "station_id": station_id,
                "horizon": horizon,
                "points": points,
            },
            headers={"Cache-Control": "public, max-age=120, stale-while-revalidate=600"},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertResponse(BaseModel):
    draft_id: Optional[str] = None
    station_id: int
    date: str
    horizon: str
    risk_band: str
    probability: float
    previous_band: Optional[str] = None
    alert_level: str
    triggered_at: str
    reason: str
    status: str  # draft, approved, dispatched
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    dispatched_at: Optional[str] = None
    delivery: Optional[dict] = None


class AlertsListResponse(BaseModel):
    alerts: list[AlertResponse]
    pending_count: int
    total_count: int


@app.get("/alerts/feed", response_model=AlertsListResponse, tags=["alerts"])
def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status: draft, approved, dispatched"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Get alert records with optional status filter."""
    try:
        from system.alerts.alert_engine import get_pending_approvals, DRAFT_DIR, APPROVED_DIR
    except ImportError:
        raise HTTPException(status_code=503, detail="Alert engine not available")
    
    all_alerts = []
    
    # Load draft alerts
    if DRAFT_DIR.exists():
        for f in DRAFT_DIR.glob("*.json"):
            try:
                alert = json.loads(f.read_text(encoding="utf-8"))
                if status is None or alert.get("status") == status:
                    all_alerts.append(alert)
            except Exception:
                pass
    
    # Load approved alerts
    if APPROVED_DIR.exists():
        for f in APPROVED_DIR.glob("*.json"):
            try:
                alert = json.loads(f.read_text(encoding="utf-8"))
                if status is None or alert.get("status") == status:
                    all_alerts.append(alert)
            except Exception:
                pass
    
    # Sort by triggered_at descending
    all_alerts.sort(key=lambda x: x.get("triggered_at", ""), reverse=True)
    
    total = len(all_alerts)
    pending = sum(1 for a in all_alerts if a.get("status") == "draft")
    paginated = all_alerts[offset:offset + limit]
    
    return AlertsListResponse(
        alerts=paginated,
        pending_count=pending,
        total_count=total,
    )


@app.post("/alerts/{draft_id}/approve", tags=["alerts"], dependencies=[Depends(require_admin_key)])
def approve_alert(draft_id: str, approver: str = Query(..., description="Approver identifier")):
    """Approve a draft alert for dispatch (admin only)."""
    try:
        from system.alerts.alert_engine import approve_alert
    except ImportError:
        raise HTTPException(status_code=503, detail="Alert engine not available")
    
    ok, msg = approve_alert(draft_id, approver)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"draft_id": draft_id, "status": "approved", "message": msg}


@app.post("/alerts/evaluate", tags=["alerts"], dependencies=[Depends(require_admin_key)])
def evaluate_alerts_endpoint(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    horizon: Optional[str] = Query(None, pattern="^(24h|48h|72h)$", description="Forecast horizon"),
):
    """Trigger alert evaluation for a date/horizon (admin only)."""
    try:
        from system.alerts.alert_engine import run_alerts
        from system.config.model_config import HORIZONS
    except ImportError:
        raise HTTPException(status_code=503, detail="Alert engine not available")
    
    target_date = _validate_date(date)
    horizons = [horizon] if horizon else HORIZONS
    
    alerts = run_alerts(target_date, horizons=horizons)
    
    return {
        "date": date,
        "horizons": horizons,
        "alerts_evaluated": len(alerts),
        "alerts": alerts,
    }


# ---------------------------------------------------------------------------
# Admin operations
# ---------------------------------------------------------------------------

class PipelineTriggerResponse(BaseModel):
    triggered: bool
    message: str
    run_id: Optional[str] = None


@app.post("/admin/pipeline/trigger", response_model=PipelineTriggerResponse, tags=["admin"], dependencies=[Depends(require_admin_key)])
def trigger_pipeline(
    catchup: bool = Query(True, description="Run catch-up for missing dates"),
    date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD)"),
):
    """Manually trigger the daily data pipeline (admin only)."""
    import subprocess
    import uuid as uuid_lib
    
    run_id = uuid_lib.uuid4().hex[:8]
    cmd = [sys.executable, "system/ops/daily_pipeline.py"]
    if catchup:
        cmd.append("--catchup")
    elif date:
        cmd += ["--date", date]
    
    try:
        # Run in background
        subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        return PipelineTriggerResponse(
            triggered=True,
            message=f"Pipeline started (run_id={run_id})",
            run_id=run_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {e}")


@app.post("/admin/model/reload", tags=["admin"], dependencies=[Depends(require_admin_key)])
def reload_model():
    """Force reload the model (admin only)."""
    try:
        from system.api.inference import get_model
        model = get_model()
        model.load()
        return {"reloaded": True, "model_version": model.version, "loaded": model.is_loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")


@app.get("/admin/status", tags=["admin"], dependencies=[Depends(require_admin_key)])
def admin_status():
    """Get detailed admin status (admin only)."""
    from system.api.inference import get_model
    from system.features.validate import QUALITY_DIR
    
    model = get_model()
    
    # Latest quality report
    quality = {}
    if QUALITY_DIR.exists():
        latest_q = QUALITY_DIR / "latest.json"
        if latest_q.exists():
            quality = json.loads(latest_q.read_text(encoding="utf-8"))
    
    # Latest source status
    sources = {}
    status_dir = Path(__file__).resolve().parents[2] / "system" / "data" / "processed" / "source_status"
    if status_dir.exists():
        status_files = sorted(status_dir.glob("*.json"))
        today = datetime.now(timezone.utc).date()
        for f in reversed(status_files):
            try:
                file_date = date_cls.fromisoformat(f.stem)
                if file_date > today:
                    continue
            except ValueError:
                pass
            sources = json.loads(f.read_text(encoding="utf-8"))
            break
    
    return {
        "model": {
            "loaded": model.is_loaded,
            "version": model.version if model.is_loaded else None,
        },
        "quality": quality,
        "sources": sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Monitoring endpoints (public, for uptime checks)
# ---------------------------------------------------------------------------

class MonitoringResponse(BaseModel):
    """Monitoring response with freshness and alerting info."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    latest_feature_date: Optional[str] = None
    latest_feature_age_days: Optional[int] = None
    sources: dict = Field(default_factory=dict)
    degraded: bool
    pipeline_last_run: Optional[str] = None
    alerts_pending: int = 0


@app.get("/monitoring", response_model=MonitoringResponse, tags=["monitoring"])
def monitoring_status():
    """
    Public monitoring endpoint for uptime checks and alerting.
    
    Returns health status with freshness indicators suitable for
    external monitoring systems (Cloud Monitoring, Prometheus, etc.).
    """
    try:
        from system.api.inference import get_model
        from system.features.validate import QUALITY_DIR
        from system.alerts.alert_engine import get_pending_approvals
    except ImportError:
        raise HTTPException(status_code=503, detail="Monitoring dependencies not available")
    
    model = get_model()
    
    # Feature freshness
    latest_feature_date = None
    try:
        dates = list_available_dates()
        latest_feature_date = dates[-1] if dates else None
    except Exception:
        pass
    
    now = datetime.now(timezone.utc)
    age_days = None
    if latest_feature_date:
        try:
            age_days = (now.date() - date_cls.fromisoformat(latest_feature_date)).days
        except ValueError:
            pass
    
    # Source status
    sources = {}
    degraded = False
    status_dir = Path(__file__).resolve().parents[2] / "system" / "data" / "processed" / "source_status"
    if status_dir.exists():
        status_files = sorted(status_dir.glob("*.json"))
        today = datetime.now(timezone.utc).date()
        for f in reversed(status_files):
            try:
                file_date = date_cls.fromisoformat(f.stem)
                if file_date > today:
                    continue
            except ValueError:
                pass
            sources = json.loads(f.read_text(encoding="utf-8"))
            # Check if any critical source is degraded
            for name, info in sources.get("sources", {}).items():
                if info.get("status") != "ok" and info.get("critical"):
                    degraded = True
            break
    
    # Pending alerts
    alerts_pending = 0
    try:
        alerts_pending = len(get_pending_approvals())
    except Exception:
        pass
    
    # Pipeline last run (from latest source status)
    pipeline_last_run = None
    if sources:
        pipeline_last_run = sources.get("checked_at")
    
    overall_status = "healthy" if model.is_loaded and not degraded and (age_days is None or age_days <= 2) else "degraded"
    
    return MonitoringResponse(
        status=overall_status,
        model_loaded=model.is_loaded,
        model_version=model.version if model.is_loaded else None,
        latest_feature_date=latest_feature_date,
        latest_feature_age_days=age_days,
        sources={name: {"status": info.get("status"), "lag_days": info.get("lag_days")} 
                 for name, info in sources.get("sources", {}).items()},
        degraded=degraded,
        pipeline_last_run=pipeline_last_run,
        alerts_pending=alerts_pending,
    )


@app.get("/monitoring/prometheus", tags=["monitoring"])
def prometheus_metrics():
    """
    Prometheus-format metrics for scraping.
    
    Returns basic metrics in Prometheus text format.
    """
    try:
        from system.api.inference import get_model
    except ImportError:
        raise HTTPException(status_code=503, detail="Monitoring dependencies not available")
    
    model = get_model()
    
    # Feature freshness
    latest_feature_date = None
    try:
        dates = list_available_dates()
        latest_feature_date = dates[-1] if dates else None
    except Exception:
        pass
    
    now = datetime.now(timezone.utc)
    age_days = -1
    if latest_feature_date:
        try:
            age_days = (now.date() - date_cls.fromisoformat(latest_feature_date)).days
        except ValueError:
            pass
    
    lines = [
        f'aabhas_model_loaded {1 if model.is_loaded else 0}',
        f'aabhas_feature_age_days {age_days}',
        f'aabhas_latest_feature_date "{latest_feature_date or "unknown"}"',
        f'aabhas_model_version "{model.version if model.is_loaded else "unknown"}"',
        f'aabhas_uptime_seconds {int((now - datetime.fromtimestamp(0, tz=timezone.utc)).total_seconds())}',
    ]
    
    return Response(content="\n".join(lines), media_type="text/plain")


# ---------------------------------------------------------------------------
# SPA
# ---------------------------------------------------------------------------

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
