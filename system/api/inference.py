"""
System API — Inference service.

Loads features from Hugging Face Datasets (production) or local parquet store (dev)
and runs D16 model predictions per horizon. Exposes functions used by the FastAPI app.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.config.model_config import (
    FEATURE_STORE,
    SINGLE_FILE_FALLBACK,
    HF_FEATURE_DATASET,
    HF_FEATURE_REVISION,
)
from system.api.model_loader import D16Model

# Lazy-import the model so FastAPI can start even if the artefact is missing
_model: D16Model | None = None

# Cache for HF dataset access to avoid repeated remote reads
_hf_dataset_cache: Optional[pd.DataFrame] = None
_hf_dataset_last_loaded: Optional[str] = None
_hf_dates_cache: Optional[list[str]] = None
_hf_dates_cached_at: float = 0.0
_HF_DATES_TTL_SECONDS = 900.0

# Bounded cache of scored dates. A `/predict/all` response is deterministic for
# a (date, horizon) pair, so every later request — including the station trend
# endpoint — can reuse it instead of re-running the pipelines.
_PREDICTION_CACHE: "OrderedDict[tuple[str, str], pd.DataFrame]" = OrderedDict()
_PREDICTION_CACHE_LOCK = threading.RLock()
_PREDICTION_CACHE_MAX = max(1, int(os.getenv("AABHAS_PREDICTION_CACHE_ENTRIES", "64")))
_PREDICTION_CACHE_HITS = 0
_PREDICTION_CACHE_MISSES = 0
# Layout used by scripts/upload_features_to_hf.py:
#   data/train/date=YYYY-MM-DD/data.parquet
_HF_PARTITION_PREFIX = os.getenv("HF_FEATURE_PREFIX", "data/train").strip("/")
_HF_PARTITION_RE = re.compile(
    rf"^{re.escape(_HF_PARTITION_PREFIX)}/date=(\d{{4}}-\d{{2}}-\d{{2}})/data\.parquet$"
)


def get_model() -> D16Model:
    """Singleton model loader."""
    global _model
    if _model is None:
        _model = D16Model()
        _model.load(verify_sha256=False)
    return _model


def _load_hf_streaming(dataset_id: str, revision: str, target_str: str) -> pd.DataFrame:
    """Legacy fallback for single-file datasets: stream and filter by date."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    ds = load_dataset(
        dataset_id,
        revision=revision,
        split="train",
        streaming=True,
    )
    filtered = ds.filter(lambda x: str(x.get("date"))[:10] == target_str)
    rows = list(filtered)
    if not rows:
        raise FileNotFoundError(
            f"No features found for {target_str} in HF Dataset {dataset_id}"
        )
    return pd.DataFrame(rows)


def _load_from_hf_datasets(target_date: date | str) -> pd.DataFrame:
    """Load features for a specific date from Hugging Face Datasets.

    Production layout is partitioned by date
    (``data/train/date=YYYY-MM-DD/data.parquet``). Only the requested
    partition is downloaded, so a request costs one small file instead of a
    scan over the full multi-year feature store.
    """
    global _hf_dataset_cache, _hf_dataset_last_loaded

    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    target_str = target_date.isoformat()

    # Check if we need to reload (different date or first load)
    if _hf_dataset_cache is None or _hf_dataset_last_loaded != target_str:
        dataset_id = HF_FEATURE_DATASET
        revision = HF_FEATURE_REVISION or "main"
        filename = f"{_HF_PARTITION_PREFIX}/date={target_str}/data.parquet"

        print(f"Loading features from HF Datasets: {dataset_id} @ {revision} for {target_str}")

        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=dataset_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
            features = pd.read_parquet(local_path)
        except Exception as exc:
            print(f"Partition fetch failed ({exc}); trying streaming fallback")
            features = _load_hf_streaming(dataset_id, revision, target_str)

        if features.empty:
            raise FileNotFoundError(
                f"No features found for {target_str} in HF Dataset {dataset_id}"
            )

        _hf_dataset_cache = features
        _hf_dataset_last_loaded = target_str
        print(f"Loaded {len(_hf_dataset_cache)} station records for {target_str}")

    return _hf_dataset_cache.copy()


def _list_hf_dates() -> list[str]:
    """List available dates from the HF Dataset using the repo file listing.

    Falls back to a streaming scan for datasets uploaded as one file.
    """
    global _hf_dates_cache, _hf_dates_cached_at

    now = time.monotonic()
    if _hf_dates_cache is not None and (now - _hf_dates_cached_at) < _HF_DATES_TTL_SECONDS:
        return list(_hf_dates_cache)

    dataset_id = HF_FEATURE_DATASET
    revision = HF_FEATURE_REVISION or "main"
    print(f"Listing dates from HF Datasets: {dataset_id} @ {revision}")

    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
        )
        dates = sorted(
            {match.group(1) for name in files if (match := _HF_PARTITION_RE.match(name))}
        )

        if not dates:
            # Legacy single-file dataset: stream and collect unique dates.
            try:
                from datasets import load_dataset
            except ImportError:
                raise RuntimeError("datasets library not installed. Run: pip install datasets")

            ds = load_dataset(dataset_id, revision=revision, split="train", streaming=True)
            seen: set[str] = set()
            for index, row in enumerate(ds):
                value = str(row.get("date"))[:10]
                if value:
                    seen.add(value)
                if index > 500_000:  # safety limit
                    break
            dates = sorted(seen)
    except Exception:
        # A transient HF/network error must not take /health and /dates down
        # after we have seen a good listing. Serve the last known set instead.
        if _hf_dates_cache:
            print("Warning: HF date listing failed; serving cached dates")
            return list(_hf_dates_cache)
        raise

    _hf_dates_cache = dates
    _hf_dates_cached_at = now
    return list(dates)


def _load_from_local(target_date: date | str) -> pd.DataFrame:
    """Load features for a specific date from local partitioned parquet store."""
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    part_file = FEATURE_STORE / f"date={target_date.isoformat()}" / "data.parquet"
    if part_file.exists():
        df = pd.read_parquet(part_file)
        return df

    # Fallback: single file, filter by date
    if SINGLE_FILE_FALLBACK.exists():
        df = pd.read_parquet(SINGLE_FILE_FALLBACK)
        df["date"] = pd.to_datetime(df["date"])
        return df[df["date"].dt.date == target_date].copy()

    raise FileNotFoundError(
        f"No feature store found for {target_date}. "
        f"Expected {part_file} or {SINGLE_FILE_FALLBACK}"
    )


def load_features_for_date(target_date: date | str) -> pd.DataFrame:
    """Load all station features for a given date.
    
    Priority:
    1. Hugging Face Datasets (if HF_FEATURE_DATASET is configured)
    2. Local partitioned parquet store
    3. Local single parquet fallback
    """
    # Use HF Datasets if configured (production)
    if HF_FEATURE_DATASET:
        try:
            return _load_from_hf_datasets(target_date)
        except Exception as e:
            print(f"Warning: HF Datasets load failed ({e}), falling back to local")
    
    # Fallback to local (development)
    return _load_from_local(target_date)


def list_available_dates() -> list[str]:
    """Return sorted list of dates with feature data available."""
    # If using HF Datasets, list partitions from the dataset repo
    if HF_FEATURE_DATASET:
        try:
            return _list_hf_dates()
        except Exception as e:
            print(f"Warning: HF Datasets date listing failed ({e}), falling back to local")

    # Local fallback
    if FEATURE_STORE.exists():
        parts = sorted(p.name for p in FEATURE_STORE.iterdir()
                       if p.is_dir() and p.name.startswith("date="))
        return [p.replace("date=", "") for p in parts]
    if SINGLE_FILE_FALLBACK.exists():
        df = pd.read_parquet(SINGLE_FILE_FALLBACK, columns=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return sorted(df["date"].dt.date.astype(str).unique().tolist())
    return []


def _iso_date(target_date: date | str) -> str:
    if isinstance(target_date, str):
        return date.fromisoformat(target_date).isoformat()
    return target_date.isoformat()


def _unknown_prediction(
    station_id: int,
    target_date: date | str,
    horizon: str,
    reason: str,
    model_version: str = "D16",
) -> dict:
    return {
        "station_id": station_id,
        "date": _iso_date(target_date),
        "horizon": horizon,
        "probability": None,
        "band": "unknown",
        "action": "no data",
        "threshold": None,
        "model_version": model_version,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "error": reason,
    }


def prediction_cache_stats() -> dict:
    """Small observability payload surfaced by /health."""
    with _PREDICTION_CACHE_LOCK:
        entries = len(_PREDICTION_CACHE)
    total = _PREDICTION_CACHE_HITS + _PREDICTION_CACHE_MISSES
    return {
        "entries": entries,
        "max_entries": _PREDICTION_CACHE_MAX,
        "hits": _PREDICTION_CACHE_HITS,
        "misses": _PREDICTION_CACHE_MISSES,
        "hit_rate": round(_PREDICTION_CACHE_HITS / total, 3) if total else None,
    }


def _cache_get(key: tuple[str, str]) -> Optional[pd.DataFrame]:
    global _PREDICTION_CACHE_HITS
    with _PREDICTION_CACHE_LOCK:
        cached = _PREDICTION_CACHE.get(key)
        if cached is None:
            return None
        _PREDICTION_CACHE.move_to_end(key)
        _PREDICTION_CACHE_HITS += 1
        return cached.copy()


def _cache_put(key: tuple[str, str], frame: pd.DataFrame) -> None:
    global _PREDICTION_CACHE_MISSES
    with _PREDICTION_CACHE_LOCK:
        _PREDICTION_CACHE_MISSES += 1
        _PREDICTION_CACHE[key] = frame.copy()
        _PREDICTION_CACHE.move_to_end(key)
        while len(_PREDICTION_CACHE) > _PREDICTION_CACHE_MAX:
            _PREDICTION_CACHE.popitem(last=False)


def predict_station(
    station_id: int,
    target_date: date | str,
    horizon: str,
):
    """Predict flood probability for a single station on a given date and horizon.

    Returns dict with station_id, date, horizon, probability, band, action,
    threshold, model_version, computed_at.
    """
    model = get_model()
    df = predict_all_stations(target_date, horizon)

    row = df[df["station_id"] == station_id]
    if row.empty:
        return _unknown_prediction(
            station_id,
            target_date,
            horizon,
            f"station {station_id} not found for {_iso_date(target_date)}",
            model.version,
        )

    return row.iloc[0].to_dict()


def predict_batch(
    station_ids: list[int],
    target_date: date | str,
    horizon: str,
):
    """Predict flood probabilities for multiple stations on a given date/horizon.

    Returns list of result dicts (one per station).
    """
    model = get_model()
    df = predict_all_stations(target_date, horizon)

    # Filter to requested stations that exist in the feature store
    available = set(df["station_id"].tolist())
    missing = [sid for sid in station_ids if sid not in available]
    results = []

    if missing:
        for sid in missing:
            results.append(
                _unknown_prediction(
                    sid,
                    target_date,
                    horizon,
                    f"station {sid} not found for {_iso_date(target_date)}",
                    model.version,
                )
            )

    present_df = df[df["station_id"].isin(station_ids)]
    if not present_df.empty:
        results.extend(present_df.to_dict("records"))

    # Sort results to match requested order
    order = {sid: i for i, sid in enumerate(station_ids)}
    results.sort(key=lambda r: order.get(r["station_id"], 999999))

    return results


def predict_all_stations(
    target_date: date | str,
    horizon: str,
):
    """Predict flood probabilities for all stations on a given date/horizon.

    Returns a cached DataFrame with one row per station. The returned frame is
    a copy, so callers can filter it without mutating the cache.
    """
    key = (_iso_date(target_date), horizon)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    model = get_model()
    df = load_features_for_date(target_date)
    result = model.predict(df, horizon, target_date=_iso_date(target_date))
    _cache_put(key, result)
    return result.copy()


def predict_station_series(
    station_id: int,
    target_dates: list[date] | list[str],
    horizon: str,
) -> list[dict]:
    """Station history over a bounded set of dates.

    Missing dates become explicit `unknown` points instead of failing the whole
    series, so a single gap at the edge of the feature store does not erase an
    otherwise valid trend. Raises only when no date has a usable prediction.
    """
    model = get_model()
    points: list[dict] = []
    for target_date in target_dates:
        try:
            points.append(predict_station(station_id, target_date, horizon))
        except FileNotFoundError:
            points.append(
                _unknown_prediction(
                    station_id,
                    target_date,
                    horizon,
                    f"no features for {_iso_date(target_date)}",
                    model.version,
                )
            )
    if not any(point.get("probability") is not None for point in points):
        raise FileNotFoundError(
            f"No usable {horizon} history for station {station_id} in the requested dates."
        )
    return points


def model_info() -> dict:
    """Return model metadata."""
    model = get_model()
    return {
        "version": model.version,
        "decision": "D16",
        "seed": model.seed,
        "feature_set": model._artefact.get("feature_set") if model._artefact else None,
        "n_features": model.n_features,
        "frozen_at_utc": model.frozen_at_utc,
        "dataset_sha256_v2": model.dataset_sha256_v2,
        "per_horizon_family": model._artefact.get("per_horizon_family") if model._artefact else None,
        "horizons": ["24h", "48h", "72h"],
        "loaded_at": model._loaded_at.isoformat() if model._loaded_at else None,
    }


def thresholds_info() -> dict:
    """Return threshold and risk-band configuration."""
    from system.config.model_config import THRESHOLDS, RISK_BANDS
    return {
        "thresholds": THRESHOLDS,
        "risk_bands": RISK_BANDS,
        "cost_ratio": "C_FN=10, C_FP=1",
        "period": "val_2022 all-season",
    }


__all__ = [
    "get_model",
    "load_features_for_date",
    "list_available_dates",
    "predict_station",
    "predict_batch",
    "predict_all_stations",
    "predict_station_series",
    "prediction_cache_stats",
    "model_info",
    "thresholds_info",
]