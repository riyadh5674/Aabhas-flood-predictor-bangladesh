"""
System API — Configuration constants.

Loads system/config/model_config.json and exposes typed constants.
Supports both local feature store and Hugging Face Datasets for production.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "system" / "config" / "model_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    _CFG = json.load(f)

# --- Model ---
MODEL_PATH = ROOT / _CFG["model"]["path"]
MODEL_SHA256 = _CFG["model"]["sha256"] or None
FEATURE_SET = _CFG["model"]["feature_set"]
N_FEATURES = _CFG["model"]["n_features"]
SEED = _CFG["model"]["seed"]
FROZEN_AT_UTC = _CFG["model"]["frozen_at_utc"]
PER_HORIZON_FAMILY = _CFG["model"]["per_horizon_family"]
HORIZONS = _CFG["model"]["horizons"]
HORIZON_TO_LABEL = _CFG["model"]["horizon_to_label"]

# --- Thresholds ---
THRESHOLDS = _CFG["thresholds"]["values"]

# --- Risk bands ---
RISK_BANDS = _CFG["risk_bands"]["values"]

# --- Inference ---
# Local feature store paths (development fallback)
FEATURE_STORE = ROOT / _CFG["inference"]["feature_store"]
FEATURE_STORE_FORMAT = _CFG["inference"]["feature_store_format"]
SINGLE_FILE_FALLBACK = ROOT / _CFG["inference"]["single_file_fallback"]
LATENCY_TARGET_MS = _CFG["inference"]["latency_target_ms"]
BATCH_LATENCY_TARGET_MS = _CFG["inference"]["batch_latency_target_ms"]
MAX_BATCH_SIZE = _CFG["inference"]["max_batch_size"]

# Hugging Face Datasets feature store (production)
# Set via environment variables or model_config.json
HF_FEATURE_DATASET = os.getenv("HF_FEATURE_DATASET") or _CFG["inference"].get("hf_feature_dataset")
HF_FEATURE_REVISION = os.getenv("HF_FEATURE_REVISION") or _CFG["inference"].get("hf_feature_revision", "main")

# --- API ---
API_HOST = _CFG["api"]["host"]
API_PORT = int(os.getenv("PORT", _CFG["api"]["port"]))  # HF Spaces uses PORT env var
API_WORKERS = _CFG["api"]["workers"]
API_ENDPOINTS = _CFG["api"]["endpoints"]

__all__ = [
    "MODEL_PATH",
    "MODEL_SHA256",
    "FEATURE_SET",
    "N_FEATURES",
    "SEED",
    "FROZEN_AT_UTC",
    "PER_HORIZON_FAMILY",
    "HORIZONS",
    "HORIZON_TO_LABEL",
    "THRESHOLDS",
    "RISK_BANDS",
    "FEATURE_STORE",
    "FEATURE_STORE_FORMAT",
    "SINGLE_FILE_FALLBACK",
    "LATENCY_TARGET_MS",
    "BATCH_LATENCY_TARGET_MS",
    "MAX_BATCH_SIZE",
    "HF_FEATURE_DATASET",
    "HF_FEATURE_REVISION",
    "API_HOST",
    "API_PORT",
    "API_WORKERS",
    "API_ENDPOINTS",
]