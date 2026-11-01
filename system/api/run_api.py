"""
System API — Entry point.

Usage:
    python system/api/run_api.py              # start with uvicorn
    python system/api/run_api.py --port 8080  # custom port
    python system/api/run_api.py --reload     # auto-reload on code change
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn  # noqa: E402

from system.config.model_config import API_HOST, API_PORT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Flood Predictor inference API")
    ap.add_argument("--host", default=API_HOST, help="bind host")
    ap.add_argument("--port", type=int, default=API_PORT, help="bind port")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code change")
    ap.add_argument("--workers", type=int, default=1, help="number of uvicorn workers")
    args = ap.parse_args()

    print(f"Starting Flood Predictor API on {args.host}:{args.port}")
    print(f"  Model: D16 (frozen, models/final_v2/selected_model_v2.pkl)")
    print(f"  Feature store: system/data/processed/model_ready/")
    print()

    uvicorn.run(
        "system.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
