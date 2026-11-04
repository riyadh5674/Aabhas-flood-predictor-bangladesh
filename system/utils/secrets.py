"""
Secret management utilities for Aabhas Flood Predictor.

Supports:
- Environment variables (highest priority)
- GCP Secret Manager
- Local .env file (development only)

Usage:
    from system.utils.secrets import get_secret
    
    hf_token = get_secret("HF_TOKEN")
    smtp_password = get_secret("SMTP_PASSWORD")
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Load .env file if present (development only)
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            pass

_load_dotenv()


@lru_cache(maxsize=32)
def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret by name.
    
    Priority:
    1. Environment variable (exact match or UPPER_SNAKE_CASE)
    2. GCP Secret Manager (if GCP_PROJECT_ID is set)
    3. Default value
    
    Args:
        secret_name: Name of the secret (e.g., "hf-token", "smtp-password")
        default: Default value if not found
    
    Returns:
        Secret value or default/None
    """
    # 1. Try environment variable (support both exact and normalized names)
    env_candidates = [
        secret_name,
        secret_name.upper().replace("-", "_"),
        secret_name.lower().replace("-", "_"),
    ]
    for env_var in env_candidates:
        value = os.getenv(env_var)
        if value:
            return value
    
    # 2. Try GCP Secret Manager
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            # Try exact name first, then with project prefix
            secret_paths = [
                f"projects/{project_id}/secrets/{secret_name}/versions/latest",
                f"projects/{project_id}/secrets/{secret_name.replace('_', '-')}/versions/latest",
            ]
            for path in secret_paths:
                try:
                    response = client.access_secret_version(request={"name": path})
                    return response.payload.data.decode("UTF-8")
                except Exception:
                    continue
        except Exception:
            pass
    
    return default


def get_required_secret(secret_name: str) -> str:
    """Get a secret that must exist, raise if not found."""
    value = get_secret(secret_name)
    if value is None:
        raise RuntimeError(f"Required secret '{secret_name}' not found in environment or Secret Manager")
    return value


# Convenience functions for common secrets
def get_hf_token() -> Optional[str]:
    return get_secret("HF_TOKEN") or get_secret("hf-token")


def get_smtp_password() -> Optional[str]:
    return get_secret("SMTP_PASSWORD") or get_secret("smtp-password")


def get_webhook_secret() -> Optional[str]:
    return get_secret("WEBHOOK_SECRET") or get_secret("webhook-secret")


def get_render_deploy_hook() -> Optional[str]:
    return get_secret("RENDER_DEPLOY_HOOK") or get_secret("render-deploy-hook")


def get_admin_api_keys() -> list[str]:
    """Get admin API keys as a list."""
    keys = get_secret("AABHAS_ADMIN_API_KEYS", "")
    return [k.strip() for k in keys.split(",") if k.strip()]


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        print(get_secret(sys.argv[1]) or "NOT FOUND")
    else:
        print("Available secrets (from env):")
        for key in ["HF_TOKEN", "GCP_PROJECT_ID", "AABHAS_ADMIN_API_KEYS"]:
            val = os.getenv(key)
            if val:
                print(f"  {key}: {'*' * min(len(val), 8)}...")