#!/usr/bin/env bash
# Start the Aabhas demo (API + web) from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "No .venv found. Create it first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "system/web/node_modules" ]]; then
  echo "Installing frontend dependencies (first run)..."
  (cd system/web && npm ci)
fi

echo
echo "Starting Aabhas..."
echo "  Web: http://localhost:5173"
echo "  API: http://localhost:8000/health"
echo
cd system/web
exec npm run dev
