# Start the Aabhas demo (API + web) from the repository root.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Create it first:" -ForegroundColor Yellow
    Write-Host "  py -3.13 -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

$web = Join-Path $root "system\web"
if (-not (Test-Path (Join-Path $web "node_modules"))) {
    Write-Host "Installing frontend dependencies (first run)..."
    Push-Location $web
    npm ci
    Pop-Location
}

Write-Host ""
Write-Host "Starting Aabhas..." -ForegroundColor Green
Write-Host "  Web: http://localhost:5173"
Write-Host "  API: http://localhost:8000/health"
Write-Host ""

# The Vite dev launcher starts the API itself when port 8000 is free.
Push-Location $web
try {
    npm run dev
} finally {
    Pop-Location
}
