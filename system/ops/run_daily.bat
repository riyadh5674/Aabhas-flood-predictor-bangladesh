@echo off
REM Aabhas - Daily data refresh
REM Runs ingestion + feature computation (backfilling gaps in the last 30 days),
REM then incrementally updates the precomputed static snapshot and uploads only
REM the changed files to the Hugging Face Static Space.
REM Intended Task Scheduler target: daily at 06:00 after FFWC posts previous-day data.
REM Register it with: powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1
REM Publishing uses the token stored by `hf auth login` (or HF_TOKEN if set) and
REM HF_SPACE (default riyadhuggy/aabhas-flood-outlook). Without a token the local
REM pipeline still runs and publishing is skipped with a clear log line.

setlocal

set ROOT=%~dp0..\..
set PYTHON=%ROOT%\.venv\Scripts\python.exe
set LOG=%ROOT%\system\logs\daily_pipeline.log

if not exist "%ROOT%\system\logs\" mkdir "%ROOT%\system\logs\"

echo [%date% %time%] Starting daily refresh >> "%LOG%"

"%PYTHON%" "%ROOT%\system\ops\refresh_and_publish.py" --catchup >> "%LOG%" 2>&1

if %errorlevel% equ 0 (
    echo [%date% %time%] Daily refresh completed successfully >> "%LOG%"
) else (
    echo [%date% %time%] Daily refresh FAILED with error code %errorlevel% >> "%LOG%"
)

endlocal
