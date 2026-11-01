@echo off
REM FloodGuard Bangladesh - Daily Pipeline
REM Runs ingestion + feature computation for yesterday's date
REM Intended Task Scheduler target: daily at 06:00 after FFWC posts previous-day data.
REM This script does not create or register a scheduled task.

setlocal

set ROOT=D:\Flood predictor
set PYTHON=%ROOT%\.venv\Scripts\python.exe
set LOG=%ROOT%\system\logs\daily_pipeline.log

if not exist "%ROOT%\system\logs\" mkdir "%ROOT%\system\logs\"

echo [%date% %time%] Starting daily pipeline >> "%LOG%"

"%PYTHON%" "%ROOT%\system\ops\daily_pipeline.py" >> "%LOG%" 2>&1

if %errorlevel% equ 0 (
    echo [%date% %time%] Daily pipeline completed successfully >> "%LOG%"
) else (
    echo [%date% %time%] Daily pipeline FAILED with error code %errorlevel% >> "%LOG%"
)

endlocal
