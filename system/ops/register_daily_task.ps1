# Register the Aabhas daily refresh as a Windows Scheduled Task.
#
# Usage (no admin required for a per-user task):
#   powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1
#   powershell -ExecutionPolicy Bypass -File system\ops\register_daily_task.ps1 -Time 07:30
#
# The task runs system\ops\run_daily.bat, which calls refresh_and_publish.py.
# Publishing uses the token stored by `hf auth login` (or HF_TOKEN in the user
# environment) plus HF_SPACE (default riyadhuggy/aabhas-flood-outlook). Without
# a token the pipeline still runs and the upload step is skipped.
# Check the log at system\logs\daily_pipeline.log.

param(
    [string]$TaskName = "Aabhas Daily Data Pipeline",
    [string]$Time = "06:00"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script = Join-Path $root "system\ops\run_daily.bat"

if (-not (Test-Path $script)) {
    throw "Missing daily script: $script"
}

$action = New-ScheduledTaskAction -Execute $script -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Aabhas daily ingestion, feature build and static-site publish" `
    -Force | Out-Null

Write-Host "Registered '$TaskName' daily at $Time."
Write-Host "Run now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Inspect:  Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host "Log:      $root\system\logs\daily_pipeline.log"
