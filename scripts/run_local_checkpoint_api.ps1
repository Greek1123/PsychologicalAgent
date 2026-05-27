$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = "D:\Anaconda\python.exe"
if (-not (Test-Path $python)) {
    throw "Cannot find $python. Please update scripts\run_local_checkpoint_api.ps1 to your Python path."
}

Write-Host "Starting Campus Support Agent with local checkpoint provider..."
Write-Host "Project: $projectRoot"
Write-Host "Python:  $python"
Write-Host "URL:     http://127.0.0.1:8000/app"
Write-Host ""
Write-Host "Tip: run this in another terminal after startup:"
Write-Host "  D:\Anaconda\python.exe scripts\evaluate_api_quality.py --base-url http://127.0.0.1:8000 --timeout-seconds 240"
Write-Host ""

Write-Host "Running deployment readiness check..."
& $python scripts\check_deployment_readiness.py
if ($LASTEXITCODE -ne 0) {
    throw "Deployment readiness check failed. Fix the blocked items above before starting the API."
}
Write-Host ""

& $python -m uvicorn campus_support_agent.main:app --app-dir src --port 8000
