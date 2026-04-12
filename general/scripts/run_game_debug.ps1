Param(
  [Parameter(Mandatory = $true)][string]$Game,
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $projectRoot ".venv_debug_shared/Scripts/python.exe"
$gameRoot = Join-Path $projectRoot "games/$Game/debug_service"

if (-not (Test-Path $pythonExe)) {
  Write-Host "[run] shared debug env not found, initializing for game=$Game ..."
  & (Join-Path $gameRoot "setup_env.ps1")
}
if (-not (Test-Path $pythonExe)) {
  throw "[run] failed to create shared debug env"
}

Write-Host "[run] start $Game debug service on port $Port"
Push-Location $projectRoot
try {
  & $pythonExe -m uvicorn "games.$Game.debug_service.app:app" --reload --app-dir "$projectRoot" --port $Port
} finally {
  Pop-Location
}
