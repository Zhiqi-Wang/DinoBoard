Param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $root ".venv_debug_shared/Scripts/python.exe"

if (-not (Test-Path $pythonExe)) {
  Write-Host "[run] shared debug env not found. initialize one game first, e.g. games/azul/debug_service/setup_env.ps1"
  exit 1
}

Write-Host "[run] start DinoBoard Debug Hub on port $Port"
& $pythonExe -m uvicorn general.debug_hub.app:app --reload --app-dir "$root" --port $Port

