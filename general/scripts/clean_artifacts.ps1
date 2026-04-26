Param(
  [switch]$DryRun,
  [switch]$IncludeSharedVenv
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$targets = Get-ChildItem -Path $projectRoot -Recurse -Directory -Force | Where-Object {
  $isArtifactName = $_.Name -eq ".venv_debug" -or $_.Name -eq "build" -or $_.Name -eq "__pycache__" -or $_.Name -like "*.egg-info"
  if (-not $isArtifactName) { return $false }
  if ($_.FullName -like "*\.git\*") { return $false }
  if (-not $IncludeSharedVenv -and ($_.FullName -like "*\.venv_debug_shared*")) { return $false }
  return $true
} | Sort-Object FullName -Descending

$count = @($targets).Count
if ($count -eq 0) {
  Write-Host "[clean] no artifact directories found"
  exit 0
}

Write-Host "[clean] directories matched: $count"
if ($DryRun) {
  $targets | ForEach-Object { Write-Host ("[dryrun] " + $_.FullName) }
  exit 0
}

foreach ($t in $targets) {
  Remove-Item -LiteralPath $t.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "[clean] removed directories: $count"
