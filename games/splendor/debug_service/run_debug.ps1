Param(
  [int]$Port = 8014
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
& (Join-Path $projectRoot "general/scripts/run_game_debug.ps1") -Game "splendor" -Port $Port

