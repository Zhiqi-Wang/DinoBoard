Param(
  [switch]$ForceInstall,
  [switch]$WithOnnx = $true,
  [string]$OnnxRuntimeRoot = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
& (Join-Path $projectRoot "general/scripts/setup_game_env.ps1") `
  -Game "azul" `
  -ForceInstall:$ForceInstall `
  -WithOnnx:$WithOnnx `
  -OnnxRuntimeRoot $OnnxRuntimeRoot `
  -AutoDiscoverOnnx
