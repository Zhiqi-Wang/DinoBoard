Param(
  [Parameter(Mandatory = $true)][string]$Game,
  [string]$CppModuleName,
  [switch]$ForceInstall,
  [switch]$WithOnnx,
  [string]$OnnxRuntimeRoot = "",
  [switch]$AutoDiscoverOnnx
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$root = Join-Path $projectRoot "games/$Game/debug_service"
$venv = Join-Path $projectRoot ".venv_debug_shared"
$pythonExe = Join-Path $venv "Scripts/python.exe"
$reqFile = Join-Path $projectRoot "requirements.debug_shared.txt"
$hashFile = Join-Path $venv ".requirements.debug_shared.sha256"
$venvCreated = $false

function Resolve-CppModuleName {
  if ($script:CppModuleName) {
    $script:CppModuleName = "$script:CppModuleName".Trim()
    return $script:CppModuleName
  }
  try {
    $resolvedModule = & $pythonExe -c "import importlib,sys; sys.path.insert(0, r'$projectRoot'); p=importlib.import_module('games.$Game.debug_service.plugin').DEBUG_PLUGIN; print((getattr(p, 'cpp_extension_name', None) or '').strip())"
    if ($LASTEXITCODE -eq 0) {
      $candidate = ""
      if ($resolvedModule -is [array]) {
        $candidate = ($resolvedModule | Select-Object -Last 1)
      } else {
        $candidate = $resolvedModule
      }
      $script:CppModuleName = "$candidate".Trim()
      if ($script:CppModuleName) {
        Write-Host "[setup] resolved cpp module from DEBUG_PLUGIN: $script:CppModuleName"
      }
    }
  } catch {
    Write-Host "[setup] warn: failed to resolve cpp module from DEBUG_PLUGIN"
  }
  return $script:CppModuleName
}

function Stop-DebugLockProcesses {
  $projectRootLower = $projectRoot.ToLower()
  $roots = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -match '^(python|python3|pythonw|py)(\.exe)?$' -and
      $null -ne $_.CommandLine -and (
        $_.CommandLine.ToLower().Contains('general.debug_hub.app') -or
        $_.CommandLine.ToLower().Contains('.debug_service.app')
      )
    })

  if ($roots.Count -gt 0) {
    $rootIds = @($roots | ForEach-Object { [int]$_.ProcessId })
    $treeRoots = @($roots | Where-Object { $rootIds -notcontains [int]$_.ParentProcessId })
    if ($treeRoots.Count -eq 0) {
      $treeRoots = $roots
    }
    foreach ($proc in $treeRoots) {
      cmd /c "taskkill /PID $($proc.ProcessId) /T /F" 2>$null | Out-Null
    }
  }

  $lockProcIds = @(
    Get-Process -Name python,python3,pythonw,py -ErrorAction SilentlyContinue | ForEach-Object {
      try {
        $hasProjectEngineModule = @(
          $_.Modules | Where-Object {
            $_.ModuleName -like 'cpp_*_engine*.pyd' -and
            $null -ne $_.FileName -and
            $_.FileName.ToLower().Contains($projectRootLower)
          }
        ).Count -gt 0
        if ($hasProjectEngineModule) {
          $_.Id
        }
      } catch {
        # Ignore access-denied process metadata and continue.
      }
    } | Sort-Object -Unique
  )

  foreach ($procId in $lockProcIds) {
    cmd /c "taskkill /PID $procId /T /F" 2>$null | Out-Null
  }
}

if (-not (Test-Path $pythonExe)) {
  Write-Host "[setup] create venv: $venv"
  python -m venv $venv
  if ($LASTEXITCODE -ne 0) { throw "[setup] failed to create venv" }
  $venvCreated = $true
}

$reqHash = (Get-FileHash -Algorithm SHA256 $reqFile).Hash
$oldHash = ""
if (Test-Path $hashFile) {
  $oldHash = (Get-Content $hashFile -Raw).Trim()
}
$needInstall = $ForceInstall -or $venvCreated -or ($oldHash -ne $reqHash)

if ($WithOnnx -and -not $OnnxRuntimeRoot -and $AutoDiscoverOnnx) {
  $workspaceRoot = Split-Path -Parent $projectRoot
  $dinoBundledOnnx = Join-Path $projectRoot "third_party/onnxruntime-win-x64-1.17.3"
  if (Test-Path $dinoBundledOnnx) {
    $OnnxRuntimeRoot = $dinoBundledOnnx
  } else {
    $candidate = Get-ChildItem -Path $workspaceRoot -Directory -Recurse -Filter "onnxruntime-win-x64-1.17.3" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($candidate) {
      $OnnxRuntimeRoot = $candidate.FullName
    }
  }
}

if ($needInstall) {
  Write-Host "[setup] install requirements"
  & $pythonExe -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw "[setup] pip upgrade failed" }
  & $pythonExe -m pip install -r $reqFile
  if ($LASTEXITCODE -ne 0) { throw "[setup] pip install requirements failed" }
} else {
  Write-Host "[setup] requirements unchanged, skip reinstall"
}

Push-Location $root
try {
  # Ensure stale debug processes do not keep cpp_*_engine*.pyd locked.
  Stop-DebugLockProcesses
  Start-Sleep -Milliseconds 500

  if ($WithOnnx) {
    if (-not $OnnxRuntimeRoot) {
      throw "[setup] WithOnnx requires -OnnxRuntimeRoot"
    }
    Write-Host "[setup] install cpp extension with ONNX Runtime: $OnnxRuntimeRoot"
    $env:BOARD_AI_WITH_ONNX = "1"
    $env:BOARD_AI_ONNXRUNTIME_ROOT = $OnnxRuntimeRoot
  }

  & $pythonExe -m pip install -e .
  if ($LASTEXITCODE -ne 0) {
    if ($WithOnnx) {
      throw "[setup] pip install -e . failed (WithOnnx)"
    }
    throw "[setup] pip install -e . failed"
  }
} finally {
  Pop-Location
  Remove-Item Env:\BOARD_AI_WITH_ONNX -ErrorAction SilentlyContinue
  Remove-Item Env:\BOARD_AI_ONNXRUNTIME_ROOT -ErrorAction SilentlyContinue
}

Set-Content -Path $hashFile -Value $reqHash -Encoding ASCII

$null = Resolve-CppModuleName

if ($CppModuleName) {
  Write-Host "[setup] verify imports"
  & $pythonExe -c "import fastapi,uvicorn,pydantic,$CppModuleName; print('env_ok', fastapi.__version__, uvicorn.__version__, pydantic.__version__)"
  if ($LASTEXITCODE -ne 0) { throw "[setup] import verification failed for module: $CppModuleName" }
}

Write-Host "[setup] done"
