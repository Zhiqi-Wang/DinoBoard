Param(
  [Parameter(Mandatory = $true)][string]$Game,
  [string]$OutputRoot = "",
  [int]$SleepSeconds = 3,
  [int]$RefreshSeconds = 5,
  [switch]$NoFollow,
  [switch]$OpenLogWindow,
  [switch]$ShowProcessWindow,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $projectRoot ".venv_debug_shared/Scripts/python.exe"
$runModule = "games.$Game.train.plugin"
$runModuleFile = Join-Path $projectRoot "games/$Game/train/plugin.py"
$setupScript = Join-Path $projectRoot "games/$Game/debug_service/setup_env.ps1"

if (-not (Test-Path $runModuleFile)) {
  throw "[run-train] training entry not found: $runModuleFile"
}
if (-not (Test-Path $pythonExe)) {
  if (Test-Path $setupScript) {
    Write-Host "[run-train] shared env missing, setup game env for $Game ..."
    & $setupScript
  }
}
if (-not (Test-Path $pythonExe)) {
  throw "[run-train] python env not found: $pythonExe"
}

$cppModuleName = ""
try {
  $resolvedModule = & $pythonExe -c "import importlib,sys; sys.path.insert(0, r'$projectRoot'); p=importlib.import_module('games.$Game.debug_service.plugin').DEBUG_PLUGIN; print((getattr(p, 'cpp_extension_name', None) or '').strip())"
  if ($LASTEXITCODE -eq 0) {
    if ($resolvedModule -is [array]) {
      $cppModuleName = ($resolvedModule | Select-Object -Last 1).ToString().Trim()
    } else {
      $cppModuleName = "$resolvedModule".Trim()
    }
  }
} catch {
  $cppModuleName = ""
}

if ($cppModuleName) {
  $importOk = $false
  try {
    & $pythonExe -c "import importlib; importlib.import_module('$cppModuleName')" 1>$null 2>$null
    $importOk = ($LASTEXITCODE -eq 0)
  } catch {
    $importOk = $false
  }
  if (-not $importOk) {
    if (Test-Path $setupScript) {
      Write-Host "[run-train] cpp module '$cppModuleName' missing, setup game env for $Game ..."
      & $setupScript
    }
    $importOkAfterSetup = $false
    try {
      & $pythonExe -c "import importlib; importlib.import_module('$cppModuleName')" 1>$null 2>$null
      $importOkAfterSetup = ($LASTEXITCODE -eq 0)
    } catch {
      $importOkAfterSetup = $false
    }
    if (-not $importOkAfterSetup) {
      throw "[run-train] cpp module import failed after setup: $cppModuleName"
    }
  }
}

$runsRoot = if ($OutputRoot) { $OutputRoot } else { Join-Path $projectRoot "games/$Game/train/runs" }
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $runsRoot $stamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$stdoutLog = Join-Path $runDir "train.log"
$stderrLog = Join-Path $runDir "train.err.log"
$argsLog = Join-Path $runDir "command.txt"

$trainArgsList = New-Object System.Collections.Generic.List[string]
$trainArgsList.Add("-m")
$trainArgsList.Add($runModule)

$hasOutput = $false
for ($i = 0; $i -lt $TrainArgs.Count; $i++) {
  $a = $TrainArgs[$i]
  if ($a -eq "--output") { $hasOutput = $true }
  if ($a.StartsWith("--output=")) { $hasOutput = $true }
  $trainArgsList.Add($a)
}
if (-not $hasOutput) {
  $trainArgsList.Add("--output")
  $trainArgsList.Add($runDir)
}

$cmdPreview = @()
$cmdPreview += $pythonExe
function Quote-Arg([string]$s) {
  if ($null -eq $s) { return "" }
  if ($s.Contains("`"")) {
    $s = $s.Replace("`"", "`"`"")
  }
  if ($s.IndexOfAny([char[]]@(' ', "`t")) -ge 0) {
    return "`"$s`""
  }
  return $s
}

$quotedArgs = @()
foreach ($a in $trainArgsList) {
  $quotedArgs += (Quote-Arg $a)
}
$cmdPreview += $quotedArgs
Set-Content -Path $argsLog -Value ($cmdPreview -join " ") -Encoding UTF8

$argString = $quotedArgs -join " "

$windowStyle = if ($ShowProcessWindow) { "Normal" } else { "Hidden" }

$p = Start-Process -FilePath $pythonExe `
  -ArgumentList $argString `
  -WorkingDirectory $projectRoot `
  -WindowStyle $windowStyle `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru

Write-Host "[run-train] started: game=$Game pid=$($p.Id)"
Write-Host "[run-train] run dir: $runDir"
Write-Host "[run-train] stdout:  $stdoutLog"
Write-Host "[run-train] stderr:  $stderrLog"
Write-Host "[run-train] command: $argsLog"

if ($SleepSeconds -gt 0) {
  Write-Host "[run-train] sleep $SleepSeconds sec before opening logs ..."
  Start-Sleep -Seconds $SleepSeconds
}

if ($OpenLogWindow) {
  $tailCmd = "Get-Content -Path '$stdoutLog' -Wait"
  Start-Process powershell -ArgumentList @("-NoExit", "-Command", $tailCmd) | Out-Null
}

if (-not $NoFollow) {
  $cursor = 0
  Write-Host "[run-train] following log every $RefreshSeconds sec (Ctrl+C to stop viewer only)"
  while (-not $p.HasExited) {
    if (Test-Path $stdoutLog) {
      $lines = Get-Content -Path $stdoutLog
      if ($lines.Count -gt $cursor) {
        for ($i = $cursor; $i -lt $lines.Count; $i++) {
          Write-Host $lines[$i]
        }
        $cursor = $lines.Count
      }
    }
    Start-Sleep -Seconds ([Math]::Max(1, $RefreshSeconds))
    $p.Refresh()
  }
  # Flush any remaining stdout lines after process exits.
  if (Test-Path $stdoutLog) {
    $lines = Get-Content -Path $stdoutLog
    if ($lines.Count -gt $cursor) {
      for ($i = $cursor; $i -lt $lines.Count; $i++) {
        Write-Host $lines[$i]
      }
      $cursor = $lines.Count
    }
  }

  # Ensure process state is finalized before reading ExitCode.
  $p.WaitForExit()
  $p.Refresh()
  $exitCode = $null
  try {
    $exitCode = $p.ExitCode
  } catch {
    $exitCode = $null
  }
  if ($null -eq $exitCode) {
    $fallback = "unknown"
    if (Test-Path $stdoutLog) {
      $stdoutText = Get-Content -Path $stdoutLog -Raw
      if ($stdoutText -match '"phase"\s*:\s*"completed"') {
        $fallback = "succeeded (phase=completed)"
      } elseif ((Test-Path $stderrLog) -and ((Get-Item $stderrLog).Length -gt 0)) {
        $fallback = "failed (stderr not empty)"
      }
    } elseif ((Test-Path $stderrLog) -and ((Get-Item $stderrLog).Length -gt 0)) {
      $fallback = "failed (stderr not empty)"
    }
    Write-Host "[run-train] process exited, but exit code is unavailable; status: $fallback"
  } else {
    Write-Host "[run-train] process exited with code $exitCode"
  }

  # Throughput summary from stdout log.
  if (Test-Path $stdoutLog) {
    $stdoutText = Get-Content -Path $stdoutLog -Raw
    if (-not [string]::IsNullOrEmpty($stdoutText)) {
      $m = [regex]::Matches($stdoutText, 'samples/s=([0-9]+(?:\.[0-9]+)?)')
      if ($m.Count -gt 0) {
        $last = $m[$m.Count - 1].Groups[1].Value
        Write-Host "[run-train] throughput: selfplay samples/s=$last"
      } else {
        $m2 = [regex]::Matches($stdoutText, '"selfplay_samples_per_sec"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        if ($m2.Count -gt 0) {
          $last2 = $m2[$m2.Count - 1].Groups[1].Value
          Write-Host "[run-train] throughput: selfplay samples/s=$last2"
        }
      }
    }
  }

  if (Test-Path $stderrLog) {
    $errLines = Get-Content -Path $stderrLog
    if ($errLines.Count -gt 0) {
      Write-Host "[run-train] stderr tail:"
      $tailN = [Math]::Min(20, $errLines.Count)
      for ($i = $errLines.Count - $tailN; $i -lt $errLines.Count; $i++) {
        Write-Host $errLines[$i]
      }
    }
  }
}
