<#
    c500-guard installer (Windows)

    Installs prerequisites, builds the Python environment, fetches and verifies
    go2rtc, and registers scheduled tasks so the alarm and the Ctrl+Alt+F4 key
    start automatically at logon.

    Run from an ELEVATED PowerShell:
        Set-ExecutionPolicy -Scope Process Bypass -Force
        .\install.ps1

    Safe to run repeatedly; every step checks before acting.
#>
[CmdletBinding()]
param(
    [switch]$NoTasks,          # install everything but do not register autostart
    [switch]$Uninstall,        # remove the scheduled tasks again
    [string]$BridgeBundle = '' # folder containing go2rtc.exe + go2rtc-build.json
)

$ErrorActionPreference = 'Stop'
$Root      = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Venv      = Join-Path $Root '.venv'
$VenvPy    = Join-Path $Venv 'Scripts\python.exe'
$Src       = Join-Path $Root 'src'
$TaskGuard = 'c500-guard watcher'
$TaskService = 'c500-guard service'
$TaskKey   = 'c500-guard hotkey'
if (-not $BridgeBundle) { $BridgeBundle = Join-Path $Root 'tools\go2rtc' }

function Say  ($m) { Write-Host "  $m" }
function Good ($m) { Write-Host "  [ OK ] $m"   -ForegroundColor Green }
function Warn ($m) { Write-Host "  [WARN] $m"   -ForegroundColor Yellow }
function Die  ($m) { Write-Host "  [FAIL] $m"   -ForegroundColor Red; exit 1 }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host ''
Write-Host 'c500-guard installer' -ForegroundColor Cyan
Write-Host '====================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Current Windows setup and repair instructions:' -ForegroundColor Cyan
Write-Host "    $Root\docs\WINDOWS.md"
Write-Host '  The older PDF covers historical camera-login troubleshooting.'

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
    foreach ($t in @($TaskGuard, $TaskService, $TaskKey)) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $t -Confirm:$false
            Good "removed scheduled task: $t"
        } else { Say "not present: $t" }
    }
    Write-Host ''
    Say 'Autostart removed. Program files and your camera login were left in place.'
    exit 0
}

if (-not (Test-Admin)) {
    Warn 'Not running as Administrator.'
    Warn 'Autostart registration and the global hotkey both need elevation.'
    Warn 'Close this window and re-open PowerShell with "Run as administrator".'
    if (-not $NoTasks) { Die 'Aborting so you do not get a half-installed system.' }
}

# ------------------------------------------------------------------ python
Say 'Checking Python...'
function Find-Python {
    # New user installs are usable before a new logon refreshes PATH.
    $candidates = @()
    foreach ($version in @('313', '312', '311', '310', '314')) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$version\python.exe"
        if (Test-Path -LiteralPath $candidate) {
            $candidates += @{Exe=$candidate; Args=@()}
        }
    }
    foreach ($name in @('py', 'python')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notlike '*\WindowsApps\*') {
            $candidates += @{Exe=$command.Source; Args=$(if ($name -eq 'py') { @('-3') } else { @() })}
        }
    }
    foreach ($candidate in $candidates) {
        try {
            $candidateArgs = @($candidate.Args)
            $versionText = & $candidate.Exe @candidateArgs --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $versionText -match 'Python (\d+)\.(\d+)') {
                if ([int]$Matches[1] -gt 3 -or
                    ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10)) {
                    return $candidate
                }
            }
        } catch { }
    }
    return $null
}

function Resolve-TaskPython ([string]$VenvPython) {
    # Windows venv pythonw is a redirector that can leave its base-interpreter
    # child alive when Task Scheduler stops the launcher. These two entrypoints
    # need only the stdlib, so let Scheduler own the base pythonw process itself.
    $baseExecutable = (& $VenvPython -c 'import sys; print(sys._base_executable)' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $baseExecutable -PathType Leaf)) {
        throw 'Could not resolve the virtual environment base Python for scheduled tasks.'
    }
    $taskPython = Join-Path (Split-Path -Parent $baseExecutable) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
        throw "Base pythonw.exe is missing: $taskPython"
    }
    return [IO.Path]::GetFullPath($taskPython)
}
$py = Find-Python
if (-not $py) {
    Say 'Installing Python 3.13 via winget...'
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Die 'winget not available. Install Python 3.10+ from python.org, tick "Add to PATH", then re-run.'
    }
    winget install --id Python.Python.3.13 -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path','User')
    if ($LASTEXITCODE -ne 0) { Die 'Python installation failed' }
    $py = Find-Python
    if (-not $py) {
        Die 'Python installed but is not on PATH yet. Close this window, open a NEW elevated PowerShell, and re-run this script.'
    }
}
$pyArgs = @($py.Args)
Good "Python: $(& $py.Exe @pyArgs --version 2>&1)"

# ------------------------------------------------------------------ ffmpeg
Say 'Checking ffmpeg...'
$localFfmpeg = Join-Path $Root 'tools\ffmpeg\bin'
if (Test-Path -LiteralPath (Join-Path $localFfmpeg 'ffmpeg.exe')) {
    $env:Path = $localFfmpeg + ';' + $env:Path
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Say 'Installing ffmpeg via winget...'
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { Die 'ffmpeg installation failed' }
        $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
                    [Environment]::GetEnvironmentVariable('Path','User')
    } else {
        Die 'ffmpeg missing and winget unavailable. Install ffmpeg and put it on PATH.'
    }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Die 'ffmpeg still not on PATH. Open a NEW PowerShell and re-run this script.'
}
Good "ffmpeg: $((Get-Command ffmpeg).Source)"

# --------------------------------------------------------------------- venv
$venvHealthy = $false
if (Test-Path -LiteralPath $VenvPy) {
    try {
        & $VenvPy -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>$null
        $venvHealthy = $LASTEXITCODE -eq 0
    } catch { }
}
if (-not $venvHealthy -and (Test-Path -LiteralPath $Venv)) {
    # A venv executable may exist while its original Python installation is gone.
    # Preserve the old directory and verify both move targets stay inside Root.
    $resolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolvedVenv = [IO.Path]::GetFullPath($Venv)
    $venvBackup = [IO.Path]::GetFullPath("$Venv.broken-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')")
    if (-not $resolvedVenv.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not $venvBackup.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Die 'Refusing to move a virtual environment outside the install directory.'
    }
    Warn "Existing virtual environment is broken; preserving it at $venvBackup"
    Move-Item -LiteralPath $resolvedVenv -Destination $venvBackup
}
if (-not $venvHealthy) {
    Say 'Creating virtual environment...'
    & $py.Exe @pyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Die 'virtual environment creation failed' }
}
if (-not (Test-Path $VenvPy)) { Die "venv creation failed ($VenvPy missing)" }
Say 'Installing numpy and pillow...'
& $VenvPy -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Die 'pip upgrade failed' }
& $VenvPy -m pip install --quiet numpy pillow
if ($LASTEXITCODE -ne 0) { Die 'pip install failed' }
Good 'Python packages installed'

if (Test-Path (Join-Path $Root 'requirements-ml.txt')) {
    & $VenvPy -m pip install --quiet -r (Join-Path $Root 'requirements-ml.txt')
    if ($LASTEXITCODE -ne 0) { Die 'Local AI packages failed to install' }
    & $VenvPy (Join-Path $Src 'install_ml.py')
    if ($LASTEXITCODE -ne 0) { Die 'Local AI model verification failed' }
    & $VenvPy (Join-Path $Src 'ml_selftest.py')
    if ($LASTEXITCODE -ne 0) { Die 'Local AI model could not complete its CPU self-test' }
    Good 'Local AI runtime and model installed; CPU inference passed'
}

# ------------------------------------------------------------------ go2rtc
Say 'Checking the tested local go2rtc build...'
$bridgeStateBin = Join-Path $env:LOCALAPPDATA 'c500-guard\bin'
$bridgeTarget = Join-Path $bridgeStateBin 'go2rtc.exe'
$bridgeManifestTarget = Join-Path $bridgeStateBin 'go2rtc-build.json'
$bundleBinary = Join-Path $BridgeBundle 'go2rtc.exe'
$bundleManifest = Join-Path $BridgeBundle 'go2rtc-build.json'
if ((Test-Path -LiteralPath $bundleBinary) -and (Test-Path -LiteralPath $bundleManifest)) {
    $build = Get-Content -LiteralPath $bundleManifest -Raw | ConvertFrom-Json
    if ($build.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or
        (Get-FileHash -LiteralPath $bundleBinary -Algorithm SHA256).Hash -ne $build.sha256) {
        Die 'Bundled go2rtc checksum does not match go2rtc-build.json'
    }
    New-Item -ItemType Directory -Path $bridgeStateBin -Force | Out-Null
    $sameBinary = (Test-Path -LiteralPath $bridgeTarget) -and
        (Get-FileHash -LiteralPath $bridgeTarget -Algorithm SHA256).Hash -eq $build.sha256
    if (-not $sameBinary) {
        $activeBridge = Get-Process -Name go2rtc -ErrorAction SilentlyContinue | Where-Object {
            $_.Path -and [IO.Path]::GetFullPath($_.Path) -eq [IO.Path]::GetFullPath($bridgeTarget)
        }
        if ($activeBridge) {
            Die 'Stop c500-guard service, run ctl.py stop and bridge.py stop, then rerun to update the bridge.'
        }
        $backupStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        foreach ($existing in @($bridgeTarget, $bridgeManifestTarget)) {
            if (Test-Path -LiteralPath $existing) {
                Copy-Item -LiteralPath $existing -Destination "$existing.bak-$backupStamp"
            }
        }
        Copy-Item -LiteralPath $bundleBinary -Destination $bridgeTarget -Force
    }
    Copy-Item -LiteralPath $bundleManifest -Destination $bridgeManifestTarget -Force
} elseif ((Test-Path -LiteralPath $bundleBinary) -or (Test-Path -LiteralPath $bundleManifest)) {
    Die 'Incomplete go2rtc bundle: keep go2rtc.exe and go2rtc-build.json together.'
} elseif (-not ((Test-Path -LiteralPath $bridgeTarget) -and (Test-Path -LiteralPath $bridgeManifestTarget))) {
    Die "The tested bridge bundle is missing. Restore $BridgeBundle (go2rtc.exe + go2rtc-build.json), or pass -BridgeBundle with that folder."
}
& $VenvPy (Join-Path $Src 'bridge.py') install
if ($LASTEXITCODE -ne 0) { Die 'go2rtc download or checksum verification failed' }
& $VenvPy (Join-Path $Src 'bridge.py') init
if ($LASTEXITCODE -ne 0) { Die 'bridge configuration initialization failed' }
Good 'go2rtc installed and verified'

# ------------------------------------------------------------------- zones
$zones = Join-Path $Root 'config\zones.json'
if (-not (Test-Path $zones)) {
    Copy-Item (Join-Path $Root 'config\zones.example.json') $zones
    Good 'created config\zones.json from the example'
}

# ------------------------------------------------------------ autostart
if (-not $NoTasks) {
    Say 'Registering autostart tasks...'
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $TaskPython = Resolve-TaskPython $VenvPy
    Good "Task interpreter: $TaskPython"

    # The supervisor restores bridge + optional relay before starting the watcher,
    # retries failed processes, and respects ctl.py stop and saved armed state.
    $aService = New-ScheduledTaskAction -Execute $TaskPython `
                -Argument "`"$(Join-Path $Src 'service.py')`"" `
                -WorkingDirectory $Src
    # The hotkey: needs an interactive desktop session to hook the keyboard.
    $aKey   = New-ScheduledTaskAction -Execute $TaskPython `
                -Argument "`"$(Join-Path $Src 'hotkey_win.py')`" --log `"$env:LOCALAPPDATA\c500-guard\hotkey.log`"" `
                -WorkingDirectory $Src

    $trigger  = New-ScheduledTaskTrigger -AtLogOn -User $user
    $principal= New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
                  -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
                  -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
                  -StartWhenAvailable

    # Retire the earlier direct watcher task to prevent duplicate capture loops.
    if (Get-ScheduledTask -TaskName $TaskGuard -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskGuard -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskGuard -Confirm:$false
    }
    foreach ($pair in @(@{n=$TaskService;a=$aService}, @{n=$TaskKey;a=$aKey})) {
        Register-ScheduledTask -TaskName $pair.n -Action $pair.a -Trigger $trigger `
            -Principal $principal -Settings $settings -Force | Out-Null
        Enable-ScheduledTask -TaskName $pair.n | Out-Null
        Good "registered: $($pair.n)"
    }
}

# ------------------------------------------------------------------- report
Write-Host ''
& $VenvPy (Join-Path $Src 'ctl.py') doctor
if ($LASTEXITCODE -ne 0) {
    Warn 'Setup files are ready, but runtime checks are incomplete. Connect/start the camera and repeat ctl.py doctor.'
}
Write-Host ''
Write-Host 'NEXT STEPS' -ForegroundColor Cyan
Write-Host '  1. Connect the camera (once):'
Write-Host "       $VenvPy `"$Src\bridge.py`" start"
Write-Host "       $VenvPy `"$Src\bridge.py`" login-details"
Write-Host '     Open the printed URL, expand Xiaomi, and sign in with your NUMERIC'
Write-Host '     Mi Account ID (not your email). Choose the region your Xiaomi Home'
Write-Host '     app uses. Then add the camera and note its name.'
Write-Host ''
Write-Host '  2. Start it (saved ON/OFF state is preserved):'
Write-Host "       $VenvPy `"$Src\ctl.py`" start"
Write-Host '     Then open http://127.0.0.1:8787'
if (-not $NoTasks) {
    Write-Host "       Start-ScheduledTask -TaskName '$TaskService'"
    Write-Host "       Start-ScheduledTask -TaskName '$TaskKey'"
    Write-Host '     These tasks also start automatically at your next sign-in.'
}
Write-Host ''
Write-Host '  3. While disarmed, draw and save zones at http://127.0.0.1:8787/zones'
Write-Host '     Confirm live frames and Local AI, test the speaker, then arm explicitly.'
Write-Host ''
Write-Host '  Ctrl+Alt+F4 turns the system on and off from anywhere.' -ForegroundColor Green
Write-Host ''
Write-Host 'IF SOMETHING GOES WRONG' -ForegroundColor Cyan
Write-Host "  Read: $Root\docs\c500-guard-runbook.pdf  (section 4 = every known error + fix)"
Write-Host '  The two that cost the most time:'
Write-Host '    - Xiaomi login must use your NUMERIC Mi Account ID, not your email (error 70016)'
Write-Host '    - Login can succeed and still list no devices if the region is wrong'
Write-Host ''
