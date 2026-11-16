# One-shot administrator controller. Triggerless scheduled tasks call this file.
[CmdletBinding()]
param([ValidateSet('On','Off','Status')][string]$Action = 'Status',
      [string]$Root = '', [switch]$Pause)
# -Pause remains accepted for old task definitions; controllers always exit.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repo = if ($Root) {[IO.Path]::GetFullPath($Root)} else {[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))}
$State = Join-Path $env:LOCALAPPDATA 'c500-guard'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$Marker = Join-Path $State 'power-off'
$OffGeneration = Join-Path $State 'power-off-request.json'
$Paused = Join-Path $State 'service-paused'
$ReceiptPath = Join-Path $State 'power-receipt.json'
$RequestId = [Guid]::NewGuid().ToString()
$StartedAt = [DateTime]::UtcNow.ToString('o')
$Receipt = [ordered]@{
    schema_version = 1; action = $Action.ToLowerInvariant(); request_id = $RequestId
    started_at = $StartedAt; completed_at = $null; ok = $false; complete = $false; power_state = 'transitioning'
    errors = @(); remaining_processes = @(); unverified_processes = @(); listening_ports = @()
    stopped_processes = @(); checks = [ordered]@{}
}
$TransitionLock = $null
$ExpectedOffGeneration = $null

function Get-OffRequest {
    # This generation record is never deleted. It detects an OFF racing the
    # short interval between ON checking and removing the ordinary stop marker.
    if (Test-Path -LiteralPath $OffGeneration) {
        # Readers share delete access so an overlapping OFF can atomically replace
        # this record while ON reads the complete previous generation.
        $stream = [IO.File]::Open($OffGeneration, [IO.FileMode]::Open, [IO.FileAccess]::Read,
            ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
        $reader = [IO.StreamReader]::new($stream)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    }
    return ''
}
function Write-AtomicText([string]$Destination, [string]$Data, [string]$Prefix) {
    $temp = Join-Path $State ("$Prefix.$RequestId.tmp")
    [IO.File]::WriteAllText($temp, $Data, [Text.UTF8Encoding]::new($false))
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            if (Test-Path -LiteralPath $Destination) {
                # PowerShell 5.1 converts plain $null to an illegal empty path here.
                [IO.File]::Replace($temp, $Destination, [NullString]::Value)
            } else { [IO.File]::Move($temp, $Destination) }
            return
        } catch [IO.IOException] {
            if ($attempt -eq 19) { throw }
            Start-Sleep -Milliseconds 25
        }
    }
}
function Write-OffRequest {
    $data = @{request_id=$RequestId; requested_at=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress
    Write-AtomicText $OffGeneration $data 'power-off-request'
    [IO.File]::WriteAllText($Marker, $data)
    [IO.File]::WriteAllText($Paused, "Full program OFF. Open the BarrierWatcher app to start it again.`r`n")
}
function Assert-OnAllowed {
    if ((Test-Path -LiteralPath $Marker) -or (Get-OffRequest) -ne $ExpectedOffGeneration) {
        throw 'An OFF request cancelled startup.'
    }
}
function Confirm-OnRequest([string]$InitialRequest) {
    if ((Get-OffRequest) -ne $InitialRequest) { throw 'A newer OFF request cancelled the queued startup.' }
}
function Write-Receipt {
    $Receipt.completed_at = [DateTime]::UtcNow.ToString('o')
    $Receipt.complete = $Receipt.ok
    # Same-volume rename replaces the old receipt; readers never see partial JSON.
    Write-AtomicText $ReceiptPath ($Receipt | ConvertTo-Json -Depth 8) 'power-result'
    # Keep the last 20 receipts as well, so an earlier ON/OFF can still be checked.
    try {
        $history = Join-Path $State 'power-receipts'
        New-Item -ItemType Directory -Path $history -Force | Out-Null
        $name = '{0}-{1}.json' -f ([DateTime]::Now.ToString('yyyyMMdd-HHmmss')), $Receipt.action
        Copy-Item -LiteralPath $ReceiptPath -Destination (Join-Path $history $name) -Force
        Get-ChildItem -LiteralPath $history -Filter '*.json' | Sort-Object Name -Descending |
            Select-Object -Skip 20 | Remove-Item -Force -ErrorAction SilentlyContinue
    } catch { } # History is a convenience; the main receipt above is authoritative.
}
function Get-OwnedTasks {
    $wanted = @{'c500-guard service'='service.py'; 'c500-guard hotkey'='hotkey_win.py';
                'c500-guard watcher'='watch.py'; 'c500-guard'='watch.py'}
    $found = @(Get-ScheduledTask -ErrorAction Stop | Where-Object {
        $_.TaskPath -eq '\' -and $wanted.ContainsKey($_.TaskName)
    })
    foreach ($task in $found) {
        $actions = @($task.Actions)
        $script = Join-Path (Join-Path $Repo 'src') $wanted[$task.TaskName]
        $pattern = '^"' + [Regex]::Escape($script) + '"(?:\s|$)'
        if ($actions.Count -ne 1 -or
            [IO.Path]::GetFileName($actions[0].Execute) -notmatch '^python(?:\d+(?:\.\d+)*)?w?\.exe$' -or
            $actions[0].Arguments -notmatch $pattern) {
            throw "Cannot verify scheduled-task ownership: $($task.TaskName)."
        }
    }
    return $found
}
function Disable-Workers {
    foreach ($task in @(Get-OwnedTasks)) {
        Disable-ScheduledTask -InputObject $task -ErrorAction Stop | Out-Null
    }
}
function Stop-Workers {
    # Disable before stop prevents Scheduler's legacy retry policy from racing us.
    $tasks = @(Get-OwnedTasks)
    foreach ($task in $tasks) { Disable-ScheduledTask -InputObject $task -ErrorAction Stop | Out-Null }
    foreach ($task in ($tasks | Sort-Object @{Expression={if($_.TaskName -eq 'c500-guard hotkey'){0}else{1}}})) {
        Stop-ScheduledTask -InputObject $task -ErrorAction Stop
    }
}
function Invoke-ProcessCheck([string]$Mode) {
    $raw = & $Python (Join-Path $PSScriptRoot 'power_processes.py') $Mode --root $Repo
    $code = $LASTEXITCODE
    try { $result = ($raw | Out-String) | ConvertFrom-Json -ErrorAction Stop }
    catch { throw 'The guarded process inventory did not return a valid result.' }
    if ($null -eq $result -or $null -eq $result.PSObject.Properties['owned'] -or
        $null -eq $result.PSObject.Properties['unverified']) {
        throw 'The guarded process inventory returned an incomplete result.'
    }
    if ($code -ne 0 -and $Mode -eq 'inventory' -and @($result.unverified).Count -eq 0) {
        throw 'The guarded process inventory failed.'
    }
    if ($Mode -eq 'stop' -and ($null -eq $result.PSObject.Properties['stopped'] -or
        $null -eq $result.PSObject.Properties['errors'])) { throw 'The guarded stop did not return a complete result.' }
    return $result
}
function Get-C500Listeners {
    $tcpPorts = @(8787,1984,8554,32108)
    # TIME_WAIT is normal after shutdown and is deliberately not counted.
    $tcp = @(Get-NetTCPConnection -ErrorAction Stop | Where-Object {
        $_.State -eq 'Listen' -and $_.LocalPort -in $tcpPorts
    } | ForEach-Object { @{protocol='tcp'; port=[int]$_.LocalPort; pid=[int]$_.OwningProcess} })
    $udp = @(Get-NetUDPEndpoint -ErrorAction Stop | Where-Object {
        $_.LocalPort -eq 32108
    } | ForEach-Object { @{protocol='udp'; port=[int]$_.LocalPort; pid=[int]$_.OwningProcess} })
    return @($tcp) + @($udp)
}
function Invoke-QuietDisarm {
    try {
        $request = [Net.HttpWebRequest]::Create('http://127.0.0.1:8787/disarm')
        $request.Method = 'POST'; $request.Timeout = 1200; $request.ContentLength = 0
        $request.ReadWriteTimeout = 1200; $request.Proxy = $null
        $request.Headers.Add('X-Via','ctl')
        $response = $request.GetResponse(); $response.Close()
    } catch { } # A stopped watcher is expected; guarded cleanup is authoritative.
}
function Get-StopVerification {
    $after = Invoke-ProcessCheck 'inventory'
    $Receipt.remaining_processes = @($after.owned)
    $Receipt.unverified_processes = @($after.unverified)
    $Receipt.listening_ports = @(Get-C500Listeners)
    $tasks = @(Get-OwnedTasks)
    $stillEnabled = @($tasks | Where-Object { $_.Settings.Enabled })
    $stillRunning = @($tasks | Where-Object { $_.State -in @('Running','Queued') })
    $Receipt.checks['legacy_tasks_disabled'] = $stillEnabled.Count -eq 0
    $Receipt.checks['legacy_tasks_stopped'] = $stillRunning.Count -eq 0
    $Receipt.checks['stop_marker_present'] = Test-Path -LiteralPath $Marker
    $Receipt.checks['no_owned_processes'] = $Receipt.remaining_processes.Count -eq 0
    $Receipt.checks['ownership_verified'] = $Receipt.unverified_processes.Count -eq 0
    $Receipt.checks['no_listeners'] = $Receipt.listening_ports.Count -eq 0
    return $stillEnabled.Count -eq 0 -and $stillRunning.Count -eq 0 -and
        $Receipt.remaining_processes.Count -eq 0 -and $Receipt.unverified_processes.Count -eq 0 -and
        $Receipt.listening_ports.Count -eq 0 -and $Receipt.checks['stop_marker_present']
}
function Test-StartupHealth {
    param($Status, [switch]$RequireArmed)
    # Missing/old API fields are not healthy; strict-mode property errors fail closed.
    try {
        $aiAge = $Status.ml.age_s
        $numericAge = $aiAge -is [double] -or $aiAge -is [single] -or $aiAge -is [decimal] -or
            $aiAge -is [int] -or $aiAge -is [long]
        if (-not $numericAge -or [double]::IsNaN([double]$aiAge) -or
            [double]::IsInfinity([double]$aiAge) -or $aiAge -lt 0 -or $aiAge -ge 3) { return $false }
        if ($RequireArmed -and -not $Status.armed) { return $false }
        return $Status.frames -gt 0 -and $null -ne $Status.last_frame_age -and
            $Status.last_frame_age -lt 3 -and $Status.ml.available -and
            $Status.ml.state -eq 'ready' -and $Status.ml.inferences -gt 0 -and
            $Status.input_helper.connected -and $Status.alignment.state -in @('tracking','holding','disabled') -and
            $null -ne $Status.loop_age_s -and $Status.loop_age_s -lt 5
    } catch { return $false }
}
function Start-Workers {
    $workers = @(Get-OwnedTasks | Where-Object { $_.TaskName -in @('c500-guard service','c500-guard hotkey') })
    if ($workers.Count -ne 2) { throw 'The installed supervisor or input-helper task is missing.' }
    foreach ($task in $workers) {
        if (@($task.Triggers | Where-Object {$null -ne $_}).Count -gt 0 -or $task.Settings.RestartCount -gt 0) {
            throw "Automatic startup/retry settings must be removed from $($task.TaskName)."
        }
    }
    foreach ($task in $workers) {
        Assert-OnAllowed
        Enable-ScheduledTask -InputObject $task -ErrorAction Stop | Out-Null
        Assert-OnAllowed
        Start-ScheduledTask -InputObject $task -ErrorAction Stop
    }
}
function Stop-FullStack {
    if (-not (Test-Path -LiteralPath $Marker)) { Write-OffRequest }
    # Give the watcher a bounded chance to silence the alarm and restore the
    # saved speaker volume before terminating its task/process tree. A watcher
    # that already observed the marker restores volume in normal shutdown.
    try { Invoke-QuietDisarm } catch { } # Best effort must never prevent full OFF.
    # Then release hooks and recover verified orphaned launcher children and
    # binaries. Do not trust PID files alone.
    try { Stop-Workers } catch { $Receipt.errors += $_.Exception.Message }
    try {
        $stopped = Invoke-ProcessCheck 'stop'
        $Receipt.stopped_processes = @($stopped.stopped)
        foreach ($problem in @($stopped.errors)) { $Receipt.errors += $problem.error }
    } catch { $Receipt.errors += $_.Exception.Message }
    Start-Sleep -Milliseconds 350
    $verified = Get-StopVerification
    $Receipt.ok = $Receipt.errors.Count -eq 0 -and $verified
    $Receipt.power_state = if ($Receipt.ok) {'off'} else {'error'}
    if (-not $Receipt.ok -and $Receipt.errors.Count -eq 0) {
        $Receipt.errors += 'Shutdown could not verify that every C500 component stopped. Review the receipt.'
    }
}

# Status never changes marker, task state, or the last transition receipt.
if ($Action -eq 'Status') {
    try {
        $verified = Get-StopVerification
        if ($verified) { $Receipt.power_state = 'off'; $Receipt.ok = $true }
        elseif ($Receipt.remaining_processes.Count -gt 0) { $Receipt.power_state = 'running'; $Receipt.ok = $true }
        else { $Receipt.power_state = 'unknown' }
        if ($Receipt.unverified_processes.Count -gt 0) { $Receipt.ok = $false; $Receipt.power_state = 'unknown' }
        if ($Receipt.checks['stop_marker_present'] -and -not $verified) {
            $Receipt.ok = $false; $Receipt.power_state = 'stopping_or_incomplete'
        }
    } catch { $Receipt.errors += $_.Exception.Message; $Receipt.power_state = 'unknown' }
    $Receipt.complete = $Receipt.ok
    $Receipt.completed_at = [DateTime]::UtcNow.ToString('o')
    $Receipt | ConvertTo-Json -Depth 8
    if ($Receipt.ok) { exit 0 } else { exit 1 }
}

try {
    New-Item -ItemType Directory -Path $State -Force | Out-Null
    $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'The full-program controller requires the installed administrator scheduled task.'
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'The installed C500 Python runtime is missing.' }
    $InitialOffRequest = Get-OffRequest
    $ExpectedOffGeneration = $InitialOffRequest
    if ($Action -eq 'Off') {
        # Cancellation is visible even if another transition currently owns the lock.
        Write-OffRequest
        try { Disable-Workers } catch { $Receipt.errors += $_.Exception.Message }
    }
    $lockPath = Join-Path $State 'power-transition.lock'
    $deadline = [DateTime]::UtcNow.AddSeconds(95)
    while (-not $TransitionLock) {
        try { $TransitionLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
        catch [IO.IOException] {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Another power transition did not finish in time.' }
            Start-Sleep -Milliseconds 200
        }
    }
    if ($Action -eq 'Off') { Stop-FullStack }
    else {
        # A later OFF wins over an ON request queued behind another transition.
        Confirm-OnRequest $InitialOffRequest
        if (Test-Path -LiteralPath $Marker) { Remove-Item -LiteralPath $Marker -Force }
        Assert-OnAllowed
        if (Test-Path -LiteralPath $Paused) { Remove-Item -LiteralPath $Paused -Force }
        Start-Workers
        $ready = $false
        $deadline = [DateTime]::UtcNow.AddSeconds(75)
        while ([DateTime]::UtcNow -lt $deadline) {
            Assert-OnAllowed
            try {
                $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8787/state' -TimeoutSec 2 -ErrorAction Stop
                $ready = Test-StartupHealth $status
                if ($ready) { break }
            } catch { $ready = $false }
            Start-Sleep -Milliseconds 500
        }
        Assert-OnAllowed
        if (-not $ready) { throw 'Camera, AI and input helper did not become healthy within 75 seconds.' }
        $armed = Invoke-RestMethod -Uri 'http://127.0.0.1:8787/arm' -Method Post -Headers @{'X-Via'='ctl'} -TimeoutSec 3
        Assert-OnAllowed
        if (-not $armed.armed) { throw 'The watcher did not confirm alerts armed.' }
        # Arming resets the detector epoch. Wait for actual inference in the new
        # epoch rather than recording the pre-arm model availability as readiness.
        $ready = $false
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        while ([DateTime]::UtcNow -lt $deadline) {
            Assert-OnAllowed
            try {
                $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8787/state' -TimeoutSec 2 -ErrorAction Stop
                $ready = Test-StartupHealth $status -RequireArmed
                if ($ready) { break }
            } catch { $ready = $false }
            Start-Sleep -Milliseconds 500
        }
        Assert-OnAllowed
        if (-not $ready) { throw 'Camera, AI and input helper did not confirm healthy armed operation within 10 seconds.' }
        $Receipt.checks['camera_ready'] = $true
        $Receipt.checks['ai_ready'] = $true
        $Receipt.checks['input_helper_ready'] = $true
        $Receipt.checks['alerts_armed'] = $true
        $Receipt.ok = $true; $Receipt.power_state = 'on'
    }
} catch {
    $Receipt.errors += $_.Exception.Message
    $Receipt.ok = $false; $Receipt.power_state = 'error'
    if ($TransitionLock -and $Action -eq 'On') {
        # Partial startup must not leave a hook or camera process behind.
        try { Stop-FullStack } catch { $Receipt.errors += $_.Exception.Message }
        $Receipt.ok = $false
        $Receipt.power_state = if ($Receipt.checks.Contains('no_owned_processes') -and
            $Receipt.checks['no_owned_processes'] -and $Receipt.checks['ownership_verified'] -and
            $Receipt.checks['no_listeners'] -and $Receipt.checks['legacy_tasks_disabled'] -and
            $Receipt.checks['legacy_tasks_stopped']) {'off'} else {'error'}
    }
} finally {
    try { Write-Receipt } catch {
        $Receipt.ok = $false
        Write-Error "Unable to save C500 power receipt: $($_.Exception.Message)" -ErrorAction Continue
    }
    if ($TransitionLock) { $TransitionLock.Dispose() }
}
if ($Receipt.ok) { exit 0 } else { exit 1 }
