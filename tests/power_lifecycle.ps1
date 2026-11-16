# Mock-only tests: parse production functions, never execute the controller body.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$controller = Join-Path $Repo 'src\power_win.ps1'
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($controller, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$functions = @($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]}, $false))
foreach ($definition in $functions) { . ([scriptblock]::Create($definition.Extent.Text)) }
foreach ($path in @('tools\c500-power.ps1','src\install_power.ps1')) {
    $tokens = $null; $parseErrors = $null
    [void][Management.Automation.Language.Parser]::ParseFile((Join-Path $Repo $path), [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
}
$State = Join-Path ([IO.Path]::GetTempPath()) ('c500-power-test-' + [Guid]::NewGuid().ToString())
[void][IO.Directory]::CreateDirectory($State)
$Marker = Join-Path $State 'power-off'
$Paused = Join-Path $State 'service-paused'
$OffGeneration = Join-Path $State 'power-off-request.json'
$ReceiptPath = Join-Path $State 'power-receipt.json'
$RequestId = [Guid]::NewGuid().ToString()
$script:checks = 0
function Assert($condition, [string]$message) {
    if (-not $condition) { throw $message }
    $script:checks++
}
function Assert-Throws([scriptblock]$operation, [string]$message) {
    $caught = $false
    try { & $operation } catch { $caught = $true }
    Assert $caught $message
}
function Reset-Receipt {
    $script:Receipt = [ordered]@{ok=$false;complete=$false;completed_at=$null;power_state='transitioning';errors=@();
        remaining_processes=@();unverified_processes=@();listening_ports=@();stopped_processes=@();checks=[ordered]@{}}
}
function New-Worker([string]$name, [string]$file) {
    return [pscustomobject]@{TaskName=$name;TaskPath='\';State='Disabled';Triggers=@();
        Settings=[pscustomobject]@{Enabled=$false;RestartCount=0};
        Actions=@([pscustomobject]@{Execute='C:\Python\pythonw.exe';Arguments='"'+(Join-Path $Repo ('src\'+$file))+'"'})}
}
# Every OS operation referenced by an extracted function is replaced here.
function Get-ScheduledTask { param($ErrorAction) return $script:taskRows }
function Disable-ScheduledTask { param($InputObject,$ErrorAction) $script:events += 'disable'; $InputObject.Settings.Enabled=$false }
function Stop-ScheduledTask { param($InputObject,$ErrorAction) $script:events += 'stop'; $InputObject.State='Disabled' }
function Enable-ScheduledTask { param($InputObject,$ErrorAction) $script:events += 'enable'; $InputObject.Settings.Enabled=$true; if ($script:cancelOnEnable) { Write-OffRequest } }
function Start-ScheduledTask { param($InputObject,$ErrorAction) $script:events += 'start'; $InputObject.State='Running' }
function Invoke-QuietDisarm {
    $script:events += 'disarm'
    $script:disarmSawMarker = Test-Path -LiteralPath $Marker
    if ($script:disarmFailure) { throw 'watcher not answering' }
}
function Start-Sleep { param($Milliseconds,$Seconds) }
function Invoke-ProcessCheck { param($Mode) return $script:snapshot }
function Get-NetTCPConnection { param($ErrorAction) if ($script:listenerFailure) { throw 'TCP inventory denied' }; return $script:tcpRows }
function Get-NetUDPEndpoint { param($ErrorAction) if ($script:listenerFailure) { throw 'UDP inventory denied' }; return $script:udpRows }

try {
    $script:taskRows = @((New-Worker 'c500-guard service' 'service.py'), (New-Worker 'c500-guard hotkey' 'hotkey_win.py'))
    $script:events=@(); $script:cancelOnEnable=$false; $script:listenerFailure=$false
    $script:disarmFailure=$false; $script:disarmSawMarker=$false
    $script:tcpRows=@(); $script:udpRows=@()
    $script:snapshot=[pscustomobject]@{owned=@();unverified=@();stopped=@();errors=@()}
    Reset-Receipt
    Assert (-not (Get-StopVerification)) 'No marker is not verified persistent OFF.'
    Write-OffRequest
    Assert (Get-StopVerification) 'Clean stopped workers should verify OFF.'

    $script:taskRows[0].Settings.Enabled=$true
    Assert (-not (Get-StopVerification)) 'Enabled tasks must prevent OFF success.'
    $script:taskRows[0].Settings.Enabled=$false
    $script:taskRows[0].State='Queued'
    Assert (-not (Get-StopVerification)) 'Queued tasks must prevent OFF success.'
    $script:taskRows[0].State='Disabled'
    $script:snapshot.unverified=@([pscustomobject]@{pid=99;error='access denied'})
    Assert (-not (Get-StopVerification)) 'Unreadable candidates must prevent OFF success.'
    $script:snapshot.unverified=@()
    $script:snapshot.owned=@([pscustomobject]@{pid=99;component='input helper'})
    Assert (-not (Get-StopVerification)) 'Surviving hooks must prevent OFF success.'
    $script:snapshot.owned=@()
    $script:udpRows=@([pscustomobject]@{LocalPort=32108;OwningProcess=99})
    Assert (-not (Get-StopVerification)) 'UDP relay must prevent OFF success.'
    $script:udpRows=@()
    $script:tcpRows=@([pscustomobject]@{State='Listen';LocalPort=32108;OwningProcess=99})
    Assert (-not (Get-StopVerification)) 'TCP relay must prevent OFF success.'
    $script:tcpRows=@([pscustomobject]@{State='TimeWait';LocalPort=8787;OwningProcess=99})
    Assert (Get-StopVerification) 'TIME_WAIT must not block shutdown.'
    $script:tcpRows=@()
    $script:listenerFailure=$true
    Assert-Throws { Get-StopVerification } 'Listener lookup failure must not look empty.'
    $script:listenerFailure=$false

    $initial = Get-OffRequest
    $RequestId = [Guid]::NewGuid().ToString()
    Write-OffRequest
    Assert-Throws { Confirm-OnRequest $initial } 'New OFF must cancel queued ON.'
    $ExpectedOffGeneration = $initial
    Remove-Item -LiteralPath $Marker
    Assert-Throws { Assert-OnAllowed } 'Erasing a raced marker must not erase OFF generation.'
    $ExpectedOffGeneration = Get-OffRequest
    Assert-OnAllowed
    $script:taskRows[0].Settings.RestartCount=999
    Assert-Throws { Start-Workers } 'Retry-enabled tasks must not start.'
    Assert ($script:events.Count -eq 0) 'Validate every worker before enabling any.'
    $script:taskRows[0].Settings.RestartCount=0
    $script:taskRows[1].Triggers=@([pscustomobject]@{Enabled=$true})
    Assert-Throws { Start-Workers } 'Logon-triggered workers must not start.'
    $script:taskRows[1].Triggers=@()
    $script:cancelOnEnable=$true
    Assert-Throws { Start-Workers } 'OFF during enable must cancel actual process start.'
    Assert (-not ($script:events -contains 'start')) 'A cancelled startup must not start workers.'
    $script:cancelOnEnable=$false
    $script:events=@()
    Reset-Receipt
    $Receipt.errors += 'Startup health failed'
    Stop-FullStack
    Assert ($script:events -join ',' -eq 'disarm,disable,disable,stop,stop') 'Disarm before forced termination, and disable all workers before stopping any.'
    Assert $script:disarmSawMarker 'Publish the stop marker before requesting volume restoration.'
    Assert ($Receipt.checks['no_owned_processes'] -and $Receipt.checks['ownership_verified']) 'Rollback must verify cleanup.'
    Assert (-not $Receipt.ok) 'Successful rollback must not hide the original startup failure.'
    $Receipt.errors=@(); Stop-FullStack
    Assert $Receipt.ok 'Clean OFF must succeed.'
    $script:events=@(); $script:disarmFailure=$true; $script:disarmSawMarker=$false
    Remove-Item -LiteralPath $Marker
    Stop-FullStack
    Assert $script:disarmSawMarker 'Rollback without an existing marker must publish it before disarm.'
    Assert ($script:events -join ',' -eq 'disarm,disable,disable,stop,stop') 'A failed graceful disarm must still force-stop every worker.'
    Assert $Receipt.ok 'An unreachable disarm endpoint must not prevent verified full OFF.'
    $script:disarmFailure=$false

    $healthy = [pscustomobject]@{armed=$true;frames=4;last_frame_age=0.2;loop_age_s=0.1;
        ml=[pscustomobject]@{available=$true;state='ready';age_s=0.2;inferences=2};
        input_helper=[pscustomobject]@{connected=$true};alignment=[pscustomobject]@{state='tracking'}}
    Assert (Test-StartupHealth $healthy) 'Healthy full stack should pass startup.'
    Assert (Test-StartupHealth $healthy -RequireArmed) 'Fresh inference while armed should pass post-arm startup.'
    $healthy.armed=$false
    Assert (Test-StartupHealth $healthy) 'Pre-arm health must allow paused alerts.'
    Assert (-not (Test-StartupHealth $healthy -RequireArmed)) 'Post-arm success requires alerts still armed.'
    $healthy.armed=$true
    $healthy.ml.state='waiting_for_frames'
    Assert (-not (Test-StartupHealth $healthy)) 'Available model awaiting inference must not pass startup.'
    Assert (-not (Test-StartupHealth $healthy -RequireArmed)) 'An arming epoch reset must not pass post-arm startup.'
    $healthy.ml.state='ready'; $healthy.ml.inferences=0
    Assert (-not (Test-StartupHealth $healthy)) 'At least one actual inference is required.'
    $healthy.ml.inferences=2
    foreach ($badAge in @($null,-0.1,3,12,[double]::NaN,[double]::PositiveInfinity,[double]::NegativeInfinity,'0.2',$true)) {
        $healthy.ml.age_s=$badAge
        Assert (-not (Test-StartupHealth $healthy)) 'ML age must be numeric, finite, nonnegative and less than 3 seconds.'
    }
    $healthy.ml.age_s=0.2
    $healthy.ml.available=$false
    Assert (-not (Test-StartupHealth $healthy)) 'Answering API without AI is not healthy.'
    $healthy.ml.available=$true; $healthy.loop_age_s=30
    Assert (-not (Test-StartupHealth $healthy)) 'Answering API with stalled detector is not healthy.'
    Assert (-not (Test-StartupHealth ([pscustomobject]@{frames=4}))) 'Incomplete API shape must fail closed.'
    $script:taskRows[0].Actions[0].Arguments='"C:\Other\service.py"'
    Assert-Throws { Get-OwnedTasks } 'Task-name-only ownership must be rejected.'
    $script:taskRows[0]=New-Worker 'c500-guard service' 'service.py'
    Write-Receipt
    $first = [IO.File]::ReadAllText($ReceiptPath) | ConvertFrom-Json
    Assert $first.complete 'Receipt must contain confirmed completion.'
    $Receipt.ok=$false; Write-Receipt
    $second = [IO.File]::ReadAllText($ReceiptPath) | ConvertFrom-Json
    Assert (-not $second.complete) 'Atomic replacement must update the receipt.'
    Write-Output "$script:checks mocked controller checks passed (Windows PowerShell $($PSVersionTable.PSVersion))."
} finally {
    # The target is a fixed freshly-created test directory under the actual temp root.
    $resolved = [IO.Path]::GetFullPath($State)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if ($resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        [IO.Path]::GetFileName($resolved).StartsWith('c500-power-test-')) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
