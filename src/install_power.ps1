<# Register explicit, windowless power controls. No logon or timed triggers. #>
[CmdletBinding()]
param([string]$Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
$src = Join-Path $Root 'src'
$venvPython = Join-Path $Root '.venv\Scripts\python.exe'
$basePython = (& $venvPython -c 'import sys; print(sys._base_executable)' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $basePython)) { throw 'Cannot resolve task Python.' }
$taskPython = Join-Path (Split-Path -Parent $basePython) 'pythonw.exe'
if (!(Test-Path -LiteralPath $taskPython)) { throw 'Windowless Python is missing.' }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Highest
$runtimeSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$powerSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
$actions = @{
    'c500-guard service' = (New-ScheduledTaskAction -Execute $taskPython -Argument ('"' + (Join-Path $src 'service.py') + '"') -WorkingDirectory $src)
    'c500-guard hotkey' = (New-ScheduledTaskAction -Execute $taskPython -Argument ('"' + (Join-Path $src 'hotkey_win.py') + '" --log "' + (Join-Path $env:LOCALAPPDATA 'c500-guard\hotkey.log') + '"') -WorkingDirectory $src)
}
foreach ($name in $actions.Keys) {
    Register-ScheduledTask -TaskName $name -Action $actions[$name] -Principal $principal -Settings $runtimeSettings -Force | Out-Null
    Disable-ScheduledTask -TaskName $name | Out-Null
}
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach ($action in @('On', 'Off')) {
    $argument = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $src 'power_win.ps1') + '" -Action ' + $action
    $taskAction = New-ScheduledTaskAction -Execute $powershell -Argument $argument -WorkingDirectory $src
    Register-ScheduledTask -TaskName ('C500 Guard ' + $action.ToUpperInvariant()) -Action $taskAction -Principal $principal -Settings $powerSettings -Force | Out-Null
}
$old = Get-ScheduledTask -TaskName 'c500-guard watcher' -ErrorAction SilentlyContinue
if ($old) {
    Disable-ScheduledTask -TaskName $old.TaskName | Out-Null
    Stop-ScheduledTask -TaskName $old.TaskName
    Unregister-ScheduledTask -TaskName $old.TaskName -Confirm:$false
}
& (Join-Path $src 'make_shortcuts.ps1') -Root $Root
Write-Output 'Explicit BarrierWatcher ON/OFF tasks and the BarrierWatcher app shortcuts installed. No task has a logon trigger or scheduled retry.'
