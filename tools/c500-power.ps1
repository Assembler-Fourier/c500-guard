# Compatible entry point for the existing C500 Guard ON/OFF scheduled tasks.
[CmdletBinding()]
param([ValidateSet('Off','On','Status')][string]$Action = 'Status', [switch]$Pause)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
& (Join-Path $root 'src\power_win.ps1') -Action $Action -Root $root
exit $LASTEXITCODE
