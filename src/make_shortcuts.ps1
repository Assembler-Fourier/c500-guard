<# BarrierWatcher app shortcuts: desktop and Start menu. Needs no administrator rights.
   Creates (or refreshes) only BarrierWatcher's own shortcuts and removes the old "C500 ON/OFF" ones. #>
[CmdletBinding()]
param([string]$Root = '')
$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$Root = [IO.Path]::GetFullPath($Root)
$src = Join-Path $Root 'src'
$assets = Join-Path $Root 'assets'
$launcher = Join-Path $src 'barrierwatcher_app.pyw'
# The app runs on the PC's own windowless Python (standard library only), like the scheduled tasks.
$venvPython = Join-Path $Root '.venv\Scripts\python.exe'
$basePython = (& $venvPython -c 'import sys; print(sys._base_executable)' | Out-String).Trim()
$pythonw = Join-Path (Split-Path -Parent $basePython) 'pythonw.exe'
foreach ($required in @($launcher, $pythonw, (Join-Path $assets 'barrierwatcher.ico'), (Join-Path $assets 'barrierwatcher-off.ico'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing $required" }
}
$desktop = [Environment]::GetFolderPath('Desktop')
$menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'BarrierWatcher'
New-Item -ItemType Directory -Path $menu -Force | Out-Null
$open = 'Open BarrierWatcher: starts the alarm system; closing the window switches everything off.'
$off = 'Switch the whole BarrierWatcher program off: camera, AI, alarms, helpers and the app window.'
$items = @(
    @{ Folder = $desktop; Name = 'BarrierWatcher';          Mode = 'open';     Icon = 'barrierwatcher.ico';     Text = $open },
    @{ Folder = $desktop; Name = 'BarrierWatcher OFF';      Mode = 'off';      Icon = 'barrierwatcher-off.ico'; Text = $off },
    @{ Folder = $menu;    Name = 'BarrierWatcher';          Mode = 'open';     Icon = 'barrierwatcher.ico';     Text = $open },
    @{ Folder = $menu;    Name = 'BarrierWatcher OFF';      Mode = 'off';      Icon = 'barrierwatcher-off.ico'; Text = $off },
    @{ Folder = $menu;    Name = 'BarrierWatcher Settings'; Mode = 'settings'; Icon = 'barrierwatcher.ico';     Text = 'Open BarrierWatcher settings (starts it first if needed).' }
)
$shell = New-Object -ComObject WScript.Shell
foreach ($item in $items) {
    $shortcut = $shell.CreateShortcut((Join-Path $item.Folder ($item.Name + '.lnk')))
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = '"' + $launcher + '" ' + $item.Mode
    $shortcut.WorkingDirectory = $Root
    $shortcut.WindowStyle = 1
    $shortcut.Description = $item.Text
    $shortcut.IconLocation = (Join-Path $assets $item.Icon) + ',0'
    $shortcut.Save()
}
# The previous name's shortcuts and Start menu folder (ours only).
foreach ($old in @('Gatewatch.lnk', 'Gatewatch OFF.lnk')) {
    $path = Join-Path $desktop $old
    if ((Test-Path -LiteralPath $path) -and $shell.CreateShortcut($path).Arguments -match 'gatewatch\.vbs') { Remove-Item -LiteralPath $path -Force }
}
$oldMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Gatewatch'
if (Test-Path -LiteralPath $oldMenu) {
    $foreign = @(Get-ChildItem -LiteralPath $oldMenu | Where-Object { $_.Extension -ne '.lnk' -or $shell.CreateShortcut($_.FullName).Arguments -notmatch 'gatewatch\.vbs' })
    if ($foreign.Count -eq 0) { Remove-Item -LiteralPath $oldMenu -Recurse -Force }
}
foreach ($old in @('C500 ON.lnk', 'C500 OFF.lnk')) {
    $path = Join-Path $desktop $old
    if (Test-Path -LiteralPath $path) {
        $target = $shell.CreateShortcut($path).Arguments
        if ($target -match 'power_shortcut\.vbs') { Remove-Item -LiteralPath $path -Force }   # only our own old ones
    }
}
Write-Output "BarrierWatcher shortcuts ready: desktop (BarrierWatcher, BarrierWatcher OFF) and Start menu folder '$menu'."
