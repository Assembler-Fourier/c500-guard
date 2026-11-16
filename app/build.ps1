[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$SdkDirectory,
      [string]$Compiler = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
$ErrorActionPreference = 'Stop'
$sdk = (Resolve-Path -LiteralPath $SdkDirectory).Path
if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) { throw 'C# compiler not found. Supply -Compiler.' }
$core = Join-Path $sdk 'Microsoft.Web.WebView2.Core.dll'
$forms = Join-Path $sdk 'Microsoft.Web.WebView2.WinForms.dll'
$loader = Join-Path $sdk 'WebView2Loader.dll'
foreach ($file in @($core,$forms,$loader)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required SDK file missing: $file" }
}
$exe = Join-Path $PSScriptRoot 'BarrierWatcher.exe'
$source = Join-Path $PSScriptRoot 'BarrierWatcherWindow.cs'
$manifest = Join-Path $PSScriptRoot 'app.manifest'
$icon = Join-Path (Split-Path -Parent $PSScriptRoot) 'assets\barrierwatcher.ico'
& $Compiler /nologo /target:winexe /platform:x64 /optimize+ "/out:$exe" "/win32manifest:$manifest" "/win32icon:$icon" "/reference:$core" "/reference:$forms" /reference:System.Windows.Forms.dll /reference:System.Drawing.dll $source
if ($LASTEXITCODE -ne 0) { throw 'Native window compilation failed.' }
foreach ($file in @($core,$forms,$loader)) {
    $destination = Join-Path $PSScriptRoot (Split-Path -Leaf $file)
    if ([IO.Path]::GetFullPath($file) -ne [IO.Path]::GetFullPath($destination)) {
        Copy-Item -LiteralPath $file -Destination $destination -Force
    }
}
Write-Output "Built $exe. No application was started."
