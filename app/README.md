# Optional native Windows window

`BarrierWatcherWindow.cs` is the application's small Windows Forms/WebView2 host. It displays the local dashboard; the Python launcher retains ownership of starting and stopping the monitored program. No compiled executable or Microsoft DLL is distributed in this source package.

To build on Windows, install a compatible .NET Framework C# compiler and obtain the Microsoft WebView2 SDK separately. Supply an SDK directory containing `Microsoft.Web.WebView2.Core.dll`, `Microsoft.Web.WebView2.WinForms.dll` and the matching x64 `WebView2Loader.dll`:

```powershell
.\app\build.ps1 -SdkDirectory C:\path\to\webview2-runtime-files
```

The build script writes ignored binaries under `app/`. It performs no download and does not start the application. The WebView2 Runtime must also be installed. This build path is provided from the source/compiler contract; it has not been rebuilt as part of the public-package test run. The launcher can use a browser window when the native executable is absent.



## Generated icons

The public source release omits binary icon files. Before building, run `python assets/make_icons.py` from the repository root (Pillow is required). This regenerates the application icons from the included drawing code without camera imagery.
