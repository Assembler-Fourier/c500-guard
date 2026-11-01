# Windows setup

## 0. Before you run anything

Sanity-check the installer without executing it (optional but cheap):

```powershell
$e=$null
[System.Management.Automation.Language.Parser]::ParseFile("$PWD\install.ps1",[ref]$null,[ref]$e)
$e   # prints nothing if the script is syntactically clean
```

## 1. Install

Keep the application in its permanent location, such as `C:\c500-guard`.
Scheduled tasks and the virtual environment refer to that location. Do not
move a populated `.venv` between PCs or remove its base Python installation.

The tested Windows bundle includes:

- Python 3.13 (3.13.15 was used for this deployment).
- `tools\ffmpeg\bin\ffmpeg.exe`; this deployment uses FFmpeg 9.0.1.
- `tools\go2rtc\go2rtc.exe` and `go2rtc-build.json`, kept together. This is
  the local v1.9.14 build with the Xiaomi command-queue fix documented in
  `patches\LOCAL-PATCH-PROVENANCE.md`.
- `requirements-ml.txt`, the model installer, and the YOLOX license/provenance
  in `models\`. Missing model weights are downloaded and verified at setup.
- OpenCV 4.14.0.94 provides automatic zone alignment. Keep the private
  `alignment-*.npz` reference with the camera state; see `CURRENT-SETUP.md`.

**PowerShell as Administrator under the account that will operate the camera**,
in that application folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\install.ps1
```

The installer prefers the tested Python 3.13, creates or repairs `.venv`,
installs ONNX Runtime and runs a real CPU inference self-test. It uses bundled
FFmpeg before checking PATH. Missing Python/FFmpeg can be installed through
winget; Python packages and a missing model need internet access.

The installer verifies the local bridge bundle against its accompanying
manifest and preserves an already-installed matching build. A missing bundle
is acceptable only when the private state folder already contains the verified
local executable and manifest. To supply a bundle stored elsewhere, use
`.\install.ps1 -BridgeBundle 'D:\camera-tools\go2rtc'`. A fresh setup does not
silently substitute the unpatched upstream binary for this camera.

Registration does not start the scheduled tasks immediately. Follow step 4
to start them in the current session, or sign out and back in.

Elevation is required for two reasons: registering the autostart tasks, and the
global keyboard hook. A hook installed by a non-elevated process cannot see
keystrokes while an elevated window has focus — so a non-elevated install gives
you a hotkey that works *most* of the time, which is worse than one that always
works.

## 2. Connect the camera (once)

```powershell
.\.venv\Scripts\python.exe src\bridge.py start
.\.venv\Scripts\python.exe src\bridge.py login-details
```

Open the printed URL, scroll to **Xiaomi**, and sign in.

**Two things that will waste your time if you get them wrong:**

- **Username must be your numeric Mi Account ID**, e.g. `1234567890` — not your
  email. An email that is only a *recovery* address is rejected with
  `code 70016 / 登录验证失败`, which looks like a wrong password but is not.
  Find the ID at <https://account.xiaomi.com> — it is in the page URL as `userId=`.
- **Region must match your Xiaomi Home app.** The wrong region returns an empty
  device list even after a successful login. Try the region you use in the app
  first; the C500 in this deployment is on **Europe (`de`)**.

Then pick the camera, press **add**, and name it `c500`.

For this deployment, the LAN relay handles both UDP discovery and the camera's
negotiated TCP video connection. The relay is enabled by the private
`%LOCALAPPDATA%\c500-guard\camera.json` record. The supervisor starts it through
`bridge.py`; do not create a separate competing relay or video reader. On a
fresh configuration, use `bridge.py add-stream --name c500 --src <camera URL>
--lan-shim` with the camera URL supplied by the local bridge. Treat that URL as
private. Keep the physical camera IP in `camera.json`; the saved bridge stream
uses the loopback relay. If DHCP changes the camera IP, update the relay's
private camera record and restart the bridge during maintenance.

Verify:

```powershell
.\.venv\Scripts\python.exe src\bridge.py status
```

You want `"streams": ["c500"]`.

## 3. Draw your zones

Start the watcher as shown below, leave it **disarmed**, and open
[Adjust zones](http://127.0.0.1:8787/zones). Select a zone and drag its rectangle
over the live image. You can edit labels, motion/light thresholds, and normalized
`[x, y, width, height]` coordinates. Saving validates the settings, retains an
exact backup beside `config\zones.json`, and resets detection history. The
editor reuses the watcher's existing camera session.

**Exclude anything that changes by itself** — an on-screen clock, a second
monitor in shot, the edge of the picture. A ticking timestamp inside a zone will
fire the siren every single second.

## 4. Run

```powershell
.\.venv\Scripts\python.exe src\ctl.py start
Start-ScheduledTask -TaskName 'c500-guard service'
Start-ScheduledTask -TaskName 'c500-guard hotkey'
```

Open <http://127.0.0.1:8787>.

The tasks `c500-guard service` and `c500-guard hotkey` run at this user's
**sign-in**, in the interactive desktop with highest privileges. The service
checks/restarts the bridge, optional LAN relay, and watcher. The obsolete
`c500-guard watcher` task is removed. Signing in is required after a reboot;
these tasks do not provide a siren before login.

New state starts disarmed. Existing saved ON/OFF state is preserved. Confirm
live video and Local AI status, test the speaker, and arm explicitly. `ctl.py
stop` stops the watcher and creates `service-paused`; automatic watcher restarts
remain paused until `ctl.py start` clears that marker. `ctl.py off` only disarms
and leaves the live preview running.

## The on/off key

**Ctrl+Alt+F4** toggles the whole system, from any window.

The keystroke is *swallowed* by the hook when it matches, so it can never reach
the foreground application as `Alt+F4` and close a window.

- Turning **ON** plays a short two-note rising confirmation.
- Turning **OFF** is **silent by default**, because while the system is off it
  must make no sound at all. If you would rather have confirmation both ways,
  add `--beep-both` to the hotkey task's arguments.

While off: no siren, no chirp, no fault tone, no notification, and nothing
written to the event log except the single "disarmed" line. It keeps decoding
video so it is instantly ready, but raises nothing.

The off state is stored in `%LOCALAPPDATA%\c500-guard\armed.json` and **survives
reboots** — if someone switches it off on Friday it is still off on Monday.
`ctl.py doctor` will say so.

## Troubleshooting

```powershell
.\.venv\Scripts\python.exe src\ctl.py doctor
```

| Symptom | Cause |
|---|---|
| `no usable frame ... since startup` | camera not on this LAN, or the bridge is down |
| Xiaomi login fails, `70016` | using an email instead of the numeric Mi Account ID |
| Login works, no devices | wrong region, or the camera is on a different Xiaomi account |
| Siren fires constantly | a zone overlaps a clock, a screen edge, or foliage |
| Hotkey does nothing | hotkey task not running, or it is not elevated |
| No sound | check Windows volume and output device; run `ctl.py test-alarm` |
| Python executable exists but will not run | its original Python was removed; reinstall Python and rerun the installer to rebuild the broken `.venv` |
| AI unavailable | use the `.venv` Python commands in `LOCAL-ML.md`; inspect the dashboard error |
| Service runs but watcher stays stopped | `ctl.py stop` may have paused it; resume with `ctl.py start` |
| Stream fails after an upstream go2rtc replacement | restore the tested executable and matching `go2rtc-build.json`; keep the UDP/TCP relay source |

Logs, in `%LOCALAPPDATA%\c500-guard\`:

| File | Contents |
|---|---|
| `watcher.log` | watcher console output and any crash |
| `hotkey.log` | every toggle, and hook failures |
| `bridge.log` | go2rtc, including camera connection errors |
| `service.log` | supervisor startup, retries and intentional pause |
| `lan-shim.log` | LAN discovery and TCP relay connection errors |
| `events.jsonl` | every alarm, fault, acknowledgement, arm and disarm |

## Removing autostart

```powershell
.\install.ps1 -Uninstall
```

This removes task registration; it does not delete the application or private
camera state. To stop current processes first, use the maintenance commands.

## Updating or restoring this installation

During an agreed maintenance period, disarm and stop the running helpers before
replacing Python, loaded packages, bridge executables, or application files:

```powershell
.\.venv\Scripts\python.exe src\ctl.py off
.\.venv\Scripts\python.exe src\ctl.py stop
Stop-ScheduledTask -TaskName 'c500-guard service'
Stop-ScheduledTask -TaskName 'c500-guard hotkey'
.\.venv\Scripts\python.exe src\bridge.py stop
```

Preserve `config\zones.json`, `models\`, `tools\`, and the complete private
`%LOCALAPPDATA%\c500-guard` folder in a secure backup. The private folder includes
camera credentials, bridge settings, camera IP, armed state and the active
bridge executable/manifest; do not publish it. A source-only checkout excludes
several binary assets and cannot reproduce this deployment without restoring
the bundle or fetching the documented model/dependencies.

Rerun `install.ps1` and step 4 after restoring files. A changed running bridge
is never replaced by the installer; stop it first. Verify `ctl.py doctor`, Local
AI, the live picture, speaker and hotkey. A complete unattended fresh install
has not been tested; this procedure documents the checked components and the
required manual Xiaomi login and scene calibration.
