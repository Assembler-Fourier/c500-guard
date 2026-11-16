#!/usr/bin/env python3
"""Per-OS behaviour for c500-guard, in one place.

Everything the watcher needs that differs between Windows, macOS and Linux
lives here: looping audio, keeping the machine awake, reading a single
keypress, capture-device syntax for ffmpeg, and where private state goes.
The rest of the program is platform-neutral.
"""

from __future__ import annotations

import os
import ntpath
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WINDOWS = sys.platform.startswith("win")
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")

OS_NAME = "windows" if WINDOWS else "macos" if MACOS else "linux" if LINUX else sys.platform

# Bundled app tools also work at logon before Windows refreshes the user's PATH.
_tool_bin = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin"
if WINDOWS and (_tool_bin / "ffmpeg.exe").is_file():
    os.environ["PATH"] = str(_tool_bin) + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Where private state lives (never inside the repo, never world-readable)
# ---------------------------------------------------------------------------

def state_dir() -> Path:
    if WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        path = base / "c500-guard"
    elif MACOS:
        path = Path.home() / "Library" / "Application Support" / "c500-guard"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        path = base / "c500-guard"
    path.mkdir(parents=True, exist_ok=True)
    harden(path)
    return path


def harden(path: Path) -> None:
    """Owner-only permissions. On Windows, ACLs are inherited from LOCALAPPDATA."""
    if WINDOWS:
        return
    try:
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Looping alarm audio
# ---------------------------------------------------------------------------

class AudioPlayer:
    """Loops a WAV until stopped. Never times out on its own."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._winsound = None
        self._lock = threading.Lock()
        if WINDOWS:
            import winsound            # stdlib on Windows
            self._winsound = winsound

    @staticmethod
    def available() -> tuple[bool, str]:
        """Report whether this machine can actually make a noise."""
        if WINDOWS:
            try:
                import winsound  # noqa: F401
                return True, "winsound (built in)"
            except ImportError:
                return False, "winsound missing"
        if MACOS:
            return (bool(shutil.which("afplay")), "afplay"
                    if shutil.which("afplay") else "afplay not found")
        for player in ("paplay", "aplay", "ffplay"):
            if shutil.which(player):
                return True, player
        return False, "no paplay/aplay/ffplay found - install alsa-utils or pulseaudio"

    def start_loop(self, path: Path) -> None:
        """Begin looping `path`. Replaces whatever was playing."""
        with self._lock:
            self._stop_locked()
            if self._winsound:
                # SND_LOOP loops natively and seamlessly; no respawn needed.
                self._winsound.PlaySound(
                    str(path),
                    self._winsound.SND_FILENAME | self._winsound.SND_ASYNC
                    | self._winsound.SND_LOOP)
                return
            self._proc = _spawn_repeating(path)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._winsound:
            self._winsound.PlaySound(None, self._winsound.SND_PURGE)
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


def _player_command(path: Path) -> list[str] | None:
    if MACOS and shutil.which("afplay"):
        return ["afplay", str(path)]
    for player, args in (("paplay", []), ("aplay", ["-q"]),
                         ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"])):
        if shutil.which(player):
            return [player, *args, str(path)]
    return None


def _spawn_repeating(path: Path) -> subprocess.Popen | None:
    """POSIX players do not loop, so re-run one in a helper shell loop."""
    cmd = _player_command(path)
    if not cmd:
        return None
    quoted = " ".join(f"'{c}'" for c in cmd)
    return subprocess.Popen(["/bin/sh", "-c", f"while :; do {quoted}; done"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


# ---------------------------------------------------------------------------
# Keep the machine awake while armed
# ---------------------------------------------------------------------------

class SleepInhibitor:
    """Best effort. Reports honestly whether it actually took hold."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self.active = False
        self.detail = "not started"

    def __enter__(self):
        if WINDOWS:
            try:
                import ctypes
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                flags = 0x80000000 | 0x00000001 | 0x00000002
                if ctypes.windll.kernel32.SetThreadExecutionState(flags):
                    self.active, self.detail = True, "SetThreadExecutionState"
                else:
                    self.detail = "SetThreadExecutionState refused"
            except Exception as exc:                     # noqa: BLE001
                self.detail = f"unavailable: {exc}"
        elif MACOS and shutil.which("caffeinate"):
            self._proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.active, self.detail = True, "caffeinate"
        elif LINUX and shutil.which("systemd-inhibit"):
            self.active, self.detail = True, "systemd-inhibit (wrap the service)"
        else:
            self.detail = "no sleep inhibitor available on this system"
        return self

    def __exit__(self, *_exc) -> None:
        if WINDOWS and self.active:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:                            # noqa: BLE001
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self.active = False
        self.detail = "not active"


# ---------------------------------------------------------------------------
# Speaker health: an alarm on a muted or silent output is no alarm at all
# ---------------------------------------------------------------------------

def audio_output_status() -> dict:
    """Default speaker name, volume (0..1) and mute state. Read-only: never changes them.

    Uses Windows Core Audio through ctypes (IMMDeviceEnumerator ->
    IMMDevice -> IAudioEndpointVolume), so no extra package is needed.
    """
    if not WINDOWS:
        return {"available": False, "detail": "Speaker level check is Windows-only"}
    import ctypes
    from ctypes import wintypes as wt

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", wt.DWORD)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                    ("r3", ctypes.c_ushort), ("value", ctypes.c_void_p), ("pad", ctypes.c_void_p)]

    ole32 = ctypes.OleDLL("ole32")

    def guid(text):
        value = GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(value))
        return value

    def method(obj, index, *argtypes):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])

    def release(obj):
        if obj:
            vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
            ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(obj)

    initialized = False
    try:
        ole32.CoInitializeEx(None, 0)
        initialized = True
    except OSError:
        pass   # Already initialized differently on this thread; still usable.
    enumerator, device = ctypes.c_void_p(), ctypes.c_void_p()
    volume, store = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        ole32.CoCreateInstance(ctypes.byref(guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")), None, 0x17,
                               ctypes.byref(guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),
                               ctypes.byref(enumerator))
        # GetDefaultAudioEndpoint(eRender, eMultimedia): where PlaySound goes.
        method(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
            enumerator, 0, 1, ctypes.byref(device))
        method(device, 3, ctypes.POINTER(GUID), wt.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, ctypes.byref(guid("{5CDF2C82-841E-4546-9722-0CF74078229A}")), 0x17, None,
            ctypes.byref(volume))
        level, muted = ctypes.c_float(), wt.BOOL()
        method(volume, 9, ctypes.POINTER(ctypes.c_float))(volume, ctypes.byref(level))   # GetMasterVolumeLevelScalar
        method(volume, 15, ctypes.POINTER(wt.BOOL))(volume, ctypes.byref(muted))         # GetMute
        name = ""
        try:
            method(device, 4, wt.DWORD, ctypes.POINTER(ctypes.c_void_p))(device, 0, ctypes.byref(store))
            key = PROPERTYKEY(guid("{A45C254E-DF1C-4EFD-8020-67D146A850E0}"), 14)   # Device_FriendlyName
            value = PROPVARIANT()
            method(store, 5, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT))(
                store, ctypes.byref(key), ctypes.byref(value))
            if value.vt == 31 and value.value:   # VT_LPWSTR
                name = ctypes.wstring_at(value.value)[:120]
            ole32.PropVariantClear(ctypes.byref(value))
        except OSError:
            pass
        return {"available": True, "muted": bool(muted.value),
                "volume": round(min(1.0, max(0.0, level.value)), 3), "device": name}
    except OSError as exc:
        return {"available": False, "detail": f"No default speaker found ({exc})"}
    finally:
        for obj in (store, volume, device, enumerator):
            release(obj)
        if initialized:
            ole32.CoUninitialize()


def set_audio_output(volume: float | None = None, muted: bool | None = None) -> dict:
    """Set the default speaker's level (0..1) and mute state. Windows only.

    Deliberately separate from `audio_output_status`, which is guaranteed never
    to change anything: a read that could write would be a trap. The alarm is
    the only caller, it saves the previous state first, and it puts it back when
    the alarm ends - a guard's music should come back the way they left it.
    """
    if not WINDOWS:
        return {"applied": False, "detail": "Speaker control is Windows-only"}
    if volume is None and muted is None:
        return {"applied": False, "detail": "nothing to change"}
    import ctypes
    from ctypes import wintypes as wt

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    ole32 = ctypes.OleDLL("ole32")

    def guid(text):
        value = GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(value))
        return value

    def method(obj, index, *argtypes):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])

    def release(obj):
        if obj:
            vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
            ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(obj)

    initialized = False
    try:
        ole32.CoInitializeEx(None, 0)
        initialized = True
    except OSError:
        pass
    enumerator, device, endpoint = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        ole32.CoCreateInstance(ctypes.byref(guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")), None, 0x17,
                               ctypes.byref(guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),
                               ctypes.byref(enumerator))
        method(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
            enumerator, 0, 1, ctypes.byref(device))
        method(device, 3, ctypes.POINTER(GUID), wt.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, ctypes.byref(guid("{5CDF2C82-841E-4546-9722-0CF74078229A}")), 0x17, None,
            ctypes.byref(endpoint))
        applied = {}
        if volume is not None:
            level = min(1.0, max(0.0, float(volume)))
            # SetMasterVolumeLevelScalar(float, LPCGUID)
            method(endpoint, 7, ctypes.c_float, ctypes.c_void_p)(endpoint, ctypes.c_float(level), None)
            applied["volume"] = round(level, 3)
        if muted is not None:
            # SetMute(BOOL, LPCGUID)
            method(endpoint, 14, wt.BOOL, ctypes.c_void_p)(endpoint, wt.BOOL(bool(muted)), None)
            applied["muted"] = bool(muted)
        return {"applied": True, **applied}
    except OSError as exc:
        return {"applied": False, "detail": f"Speaker refused the change ({exc})"}
    finally:
        for obj in (endpoint, device, enumerator):
            release(obj)
        if initialized:
            ole32.CoUninitialize()


def rotate_log(path: Path, max_bytes: int = 5_000_000) -> None:
    """Keep one previous generation. Call just before opening a log to append."""
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass   # Missing, or still open elsewhere on Windows: keep appending.


# ---------------------------------------------------------------------------
# Single-keypress input (any key acknowledges)
# ---------------------------------------------------------------------------

class KeyReader:
    """Yields one character per keypress. No Enter needed. Restores the tty."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._fd = None
        self._saved = None
        self.usable = False
        self.reason = ""

    def __enter__(self):
        try:
            if not self.stream.isatty():
                self.reason = "not a terminal - acknowledge from the web page"
                return self
        except (AttributeError, ValueError):
            self.reason = "no usable stdin"
            return self
        if WINDOWS:
            try:
                import msvcrt  # noqa: F401
                self.usable = True
            except ImportError:
                self.reason = "msvcrt unavailable"
            return self
        try:
            import termios, tty
            self._fd = self.stream.fileno()
            self._saved = termios.tcgetattr(self._fd)
            # cbreak, not raw: leaves ISIG on so Ctrl-C still interrupts.
            tty.setcbreak(self._fd)
            self.usable = True
        except Exception as exc:                          # noqa: BLE001
            self.reason = f"raw mode unavailable: {exc}"
        return self

    def read_key(self) -> str:
        """Blocks until a key is pressed. Returns '' when input ends."""
        if not self.usable:
            time.sleep(0.2)
            return ""
        if WINDOWS:
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
                time.sleep(0.03)
        return self.stream.read(1)

    def __exit__(self, *_exc) -> None:
        self.restore()

    def restore(self) -> None:
        """Put the terminal back however we exit - never leave a broken shell."""
        fd, saved = self._fd, self._saved
        self._fd = self._saved = None
        if fd is not None and saved is not None:
            try:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:                             # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# ffmpeg capture syntax
# ---------------------------------------------------------------------------

def screen_input(spec: str, fps: float) -> list[str]:
    """spec: '' for the whole desktop, or 'x,y,w,h', or a macOS device index."""
    if WINDOWS:
        args = ["-f", "gdigrab", "-framerate", str(fps), "-draw_mouse", "0"]
        if spec and "," in spec:
            x, y, w, h = (int(v) for v in spec.split(","))
            args += ["-offset_x", str(x), "-offset_y", str(y),
                     "-video_size", f"{w}x{h}"]
        return args + ["-i", "desktop"]
    if MACOS:
        index, _, _rect = spec.partition("@")
        return ["-f", "avfoundation", "-capture_cursor", "0",
                "-capture_mouse_clicks", "0", "-framerate", str(fps),
                "-i", f"{index or '4'}:"]
    return ["-f", "x11grab", "-framerate", str(fps),
            "-i", os.environ.get("DISPLAY", ":0.0")]


def camera_input(name: str, fps: float) -> list[str]:
    if WINDOWS:
        return ["-f", "dshow", "-framerate", str(fps), "-i", f"video={name}"]
    if MACOS:
        return ["-f", "avfoundation", "-framerate", "30", "-i", f"{name or '0'}:"]
    return ["-f", "v4l2", "-framerate", str(fps), "-i", name or "/dev/video0"]


_USB_ID = re.compile(r"^([0-9a-f]{4}):([0-9a-f]{4})$")
_DSHOW_NAME = re.compile(r'^\[[^\]]*\]\s+"(?P<name>[^"]+)"\s+\((?P<kind>video|audio|none)\)\s*$')
_DSHOW_ALT = re.compile(r'^\[[^\]]*\]\s+Alternative name\s+"(?P<alt>[^"]+)"\s*$')


def parse_usb_id(value) -> tuple[str, str]:
    """'345F:2109' -> ('345f', '2109'); anything else is refused."""
    match = _USB_ID.match(str(value or "").strip().lower())
    if not match:
        raise ValueError(f"USB id must look like 345f:2109, not {value!r}")
    return match.group(1), match.group(2)


def parse_dshow_devices(text: str) -> list[dict]:
    """Video devices from `ffmpeg -list_devices`, in DirectShow (and OpenCV) order."""
    devices, last = [], None
    for line in text.splitlines():
        line = line.strip()
        named = _DSHOW_NAME.match(line)
        if named:
            last = None
            if named.group("kind") == "video":
                last = {"name": named.group("name"), "path": "", "usb_id": ""}
                devices.append(last)
            continue
        alternative = _DSHOW_ALT.match(line)
        if alternative and last is not None and not last["path"]:
            last["path"] = alternative.group("alt")
            ids = re.search(r"vid_([0-9a-f]{4})&pid_([0-9a-f]{4})", last["path"].lower())
            if ids:
                last["usb_id"] = f"{ids.group(1)}:{ids.group(2)}"
    return devices


def dshow_video_devices(timeout: float = 15.0) -> list[dict]:
    """List video capture devices without opening any of them."""
    command = list_devices_command()
    if not WINDOWS or not command:
        return []
    result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                            timeout=timeout, stdin=subprocess.DEVNULL,
                            **background_subprocess_kwargs())
    return parse_dshow_devices((result.stderr or "") + "\n" + (result.stdout or ""))


def find_usb_camera(usb_id: str, devices: list[dict] | None = None) -> tuple[int, dict] | None:
    """(index, device) of the one camera with this USB id; None if absent or ambiguous."""
    wanted = ":".join(parse_usb_id(usb_id))
    devices = dshow_video_devices() if devices is None else devices
    found = [(index, device) for index, device in enumerate(devices)
             if device.get("usb_id") == wanted and device.get("path")]
    return found[0] if len(found) == 1 else None


def normal_priority(handle=None) -> dict:
    """Run at normal CPU and memory priority, without power throttling.

    Task Scheduler starts programs at below-normal priority with low memory
    priority, so Windows pages them out first when RAM is short. Measured on
    13 Nov 2026 with 87% of 7.6 GB in use: the watcher had 147 of 524 MB
    resident and froze for 4-21 s at a time. A keyboard hook in such a process
    delays every keystroke. This changes only the given process (default: this
    one) for as long as it runs; no Windows setting is touched.
    """
    result = {"priority": False, "memory": False, "throttling": False}
    if not WINDOWS:
        return result
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        target = wintypes.HANDLE(int(handle)) if handle is not None else kernel32.GetCurrentProcess()
        result["priority"] = bool(kernel32.SetPriorityClass(target, 0x00000020))       # NORMAL_PRIORITY_CLASS
        memory = wintypes.ULONG(5)                                                      # MEMORY_PRIORITY_NORMAL
        result["memory"] = bool(kernel32.SetProcessInformation(target, 0, ctypes.byref(memory),
                                                               ctypes.sizeof(memory)))   # ProcessMemoryPriority

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        state = PowerThrottling(1, 1, 0)        # control EXECUTION_SPEED, state off: never throttle
        result["throttling"] = bool(kernel32.SetProcessInformation(target, 4, ctypes.byref(state),
                                                                   ctypes.sizeof(state)))  # ProcessPowerThrottling
    except (OSError, AttributeError, ValueError, TypeError):
        pass
    return result


def list_devices_command() -> list[str] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    if WINDOWS:
        return [ffmpeg, "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    if MACOS:
        return [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    return None


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

_CHILD_JOB = None


def tie_to_this_process(process) -> bool:
    """Make Windows end this child whenever this process ends, even by a crash or a kill.

    One Job object with KILL_ON_JOB_CLOSE is held for the life of this process;
    only children handed to this function are in it. Nothing else is affected,
    and nothing is left running when the watcher stops for any reason.
    """
    global _CHILD_JOB
    handle = getattr(process, "_handle", None)
    if not WINDOWS or handle is None:
        return False
    try:
        import ctypes
        from ctypes import wintypes as wt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if _CHILD_JOB is None:
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
            kernel.CreateJobObjectW.restype = wt.HANDLE
            kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
            kernel.SetInformationJobObject.restype = wt.BOOL

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD),
                            ("SchedulingClass", wt.DWORD)]

            class Counters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Counters),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            job = kernel.CreateJobObjectW(None, None)
            if not job:
                return False
            info = Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                kernel.CloseHandle(job)
                return False
            _CHILD_JOB = job
        kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        kernel.AssignProcessToJobObject.restype = wt.BOOL
        return bool(kernel.AssignProcessToJobObject(_CHILD_JOB, int(handle)))
    except Exception:                                              # noqa: BLE001 - optional safety net
        return False


def background_subprocess_kwargs() -> dict:
    """Keep Windows console helpers from opening a window under pythonw.

    Redirecting stdout/stderr alone does not suppress console creation. Use
    CREATE_NO_WINDOW by itself: Windows ignores it with DETACHED_PROCESS.
    POSIX callers keep their existing process/session behavior.
    """
    return {"creationflags": 0x08000000} if WINDOWS else {}


def _valid_pid(pid: int) -> bool:
    return isinstance(pid, int) and not isinstance(pid, bool) and 0 < pid <= 0xffffffff


def _windows_identity_matches(image: str, argv: list[str], expected: str) -> bool:
    """Match an executable or Python script, never text inside another argument."""
    def normalized(value):
        return ntpath.normcase(ntpath.normpath(value.removeprefix('\\\\?\\')))

    if not image or not expected:
        return False
    wanted = normalized(expected)
    absolute = ntpath.isabs(expected)
    if expected.lower().endswith(('.py', '.pyw')):
        interpreter = ntpath.basename(image).lower()
        if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?w?\.exe', interpreter):
            return False
        # The car-memory updater uses isolated mode. Accept only this known
        # prefix; -c/-m text and an unrelated script's arguments are not identity.
        script_index = 2 if len(argv) > 1 and argv[1] == '-I' else 1
        if len(argv) <= script_index:
            return False
        script = normalized(argv[script_index])
        return script == wanted if absolute else ntpath.basename(script) == wanted
    actual = normalized(image)
    if absolute:
        return actual == wanted
    return ntpath.basename(actual) == wanted or ntpath.splitext(ntpath.basename(actual))[0] == wanted


class _WindowsProcess:
    """Own a process handle throughout identity checks and a guarded stop.

    Holding this handle keeps its PID from being recycled. No shell or WMI
    process is spawned for health checks. Command-line querying is read-only;
    unsupported native API layouts, cross-bitness and access denial fail closed.
    https://devblogs.microsoft.com/oldnewthing/20150918-00/?p=91561
    """

    def __init__(self, pid: int):
        if not _valid_pid(pid):
            raise ValueError('Invalid process ID')
        self.pid = pid
        self.handle = None

    def __enter__(self):
        import ctypes
        from ctypes import wintypes as wt
        self.ctypes, self.wt = ctypes, wt
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.kernel.OpenProcess.restype = wt.HANDLE
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.kernel.CloseHandle.restype = wt.BOOL
        self.kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        self.kernel.WaitForSingleObject.restype = wt.DWORD
        # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, non-inheritable.
        self.handle = self.kernel.OpenProcess(0x1000 | 0x100000, False, self.pid)
        if not self.handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such process.
                raise ProcessLookupError(f'Process {self.pid} is no longer running')
            raise PermissionError(f'Cannot inspect process {self.pid} (Windows error {error})')
        return self

    def __exit__(self, *_exc):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def alive(self) -> bool:
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status not in (0, 258):
            raise OSError(f'Cannot read process {self.pid} state')
        return status == 258  # WAIT_TIMEOUT means the process has not exited.

    def matches(self, expected: str) -> bool:
        c, wt = self.ctypes, self.wt
        query_image = self.kernel.QueryFullProcessImageNameW
        query_image.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, c.POINTER(wt.DWORD)]
        query_image.restype = wt.BOOL
        size = wt.DWORD(32768)
        image = c.create_unicode_buffer(size.value)
        if not query_image(self.handle, 0, image, c.byref(size)):
            raise PermissionError(f'Cannot read executable for process {self.pid}')
        argv = self._arguments() if expected.lower().endswith(('.py', '.pyw')) else []
        return _windows_identity_matches(image.value, argv, expected)

    def _arguments(self) -> list[str]:
        c, wt = self.ctypes, self.wt
        # Do not guess the layout when caller and target have different bitness.
        wow = self.kernel.IsWow64Process
        wow.argtypes = [wt.HANDLE, c.POINTER(wt.BOOL)]
        wow.restype = wt.BOOL
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = wt.HANDLE
        own, other = wt.BOOL(), wt.BOOL()
        if (not wow(self.kernel.GetCurrentProcess(), c.byref(own))
                or not wow(self.handle, c.byref(other)) or own.value != other.value):
            raise PermissionError(f'Cannot verify command-line architecture for process {self.pid}')

        class UnicodeString(c.Structure):
            _fields_ = [('Length', wt.USHORT), ('MaximumLength', wt.USHORT), ('Buffer', c.c_void_p)]

        query = c.WinDLL('ntdll').NtQueryInformationProcess
        query.argtypes = [wt.HANDLE, wt.ULONG, c.c_void_p, wt.ULONG, c.POINTER(wt.ULONG)]
        query.restype = wt.LONG
        needed = wt.ULONG()
        query(self.handle, 60, None, 0, c.byref(needed))  # ProcessCommandLineInformation
        if not c.sizeof(UnicodeString) <= needed.value <= 131072:
            raise PermissionError(f'Cannot read command line for process {self.pid}')
        buffer = c.create_string_buffer(needed.value)
        if query(self.handle, 60, buffer, len(buffer), c.byref(needed)) < 0:
            raise PermissionError(f'Cannot read command line for process {self.pid}')
        value = UnicodeString.from_buffer(buffer)
        start, end = c.addressof(buffer), c.addressof(buffer) + len(buffer)
        if (not value.Buffer or value.Length % 2 or value.Length > value.MaximumLength
                or value.Buffer < start + c.sizeof(UnicodeString)
                or value.Buffer + value.Length > end):
            raise PermissionError(f'Invalid command-line layout for process {self.pid}')
        command_line = c.wstring_at(value.Buffer, value.Length // 2)
        split = c.WinDLL('shell32', use_last_error=True).CommandLineToArgvW
        split.argtypes = [wt.LPCWSTR, c.POINTER(c.c_int)]
        split.restype = c.POINTER(wt.LPWSTR)
        count = c.c_int()
        args = split(command_line, c.byref(count))
        if not args:
            raise PermissionError(f'Cannot parse command line for process {self.pid}')
        self.kernel.LocalFree.argtypes = [wt.HLOCAL]
        self.kernel.LocalFree.restype = wt.HLOCAL
        try:
            return [args[index] for index in range(count.value)]
        finally:
            self.kernel.LocalFree(c.cast(args, wt.HLOCAL))


def process_alive(pid: int, must_contain: str = '', *, strict: bool = False) -> bool:
    """Check an exact PID and optional identity. Strict mode rejects uncertainty."""
    if not _valid_pid(pid):
        if strict:
            raise ValueError('Invalid process ID')
        return False
    try:
        if WINDOWS:
            with _WindowsProcess(pid) as process:
                if not process.alive():
                    return False
                matched = not must_contain or process.matches(must_contain)
                if strict and not matched:
                    raise PermissionError(f'Process {pid} identity does not match {must_contain}')
                return matched
        out = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip()) and (not must_contain or must_contain in out)
    except ProcessLookupError:
        return False
    except (OSError, subprocess.SubprocessError):
        if strict:
            raise
        return False


def stop_process_tree(pid: int, identity: str) -> bool:
    """Stop only a positively identified process. False means already gone."""
    if not identity or not _valid_pid(pid):
        raise ValueError('A valid PID and exact process identity are required')
    if WINDOWS:
        try:
            with _WindowsProcess(pid) as process:
                if not process.alive():
                    return False
                if not process.matches(identity):
                    raise PermissionError(f'Refusing to stop process {pid}: identity mismatch')
                # Keep the verified handle open through taskkill and the exit
                # check. Its PID cannot be reused while this handle is held.
                result = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                                        capture_output=True, timeout=20,
                                        **background_subprocess_kwargs())
                if result.returncode and process.alive():
                    raise OSError(f'Could not stop process {pid} (exit {result.returncode})')
                return True
        except ProcessLookupError:
            return False
    if not process_alive(pid, identity, strict=True):
        if process_alive(pid):
            raise PermissionError(f'Refusing to stop process {pid}: identity mismatch')
        return False
    import signal
    os.kill(pid, signal.SIGTERM)
    return True


def popen_detached(cmd: list[str], log_handle):
    """Start a long-lived child that survives this process, without a console window."""
    kwargs = dict(stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if WINDOWS:
        kwargs.update(background_subprocess_kwargs())
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def summary() -> dict:
    audio_ok, audio_detail = AudioPlayer.available()
    return {
        "os": OS_NAME,
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg") or "MISSING",
        "audio_ok": audio_ok,
        "audio": audio_detail,
        "state_dir": str(state_dir()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
