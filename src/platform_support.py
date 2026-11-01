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
    if expected.lower().endswith('.py'):
        interpreter = ntpath.basename(image).lower()
        if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?w?\.exe', interpreter):
            return False
        # All app launchers pass the absolute script directly after Python.
        # A -c string or unrelated script's argument is not process identity.
        if len(argv) < 2:
            return False
        script = normalized(argv[1])
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
        argv = self._arguments() if expected.lower().endswith('.py') else []
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
