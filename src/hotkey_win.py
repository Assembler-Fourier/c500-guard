#!/usr/bin/env python3
"""Global on/off and alarm acknowledgement for c500-guard (Windows).

Installs a low-level keyboard hook, so the combination works no matter which
window has focus. The keystroke is SWALLOWED when it matches, which matters:
it means the press can never fall through to the foreground app as Alt+F4 and
close somebody's window.

Any physical key-down or mouse-button-down also acknowledges the alarm that
was active when the input arrived. Ordinary input still reaches its app;
injected input, mouse movement and scrolling do not acknowledge alarms. No
typed text, key codes or mouse coordinates are retained or logged.

Talks to the watcher over loopback HTTP only. It holds no credentials and
never touches the camera. Hook callbacks only inspect flags and enqueue work;
HTTP, logging and sound run on worker threads.

Run:  pythonw.exe hotkey_win.py           (no console window)
      python.exe  hotkey_win.py --debug   (prints service status, never input)
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import ctypes
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import platform_support as ps

# Virtual key codes
VK_F4, VK_CONTROL, VK_MENU = 0x73, 0x11, 0x12
WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_KEYUP, WM_SYSKEYUP = 0x0101, 0x0105
WM_LBUTTONDOWN, WM_RBUTTONDOWN = 0x0201, 0x0204
WM_MBUTTONDOWN, WM_XBUTTONDOWN = 0x0207, 0x020B
MOUSE_BUTTON_DOWN = frozenset((WM_LBUTTONDOWN, WM_RBUTTONDOWN,
                             WM_MBUTTONDOWN, WM_XBUTTONDOWN))
LLKHF_INJECTED, LLKHF_LOWER_IL_INJECTED = 0x10, 0x02
LLMHF_INJECTED, LLMHF_LOWER_IL_INJECTED = 0x01, 0x02
HC_ACTION = 0
WM_TIMER = 0x0113
HOOK_REFRESH_MS = 5 * 60 * 1000
POWER_CHECK_MS = 1000
SWALLOW = 1          # non-zero return stops the key reaching any other app
INPUT_POLL_SECONDS = .2
INPUT_CACHE_MAX_AGE = .75
INPUT_HTTP_TIMEOUT = .6

DEFAULT_API = "http://127.0.0.1:8787"


@contextlib.contextmanager
def helper_lock(path: Path):
    """A lifetime OS lock excludes direct starts as well as scheduled starts.

    The file may remain after shutdown; the OS releases its byte lock even if
    the helper is terminated. Its existence never means a process is alive.
    """
    import msvcrt
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _win_types():
    """Built lazily: ctypes.wintypes and WINFUNCTYPE only exist on Windows,
    so importing this module must not require them."""
    import ctypes.wintypes as wintypes

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                    ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]

    class MSLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD),
                    ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]

    # LRESULT and ULONG_PTR are pointer-sized on x64, unlike Windows LONG.
    hookproc = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int,
                                  wintypes.WPARAM, wintypes.LPARAM)
    return wintypes, KBDLLHOOKSTRUCT, MSLLHOOKSTRUCT, hookproc


class Hotkey:
    def __init__(self, api: str, debug: bool = False, beep_on_off: bool = False,
                 log: Path | None = None, *, state: Path | None = None):
        self.api = api.rstrip("/")
        self.debug = debug
        self.beep_on_off = beep_on_off
        self.log_path = log
        self.state = state if state is not None else ps.state_dir()
        self.power_off = self.state / "power-off"
        self._wintypes, self._KBD, self._MOUSE, hookproc = _win_types()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api(hookproc)
        self._proc = hookproc(self._on_key)
        self._mouse_proc = hookproc(self._on_mouse)
        self._hook = None
        self._mouse_hook = None
        self._last_fire = float("-inf")
        self._f4_down = False
        self._requests = queue.Queue(maxsize=1)
        self._pending = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._toggle_worker, daemon=True,
                                        name="c500-guard-hotkey-toggle")
        # Immutable snapshots and nonblocking queues keep the hook thread out
        # of HTTP locks. A cached ID tags the exact alarm observed at input
        # time; the server rejects it if that alarm has already ended.
        self._alarm_snapshot = (None, float("-inf"))
        self._last_queued_alarm_id = None
        self._ack_requests = queue.Queue(maxsize=1)
        # One coalesced timestamp covers an alarm that started after our last
        # poll. deque append/popleft are atomic; no key or mouse data is kept.
        self._fallback_inputs = deque(maxlen=1)
        self._input_wake = threading.Event()
        # Separate workers ensure a slow acknowledgement/poll cannot delay
        # the user's on/off command or consume its queue slot.
        self._input_worker = threading.Thread(target=self._input_loop, daemon=True,
                                              name="c500-guard-input-ack")

    def _configure_api(self, hookproc) -> None:
        """ctypes defaults truncate handles/LRESULT on 64-bit Windows."""
        wt = self._wintypes
        signatures = {
            "SetWindowsHookExW": ([ctypes.c_int, hookproc, wt.HINSTANCE, wt.DWORD], wt.HANDLE),
            "CallNextHookEx": ([wt.HANDLE, ctypes.c_int, wt.WPARAM, wt.LPARAM], ctypes.c_ssize_t),
            "UnhookWindowsHookEx": ([wt.HANDLE], wt.BOOL),
            "GetAsyncKeyState": ([ctypes.c_int], wt.SHORT),
            "GetMessageW": ([ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT], wt.BOOL),
            "TranslateMessage": ([ctypes.POINTER(wt.MSG)], wt.BOOL),
            "DispatchMessageW": ([ctypes.POINTER(wt.MSG)], ctypes.c_ssize_t),
            "SetTimer": ([wt.HWND, ctypes.c_size_t, wt.UINT, ctypes.c_void_p], ctypes.c_size_t),
            "KillTimer": ([wt.HWND, ctypes.c_size_t], wt.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.user32, name)
            function.argtypes = arguments
            function.restype = result
        self.kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wt.HMODULE

    def _toggle_worker(self) -> None:
        """All HTTP, sound, printing and file I/O stays off the hook thread."""
        while not self._stop.is_set():
            try:
                self._requests.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                if not self._stop.is_set():
                    self.toggle()
            except Exception as exc:
                # Keep the worker alive if a response/sound/logging path fails.
                self.say(f"hotkey toggle failed: {exc}")
            finally:
                self._pending.clear()
                self._requests.task_done()

    def _queue_ack(self, via: str) -> None:
        """Hook-thread path: do not log, make HTTP calls, or wait for locks."""
        pressed_at = time.monotonic()
        alarm_id, sampled_at = self._alarm_snapshot
        if (not alarm_id or pressed_at - sampled_at > INPUT_CACHE_MAX_AGE or
                alarm_id == self._last_queued_alarm_id):
            # A press during the first polling interval of a new alarm must
            # still count. The worker checks this timestamp against the alarm
            # start time, so input preceding a future alarm cannot silence it.
            self._fallback_inputs.append((pressed_at, via))
            self._input_wake.set()
            return
        previous_id = self._last_queued_alarm_id
        self._last_queued_alarm_id = alarm_id
        try:
            self._ack_requests.put_nowait((alarm_id, via, pressed_at))
        except queue.Full:
            if self._last_queued_alarm_id == alarm_id:
                self._last_queued_alarm_id = previous_id
            return
        self._input_wake.set()

    def _refresh_alarm_status(self) -> tuple[str | None, float | None]:
        try:
            with urllib.request.urlopen(self.api + "/input-status",
                                        timeout=INPUT_HTTP_TIMEOUT) as response:
                data = json.load(response)
            alarm_id = data.get("alarm_id")
            if not isinstance(alarm_id, str) or not 0 < len(alarm_id) <= 128:
                alarm_id = None
            started_at = data.get("alarm_started_at")
            if (not isinstance(started_at, (int, float)) or isinstance(started_at, bool)
                    or not math.isfinite(started_at) or started_at < 0):
                started_at = None
            self._alarm_snapshot = (alarm_id, time.monotonic())
            return alarm_id, started_at
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            # Never acknowledge from stale state after a watcher outage.
            # Poll failures stay quiet rather than filling the disk with logs.
            self._alarm_snapshot = (None, time.monotonic())
            return None, None

    def acknowledge(self, alarm_id: str, via: str) -> bool | None:
        """Send the observed alarm ID once. A stale ID is a harmless no-op."""
        request = urllib.request.Request(
            self.api + "/ack", method="POST",
            data=json.dumps({"alarm_id": alarm_id}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Via": via})
        try:
            with urllib.request.urlopen(request, timeout=INPUT_HTTP_TIMEOUT) as response:
                # Consume the response so the local connection closes cleanly.
                result = json.load(response)
            return result.get("acknowledged") is True
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _deliver_ack(self, alarm_id: str, via: str, pressed_at: float,
                     *, allow_lookup: bool = True) -> None:
        try:
            accepted = self.acknowledge(alarm_id, via)
        except Exception:
            accepted = None
        if accepted is True:
            return
        if self._last_queued_alarm_id == alarm_id:
            self._last_queued_alarm_id = None
        if not allow_lookup or self._stop.is_set():
            return
        # The cached ID may have been superseded before the user's press.
        # Resolve once, bound to that press's time. A timeout may have already
        # silenced the alarm; the fresh status then has no active ID to retry.
        current_id, started_at = self._refresh_alarm_status()
        if (current_id and started_at is not None and started_at <= pressed_at
                and current_id != self._last_queued_alarm_id and not self._stop.is_set()):
            self._last_queued_alarm_id = current_id
            self._deliver_ack(current_id, via, pressed_at, allow_lookup=False)

    def _input_loop(self) -> None:
        next_poll = 0.0
        while not self._stop.is_set():
            try:
                alarm_id, via, pressed_at = self._ack_requests.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    if not self._stop.is_set():
                        self._deliver_ack(alarm_id, via, pressed_at)
                finally:
                    self._ack_requests.task_done()
                    next_poll = 0.0
            if self._stop.is_set():
                break
            now = time.monotonic()
            if now >= next_poll:
                try:
                    fallback = self._fallback_inputs.popleft()
                except IndexError:
                    fallback = None
                alarm_id, started_at = self._refresh_alarm_status()
                next_poll = time.monotonic() + INPUT_POLL_SECONDS
                if fallback and alarm_id and started_at is not None:
                    pressed_at, via = fallback
                    if (started_at <= pressed_at and
                            alarm_id != self._last_queued_alarm_id and not self._stop.is_set()):
                        self._last_queued_alarm_id = alarm_id
                        self._deliver_ack(alarm_id, via, pressed_at, allow_lookup=False)
                        next_poll = 0.0
            self._input_wake.wait(max(0, min(INPUT_POLL_SECONDS,
                                           next_poll - time.monotonic())))
            self._input_wake.clear()

    # -- plumbing ---------------------------------------------------------
    def say(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  pid={os.getpid()} {message}"
        if self.debug:
            print(line, flush=True)
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass

    def beep(self, armed: bool) -> None:
        """Confirmation that the press registered.

        Nothing is played when switching OFF by default: while the system is
        off it must make no sound at all. Pass --beep-both if you would rather
        have confirmation in both directions.
        """
        if not armed and not self.beep_on_off:
            return
        try:
            import winsound
            if armed:
                winsound.Beep(880, 90); winsound.Beep(1320, 110)
            else:
                winsound.Beep(520, 90); winsound.Beep(360, 110)
        except Exception:                                  # noqa: BLE001
            pass

    def toggle(self) -> None:
        request = urllib.request.Request(self.api + "/toggle", method="POST",
                                         headers={"X-Via": "hotkey"})
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                armed = json.load(response).get("armed")
            self.say(f"toggled -> {'ON' if armed else 'OFF'}")
            self.beep(bool(armed))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # The watcher is not running. Stay quiet and keep waiting for it.
            self.say(f"watcher not reachable at {self.api}: {exc}")

    # -- the hook ---------------------------------------------------------
    def _forward_input(self, hook, code, wparam, lparam):
        """No Python exception may escape a native callback as its result.

        Zero is the fail-open fallback if forwarding itself fails. Never log
        or inspect more input here: this path runs on the Windows hook thread.
        """
        try:
            return self.user32.CallNextHookEx(hook, code, wparam, lparam)
        except BaseException:
            return 0

    def _on_key(self, code, wparam, lparam):
        try:
            if self._handle_key(code, wparam, lparam):
                return SWALLOW
        except BaseException:
            # ctypes otherwise reports an ignored exception and has no defined
            # Python return value to pass back to Windows. Always forward.
            self._f4_down = False
        return self._forward_input(self._hook, code, wparam, lparam)

    def _handle_key(self, code, wparam, lparam):
        if code == HC_ACTION and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP):
            data = ctypes.cast(lparam, ctypes.POINTER(self._KBD)).contents
            if data.flags & (LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED):
                return False
            if data.vkCode == VK_F4:
                if wparam in (WM_KEYUP, WM_SYSKEYUP):
                    if self._f4_down:
                        self._f4_down = False
                        return True
                    return False
                # Swallow auto-repeat even if a modifier was released first.
                if self._f4_down:
                    return True
                # 0x8000 = key currently down.
                ctrl = self.user32.GetAsyncKeyState(VK_CONTROL) & 0x8000
                alt = self.user32.GetAsyncKeyState(VK_MENU) & 0x8000
                if ctrl and alt:
                    self._f4_down = True
                    now = time.monotonic()
                    # Collapse repeats and presses while a previous HTTP call
                    # is outstanding. Never block this Windows hook callback.
                    if now - self._last_fire > 0.5 and not self._pending.is_set():
                        self._pending.set()
                        try:
                            self._requests.put_nowait(True)
                            self._last_fire = now
                        except queue.Full:
                            self._pending.clear()
                    return True
            if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self._queue_ack("input-any-key")
        return False

    def _on_mouse(self, code, wparam, lparam):
        try:
            if code == HC_ACTION and wparam in MOUSE_BUTTON_DOWN:
                data = ctypes.cast(lparam, ctypes.POINTER(self._MOUSE)).contents
                if not data.flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED):
                    self._queue_ack("input-mouse")
        except BaseException:
            pass
        # Acknowledging must never suppress a normal click.
        return self._forward_input(self._mouse_hook, code, wparam, lparam)

    def _refresh_hooks(self, module) -> None:
        """Re-register both hooks; the new one is live before the old one goes.

        Windows silently removes a low-level hook whose callback once exceeded
        LowLevelHooksTimeout (for example under heavy CPU load), with no error
        to observe. Periodic re-registration bounds that outage to minutes.
        """
        for attr, kind, proc in (("_hook", WH_KEYBOARD_LL, self._proc),
                                 ("_mouse_hook", WH_MOUSE_LL, self._mouse_proc)):
            fresh = self.user32.SetWindowsHookExW(kind, proc, module, 0)
            if not fresh:
                self.say(f"hook refresh failed (error {ctypes.get_last_error()}); keeping the current hook")
                continue
            old = getattr(self, attr)
            setattr(self, attr, fresh)
            if old:
                self.user32.UnhookWindowsHookEx(old)

    def run(self) -> int:
        result = 1
        self.say("input helper starting")
        try:
            if self._power_off_requested():
                self.say("Program is fully OFF; no input hooks installed.")
                result = 0
                return result
            with helper_lock(self.state / "hotkey.lock") as acquired:
                if not acquired:
                    self.say("Another input helper is already running; duplicate start ignored.")
                    result = 0
                    return result
                result = self._run_hooks()
                return result
        except OSError as exc:
            self.say(f"Input helper cannot run safely: {type(exc).__name__}")
            return result
        finally:
            self.say(f"input helper exited; result={result}")

    def _power_off_requested(self) -> bool:
        try:
            self.power_off.stat()
        except FileNotFoundError:
            return False
        except OSError:
            # If the persistent OFF state is unreadable, do not reserve input.
            return True
        return True

    def _run_hooks(self) -> int:
        # Recheck after the singleton lock, closing a race with full shutdown.
        if self._power_off_requested():
            self.say("Program is fully OFF; no input hooks installed.")
            return 0
        module = self.kernel32.GetModuleHandleW(None)
        timer = None
        self._hook = self.user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, module, 0)
        if not self._hook:
            err = ctypes.get_last_error()
            self.say(f"FATAL: could not install keyboard hook (error {err}). "
                     f"If the watcher runs elevated, this must run elevated too.")
            return 1
        try:
            self._mouse_hook = self.user32.SetWindowsHookExW(
                WH_MOUSE_LL, self._mouse_proc, module, 0)
            if not self._mouse_hook:
                self.say(f"FATAL: could not install mouse hook (error {ctypes.get_last_error()}).")
                return 1
            self._worker.start()
            self._input_worker.start()
            self.say(f"Ctrl+Alt+F4 ready; physical key/click acknowledgement ready; "
                     f"talking to {self.api}")
            # A thread timer (no window) posts WM_TIMER to this message loop.
            timer = self.user32.SetTimer(None, 0, POWER_CHECK_MS, None)
            if not timer:
                self.say("FATAL: cannot schedule the full-OFF check; removing input hooks.")
                return 1
            next_refresh = time.monotonic() + HOOK_REFRESH_MS / 1000
            message = self._wintypes.MSG()
            while True:
                result = self.user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == 0:
                    return 0
                if result == -1:
                    self.say(f"FATAL: keyboard message loop failed (error {ctypes.get_last_error()})")
                    return 1
                if message.message == WM_TIMER and message.wParam == timer:
                    if self._power_off_requested():
                        self.say("Program is fully OFF; removing input hooks.")
                        return 0
                    now = time.monotonic()
                    if now >= next_refresh:
                        self._refresh_hooks(module)
                        next_refresh = now + HOOK_REFRESH_MS / 1000
                    continue
                self.user32.TranslateMessage(ctypes.byref(message))
                self.user32.DispatchMessageW(ctypes.byref(message))
        finally:
            if timer:
                self.user32.KillTimer(None, timer)
            self._stop.set()
            self._input_wake.set()
            if self._mouse_hook:
                self.user32.UnhookWindowsHookEx(self._mouse_hook)
                self._mouse_hook = None
            if self._hook:
                self.user32.UnhookWindowsHookEx(self._hook)
                self._hook = None
            if self._worker.is_alive():
                self._worker.join(timeout=.3)
            if self._input_worker.is_alive():
                self._input_worker.join(timeout=.3)


def main() -> int:
    if not sys.platform.startswith("win"):
        print("hotkey_win.py is Windows-only. On macOS and Linux use the "
              "Turn ON/OFF button on the web page, or 'ctl.py toggle'.",
              file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--beep-both", action="store_true",
                        help="also beep when switching OFF (default: silent when going off)")
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    # A keyboard hook in a paged-out, below-normal process delays every
    # keystroke on the PC. Run at normal priority for as long as we run.
    ps.normal_priority()
    log = Path(args.log) if args.log else None
    return Hotkey(args.api, args.debug, args.beep_both, log).run()


if __name__ == "__main__":
    raise SystemExit(main())
