#!/usr/bin/env python3
"""c500-guard control: one command for the whole system.

  ctl.py doctor        check everything and say what is wrong
  ctl.py status        bridge + watcher + armed state
  ctl.py on / off      turn the alarm system on or off (same as Ctrl+Alt+F4)
  ctl.py toggle        flip it
  ctl.py ack           acknowledge a sounding alarm
  ctl.py start / stop  start or stop the watcher process
  ctl.py test-alarm    play the three tones
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import platform_support as ps          # noqa: E402

REPO = HERE.parent
STATE = ps.state_dir()
WATCHER_PID = STATE / "watcher.pid"
PAUSED = STATE / "service-paused"
START_LOCK = STATE / "watcher-start.lock"
API = "http://127.0.0.1:8787"
DEFAULT_SOURCE = "rtsp://127.0.0.1:8554/c500"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if ps.WINDOWS:
    # Enable ANSI on Windows 10+ consoles; fall back to no colour if refused.
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.SetConsoleMode(handle, 7)
    except Exception:                                     # noqa: BLE001
        GREEN = RED = YELLOW = DIM = RESET = ""


def venv_python() -> str:
    candidates = [REPO / ".venv" / "Scripts" / "python.exe",
                  REPO / ".venv" / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def api(path: str, method: str = "GET"):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"X-Via": "ctl"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def watcher_state():
    try:
        return api("/state")
    except (OSError, urllib.error.URLError, ValueError):
        return None


def bridge(action: str) -> subprocess.CompletedProcess:
    return subprocess.run([venv_python(), str(HERE / "bridge.py"), action],
                          capture_output=True, text=True, timeout=180,
                          **ps.background_subprocess_kwargs())


def bridge_status() -> dict:
    try:
        result = bridge("status")
        state = json.loads(result.stdout or "{}")
        return state if result.returncode == 0 and isinstance(state, dict) else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def ensure_bridge() -> bool:
    """Start the bridge and relay, then verify readiness before opening video."""
    try:
        result = bridge("start")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Bridge could not start: {exc}")
        return False
    if result.returncode:
        print(f"Bridge start failed (exit {result.returncode}). See {STATE / 'bridge.log'}")
        return False
    state = bridge_status()
    if not state.get("bridge_running") or state.get("api_error"):
        print(f"Bridge is not healthy. See {STATE / 'bridge.log'}")
        return False
    if "lan_relay_pid" in state and not state["lan_relay_pid"]:
        print(f"LAN relay is not running. See {STATE / 'lan-shim.log'}")
        return False
    return True


@contextlib.contextmanager
def process_lock(path: Path):
    """An OS-released lock also protects manual starts against the supervisor."""
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if ps.WINDOWS:
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if ps.WINDOWS:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------

def cmd_doctor(_args) -> int:
    print(f"c500-guard doctor  ({ps.OS_NAME})\n")
    problems = []

    def check(label, ok, detail="", fatal=True, fix=""):
        mark = f"{GREEN}PASS{RESET}" if ok else (f"{RED}FAIL{RESET}" if fatal
                                                 else f"{YELLOW}WARN{RESET}")
        # `fix` is a remedy, so it is shown only when there is something to fix.
        # Printing it beside a PASS reads like a failure.
        shown = detail if ok else (fix or detail)
        print(f"  [{mark}] {label}" + (f"  {DIM}{shown}{RESET}" if shown else ""))
        if not ok and fatal:
            problems.append(label)

    info = ps.summary()
    check("Python 3.10+", sys.version_info >= (3, 10), info["python"])
    check("ffmpeg on PATH", info["ffmpeg"] != "MISSING", info["ffmpeg"])
    check("audio output", info["audio_ok"], info["audio"])
    try:
        import numpy, PIL                                  # noqa: F401
        check("numpy + pillow", True)
    except ImportError as exc:
        check("numpy + pillow", False, str(exc))
    check("zones file", (REPO / "config" / "zones.json").is_file(),
          str(REPO / "config" / "zones.json"))

    bridge_state = bridge_status()
    check("go2rtc bridge running", bool(bridge_state.get("bridge_running"))
          and not bridge_state.get("api_error"),
          f"streams={bridge_state.get('streams', [])}")
    check("camera stream configured", "c500" in bridge_state.get("streams", []),
          ", ".join(bridge_state.get("streams", [])),
          fix="run: bridge.py login-details, then add the camera at /add.html")

    # Only installs whose camera needs the broadcast relay report this key. If
    # the relay dies the camera goes dark, so it belongs in the health check.
    if "lan_relay_pid" in bridge_state:
        check("LAN discovery relay", bool(bridge_state.get("lan_relay_pid")),
              f"pid={bridge_state.get('lan_relay_pid')}",
              fix="relay not running; run: bridge.py start")

    state = watcher_state()
    check("watcher running", state is not None, API,
          fix="run: ctl.py start")
    if state is not None:
        if "armed" not in state:
            # Something else is on this port - an older build, or another app.
            check("watcher is this version", False,
                  "port 8787 answered but without an 'armed' field; stop the "
                  "other process before starting c500-guard")
        else:
            check("system armed", state["armed"],
                  "ON" if state["armed"] else "OFF - press Ctrl+Alt+F4 to turn on",
                  fatal=False)
            age = state.get("last_frame_age")
            check("video flowing", state.get("frames", 0) > 0 and age is not None and age < 3,
                  f"last frame {age}s ago")
            ml = state.get("ml", {})
            check("local AI model", ml.get("available", False),
                  f"{ml.get('inferences', 0)} observations",
                  fix=ml.get("error") or "model is loading or unavailable")
            loop_age = state.get("loop_age_s")
            if isinstance(loop_age, (int, float)):
                check("detector loop running", loop_age < 5, f"last cycle {loop_age}s ago",
                      fix=f"no detection cycle for {loop_age}s; the watcher restarts itself")
            audio = state.get("audio") or {}
            if audio.get("available"):
                percent = round(float(audio.get("volume") or 0) * 100)
                check("speaker not muted", not audio.get("muted"), audio.get("device") or "default speaker",
                      fix="the speaker is MUTED - alarms are silent until it is unmuted")
                check("speaker volume", percent >= 30, f"{percent}%", fatal=False,
                      fix=f"volume is {percent}% - raise it so the siren is heard")
            elif audio:
                check("speaker level readable", False, fatal=False,
                      fix=audio.get("detail") or "speaker state unknown")
            if ps.WINDOWS and "input_helper" in state:
                check("key/click acknowledgement helper", bool(state["input_helper"].get("connected")),
                      "connected", fatal=False,
                      fix="not polling; run: Start-ScheduledTask -TaskName 'c500-guard hotkey'")
            alignment = state.get("alignment", {})
            if alignment.get("state") not in (None, "disabled"):
                check("automatic zone alignment", alignment.get("state") in ("tracking", "holding"),
                      alignment.get("message", ""),
                      fix=alignment.get("message") or "check the current view and reference")

    print()
    if problems:
        print(f"{RED}{len(problems)} problem(s):{RESET} " + ", ".join(problems))
        print(f"{DIM}  See {REPO / 'docs' / 'c500-guard-runbook.pdf'} "
              f"section 4 - every known error and its fix.{RESET}")
        return 1
    print(f"{GREEN}All checks passed.{RESET}")
    return 0


def cmd_status(_args) -> int:
    print(bridge("status").stdout.strip())
    state = watcher_state()
    if not state:
        print(json.dumps({"watcher_running": False}))
        return 1
    print(json.dumps({"watcher_running": True,
                      "armed": state.get("armed"),
                      "alarm": state.get("alarm"), "reason": state.get("reason"),
                      "frames": state.get("frames"), "restarts": state.get("restarts"),
                      "last_frame": state.get("last_frame_clock"),
                      "last_frame_age_s": state.get("last_frame_age")}, indent=2))
    return 0


def _switch(path: str) -> int:
    try:
        result = api(path, "POST")
    except (OSError, urllib.error.URLError):
        print(f"{RED}Watcher is not running.{RESET} Start it with: ctl.py start")
        return 1
    print("ON  - watching" if result.get("armed") else "OFF - not watching")
    return 0


def cmd_on(_a):     return _switch("/arm")
def cmd_off(_a):    return _switch("/disarm")
def cmd_toggle(_a): return _switch("/toggle")


def cmd_ack(_args) -> int:
    try:
        api("/ack", "POST")
    except (OSError, urllib.error.URLError):
        print(f"{RED}Watcher is not running.{RESET}")
        return 1
    print("Acknowledged.")
    return 0


def cmd_start(args) -> int:
    if getattr(args, "resume", True):
        PAUSED.unlink(missing_ok=True)
    with process_lock(START_LOCK) as acquired:
        if not acquired:
            print("Watcher startup is already in progress.")
            return 1
        return start_watcher(args)


def start_watcher(args) -> int:
    if not ensure_bridge():
        return 1
    if PAUSED.exists():
        print("Watcher startup cancelled by ctl.py stop.")
        return 0
    state = watcher_state()
    if state is not None and "armed" not in state:
        print("Port 8787 answers for another application. Watcher was not started.")
        return 1
    if state is not None:
        print("Watcher is already running.")
        return 0
    if WATCHER_PID.exists():
        try:
            pid = int(WATCHER_PID.read_text().strip())
            if ps.process_alive(pid, str(HERE / 'watch.py'), strict=True):
                print(f'Watcher pid {pid} is running but not responding. Refusing a duplicate start.')
                return 1
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Cannot verify the recorded watcher process: {exc}. Refusing a duplicate start.')
            return 1
    cmd = [venv_python(), str(HERE / "watch.py"), "watch", args.source,
           "--zones", str(REPO / "config" / "zones.json"), "--web", "8787"]
    ps.rotate_log(STATE / "watcher.log")
    with (STATE / "watcher.log").open("ab") as handle:
        child = ps.popen_detached(cmd, handle)
    WATCHER_PID.write_text(str(child.pid) + "\n")
    for _ in range(60):
        if PAUSED.exists():
            break
        if watcher_state():
            if PAUSED.exists():
                break
            print(f"Watcher started (pid {child.pid}). Open {API}")
            return 0
        if child.poll() is not None:
            print(f"Watcher exited (exit {child.returncode}). See {STATE / 'watcher.log'}")
            WATCHER_PID.unlink(missing_ok=True)
            return 1
        time.sleep(1)
    # The watcher may have become ready at the timeout boundary. Otherwise,
    # stop its entire verified tree: terminating a venv launcher alone can
    # orphan the actual Python watcher and ffmpeg process.
    if not PAUSED.exists() and watcher_state():
        print(f"Watcher started (pid {child.pid}). Open {API}")
        return 0
    try:
        ps.stop_process_tree(child.pid, str(HERE / 'watch.py'))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Watcher startup failed and cleanup could not verify a safe stop: {exc}. '
              'The recorded PID was retained; no duplicate will be started.')
        return 1
    WATCHER_PID.unlink(missing_ok=True)
    print(f"{RED}Watcher did not come up.{RESET} See {STATE / 'watcher.log'}")
    return 1


def cmd_stop(_args) -> int:
    # The supervisor must distinguish an intentional stop from a crash.
    PAUSED.write_text("Stopped by ctl.py stop. Run ctl.py start to resume.\n")
    with process_lock(START_LOCK) as acquired:
        if not acquired:
            print("Supervisor paused; an in-progress startup will cancel.")
            return 0
        return stop_watcher()


def stop_watcher() -> int:
    try:
        pid = int(WATCHER_PID.read_text().strip())
    except (OSError, ValueError):
        print("Watcher autostart paused; no watcher pid recorded.")
        return 0
    try:
        ps.stop_process_tree(pid, str(HERE / 'watch.py'))
        WATCHER_PID.unlink(missing_ok=True)
        print("Watcher stopped; supervisor paused until ctl.py start.")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Could not stop pid {pid}: {exc}")
        return 1


def cmd_test_alarm(args) -> int:
    return subprocess.run([venv_python(), str(HERE / "watch.py"),
                           "test-alarm", "--seconds", str(args.seconds)],
                          **ps.background_subprocess_kwargs()).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("doctor", cmd_doctor), ("status", cmd_status), ("on", cmd_on),
                     ("off", cmd_off), ("toggle", cmd_toggle), ("ack", cmd_ack),
                     ("stop", cmd_stop)):
        sub.add_parser(name).set_defaults(func=fn)
    start = sub.add_parser("start")
    start.add_argument("--source", default=DEFAULT_SOURCE)
    start.set_defaults(func=cmd_start)
    test = sub.add_parser("test-alarm")
    test.add_argument("--seconds", type=float, default=3.0)
    test.set_defaults(func=cmd_test_alarm)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
