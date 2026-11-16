#!/usr/bin/env python3
"""c500-guard supervisor: run only after the explicit whole-program ON command.

Run by the clearly named 'c500-guard service' scheduled task in the current
user's interactive session. The watcher restores its own saved armed state.
ctl.py stop pauses restarts; ctl.py start resumes them.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import ctl


LOOP_STALL_S = 45   # the watcher's own watchdog normally acts after 20 s
REFRESH_WAIT_S = 20  # longest wait for a refreshing watcher to finish closing


def restart_requested() -> bool:
    return (ctl.STATE / "restart-request").exists()


def finish_refresh(wait_s: float = REFRESH_WAIT_S) -> str:
    """The dashboard asked for a clean restart: wait for the old watcher to
    close (it restores the speaker and releases the camera), then start a new
    one straight away instead of at the next 15-second check."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if ctl.POWER_OFF.exists():
            return "Full program OFF during refresh."
        if ctl.watcher_state() is None:
            break
        time.sleep(.25)
    else:
        # Still answering: stop only the verified watcher tree.
        if ctl.WATCHER_PID.exists():
            ctl.stop_watcher()
    try:
        (ctl.STATE / "restart-request").unlink()
    except OSError:
        pass
    result = ctl.cmd_start(argparse.Namespace(source=ctl.configured_source(), resume=False))
    return "Refreshed: fresh watcher started." if result == 0 else "Refresh: restart failed; will retry."


def loop_stalled(state) -> bool:
    """The API can answer while the detector thread is stuck; check its heartbeat."""
    age = state.get("loop_age_s") if isinstance(state, dict) else None
    return isinstance(age, (int, float)) and not isinstance(age, bool) and age > LOOP_STALL_S


def supervise(interval: float = 15) -> None:
    last_status = None
    missing_checks = 0
    stalled_checks = 0
    while True:
        if ctl.POWER_OFF.exists():
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Full program OFF; supervisor exiting.", flush=True)
            return
        if ctl.PAUSED.exists():
            missing_checks = stalled_checks = 0
            status = "Paused by ctl.py stop; waiting for ctl.py start."
        elif restart_requested():
            missing_checks = stalled_checks = 0
            try:
                status = finish_refresh()
            except Exception as exc:
                status = f"Refresh failed: {type(exc).__name__}: {exc}"
        else:
            try:
                state = ctl.watcher_state()
                stalled_checks = stalled_checks + 1 if loop_stalled(state) else 0
                if stalled_checks:
                    # Two consecutive stalled checks: stop the verified watcher;
                    # the next cycle starts a fresh one with the saved ON/OFF.
                    if stalled_checks >= 2 and ctl.WATCHER_PID.exists():
                        status = ("Detector stalled; watcher stopped for restart." if ctl.stop_watcher() == 0
                                  else "Detector stalled; process could not be verified for recovery.")
                    else:
                        status = "Detector loop not responding; checking again."
                elif state is None:
                    missing_checks += 1
                    # One slow API response is not a reason to restart. After
                    # three failed checks, terminate only the verified owned
                    # watcher tree; ctl refuses unknown or recycled PIDs.
                    if missing_checks >= 3 and ctl.WATCHER_PID.exists() and ctl.stop_watcher() != 0:
                        status = "Watcher unresponsive; process could not be verified for recovery."
                    else:
                        result = ctl.cmd_start(argparse.Namespace(
                            source=ctl.configured_source(), resume=False))
                        if result == 0:
                            missing_checks = 0
                        status = "Watcher started." if result == 0 else "Startup failed; will retry."
                elif "armed" not in state:
                    missing_checks = 0
                    status = "Port 8787 is occupied by another application; waiting."
                elif not ctl.needs_bridge(ctl.configured_source()):
                    missing_checks = 0
                    status = "Watcher running (no camera bridge needed)."
                elif ctl.ensure_bridge():
                    missing_checks = 0
                    status = "Bridge and watcher running."
                else:
                    missing_checks = 0
                    status = "Bridge unavailable; will retry."
            except Exception as exc:
                status = f"Startup check failed: {type(exc).__name__}: {exc}"
        if status != last_status:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), status, flush=True)
            last_status = status
        # Cooperate promptly with a full stop instead of waiting a whole retry
        # interval. The one-shot controller also terminates a stalled service.
        deadline = time.monotonic() + interval
        while time.monotonic() < deadline:
            if ctl.POWER_OFF.exists():
                return
            if restart_requested():
                break
            time.sleep(min(.25, max(0, deadline - time.monotonic())))


def main() -> int:
    if ctl.POWER_OFF.exists():
        return 0
    ctl.ps.normal_priority()
    log_path = ctl.STATE / "service.log"
    ctl.ps.rotate_log(log_path)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            with ctl.process_lock(ctl.STATE / "service.lock") as acquired:
                if not acquired:
                    print("Another c500-guard supervisor is already running.")
                    return 0
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "c500-guard supervisor starting.")
                supervise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
