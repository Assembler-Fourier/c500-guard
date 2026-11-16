"""One-shot Windows C500 inventory and guarded cleanup. Never records arguments."""

from __future__ import annotations

import argparse
import ctypes
import json
import ntpath
import os
from pathlib import Path
import re
import sys
import time

import platform_support as ps


REPO = Path(__file__).resolve().parent.parent
STATE = ps.state_dir()
PYTHON_NAME = re.compile(r"python(?:\d+(?:\.\d+)*)?w?\.exe", re.I)
SCRIPT_COMPONENTS = (
    ("barrierwatcher_app.pyw", "app launcher"),
    ("hotkey_win.py", "input helper"),
    ("service.py", "supervisor"),
    ("ctl.py", "watcher controller"),
    ("bridge.py", "bridge controller"),
    ("watch.py", "watcher"),
    ("lan_shim.py", "LAN relay"),
)
STOP_ORDER = tuple(component for _, component in SCRIPT_COMPONENTS) + (
    "native app", "car-memory updater", "camera bridge", "video decoder")


def expected_identities(repo=REPO, state=STATE):
    identities = [(str(repo / "src" / name), component) for name, component in SCRIPT_COMPONENTS]
    identities += [(str(repo / "app" / "BarrierWatcher.exe"), "native app"),
                   (str(state / "bin" / "go2rtc.exe"), "camera bridge"),
                   (str(repo / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"), "video decoder")]
    try:
        config = json.loads((repo / "config" / "car-memory.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return identities  # Optional integration has not been configured.
    # A configured updater may outlive its launcher if assigning its job failed.
    # Do not silently report OFF complete when its identity cannot be established.
    script = config.get("script") if isinstance(config, dict) else None
    if (not isinstance(script, str) or not Path(script).is_absolute()
            or not script.lower().endswith((".py", ".pyw"))):
        raise ValueError("Car-memory script identity must be an absolute Python script path")
    identities.append((str(Path(script)), "car-memory updater"))
    return identities


def windows_process_rows():
    """Toolhelp gives names/PIDs even when command-line inspection is denied."""
    from ctypes import wintypes as wt

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD),
                    ("th32ProcessID", wt.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                    ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", wt.LONG),
                    ("dwFlags", wt.DWORD), ("szExeFile", wt.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32FirstW.restype = wt.BOOL
    kernel.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.restype = wt.BOOL
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(f"Cannot enumerate processes (Windows error {ctypes.get_last_error()})")
    rows = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while found:
            rows.append({"pid": entry.th32ProcessID, "parent_pid": entry.th32ParentProcessID,
                         "name": entry.szExeFile})
            found = kernel.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() not in (0, 18):  # ERROR_NO_MORE_FILES
            raise OSError("Process enumeration did not finish")
    finally:
        kernel.CloseHandle(handle)
    return rows


def inventory(rows=None, process_factory=None, identities=None):
    """Unknown candidate ownership prevents a false 'nothing running' result."""
    rows = windows_process_rows() if rows is None else rows
    process_factory = ps._WindowsProcess if process_factory is None else process_factory
    identities = expected_identities() if identities is None else identities
    native_names = {"go2rtc.exe", "ffmpeg.exe"} | {
        ntpath.basename(identity).lower() for identity, _ in identities
        if not identity.lower().endswith((".py", ".pyw"))}
    owned, unverified = [], []
    for row in rows:
        name, pid = row["name"], row["pid"]
        if pid == os.getpid():
            continue
        is_python = bool(PYTHON_NAME.fullmatch(name))
        if not is_python and name.lower() not in native_names:
            continue
        candidates = [(identity, component) for identity, component in identities
                      if identity.lower().endswith((".py", ".pyw")) == is_python]
        try:
            with process_factory(pid) as process:
                if not process.alive():
                    continue
                for identity, component in candidates:
                    if process.matches(identity):
                        owned.append({**row, "identity": identity, "component": component})
                        break
        except ProcessLookupError:
            continue
        except OSError as exc:
            unverified.append({**row, "error": str(exc)})
    return {"owned": owned, "unverified": unverified}


def cleanup_pid_files(state=STATE, repo=REPO):
    """Remove dead records only; a recycled or unreadable PID is preserved."""
    records = (("watcher.pid", repo / "src" / "watch.py"),
               ("bridge.pid", state / "bin" / "go2rtc.exe"),
               ("lan-shim.pid", repo / "src" / "lan_shim.py"))
    for filename, identity in records:
        path = state / filename
        try:
            pid = int(path.read_text().strip())
            if not ps.process_alive(pid, str(identity), strict=True):
                path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def stop_owned(*, inventory_fn=inventory, stop_fn=ps.stop_process_tree, sleep=time.sleep):
    """Exact identity is verified again, under a held handle, before every stop."""
    stopped, errors = [], []
    # Stop the app launcher before its children so it cannot launch replacements
    # while OFF is taking inventory. Final snapshots still verify every survivor.
    order = {component: index for index, component in enumerate(STOP_ORDER)}
    for _ in range(3):
        snapshot = inventory_fn()
        if not snapshot["owned"]:
            break
        for row in sorted(snapshot["owned"], key=lambda item: order.get(item["component"], 99)):
            try:
                if stop_fn(row["pid"], row["identity"]):
                    stopped.append({"pid": row["pid"], "component": row["component"]})
            except OSError as exc:
                errors.append({"pid": row["pid"], "error": str(exc)})
        sleep(.25)
    final = inventory_fn()
    # An earlier failed stop is not a failure if a later verified parent stop
    # removed it. The final inventory is the authority for completion.
    remaining_ids = {row["pid"] for row in final["owned"] + final["unverified"]}
    errors = [error for error in errors if error["pid"] in remaining_ids]
    return {**final, "stopped": stopped, "errors": errors,
            "ok": not final["owned"] and not final["unverified"] and not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "stop"))
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    if not ps.WINDOWS:
        print(json.dumps({"ok": False, "error": "Windows only"}))
        return 2
    try:
        identities = expected_identities(args.root.resolve(), STATE)
        snapshot = lambda: inventory(identities=identities)
        result = snapshot() if args.action == "inventory" else stop_owned(inventory_fn=snapshot)
        if args.action == "stop":
            cleanup_pid_files(repo=args.root.resolve())
        print(json.dumps(result))
        return 0 if not result.get("unverified") and result.get("ok", True) else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "owned": [], "unverified": [{"error": "Inventory unavailable"}]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
