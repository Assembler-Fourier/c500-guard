#!/usr/bin/env python3
"""Manage the local go2rtc bridge that turns the Xiaomi camera into RTSP.

Downloads the official go2rtc binary for this platform, verifies it against a
pinned SHA-256, and runs it bound to loopback only with mandatory auth.

It never prints, stores or transmits Xiaomi credentials. You log in once, in
your own browser, on the machine itself.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import platform_support as ps

VERSION = "v1.9.14"
BASE = f"https://github.com/AlexxIT/go2rtc/releases/download/{VERSION}"

# Pinned digests of the official release assets. A download that does not match
# is discarded, so a tampered or truncated file can never be executed.
ASSETS = {
    "windows-amd64": ("go2rtc_win64.zip",
                      "dd4167d75cb04abe618855b7c71f8658bd009f60c1a71835d134d2c11c939907",
                      "go2rtc.exe"),
    "macos-arm64":   ("go2rtc_mac_arm64.zip",
                      "919b78adc759d6b3883d1e1b2ac915ac0985bb903ff1897b4d228527bd64690c",
                      "go2rtc"),
    "macos-x86_64":  ("go2rtc_mac_amd64.zip",
                      "9b0b9a27a4dc3a5b8b93376e7e8fc2787c6af624a512842622be84aec0171c7a",
                      "go2rtc"),
    "linux-x86_64":  ("go2rtc_linux_amd64",
                      "32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6",
                      None),
}

STATE = ps.state_dir()
POWER_OFF = STATE / "power-off"
BIN_DIR = STATE / "bin"
CONFIG = STATE / "go2rtc.yaml"
LOGIN = STATE / "bridge-login.json"
PID_FILE = STATE / "bridge.pid"
LOG = STATE / "bridge.log"
CAMERA = STATE / "camera.json"
SHIM_PID = STATE / "lan-shim.pid"
SHIM_LOG = STATE / "lan-shim.log"
URL = "http://127.0.0.1:1984"


def platform_key() -> str:
    import platform
    machine = platform.machine().lower()
    if ps.WINDOWS:
        return "windows-amd64"
    if ps.MACOS:
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
    return "linux-x86_64"


def binary_path() -> Path:
    return BIN_DIR / ("go2rtc.exe" if ps.WINDOWS else "go2rtc")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def fetch() -> Path:
    """Download and verify go2rtc. Idempotent: returns early if already valid."""
    target = binary_path()
    local_build = BIN_DIR / "go2rtc-build.json"
    if local_build.exists() and target.is_file():
        expected = json.loads(local_build.read_text())["sha256"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit("Local camera bridge checksum mismatch; refusing to use it.")
        print("Verified local go2rtc build with Xiaomi command-queue fix.")
        return target
    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"go2rtc already installed at {target}")
        return target
    key = platform_key()
    if key not in ASSETS:
        raise SystemExit(f"No pinned go2rtc build for {key}.")
    name, digest, member = ASSETS[key]
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ({VERSION}) ...")
    try:
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=180) as response:
            blob = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"Download failed: {exc}")
    actual = hashlib.sha256(blob).hexdigest()
    if actual != digest:
        raise SystemExit("SHA-256 MISMATCH - refusing to install.\n"
                         f"  expected {digest}\n  got      {actual}")
    print(f"SHA-256 verified: {actual[:16]}...")
    if member:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            target.write_bytes(archive.read(member))
    else:
        target.write_bytes(blob)
    if not ps.WINDOWS:
        os.chmod(target, 0o755)
    print(f"Installed {target}")
    return target


def init() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    ps.harden(STATE)
    if CONFIG.exists() and LOGIN.exists():
        print("Existing private configuration preserved.")
        return
    if CONFIG.exists() or LOGIN.exists():
        raise SystemExit("Partial configuration found. Review it before continuing; "
                         "refusing to overwrite credentials.")
    auth = {"username": "c500-local", "password": secrets.token_urlsafe(24)}
    config = {
        "app": {"modules": ["api", "ws", "rtsp", "mp4", "xiaomi"]},
        "api": {"listen": "127.0.0.1:1984", **auth, "local_auth": True,
                "allow_paths": ["/", "/api", "/api/xiaomi", "/api/streams",
                                "/api/ws", "/api/frame.mp4", "/api/stream.mp4"]},
        "rtsp": {"listen": "127.0.0.1:8554"},
        "log": {"level": "info"},
        "streams": {},
    }
    write_yaml(config, {})
    LOGIN.write_text(json.dumps(auth) + "\n")
    ps.harden(LOGIN)
    print("Private configuration created. No Xiaomi account is connected yet.")


def write_yaml(config: dict, xiaomi: dict) -> None:
    """Emit real YAML.

    go2rtc appends its own YAML when you log in, so the file must never be left
    as a JSON object - appending YAML to JSON produces a file that go2rtc can
    read but not save to, which silently loses your streams on restart.
    """
    def emit(obj, indent=0):
        pad = "  " * indent
        out = []
        for key, value in obj.items():
            k = json.dumps(str(key))
            if isinstance(value, dict):
                # An empty mapping is omitted, never written as "{}".
                # go2rtc patches this file by splicing text: it can append a
                # top-level key that is missing, but it silently fails to
                # insert into an existing empty flow mapping. Writing
                # "streams: {}" here loses every stream you add, at the next
                # restart, with no error anywhere. See runbook 4.4.
                if not value:
                    continue
                out.append(f"{pad}{k}:")
                out.append(emit(value, indent + 1))
            elif isinstance(value, list):
                out.append(f"{pad}{k}: " + json.dumps(value))
            elif isinstance(value, bool):
                out.append(f"{pad}{k}: {'true' if value else 'false'}")
            else:
                out.append(f"{pad}{k}: " + json.dumps(value))
        return "\n".join(out)

    document = emit(config)
    if xiaomi:
        document += "\n" + emit({"xiaomi": xiaomi})
    # LF explicitly. Path.write_text() emits CRLF on Windows while
    # go2rtc appends LF, leaving a half-and-half file for a text-
    # splicing patcher. Not the cause of 4.4, but no reason to risk it.
    CONFIG.write_bytes((document + "\n").encode())
    ps.harden(CONFIG)


def repair_config() -> bool:
    """Fix a config left as JSON-plus-appended-YAML by go2rtc. Returns True if changed."""
    if not CONFIG.exists():
        return False
    raw = CONFIG.read_text()
    if not raw.lstrip().startswith("{"):
        return False
    import re
    config, end = json.JSONDecoder().raw_decode(raw)
    tail = raw[end:].strip()
    xiaomi = {}
    match = re.match(r'^xiaomi:\s*\n\s*"?([0-9]+)"?:\s*(.+)\s*$', tail, re.S)
    if match:
        xiaomi[match.group(1)] = match.group(2).strip()
    elif tail:
        print("Config has an unrecognised trailing block; leaving it alone.")
        return False
    backup = CONFIG.with_suffix(f".yaml.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(raw)
    ps.harden(backup)
    write_yaml(config, xiaomi)
    print(f"Repaired config into valid YAML (backup: {backup.name}).")
    return True


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def auth_header() -> str:
    auth = json.loads(LOGIN.read_text())
    token = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
    return "Basic " + token


def request(path: str, authenticated: bool = True):
    headers = {"Authorization": auth_header()} if authenticated else {}
    req = urllib.request.Request(URL + path, headers=headers)
    with urllib.request.urlopen(req, timeout=6) as response:
        return json.load(response)


def running_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if ps.process_alive(pid, str(binary_path()), strict=True) else None


def port_listening(port: int) -> bool:
    """True only if something is actually accepting connections.

    Deliberately not a bind() test: after a stop, the socket lingers in
    TIME_WAIT, so bind() fails while nothing is listening. Treating that as
    "occupied" makes a restart impossible for ~30 seconds.
    """
    probe = socket.socket()
    probe.settimeout(1.0)
    try:
        probe.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def wait_for_ports(timeout: float = 90.0) -> bool:
    """Wait for listeners to exit; TIME_WAIT is not a running server."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        free = not any(port_listening(port) for port in (1984, 8554))
        if free:
            return True
        time.sleep(1)
    return False


def shim_pid() -> int | None:
    try:
        pid = int(SHIM_PID.read_text().strip())
    except (OSError, ValueError):
        return None
    script = Path(__file__).resolve().parent / 'lan_shim.py'
    return pid if ps.process_alive(pid, str(script), strict=True) else None


def start_shim() -> None:
    """Run the LAN discovery relay, if this install needs one.

    Written by `add-stream --lan-shim`. Absent on cameras that answer a plain
    unicast search, in which case nothing here runs at all.
    """
    if POWER_OFF.exists() or not CAMERA.exists():
        return
    if shim_pid():
        return
    try:
        camera = json.loads(CAMERA.read_text())["ip"]
    except (OSError, ValueError, KeyError):
        print(f"Ignoring unreadable {CAMERA}; no LAN relay started.")
        return
    script = Path(__file__).resolve().parent / "lan_shim.py"
    ps.rotate_log(SHIM_LOG)
    if POWER_OFF.exists():
        return
    with SHIM_LOG.open("ab") as handle:
        child = ps.popen_detached(
            [sys.executable, str(script), "--camera", camera], handle)
    SHIM_PID.write_text(str(child.pid) + chr(10))
    ps.harden(SHIM_PID)
    if POWER_OFF.exists():
        stop_shim()
        return
    time.sleep(0.5)
    if child.poll() is not None:
        raise SystemExit(f"LAN relay exited immediately. See {SHIM_LOG}")
    print(f"LAN relay started for camera {camera}.")


def stop_shim() -> None:
    pid = shim_pid()
    if not pid:
        return
    try:
        ps.stop_process_tree(pid, str(Path(__file__).resolve().parent / 'lan_shim.py'))
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not stop LAN relay pid {pid}: {exc}")
        return
    SHIM_PID.unlink(missing_ok=True)
    print("LAN relay stopped.")


def start() -> None:
    if POWER_OFF.exists():
        print("The whole program is OFF. Open the BarrierWatcher app (desktop or Start menu) to start it.")
        return
    init()
    repair_config()
    if running_pid():
        print("Bridge is already running.")
        start_shim()
        return status()
    if port_listening(1984):
        # Another go2rtc (or another app) already owns the port. Starting a
        # second one would just fail after a long wait, so say so plainly.
        print("Port 1984 is already served by another process.")
        try:
            request("/api/xiaomi")
            print("It answers our credentials, so it is a compatible bridge - reusing it.")
            start_shim()
        except urllib.error.HTTPError:
            print("It does NOT accept our credentials. Stop that process, or this "
                  "install will not be able to reach the camera.")
        except (OSError, urllib.error.URLError) as exc:
            print(f"It does not answer the go2rtc API ({exc}). Stop it before continuing.")
        return status()
    if not binary_path().is_file():
        fetch()
    if POWER_OFF.exists():
        return
    if not wait_for_ports():
        raise SystemExit("Ports 1984/8554 still busy after 90s. Something else is using them.")
    if POWER_OFF.exists():
        return
    ps.rotate_log(LOG)
    with LOG.open("ab") as handle:
        child = ps.popen_detached([str(binary_path()), "-config", str(CONFIG)], handle)
    PID_FILE.write_text(str(child.pid) + "\n")
    ps.harden(PID_FILE)
    for _ in range(100):
        if POWER_OFF.exists():
            stop()
            return
        if child.poll() is not None:
            raise SystemExit(f"go2rtc exited immediately. See {LOG}")
        try:
            request("/api/xiaomi")
            print("Bridge started (loopback only).")
            start_shim()
            if POWER_OFF.exists():
                stop()
                return
            return status()
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    raise SystemExit("Bridge started but never became healthy.")


def stop() -> None:
    stop_shim()
    pid = running_pid()
    if not pid:
        print("Bridge is not running.")
        return
    try:
        ps.stop_process_tree(pid, str(binary_path()))
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Could not stop pid {pid}: {exc}")
    PID_FILE.unlink(missing_ok=True)
    print("Bridge stopped. Saved Xiaomi login is preserved.")


def status() -> None:
    pid = running_pid()
    if not pid:
        if port_listening(1984):
            try:
                accounts, streams = request("/api/xiaomi"), request("/api/streams")
                report = {"bridge_running": True, "pid": None,
                                  "started_by": "another process",
                                  "xiaomi_accounts": len(accounts),
                                  "streams": list(streams.keys())}
                if CAMERA.exists():
                    report["lan_relay_pid"] = shim_pid()
                print(json.dumps(report, indent=2))
                return
            except (OSError, urllib.error.URLError):
                pass
            print(json.dumps({"bridge_running": False,
                              "port_1984_listening_but_not_go2rtc": True}))
            return
        print(json.dumps({"bridge_running": False}))
        return
    try:
        accounts, streams = request("/api/xiaomi"), request("/api/streams")
    except (OSError, urllib.error.URLError) as exc:
        print(json.dumps({"bridge_running": True, "pid": pid, "api_error": str(exc)}))
        return
    report = {"bridge_running": True, "pid": pid,
              "xiaomi_accounts": len(accounts),
              "streams": list(streams.keys())}
    if CAMERA.exists():
        report["lan_relay_pid"] = shim_pid()
    print(json.dumps(report, indent=2))


def security_test() -> None:
    failures = []
    for path in ("/api/xiaomi", "/api/streams"):
        try:
            request(path, authenticated=False)
            failures.append(f"unauthenticated access allowed: {path}")
        except urllib.error.HTTPError as error:
            if error.code != 401:
                failures.append(f"expected 401 on {path}, got {error.code}")
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"could not reach {path}: {exc}")
    for path in ("/api/config", "/api/log", "/api/restart"):
        try:
            request(path)
            failures.append(f"management endpoint reachable: {path}")
        except urllib.error.HTTPError as error:
            if error.code not in (403, 404):
                failures.append(f"unexpected {error.code} on {path}")
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"could not reach {path}: {exc}")
    if failures:
        for line in failures:
            print("FAIL:", line)
        raise SystemExit(1)
    print("PASS: unauthenticated requests denied; config/log/restart blocked.")


def login_details() -> None:
    print("LOCAL BRIDGE LOGIN ONLY - this is NOT your Xiaomi password.")
    auth = json.loads(LOGIN.read_text())
    print("  URL     :", URL + "/add.html")
    print("  Username:", auth["username"])
    print("  Password:", auth["password"])


def add_stream(name: str, src: str, lan_shim: bool = False) -> None:
    import urllib.parse
    if lan_shim:
        parts = urllib.parse.urlsplit(src)
        host = parts.hostname or ""
        if not host:
            raise SystemExit("--lan-shim needs a camera address in --src")
        CAMERA.write_text(json.dumps({"ip": host}) + chr(10))
        ps.harden(CAMERA)
        userinfo = src.split("@", 1)[0].split("//", 1)[1]
        query = src.split("?", 1)[1] if "?" in src else ""
        src = f"xiaomi://{userinfo}@127.0.0.1" + (f"?{query}" if query else "")
        print(f"LAN relay recorded for {host}; go2rtc will be pointed at 127.0.0.1.")
        start_shim()
    query = urllib.parse.urlencode({"name": name, "src": src})
    req = urllib.request.Request(f"{URL}/api/streams?{query}", method="PUT",
                                 headers={"Authorization": auth_header()})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        if error.code != 400:            # 400 = added, but config save complained
            raise SystemExit(f"Could not add stream: {error}")
    # go2rtc may have written a JSON+YAML hybrid; normalise it so it persists.
    stop_needed = repair_config()
    print(f"Stream '{name}' added." + (" Restart the bridge to persist it."
                                       if stop_needed else ""))
    # go2rtc answers 400 when it accepted the stream but could not save the
    # config, and we deliberately ignore that above. Ignoring it silently is
    # how a camera survives until the next restart and then vanishes, so
    # confirm against the file rather than trusting the reply.
    if CONFIG.exists() and f"\n  {name}:" not in CONFIG.read_text():
        print(f"WARNING: '{name}' is running but was NOT saved to {CONFIG}.")
        print("         It will disappear on the next restart. Runbook 4.4.")
    status()


def main() -> None:
    actions = {"install": fetch, "init": init, "start": start, "stop": stop,
               "status": status, "security-test": security_test,
               "login-details": login_details, "repair-config": repair_config}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=list(actions) + ["add-stream"])
    parser.add_argument("--name", default="c500")
    parser.add_argument("--src", default="")
    parser.add_argument("--lan-shim", action="store_true",
                        help="camera only answers a broadcast search; relay through lan_shim.py")
    args = parser.parse_args()
    if args.action == "add-stream":
        if not args.src:
            raise SystemExit("--src is required (copy the xiaomi:// URL from /add.html)")
        add_stream(args.name, args.src, args.lan_shim)
    else:
        actions[args.action]()


if __name__ == "__main__":
    main()
