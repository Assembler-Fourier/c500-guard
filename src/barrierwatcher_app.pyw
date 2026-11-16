"""BarrierWatcher as a normal app: open it and everything starts, close it and everything stops.

    pythonw barrierwatcher_app.pyw            open (start the system if needed, then the app window)
    pythonw barrierwatcher_app.pyw settings   the same, opening Settings
    pythonw barrierwatcher_app.pyw off        switch everything off (closes the app window too)

The app window is a Chrome app-mode window inside the user's own Chrome (no second browser in memory);
the launcher follows it by its window handle:
* closing the window (X) switches the whole system off through the "C500 Guard OFF" task, which stops
  every process and checks nothing is left; the launcher then exits;
* if the system goes off another way (F4, the OFF shortcut, the stick pulled) the window closes itself;
* if the window goes because Windows is shutting down, the launcher just exits (Windows stops everything);
* tests (BARRIERWATCHER_QUIET=1) use a private hidden profile tied to the launcher by a job object.
Only BarrierWatcher's own tasks, files and app window are touched. Standard library only.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import platform_support as ps  # noqa: E402  (job object helper; standard library only)

STATE = Path(os.environ.get("BARRIERWATCHER_STATE") or
             Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "c500-guard")
BASE = os.environ.get("BARRIERWATCHER_BASE", "http://127.0.0.1:8787")
SCHTASKS = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "schtasks.exe"
PROFILE = STATE / "app-window"
APP_EXE = ROOT / "app" / "BarrierWatcher.exe"     # BarrierWatcher's own window (WebView2)
ON_TASK, OFF_TASK = "C500 Guard ON", "C500 Guard OFF"
NO_WINDOW = 0x08000000
DRY = os.environ.get("BARRIERWATCHER_DRY") == "1"       # tests: never start or stop the real system
QUIET = os.environ.get("BARRIERWATCHER_QUIET") == "1"   # tests: no small windows, browser hidden
BG, PANEL, TEXT, MUTED, LIME, RED = "#0b0c0b", "#121512", "#eef0e9", "#92998e", "#d8f78a", "#ff896f"


def display_name() -> str:
    try:
        name = (json.loads((ROOT / "config" / "zones.json").read_text(encoding="utf-8-sig")).get("system") or {}).get("name")
        return name if isinstance(name, str) and 0 < len(name.strip()) <= 24 else "BarrierWatcher"
    except (OSError, ValueError, AttributeError):
        return "BarrierWatcher"


NAME = display_name()


def log(message: str) -> None:
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        with (STATE / "app.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} {message}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- system state
def system_state() -> dict | None:
    try:
        with urllib.request.urlopen(BASE + "/state", timeout=1.5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def task_status(task: str) -> str:
    try:
        result = subprocess.run([str(SCHTASKS), "/Query", "/TN", task, "/FO", "CSV", "/NH"], capture_output=True,
                                text=True, timeout=10, creationflags=NO_WINDOW)
        return result.stdout.strip().split('","')[-1].strip('"') if result.returncode == 0 else "Missing"
    except (OSError, subprocess.SubprocessError):
        return "Unknown"


def run_task(task: str) -> bool:
    if DRY:
        log(f"dry run: would start {task}")
        return True
    try:
        return subprocess.run([str(SCHTASKS), "/Run", "/TN", task], capture_output=True, timeout=10,
                              creationflags=NO_WINDOW).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def receipt() -> dict:
    try:
        return json.loads((STATE / "power-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def switched_off() -> bool:
    return (STATE / "power-off").exists()


def shutting_down() -> bool:
    """True while Windows is shutting down, restarting or signing out (SM_SHUTTINGDOWN)."""
    if os.environ.get("BARRIERWATCHER_FAKE_SHUTDOWN") == "1":     # tests
        return True
    try:
        return bool(ctypes.WinDLL("user32").GetSystemMetrics(0x2000))
    except (OSError, AttributeError):
        return False


def windows_is_ending() -> bool:
    """The window went because Windows is shutting down: it stops every process itself, and an OFF started
    now could not finish (seen 14 Nov 21:54). Wait briefly in case the shutdown is cancelled."""
    if not shutting_down():
        return False
    log("Windows is shutting down; it stops everything itself")
    deadline = time.time() + (2 if os.environ.get("BARRIERWATCHER_FAKE_SHUTDOWN") == "1" else 30)
    while time.time() < deadline:
        time.sleep(1.0)
        if not shutting_down():
            log("Windows shutdown was cancelled; switching off as normal")
            return False
    return True


def switch_off(reason: str) -> dict:
    """Run the verified OFF task and wait for it (max 45 s). Returns its receipt."""
    log(f"switching off: {reason}")
    if not run_task(OFF_TASK):
        return {"ok": False, "errors": ["Windows could not start the OFF task."]}
    started = time.time()
    time.sleep(1.0)
    while time.time() - started < 45 and task_status(OFF_TASK) in ("Running", "Queued"):
        time.sleep(0.4)
    result = receipt()
    log(f"off finished: state={result.get('power_state')} ok={result.get('ok')} "
        f"remaining={len(result.get('remaining_processes') or [])}")
    return result


# ---------------------------------------------------------------- the app window (Chrome app mode)
def chrome_path() -> Path | None:
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            for relative in (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe"):
                candidate = Path(base) / relative
                if candidate.is_file():
                    return candidate
    return None


def window_command(page: str) -> list[str]:
    browser = chrome_path()
    if browser is None:
        raise FileNotFoundError("Chrome or Edge is needed for the app window")
    if not QUIET:
        # A window of the Chrome that is already running: about 60 MB instead of ~300 MB for a second
        # browser of its own (measured 14 Nov: 7 extra processes, 314 MB private, with 1 GB RAM free).
        return [str(browser), f"--app={BASE}{page}?app=1"]
    return [str(browser), f"--user-data-dir={PROFILE}", f"--app={BASE}{page}?app=1", "--start-maximized",
            "--no-first-run", "--no-default-browser-check", "--disable-sync", "--disable-extensions",
            "--disable-features=Translate,MediaRouter", "--hide-crash-restore-bubble", "--headless=new"]


def app_windows() -> list[int]:
    """The visible BarrierWatcher app windows: Chrome/Edge windows whose title is the app's own page title.

    A normal browser tab shows "<page> - Google Chrome", an app window only "<page>", so a dashboard
    opened in an ordinary tab is never taken for the app.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    host = BASE.split("://", 1)[-1]

    def callback(hwnd, _param):
        handle = ctypes.c_void_p(hwnd)
        if not user32.IsWindowVisible(handle):
            return True
        name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(handle, name, 64)
        own_window = name.value.startswith("WindowsForms10.Window")           # BarrierWatcher.exe
        if name.value != "Chrome_WidgetWin_1" and not own_window:
            return True
        text = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(handle, text, 256)
        title = text.value
        if title.endswith((" - Google Chrome", " - Microsoft Edge", " - Microsoft​ Edge")):
            return True
        if title == NAME or title.startswith(NAME + " /") or (not own_window and title.startswith(host)):
            found.append(hwnd)
        return True
    user32.EnumWindows(callback_type(callback), None)
    return found


def maximise(hwnds) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    for hwnd in hwnds:
        user32.ShowWindow(ctypes.c_void_p(hwnd), 3)          # SW_MAXIMIZE
        user32.SetForegroundWindow(ctypes.c_void_p(hwnd))


def bring_forward(hwnd: int) -> None:
    """Like clicking a running app's icon: un-minimise it, maximise it and put it in front."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    handle = ctypes.c_void_p(hwnd)
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, 9)                           # SW_RESTORE
    user32.ShowWindow(handle, 3)                               # SW_MAXIMIZE
    user32.SetForegroundWindow(handle)


def open_another_window(page: str) -> list[int]:
    """Open one more app window (e.g. Settings) and maximise it like the first."""
    before = set(app_windows())
    subprocess.Popen(window_command(page), creationflags=NO_WINDOW)
    deadline = time.time() + 15
    while time.time() < deadline:
        time.sleep(0.4)
        new = [hwnd for hwnd in app_windows() if hwnd not in before]
        if new:
            maximise(new)
            return new
    return []


def close_app_windows() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    for hwnd in app_windows():
        user32.PostMessageW(ctypes.c_void_p(hwnd), 0x0010, 0, 0)   # WM_CLOSE, like clicking X


def profile_processes() -> list[int]:
    """PIDs of browser processes using the app's private profile (and nothing else)."""
    try:
        script = ("Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' OR Name='msedge.exe'\" | "
                  "Where-Object { $_.CommandLine -like '*" + str(PROFILE).replace("'", "''") + "*' } | "
                  "ForEach-Object { $_.ProcessId }")
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW)
        return [int(line) for line in result.stdout.split() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def close_windows(pids: set[int]) -> None:
    """Ask the app's own windows to close (WM_CLOSE), like clicking X."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _param):
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(ctypes.c_void_p(hwnd)):
            found.append(hwnd)
        return True
    user32.EnumWindows(callback_type(callback), None)
    for hwnd in found:
        user32.PostMessageW(ctypes.c_void_p(hwnd), 0x0010, 0, 0)


def end_profile_processes(main=None) -> None:
    """Nothing of the app window may be left behind."""
    pids = set(profile_processes())
    if main is not None and main.poll() is None:
        pids.add(main.pid)
    if pids:
        close_windows(pids)
        deadline = time.time() + 6
        while time.time() < deadline and (main is None or main.poll() is None) and profile_processes():
            time.sleep(0.5)
    for pid in profile_processes():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)


# ---------------------------------------------------------------- small native windows
class Splash:
    def __init__(self, title: str, line: str):
        self.root = tk.Tk()
        self.root.title(NAME)
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.iconbitmap(str(ROOT / "assets" / "barrierwatcher.ico"))
        except tk.TclError:
            pass
        width, height = 460, 230
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        frame = tk.Frame(self.root, bg=PANEL, highlightbackground="#2b3229", highlightthickness=1)
        frame.pack(fill="both", expand=True)
        logo = tk.Canvas(frame, width=64, height=64, bg=PANEL, highlightthickness=0)
        logo.place(x=28, y=28)
        s = 64 / 40
        logo.create_rectangle(4 * s, 7 * s, 12 * s, 34 * s, fill=LIME, outline="")
        logo.create_rectangle(2 * s, 32 * s, 14 * s, 37 * s, fill=LIME, outline="")
        logo.create_rectangle(12 * s, 10 * s, 38 * s, 18 * s, fill=TEXT, outline="")
        for left, width_ in ((16.5, 4.5), (25.5, 4.5), (34, 4)):
            logo.create_rectangle(left * s, 10 * s, (left + width_) * s, 18 * s, fill="#ff6b57", outline="")
        logo.create_oval(5.8 * s, 11.8 * s, 10.2 * s, 16.2 * s, fill=PANEL, outline="")
        tk.Label(frame, text=NAME, bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 18)).place(x=108, y=30)
        self.title = tk.Label(frame, text=title, bg=PANEL, fg=LIME, font=("Segoe UI", 11))
        self.title.place(x=110, y=66)
        self.line = tk.Label(frame, text=line, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=400,
                             justify="left")
        self.line.place(x=28, y=120)
        self.buttons = tk.Frame(frame, bg=PANEL)
        self.buttons.place(x=28, y=180)
        self.result = None

    def say(self, title: str, line: str = "", colour: str = LIME) -> None:
        self.root.after(0, lambda: (self.title.config(text=title, fg=colour), self.line.config(text=line)))

    def ask(self, title: str, line: str, choices: list[tuple[str, str]]) -> str:
        """Show a problem with buttons; returns the chosen key."""
        done = threading.Event()

        def build():
            self.title.config(text=title, fg=RED)
            self.line.config(text=line)
            for child in self.buttons.winfo_children():
                child.destroy()
            for key, label in choices:
                tk.Button(self.buttons, text=label, bg="#1c211a", fg=TEXT, activebackground="#2b3229",
                          activeforeground=TEXT, relief="flat", padx=14, pady=4, font=("Segoe UI", 9),
                          command=lambda k=key: (setattr(self, "result", k), done.set())).pack(side="left", padx=(0, 8))
        self.root.after(0, build)
        done.wait()
        return self.result

    def close(self) -> None:
        self.root.after(0, self.root.destroy)


class _Silent:
    def say(self, *args, **kwargs):
        pass

    def ask(self, title, line, choices):
        log(f"would ask: {title}: {line}")
        return "close"


def run_with_splash(title: str, line: str, work) -> object:
    if QUIET:
        return work(_Silent())
    splash = Splash(title, line)
    outcome = {}

    def worker():
        try:
            outcome["value"] = work(splash)
        except Exception as exc:                                  # noqa: BLE001
            log(f"error: {type(exc).__name__}: {exc}")
            outcome["value"] = None
        splash.close()
    threading.Thread(target=worker, daemon=True).start()
    splash.root.mainloop()
    return outcome.get("value")


# ---------------------------------------------------------------- the three actions
def start_system(splash: Splash) -> bool:
    """True when the system is up and armed (or at least answering)."""
    while True:
        if system_state() is not None:
            return True
        splash.say("Starting…", "Checking the camera, AI and helpers. This usually takes 20–40 seconds.")
        started = time.time()
        if not run_task(ON_TASK):
            choice = splash.ask("Could not start", "Windows could not start the power task. It may need repair.",
                                [("retry", "Try again"), ("close", "Close")])
            if choice == "retry":
                continue
            return False
        time.sleep(4)
        while time.time() - started < 150:
            state = system_state()
            if state is not None and state.get("armed"):
                return True
            if time.time() - started > 8 and task_status(ON_TASK) not in ("Running", "Queued"):
                if state is not None:
                    return True
                break
            time.sleep(1)
        info = receipt()
        problem = "; ".join(info.get("errors") or []) or "It did not answer in time."
        log(f"start failed: {problem}")
        choice = splash.ask("Did not start",
                            f"{problem}\nCheck the capture stick is plugged in and the keyboard shows the ANPR "
                            "Entrance camera.", [("retry", "Try again"), ("close", "Close")])
        if choice != "retry":
            return False


def watch_exe_window(page: str) -> bool:
    """BarrierWatcher's own window (BarrierWatcher.exe): follow its process until it closes or the system goes off.
    False means it could not start (for example the WebView2 engine is missing): the Chrome window is used instead."""
    url = f"{BASE}/?app=1" + ("#settings" if page == "/settings" else "")
    try:
        window = subprocess.Popen([str(APP_EXE), "--url", url])
    except OSError as exc:
        log(f"own window could not start: {exc}")
        return False
    ps.tie_to_this_process(window)
    started = time.time()
    log(f"window open (BarrierWatcher.exe pid {window.pid}) {page}")
    car_memory = start_car_memory()
    away_since = None
    try:
        while True:
            time.sleep(1.0)
            code = window.poll()
            if code is not None:
                if code != 0 and time.time() - started < 15:
                    log(f"own window ended at once (code {code}); using the Chrome window instead")
                    return False
                if windows_is_ending():
                    return True
                log("window closed by the user")
                stop_car_memory(car_memory)
                if system_state() is not None or task_status(ON_TASK) == "Running":
                    run_with_splash("Switching off…", "Stopping the camera, AI, alarms and helpers.",
                                    lambda splash: switch_off("app window closed"))
                return True
            if system_state() is not None:
                away_since = None
                continue
            away_since = away_since or time.time()
            if time.time() - away_since > 8 and switched_off():
                log("system was switched off elsewhere; closing the app window")
                window.terminate()
                return True
    finally:
        stop_car_memory(car_memory)


def watch_window(page: str) -> None:
    """Open the app window and keep the app and the system together until either ends."""
    if not QUIET:
        if APP_EXE.is_file() and watch_exe_window(page):
            return None
        return watch_shared_window(page)
    PROFILE.mkdir(parents=True, exist_ok=True)
    window = subprocess.Popen(window_command(page), creationflags=NO_WINDOW)
    ps.tie_to_this_process(window)
    log(f"window open (pid {window.pid}) {page}")
    away_since = None
    while True:
        time.sleep(1.0)
        if window.poll() is not None:
            time.sleep(1.5)
            if profile_processes():           # another app window of ours is still open
                continue
            if windows_is_ending():
                return
            log("window closed by the user")
            if system_state() is not None or task_status(ON_TASK) == "Running":
                run_with_splash("Switching off…", "Stopping the camera, AI, alarms and helpers.",
                                lambda splash: switch_off("app window closed"))
            end_profile_processes()
            return
        state = system_state()
        if state is not None:
            away_since = None
            continue
        away_since = away_since or time.time()
        if time.time() - away_since > 8 and switched_off():
            log("system was switched off elsewhere; closing the app window")
            end_profile_processes(window)
            return


def start_car_memory():
    """The separate car memory updates itself while the app is open and closes with it."""
    if DRY or QUIET:
        return None
    try:
        config = json.loads((ROOT / "config" / "car-memory.json").read_text(encoding="utf-8"))
        script = Path(config["script"])
        python = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        if not script.is_file() or not python.is_file():
            return None
        process = subprocess.Popen([str(python), "-I", str(script), "watch"], cwd=str(script.parent),
                                   creationflags=NO_WINDOW | 0x00004000)      # below-normal priority
        ps.tie_to_this_process(process)
        log(f"car memory updating (pid {process.pid})")
        return process
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log(f"car memory not started: {exc}")
        return None


def stop_car_memory(process) -> None:
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(5)
        except (OSError, subprocess.SubprocessError):
            pass


def watch_shared_window(page: str) -> None:
    """The app window lives in the user's own Chrome; follow it by its window handle and title."""
    before = set(app_windows())
    subprocess.Popen(window_command(page), creationflags=NO_WINDOW)
    deadline = time.time() + 25
    windows = []
    while time.time() < deadline:
        time.sleep(0.5)
        windows = app_windows()
        if windows:
            break
    if not windows:
        log("the app window did not appear; leaving the system as it is")
        return
    maximise([hwnd for hwnd in windows if hwnd not in before] or windows)
    log(f"window open (shared browser, {len(windows)} window(s)) {page}")
    car_memory = start_car_memory()
    try:
        follow_windows(windows, before_off=lambda: stop_car_memory(car_memory))
    finally:
        stop_car_memory(car_memory)


def follow_windows(windows, before_off=lambda: None) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    # Follow the windows themselves, not their titles: a reload, an error page during a refresh or a
    # renamed app changes the title but is still the same open app window.
    tracked = set(windows)
    away_since = None
    while True:
        time.sleep(1.0)
        tracked = {hwnd for hwnd in tracked if user32.IsWindow(ctypes.c_void_p(hwnd))} | set(app_windows())
        if not tracked:
            if windows_is_ending():
                return
            log("window closed by the user")
            before_off()
            if system_state() is not None or task_status(ON_TASK) == "Running":
                run_with_splash("Switching off…", "Stopping the camera, AI, alarms and helpers.",
                                lambda splash: switch_off("app window closed"))
            return
        if system_state() is not None:
            away_since = None
            continue
        away_since = away_since or time.time()
        if time.time() - away_since > 8 and switched_off():
            log("system was switched off elsewhere; closing the app window")
            for hwnd in tracked:
                user32.PostMessageW(ctypes.c_void_p(hwnd), 0x0010, 0, 0)    # WM_CLOSE
            return


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "open").lower()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    import hashlib
    lock = "Local\\BarrierWatcher.App." + hashlib.sha1(str(STATE).lower().encode()).hexdigest()[:12]
    mutex = kernel.CreateMutexW(None, False, lock)        # one app per state folder (tests use their own)
    already = ctypes.get_last_error() == 183
    log(f"start mode={mode} already_running={already}")
    if mode == "off":
        run_with_splash("Switching off…", "Stopping the camera, AI, alarms and helpers.",
                        lambda splash: switch_off("OFF shortcut"))
        return 0
    page = "/settings" if mode == "settings" else "/"
    if already:
        # The app is already open. Like any app: its icon brings the open window to the front; Settings
        # opens as one more (maximised) window. The running launcher follows every app window.
        windows = [] if QUIET else app_windows()
        if windows and (page == "/" or APP_EXE.is_file()):      # own window: Settings is inside it
            bring_forward(windows[0])
            log("already open: brought the app window to the front")
        else:
            opened = [] if QUIET else open_another_window(page)
            if QUIET:
                subprocess.Popen(window_command(page), creationflags=NO_WINDOW)
            log(f"already open: opened another window {page} ({len(opened)} found)")
        return 0
    if not run_with_splash("Starting…", "Checking whether it is already running.", start_system):
        log("not started")
        return 1
    watch_window(page)
    log("app closed")
    if mutex:
        kernel.CloseHandle(ctypes.c_void_p(mutex))
    return 0


if __name__ == "__main__":
    sys.exit(main())
