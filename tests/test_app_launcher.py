"""BarrierWatcher app lifecycle, end to end with a fake server, a hidden browser and a temporary profile.

Opt-in (starts real browser processes for ~1 minute): set BARRIERWATCHER_APP_TESTS=1.
Never touches the real system: dry-run mode only logs what it would start.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "barrierwatcher_app.pyw"
BASE_PYTHON = Path(getattr(sys, "_base_executable", sys.executable))


class FakeWatcher(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"armed": true}' if self.path.startswith("/state") else b"<!doctype html><title>BW</title>ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def profile_pids(profile: Path) -> list[int]:
    script = ("Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' OR Name='msedge.exe'\" | Where-Object "
              "{ $_.CommandLine -like '*" + str(profile) + "*' } | ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout
    return [int(line) for line in out.split() if line.strip().isdigit()]


@unittest.skipUnless(os.environ.get("BARRIERWATCHER_APP_TESTS") == "1" and sys.platform == "win32",
                     "set BARRIERWATCHER_APP_TESTS=1 to run the app lifecycle tests")
class AppLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.state = Path(self.folder.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeWatcher)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        env = {**os.environ, "BARRIERWATCHER_DRY": "1", "BARRIERWATCHER_QUIET": "1",
               "BARRIERWATCHER_STATE": str(self.state),
               "BARRIERWATCHER_BASE": f"http://127.0.0.1:{self.server.server_address[1]}"}
        self.app = subprocess.Popen([str(BASE_PYTHON), str(APP)], env=env)

    def tearDown(self):
        if self.app.poll() is None:
            self.app.kill()
        for pid in profile_pids(self.state / "app-window"):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        try:
            self.server.shutdown()
            self.server.server_close()
        except OSError:
            pass
        time.sleep(1)
        self.folder.cleanup()

    def log(self):
        try:
            return (self.state / "app.log").read_text(encoding="utf-8")
        except OSError:
            return ""

    def wait_for(self, text, seconds=40):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if text in self.log():
                return
            time.sleep(0.3)
        self.fail(f"never logged {text!r}:\n{self.log()}")

    def window_pid(self):
        line = next(l for l in self.log().splitlines() if "window open (pid" in l)
        return int(line.split("window open (pid ")[1].split(")")[0])

    def test_closing_the_window_switches_everything_off_and_leaves_nothing(self):
        self.wait_for("window open (pid")
        subprocess.run(["taskkill", "/PID", str(self.window_pid()), "/T", "/F"], capture_output=True)
        self.wait_for("app closed")
        log = self.log()
        self.assertIn("window closed by the user", log)
        self.assertIn("dry run: would start C500 Guard OFF", log)
        self.app.wait(10)
        self.assertEqual(0, self.app.returncode)
        self.assertEqual([], profile_pids(self.state / "app-window"))

    def test_window_closes_itself_when_the_system_is_switched_off_elsewhere(self):
        self.wait_for("window open (pid")
        self.server.shutdown()
        self.server.server_close()
        (self.state / "power-off").write_text("off", encoding="utf-8")
        self.wait_for("switched off elsewhere", 30)
        self.wait_for("app closed")
        self.assertNotIn("would start C500 Guard OFF", self.log())
        self.app.wait(10)
        self.assertEqual([], profile_pids(self.state / "app-window"))


if __name__ == "__main__":
    unittest.main()
