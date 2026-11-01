"""Regression coverage for background Windows helpers creating console windows."""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / "src"
NO_WINDOW = 0x08000000


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackgroundProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.ps = load_module("background_platform_support", "platform_support.py")
        # ctl/bridge import-time state must never touch the actual deployment.
        with patch.object(self.ps, "state_dir", return_value=self.state), \
             patch.dict(sys.modules, {"platform_support": self.ps}):
            self.ctl = load_module("background_ctl", "ctl.py")
            self.bridge = load_module("background_bridge", "bridge.py")
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_console_flag_is_windows_only(self):
        with patch.object(self.ps, "WINDOWS", True):
            self.assertEqual(self.ps.background_subprocess_kwargs(), {"creationflags": NO_WINDOW})
        with patch.object(self.ps, "WINDOWS", False):
            self.assertEqual(self.ps.background_subprocess_kwargs(), {})

    def test_native_health_probe_spawns_no_console_helper(self):
        process = Mock()
        process.alive.return_value = True
        process.matches.return_value = True
        with patch.object(self.ps, "WINDOWS", True), \
             patch.object(self.ps, "_WindowsProcess") as native, \
             patch.object(self.ps.subprocess, "run") as run:
            native.return_value.__enter__.return_value = process
            self.assertTrue(self.ps.process_alive(1234, 'go2rtc'))
        native.assert_called_once_with(1234)
        process.matches.assert_called_once_with('go2rtc')
        run.assert_not_called()

    def test_supervisor_bridge_helpers_keep_capture_and_suppress_console(self):
        result = subprocess.CompletedProcess([], 0, '{"bridge_running":true}', "")
        with patch.object(self.ps, "WINDOWS", True), \
             patch.object(self.ctl.subprocess, "run", return_value=result) as run:
            self.assertTrue(self.ctl.ensure_bridge())
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ["start", "status"])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["creationflags"], NO_WINDOW)
            self.assertTrue(call.kwargs["capture_output"])
            self.assertTrue(call.kwargs["text"])
            self.assertEqual(call.kwargs["timeout"], 180)

    def test_all_windows_stop_helpers_suppress_console(self):
        self.ctl.WATCHER_PID.write_text("1234")
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(self.ps, "WINDOWS", True), \
             patch.object(self.ps, "_WindowsProcess") as native, \
             patch.object(self.ps.subprocess, "run", return_value=result) as run, \
             patch.object(self.bridge, "shim_pid", return_value=1235), \
             patch.object(self.bridge, "running_pid", return_value=1236):
            native.return_value.__enter__.return_value.alive.return_value = True
            native.return_value.__enter__.return_value.matches.return_value = True
            self.assertEqual(self.ctl.stop_watcher(), 0)
            self.bridge.stop()
        self.assertEqual(len(run.call_args_list), 3)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], "taskkill")
            self.assertEqual(call.kwargs["creationflags"], NO_WINDOW)
            self.assertTrue(call.kwargs["capture_output"])

    def test_alarm_helper_suppresses_console_without_losing_exit_status(self):
        with patch.object(self.ps, "WINDOWS", True), \
             patch.object(self.ctl.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 7)) as run:
            self.assertEqual(self.ctl.cmd_test_alarm(argparse.Namespace(seconds=1)), 7)
        self.assertEqual(run.call_args.kwargs["creationflags"], NO_WINDOW)

    def test_background_children_keep_logs_without_conflicting_flags(self):
        log = io.BytesIO()
        with patch.object(self.ps, "WINDOWS", True), \
             patch.object(self.ps.subprocess, "Popen") as popen:
            self.ps.popen_detached(["python.exe", "helper.py"], log)
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], NO_WINDOW)
        self.assertIs(kwargs["stdout"], log)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("start_new_session", kwargs)

    def test_posix_background_children_keep_independent_session(self):
        with patch.object(self.ps, "WINDOWS", False), \
             patch.object(self.ps.subprocess, "Popen") as popen:
            self.ps.popen_detached(["python", "helper.py"], io.BytesIO())
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("creationflags", popen.call_args.kwargs)

    @unittest.skipUnless(sys.platform == "win32", "Windows console API")
    def test_real_windows_helper_has_no_console_and_retains_stdout(self):
        result = subprocess.run(
            [sys.executable, "-c", "import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())"],
            capture_output=True, text=True, timeout=10,
            **self.ps.background_subprocess_kwargs())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")

    @unittest.skipUnless(sys.platform == "win32", "Windows console API")
    def test_real_background_child_has_no_console_and_retains_log(self):
        logfile = self.state / "synthetic-helper.log"
        with logfile.open("wb") as log:
            child = self.ps.popen_detached(
                [sys.executable, "-c", "import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())"], log)
            self.assertEqual(child.wait(timeout=10), 0)
        self.assertEqual(logfile.read_text().strip(), "0")


if __name__ == "__main__":
    unittest.main()
