"""Startup regression tests with an isolated state directory and no camera I/O."""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / "src"


class StopLoop(Exception):
    pass


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        fake_ps = types.ModuleType("platform_support")
        fake_ps.WINDOWS = sys.platform == "win32"
        fake_ps.OS_NAME = "test"
        fake_ps.state_dir = lambda: self.state
        fake_ps.summary = lambda: {
            "python": "3.12", "ffmpeg": "ffmpeg", "audio_ok": True, "audio": "mock"}
        fake_ps.popen_detached = Mock()
        fake_ps.rotate_log = Mock()
        fake_ps.process_alive = Mock(return_value=False)
        self.modules = patch.dict(sys.modules, {"platform_support": fake_ps})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location("ctl", SRC / "ctl.py")
        self.ctl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ctl)
        spec = importlib.util.spec_from_file_location("service", SRC / "service.py")
        self.service = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"ctl": self.ctl}):
            spec.loader.exec_module(self.service)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_missing_watcher_is_a_doctor_failure(self):
        healthy = {"bridge_running": True, "streams": ["c500"]}
        with patch.object(self.ctl, "bridge_status", return_value=healthy), \
             patch.object(self.ctl, "watcher_state", return_value=None), \
             patch.object(Path, "is_file", return_value=True), \
             patch.dict(sys.modules, {"numpy": types.ModuleType("numpy"),
                                      "PIL": types.ModuleType("PIL")}):
            self.assertEqual(self.ctl.cmd_doctor(None), 1)

    def test_failed_bridge_prevents_watcher_launch(self):
        with patch.object(self.ctl, "bridge", return_value=
                          subprocess.CompletedProcess([], 1, "", "failed")):
            self.assertEqual(self.ctl.cmd_start(argparse.Namespace(source=self.ctl.DEFAULT_SOURCE)), 1)
        self.ctl.ps.popen_detached.assert_not_called()

    def test_zero_exit_with_unhealthy_bridge_or_missing_relay_is_failure(self):
        for state in ({"bridge_running": False},
                      {"bridge_running": True, "api_error": "unavailable"},
                      {"bridge_running": True, "lan_relay_pid": None}):
            with self.subTest(state=state), \
                 patch.object(self.ctl, "bridge", return_value=
                              subprocess.CompletedProcess([], 0, "", "")), \
                 patch.object(self.ctl, "bridge_status", return_value=state):
                self.assertFalse(self.ctl.ensure_bridge())

    def test_capture_source_never_starts_the_camera_bridge(self):
        with patch.object(self.ctl, "bridge") as bridge,              patch.object(self.ctl, "watcher_state", side_effect=[None, {"armed": False}]),              patch.object(self.ctl.time, "sleep"):
            self.ctl.ps.popen_detached.return_value = Mock(pid=4321, poll=Mock(return_value=None))
            self.assertEqual(self.ctl.cmd_start(argparse.Namespace(source="usbcam:12ab:3456")), 0)
        bridge.assert_not_called()
        command = self.ctl.ps.popen_detached.call_args[0][0]
        self.assertIn("usbcam:12ab:3456", command)

    def test_configured_source_and_bridge_need(self):
        self.assertEqual(self.ctl.configured_source(), self.ctl.DEFAULT_SOURCE
                         if not self.ctl.SOURCE_FILE.is_file() else self.ctl.configured_source())
        self.assertTrue(self.ctl.needs_bridge("rtsp://127.0.0.1:8554/c500"))
        self.assertFalse(self.ctl.needs_bridge("usbcam:12ab:3456"))
        self.assertFalse(self.ctl.needs_bridge("rtsp://192.0.2.5:554/Streaming/Channels/101"))
        with tempfile.TemporaryDirectory() as folder:
            source_file = Path(folder) / "source.json"
            with patch.object(self.ctl, "SOURCE_FILE", source_file):
                self.assertEqual(self.ctl.configured_source(), self.ctl.DEFAULT_SOURCE)
                source_file.write_text('{"source": "usbcam:12ab:3456"}', encoding="utf-8")
                self.assertEqual(self.ctl.configured_source(), "usbcam:12ab:3456")
                source_file.write_text('not json', encoding="utf-8")
                self.assertEqual(self.ctl.configured_source(), self.ctl.DEFAULT_SOURCE)

    def test_supervisor_with_capture_source_does_not_touch_the_bridge(self):
        with patch.object(self.ctl, "configured_source", return_value="usbcam:12ab:3456"),              patch.object(self.ctl, "watcher_state", return_value={"armed": True, "loop_age_s": .1}),              patch.object(self.ctl, "ensure_bridge") as bridge,              patch.object(self.service.time, "sleep", side_effect=StopLoop):
            with self.assertRaises(StopLoop):
                self.service.supervise()
        bridge.assert_not_called()

    def test_refresh_request_starts_a_fresh_watcher_after_the_old_one_closes(self):
        (self.ctl.STATE / "restart-request").write_text("now")
        with patch.object(self.ctl, "watcher_state", side_effect=[{"armed": True}, None]),              patch.object(self.ctl, "cmd_start", return_value=0) as start,              patch.object(self.service.time, "sleep"):
            self.assertTrue(self.service.restart_requested())
            status = self.service.finish_refresh(wait_s=5)
        start.assert_called_once()
        self.assertIn("Refreshed", status)
        self.assertFalse(self.service.restart_requested())

    def test_supervisor_honours_saved_stop(self):
        self.ctl.PAUSED.write_text("paused")
        with patch.object(self.ctl, "cmd_start") as start, \
             patch.object(self.ctl, "ensure_bridge") as bridge, \
             patch.object(self.service.time, "sleep", side_effect=StopLoop):
            with self.assertRaises(StopLoop):
                self.service.supervise()
        start.assert_not_called()
        bridge.assert_not_called()

    def test_service_does_not_clear_stop_requested_during_bridge_start(self):
        def stop_during_start():
            self.ctl.PAUSED.write_text("paused")
            return True
        with patch.object(self.ctl, "ensure_bridge", side_effect=stop_during_start):
            self.assertEqual(self.ctl.cmd_start(
                argparse.Namespace(source=self.ctl.DEFAULT_SOURCE, resume=False)), 0)
        self.assertTrue(self.ctl.PAUSED.exists())
        self.ctl.ps.popen_detached.assert_not_called()

    def test_existing_disarmed_watcher_remains_disarmed(self):
        self.ctl.PAUSED.write_text("paused")
        with patch.object(self.ctl, "ensure_bridge", return_value=True), \
             patch.object(self.ctl, "watcher_state", return_value={"armed": False}), \
             patch.object(self.ctl, "api") as api:
            self.assertEqual(self.ctl.cmd_start(argparse.Namespace(source="test")), 0)
        self.assertFalse(self.ctl.PAUSED.exists())
        self.ctl.ps.popen_detached.assert_not_called()
        api.assert_not_called()

    def test_process_lock_prevents_duplicate_start(self):
        with self.ctl.process_lock(self.ctl.START_LOCK) as first:
            self.assertTrue(first)
            with self.ctl.process_lock(self.ctl.START_LOCK) as second:
                self.assertFalse(second)
        with self.ctl.process_lock(self.ctl.START_LOCK) as released:
            self.assertTrue(released)

    def test_child_exit_fails_promptly(self):
        child = Mock(pid=1234, returncode=2)
        child.poll.return_value = 2
        self.ctl.ps.popen_detached.return_value = child
        with patch.object(self.ctl, "ensure_bridge", return_value=True), \
             patch.object(self.ctl, "watcher_state", return_value=None), \
             patch.object(self.ctl.time, "sleep") as sleep:
            self.assertEqual(self.ctl.cmd_start(argparse.Namespace(source="test")), 1)
        sleep.assert_not_called()
        self.assertFalse(self.ctl.WATCHER_PID.exists())


if __name__ == "__main__":
    unittest.main()
