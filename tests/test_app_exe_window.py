"""BarrierWatcher's own window (BarrierWatcher.exe) replaces the Chrome app window; Chrome stays as a fallback."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_app_shutdown import load_launcher


class FakeWindow:
    def __init__(self, codes):
        self.codes = list(codes)
        self.pid = 4321
        self.terminated = False

    def poll(self):
        return self.codes.pop(0) if self.codes else None

    def terminate(self):
        self.terminated = True


class ExeWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = load_launcher(Path(self.folder.name) / "state")
        exe = Path(self.folder.name) / "BarrierWatcher.exe"
        exe.write_bytes(b"")
        mock.patch.object(self.app, "APP_EXE", exe).start()
        mock.patch.object(self.app.time, "sleep").start()
        mock.patch.object(self.app.ps, "tie_to_this_process").start()
        self.splash = mock.patch.object(self.app, "run_with_splash").start()
        mock.patch.object(self.app, "start_car_memory", return_value=None).start()
        self.shared = mock.patch.object(self.app, "watch_shared_window").start()

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()

    def run_window(self, window, page="/", **patches):
        popen = mock.patch.object(self.app.subprocess, "Popen", return_value=window).start()
        for name, value in patches.items():
            mock.patch.object(self.app, name, value).start()
        self.app.watch_window(page)
        return popen

    def test_opens_the_own_window_and_closing_it_switches_off(self):
        popen = self.run_window(FakeWindow([None, None, 0]), system_state=lambda: {"armed": True},
                                windows_is_ending=lambda: False)
        self.assertEqual([str(self.app.APP_EXE), "--url", "http://127.0.0.1:8787/?app=1"], popen.call_args.args[0])
        self.splash.assert_called_once()                       # the OFF, with its splash
        self.shared.assert_not_called()

    def test_settings_opens_on_the_settings_page_inside_the_window(self):
        popen = self.run_window(FakeWindow([0]), page="/settings", system_state=lambda: None,
                                task_status=lambda task: "Ready", windows_is_ending=lambda: False)
        self.assertTrue(popen.call_args.args[0][-1].endswith("/?app=1#settings"))

    def test_a_window_that_cannot_start_falls_back_to_chrome(self):
        self.run_window(FakeWindow([3]))
        self.shared.assert_called_once_with("/")
        self.splash.assert_not_called()

    def test_switched_off_elsewhere_closes_the_window(self):
        window = FakeWindow([None] * 30)
        clock = iter(range(0, 1000, 5))
        self.run_window(window, system_state=lambda: None, switched_off=lambda: True)
        self.assertTrue(window.terminated)
        self.splash.assert_not_called()

    def test_windows_shutting_down_needs_no_off(self):
        self.run_window(FakeWindow([None, 0]), windows_is_ending=lambda: True, system_state=lambda: {"armed": True})
        self.splash.assert_not_called()

    def test_the_icon_brings_the_own_window_forward_even_for_settings(self):
        mock.patch.object(self.app, "app_windows", return_value=[777]).start()
        forward = mock.patch.object(self.app, "bring_forward").start()
        another = mock.patch.object(self.app, "open_another_window").start()
        import ctypes
        import hashlib
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        name = "Local\\BarrierWatcher.App." + hashlib.sha1(str(self.app.STATE).lower().encode()).hexdigest()[:12]
        mutex = kernel.CreateMutexW(None, False, name)
        try:
            with mock.patch.object(self.app.sys, "argv", ["barrierwatcher_app.pyw", "settings"]):
                self.assertEqual(0, self.app.main())
        finally:
            kernel.CloseHandle(ctypes.c_void_p(mutex))
        forward.assert_called_once_with(777)
        another.assert_not_called()


if __name__ == "__main__":
    unittest.main()
