"""Repeated shortcuts reuse the native app; the Chrome fallback opens Settings separately.

Use an explicit app layout, so a native executable installed beside the tests cannot change their meaning.
Every window operation is mocked; the mutex and state directory belong only to this test.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from test_app_shutdown import load_launcher


@unittest.skipUnless(sys.platform == "win32", "Windows only")
class SingleInstanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.state = Path(self.folder.name)
        self.app = load_launcher(self.state)
        self.native_exe = self.state / "BarrierWatcher.exe"
        mock.patch.object(self.app, "APP_EXE", self.native_exe).start()
        mock.patch.object(self.app, "QUIET", False).start()
        # Hold the app's own lock, as a running BarrierWatcher would.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        name = "Local\\BarrierWatcher.App." + hashlib.sha1(str(self.app.STATE).lower().encode()).hexdigest()[:12]
        self.mutex = kernel.CreateMutexW(None, False, name)
        self.kernel = kernel
        self.assertTrue(self.mutex)

    def tearDown(self):
        self.kernel.CloseHandle(ctypes.c_void_p(self.mutex))
        mock.patch.stopall()
        self.folder.cleanup()

    def run_main(self, mode):
        with mock.patch.object(sys, "argv", ["barrierwatcher_app.pyw", mode]):
            return self.app.main()

    def test_icon_brings_the_open_window_forward_instead_of_a_second_one(self):
        forward = mock.patch.object(self.app, "bring_forward").start()
        another = mock.patch.object(self.app, "open_another_window").start()
        mock.patch.object(self.app, "app_windows", return_value=[4242]).start()
        self.assertEqual(0, self.run_main("open"))
        forward.assert_called_once_with(4242)
        another.assert_not_called()
        self.assertIn("brought the app window to the front", (self.state / "app.log").read_text(encoding="utf-8"))

    def test_settings_opens_one_more_window(self):
        forward = mock.patch.object(self.app, "bring_forward").start()
        another = mock.patch.object(self.app, "open_another_window", return_value=[77]).start()
        mock.patch.object(self.app, "app_windows", return_value=[4242]).start()
        self.assertEqual(0, self.run_main("settings"))
        another.assert_called_once_with("/settings")
        forward.assert_not_called()

    def test_settings_reuses_the_existing_native_window(self):
        self.native_exe.write_bytes(b"synthetic executable; never launched")
        forward = mock.patch.object(self.app, "bring_forward").start()
        another = mock.patch.object(self.app, "open_another_window").start()
        mock.patch.object(self.app, "app_windows", return_value=[4242]).start()
        self.assertEqual(0, self.run_main("settings"))
        forward.assert_called_once_with(4242)
        another.assert_not_called()

    def test_no_window_found_opens_one(self):
        another = mock.patch.object(self.app, "open_another_window", return_value=[]).start()
        mock.patch.object(self.app, "app_windows", return_value=[]).start()
        self.assertEqual(0, self.run_main("open"))
        another.assert_called_once_with("/")


if __name__ == "__main__":
    unittest.main()
