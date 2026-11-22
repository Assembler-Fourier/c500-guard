"""The app window's title bar is drawn dark instead of white (21 Nov 2026). Fake DWM calls only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from test_app_shutdown import load_launcher


@unittest.skipUnless(sys.platform == "win32", "Windows only")
class DarkTitleBarTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.app = load_launcher(Path(self.folder.name))

    def test_dark_mode_caption_text_and_border_colours_are_set(self):
        calls = []

        class Dwm:
            def DwmSetWindowAttribute(self, hwnd, attribute, value, size):
                calls.append((hwnd.value, attribute, value._obj.value, size))
                return 0
        with mock.patch.object(self.app.ctypes, "WinDLL", return_value=Dwm()):
            self.assertTrue(self.app.dark_title_bar(4242))
        self.assertEqual([c[1] for c in calls], [20, 35, 36, 34])
        self.assertEqual(calls[0][2], 1)                          # dark mode on
        self.assertEqual(calls[1][2], 0x09 | (0x0c << 8) | (0x10 << 16))   # COLORREF of #090c10
        self.assertTrue(all(c[0] == 4242 and c[3] == 4 for c in calls))

    def test_failure_is_reported_and_harmless(self):
        class Dwm:
            def DwmSetWindowAttribute(self, *args):
                return -2147024809                                # E_INVALIDARG: an older Windows
        with mock.patch.object(self.app.ctypes, "WinDLL", return_value=Dwm()):
            self.assertFalse(self.app.dark_title_bar(1))
        with mock.patch.object(self.app.ctypes, "WinDLL", side_effect=OSError("no dwmapi")):
            self.assertFalse(self.app.dark_title_bar(1))

    def test_each_new_app_window_is_styled_once(self):
        windows, styled_calls = [[11], [11, 12], [11, 12]], []
        with mock.patch.object(self.app, "app_windows", side_effect=windows), \
                mock.patch.object(self.app, "dark_title_bar", side_effect=lambda hwnd: styled_calls.append(hwnd)):
            styled = set()
            for _ in windows:
                self.app.style_new_windows(styled)
        self.assertEqual(styled_calls, [11, 12])


if __name__ == "__main__":
    unittest.main()
