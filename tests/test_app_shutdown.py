"""When Windows itself shuts down, the app window disappears; the launcher must not start an OFF it can't finish."""
from importlib.machinery import SourceFileLoader
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_launcher(state: Path):
    with mock.patch.dict(os.environ, {"BARRIERWATCHER_STATE": str(state), "BARRIERWATCHER_DRY": "1"}):
        loader = SourceFileLoader("barrierwatcher_app_under_test", str(ROOT / "src" / "barrierwatcher_app.pyw"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    return module


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = load_launcher(Path(self.folder.name))
        mock.patch.object(self.app.time, "sleep").start()

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()

    def log(self):
        try:
            return (Path(self.folder.name) / "app.log").read_text(encoding="utf-8")
        except OSError:
            return ""

    def test_normal_close_is_not_a_shutdown(self):
        with mock.patch.object(self.app, "shutting_down", return_value=False):
            self.assertFalse(self.app.windows_is_ending())
        self.assertEqual("", self.log())

    def test_windows_shutting_down_skips_the_off(self):
        clock = iter(range(0, 1000, 5))
        with mock.patch.object(self.app, "shutting_down", return_value=True), \
                mock.patch.object(self.app.time, "time", side_effect=lambda: next(clock)):
            self.assertTrue(self.app.windows_is_ending())
        self.assertIn("Windows is shutting down", self.log())

    def test_a_cancelled_shutdown_still_switches_off(self):
        answers = iter([True, True, False])
        clock = iter(range(0, 1000, 1))
        with mock.patch.object(self.app, "shutting_down", side_effect=lambda: next(answers)), \
                mock.patch.object(self.app.time, "time", side_effect=lambda: next(clock)):
            self.assertFalse(self.app.windows_is_ending())
        self.assertIn("shutdown was cancelled", self.log())

    def test_real_shutdown_check_runs_on_this_pc(self):
        if os.name != "nt":
            self.skipTest("Windows only")
        self.assertFalse(self.app.shutting_down())      # this PC is not shutting down while tests run


if __name__ == "__main__":
    unittest.main()
