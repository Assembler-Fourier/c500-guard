"""The car memory inside the app: page route, "Cars" link, and the updater's life tied to the app window."""
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request
from unittest import mock

from test_watch import watch, watcher
from test_app_shutdown import load_launcher
import dashboard_ui


class CarMemoryPageTests(unittest.TestCase):
    def setUp(self):
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.server = watch.start_web(self.obj, 0)
        self.url = 'http://127.0.0.1:' + str(self.server.server_address[1])
        self.report = watch.STATE_DIR / 'car-memory' / 'report.html'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.report.unlink(missing_ok=True)
        mock.patch.stopall()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=3) as response:
            return response.read().decode('utf-8')

    def test_waiting_page_until_the_car_memory_has_written_one(self):
        self.report.unlink(missing_ok=True)
        self.assertIn('The first local history report is being prepared.', self.get('/car-memory'))
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text('<!doctype html><title>Car memory</title>3 in', encoding='utf-8')
        self.assertIn('3 in', self.get('/car-memory'))

    def test_cars_page_is_inside_the_app(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        self.assertIn('data-page="vehicles"', page)
        self.assertIn('data-src="/car-memory"', page)


class LauncherCarMemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = load_launcher(Path(self.folder.name) / 'state')

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()

    def test_nothing_starts_in_tests_or_without_its_config(self):
        self.assertIsNone(self.app.start_car_memory())                  # DRY
        mock.patch.object(self.app, 'DRY', False).start()
        mock.patch.object(self.app, 'ROOT', Path(self.folder.name)).start()
        self.assertIsNone(self.app.start_car_memory())                  # no config/car-memory.json

    def test_starts_below_normal_tied_to_the_app(self):
        root = Path(self.folder.name)
        (root / 'config').mkdir()
        (root / '.venv' / 'Scripts').mkdir(parents=True)
        (root / '.venv' / 'Scripts' / 'pythonw.exe').write_bytes(b'')
        script = root / 'car-memory' / 'carmem.py'
        script.parent.mkdir()
        script.write_text('# fake', encoding='utf-8')
        (root / 'config' / 'car-memory.json').write_text(json.dumps({'script': str(script)}), encoding='utf-8')
        mock.patch.object(self.app, 'DRY', False).start()
        mock.patch.object(self.app, 'ROOT', root).start()
        popen = mock.patch.object(self.app.subprocess, 'Popen').start()
        tie = mock.patch.object(self.app.ps, 'tie_to_this_process').start()
        process = self.app.start_car_memory()
        args, kwargs = popen.call_args
        self.assertEqual([str(root / '.venv' / 'Scripts' / 'pythonw.exe'), '-I', str(script), 'watch'], args[0])
        self.assertTrue(kwargs['creationflags'] & 0x00004000)            # below normal
        tie.assert_called_once_with(process)

    def test_updater_stops_before_the_system_is_switched_off(self):
        order = []
        mock.patch.object(self.app.time, 'sleep').start()
        mock.patch.object(self.app, 'app_windows', return_value=[]).start()
        mock.patch.object(self.app, 'windows_is_ending', return_value=False).start()
        mock.patch.object(self.app, 'system_state', side_effect=lambda: order.append('state') or {'armed': True}).start()
        mock.patch.object(self.app, 'run_with_splash', side_effect=lambda *a: order.append('off')).start()
        self.app.follow_windows([], before_off=lambda: order.append('car memory stopped'))
        self.assertEqual(['car memory stopped', 'state', 'off'], order)


if __name__ == '__main__':
    unittest.main()
