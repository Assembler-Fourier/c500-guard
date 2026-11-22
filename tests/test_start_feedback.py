"""A failed start keeps the app window so it can say why; the start window shows fixable problems live (21 Nov 2026).
Synthetic processes, a fake watcher state and a dry-run launcher only."""
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import platform_support as ps
from test_app_shutdown import load_launcher

ROWS = [
    {'pid': 101, 'identity': r'C:\C500\src\barrierwatcher_app.pyw', 'component': 'app launcher'},
    {'pid': 102, 'identity': r'C:\C500\app\BarrierWatcher.exe', 'component': 'native app'},
    {'pid': 103, 'identity': r'C:\C500\src\watch.py', 'component': 'watcher'},
    {'pid': 104, 'identity': r'C:\C500\src\hotkey_win.py', 'component': 'input helper'},
]


class FailedStartCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        spec = importlib.util.spec_from_file_location('power_start_feedback_test', SRC / 'power_processes.py')
        self.power = importlib.util.module_from_spec(spec)
        with mock.patch.object(ps, 'state_dir', return_value=Path(self.temp.name)):
            spec.loader.exec_module(self.power)

    def run_stop(self, keep):
        remaining, stopped = list(ROWS), []

        def stop(pid, identity):
            stopped.append(pid)
            remaining[:] = [row for row in remaining if row['pid'] != pid]
            return True
        snapshot = lambda: {'owned': list(remaining), 'unverified': []}
        inventory = (lambda: self.power.without_launcher(snapshot())) if keep else snapshot
        result = self.power.stop_owned(inventory_fn=inventory, stop_fn=stop, sleep=mock.Mock())
        return result, stopped, remaining

    def test_failed_start_keeps_only_the_launcher_window(self):
        result, stopped, remaining = self.run_stop(keep=True)
        self.assertTrue(result['ok'])
        self.assertEqual(sorted(stopped), [102, 103, 104], 'Camera, AI, hooks and the app page still stop')
        self.assertEqual([row['component'] for row in remaining], ['app launcher'])

    def test_normal_off_still_stops_the_launcher_first(self):
        result, stopped, remaining = self.run_stop(keep=False)
        self.assertTrue(result['ok'])
        self.assertEqual(stopped[0], 101)
        self.assertEqual(remaining, [])

    def test_keep_launcher_flag_only_hides_the_launcher_from_the_inventory(self):
        with mock.patch.object(self.power, 'inventory', return_value={'owned': list(ROWS), 'unverified': []}), \
                mock.patch.object(self.power, 'expected_identities', return_value=[]), \
                mock.patch.object(self.power.ps, 'WINDOWS', True), \
                mock.patch('builtins.print') as printed:
            for argv, expected in ((['inventory', '--keep-launcher'], {102, 103, 104}),
                                   (['inventory'], {101, 102, 103, 104})):
                with mock.patch.object(sys, 'argv', ['power_processes.py', *argv]):
                    self.power.main()
                import json
                owned = {row['pid'] for row in json.loads(printed.call_args.args[0])['owned']}
                self.assertEqual(owned, expected)

    def test_power_script_keeps_the_launcher_only_after_a_failed_start(self):
        script = (SRC / 'power_win.ps1').read_text(encoding='utf-8-sig')
        self.assertEqual(script.count('Stop-FullStack -KeepLauncher'), 1)
        failed = script.split('} catch {\n', 1)[1] if '} catch {\n' in script else script.split('} catch {\r\n', 1)[1]
        self.assertIn("Stop-FullStack -KeepLauncher", failed.split('} finally {', 1)[0])
        self.assertRegex(script, r"if \(\$Action -eq 'Off'\) \{ Stop-FullStack \}")
        self.assertIn("$extra = @('--keep-launcher')", script)


class StartupHintTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.app = load_launcher(Path(self.folder.name))

    def test_keyboard_menu_tells_the_operator_what_to_do(self):
        title, line, colour = self.app.startup_hint({'frames': 50, 'alignment': {'state': 'keyboard_menu'}})
        self.assertEqual(title, 'Make the camera full-screen')
        self.assertIn('Live View menu bar', line)
        self.assertIn('carries on by itself', line)
        self.assertEqual(colour, self.app.RED)

    def test_other_states_keep_the_normal_wording(self):
        self.assertIn('first picture', self.app.startup_hint({'frames': 0, 'alignment': {}})[1])
        self.assertEqual(self.app.startup_hint(None)[:2], self.app.STARTING)
        self.assertEqual(self.app.startup_hint({'frames': 9, 'alignment': {'state': 'tracking'}})[:2], self.app.STARTING)

    def test_start_window_updates_live_and_still_explains_a_failed_start(self):
        said, asked = [], []
        splash = mock.Mock(say=lambda *args: said.append(args), ask=lambda t, l, c: asked.append((t, l)) or 'close')
        states = iter([None] + [{'frames': 30, 'alignment': {'state': 'keyboard_menu'}}] * 3 + [None] * 200)
        clock = iter(range(0, 2000, 3))
        with mock.patch.object(self.app, 'system_state', side_effect=lambda: next(states)), \
                mock.patch.object(self.app, 'run_task', return_value=True), \
                mock.patch.object(self.app, 'task_status', side_effect=['Running'] * 4 + ['Ready'] * 200), \
                mock.patch.object(self.app.time, 'time', side_effect=lambda: next(clock)), \
                mock.patch.object(self.app.time, 'sleep'), \
                mock.patch.object(self.app, 'receipt', return_value={'errors': ['the keyboard menu bar is across the picture']}):
            self.assertFalse(self.app.start_system(splash))
        self.assertIn('Make the camera full-screen', [entry[0] for entry in said])
        self.assertEqual(asked[0][0], 'Did not start')
        self.assertIn('keyboard menu bar', asked[0][1])


if __name__ == '__main__':
    unittest.main()
