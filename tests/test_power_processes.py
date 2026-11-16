"""Guarded full-program inventory/cleanup, using synthetic processes only."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import platform_support as ps


class PowerProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location('power_inventory_test', SRC / 'power_processes.py')
        self.power = importlib.util.module_from_spec(spec)
        with patch.object(ps, 'state_dir', return_value=self.state):
            spec.loader.exec_module(self.power)
        self.identities = [(r'C:\C500\src\watch.py', 'watcher'),
                           (r'C:\C500\tools\ffmpeg\bin\ffmpeg.exe', 'video decoder')]

    def test_only_exact_owned_processes_are_listed(self):
        rows = [{'pid': 101, 'name': 'python.exe'}, {'pid': 102, 'name': 'python.exe'},
                {'pid': 103, 'name': 'notepad.exe'}, {'pid': os.getpid(), 'name': 'python.exe'}]
        opened = []

        @contextlib.contextmanager
        def process_factory(pid):
            opened.append(pid)
            yield Mock(alive=Mock(return_value=True), matches=Mock(return_value=pid == 101))

        result = self.power.inventory(rows, process_factory, self.identities)
        self.assertEqual([row['pid'] for row in result['owned']], [101])
        self.assertEqual(result['unverified'], [])
        self.assertEqual(opened, [101, 102])

    def test_new_app_and_configured_car_memory_are_exact_owned_components(self):
        root = self.state / 'app'
        (root / 'config').mkdir(parents=True)
        car_script = self.state / 'separate car memory' / 'carmem.py'
        (root / 'config' / 'car-memory.json').write_text(json.dumps({'script': str(car_script)}))
        identities = self.power.expected_identities(root, self.state)
        expected = {101: (str(root / 'src' / 'barrierwatcher_app.pyw'), 'app launcher'),
                    102: (str(root / 'app' / 'BarrierWatcher.exe'), 'native app'),
                    103: (str(car_script), 'car-memory updater')}
        rows = [{'pid': 101, 'name': 'pythonw.exe'},
                {'pid': 102, 'name': 'BarrierWatcher.exe'},
                {'pid': 103, 'name': 'pythonw.exe'},
                {'pid': 104, 'name': 'BarrierWatcher.exe'},
                {'pid': 105, 'name': 'pythonw.exe'}]

        @contextlib.contextmanager
        def process_factory(pid):
            yield Mock(alive=Mock(return_value=True),
                       matches=Mock(side_effect=lambda identity: pid in expected and expected[pid][0] == identity))

        result = self.power.inventory(rows, process_factory, identities)
        self.assertEqual({row['pid']: (row['identity'], row['component']) for row in result['owned']}, expected)
        self.assertEqual(result['unverified'], [])

    def test_unreadable_native_app_cannot_make_off_report_complete(self):
        identities = [(r'C:\C500\app\BarrierWatcher.exe', 'native app')]
        result = self.power.inventory([{'pid': 102, 'name': 'BarrierWatcher.exe'}],
            Mock(side_effect=PermissionError('access denied')), identities)
        self.assertEqual(result['owned'], [])
        self.assertEqual(result['unverified'][0]['pid'], 102)
        stopped = self.power.stop_owned(inventory_fn=Mock(return_value=result), stop_fn=Mock(), sleep=Mock())
        self.assertFalse(stopped['ok'])

    def test_malformed_car_memory_identity_prevents_a_false_complete_inventory(self):
        root = self.state / 'app'
        (root / 'config').mkdir(parents=True)
        config = root / 'config' / 'car-memory.json'
        for data in ['{', '{}', '{"script":"relative.py"}', '{"script":"C:/other.exe"}']:
            config.write_text(data)
            with self.subTest(data=data), self.assertRaises((ValueError, OSError)):
                self.power.expected_identities(root, self.state)

    def test_off_stops_launcher_before_its_children_and_verifies_final_inventory(self):
        rows = [
            {'pid': 103, 'identity': r'C:\Separate\carmem.py', 'component': 'car-memory updater'},
            {'pid': 102, 'identity': r'C:\C500\app\BarrierWatcher.exe', 'component': 'native app'},
            {'pid': 101, 'identity': r'C:\C500\src\barrierwatcher_app.pyw', 'component': 'app launcher'},
        ]
        remaining = list(rows)
        order = []

        def stop(pid, identity):
            order.append(pid)
            remaining[:] = [row for row in remaining if row['pid'] != pid]
            return True

        result = self.power.stop_owned(
            inventory_fn=lambda: {'owned': list(remaining), 'unverified': []}, stop_fn=stop, sleep=Mock())
        self.assertEqual(order[0], 101)
        self.assertEqual(set(order), {101, 102, 103})
        self.assertTrue(result['ok'])

    def test_unreadable_candidate_prevents_false_empty_inventory(self):
        for name in ('pythonw.exe', 'go2rtc.exe', 'ffmpeg.exe'):
            with self.subTest(name=name):
                result = self.power.inventory([{'pid': 101, 'name': name}],
                    Mock(side_effect=PermissionError('access denied')), self.identities)
                self.assertEqual(result['owned'], [])
                self.assertEqual(result['unverified'][0]['pid'], 101)

    def test_disappeared_process_is_not_an_unknown_live_process(self):
        result = self.power.inventory([{'pid': 101, 'name': 'python.exe'}],
            Mock(side_effect=ProcessLookupError('gone')), self.identities)
        self.assertEqual(result, {'owned': [], 'unverified': []})

    def test_stop_rechecks_identity_and_final_inventory_is_authoritative(self):
        owned = {'pid': 101, 'identity': self.identities[0][0], 'component': 'watcher'}
        inventory = Mock(side_effect=[{'owned': [owned], 'unverified': []},
                                     {'owned': [], 'unverified': []},
                                     {'owned': [], 'unverified': []}])
        stop = Mock(side_effect=PermissionError('exited during stop'))
        result = self.power.stop_owned(inventory_fn=inventory, stop_fn=stop, sleep=Mock())
        stop.assert_called_once_with(101, self.identities[0][0])
        self.assertTrue(result['ok'])
        self.assertEqual(result['errors'], [])

    def test_unknown_process_is_never_stopped_and_blocks_success(self):
        inventory = Mock(return_value={'owned': [], 'unverified': [{'pid': 101, 'error': 'denied'}]})
        stop = Mock()
        result = self.power.stop_owned(inventory_fn=inventory, stop_fn=stop, sleep=Mock())
        self.assertFalse(result['ok'])
        stop.assert_not_called()

    def test_recycled_and_unreadable_pid_records_are_preserved(self):
        record = self.state / 'watcher.pid'
        for error in (PermissionError('identity mismatch'), PermissionError('denied')):
            record.write_text('101')
            with patch.object(ps, 'process_alive', side_effect=error):
                self.power.cleanup_pid_files(state=self.state, repo=Path('C:/C500'))
            self.assertEqual(record.read_text(), '101')
        with patch.object(ps, 'process_alive', return_value=False) as alive:
            self.power.cleanup_pid_files(state=self.state, repo=Path('C:/C500'))
        self.assertFalse(record.exists())
        self.assertTrue(alive.call_args.kwargs['strict'])


if __name__ == '__main__':
    unittest.main()
