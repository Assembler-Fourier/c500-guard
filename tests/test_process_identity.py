"""PID reuse, command-line identity, guarded stops and real Windows checks."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / 'src'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProcessIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.ps = load('identity_platform', 'platform_support.py')
        with patch.object(self.ps, 'state_dir', return_value=self.state), \
             patch.dict(sys.modules, {'platform_support': self.ps}):
            self.ctl = load('identity_ctl', 'ctl.py')
            self.bridge = load('identity_bridge', 'bridge.py')
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_exact_executable_matching_rejects_substrings_and_arguments(self):
        matches = self.ps._windows_identity_matches
        self.assertTrue(matches(r'C:\App\go2rtc.exe', [], r'c:/app/GO2RTC.exe'))
        self.assertTrue(matches(r'C:\App\go2rtc.exe', [], 'go2rtc'))
        self.assertFalse(matches(r'C:\Other\go2rtc.exe', [], r'C:\App\go2rtc.exe'))
        self.assertFalse(matches(r'C:\App\not-go2rtc.exe', [], 'go2rtc'))
        self.assertFalse(matches(r'C:\Windows\notepad.exe', [r'C:\App\go2rtc.exe'], 'go2rtc'))

    def test_python_identity_requires_interpreter_and_exact_script_position(self):
        matches = self.ps._windows_identity_matches
        script = r'C:\App with spaces\src\watch.py'
        self.assertTrue(matches(r'C:\Python\python.exe', ['python', script, 'watch'], script))
        self.assertTrue(matches(r'C:\Python\pythonw.exe', ['pythonw', script], script))
        for image, args in [
                (r'C:\Windows\notepad.exe', ['notepad', script]),
                (r'C:\Python\python.exe', ['python', '-c', script]),
                (r'C:\Python\python.exe', ['python', '-m', 'other', script]),
                (r'C:\Python\python.exe', ['python', r'C:\Other\watch.py', script]),
                (r'C:\Python\python.exe', ['python', script + '.backup']),
                (r'C:\Python\python.exe', ['python', 'watch.py'])]:
            with self.subTest(image=image, args=args):
                self.assertFalse(matches(image, args, script))

    def test_windowless_and_isolated_python_identity_require_the_actual_script(self):
        matches = self.ps._windows_identity_matches
        image = r'C:\Python\pythonw.exe'
        launcher, updater = r'C:\C500\src\barrierwatcher_app.pyw', r'C:\Separate\car memory\carmem.py'
        self.assertTrue(matches(image, ['pythonw', launcher, 'open'], launcher))
        self.assertTrue(matches(image, ['pythonw', '-I', updater, 'watch'], updater))
        self.assertTrue(matches(image, ['pythonw', '-I', launcher], launcher))
        for args in [['pythonw', '-c', updater], ['pythonw', '-I', '-c', updater],
                     ['pythonw', '-m', 'other', updater], ['pythonw', '-I', 'other.py', updater],
                     ['pythonw', '-I', updater + '.backup'], ['pythonw', '-I']]:
            with self.subTest(args=args):
                self.assertFalse(matches(image, args, updater))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows native process APIs')
    def test_real_isolated_windowless_script_has_verified_identity(self):
        script = self.state / 'isolated child.pyw'
        script.write_text('import time\ntime.sleep(60)\n')
        child = subprocess.Popen([sys.executable, '-I', str(script)],
                                 **self.ps.background_subprocess_kwargs())
        try:
            self.assertTrue(self.ps.process_alive(child.pid, str(script), strict=True))
            self.assertTrue(self.ps.stop_process_tree(child.pid, str(script)))
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_invalid_pid_never_opens_a_process(self):
        with patch.object(self.ps, '_WindowsProcess') as native:
            for pid in (-1, 0, True, '1234', 2**32):
                self.assertFalse(self.ps.process_alive(pid))
                with self.assertRaises(ValueError):
                    self.ps.process_alive(pid, 'watch.py', strict=True)
        native.assert_not_called()

    def test_unknown_identity_fails_closed_and_strict_mode_explains_failure(self):
        with patch.object(self.ps, 'WINDOWS', True), \
             patch.object(self.ps, '_WindowsProcess', side_effect=PermissionError('access denied')):
            self.assertFalse(self.ps.process_alive(1234, 'watch.py'))
            with self.assertRaises(PermissionError):
                self.ps.process_alive(1234, 'watch.py', strict=True)
        with patch.object(self.ps, 'WINDOWS', True), \
             patch.object(self.ps, '_WindowsProcess', side_effect=ProcessLookupError()):
            self.assertFalse(self.ps.process_alive(1234, 'watch.py', strict=True))

    def test_identity_mismatch_is_not_a_dead_process(self):
        with patch.object(self.ps, 'WINDOWS', True), patch.object(self.ps, '_WindowsProcess') as native:
            process = native.return_value.__enter__.return_value
            process.alive.return_value = True
            process.matches.return_value = False
            self.assertFalse(self.ps.process_alive(1234, 'watch.py'))
            with self.assertRaises(PermissionError):
                self.ps.process_alive(1234, 'watch.py', strict=True)

    def test_cross_bitness_command_line_query_is_refused_before_native_layout_read(self):
        import ctypes
        from ctypes import wintypes
        process = self.ps._WindowsProcess(1234)
        process.ctypes, process.wt, process.handle = ctypes, wintypes, 2
        process.kernel = Mock()
        process.kernel.GetCurrentProcess.return_value = 1

        def wow(handle, pointer):
            ctypes.cast(pointer, ctypes.POINTER(wintypes.BOOL))[0] = handle == 1
            return True

        process.kernel.IsWow64Process.side_effect = wow
        with self.assertRaisesRegex(PermissionError, 'architecture'):
            process._arguments()

    def test_verified_handle_stays_open_through_taskkill(self):
        opened = []
        process = Mock()
        process.alive.return_value = True
        process.matches.return_value = True

        @contextlib.contextmanager
        def held(pid):
            opened.append(pid)
            try:
                yield process
            finally:
                opened.remove(pid)

        def stop(command, **kwargs):
            self.assertEqual(opened, [1234])
            self.assertEqual(command, ['taskkill', '/PID', '1234', '/T', '/F'])
            self.assertEqual(kwargs['creationflags'], 0x08000000)
            process.matches.assert_called_once_with(r'C:\App\watch.py')
            return subprocess.CompletedProcess(command, 0)

        with patch.object(self.ps, 'WINDOWS', True), \
             patch.object(self.ps, '_WindowsProcess', side_effect=held), \
             patch.object(self.ps.subprocess, 'run', side_effect=stop):
            self.assertTrue(self.ps.stop_process_tree(1234, r'C:\App\watch.py'))
        self.assertEqual(opened, [])

    def test_stop_mismatch_or_unknown_never_runs_taskkill_and_retains_pid(self):
        self.ctl.WATCHER_PID.write_text('1234')
        for error in (PermissionError('identity mismatch'), PermissionError('access denied')):
            with self.subTest(error=error), \
                 patch.object(self.ps, 'stop_process_tree', side_effect=error), \
                 patch.object(self.ps.subprocess, 'run') as run:
                self.assertEqual(self.ctl.stop_watcher(), 1)
                self.assertEqual(self.ctl.WATCHER_PID.read_text(), '1234')
                run.assert_not_called()

    def test_guarded_stop_rejects_mismatch_before_spawn(self):
        with patch.object(self.ps, 'WINDOWS', True), \
             patch.object(self.ps, '_WindowsProcess') as native, \
             patch.object(self.ps.subprocess, 'run') as run:
            native.return_value.__enter__.return_value.alive.return_value = True
            native.return_value.__enter__.return_value.matches.return_value = False
            with self.assertRaises(PermissionError):
                self.ps.stop_process_tree(1234, r'C:\App\watch.py')
        run.assert_not_called()

    def test_failed_taskkill_keeps_record_for_retry(self):
        self.ctl.WATCHER_PID.write_text('1234')
        with patch.object(self.ps, 'WINDOWS', True), \
             patch.object(self.ps, '_WindowsProcess') as native, \
             patch.object(self.ps.subprocess, 'run', return_value=subprocess.CompletedProcess([], 5)):
            native.return_value.__enter__.return_value.alive.return_value = True
            native.return_value.__enter__.return_value.matches.return_value = True
            self.assertEqual(self.ctl.stop_watcher(), 1)
        self.assertTrue(self.ctl.WATCHER_PID.exists())

    def test_watcher_start_refuses_unknown_or_unresponsive_recorded_process(self):
        self.ctl.WATCHER_PID.write_text('1234')
        with patch.object(self.ctl, 'ensure_bridge', return_value=True), \
             patch.object(self.ctl, 'watcher_state', return_value=None), \
             patch.object(self.ps, 'popen_detached') as spawn:
            for result in (True, PermissionError('identity mismatch'), PermissionError('access denied')):
                with patch.object(self.ps, 'process_alive', side_effect=[result]):
                    self.assertEqual(self.ctl.start_watcher(argparse.Namespace(source='test')), 1)
            spawn.assert_not_called()

    def test_bridge_and_relay_health_require_exact_identity(self):
        self.bridge.PID_FILE.write_text('1234')
        self.bridge.SHIM_PID.write_text('1235')
        with patch.object(self.ps, 'process_alive', return_value=True) as alive:
            self.assertEqual(self.bridge.running_pid(), 1234)
            alive.assert_called_with(1234, str(self.bridge.binary_path()), strict=True)
            self.assertEqual(self.bridge.shim_pid(), 1235)
            alive.assert_called_with(1235, str(SRC / 'lan_shim.py'), strict=True)

    def test_unknown_relay_does_not_spawn_duplicate(self):
        self.bridge.CAMERA.write_text(json.dumps({'ip': '192.0.2.1'}))
        self.bridge.SHIM_PID.write_text('1235')
        with patch.object(self.ps, 'process_alive', side_effect=PermissionError('unknown identity')), \
             patch.object(self.ps, 'popen_detached') as spawn:
            with self.assertRaises(PermissionError):
                self.bridge.start_shim()
        spawn.assert_not_called()
        self.assertTrue(self.bridge.SHIM_PID.exists())

    def test_failed_start_stops_verified_tree_instead_of_launcher_only(self):
        child = Mock(pid=1234)
        child.poll.return_value = None
        with patch.object(self.ctl, 'ensure_bridge', return_value=True), \
             patch.object(self.ctl, 'watcher_state', return_value=None), \
             patch.object(self.ctl.time, 'sleep'), \
             patch.object(self.ps, 'popen_detached', return_value=child), \
             patch.object(self.ps, 'stop_process_tree', return_value=True) as stop:
            self.assertEqual(self.ctl.start_watcher(argparse.Namespace(source='test')), 1)
        stop.assert_called_once_with(1234, str(SRC / 'watch.py'))
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        self.assertFalse(self.ctl.WATCHER_PID.exists())

    def test_failed_start_cleanup_retains_pid_when_stop_is_not_verified(self):
        child = Mock(pid=1234)
        child.poll.return_value = None
        with patch.object(self.ctl, 'ensure_bridge', return_value=True), \
             patch.object(self.ctl, 'watcher_state', return_value=None), \
             patch.object(self.ctl.time, 'sleep'), \
             patch.object(self.ps, 'popen_detached', return_value=child), \
             patch.object(self.ps, 'stop_process_tree', side_effect=PermissionError('unknown process')):
            self.assertEqual(self.ctl.start_watcher(argparse.Namespace(source='test')), 1)
        self.assertEqual(self.ctl.WATCHER_PID.read_text().strip(), '1234')
        child.terminate.assert_not_called()

    def test_start_that_becomes_ready_at_timeout_boundary_is_retained(self):
        child = Mock(pid=1234)
        child.poll.return_value = None
        with patch.object(self.ctl, 'ensure_bridge', return_value=True), \
             patch.object(self.ctl, 'watcher_state', side_effect=[None]*61 + [{'armed': False}]), \
             patch.object(self.ctl.time, 'sleep'), \
             patch.object(self.ps, 'popen_detached', return_value=child), \
             patch.object(self.ps, 'stop_process_tree') as stop:
            self.assertEqual(self.ctl.start_watcher(argparse.Namespace(source='test')), 0)
        stop.assert_not_called()
        self.assertTrue(self.ctl.WATCHER_PID.exists())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows native process APIs')
    def test_real_process_identity_and_guarded_stop(self):
        script = self.state / 'identity child.py'
        script.write_text('import ctypes,json,os,time\n'
                          'print(json.dumps({"pid":os.getpid(),"console":ctypes.windll.kernel32.GetConsoleWindow()}),flush=True)\n'
                          'time.sleep(60)\n')
        child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 **self.ps.background_subprocess_kwargs())
        try:
            self.assertTrue(self.ps.process_alive(child.pid, str(script), strict=True))
            self.assertFalse(self.ps.process_alive(child.pid, str(script) + '.other.py'))
            self.assertFalse(self.ps.process_alive(0xffffffff))
            with self.assertRaises(PermissionError):
                self.ps.stop_process_tree(child.pid, str(script) + '.other.py')
            self.assertIsNone(child.poll())
            details = json.loads(child.stdout.readline())
            self.assertEqual(details['console'], 0)
            self.assertTrue(self.ps.stop_process_tree(child.pid, str(script)))
            child.wait(timeout=5)
            self.assertFalse(self.ps.process_alive(child.pid))
            self.assertFalse(self.ps.process_alive(details['pid']))
        finally:
            if child.poll() is None:
                self.ps.stop_process_tree(child.pid, str(script))
                child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()


if __name__ == '__main__':
    unittest.main()
