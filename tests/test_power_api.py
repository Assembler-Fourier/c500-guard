import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test_watch import watch, watcher
import power_control


class PowerDispatchTests(unittest.TestCase):
    def test_fixed_task_runs_windowless_without_shell_or_controller_child(self):
        with mock.patch.object(power_control.ps, 'WINDOWS', True), mock.patch.object(power_control.subprocess, 'run') as run:
            run.return_value.returncode = 0
            power_control.request_full_off()
        self.assertEqual(run.call_args.args[0][1:], ['/Run', '/TN', 'C500 Guard OFF'])
        self.assertNotIn('shell', run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs['timeout'], 8)
        if power_control.ps.WINDOWS:
            self.assertTrue(run.call_args.kwargs['creationflags'] & subprocess.CREATE_NO_WINDOW)

    def test_rejected_and_uncertain_dispatches_are_not_success(self):
        for failure in (None, OSError('unavailable'), subprocess.TimeoutExpired('schtasks', 8)):
            with self.subTest(failure=failure), mock.patch.object(power_control.ps, 'WINDOWS', True), mock.patch.object(power_control.subprocess, 'run') as run:
                run.return_value.returncode = 1
                run.side_effect = failure
                with self.assertRaises(RuntimeError):
                    power_control.request_full_off()

    def test_full_off_marker_prevents_direct_watcher_start(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(watch, 'STATE_DIR', Path(directory)), mock.patch.object(watch, 'Watcher') as build:
            (Path(directory) / 'power-off').touch()
            watch.cmd_watch(mock.Mock())
            build.assert_not_called()

    def test_off_during_constructor_prevents_capture_and_server_start(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(watch, 'STATE_DIR', Path(directory)), mock.patch.object(watch, 'start_web') as server:
            obj = watcher()
            obj.reader = mock.Mock()
            (Path(directory) / 'power-off').touch()
            obj.run()
            obj.reader.start.assert_not_called()
            server.assert_not_called()


class PowerAPITests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.server = watch.start_web(self.obj, 0)
        self.url = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, via='web', origin=None):
        headers = {'X-Via': via}
        if origin:
            headers['Origin'] = origin
        return urllib.request.urlopen(urllib.request.Request(self.url + '/power/off', data=b'', headers=headers), timeout=3)

    def test_success_is_only_shutdown_requested_and_silences_alarm(self):
        with mock.patch.object(watch, 'request_full_off') as dispatch:
            with self.post() as response:
                self.assertEqual(response.status, 202)
                payload = json.load(response)
            dispatch.assert_called_once_with()
            self.assertEqual(payload['status'], 'stopping')
            self.assertFalse(self.obj.armed)

    def test_dispatch_failure_keeps_running_state_and_reports_failure(self):
        with mock.patch.object(watch, 'request_full_off', side_effect=RuntimeError('task unavailable')):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post()
            self.assertEqual(error.exception.code, 503)
            self.assertFalse(json.load(error.exception)['ok'])
            self.assertTrue(self.obj.armed)

    def test_cross_origin_and_any_key_cannot_shut_down_the_program(self):
        for via, origin in [('web', 'http://example.org'), ('input-any-key', None), ('input-mouse', None), ('hotkey', None), ('', None)]:
            with self.subTest(via=via), mock.patch.object(watch, 'request_full_off') as dispatch:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.post(via=via, origin=origin)
                self.assertEqual(error.exception.code, 403)
                dispatch.assert_not_called()
