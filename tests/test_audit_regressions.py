"""Regressions reproduced during the Claude/live installation audit."""
import threading
import unittest
from unittest import mock

from test_watch import watch, watcher


class ReleaseBoundary:
    """Schedule an alarm exactly when the test releases the shared state lock."""
    def __init__(self):
        self.lock = threading.RLock()
        self.after = None

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()
        callback, self.after = self.after, None
        if callback:
            callback()


class AuditRegressionTests(unittest.TestCase):
    def test_camera_loss_full_stop_applies_even_with_alerts_paused(self):
        obj = watcher()
        obj.armed = False
        obj.first_frame_seen = True
        obj.camera_gone_s = 120.
        obj.last_frame_at = 10.
        with mock.patch.object(obj, '_request_full_stop') as stop:
            obj.check_health(129.)
            stop.assert_not_called()
            obj.check_health(131.)
            stop.assert_called_once()

    def test_failed_automatic_shutdown_retains_fault_sound_and_reports_problem(self):
        obj = watcher()
        obj.buzzer.set('fault')
        with mock.patch('power_control.request_full_off', side_effect=RuntimeError('scheduler unavailable')):
            obj._request_full_stop('camera offline')
        self.assertEqual(obj.buzzer.want, 'fault')
        self.assertIn('Full OFF failed', obj.control_warning)

    def test_real_alarm_wins_at_siren_test_start_boundary(self):
        obj = watcher()
        boundary = ReleaseBoundary()
        obj._state_lock = boundary
        boundary.after = lambda: obj.raise_alarm('fault', 'camera lost')
        with mock.patch.object(watch.threading, 'Thread'):
            obj.play_test_tone(.2)
        self.assertEqual(obj.alarm, 'fault')
        self.assertEqual(obj.buzzer.want, 'fault')

    def test_real_alarm_wins_at_siren_test_finish_boundary(self):
        obj = watcher()
        boundary = ReleaseBoundary()
        obj._state_lock = boundary
        with mock.patch.object(watch.threading, 'Thread') as worker:
            obj.play_test_tone(.2)
            finish = worker.call_args.kwargs['target']
        boundary.after = lambda: obj.raise_alarm('fault', 'camera lost')
        with mock.patch.object(watch.time, 'sleep'):
            finish()
        self.assertEqual(obj.alarm, 'fault')
        self.assertEqual(obj.buzzer.want, 'fault')
