"""Slow recording history must never hold up live controls or alignment."""
import threading
import time
import unittest
from unittest import mock

from test_watch import watch, watcher


class AlertRecordingLockTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.recent_alerts.clear()
        self.obj._remember_alert('2026-11-20T19:00:00', 'main', 'Synthetic car',
                                 'synthetic-1', None)
        self.version = self.obj.alerts_version
        self.entered, self.release = threading.Event(), threading.Event()
        self.result, self.errors = [], []
        self.obj.recorder = mock.Mock()

        def recording_for(_alarm_id):
            self.entered.set()
            if not self.release.wait(3.):
                raise RuntimeError('test did not release recording lookup')
            return 'synthetic.mp4'

        self.obj.recorder.recording_for.side_effect = recording_for

        def history():
            try:
                self.result.append(self.obj.alerts())
            except BaseException as exc:
                self.errors.append(exc)

        self.thread = threading.Thread(target=history)
        self.thread.start()
        self.addCleanup(self.finish)
        self.assertTrue(self.entered.wait(1.), 'history did not reach recording lookup')

    def finish(self):
        self.release.set()
        self.thread.join(2.)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual([], self.errors)

    def control_completes(self, operation):
        done, failures = threading.Event(), []

        def control():
            try:
                operation()
            except BaseException as exc:
                failures.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=control)
        thread.start()
        try:
            self.assertTrue(done.wait(1.), 'recording history blocked a live control')
        finally:
            self.release.set()
            thread.join(2.)
        self.assertEqual([], failures)

    def test_acknowledgement_does_not_wait_for_recording_lookup(self):
        self.obj.alarm = 'main'
        self.obj.alarm_id = 'synthetic-1'
        self.obj.alarm_reason = 'Synthetic car'
        self.obj.alarm_since = time.monotonic()-2.
        self.control_completes(lambda: self.assertTrue(self.obj.acknowledge(via='web')))
        self.assertIsNone(self.obj.alarm)

    def test_new_alarm_does_not_wait_for_recording_lookup(self):
        self.obj._queue_snapshot = mock.Mock(return_value=None)
        with mock.patch.object(watch, 'notify'):
            self.control_completes(lambda: self.assertTrue(self.obj.raise_alarm('main', 'New car')))
        self.assertEqual('main', self.obj.alarm)

    def test_alignment_does_not_wait_and_snapshot_version_stays_consistent(self):
        frame = object()
        self.obj._align_frame_locked = mock.Mock(return_value=frame)

        def change():
            self.assertIs(frame, self.obj.align_frame(frame))
            with self.obj._state_lock:
                self.obj.recent_alerts[0]['acked'] = {'via': 'web', 'after_s': 2.}
                self.obj.alerts_version += 1
                self.obj._remember_alert('2026-11-20T19:00:01', 'main', 'Next car', 'synthetic-2', None)

        self.control_completes(change)
        self.thread.join(2.)
        report = self.result[0]
        self.assertEqual(self.version, report['version'])
        self.assertEqual(1, len(report['alerts']))
        self.assertIsNone(report['alerts'][0]['acked'])
        self.assertEqual('synthetic.mp4', report['alerts'][0]['recording'])


class AlignmentTimingTests(unittest.TestCase):
    def test_lock_wait_is_separate_from_alignment_work(self):
        obj = watcher()
        frame = object()
        obj._align_frame_locked = mock.Mock(return_value=frame)
        with mock.patch.object(watch.time, 'perf_counter', side_effect=(10., 10.2, 10.5)):
            self.assertIs(frame, obj.align_frame(frame))
        self.assertEqual(200., obj.alignment['lock_wait_ms'])
        self.assertEqual(300., obj.alignment['work_ms'])
        self.assertEqual(500., obj.alignment['total_ms'])


if __name__ == '__main__':
    unittest.main()
