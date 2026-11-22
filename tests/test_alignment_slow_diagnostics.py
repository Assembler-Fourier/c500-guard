"""In-memory timing evidence must not change alignment or alarm decisions."""
import threading
import unittest
from unittest import mock

from test_watch import watch, watcher


class SlowAlignmentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.now = 100.
        self.wall = 9000.
        self.perf = mock.patch.object(watch.time, 'perf_counter', side_effect=lambda: self.now)
        self.mono = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        self.wallclock = mock.patch.object(watch.time, 'time', side_effect=lambda: self.wall)
        for patch in (self.perf, self.mono, self.wallclock):
            patch.start()
            self.addCleanup(patch.stop)

    def run_alignment(self, seconds, value=None):
        def work(frame):
            self.now += seconds
            return value
        self.obj._align_frame_locked = work
        return self.obj.align_frame(object())

    def test_normal_frames_preserve_old_slow_record_and_return_value(self):
        accepted = object()
        self.assertIs(self.run_alignment(4.3, accepted), accepted)
        self.wall = -1000.  # Wall clock changes cannot affect duration.
        self.assertIsNone(self.run_alignment(.025))
        report = self.obj.diagnostics()
        self.assertEqual(len(report['alignment_slow']), 1)
        self.assertEqual(report['alignment_slow'][0]['work_ms'], 4300.)
        self.assertEqual(report['alignment_slow'][0]['finished_wall'], 9000.)
        self.assertEqual(self.obj.alignment['work_ms'], 25.)
        self.obj.log_event.assert_not_called()

    def test_history_is_bounded_to_latest_eight(self):
        for i in range(11):
            self.wall = i
            self.run_alignment(2.)
        rows = self.obj.diagnostics()['alignment_slow']
        self.assertEqual([r['finished_wall'] for r in rows], list(range(3, 11)))

    def test_lock_wait_is_distinguished_from_actual_work(self):
        clock = self
        class DelayedLock:
            def __enter__(self):
                clock.now += 4.
            def __exit__(self, *_):
                pass
        self.obj._state_lock = DelayedLock()
        self.run_alignment(.03)
        row = self.obj.diagnostics()['alignment_slow'][0]
        self.assertEqual(row['lock_wait_ms'], 4000.)
        self.assertEqual(row['work_ms'], 30.)
        self.assertEqual(row['total_ms'], 4030.)

    def test_named_step_is_visible_while_running_and_durable_afterwards(self):
        def save_pose():
            self.assertEqual(self.obj.diagnostics()['alignment_step'], 'pose_persistence')
            self.now += 4.2
            return 'unchanged'
        def work(frame):
            return self.obj._alignment_step('pose_persistence', save_pose)
        self.obj._align_frame_locked = work
        self.assertEqual(self.obj.align_frame(None), 'unchanged')
        report = self.obj.diagnostics()
        self.assertIsNone(report['alignment_step'])
        self.assertEqual(report['alignment_slow'][0]['steps_ms'], {'pose_persistence': 4200.})
        report['alignment_slow'][0]['steps_ms']['pose_persistence'] = -1
        report['alignment_slow'][0]['work_ms'] = -1
        again = self.obj.diagnostics()['alignment_slow'][0]
        self.assertEqual(again['work_ms'], 4200.)
        self.assertEqual(again['steps_ms']['pose_persistence'], 4200.)

    def test_slow_exception_retains_evidence_and_original_exception(self):
        def failed():
            self.now += 4.
            raise ValueError('fixture failure')
        self.obj._align_frame_locked = lambda frame: self.obj._alignment_step('reference_match', failed)
        with self.assertRaisesRegex(ValueError, 'fixture failure'):
            self.obj.align_frame(None)
        self.assertEqual(self.obj.diagnostics()['alignment_slow'][0]['steps_ms'], {'reference_match': 4000.})
        self.assertIsNone(self.obj.diagnostics()['alignment_step'])
        self.obj.log_event.assert_not_called()

    def test_durable_snapshot_does_not_acquire_state_lock(self):
        self.run_alignment(4.)
        self.obj._state_lock = threading.Lock()
        with self.obj._state_lock:
            self.assertEqual(len(self.obj.diagnostics()['alignment_slow']), 1)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.obj.ack_quiet_s, 10.)


if __name__ == '__main__':
    unittest.main()
