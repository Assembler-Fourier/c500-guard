"""Calibration persistence may wait on disk without stalling camera evidence."""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from barrier_workflow import BarrierWorkflow


class WorkflowContentionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.state = 'down'
        self.monitor = SimpleNamespace(
            update=mock.Mock(),
            status=lambda: {'configured': True, 'barriers': [dict(
                id='exit', state=self.state, fresh=True, calibrated=True,
                observed_at=self.now, position_since=self.now, confidence=.9,
                active_line=[[0., 0.], [1., 1.]])]},
            capture=mock.Mock(return_value={'ok': True}),
            road_clear=lambda *a, **kw: {'clear': True})
        self.flow = BarrierWorkflow('synthetic', '.', None, monitor=self.monitor,
                                    clock=lambda: self.now)
        self.frame = np.zeros((4, 4, 3), np.uint8)
        self.zone = SimpleNamespace(name='Exit', route_group='campus-exit', route_role='exit',
            motion_active=False, light_active=False, polygon=None, rect=(0, 0, 1, 1),
            exclude_polygons=[], mask_for=lambda w, h: np.ones((h, w), bool))
        self.alarm = dict(id='car', route='campus-exit', started_at=100.)

    def observe(self, detections=None, alarm=None, pending=()):
        primary = dict(available=True, state='ready', observed_at=self.now,
                       detections=[{'regions': ['Exit']}] if detections is None else detections)
        return self.flow.observe(self.frame, self.now, primary, [self.zone], True,
                                 alarm=alarm, pending_routes=pending)

    def during_capture(self, operation):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def capture(*args):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test capture was not released')
            return {'ok': False, 'reason': 'synthetic write refused'}
        self.monitor.capture.side_effect = capture
        writer = threading.Thread(target=lambda: self.flow.calibrate(
            {'action': 'capture', 'id': 'exit', 'position': 'down'}), daemon=True)
        result, errors = [], []
        def read():
            try:
                result.append(operation())
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        reader = threading.Thread(target=read, daemon=True)
        writer.start()
        self.assertTrue(entered.wait(1.))
        try:
            reader.start()
            self.assertTrue(done.wait(.4), 'camera/status waited on calibration persistence')
        finally:
            release.set()
            writer.join(2.)
            reader.join(2.)
        self.assertFalse(writer.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual([], errors)
        return result[0]

    def test_observation_skips_a_blocked_calibration_and_discards_pre_gap_transition(self):
        self.observe(alarm=self.alarm)
        self.flow.status()
        self.now = 101.
        def skip():
            intents = self.observe(alarm=self.alarm)
            return intents, self.flow.status()
        intents, status = self.during_capture(skip)
        self.assertEqual([], intents)
        self.assertFalse(status['trusted'])
        self.assertFalse(status['reminder_active'])
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(status['barriers'][0]['fresh'])
        self.assertIsNone(status['barriers'][0]['active_line'])
        self.state, self.now = 'up', 101.5
        self.assertEqual([], self.observe(alarm=self.alarm), 'old DOWN cannot acknowledge new UP across a skipped frame')

    def test_status_during_calibration_returns_unknown_without_waiting(self):
        self.observe()
        self.flow.status()
        status = self.during_capture(self.flow.status)
        self.assertFalse(status['trusted'])
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertTrue(status['configured'], 'calibration controls remain present while saving')
        self.assertTrue(self.flow.status()['trusted'], 'a read-only status collision does not discard evidence')

    def test_work_expiring_source_image_does_not_offer_a_transition(self):
        self.observe(alarm=self.alarm)
        self.state, self.now = 'up', 101.
        # Preserve the image timestamp even though the simulated monitor work
        # advances the clock beyond the evidence window.
        self.monitor.status = lambda: {'barriers': [dict(id='exit', state='up',
            fresh=True, calibrated=True, observed_at=101., position_since=101.)]}
        self.monitor.update.side_effect = lambda *a, **kw: setattr(self, 'now', 104.1)
        self.assertEqual([], self.observe(alarm=self.alarm))
        self.assertFalse(self.flow.status()['trusted'])

    def test_model_expiring_during_work_cannot_confirm_current_image_occupancy(self):
        self.observe(alarm=self.alarm)
        self.now = 102.
        self.zone.motion_active = True
        primary = dict(available=True, state='ready', observed_at=100., detections=[])
        self.monitor.update.side_effect = lambda *a, **kw: setattr(self, 'now', 103.1)
        self.flow.observe(self.frame, 102., primary, [self.zone], True, alarm=self.alarm)
        status = self.flow.status()
        self.assertTrue(status['trusted'], 'the camera image is still current')
        self.assertEqual('unknown', status['service']['routes']['campus-exit']['occupancy'])

    def test_learning_waits_for_image_activity_and_outstanding_visits_to_clear(self):
        with mock.patch.object(self.flow, '_learn_arms') as learn:
            for change in ('motion_active', 'light_active'):
                with self.subTest(activity=change):
                    setattr(self.zone, change, True)
                    self.observe(detections=[])
                    self.assertEqual('occupied', self.flow.status()['road_readings']['campus-exit']['occupancy'])
                    learn.assert_not_called()
                    setattr(self.zone, change, False)
            self.observe(detections=[], alarm=self.alarm)
            self.observe(detections=[], pending=('campus-exit',))
            learn.assert_not_called()
            self.observe(detections=[])
            learn.assert_called_once()

    def during_work(self, target, operation, check):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        result, errors = [], []
        original = getattr(*target)
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test operation was not released')
            return original(*args, **kwargs)
        def work():
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        def read():
            try:
                result.append(check())
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        with mock.patch.object(*target, side_effect=blocked):
            worker = threading.Thread(target=work, daemon=True)
            reader = threading.Thread(target=read, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(1.))
            try:
                reader.start()
                self.assertTrue(done.wait(.4), 'status waited for unfinished work')
            finally:
                release.set()
                worker.join(2.)
                reader.join(2.)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual([], errors)
        return result[0]

    def test_normal_observation_displays_only_last_completed_snapshot(self):
        self.observe(alarm=self.alarm)
        self.state, self.now = 'up', 101.
        def check():
            first = self.flow.status()
            first['barriers'][0]['active_line'][0][0] = .75
            second = self.flow.status()
            self.assertEqual(0., second['barriers'][0]['active_line'][0][0])
            return second
        status = self.during_work((self.monitor, 'update'),
            lambda: self.observe(alarm=self.alarm), check)
        self.assertEqual('down', status['barriers'][0]['state'])
        self.assertEqual(100., status['barriers'][0]['observed_at'])
        self.assertTrue(status['barriers'][0]['fresh'])
        self.assertTrue(status['cached'])
        self.assertFalse(status['trusted'], 'display snapshot cannot authorize acknowledgement')
        self.assertEqual({'routes': {}}, status['service'])
        self.assertEqual({}, status['road_readings'])
        self.assertFalse(status['reminder_active'])
        self.assertEqual('up', self.flow.status()['barriers'][0]['state'])

    def test_cached_snapshot_never_renews_its_source_time(self):
        self.observe()
        self.now = 102.9
        def expired():
            self.now = 103.
            return self.flow.status()
        status = self.during_work((self.monitor, 'update'), self.observe, expired)
        self.assertFalse(status['trusted'])
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(status['barriers'][0]['fresh'])

    def test_unaligned_or_stale_incoming_observation_invalidates_display_immediately(self):
        for aligned, stamp in ((False, 101.), (True, 97.)):
            with self.subTest(aligned=aligned, stamp=stamp):
                self.now = 100.
                self.observe()
                self.now = 101.
                operation = lambda: self.flow.observe(self.frame, stamp, {}, [self.zone], aligned)
                status = self.during_work((self.monitor, 'update'), operation, self.flow.status)
                self.assertEqual('unknown', status['barriers'][0]['state'])
                self.assertFalse(status['trusted'])

    def test_future_snapshot_time_and_future_row_time_are_never_displayed_as_known(self):
        for future_snapshot in (False, True):
            with self.subTest(future_snapshot=future_snapshot):
                self.now = 100.
                self.observe()
                stamp, saved = self.flow._status_snapshot
                if future_snapshot:
                    self.flow._status_snapshot = (102., saved)
                else:
                    saved['barriers'][0]['observed_at'] = 102.
                self.now = 101.
                status = self.during_work((self.monitor, 'update'), self.observe, self.flow.status)
                self.assertEqual('unknown', status['barriers'][0]['state'])
                self.assertFalse(status['barriers'][0]['fresh'])
                self.assertFalse(status['trusted'])

    def test_skipped_observation_invalidates_an_inflight_display_snapshot(self):
        self.observe()
        self.now = 101.
        def skip():
            self.assertEqual([], self.observe())
            return self.flow.status()
        status = self.during_work((self.monitor, 'update'), self.observe, skip)
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(self.flow.status()['trusted'])

    def test_recovery_observation_cannot_redisplay_snapshot_from_before_a_skipped_frame(self):
        self.observe()
        self.now = 101.
        self.during_work((self.monitor, 'status'), self.flow.status, self.observe)
        self.now = 101.1
        status = self.during_work((self.monitor, 'update'), self.observe, self.flow.status)
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(status['trusted'])

    def test_reset_owner_never_serves_the_old_display_snapshot(self):
        self.observe()
        status = self.during_work((self.flow.cycles, 'reset'), self.flow.reset, self.flow.status)
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(status['trusted'])

    def test_undo_owner_never_serves_the_old_display_snapshot(self):
        self.observe()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'references.npz'
            path.write_bytes(b'new')
            path.with_name(path.name + '.before-learn-test').write_bytes(b'old')
            self.monitor.storage_path = path
            self.monitor.reload = mock.Mock()
            status = self.during_work((self.monitor, 'reload'),
                self.flow.undo_last_learning, self.flow.status)
        self.assertEqual('unknown', status['barriers'][0]['state'])
        self.assertFalse(status['trusted'])

    def test_waiting_mutation_does_not_claim_the_observation_lock(self):
        self.observe()
        entered, release = threading.Event(), threading.Event()
        mutation_started, mutation_entered, mutation_release = (threading.Event() for _ in range(3))
        errors = []
        def update(*args, **kwargs):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('observation not released')
        def reset():
            mutation_entered.set()
            if not mutation_release.wait(3.):
                raise RuntimeError('reset not released')
        def work(operation):
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        def mutation():
            mutation_started.set()
            self.flow.reset()
        with mock.patch.object(self.monitor, 'update', side_effect=update), \
                mock.patch.object(self.flow.cycles, 'reset', side_effect=reset):
            observer = threading.Thread(target=lambda: work(self.observe), daemon=True)
            writer = threading.Thread(target=lambda: work(mutation), daemon=True)
            observer.start()
            self.assertTrue(entered.wait(1.))
            writer.start()
            self.assertTrue(mutation_started.wait(1.))
            try:
                self.assertFalse(mutation_entered.is_set())
                status = self.flow.status()
                self.assertEqual('down', status['barriers'][0]['state'])
                self.assertTrue(status['cached'])
                release.set()
                self.assertTrue(mutation_entered.wait(1.))
                self.assertEqual('unknown', self.flow.status()['barriers'][0]['state'])
            finally:
                release.set()
                mutation_release.set()
                observer.join(2.)
                writer.join(2.)
        self.assertEqual([], errors)


if __name__ == '__main__':
    unittest.main()
