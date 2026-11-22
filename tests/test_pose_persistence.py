"""Slow native flushes must not block detection or publish obsolete poses."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from test_watch import watch, watcher
from pose_persistence import PoseWriter


def matrix(x):
    return [[1., 0., float(x)], [0., 1., 0.]]


class PosePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'pose.json'
        self.entered, self.release = threading.Event(), threading.Event()
        self.writers = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.release.set()
        for writer in self.writers:
            self.assertTrue(writer.stop(2.))

    def writer(self, fsync=lambda _: None):
        writer = PoseWriter(interval_s=0, fsync=fsync)
        self.writers.append(writer)
        writer.start()
        return writer

    def blocked_flush(self, fd):
        self.entered.set()
        if not self.release.wait(3):
            raise OSError('fixture timed out')

    def until(self, predicate):
        end = time.monotonic()+2.
        while time.monotonic()<end:
            if predicate():
                return
            time.sleep(.005)
        self.fail('worker condition not reached')

    def test_latest_only_coalescing_and_immutable_input(self):
        writer = self.writer(self.blocked_flush)
        writer.submit(self.path, matrix(0), 'first')
        self.assertTrue(self.entered.wait(1))
        for i in range(1, 101):
            value = matrix(i)
            writer.submit(self.path, value, 'last')
            value[0][2] = -1
        self.assertTrue(writer.status()['pending'])
        self.release.set()
        self.until(lambda: writer.status()['saved']==2)
        self.assertEqual(json.loads(self.path.read_text())['current_to_reference'], matrix(100))
        self.assertEqual(writer.saved()['current_to_reference'], matrix(100))
        returned = writer.saved();returned['current_to_reference'][0][2] = -2
        self.assertEqual(writer.saved()['current_to_reference'], matrix(100))

    def test_invalidation_discards_inflight_and_pending_old_reference(self):
        self.path.write_text('previous valid pose')
        writer = self.writer(self.blocked_flush)
        writer.submit(self.path, matrix(1), 'old')
        self.assertTrue(self.entered.wait(1))
        writer.submit(self.path, matrix(2), 'old queued')
        writer.invalidate()
        other = self.path.with_name('other-profile.json')
        writer.submit(other, matrix(3), 'new')
        self.release.set()
        self.until(lambda: writer.saved() is not None)
        self.assertEqual(self.path.read_text(), 'previous valid pose')
        self.assertEqual(json.loads(other.read_text())['current_to_reference'], matrix(3))
        self.assertEqual(writer.status()['saved'], 1)

    def test_stop_is_bounded_reports_blocked_io_and_prevents_publication(self):
        writer = self.writer(self.blocked_flush)
        writer.submit(self.path, matrix(1), 'old')
        self.assertTrue(self.entered.wait(1))
        start = time.monotonic()
        self.assertFalse(writer.stop(.025))
        self.assertLess(time.monotonic()-start, .25)
        self.assertTrue(writer.status()['shutdown_waiting_for_io'])
        self.assertFalse(writer.submit(self.path, matrix(2), 'rejected'))
        self.release.set()
        self.assertTrue(writer.stop(1.))
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_failed_write_does_not_claim_success_and_can_retry(self):
        flush = mock.Mock(side_effect=[OSError('disk failure'), None])
        writer = self.writer(flush)
        writer.submit(self.path, matrix(4), 'failed')
        self.until(lambda: bool(writer.status()['error']))
        self.assertIsNone(writer.saved())
        self.assertFalse(self.path.exists())
        writer.submit(self.path, matrix(4), 'retried')
        self.until(lambda: writer.saved() is not None)
        self.assertEqual(writer.status()['error'], '')
        self.assertEqual(json.loads(self.path.read_text())['saved'], 'retried')

    def test_watcher_does_not_wait_for_native_write_or_mark_unsaved_pose_persisted(self):
        obj = watcher();obj._pose_writer = self.writer(self.blocked_flush)
        obj._alignment_pose = None
        with mock.patch.object(obj, '_pose_path', return_value=self.path):
            obj._maybe_save_pose(matrix(1), (360,640,3))
            self.assertTrue(self.entered.wait(1))
            start = time.monotonic()
            with obj._state_lock:
                obj._maybe_save_pose(matrix(2), (360,640,3))
                obj.acknowledge(via='test')
            self.assertLess(time.monotonic()-start, .1)
            self.assertIsNone(obj._alignment_pose)
            self.release.set()
            self.until(lambda: obj._pose_writer.status()['saved']==2)
            obj._maybe_save_pose(matrix(2), (360,640,3))
            self.assertEqual(obj._alignment_pose.tolist(), matrix(2))

    def test_return_to_saved_pose_replaces_a_different_pending_pose(self):
        obj = watcher()
        obj._pose_writer = mock.Mock()
        obj._pose_writer.saved.return_value = {'path':self.path,'current_to_reference':matrix(0)}
        obj._pose_writer.status.return_value = {'pending':True,'writing':True}
        obj._pose_writer.submit.return_value = True
        obj._pose_requested = (self.path, watch.np.asarray(matrix(3)))
        with mock.patch.object(obj,'_pose_path',return_value=self.path):
            obj._maybe_save_pose(matrix(0),(360,640,3))
        self.assertEqual(obj._pose_writer.submit.call_args.args[1],matrix(0))

    def test_watcher_shutdown_stops_owned_writer_and_drops_pending_work(self):
        obj = watcher()
        writer = self.writer()
        obj._pose_writer = writer
        # Every external cleanup is fake; the persistence lifecycle is real.
        for name in ('buzzer','loudness','presence','reader','sleep_inhibitor','semantic',
                     'secondary','recorder','sightings','learning'):
            setattr(obj,name,mock.Mock())
        obj.sleep_inhibitor = mock.MagicMock()
        obj.shutdown()
        self.assertTrue(writer.status()['stopped'])
        self.assertFalse(writer.status()['running'])
        self.assertFalse(writer.submit(self.path,matrix(8),'after stop'))

    def test_blocked_rename_cannot_hold_detection_or_status_lock(self):
        real_replace = os.replace
        def blocked_replace(source, target):
            self.entered.set()
            if not self.release.wait(3):
                raise OSError('fixture timed out')
            real_replace(source, target)
        obj = watcher()
        writer = self.writer()
        obj._pose_writer = writer
        with mock.patch('pose_persistence.os.replace',side_effect=blocked_replace), \
                mock.patch.object(obj,'_pose_path',return_value=self.path):
            obj._maybe_save_pose(matrix(1),(360,640,3))
            self.assertTrue(self.entered.wait(1))
            start = time.monotonic()
            with obj._state_lock:
                obj._maybe_save_pose(matrix(2),(360,640,3))
                obj.acknowledge(via='test')
                self.assertTrue(writer.status()['writing'])
                self.assertIsNone(writer.saved())
            self.assertLess(time.monotonic()-start,.1)
            # A camera-reference change must defer if an old rename is already
            # inside the native call, rather than risk its later publication.
            start = time.monotonic()
            self.assertFalse(writer.invalidate(timeout=.025))
            self.assertLess(time.monotonic()-start,.25)
            self.release.set()
            self.until(lambda: not writer.status()['writing'])
        self.assertTrue(writer.invalidate(timeout=.025))
        self.path.write_text('new calibrated reference')
        self.assertTrue(writer.stop())
        self.assertEqual(self.path.read_text(),'new calibrated reference')
        self.assertIsNone(writer.saved())

    def test_watcher_defers_reference_change_when_publication_fence_is_busy(self):
        obj = watcher();obj._pose_writer = mock.Mock()
        obj._pose_writer.invalidate.return_value = False
        with self.assertRaisesRegex(OSError,'retry the camera-reference change'):
            obj._invalidate_pose_writes()


class BarrierWarmupContinuityTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.args.warmup = 2.
        self.obj.warmup_started = 109.5
        self.obj._semantic_warmup_started = 100.
        self.obj._pending_rearm = False
        self.obj._alignment_blocked = False
        self.obj.alignment = {'state':'tracking'}
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.observe.return_value = []
        self.obj.barrier_workflow.status.return_value = {'barriers':[]}
        self.obj._observe_arm_angles = mock.Mock()
        self.obj._ack_on_any_arm = mock.Mock()
        self.obj._note_arm_memory = mock.Mock()
        self.clock = mock.patch.object(watch.time,'monotonic',return_value=110.)
        self.clock.start();self.addCleanup(self.clock.stop)

    def trusted(self):
        self.obj._observe_barrier_service(None,110.)
        return self.obj.barrier_workflow.observe.call_args.args[4]

    def test_background_rewarming_does_not_blank_fresh_barriers(self):
        self.assertTrue(self.trusted())
        self.assertEqual(self.obj._semantic_warmup_remaining(), 0.)
        self.assertEqual(self.obj.ack_quiet_s, 10.)

    def test_startup_and_full_rearm_still_require_confirmation(self):
        self.obj._semantic_warmup_started = None
        self.assertFalse(self.trusted())
        self.obj._semantic_warmup_started = 100.
        self.obj._pending_rearm = True
        self.assertFalse(self.trusted())

    def test_unverified_alignment_remains_untrusted(self):
        self.obj.alignment = {'state':'holding'}
        self.assertFalse(self.trusted())
        self.obj.alignment = {'state':'tracking'}
        self.obj._alignment_blocked = True
        self.assertFalse(self.trusted())

    def test_state_reports_ai_and_pixel_readiness_separately(self):
        report = self.obj.state_dict()
        self.assertEqual(report['warmup_remaining_s'], 1.5)
        self.assertEqual(report['background_warmup_remaining_s'], 1.5)
        self.assertEqual(report['semantic_warmup_remaining_s'], 0.)
        self.obj._pending_rearm = True
        self.assertEqual(self.obj.state_dict()['semantic_warmup_remaining_s'], 2.)

    def test_stale_frame_remains_untrusted(self):
        self.obj._observe_barrier_service(None,105.)
        self.assertFalse(self.obj.barrier_workflow.observe.call_args.args[4])


if __name__ == '__main__':
    unittest.main()
