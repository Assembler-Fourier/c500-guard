"""A blocked optional archive must not stop observation or whole-program shutdown."""
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from test_watch import watcher
from vehicle_sightings import VehicleSightings


class HistoryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='history-worker-')
        self.now = 100.
        self.store = VehicleSightings(self.temp.name, clock=lambda: self.now,
                                      wall=lambda: 100000.+self.now, async_writes=True, max_pending=2)
        self.frame = np.zeros((100,100,3), dtype=np.uint8)
        self.zones = [SimpleNamespace(name='Exit', route_role='exit')]
        self.entered, self.release = threading.Event(), threading.Event()

    def tearDown(self):
        self.release.set()
        self.store.flush(timeout=1.)
        if self.store._writer:
            self.store._writer.join(1.)
        self.temp.cleanup()

    def vehicle(self, ident):
        return {'track_id': ident, 'label': 'car', 'confidence': .9,
                'box_n': [.2,.2,.8,.8], 'regions': ['Exit']}

    def feed(self, objects, step=.25):
        self.now += step
        self.store.observe(objects, self.now, self.frame, self.zones)

    def block(self, *args, **kwargs):
        self.entered.set()
        self.release.wait(2.)

    def test_slow_record_write_does_not_block_next_vehicle(self):
        with mock.patch.object(self.store, '_write', side_effect=self.block):
            self.feed([self.vehicle(1)])
            self.feed([self.vehicle(1)])
            self.feed([], step=5.)
            self.assertTrue(self.entered.wait(1.))
            self.feed([self.vehicle(2)])
            self.assertIn(2, self.store._tracks)
            self.assertTrue(self.store.status()['writing'])
            self.assertFalse(self.release.is_set())

    def test_slow_retention_purge_is_not_on_observation_thread(self):
        with mock.patch.object(self.store, 'purge', side_effect=self.block):
            self.feed([], step=1000.)
            self.assertTrue(self.entered.wait(1.))
            self.feed([self.vehicle(2)])
            self.assertIn(2, self.store._tracks)

    def test_queue_is_bounded_and_shutdown_cancels_unstarted_records(self):
        calls = []
        def write(ident, *_args, **_kwargs):
            calls.append(ident)
            self.block()
        with mock.patch.object(self.store, '_write', side_effect=write):
            self.assertTrue(self.store._submit(('write', 1, {}, False)))
            self.assertTrue(self.entered.wait(1.))
            self.assertTrue(self.store._submit(('write', 2, {}, False)))
            self.assertTrue(self.store._submit(('write', 3, {}, False)))
            self.assertFalse(self.store._submit(('write', 4, {}, False)))
            self.assertEqual(2, self.store.status()['pending_writes'])
            started = time.monotonic()
            self.store.flush(timeout=.03)
            self.assertLess(time.monotonic()-started, .5)
            self.assertEqual(0, self.store.status()['pending_writes'])
            self.assertTrue(self.store.status()['closing'])
            self.assertFalse(self.store._submit(('write', 5, {}, False)))
            self.release.set()
            self.store._writer.join(1.)
            self.assertFalse(self.store._writer.is_alive())
            self.assertEqual([1], calls)

    def test_storage_error_does_not_kill_worker_or_following_job(self):
        done = threading.Event()
        calls = []
        def write(ident, *_args, **_kwargs):
            calls.append(ident)
            if ident == 1:
                raise OSError('synthetic disk failure')
            done.set()
        with mock.patch.object(self.store, '_write', side_effect=write):
            self.store._submit(('write', 1, {}, False))
            self.store._submit(('write', 2, {}, False))
            self.assertTrue(done.wait(1.))
            self.assertEqual([1,2], calls)
            self.assertIn('OSError', self.store.last_error)

    def test_thread_start_failure_can_retry_without_growing_queue(self):
        with mock.patch('vehicle_sightings.threading.Thread.start', side_effect=RuntimeError('no thread')):
            self.assertFalse(self.store._submit(('purge',)))
        self.assertIsNone(self.store._writer)
        self.assertEqual(0, self.store.status()['pending_writes'])
        self.assertTrue(self.store._submit(('purge',)))
        self.store.flush()
        self.assertFalse(self.store._writer.is_alive())

    def test_normal_shutdown_saves_detached_crops_then_exits(self):
        self.feed([self.vehicle(1)])
        self.feed([self.vehicle(1)])
        self.store.flush()
        self.assertEqual(1, self.store.written)
        self.assertEqual(1, len(list(Path(self.temp.name).rglob('meta.json'))))
        self.assertFalse(self.store._writer.is_alive())
        self.feed([self.vehicle(2)])
        self.assertEqual({}, self.store._tracks)

    def test_persistent_track_has_bounded_brightness_history(self):
        for _ in range(200):
            self.feed([self.vehicle(1)])
        self.assertEqual(60, len(self.store._tracks[1]['brightness']))

    def test_live_watcher_selects_async_storage_without_starting_extra_process(self):
        obj = watcher()
        self.assertTrue(obj.sightings.async_writes)
        self.assertIsNone(obj.sightings._writer)
        obj.sightings.flush()


if __name__ == '__main__':
    unittest.main()
