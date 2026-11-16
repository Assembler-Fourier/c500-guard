"""Bounded scheduling and image-free diagnostics; no model/camera/live services."""
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semantic_monitor import SemanticMonitor

REAL_MONOTONIC = time.monotonic


class LatestPendingTests(unittest.TestCase):
    def make_monitor(self, **kwargs):
        zones = [SimpleNamespace(rect=(0., 0., 1., 1.), motion=True,
                                 severity='main', name='Gate')]
        with patch.object(threading.Thread, 'start'):
            monitor = SemanticMonitor(zones, Path('no-model.onnx'), **kwargs)
        self.addCleanup(monitor.stop)
        return monitor

    @staticmethod
    def frame(value):
        return np.full((100, 100, 3), value, np.uint8)

    def send(self, monitor, now, value, stamp=None, enabled=True):
        with patch('semantic_monitor.time.monotonic', return_value=now):
            return monitor.update(self.frame(value), enabled, stamp=now if stamp is None else stamp)

    @staticmethod
    def await_condition(predicate):
        deadline = REAL_MONOTONIC() + 2.
        while REAL_MONOTONIC() < deadline:
            if predicate():
                return
            time.sleep(.002)
        raise AssertionError('Fake worker did not reach expected state')

    @staticmethod
    def start_worker(monitor):
        monitor._thread = threading.Thread(target=monitor._run, daemon=True)
        monitor._thread.start()

    def test_latest_pending_replaces_older_pixels_without_speeding_cadence(self):
        monitor = self.make_monitor(interval=.5)
        self.send(monitor, 100., 1)
        self.send(monitor, 100.25, 2)
        self.assertEqual(1, monitor._pipeline['inputs_submitted'])
        self.send(monitor, 100.6, 3, stamp=100.55)
        self.assertEqual(1, monitor._input.qsize())
        epoch, generation, stamp, views = monitor._input.get_nowait()
        self.assertEqual((monitor._epoch, monitor._worker_generation, 100.55),
                         (epoch, generation, stamp))
        self.assertEqual(3, int(views[0][0][0, 0, 0]))
        self.assertEqual(2, monitor._pipeline['inputs_submitted'])
        self.assertEqual(1, monitor._pipeline['pending_replaced'])
        self.assertEqual(100.6, monitor._input_times[(epoch, generation, stamp)])

    def test_older_duplicate_invalid_and_expired_inputs_cannot_replace_latest(self):
        monitor = self.make_monitor(interval=.1)
        self.send(monitor, 100., 1)
        for source in (100., 99.9, 97., 101., True, float('nan')):
            self.send(monitor, 100.5, 2, stamp=source)
        self.assertEqual(100., monitor._input.get_nowait()[2])
        self.assertEqual(1, monitor._pipeline['inputs_submitted'])

    def test_reset_recovery_and_off_cannot_relabel_pixels_copied_before_transition(self):
        for action in ('reset', 'recovery', 'off'):
            with self.subTest(action=action):
                monitor = self.make_monitor(interval=.1)
                monitor._enabled = True
                original = np.asarray
                called = False

                def copy_with_transition(*args, **kwargs):
                    nonlocal called
                    array = original(*args, **kwargs)
                    if not called:
                        called = True
                        if action == 'recovery':
                            monitor._retry_failure(RuntimeError('fake failure'), 2.)
                        else:
                            monitor.reset()
                            if action == 'off':
                                monitor._enabled = False
                    return array

                with patch('semantic_monitor.np.asarray', side_effect=copy_with_transition):
                    self.send(monitor, 100., 1)
                self.assertTrue(monitor._input.empty())
                self.assertEqual(0, monitor._pipeline['inputs_submitted'])

    def test_source_expiring_during_copy_is_not_queued(self):
        monitor = self.make_monitor(interval=.1)
        clock = [100.]
        original = np.asarray

        def slow_copy(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 103.1
            return result

        with patch('semantic_monitor.time.monotonic', side_effect=lambda: clock[0]), \
             patch('semantic_monitor.np.asarray', side_effect=slow_copy):
            monitor.update(self.frame(1), True, stamp=100.)
        self.assertTrue(monitor._input.empty())

    def test_only_latest_pending_follows_a_slow_inflight_inference(self):
        monitor = self.make_monitor(interval=.001)
        entered, release = threading.Event(), threading.Event()
        seen = []

        class Detector:
            available = True
            def status(self):
                return {'available': True}
            def detect(self, frame):
                seen.append(int(frame[0, 0, 0]))
                if len(seen) == 1:
                    entered.set()
                    if not release.wait(2.):
                        raise AssertionError('Test did not release fake inference')
                return []

        with patch('object_detector.ObjectDetector', return_value=Detector()):
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            monitor.update(self.frame(1), True)
            self.assertTrue(entered.wait(1.))
            try:
                monitor._last_submit = 0.
                monitor.update(self.frame(2), True)
                monitor._last_submit = 0.
                latest_stamp = time.monotonic()
                pixels = self.frame(3)
                monitor.update(pixels, True, stamp=latest_stamp)
                pixels.fill(9)  # Queued work owns its pixels.
                self.assertEqual(1, monitor._input.qsize())
                self.assertEqual(1, monitor._pipeline['pending_replaced'])
            finally:
                release.set()
            self.await_condition(lambda: monitor.status()['inferences'] == 2)
            self.assertEqual([1, 3], seen)
            self.assertEqual(latest_stamp, monitor.status()['observed_at'])
            self.assertEqual(1, monitor.status()['pipeline']['undelivered_replaced'])
            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())

    def test_queue_wait_and_delivery_timings_keep_exact_source_stamp(self):
        monitor = self.make_monitor(interval=10.)
        clock = [100.]
        original_get = monitor._input.get

        def dequeue(*args, **kwargs):
            item = original_get(*args, **kwargs)
            if kwargs.get('timeout') == .25:
                clock[0] = 100.2
            return item

        class Detector:
            available = True
            def status(self):
                return {'available': True}
            def detect(self, frame):
                clock[0] = 100.3
                return []

        with patch('semantic_monitor.time.monotonic', side_effect=lambda: clock[0]), \
             patch.object(monitor._input, 'get', side_effect=dequeue), \
             patch('object_detector.ObjectDetector', return_value=Detector()):
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            monitor.update(self.frame(1), True, stamp=99.9)
            self.await_condition(lambda: monitor.status()['inferences'] == 1)
            first = monitor.status()
            timing = first['pipeline']['last_inference']
            self.assertEqual(99.9, timing['observed_at'])
            self.assertEqual(100., timing['source_to_enqueue_ms'])
            self.assertEqual(200., timing['queue_wait_ms'])
            self.assertEqual(400., timing['source_to_result_ms'])
            self.assertEqual(100., first['view_diagnostics'][0]['inference_ms'])
            clock[0] = 100.8
            monitor.update(self.frame(2), True, stamp=100.8)
            report = monitor.status()
            self.assertEqual(500., report['pipeline']['last_delivery']['result_wait_ms'])
            self.assertEqual(900., report['pipeline']['last_delivery']['source_to_delivery_ms'])
            self.assertEqual(99.9, report['pipeline']['last_delivery']['observed_at'])
            self.assertEqual(1, report['pipeline']['results_consumed'])
            monitor.stop()

    def test_unmerged_diagnostics_are_bounded_image_free_and_clear_when_invalid(self):
        monitor = self.make_monitor(interval=10., context_view=True)
        rects = [(10, 20, 70, 90), (40, 10, 100, 80)]
        raw = [dict(label='car', confidence=.8, bbox=[5, 6, 20, 30],
                    image=self.frame(7), unexpected='not retained') for _ in range(55)]

        class Detector:
            available = True
            calls = 0
            def status(self):
                return {'available': True}
            def detect(self, frame):
                self.calls += 1
                return raw if self.calls in (1, 3) else []

        with patch('object_detector.ObjectDetector', return_value=Detector()), \
             patch('semantic_monitor.inference_rects', return_value=rects):
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            monitor.update(self.frame(1), True)
            self.await_condition(lambda: monitor.status()['inferences'] == 1)
            report = monitor.status()
            views = report['view_diagnostics']
            self.assertEqual(['focused', 'focused', 'context'], [v['kind'] for v in views])
            self.assertEqual([55, 0, 55], [v['detection_count'] for v in views])
            self.assertEqual([50, 0, 50], [len(v['detections']) for v in views])
            self.assertEqual([15., 26., 30., 50.], views[0]['detections'][0]['bbox_frame'])
            self.assertEqual([5., 6., 20., 30.], views[2]['detections'][0]['bbox_frame'])
            self.assertEqual({'label', 'confidence', 'bbox', 'bbox_frame'},
                             set(views[0]['detections'][0]))
            json.dumps(views, allow_nan=False)
            views[0]['detections'][0]['bbox'][0] = 999
            self.assertEqual(5., monitor.status()['view_diagnostics'][0]['detections'][0]['bbox'][0])
            monitor.reset()
            report = monitor.status()
            self.assertEqual([], report['view_diagnostics'])
            self.assertIsNone(report['pipeline']['last_inference'])
            self.assertIsNone(report['pipeline']['last_delivery'])
            monitor.stop()

    def test_pending_bookkeeping_stays_bounded_and_off_never_confirms(self):
        monitor = self.make_monitor(interval=.1)
        for i in range(100):
            self.send(monitor, 100+i*.2, i, enabled=False)
        self.assertEqual(1, monitor._input.qsize())
        self.assertLessEqual(len(monitor._input_times), 2)
        self.assertEqual(99, monitor._pipeline['pending_replaced'])
        monitor._output.put((monitor._epoch, monitor._worker_generation, 119.8,
                             [dict(label='car', confidence=.9, bbox=[10, 10, 50, 60])]))
        self.assertEqual([], self.send(monitor, 119.9, 1, enabled=False))
        self.assertEqual(0, monitor._pipeline['results_consumed'])
        self.assertEqual(0, monitor._tracker.track_count)


if __name__ == '__main__':
    unittest.main()
