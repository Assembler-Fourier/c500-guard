"""Optional entrance inference shares one worker/source; no model or GPU work."""
import copy
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import semantic_monitor as sm

REAL_MONOTONIC = time.monotonic
PIN = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
GPU = 'Intel graphics (OpenVINO)'
RECTS = [(0, 0, 40, 90), (40, 0, 100, 90)]


class Zone(SimpleNamespace):
    def contains_point(self, x, y):
        left, top, width, height = self.rect
        return left <= x <= left + width and top <= y <= top + height


def detection(box, confidence=.8, label='car'):
    return {'label': label, 'confidence': confidence, 'raw_confidence': confidence, 'bbox': list(box)}


class FakeDetector:
    def __init__(self, label, confidence, output, provider=GPU, callback=None):
        self.label, self.confidence, self.output = label, confidence, output
        self.provider, self.callback = provider, callback
        self.available, self.calls = True, 0
        self.thread_ids, self.shapes = [], []

    def status(self):
        return {'available': self.available, 'provider': self.provider,
                'model': self.label, 'error': '' if self.available else 'fake GPU unavailable'}

    def detect(self, frame):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        self.shapes.append(frame.shape)
        if self.callback is not None:
            self.callback(self, frame)
        rows = self.output(self.calls) if callable(self.output) else self.output
        return [copy.deepcopy(row) for row in rows if row['raw_confidence'] >= self.confidence]


class EntranceVerifierTests(unittest.TestCase):
    @staticmethod
    def wait_for(predicate):
        deadline = REAL_MONOTONIC() + 2.
        while REAL_MONOTONIC() < deadline:
            if predicate():
                return
            time.sleep(.002)
        raise AssertionError('Fake worker did not finish its bounded operation')

    def start(self, *, optional=True, context=True, tiny_output=None, extra_output=None,
              extra_provider=GPU, extra_init_error=None, callback=None, tiny_callback=None):
        zones = [Zone(name='Entrance', rect=(.4, .2, .5, .6), ai_enabled=True,
                      classes=['car', 'bus', 'truck'], ai_confidence=.5, motion=False,
                      route_role='entrance', severity='main', arrival_direction=None),
                 Zone(name='Exit', rect=(0., .2, .3, .6), ai_enabled=True,
                      classes=['car', 'bus', 'truck'], ai_confidence=.5, motion=False,
                      route_role='exit', severity='main', arrival_direction=None)]
        if tiny_output is None:
            tiny_output = lambda call: [detection((10, 20, 30, 60), .55)] if call % 3 == 2 else []
        if extra_output is None:
            extra_output = [detection((50, 20, 70, 60), .8)]
        self.offset = 0.
        self.enterContext(patch('semantic_monitor.time.monotonic', side_effect=lambda: REAL_MONOTONIC() + self.offset))
        with patch.object(threading.Thread, 'start'):
            monitor = sm.SemanticMonitor(zones, Path('tiny-fixture.onnx'), interval=.25,
                context_view=context, entrance_verifier_path=Path('s-fixture.onnx') if optional else None)
        self.addCleanup(monitor.stop)
        self.requests, self.instances = [], []

        def factory(path, **kwargs):
            self.requests.append((path, dict(kwargs)))
            secondary = Path(path).name == 's-fixture.onnx'
            if secondary and extra_init_error:
                raise extra_init_error
            instance = FakeDetector('S' if secondary else 'Tiny', kwargs['confidence'],
                                    extra_output if secondary else tiny_output,
                                    provider=extra_provider if secondary else GPU,
                                    callback=callback if secondary else tiny_callback)
            self.instances.append(instance)
            return instance

        self.enterContext(patch('object_detector.ObjectDetector', side_effect=factory))
        self.enterContext(patch('semantic_monitor.inference_rects', return_value=RECTS))
        self.finished = threading.Event()
        original_get = monitor._input.get
        consumed = False

        def work_get(*args, **kwargs):
            nonlocal consumed
            if kwargs.get('timeout') == .25 and consumed:
                self.finished.set()  # Previous source has finished processing.
            value = original_get(*args, **kwargs)
            if kwargs.get('timeout') == .25:
                consumed = True
            return value

        self.enterContext(patch.object(monitor._input, 'get', side_effect=work_get))
        monitor._thread = threading.Thread(target=monitor._run, daemon=True)
        monitor._thread.start()
        self.wait_for(lambda: monitor.status()['available'])
        return monitor

    @staticmethod
    def frame():
        return np.zeros((100, 100, 3), np.uint8)

    def submit(self, monitor):
        self.finished.clear()
        monitor._last_submit = 0.
        stamp = sm.time.monotonic()
        monitor.update(self.frame(), True, stamp=stamp)
        return stamp

    def deliver(self, monitor):
        previous = monitor._pipeline['results_published']
        stamp = self.submit(monitor)
        self.wait_for(lambda: monitor._pipeline['results_published'] == previous + 1)
        report = monitor.status()
        monitor._last_submit = sm.time.monotonic()
        events = monitor.update(self.frame(), True, stamp=sm.time.monotonic())
        return events, report, stamp

    def test_default_keeps_only_three_tiny_views_and_original_detections(self):
        monitor = self.start(optional=False)
        events, report, _ = self.deliver(monitor)
        self.assertEqual([], events)
        self.assertEqual(1, len(self.requests))
        self.assertEqual({'confidence': .5, 'threads': 2}, self.requests[0][1])
        self.assertEqual(3, self.instances[0].calls)
        self.assertEqual(.55, report['detections'][0]['confidence'])
        self.assertEqual('disabled', report['entrance_verifier']['state'])
        self.assertEqual(3, len(report['view_diagnostics']))

    def test_explicit_s_profile_is_pinned_gpu_only_and_same_worker(self):
        monitor = self.start()
        self.deliver(monitor)
        self.assertEqual(2, len(self.requests))
        kwargs = self.requests[1][1]
        self.assertEqual(PIN, kwargs['expected_sha256'])
        self.assertEqual((640, 640), kwargs['input_size'])
        self.assertEqual('YOLOX-S entrance check', kwargs['model_label'])
        self.assertEqual('graphics', kwargs['engine'])
        self.assertEqual('f32', kwargs['gpu_precision'])
        self.assertEqual(1, kwargs['threads'])
        self.assertFalse(kwargs['allow_cpu_fallback'])
        tiny, extra = self.instances
        self.assertEqual(3, tiny.calls)
        self.assertEqual(1, extra.calls)
        self.assertEqual([(100, 100, 3)], extra.shapes)
        self.assertEqual({monitor._thread.ident}, set(tiny.thread_ids + extra.thread_ids))

    def test_four_views_keep_one_source_timestamp_and_one_temporal_observation(self):
        monitor = self.start()
        first, report, stamp = self.deliver(monitor)
        self.assertEqual([], first, 'four passes must not confirm the first source frame')
        self.assertEqual(1, len(report['detections']))
        self.assertEqual(.8, report['detections'][0]['confidence'])
        self.assertEqual(4, report['inference_views'])
        self.assertEqual(4, len(report['view_diagnostics']))
        for view in report['view_diagnostics']:
            self.assertEqual((stamp, monitor._epoch, monitor._worker_generation),
                             (view['observed_at'], view['epoch'], view['generation']))
        self.assertEqual(stamp, report['observed_at'])
        self.assertEqual(stamp, monitor._pipeline['last_delivery']['observed_at'])
        second, _, second_stamp = self.deliver(monitor)
        self.assertEqual(1, len(second))
        proof = monitor.event_context(second[0][2])
        self.assertEqual(2, proof['observations'])
        self.assertEqual(second_stamp, proof['observed_at'])
        self.assertEqual(2, monitor._pipeline['results_consumed'])

    def test_only_enabled_entrance_ground_points_and_classes_accept_s(self):
        extra = [detection((50, 20, 70, 60), .8),
                 detection((10, 20, 30, 60), .99),  # Exit cannot be upgraded by S.
                 detection((50, 20, 70, 60), .99, 'person'),
                 detection((50, 0, 70, 10), .99),
                 detection((50, 20, 70, 60), .49),
                 detection((-1, 20, 70, 60), .99)]
        tiny = lambda call: [detection((10, 20, 30, 60), .7)] if call % 3 == 1 else []
        monitor = self.start(tiny_output=tiny, extra_output=extra)
        _, report, _ = self.deliver(monitor)
        self.assertEqual(2, len(report['detections']))
        self.assertEqual([.7, .8], [row['confidence'] for row in report['detections']])
        self.assertEqual(1, report['entrance_verifier']['accepted'])
        monitor.zones[0].ai_enabled = False
        before = self.instances[1].calls
        _, report, _ = self.deliver(monitor)
        self.assertEqual(before, self.instances[1].calls)
        self.assertEqual(1, len(report['detections']))

    def test_missing_whole_view_never_adds_secondary_pass(self):
        monitor = self.start(context=False, tiny_output=[])
        _, report, _ = self.deliver(monitor)
        self.assertEqual(2, report['inference_views'])
        self.assertEqual(0, self.instances[1].calls)
        self.assertEqual('unavailable', report['entrance_verifier']['state'])
        self.assertIn('whole', report['entrance_verifier']['error'].lower())

    def test_missing_s_does_not_retry_or_prevent_primary_results(self):
        monitor = self.start(extra_init_error=FileNotFoundError('missing S fixture'))
        for _ in range(2):
            _, report, _ = self.deliver(monitor)
            self.assertEqual('ready', report['state'])
            self.assertEqual('unavailable', report['entrance_verifier']['state'])
            self.assertIn('missing S fixture', report['entrance_verifier']['error'])
            self.assertEqual(.55, report['detections'][0]['confidence'])
        self.assertEqual(2, len(self.requests))
        self.assertEqual(0, monitor.status()['recovery_attempts'])

    def test_cpu_secondary_is_rejected_without_running_it(self):
        monitor = self.start(extra_provider='CPU')
        _, report, _ = self.deliver(monitor)
        self.assertEqual('unavailable', report['entrance_verifier']['state'])
        self.assertEqual(0, self.instances[1].calls)
        self.assertEqual(.55, report['detections'][0]['confidence'])

    def test_runtime_secondary_gpu_loss_keeps_primary_and_disables_extra(self):
        def gpu_lost(detector, frame):
            detector.available = False
        monitor = self.start(callback=gpu_lost)
        for _ in range(2):
            _, report, _ = self.deliver(monitor)
            self.assertEqual('unavailable', report['entrance_verifier']['state'])
            self.assertEqual(.55, report['detections'][0]['confidence'])
            self.assertEqual('ready', report['state'])
        self.assertEqual(1, self.instances[1].calls)
        self.assertEqual(0, monitor.status()['recovery_attempts'])

    def test_over_budget_secondary_keeps_fresh_primary_and_disables_extra(self):
        def slow(_detector, _frame):
            self.offset += .501  # Controlled clock; no sleep/model workload.
        monitor = self.start(callback=slow)
        for _ in range(2):
            _, report, _ = self.deliver(monitor)
            self.assertEqual(.55, report['detections'][0]['confidence'])
            self.assertIn('500 ms', report['entrance_verifier']['error'])
        self.assertEqual(1, self.instances[1].calls)
        self.assertEqual(2, monitor._pipeline['results_published'])

    def test_stale_reset_off_and_stop_during_s_cannot_publish_old_source(self):
        for transition in ('stale', 'reset', 'off', 'stop'):
            with self.subTest(transition=transition):
                holder = {}
                def invalidate(_detector, _frame):
                    monitor = holder['monitor']
                    if transition == 'stale':
                        self.offset += 3.1
                    elif transition == 'reset':
                        monitor.reset()
                    elif transition == 'stop':
                        monitor.stop()
                    else:
                        monitor._last_submit = sm.time.monotonic()
                        monitor.update(self.frame(), False, stamp=sm.time.monotonic())
                monitor = self.start(callback=invalidate)
                holder['monitor'] = monitor
                self.submit(monitor)
                if transition == 'stop':
                    self.wait_for(lambda: not monitor._thread.is_alive())
                else:
                    self.assertTrue(self.finished.wait(1.))
                self.assertEqual(0, monitor._pipeline['results_published'])
                self.assertTrue(monitor._output.empty())
                self.assertEqual([], monitor.status()['detections'])
                self.assertEqual([], monitor.status()['view_diagnostics'])
                monitor.stop()

    def test_source_expiring_after_tiny_never_starts_secondary_or_publishes(self):
        def expires(detector, _frame):
            if detector.calls == 3:
                self.offset += 3.1
        monitor = self.start(tiny_callback=expires)
        self.submit(monitor)
        self.assertTrue(self.finished.wait(1.))
        self.assertEqual(3, self.instances[0].calls)
        self.assertEqual(0, self.instances[1].calls)
        self.assertEqual(0, monitor._pipeline['results_published'])
        self.assertTrue(monitor._output.empty())


if __name__ == '__main__':
    unittest.main()
