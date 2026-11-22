"""Whole-picture entrance reuse: synthetic frames/models, no camera or GPU work."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.environ.get('BW_SECONDARY_SOURCE', str(Path(__file__).resolve().parents[1] / 'src')))
import semantic_monitor as sm
from regional_inference import inference_rects

REAL_MONOTONIC = time.monotonic
GPU = 'Intel graphics (OpenVINO)'


class Zone(SimpleNamespace):
    def contains_point(self, x, y):
        left, top, width, height = self.rect
        return left <= x <= left + width and top <= y <= top + height


def zone(**changes):
    return Zone(**(dict(name='Entrance', rect=(0., 0., 1., 1.), ai_enabled=True,
                       classes=['car'], ai_confidence=.5, motion=False,
                       route_role='entrance', severity='main', arrival_direction=None,
                       ai_confirm_observations=2) | changes))


def car(box=(50, 20, 70, 60), score=.8, label='car'):
    return dict(label=label, confidence=score, raw_confidence=score, bbox=list(box))


class FakeDetector:
    def __init__(self, name, confidence, rows, callback):
        self.name, self.confidence, self.rows, self.callback = name, confidence, rows, callback
        self.available, self.provider, self.calls = True, GPU, 0
        self.shapes, self.thread_ids = [], []

    def status(self):
        return dict(available=self.available, provider=self.provider, model=self.name, error='')

    def detect(self, frame):
        self.calls += 1
        self.shapes.append(frame.shape)
        self.thread_ids.append(threading.get_ident())
        if self.callback:
            self.callback(self, frame)
        rows = self.rows(self.calls) if callable(self.rows) else self.rows
        return [copy.deepcopy(row) for row in rows if row['raw_confidence'] >= self.confidence]


class WholeFrameVerifierRegressionTests(unittest.TestCase):
    def start(self, *, zones=None, context=True, primary=(), secondary=None,
              primary_callback=None, secondary_callback=None, shape=(100, 100, 3)):
        self.now = 100.
        self.frame = np.zeros(shape, dtype=np.uint8)
        self.models, self.requests = [], []
        self.enterContext(patch.object(sm.time, 'monotonic', side_effect=lambda: self.now))
        with patch.object(threading.Thread, 'start'):
            monitor = sm.SemanticMonitor(zones if zones is not None else [zone()], Path('primary.fake'),
                                         interval=.25, context_view=context,
                                         entrance_verifier_path=Path('secondary.fake'))
        self.addCleanup(monitor.stop)

        def factory(path, **kwargs):
            self.requests.append((str(path), kwargs))
            extra = Path(path).name == 'secondary.fake'
            detector = FakeDetector('secondary' if extra else 'primary', kwargs['confidence'],
                                    ([car()] if secondary is None else secondary) if extra else primary,
                                    secondary_callback if extra else primary_callback)
            self.models.append(detector)
            return detector

        self.enterContext(patch('object_detector.ObjectDetector', side_effect=factory))
        self.finished = threading.Event()
        original_get = monitor._input.get
        got_source = False

        def get_work(*args, **kwargs):
            nonlocal got_source
            if kwargs.get('timeout') == .25 and got_source:
                self.finished.set()
            value = original_get(*args, **kwargs)
            if kwargs.get('timeout') == .25:
                got_source = True
            return value

        self.enterContext(patch.object(monitor._input, 'get', side_effect=get_work))
        monitor._thread = threading.Thread(target=monitor._run, daemon=True)
        monitor._thread.start()
        self.wait_for(lambda: monitor.status()['available'])
        return monitor

    @staticmethod
    def wait_for(predicate):
        deadline = REAL_MONOTONIC() + 2.
        while REAL_MONOTONIC() < deadline:
            if predicate():
                return
            time.sleep(.001)
        raise AssertionError('Synthetic worker exceeded its two-second operation bound')

    def submit(self, monitor):
        self.now += .25
        self.finished.clear()
        monitor._last_submit = 0.
        monitor.update(self.frame, True, stamp=self.now)
        return self.now

    def deliver(self, monitor):
        before = monitor._pipeline['results_published']
        stamp = self.submit(monitor)
        self.wait_for(lambda: monitor._pipeline['results_published'] == before + 1)
        report = monitor.status()
        monitor._last_submit = self.now
        events = monitor.update(self.frame, True, stamp=self.now)
        return events, report, stamp

    def test_full_focused_frame_reuses_pixels_without_duplicate_primary_or_observation(self):
        monitor = self.start()
        first, report, stamp = self.deliver(monitor)
        self.assertEqual([1, 1], [model.calls for model in self.models])
        self.assertEqual([], first)
        self.assertEqual(2, report['inference_views'])
        self.assertEqual([[0, 0, 100, 100]], report['inference_crops'])
        self.assertEqual(['focused', 'entrance verifier'], [d['kind'] for d in report['view_diagnostics']])
        self.assertEqual('ready', report['entrance_verifier']['state'])
        self.assertEqual(stamp, report['observed_at'])
        for diagnostic in report['view_diagnostics']:
            self.assertEqual((stamp, monitor._epoch, monitor._worker_generation),
                             (diagnostic['observed_at'], diagnostic['epoch'], diagnostic['generation']))
        second, _, second_stamp = self.deliver(monitor)
        self.assertEqual(1, len(second))
        proof = monitor.event_context(second[0][2])
        self.assertEqual(2, proof['observations'])
        self.assertEqual(second_stamp, proof['observed_at'])
        self.assertEqual(2, monitor._pipeline['results_consumed'])
        self.assertEqual({monitor._thread.ident}, set(sum((m.thread_ids for m in self.models), [])))

    def test_padding_and_integer_rounding_reuse_actual_full_rectangle(self):
        # The left/top bounds are positive before int(), but round to zero.
        monitor = self.start(zones=[zone(rect=(.061, .101, .88, .86))])
        self.assertEqual([(0, 0, 100, 100)], inference_rects(monitor.zones, 100, 100))
        _, report, _ = self.deliver(monitor)
        self.assertEqual([1, 1], [m.calls for m in self.models])
        self.assertEqual([50., 20., 70., 60.], report['detections'][0]['bbox'])

    def test_broad_multi_region_union_reuses_actual_full_rectangle(self):
        zones = [zone(rect=(0., 0., .4, 1.)), zone(name='Other entrance', rect=(.6, 0., .4, 1.))]
        monitor = self.start(zones=zones, secondary=[car((65, 20, 85, 60))])
        self.assertEqual([(0, 0, 100, 100)], inference_rects(zones, 100, 100))
        _, report, _ = self.deliver(monitor)
        self.assertEqual([1, 1], [m.calls for m in self.models])
        self.assertEqual(1, report['entrance_verifier']['accepted'])

    def test_full_reuse_preserves_primary_geometry_and_merges_duplicate_boxes(self):
        monitor = self.start(shape=(100, 200, 3), primary=[car((100, 20, 140, 60), .7)],
                             secondary=[car((100, 20, 140, 60), .9)])
        first, report, _ = self.deliver(monitor)
        self.assertEqual([], first)
        self.assertEqual([1, 1], [m.calls for m in self.models])
        self.assertEqual([(100, 200, 3)], self.models[1].shapes)
        self.assertEqual(1, len(report['detections']))
        self.assertEqual([100., 20., 140., 60.], report['detections'][0]['bbox'])
        self.assertEqual(.9, report['detections'][0]['confidence'])
        second, _, _ = self.deliver(monitor)
        self.assertEqual(1, len(second))
        self.assertEqual(2, monitor.event_context(second[0][2])['observations'])

    def test_disabled_context_does_not_enable_secondary_from_full_focused_view(self):
        monitor = self.start(context=False, primary=[car()])
        _, report, _ = self.deliver(monitor)
        self.assertEqual([1, 0], [m.calls for m in self.models])
        self.assertEqual(1, report['inference_views'])
        self.assertEqual(1, len(report['detections']))

    def test_disabled_entrance_and_empty_classes_do_not_enable_secondary(self):
        monitor = self.start(zones=[zone(ai_enabled=False), zone(name='Exit', route_role='exit')])
        self.deliver(monitor)
        self.assertEqual([1, 0], [m.calls for m in self.models])
        monitor.zones[0].ai_enabled, monitor.zones[0].classes = True, []
        monitor.reset()
        self.deliver(monitor)
        self.assertEqual([2, 0], [m.calls for m in self.models])

    def test_full_focused_marker_does_not_relax_entrance_classes_or_region(self):
        zones = [zone(rect=(.5, .2, .4, .6)), zone(name='Whole exit', route_role='exit')]
        rows = [car(), car(label='person'), car((10, 20, 30, 60)), car((50, 0, 70, 10)),
                car((-1, 20, 70, 60)), car(score=.49)]
        monitor = self.start(zones=zones, secondary=rows)
        _, report, _ = self.deliver(monitor)
        self.assertEqual(1, self.models[1].calls)
        self.assertEqual(2, self.models[0].calls)
        self.assertEqual(1, len(report['detections']))
        self.assertEqual(1, report['entrance_verifier']['accepted'])
        self.assertEqual([50., 20., 70., 60.], report['detections'][0]['bbox'])

    def test_origin_crop_that_is_smaller_than_frame_is_never_mistaken_for_whole(self):
        monitor = self.start(zones=[zone(rect=(0., 0., .4, .4))], secondary=[])
        _, report, _ = self.deliver(monitor)
        self.assertEqual([2, 1], [m.calls for m in self.models])
        self.assertEqual([(100, 100, 3)], self.models[1].shapes)
        self.assertEqual(['focused', 'context', 'entrance verifier'],
                         [d['kind'] for d in report['view_diagnostics']])

    def test_profile_reset_between_narrow_and_full_needs_two_fresh_sources(self):
        monitor = self.start(zones=[zone(rect=(.4, .2, .5, .6))])
        first, _, _ = self.deliver(monitor)
        self.assertEqual([], first)
        old_epoch = monitor._epoch
        monitor.zones = [zone()]
        monitor.reset()
        after_reset, report, _ = self.deliver(monitor)
        self.assertEqual([], after_reset)
        self.assertGreater(report['tracking_generation'][0], old_epoch)
        self.assertEqual(2, self.models[1].calls)
        final, _, _ = self.deliver(monitor)
        self.assertEqual(1, len(final))
        self.assertEqual(2, monitor.event_context(final[0][2])['observations'])
        self.assertEqual([4, 3], [m.calls for m in self.models])
        monitor.zones = [zone(rect=(.4, .2, .5, .6))]
        monitor.reset()
        again, report, _ = self.deliver(monitor)
        self.assertEqual([], again)
        self.assertEqual(3, report['inference_views'])
        self.assertEqual([6, 4], [m.calls for m in self.models])

    def test_full_frame_duplicate_source_cannot_add_an_observation_or_model_pass(self):
        monitor = self.start()
        _, _, stamp = self.deliver(monitor)
        before = monitor._pipeline['inputs_submitted']
        monitor._last_submit = 0.
        self.assertEqual([], monitor.update(self.frame, True, stamp=stamp))
        self.assertEqual(before, monitor._pipeline['inputs_submitted'])
        self.assertEqual([1, 1], [m.calls for m in self.models])
        self.assertEqual(1, monitor._pipeline['results_consumed'])

    def test_reset_during_full_secondary_discards_inflight_old_geometry(self):
        holder = {}
        def reset_once(detector, _frame):
            if detector.calls == 1:
                holder['monitor'].zones = [zone(rect=(.4, .2, .5, .6))]
                holder['monitor'].reset()
        monitor = self.start(secondary_callback=reset_once)
        holder['monitor'] = monitor
        self.submit(monitor)
        self.assertTrue(self.finished.wait(2.))
        self.assertEqual(1, self.models[1].calls)
        self.assertEqual(0, monitor._pipeline['results_published'])
        self.assertTrue(monitor._output.empty())
        recovered, _, _ = self.deliver(monitor)
        self.assertEqual([], recovered)
        confirmed, _, _ = self.deliver(monitor)
        self.assertEqual(1, len(confirmed))

    def test_slow_full_secondary_discards_late_boxes_and_recovers_on_same_worker(self):
        def slow_once(detector, _frame):
            if detector.calls == 1:
                self.now += .501
        monitor = self.start(secondary_callback=slow_once)
        worker = monitor._thread
        first, report, _ = self.deliver(monitor)
        self.assertEqual([], first)
        self.assertEqual([], report['detections'])
        self.assertEqual('cooldown', report['entrance_verifier']['state'])
        self.now += 5.1
        recovered, report, _ = self.deliver(monitor)
        self.assertEqual([], recovered)
        self.assertEqual('ready', report['entrance_verifier']['state'])
        confirmed, _, _ = self.deliver(monitor)
        self.assertEqual(1, len(confirmed))
        self.assertEqual([3, 3], [m.calls for m in self.models])
        self.assertIs(worker, monitor._thread)
        self.assertEqual(2, len(self.requests))

    def test_full_secondary_respects_primary_freshness_reserve(self):
        def primary_delay(detector, _frame):
            if detector.calls == 1:
                self.now += 2.6
        monitor = self.start(primary=[car()], primary_callback=primary_delay)
        _, report, stamp = self.deliver(monitor)
        self.assertEqual([1, 0], [m.calls for m in self.models])
        self.assertEqual('deferred', report['entrance_verifier']['state'])
        self.assertEqual(stamp, report['observed_at'])
        self.assertEqual(2.6, report['age_s'])
        self.assertEqual(1, len(report['detections']))


if __name__ == '__main__':
    unittest.main()
