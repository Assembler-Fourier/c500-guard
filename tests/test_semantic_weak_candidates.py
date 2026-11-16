"""Optional weak motion evidence cannot weaken the ordinary semantic path."""
import time
import unittest
from unittest.mock import patch

import numpy as np

import test_semantic_latest_pending as pending_helpers


def detected(score, box, label='car'):
    return {'label': label, 'confidence': round(score, 4), 'raw_confidence': score,
            'bbox': list(box)}


class WeakCandidateTests(unittest.TestCase):
    make_monitor = pending_helpers.LatestPendingTests.make_monitor
    frame = staticmethod(pending_helpers.LatestPendingTests.frame)
    start_worker = staticmethod(pending_helpers.LatestPendingTests.start_worker)
    await_condition = staticmethod(pending_helpers.LatestPendingTests.await_condition)

    def deliver(self, monitor, number):
        monitor._last_submit = 0.
        stamp = time.monotonic()
        monitor.update(self.frame(number), True, stamp=stamp)
        self.await_condition(lambda: monitor.status()['inferences'] == number)
        report = monitor.status()
        monitor._last_submit = time.monotonic()
        events = monitor.update(self.frame(number), True)
        return events, report, stamp

    def test_only_explicitly_enabled_ai_zone_lowers_diagnostic_floor(self):
        monitor = self.make_monitor()
        zone = monitor.zones[0]
        zone.ai_confidence = .5
        self.assertEqual(.5, monitor._inference_confidence(.5))
        for value in ('true', 1, False, None):
            zone.approach_motion = value
            self.assertEqual(.5, monitor._inference_confidence(.5))
        zone.approach_motion = True
        self.assertEqual(.3, monitor._inference_confidence(.5))
        self.assertEqual(.2, monitor._inference_confidence(.2), 'an existing lower normal floor is retained')
        zone.ai_enabled = False
        self.assertEqual(.5, monitor._inference_confidence(.5))
        zone.ai_enabled = True
        zone.classes = []
        self.assertEqual(.5, monitor._inference_confidence(.5))

    def test_weak_focused_and_context_objects_are_diagnostic_only(self):
        monitor = self.make_monitor(interval=.1, context_view=True)
        monitor.zones[0].ai_confidence = .5
        monitor.zones[0].approach_motion = True
        observed_thresholds = []

        class Detector:
            available = True
            def status(self):
                return {'available': True, 'confidence_threshold': self.confidence}
            def detect(self, frame):
                observed_thresholds.append(self.confidence)
                box = (20, 20, 40, 60) if frame.shape[1] == 60 else (75, 20, 95, 60)
                return [detected(.4574, box, 'truck')]

        def create(model, confidence, threads):
            instance = Detector()
            instance.confidence = confidence
            return instance

        with patch('object_detector.ObjectDetector', side_effect=create) as factory, \
             patch('semantic_monitor.inference_rects', return_value=[(0, 0, 60, 100)]):
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            for number in (1, 2):
                events, report, stamp = self.deliver(monitor, number)
                self.assertEqual([], events)
                self.assertEqual([], report['detections'])
                self.assertEqual(0, monitor.status()['tracked_objects'])
                self.assertEqual(.5, report['confidence_threshold'])
                self.assertEqual(.3, report['diagnostic_confidence_threshold'])
                self.assertEqual(2, len(report['view_diagnostics']))
                for view in report['view_diagnostics']:
                    self.assertEqual(stamp, view['observed_at'])
                    self.assertEqual(monitor._epoch, view['epoch'])
                    self.assertEqual(monitor._worker_generation, view['generation'])
                    self.assertEqual(0, view['normal_detection_count'])
                    self.assertEqual(.4574, view['detections'][0]['confidence'])
            self.assertEqual([.3]*4, observed_thresholds)
            self.assertEqual(1, factory.call_count)
            self.assertEqual(2, monitor.status()['pipeline']['results_consumed'])
            monitor.reset()
            self.assertEqual([], monitor.status()['view_diagnostics'])
            monitor.stop()

    def test_views_still_count_as_one_normal_observation_and_weak_cannot_join(self):
        monitor = self.make_monitor(interval=.1, context_view=True)
        monitor.zones[0].ai_confidence = .5
        monitor.zones[0].approach_motion = True

        class Detector:
            available = True
            calls = 0
            def status(self):
                return {'available': True}
            def detect(self, frame):
                self.calls += 1
                return [detected(.8, (20, 20, 40, 60)),
                        detected(.4, (75, 20, 95, 60), 'truck')]

        detector = Detector()
        with patch('object_detector.ObjectDetector', return_value=detector), \
             patch('semantic_monitor.inference_rects', return_value=[(0, 0, 60, 100)]):
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            first, _, _ = self.deliver(monitor, 1)
            self.assertEqual([], first, 'two passes of one frame cannot confirm an arrival')
            self.assertEqual(1, monitor.status()['tracked_objects'])
            second, _, _ = self.deliver(monitor, 2)
            self.assertEqual(1, len(second))
            proof = monitor.event_context(second[0][2])
            self.assertEqual('car', proof['label'])
            self.assertEqual(2, proof['observations'])
            self.assertEqual(4, detector.calls)
            monitor.stop()

    def test_toggle_uses_original_floor_without_reloading_or_extra_passes(self):
        monitor = self.make_monitor(interval=.1)
        zone = monitor.zones[0]
        zone.ai_confidence = .5
        seen = []

        class Detector:
            available = True
            def status(self):
                return {'available': True, 'confidence_threshold': self.confidence}
            def detect(self, frame):
                seen.append(self.confidence)
                candidates = [detected(.4, (20, 20, 40, 60)), detected(.8, (60, 20, 80, 60))]
                return [d for d in candidates if d['raw_confidence'] >= self.confidence]

        def create(model, confidence, threads):
            instance = Detector()
            instance.confidence = confidence
            return instance

        with patch('object_detector.ObjectDetector', side_effect=create) as factory:
            self.start_worker(monitor)
            self.await_condition(lambda: monitor.status()['available'])
            _, before, _ = self.deliver(monitor, 1)
            zone.approach_motion = True
            _, enabled, _ = self.deliver(monitor, 2)
            zone.approach_motion = False
            _, after, _ = self.deliver(monitor, 3)
            self.assertEqual([.5, .3, .5], seen)
            self.assertEqual([1, 2, 1], [r['view_diagnostics'][0]['detection_count']
                                       for r in (before, enabled, after)])
            self.assertEqual(before['detections'], enabled['detections'])
            self.assertEqual(before['detections'], after['detections'])
            self.assertEqual(1, factory.call_count)
            monitor.stop()

    def test_normal_filter_uses_exact_decoder_score_at_rounded_boundary(self):
        from semantic_monitor import SemanticMonitor
        raw = [detected(.49996, (1, 1, 10, 10)), detected(.50004, (20, 1, 30, 10))]
        self.assertEqual([.5, .5], [d['confidence'] for d in raw])
        accepted = SemanticMonitor._normal_detections(raw, .5)
        self.assertEqual([raw[1]], accepted)
        for bad in (True, float('nan'), float('inf'), -.1, 1.1, '0.8'):
            self.assertEqual([], SemanticMonitor._normal_detections(
                [{'confidence': .8, 'raw_confidence': bad}], .5))

    def test_real_decoder_boundary_is_identical_with_weak_collection(self):
        from object_detector import decode_detections
        from semantic_monitor import SemanticMonitor
        for threshold in (.5, .55, .6, .7):
            center = np.float32(threshold)
            for score in (np.nextafter(center, np.float32(0.)), center,
                          np.nextafter(center, np.float32(1.))):
                with self.subTest(threshold=threshold, score=float(score)):
                    raw = np.zeros((1, 3549, 85), np.float32)
                    raw[0, 0, :4] = [1, 1, 1, 1]
                    raw[0, 0, 4], raw[0, 0, 7] = score, 1.
                    baseline = decode_detections(raw, (416, 416), 1., (416, 416), confidence=threshold)
                    weak = decode_detections(raw, (416, 416), 1., (416, 416), confidence=.3,
                                             normal_confidence=threshold)
                    accepted = SemanticMonitor._normal_detections(weak, threshold)
                    self.assertEqual(baseline, accepted)
                    self.assertEqual(float(score), weak[0]['raw_confidence'])

    def test_weak_rounding_ties_cannot_evict_normal_vehicle_from_output_cap(self):
        from object_detector import decode_detections
        from semantic_monitor import SemanticMonitor
        raw = np.zeros((1, 3549, 85), np.float32)
        for index in range(50):
            raw[0, index, :4] = [.5, .5, 0, 0]
            raw[0, index, 4], raw[0, index, 5] = .49996, 1.  # separate weak people
        raw[0, 100, :4] = [.5, .5, 0, 0]
        raw[0, 100, 4], raw[0, 100, 7] = .50004, 1.  # one true normal car
        baseline = decode_detections(raw, (416, 416), 1., (416, 416), confidence=.5)
        weak = decode_detections(raw, (416, 416), 1., (416, 416), confidence=.3, normal_confidence=.5)
        self.assertEqual(1, len(baseline))
        self.assertEqual('car', baseline[0]['label'])
        self.assertEqual(50, len(weak))
        self.assertEqual(baseline, SemanticMonitor._normal_detections(weak, .5))


if __name__ == '__main__':
    unittest.main()
