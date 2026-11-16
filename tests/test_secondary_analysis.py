"""Real image measurements and worker-lifecycle regressions; no camera/audio."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1]/'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import secondary_analysis as sa


class FakeDetector:
    def __init__(self, objects=()):
        self.available = True
        self.objects = list(objects)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [dict(o) for o in self.objects]

    def status(self):
        return {'available': self.available, 'error': ''}


def texture(level=100):
    rng = np.random.default_rng(582)
    gray = np.clip(level+rng.integers(-25, 26, (120, 200)), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def primary(objects=(), age=.1, available=True):
    return {'available': available, 'state': 'ready', 'age_s': age, 'detections': list(objects)}


class SecondaryImageTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = sa.SecondaryAnalyzer('not-loaded.onnx')
        self.analyzer._state['enabled'] = True
        self.detector = FakeDetector()

    def inspect(self, frame, *, detector=None, main=None, alarm=None, alignment=None, zones=()):
        stamp = time.monotonic()
        return self.analyzer._analyze(frame, sa._zones(zones), primary() if main is None else main,
                                      {'accepted': True} if alignment is None else alignment,
                                      alarm or {}, stamp, detector or self.detector,
                                      self.analyzer._generation)

    def test_constructor_does_not_start_or_load_model(self):
        self.assertIsNone(self.analyzer._thread)
        self.assertEqual(self.detector.calls, 0)

    def test_all_diagnostic_images_are_decodable_measured_images(self):
        report, images = self.inspect(texture())
        self.assertEqual(set(images), set(sa.VIEWS))
        self.assertGreater(report['landmarks']['count'], 0)
        for name, data in images.items():
            decoded = sa.cv2.imdecode(np.frombuffer(data, np.uint8), sa.cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (120, 200, 3), name)
        self.assertFalse(report['landmarks']['zone_boundaries_verified'])

    def test_uniform_covered_image_is_unreliable_and_never_learned(self):
        self.inspect(texture())
        report, _ = self.inspect(np.full((120, 200, 3), 58, np.uint8))
        self.assertEqual(report['verdict'], 'view_unreliable')
        self.assertIn('possible obstruction', ' '.join(report['quality']['reasons']))
        self.assertFalse(report['learning']['updated'])

    def test_dark_and_overexposed_frames_are_unreliable(self):
        for value in (0, 255):
            report, _ = self.inspect(np.full((120, 200, 3), value, np.uint8))
            self.assertFalse(report['quality']['reliable'])
            self.assertFalse(report['learning']['updated'])

    def test_local_headlight_change_is_measured_and_freezes_learning(self):
        frame = texture(60)
        self.inspect(frame)
        changed = frame.copy()
        changed[50:65, 20:45] = 220
        report, _ = self.inspect(changed)
        self.assertGreater(report['quality']['change_fraction'], .005)
        self.assertLess(abs(report['quality']['exposure_shift']), 2)
        self.assertFalse(report['learning']['updated'])

    def test_global_exposure_shift_is_separated_from_local_change(self):
        frame = texture(70)
        self.inspect(frame)
        report, _ = self.inspect((frame+60).astype(np.uint8))
        self.assertAlmostEqual(report['quality']['exposure_shift'], 60, delta=1)
        self.assertLess(report['quality']['change_fraction'], .01)
        self.assertGreater(report['quality']['raw_change_fraction'], .95)
        self.assertFalse(report['learning']['updated'])

    def test_stable_clear_frames_update_statistics_only(self):
        frame = texture()
        first, _ = self.inspect(frame)
        second, _ = self.inspect(frame)
        self.assertFalse(first['learning']['updated'])
        self.assertTrue(second['learning']['updated'])
        self.assertEqual(second['learning']['samples'], 1)
        self.assertFalse(second['learning']['controls_alarms'])

    def test_alarm_objects_or_stale_primary_freeze_calibration(self):
        frame = texture()
        self.inspect(frame)
        for kwargs in ({'alarm': {'id': 1}}, {'main': primary(age=5)},
                       {'detector': FakeDetector([{'label': 'car', 'confidence': .8, 'bbox': [20, 20, 70, 70]}])}):
            report, _ = self.inspect(frame, **kwargs)
            self.assertFalse(report['learning']['updated'], kwargs)

    def test_missing_model_reports_diagnostics_and_no_false_alarm_claim(self):
        self.detector.available = False
        report, _ = self.inspect(texture(), alarm={'id': 1})
        self.assertEqual(report['state'], 'diagnostics_only')
        self.assertFalse(report['available'])
        self.assertNotEqual(report['verdict'], 'false_alarm')
        self.assertFalse(report['learning']['updated'])

    def test_invalid_alignment_skips_model_and_learning(self):
        report, _ = self.inspect(texture(), alignment={'accepted': True, 'frame_aligned': False})
        self.assertEqual(self.detector.calls, 0)
        self.assertEqual(report['verdict'], 'view_unreliable')
        self.assertFalse(report['learning']['updated'])

    def test_dark_two_model_passes_are_one_observation(self):
        car = {'label': 'car', 'confidence': .8, 'bbox': [30, 30, 80, 85], 'confirmed_regions': ['Exit']}
        zones = [{'name': 'Exit', 'rect': [0., 0., 1., 1.]}]
        detector = FakeDetector([car])
        first, _ = self.inspect(texture(50), detector=detector, main=primary([car]), zones=zones)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(first['passes'], 2)
        self.assertEqual(len(first['detections']), 1)
        self.assertEqual(first['detections'][0]['observations'], 1)
        self.assertFalse(first['detections'][0]['confirmed'])
        second, _ = self.inspect(texture(50), detector=detector, main=primary([car]), zones=zones)
        self.assertEqual(second['verdict'], 'corroborated')
        self.assertEqual(second['comparison']['repeated_matches'], 1)

    def test_empty_second_view_does_not_veto_main_object(self):
        car = {'label': 'car', 'confidence': .8, 'bbox': [30, 30, 80, 85]}
        report, _ = self.inspect(texture(), main=primary([car]), alarm={'id': 1})
        self.assertEqual(report['verdict'], 'unconfirmed')
        self.assertEqual(report['comparison']['agreement'], 'not_seen_again')
        self.assertIn('does not establish a false alarm', report['comparison']['note'])

    def test_primary_match_uses_object_family_and_location(self):
        detector = FakeDetector([{'label': 'truck', 'confidence': .8, 'bbox': [10, 10, 50, 60]}])
        main = primary([{'label': 'car', 'bbox': [110, 10, 150, 60]}])
        report, _ = self.inspect(texture(), detector=detector, main=main)
        self.assertEqual(report['comparison']['matched_count'], 0)

    def test_primary_freshness_rejects_invalid_ages_and_error_states(self):
        for age in (-1, float('nan'), float('inf'), '0.1', True, None, 3):
            report, _ = self.inspect(texture(), main=primary(age=age))
            self.assertFalse(report['comparison']['primary_fresh'], age)
        for state in ('error', 'retrying', 'loading', 'off'):
            main = primary()
            main['state'] = state
            report, _ = self.inspect(texture(), main=main)
            self.assertFalse(report['comparison']['primary_fresh'], state)

    def test_repeated_non_route_object_is_never_strong_corroboration(self):
        car = {'label': 'car', 'confidence': .8, 'bbox': [30, 30, 80, 85]}
        detector = FakeDetector([car])
        zones = [{'name': 'Exit', 'rect': [0., 0., 1., 1.]}]
        self.inspect(texture(), detector=detector, main=primary([car]), zones=zones)
        report, _ = self.inspect(texture(), detector=detector, main=primary([car]), zones=zones)
        self.assertEqual(report['comparison']['repeated_object_matches'], 1)
        self.assertEqual(report['comparison']['repeated_matches'], 0)
        self.assertEqual(report['verdict'], 'unconfirmed')

    def test_other_route_objects_do_not_look_missed_but_freeze_learning(self):
        frame = texture()
        zones = [{'name': 'Exit', 'rect': [0., 0., .2, .2]}]
        self.inspect(frame, zones=zones)
        main = primary([{'label': 'car', 'bbox': [140, 60, 190, 100]}])
        report, _ = self.inspect(frame, main=main, zones=zones)
        self.assertEqual(report['comparison']['primary_count'], 0)
        self.assertEqual(report['comparison']['agreement'], 'no_primary_object')
        self.assertFalse(report['learning']['updated'])
        self.assertEqual(report['learning']['frozen_reason'], 'An object is present')

    def test_round_robin_keeps_temporal_memory_for_each_route(self):
        zones = [{'name': 'Exit', 'route_group': 'exit', 'rect': [0., 0., 1., 1.]},
                 {'name': 'Entrance', 'route_group': 'entrance', 'rect': [0., 0., 1., 1.]}]
        car = {'label': 'car', 'confidence': .8, 'bbox': [30, 30, 80, 85],
               'confirmed_regions': ['Exit']}
        detector = FakeDetector([car])
        first, _ = self.inspect(texture(), zones=zones, detector=detector, main=primary([car]))
        middle, _ = self.inspect(texture(), zones=zones, detector=detector, main=primary([car]))
        third, _ = self.inspect(texture(), zones=zones, detector=detector, main=primary([car]))
        self.assertEqual([r['route'] for r in (first, middle, third)], ['exit', 'entrance', 'exit'])
        self.assertEqual(third['detections'][0]['observations'], 2)
        self.assertEqual(third['verdict'], 'corroborated')
        self.assertFalse(middle['detections'][0]['route_confirmed'])

    def test_adaptive_noise_threshold_is_bounded_and_never_relaxes_learning(self):
        frame = texture()
        self.inspect(frame)
        self.analyzer._baselines['dim'] = {'samples': 20, 'noise': 100, 'brightness': 100, 'blur': 100}
        changed = frame.copy()
        changed[40:60, 50:80] += 20
        report, _ = self.inspect(changed)
        self.assertEqual(report['quality']['change_threshold'], 28.)
        self.assertTrue(report['quality']['noise_baseline_used'])
        self.assertGreater(report['quality']['change_fraction'], .003)
        self.assertEqual(report['quality']['adaptive_change_fraction'], 0)
        self.assertFalse(report['learning']['updated'])
        strong = frame.copy()
        strong[40:60, 50:80] = 240
        report, _ = self.inspect(strong)
        self.assertGreater(report['quality']['adaptive_change_fraction'], .003)

    def test_active_alarm_selects_route_and_crop_offsets_map_back(self):
        zones = [{'name': 'Exit', 'route_group': 'exit', 'rect': [.05, .5, .2, .3]},
                 {'name': 'Entrance', 'route_group': 'entrance', 'rect': [.6, .2, .3, .3]}]
        detector = FakeDetector([{'label': 'car', 'confidence': .8, 'bbox': [2, 4, 20, 30]}])
        report, _ = self.inspect(texture(), zones=zones, detector=detector,
                                  alarm={'route_group': 'entrance'})
        self.assertEqual(report['route'], 'entrance')
        self.assertGreater(report['crop'][0], 80)
        self.assertEqual(report['detections'][0]['bbox'][0], report['crop'][0]+2)

    def test_rectification_is_real_but_explicitly_approximate(self):
        frame = texture()
        result, detail = self.analyzer._rectify(frame, [[.14, .095], [.818, .095], [.906, .811],
                                                         [.735, .893], [.092, .865], [.103, .375]], (200, 120))
        self.assertTrue(detail['applied'])
        self.assertFalse(detail['calibration_verified'])
        self.assertFalse(detail['used_for_inference'])
        self.assertFalse(np.array_equal(result, frame))
        for invalid in (None, [], [[0, 0]]*4, [[0, 0], [1, 0], [float('nan'), 1], [0, 1]]):
            _, rejected = self.analyzer._rectify(frame, invalid, (200, 120))
            self.assertFalse(rejected['applied'])

    def test_stale_or_stopped_result_cannot_learn(self):
        frame = texture()
        self.inspect(frame)
        self.analyzer._stop.set()
        report, _ = self.inspect(frame)
        self.assertFalse(report['learning']['updated'])
        self.assertIn('stopped', report['learning']['frozen_reason'])


class SecondaryLifecycleTests(unittest.TestCase):
    def test_submit_never_waits_behind_worker_lock_or_holds_main_lock_waiting(self):
        analyzer = sa.SecondaryAnalyzer('unused')
        analyzer._state['enabled'] = True
        held, release = threading.Event(), threading.Event()
        main_lock = threading.Lock()

        def hold_worker_lock():
            with analyzer._lock:
                held.set()
                release.wait(2)

        worker = threading.Thread(target=hold_worker_lock)
        worker.start()
        try:
            self.assertTrue(held.wait(1))
            started = time.monotonic()
            with main_lock:
                accepted = analyzer.submit(texture(), [], primary(), {'accepted': True}, {})
            elapsed = time.monotonic()-started
            self.assertFalse(accepted)
            self.assertLess(elapsed, .2)
            self.assertTrue(main_lock.acquire(blocking=False))
            main_lock.release()
            self.assertTrue(analyzer._queue.empty())
            self.assertEqual(analyzer._last_submit, -float('inf'))
            self.assertEqual(analyzer._state['dropped'], 0)
        finally:
            release.set()
            worker.join(timeout=2)
        # Contention did not consume the interval: the very next call succeeds.
        self.assertTrue(analyzer.submit(texture(), [], primary(), {'accepted': True}, {}))

    def test_latest_only_queue_replaces_pending_frame(self):
        analyzer = sa.SecondaryAnalyzer('unused')
        analyzer._state['enabled'] = True
        with patch.object(sa.time, 'monotonic', return_value=100.):
            self.assertTrue(analyzer.submit(texture(50), [], {}, {}, {}, stamp=100))
        with patch.object(sa.time, 'monotonic', return_value=102.):
            self.assertTrue(analyzer.submit(texture(100), [], {}, {}, {}, stamp=102))
        self.assertEqual(analyzer._queue.qsize(), 1)
        self.assertEqual(analyzer._queue.get_nowait()[1], 102)
        self.assertEqual(analyzer._state['dropped'], 1)

    def test_stale_input_invalid_image_and_future_input_rejected(self):
        analyzer = sa.SecondaryAnalyzer('unused')
        analyzer._state['enabled'] = True
        with patch.object(sa.time, 'monotonic', return_value=100.):
            self.assertFalse(analyzer.submit(texture(), [], {}, {}, {}, stamp=90))
            self.assertFalse(analyzer.submit(texture(), [], {}, {}, {}, stamp=101))
            self.assertFalse(analyzer.submit(np.zeros((10, 20), np.uint8), [], {}, {}, {}))

    def test_old_results_and_images_expire(self):
        analyzer = sa.SecondaryAnalyzer('unused')
        analyzer._state.update(enabled=True, detections=[{'label': 'car'}], verdict='corroborated')
        analyzer._images['crop'] = b'old'
        analyzer._published_at = time.monotonic()-10
        self.assertEqual(analyzer.status()['state'], 'stale')
        self.assertEqual(analyzer.status()['detections'], [])
        self.assertIsNone(analyzer.image('crop'))
        self.assertIsNone(analyzer.image('invalid'))

    def test_stop_invalidates_in_flight_work_and_does_not_resurrect(self):
        entered, release = threading.Event(), threading.Event()
        detector = FakeDetector()
        original = detector.detect

        def delayed(frame):
            entered.set()
            release.wait(2)
            return original(frame)

        detector.detect = delayed
        with patch.object(sa, 'ObjectDetector', return_value=detector):
            analyzer = sa.SecondaryAnalyzer('unused')
            analyzer.start()
            self.assertTrue(analyzer.submit(texture(), [], primary(), {'accepted': True}, {}))
            self.assertTrue(entered.wait(2))
            self.assertFalse(analyzer.stop(timeout=.01))
            release.set()
            self.assertTrue(analyzer.stop(timeout=2))
            self.assertEqual(analyzer.status()['state'], 'off')
            self.assertEqual(analyzer.status()['processed'], 0)
            self.assertIsNone(analyzer.image('enhanced'))

    def test_persistence_contains_only_bounded_statistics_and_invalid_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'baseline.json'
            for corrupt in ('[]', '{"version":1,"lighting":[]}', 'not json', '{"version":1,"lighting":{"dark":{"samples":-1}}}'):
                path.write_text(corrupt)
                analyzer = sa.SecondaryAnalyzer('unused', learning_path=path)
                self.assertEqual(analyzer._baselines, {})
            analyzer._baselines = {'dark': {'samples': 3, 'noise': 2., 'brightness': 30., 'blur': 40.}}
            analyzer._save_learning(force=True)
            data = json.loads(path.read_text())
            self.assertEqual(set(data), {'version', 'lighting'})
            loaded = sa.SecondaryAnalyzer('unused', learning_path=path)
            self.assertEqual(loaded._baselines, analyzer._baselines)
            self.assertLess(path.stat().st_size, 1000)

    def test_stop_during_model_load_failure_remains_off(self):
        entered, release = threading.Event(), threading.Event()

        def delayed_failure(*args, **kwargs):
            entered.set()
            release.wait(2)
            raise RuntimeError('Synthetic load failure')

        with patch.object(sa, 'ObjectDetector', side_effect=delayed_failure):
            analyzer = sa.SecondaryAnalyzer('unused')
            analyzer.start()
            self.assertTrue(entered.wait(2))
            self.assertFalse(analyzer.stop(timeout=.01))
            release.set()
            self.assertTrue(analyzer.stop(timeout=2))
            self.assertEqual(analyzer.status()['state'], 'off')
            self.assertFalse(analyzer.status()['enabled'])


if __name__ == '__main__':
    unittest.main()
