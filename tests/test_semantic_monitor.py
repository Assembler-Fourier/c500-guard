import sys
import unittest
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semantic_monitor import SemanticMonitor


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.monitor = SemanticMonitor.__new__(SemanticMonitor)
        self.monitor.confidence = .60
        self.monitor.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=True,
                                              severity='main',name='Gate')]
        self.monitor._counts = {}
        self.monitor._misses = {}
        self.monitor._active = set()
        self.monitor._epoch = 0
        self.monitor._result_lock = threading.RLock()
        self.monitor._last_confirmed_at = None
        self.car = {'label':'car','confidence':.9,'bbox':[25,25,55,55]}

    def test_two_observations_and_no_repeated_alarm_for_parked_car(self):
        self.assertEqual([], self.monitor._confirm([self.car],100,100))
        events = self.monitor._confirm([self.car],100,100)
        self.assertEqual(1, len(events))
        self.monitor.mark_emitted(events[0][2])
        self.assertEqual([], self.monitor._confirm([self.car],100,100))

    def test_requires_consecutive_observations(self):
        self.monitor._confirm([self.car],100,100)
        self.monitor._confirm([],100,100)
        self.assertEqual([], self.monitor._confirm([self.car],100,100))

    def test_outside_zone_is_ignored(self):
        outside={'label':'car','bbox':[75,75,95,95]}
        for _ in range(3):
            self.assertEqual([],self.monitor._confirm([outside],100,100))

    def test_departure_and_new_arrival_retrigger(self):
        self.monitor._confirm([self.car],100,100,100)
        events = self.monitor._confirm([self.car],100,100,100.5)
        self.monitor.mark_emitted(events[0][2])
        self.monitor._confirm([],100,100,106)
        self.assertEqual([], self.monitor._confirm([self.car],100,100,106.5))
        self.assertEqual(1, len(self.monitor._confirm([self.car],100,100,107)))

    def test_reset_invalidates_old_generation_and_confirmation(self):
        self.monitor._confirm([self.car],100,100)
        self.monitor.reset()
        self.assertEqual(1,self.monitor._epoch)
        self.assertEqual([],self.monitor._confirm([self.car],100,100))

    def test_status_rejects_old_frame_and_old_zone_generation(self):
        self.monitor._status = {'available':True, 'detections':[self.car],
            'updated_at':time.monotonic()-4, '_epoch':0}
        self.assertEqual([],self.monitor.status()['detections'])
        self.monitor._status['updated_at']=time.monotonic()
        self.monitor.reset()
        self.assertEqual('waiting_for_frames',self.monitor.status()['state'])
        self.assertEqual([],self.monitor.status()['detections'])

    def test_light_correlation_context_exists_only_for_a_pending_confirmed_visit(self):
        self.monitor.zones[0].route_group = 'west'
        self.monitor._confirm([self.car],100,100)
        events = self.monitor._confirm([self.car],100,100)
        event_id = events[0][2]
        context = self.monitor.event_context(event_id)
        self.assertEqual('Gate', context['zone_name'])
        self.assertEqual('west', context['route_group'])
        self.assertEqual(1, context['track_id'])
        self.assertEqual([.25,.25,.55,.55], context['bbox_normalized'])
        self.assertEqual(2, context['observations'])
        self.monitor.mark_emitted(event_id)
        self.assertIsNone(self.monitor.event_context(event_id))
        self.monitor.reset()
        self.assertIsNone(self.monitor.event_context(event_id))

    def test_status_preserves_exact_source_stamp_across_repeated_polls(self):
        stamp = 100.123456789
        self.monitor._status = {'available': True, 'state': 'ready',
            'detections': [], 'updated_at': stamp, '_epoch': 0}
        with patch('semantic_monitor.time.monotonic', return_value=100.24):
            first = self.monitor.status()
        with patch('semantic_monitor.time.monotonic', return_value=100.38):
            second = self.monitor.status()
        self.assertEqual(stamp, first['observed_at'])
        self.assertEqual(stamp, second['observed_at'])
        self.assertNotEqual(first['age_s'], second['age_s'])
        self.monitor._status['updated_at'] = None
        self.assertIsNone(self.monitor.status()['observed_at'])

    def test_early_candidates_cannot_leak_into_stricter_or_default_zones(self):
        self.monitor.zones = [SimpleNamespace(
            rect=(.2,.2,.5,.5), motion=False, ai_enabled=True, classes=['car'],
            severity='main', name=name, ai_confidence=threshold,
            ai_confirm_observations=2, arrival_direction=[1, 0],
            arrival_min_progress=.01, stationary_confirm_s=0)
            for name, threshold in [('Exit approach', .5), ('Top road', .55),
                                    ('Barrier', .65), ('Default', 0.)]]
        self.assertEqual(.5, self.monitor._candidate_confidence())
        for score, expected in ((.52, ['Exit approach']),
                                (.58, ['Exit approach', 'Top road']),
                                (.61, ['Exit approach', 'Top road', 'Default'])):
            with self.subTest(confidence=score):
                self.monitor.reset()
                first = {**self.car, 'confidence': score}
                second = {**first, 'bbox': [27,25,57,55]}
                self.assertEqual([], self.monitor._confirm([first],100,100,100.))
                events = self.monitor._confirm([second],100,100,100.5)
                names = [self.monitor.event_context(event[2])['zone_name'] for event in events]
                self.assertEqual(expected, names)
        self.assertEqual(0., self.monitor.zones[-1].ai_confidence,
                         'the default is applied without mutating saved zones')
        self.monitor.reset()
        for index in range(3):
            departing = {**self.car, 'confidence': .58,
                         'bbox': [29-index*2,25,59-index*2,55]}
            self.assertEqual([], self.monitor._confirm([departing],100,100,200.+index*.5),
                             'weaker candidates must still pass direction checks')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=True,
                                      severity='main', name='Gate')]
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.car = {'label': 'car', 'confidence': .9, 'bbox': [25, 25, 55, 55]}

    def make_monitor(self):
        # Unit tests run the worker explicitly or inject results; no ONNX/camera I/O.
        with patch.object(threading.Thread, 'start'):
            monitor = SemanticMonitor(self.zones, Path('test-model.onnx'), interval=.01)
        self.addCleanup(monitor.stop)
        return monitor

    @staticmethod
    def await_condition(predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        raise AssertionError('Worker did not reach the expected state')

    def deliver(self, monitor, *, age=0, generation=None):
        monitor._clear_queue(monitor._output)
        monitor._output.put((monitor._epoch,
                             monitor._worker_generation if generation is None else generation,
                             time.monotonic()-age, [self.car]))
        return monitor.update(self.frame, True)

    def test_failure_invalidates_old_results_and_requires_two_fresh_observations(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor))
        monitor._retry_failure(RuntimeError('temporary ONNX error'), 2)
        report = monitor.status()
        self.assertFalse(report['available'])
        self.assertEqual('retrying', report['state'])
        self.assertEqual([], report['detections'])
        self.assertGreater(report['retry_in_s'], 0)
        monitor.update(self.frame, True)  # Consume the failure's new generation.
        self.assertEqual([], self.deliver(monitor, generation=0))
        self.assertEqual([], self.deliver(monitor))
        self.assertEqual(1, len(self.deliver(monitor)))

    def test_candidate_floor_reaches_worker_and_updates_without_reloading_model(self):
        self.zones[0].ai_confidence = .5
        self.zones += [SimpleNamespace(rect=(.1,.1,.2,.2), ai_enabled=False,
                                      ai_confidence=.1, name='Motion only'),
                       SimpleNamespace(rect=(.1,.1,.2,.2), ai_enabled=True,
                                       classes=[], ai_confidence=.1, name='No classes')]
        monitor = self.make_monitor()
        seen = []
        class Detector:
            available = True
            def status(self):
                return {'available': True, 'confidence_threshold': self.confidence}
            def detect(self, frame):
                seen.append(self.confidence)
                return []
        def create(model, confidence, threads):
            detector = Detector()
            detector.confidence = confidence
            return detector
        with patch('object_detector.ObjectDetector', side_effect=create) as factory:
            monitor._thread = threading.Thread(target=monitor._run, daemon=True)
            monitor._thread.start()
            self.await_condition(lambda: monitor.status()['available'])
            monitor.update(self.frame, True)
            self.await_condition(lambda: len(seen) == 1)
            self.assertEqual([.5], seen)
            self.zones[0].ai_confidence = .55
            monitor._last_submit = 0.
            monitor.update(self.frame, True)
            self.await_condition(lambda: len(seen) == 2)
            self.assertEqual([.5, .55], seen)
            self.assertEqual(1, factory.call_count)
            monitor.stop()

    def test_candidate_confidence_validates_bounds_and_retains_default(self):
        for value in (float('nan'), float('inf'), -.1, 1.1, True, '.5'):
            with self.subTest(value=value), patch.object(threading.Thread, 'start'):
                with self.assertRaises(ValueError):
                    SemanticMonitor(self.zones, Path('test-model.onnx'), confidence=value)
                self.zones[0].ai_confidence = value
                with self.assertRaises(ValueError):
                    SemanticMonitor(self.zones, Path('test-model.onnx'))
                del self.zones[0].ai_confidence
        monitor = self.make_monitor()
        self.assertEqual(.6, monitor._candidate_confidence())
        self.zones[0].ai_confidence = 0.
        self.assertEqual(.6, monitor._candidate_confidence())

    def test_gap_cannot_complete_a_pre_gap_confirmation_pair(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor))
        monitor._last_confirmed_at = time.monotonic() - 4
        epoch = monitor._epoch
        # Accept the fresh result as observation ONE after the gap. The old
        # pre-gap sighting cannot count toward its confirmation pair.
        self.assertEqual([], self.deliver(monitor))
        self.assertEqual(epoch, monitor._epoch)
        self.assertEqual(1, len(self.deliver(monitor)))

    def test_fresh_queued_result_after_consumer_stall_recovers_without_new_epoch(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        monitor._last_submit = 101.2
        monitor._last_confirmed_at = 98.
        monitor._output.put((monitor._epoch, monitor._worker_generation, 100.8, [self.car]))
        epoch = monitor._epoch
        with patch('semantic_monitor.time.monotonic', return_value=101.2):
            self.assertEqual([], monitor.update(self.frame, True))
        self.assertEqual(epoch, monitor._epoch)
        self.assertEqual(100.8, monitor._last_confirmed_at)
        self.assertEqual(1, monitor._tracker.track_count)
        monitor._output.put((epoch, monitor._worker_generation, 101.3, [self.car]))
        with patch('semantic_monitor.time.monotonic', return_value=101.5):
            self.assertEqual(1, len(monitor.update(self.frame, True)))

    def test_stale_queued_result_after_gap_still_invalidates_old_work(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor))
        monitor._last_confirmed_at = time.monotonic()-4
        epoch = monitor._epoch
        self.assertEqual([], self.deliver(monitor, age=4))
        self.assertEqual(epoch+1, monitor._epoch)
        self.assertIsNone(monitor._last_confirmed_at)
        self.assertEqual(0, monitor._tracker.track_count)
        self.assertEqual([], self.deliver(monitor))
        self.assertEqual(1, len(self.deliver(monitor)))

    def test_off_and_explicit_reset_reject_fresh_old_epoch_results(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.deliver(monitor)
        old_epoch = monitor._epoch
        monitor._clear_queue(monitor._output)
        monitor._output.put((old_epoch, monitor._worker_generation, time.monotonic(), [self.car]))
        self.assertEqual([], monitor.update(self.frame, False))
        self.assertEqual(0, monitor._tracker.track_count)
        self.assertGreater(monitor._epoch, old_epoch)
        # Re-enabling also rejects work produced while disabled.
        monitor._output.put((monitor._epoch, monitor._worker_generation, time.monotonic(), [self.car]))
        self.assertEqual([], monitor.update(self.frame, True))
        self.assertEqual(0, monitor._tracker.track_count)
        self.assertEqual([], self.deliver(monitor))
        old_epoch = monitor._epoch
        monitor._output.put((old_epoch, monitor._worker_generation, time.monotonic(), [self.car]))
        monitor.reset()
        self.assertEqual([], monitor.update(self.frame, True))
        self.assertEqual(0, monitor._tracker.track_count)

    def test_future_result_never_counts_toward_confirmation(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor, age=-1))
        self.assertEqual(0, monitor._tracker.track_count)
        self.assertEqual([], self.deliver(monitor))
        self.assertEqual(1, len(self.deliver(monitor)))

    def test_backoff_doubles_to_cap_and_is_not_a_busy_retry_loop(self):
        monitor = self.make_monitor()
        delays = []
        def wait(delay):
            delays.append(delay)
            return len(delays) >= 8
        with patch('object_detector.ObjectDetector', side_effect=RuntimeError('missing runtime')) as create, \
             patch.object(monitor._stop, 'wait', side_effect=wait):
            monitor._run()
        self.assertEqual([2, 4, 8, 16, 32, 60, 60, 60], delays)
        self.assertEqual(8, create.call_count)
        self.assertEqual(8, monitor.status()['recovery_attempts'])

    def test_initialization_and_inference_failures_recover_without_restart(self):
        monitor = self.make_monitor()
        monitor.RETRY_INITIAL = .02
        bad = SimpleNamespace(available=False, status=lambda: {'available': False, 'error': 'load failed'})
        class Detector:
            available = True
            def status(self):
                return {'available': self.available, 'error': '' if self.available else 'inference failed'}
            def detect(self, frame):
                return []
        broken = Detector()
        def fail(frame):
            broken.available = False
            return []
        broken.detect = fail
        healthy = Detector()
        with patch('object_detector.ObjectDetector', side_effect=[bad, broken, healthy]) as create:
            monitor._thread = threading.Thread(target=monitor._run, daemon=True)
            monitor._thread.start()
            self.await_condition(lambda: create.call_count == 2 and monitor.status()['available'])
            monitor.update(self.frame, True)
            self.await_condition(lambda: monitor.status()['recovery_attempts'] == 2)
            self.await_condition(lambda: create.call_count == 3 and monitor.status()['available'])
            monitor.update(self.frame, True)
            self.await_condition(lambda: monitor.status()['inferences'] == 1)
            self.assertEqual('ready', monitor.status()['state'])
            self.assertEqual('', monitor.status()['error'])
            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())

    def test_uncaught_detector_exception_also_enters_retry(self):
        monitor = self.make_monitor()
        entered = threading.Event()
        class Detector:
            available = True
            def status(self):
                return {'available': True, 'error': ''}
            def detect(self, frame):
                entered.set()
                raise RuntimeError('unexpected backend exception')
        with patch('object_detector.ObjectDetector', return_value=Detector()):
            monitor._thread = threading.Thread(target=monitor._run, daemon=True)
            monitor._thread.start()
            self.await_condition(lambda: monitor.status()['available'])
            monitor.update(self.frame, True)
            self.assertTrue(entered.wait(1))
            self.await_condition(lambda: monitor.status()['state'] == 'retrying')
            self.assertIn('unexpected backend exception', monitor.status()['error'])
            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())

    def test_stop_interrupts_long_retry_wait(self):
        monitor = self.make_monitor()
        monitor.RETRY_INITIAL = 60
        with patch('object_detector.ObjectDetector', side_effect=RuntimeError('temporary failure')):
            monitor._thread = threading.Thread(target=monitor._run, daemon=True)
            monitor._thread.start()
            self.await_condition(lambda: monitor.status()['state'] == 'retrying')
            started = time.monotonic()
            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())
            self.assertLess(time.monotonic() - started, .5)

    def test_pending_visit_survives_quiet_until_explicit_delivery_confirmation(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor))
        pending = self.deliver(monitor)
        self.assertEqual(1, len(pending))
        event_id = pending[0][2]
        for _ in range(4):
            self.assertEqual(event_id, self.deliver(monitor)[0][2])
        self.assertTrue(monitor.mark_emitted(event_id))
        self.assertEqual([], self.deliver(monitor))

    def test_ai_only_region_still_crops_and_submits_with_raw_motion_disabled(self):
        monitor = self.make_monitor()
        monitor.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=False,
                         ai_enabled=True, classes=['person'], severity='main', name='Walkway')]
        monitor.update(self.frame, True)
        self.assertFalse(monitor._input.empty())
        _, _, _, views = monitor._input.get_nowait()
        self.assertEqual(1, len(views))
        crop, rect = views[0]
        self.assertEqual((64, 62, 3), crop.shape)
        self.assertEqual((14,10,76,74), rect)

    def test_motion_only_regions_do_not_submit_unnecessary_ai_frames(self):
        monitor = self.make_monitor()
        monitor.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=True,
                         ai_enabled=False, classes=['car'], severity='main', name='Motion')]
        monitor.update(self.frame, True)
        self.assertTrue(monitor._input.empty())

    def test_status_reports_tracked_regions_and_clears_them_on_reset(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        monitor._status.update(available=True, updated_at=time.monotonic(), _epoch=0)
        self.deliver(monitor)
        pending = self.deliver(monitor)
        report = monitor.status()
        self.assertEqual(1, report['tracked_objects'])
        self.assertEqual(1, report['active_visits'])
        self.assertEqual(['Gate'], report['detections'][0]['regions'])
        self.assertEqual([.4,.55], report['detections'][0]['ground_point'])
        monitor.reset()
        self.assertFalse(monitor.mark_emitted(pending[0][2]))
        self.assertEqual(0, monitor.status()['tracked_objects'])
        self.assertEqual([], monitor.status()['detections'])

    def test_status_hides_result_from_previous_worker_generation(self):
        monitor = self.make_monitor()
        monitor._worker_generation = 2
        monitor._status.update(available=True, updated_at=time.monotonic(),
                               _epoch=monitor._epoch, _generation=1, detections=[self.car])
        report = monitor.status()
        self.assertEqual('waiting_for_frames', report['state'])
        self.assertEqual([], report['detections'])
        self.assertNotIn('_generation', report)
if __name__=='__main__': unittest.main()
