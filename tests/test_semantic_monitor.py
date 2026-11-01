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
        self.monitor.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=True,
                                              severity='main',name='Gate')]
        self.monitor._counts = {}
        self.monitor._misses = {}
        self.monitor._active = set()
        self.monitor._epoch = 0
        self.monitor._result_lock = threading.RLock()
        self.monitor._last_confirmed_at = None
        self.car = {'label':'car','bbox':[25,25,55,55]}

    def test_two_observations_and_no_repeated_alarm_for_parked_car(self):
        self.assertEqual([], self.monitor._confirm([self.car],100,100))
        self.assertEqual(1, len(self.monitor._confirm([self.car],100,100)))
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
        for _ in range(2): self.monitor._confirm([self.car],100,100)
        for _ in range(3): self.monitor._confirm([],100,100)
        self.assertEqual([], self.monitor._confirm([self.car],100,100))
        self.assertEqual(1, len(self.monitor._confirm([self.car],100,100)))

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


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.zones = [SimpleNamespace(rect=(.2,.2,.5,.5), motion=True,
                                      severity='main', name='Gate')]
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.car = {'label': 'car', 'bbox': [25, 25, 55, 55]}

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

    def test_gap_cannot_complete_a_pre_gap_confirmation_pair(self):
        monitor = self.make_monitor()
        monitor._enabled = True
        self.assertEqual([], self.deliver(monitor))
        monitor._last_confirmed_at = time.monotonic() - 4
        # The first result queued under the old epoch is discarded after the gap.
        self.assertEqual([], self.deliver(monitor))
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
if __name__=='__main__': unittest.main()
