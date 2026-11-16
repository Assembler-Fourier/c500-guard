"""No stale local pixels may become new image evidence after a stalled stage."""
import queue
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_watch import watch, watcher
from semantic_monitor import SemanticMonitor


class LocalFrameFreshnessTests(unittest.TestCase):
    def urgent_fixture(self):
        obj = watcher()
        obj._pending_rearm = False
        obj.args.warmup = 0.
        obj.warmup_started = 0.
        obj.semantic = None
        obj._alignment_blocked = False
        zone = watch.Zone(name='Exit', rect=(.1, .1, .3, .3), motion=True,
                          strict_change=True, on_frames=1, off_frames=8,
                          classes=[], route_group='exit')
        obj.zones = [zone]
        frame = np.full((watch.PROC_H, watch.PROC_W, 3), 40, np.float32)
        obj.bg = np.full(frame.shape[:2], 40., np.float32)
        obj.bg_noise = np.full_like(obj.bg, 5.)
        obj.visits = mock.Mock()
        obj.raise_alarm = mock.Mock(return_value=True)
        return obj, zone, frame

    def test_expired_weak_measurement_cannot_help_next_fresh_frame_confirm(self):
        obj, zone, frame = self.urgent_fixture()
        clock, calls = [100.], [0]
        measurement = {'hit': False, 'weak_hit': True, 'support_bounds': [20, 20, 8, 8],
                       'changed_pct': .8}

        def measure(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                clock[0] = 103.5
            return dict(measurement)

        with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(watch, 'measure_urgent_change', side_effect=measure):
            counts = dict(zone._counts)
            self.assertFalse(obj.process(frame, received_at=100.))
            self.assertIsNone(zone._change_previous)
            self.assertEqual(zone._counts, counts)
            obj.visits.observe.assert_not_called()
            clock[0] = 104.
            self.assertTrue(obj.process(frame, received_at=104.))
            obj.raise_alarm.assert_not_called()
            self.assertEqual(zone._change_previous[0], 104.)
            # Two genuinely fresh observations still warn promptly.
            clock[0] = 104.5
            self.assertTrue(obj.process(frame, received_at=104.5))
            obj.raise_alarm.assert_called_once()

    def test_expired_strong_measurement_does_not_activate_zone_or_learn(self):
        obj, zone, frame = self.urgent_fixture()
        clock = [100.]
        counts, background = dict(zone._counts), obj.bg.copy()

        def measure(*args, **kwargs):
            clock[0] = 104.
            return {'hit': True, 'weak_hit': True, 'support_bounds': [20, 20, 8, 8],
                    'changed_pct': 20.}

        with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(watch, 'measure_urgent_change', side_effect=measure):
            self.assertFalse(obj.process(frame, received_at=100.))
        self.assertIsNone(zone._change_previous)
        self.assertFalse(zone.motion_active)
        self.assertEqual(zone._counts, counts)
        np.testing.assert_array_equal(obj.bg, background)
        obj.visits.observe.assert_not_called()
        obj.raise_alarm.assert_not_called()

    def test_weak_confirmation_uses_acquisition_spacing_not_finish_spacing(self):
        obj, zone, frame = self.urgent_fixture()
        clock, calls = [100.], [0]

        def measure(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                clock[0] = 102.5  # Fresh, but almost two seconds of local work.
            return {'hit': False, 'weak_hit': True, 'support_bounds': [20, 20, 8, 8],
                    'changed_pct': .8}

        with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(watch, 'measure_urgent_change', side_effect=measure):
            self.assertTrue(obj.process(frame, received_at=100.))
            self.assertEqual(zone._change_previous[0], 100.)
            clock[0] = 103.
            self.assertTrue(obj.process(frame, received_at=103.))
            obj.raise_alarm.assert_not_called()
            self.assertEqual(zone._change_previous[0], 103.)

    def test_invalid_or_expired_pixels_cannot_change_alarm_visits_or_background(self):
        obj = watcher()
        frame = np.full((watch.PROC_H, watch.PROC_W, 3), 100, np.float32)
        obj.bg = np.full(frame.shape[:2], 40., np.float32)
        obj.bg_noise = np.full_like(obj.bg, 5.)
        original_bg, original_noise = obj.bg.copy(), obj.bg_noise.copy()
        obj.alarm, obj.alarm_id = 'main', 'current-other-visit'
        obj.buzzer.set('main')
        obj.visits = mock.Mock()
        obj.semantic = mock.Mock()
        obj.raise_alarm = mock.Mock()
        for stamp in (96., 97., 100.01, float('nan'), float('inf'), True, '100'):
            with self.subTest(stamp=stamp), mock.patch.object(watch.time, 'monotonic', return_value=100.):
                self.assertFalse(obj.process(frame, received_at=stamp))
        np.testing.assert_array_equal(obj.bg, original_bg)
        np.testing.assert_array_equal(obj.bg_noise, original_noise)
        self.assertEqual(obj.alarm, 'main')
        self.assertEqual(obj.alarm_id, 'current-other-visit')
        self.assertEqual(obj.buzzer.want, 'main')
        obj.visits.observe.assert_not_called()
        obj.semantic.update.assert_not_called()
        obj.semantic.reset.assert_not_called()
        obj.raise_alarm.assert_not_called()

    def test_frame_expiring_during_preprocessing_cannot_learn_or_emit(self):
        obj = watcher()
        frame = np.zeros((watch.PROC_H, watch.PROC_W, 3), np.float32)
        obj._pending_rearm = False
        obj.bg = np.full(frame.shape[:2], 40., np.float32)
        obj.bg_noise = np.full_like(obj.bg, 5.)
        original = obj.bg.copy()
        obj.visits = mock.Mock()
        obj.semantic = mock.Mock()
        obj.raise_alarm = mock.Mock()
        clock = [100.]
        obj.lighting_monitor = mock.Mock()
        obj.lighting_monitor.update.side_effect = lambda *a, **k: clock.__setitem__(0, 116.)
        with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: clock[0]):
            self.assertFalse(obj.process(frame, received_at=100.))
        np.testing.assert_array_equal(obj.bg, original)
        obj.visits.observe.assert_not_called()
        obj.semantic.update.assert_not_called()
        obj.raise_alarm.assert_not_called()

    def test_current_frame_keeps_decoder_stamp_for_primary_submission(self):
        obj = watcher()
        obj._pending_rearm = False
        obj.zones = []
        obj.args.warmup = 0.
        obj.warmup_started = 0.
        frame = np.full((watch.PROC_H, watch.PROC_W, 3), 40, np.float32)
        obj.bg = np.full(frame.shape[:2], 40., np.float32)
        obj.bg_noise = np.full_like(obj.bg, 5.)
        obj.semantic = mock.Mock()
        obj.semantic.status.return_value = {}
        obj.semantic.update.return_value = []
        obj.visits = mock.Mock()
        with mock.patch.object(watch.time, 'monotonic', return_value=100.5):
            self.assertTrue(obj.process(frame, received_at=100.))
        self.assertEqual(obj.semantic.update.call_args.kwargs['stamp'], 100.)

    def test_ai_alarm_uses_result_observation_time_not_newer_caller_frame(self):
        obj = watcher()
        obj._pending_rearm = False
        obj.zones = []
        obj.args.warmup = 0.
        obj.warmup_started = 0.
        frame = np.full((watch.PROC_H, watch.PROC_W, 3), 40, np.float32)
        obj.bg = np.full(frame.shape[:2], 40., np.float32)
        obj.bg_noise = np.full_like(obj.bg, 5.)
        obj.semantic = mock.Mock()
        obj.semantic.status.return_value = {}
        obj.semantic.update.return_value = [('main', 'car arrival', 'event')]
        obj.semantic.event_context.return_value = {'observed_at': 90.}
        obj.visits = mock.Mock()
        obj.raise_alarm = mock.Mock(return_value=True)
        with mock.patch.object(watch.time, 'monotonic', return_value=100.):
            obj.process(frame, received_at=99.5)
        obj.raise_alarm.assert_not_called()
        obj.semantic.mark_emitted.assert_not_called()
        obj.semantic.event_context.return_value = {'observed_at': 99.2}
        with mock.patch.object(watch.time, 'monotonic', return_value=100.):
            obj.process(frame, received_at=99.5)
        self.assertEqual(obj.raise_alarm.call_args.kwargs['details']['frame_received_at'], 99.2)
        obj.semantic.mark_emitted.assert_called_once_with('event')

    def run_one_frame(self, delay, stamp=100.):
        obj = watcher()
        obj.args.web = 0
        obj.args.black_level = 5
        obj.args.lost_after = 50
        obj.reader.start = mock.Mock()
        obj.presence.start = mock.Mock()
        obj.secondary.start = mock.Mock()
        obj.semantic = mock.Mock()
        obj._update_awake = mock.Mock()
        obj._submit_analysis = mock.Mock()
        obj.check_health = mock.Mock()
        obj.paint_status = mock.Mock()
        obj.render_preview = mock.Mock()
        obj.process = mock.Mock(return_value=True)
        obj.pipeline = mock.MagicMock()
        clock = [100.]
        frame = np.full((watch.PROC_H, watch.PROC_W, 3), 60, np.uint8)

        def get_one(**kwargs):
            obj._running = False
            return stamp, frame.tobytes()

        def align(value):
            clock[0] += delay
            return value

        obj.reader.frames = mock.Mock()
        obj.reader.frames.get.side_effect = get_one
        obj.align_frame = mock.Mock(side_effect=align)
        with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(threading.Thread, 'start'), mock.patch('builtins.print'):
            obj.run()
        return obj

    def test_slow_alignment_drops_frame_without_resetting_current_ai(self):
        obj = self.run_one_frame(16.)
        obj.process.assert_not_called()
        obj.semantic.reset.assert_not_called()
        obj.check_health.assert_called_with(116.)
        obj.pipeline.record_stale_drop.assert_called_once()
        obj.pipeline.frame_processed.assert_not_called()

    def test_fresh_alignment_forwards_acquisition_time_and_records_success(self):
        obj = self.run_one_frame(.2)
        self.assertEqual(obj.process.call_args.kwargs['received_at'], 100.)
        obj.pipeline.frame_received.assert_called_once_with(100.)
        obj.pipeline.frame_processed.assert_called_once_with(100.)
        obj.check_health.assert_called_with(100.2)

    def test_old_queue_item_is_rejected_before_alignment(self):
        obj = self.run_one_frame(0., stamp=90.)
        obj.align_frame.assert_not_called()
        obj.process.assert_not_called()
        obj.semantic.reset.assert_not_called()


class SemanticAcquisitionTimeTests(unittest.TestCase):
    def make_monitor(self):
        from types import SimpleNamespace
        zone = SimpleNamespace(rect=(0., 0., 1., 1.), motion=True, name='Gate', severity='main')
        with mock.patch.object(threading.Thread, 'start'):
            monitor = SemanticMonitor([zone], Path('test-model.onnx'), interval=.1)
        monitor._enabled = True
        self.addCleanup(monitor.stop)
        return monitor

    def test_queue_retains_original_receipt_stamp(self):
        monitor = self.make_monitor()
        with mock.patch('semantic_monitor.time.monotonic', return_value=100.5):
            monitor.update(np.zeros((20, 20, 3), np.uint8), True, stamp=100.)
        self.assertEqual(monitor._input.get_nowait()[2], 100.)

    def test_stale_input_does_not_consume_fresh_output_or_clear_tracking(self):
        monitor = self.make_monitor()
        monitor._last_confirmed_at = 98.
        monitor._tracker = mock.Mock()
        monitor._output.put((monitor._epoch, monitor._worker_generation, 100.5, []))
        epoch = monitor._epoch
        with mock.patch('semantic_monitor.time.monotonic', return_value=101.):
            monitor.update(np.zeros((20, 20, 3), np.uint8), True, stamp=90.)
        self.assertFalse(monitor._output.empty())
        self.assertTrue(monitor._input.empty())
        self.assertEqual(monitor._epoch, epoch)
        self.assertEqual(monitor._last_confirmed_at, 98.)
        monitor._tracker.reset.assert_not_called()

    def test_invalid_and_future_stamps_never_queue_pixels(self):
        monitor = self.make_monitor()
        for stamp in (97., 100.1, float('nan'), float('inf'), '100', True):
            with self.subTest(stamp=stamp), mock.patch('semantic_monitor.time.monotonic', return_value=100.):
                monitor.update(np.zeros((20, 20, 3), np.uint8), True, stamp=stamp)
        self.assertTrue(monitor._input.empty())

    def test_fresh_caller_cannot_make_old_result_fresh(self):
        monitor = self.make_monitor()
        monitor._tracker = mock.Mock()
        monitor._output.put((monitor._epoch, monitor._worker_generation, 90., []))
        with mock.patch('semantic_monitor.time.monotonic', return_value=100.):
            monitor.update(np.zeros((20, 20, 3), np.uint8), True, stamp=99.5)
        monitor._tracker.update.assert_not_called()
        self.assertEqual(monitor._input.get_nowait()[2], 99.5)

    def test_disabled_transition_still_invalidates_even_with_expired_input(self):
        monitor = self.make_monitor()
        epoch = monitor._epoch
        with mock.patch('semantic_monitor.time.monotonic', return_value=100.):
            monitor.update(np.zeros((20, 20, 3), np.uint8), False, stamp=80.)
        self.assertFalse(monitor._enabled)
        self.assertEqual(monitor._epoch, epoch+1)


if __name__ == '__main__':
    unittest.main()
