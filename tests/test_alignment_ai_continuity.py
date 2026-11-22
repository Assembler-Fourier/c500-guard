"""Accepted canonical corrections must not restart vehicle confirmation or visits.

Real SemanticMonitor queues/tracker, synthetic source pixels and detections only;
no model, camera, sound, or live files are used.
"""
from pathlib import Path
import threading
import unittest
from unittest import mock

from test_watch import watch, watcher
from semantic_monitor import SemanticMonitor


class AlignmentAIContinuityTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.obj = watcher()
        self.obj.args.warmup = 2.
        self.obj.zones = [watch.Zone(name='Gate', rect=(.1, .1, .8, .8), trigger='ai',
                                    route_group='entrance', classes=['car'], ai_confidence=.5,
                                    ai_confirm_observations=2)]
        with mock.patch.object(threading.Thread, 'start'):
            self.monitor = SemanticMonitor(self.obj.zones, Path('synthetic.onnx'), interval=.1)
        self.addCleanup(self.monitor.stop)
        self.obj.semantic = self.monitor
        self.obj.raise_alarm = mock.Mock(return_value=True)
        self.obj._observe_barrier_service = mock.Mock()
        self.obj._update_alarm_audio = mock.Mock()
        self.obj._feed_sightings = mock.Mock()
        self.obj._note_ai_activity = mock.Mock()
        self.obj._maybe_save_pose = mock.Mock()
        self.obj._alignment_blocked = False
        self.frame = watch.np.full((360, 640, 3), 40, watch.np.uint8)
        self.car = {'label': 'car', 'confidence': .9, 'bbox': [250, 140, 350, 230]}
        self.obj.aligner = mock.Mock()
        self.shift = 0.
        self.obj._alignment_used = ([[1., 0., 0.], [0., 1., 0.]], self.now)

    def process(self, *, car=False):
        if car:
            self.monitor._clear_queue(self.monitor._output)
            self.monitor._status.update(available=True, state='ready', updated_at=self.now,
                                        _epoch=self.monitor._epoch)
            self.monitor._output.put((self.monitor._epoch, self.monitor._worker_generation,
                                      self.now, [self.car]))
        self.assertTrue(self.obj.process(self.frame, received_at=self.now))

    def ready(self):
        self.process()
        self.now = 103.
        self.process()  # Explicit enable transition after cold-start warmup.

    def correction(self):
        self.shift = 3. if self.shift == 0. else 0.
        self.obj.aligner.update.return_value = (self.frame, {'reference_ready': True,
            'current_to_reference': [[1., 0., self.shift], [0., 1., 0.]],
            'translation_px': [self.shift, 0.]})
        self.assertIs(self.obj.align_frame(self.frame), self.frame)
        self.assertTrue(self.obj._pending_background_reset)

    def test_pair_and_consumed_identity_survive_repeated_accepted_corrections(self):
        self.ready()
        self.now += .1
        self.process(car=True)
        epoch = self.monitor._epoch
        self.obj.raise_alarm.assert_not_called()
        self.now += .2
        self.correction()
        self.process(car=True)
        self.obj.raise_alarm.assert_called_once()
        self.assertEqual(epoch, self.monitor._epoch)
        self.assertEqual(1, self.monitor._tracker.track_count)
        for _ in range(5):
            self.now += .2
            self.correction()
            self.process(car=True)
        self.obj.raise_alarm.assert_called_once()  # Same car cannot become six visits.
        self.assertEqual(epoch, self.monitor._epoch)
        self.assertTrue(self.monitor._enabled)
        self.assertLess(self.now-self.obj.warmup_started, self.obj.args.warmup)
        self.assertEqual('background', self.obj.log_event.call_args.kwargs['reset_scope'])

    def test_cold_start_correction_does_not_bypass_initial_ai_warmup(self):
        self.process()
        self.now = 100.5
        self.correction()
        self.process(car=True)
        self.assertFalse(self.monitor._enabled)
        self.assertEqual(0, self.monitor._tracker.track_count)
        self.now = 101.9
        self.process(car=True)
        self.assertFalse(self.monitor._enabled)
        self.obj.raise_alarm.assert_not_called()

    def test_true_blocked_recovery_invalidates_tracks_and_restarts_ai_warmup(self):
        self.ready()
        self.now += .1
        self.process(car=True)
        epoch = self.monitor._epoch
        self.now += .2
        self.obj._alignment_blocked = True
        self.correction()
        self.assertTrue(self.obj._pending_rearm)
        self.process(car=True)
        self.assertGreater(self.monitor._epoch, epoch)
        self.assertEqual(0, self.monitor._tracker.track_count)
        self.now += .2
        self.process(car=True)
        self.assertFalse(self.monitor._enabled)
        self.obj.raise_alarm.assert_not_called()

    def test_zone_change_overrides_simultaneous_background_only_request(self):
        self.ready()
        self.now += .1
        self.process(car=True)
        epoch = self.monitor._epoch
        self.now += .2
        self.correction()
        self.obj._pending_zones = list(self.obj.zones)
        self.process(car=True)
        self.assertGreater(self.monitor._epoch, epoch)
        self.assertEqual(0, self.monitor._tracker.track_count)
        self.now += .1
        self.process(car=True)
        self.assertFalse(self.monitor._enabled)
        self.obj.raise_alarm.assert_not_called()

    def test_user_rearm_overrides_background_only_request(self):
        self.ready()
        self.now += .1
        self.process(car=True)
        self.now += .2
        self.correction()
        self.obj._pending_rearm = True
        self.process(car=True)
        self.assertEqual(0, self.monitor._tracker.track_count)
        self.now += .1
        self.process(car=True)
        self.assertFalse(self.monitor._enabled)
        self.obj.raise_alarm.assert_not_called()

    def test_pixel_channels_stay_quiet_while_ai_confirms_during_background_warmup(self):
        motion = watch.Zone(name='Motion', rect=(.1, .1, .8, .8), motion=True, on_frames=1,
                            motion_pct=.1, classes=[])
        self.obj.zones.append(motion)
        self.ready()
        self.now += .1
        self.process(car=True)
        self.now += .2
        self.correction()
        self.process(car=True)
        self.now += .2
        self.frame[:] = 240
        self.process(car=True)
        self.assertFalse(motion.motion_active)
        self.obj.raise_alarm.assert_called_once()
        self.assertEqual('ai', self.obj.raise_alarm.call_args.kwargs['details']['source'])


if __name__ == '__main__':
    unittest.main()
