"""Traffic-region integration without live camera, model inference or audio."""
import sys
import threading
from pathlib import Path
from unittest import mock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semantic_monitor import SemanticMonitor
from test_watch import watch, watcher


class RegionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.notifications = mock.patch.object(watch, 'notify')
        self.notifications.start()
        self.addCleanup(self.notifications.stop)
        self.obj = watcher()
        self.obj.args.warmup = 0.
        self.obj.args.adapt = 0.
        self.obj._alignment_blocked = False
        self.obj.zones = [watch.Zone('Vehicle lane', (0., 0., 1., 1.),
                          trigger='ai', classes=['car', 'bus', 'truck'],
                          motion=True, light=True)]
        self.frame = watch.np.full((100, 100, 3), 80, dtype=watch.np.uint8)
        self.obj.process(self.frame)
        with mock.patch.object(threading.Thread, 'start'):
            self.semantic = SemanticMonitor(self.obj.zones, Path('unused-test-model.onnx'), interval=1000.)
        self.semantic._enabled = True
        self.semantic._last_submit = self.now
        self.obj.semantic = self.semantic
        self.addCleanup(self.semantic.stop)
        self.car = {'label': 'car', 'confidence': .92, 'bbox': [20,20,32,40]}

    def deliver(self, objects, *, after=.5, frame=None):
        self.now += after
        self.obj.last_frame_at = self.now
        self.semantic._status.update(available=True, state='ready', updated_at=self.now,
                                     _epoch=self.semantic._epoch,
                                     _generation=self.semantic._worker_generation,
                                     detections=list(objects))
        self.semantic._clear_queue(self.semantic._output)
        self.semantic._output.put((self.semantic._epoch, self.semantic._worker_generation,
                                   self.now, objects))
        self.obj.process(self.frame if frame is None else frame)

    def confirm_car(self):
        self.deliver([self.car])
        self.assertIsNone(self.obj.alarm)
        self.deliver([self.car])
        self.assertEqual('main', self.obj.alarm)

    def fresh_feed(self):
        self.obj.first_frame_seen = True
        self.obj.last_frame_at = self.now

    def test_ai_only_regions_ignore_moving_barrier_and_headlight_pixels(self):
        changed = self.frame.copy()
        changed[25:55, 10:70] = 230
        for _ in range(5):
            self.deliver([], frame=changed)
        current = self.obj.zones[0]
        self.assertGreater(current.last_pct, current.motion_pct)
        self.assertGreater(current.last_delta, current.light_delta)
        self.assertFalse(current.motion_active)
        self.assertFalse(current.light_active)
        self.assertIsNone(self.obj.alarm)

    def test_acknowledgement_expiry_preserves_current_ai_visit(self):
        self.confirm_car()
        epoch = self.semantic._epoch
        alarm_count = self.obj._alarm_sequence
        self.assertTrue(self.obj.acknowledge())
        with mock.patch.object(self.semantic, 'reset', wraps=self.semantic.reset) as reset:
            for _ in range(24):
                self.deliver([self.car])
            reset.assert_not_called()
        self.assertEqual(epoch, self.semantic._epoch)
        self.assertEqual(alarm_count, self.obj._alarm_sequence)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertEqual(0., self.obj._ack_until)

    def test_pending_visit_is_consumed_only_when_raise_alarm_accepts_it(self):
        self.deliver([self.car])
        with mock.patch.object(self.obj, 'raise_alarm', return_value=False) as raise_alarm, \
             mock.patch.object(self.semantic, 'mark_emitted', wraps=self.semantic.mark_emitted) as mark:
            self.deliver([self.car])
            self.assertTrue(raise_alarm.called)
            mark.assert_not_called()
        with mock.patch.object(self.semantic, 'mark_emitted', wraps=self.semantic.mark_emitted) as mark:
            self.deliver([self.car])
            mark.assert_called_once()
        self.assertEqual('main', self.obj.alarm)

    def test_second_car_waits_for_current_alarm_and_ten_second_quiet(self):
        self.confirm_car()
        first_id = self.obj.alarm_id
        second = {'label':'car', 'confidence':.91, 'bbox':[65,20,77,40]}
        for _ in range(3):
            self.deliver([self.car, second])
        self.assertEqual(first_id, self.obj.alarm_id)
        self.assertTrue(self.obj.acknowledge())
        for _ in range(19):
            self.deliver([self.car, second])
            self.assertIsNone(self.obj.alarm)
        self.deliver([self.car, second])
        self.assertEqual('main', self.obj.alarm)
        self.assertNotEqual(first_id, self.obj.alarm_id)
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_car_departing_during_quiet_is_not_replayed(self):
        self.confirm_car()
        second = {'label':'car', 'confidence':.91, 'bbox':[65,20,77,40]}
        self.deliver([self.car, second])
        self.deliver([self.car, second])
        self.assertTrue(self.obj.acknowledge())
        for _ in range(24):
            self.deliver([self.car])
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_person_only_region_ignores_car_and_accepts_pedestrian(self):
        self.obj.zones[0].classes = ['person']
        for _ in range(4):
            self.deliver([self.car])
        self.assertIsNone(self.obj.alarm)
        person = {'label':'person', 'confidence':.92, 'bbox':[25,15,35,50]}
        self.deliver([person])
        self.deliver([person])
        self.assertEqual('main', self.obj.alarm)
        self.assertIn('person arrival', self.obj.alarm_reason)

    def test_parking_exclusion_rejects_overlapping_car_ground_point(self):
        self.obj.zones[0].exclude_polygons = [[[.5,.3],[.8,.3],[.8,.8],[.5,.8]]]
        parked = {'label':'car', 'confidence':.92, 'bbox':[40,15,70,55]}
        for _ in range(4):
            self.deliver([parked])
        self.assertIsNone(self.obj.alarm)
        self.confirm_car()

    def test_polygon_motion_mask_ignores_changes_outside_road_and_inside_parking(self):
        current = watch.Zone('Road', (0.,0.,1.,1.), trigger='motion', motion=True,
                             polygon=[[.1,.1],[.9,.1],[.1,.9]],
                             exclude_polygons=[[[.2,.2],[.35,.2],[.35,.35],[.2,.35]]])
        self.obj.zones = [current]
        self.semantic.zones = self.obj.zones
        outside = self.frame.copy()
        mask = current.mask_for(100,100)
        outside[~mask] = 230
        for _ in range(3):
            self.deliver([], frame=outside)
        self.assertEqual(0., current.last_pct)
        self.assertIsNone(self.obj.alarm)
        inside = self.frame.copy()
        inside[mask] = 230
        self.deliver([], frame=inside)
        self.deliver([], frame=inside)
        self.assertEqual('main', self.obj.alarm)

    def test_fractional_region_bounds_retain_every_included_edge_pixel(self):
        # Pixel centre x=30.5 is inside the 10.3..30.8 region. Rounding the
        # start and width separately would accidentally drop that entire edge.
        current = watch.Zone('Narrow region', (.103,.103,.205,.205),
                             trigger='motion', motion_pct=.1)
        self.obj.zones = [current]
        self.semantic.zones = self.obj.zones
        self.assertTrue(current.mask_for(100,100)[20,30])
        changed = self.frame.copy()
        changed[20,30] = 230
        self.deliver([], frame=changed)
        self.deliver([], frame=changed)
        self.assertGreater(current.last_pct, 0.)
        self.assertEqual('main', self.obj.alarm)

    def test_accepted_raw_motion_episode_does_not_replay_at_ack_expiry(self):
        self.obj.zones[0].trigger = 'motion'
        self.obj.zones[0].light = False
        changed = self.frame.copy()
        changed[25:55, 10:70] = 230
        self.deliver([], frame=changed)
        self.deliver([], frame=changed)
        self.assertEqual('main', self.obj.alarm)
        self.obj.acknowledge()
        for _ in range(24):
            self.deliver([], frame=changed)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_ai_only_unavailable_model_raises_fault_after_startup_grace(self):
        self.now = 130.
        self.fresh_feed()
        self.semantic._status.update(available=False, state='retrying', updated_at=None)
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj.alarm)  # Allow a brief recovery/rearm delay.
        self.now += 6.
        self.fresh_feed()
        self.obj.check_health(self.now)
        self.assertEqual('fault', self.obj.alarm)
        self.assertEqual('ai', self.obj._fault_kind)
        self.assertIn('Object detection unavailable', self.obj.fault_reason)

    def test_ai_loaded_but_never_completing_inference_cannot_remain_silent(self):
        self.now = 130.
        self.fresh_feed()
        self.semantic._status.update(available=True, state='ready', updated_at=None)
        self.obj.check_health(self.now)
        self.now += 6.
        self.fresh_feed()
        self.obj.check_health(self.now)
        self.assertEqual('fault', self.obj.alarm)
        self.assertEqual('ai', self.obj._fault_kind)

    def test_stale_ai_result_does_not_report_healthy_ai_only_coverage(self):
        self.now = 130.
        self.fresh_feed()
        self.semantic._status.update(available=True, state='ready', updated_at=120.,
                                     _epoch=self.semantic._epoch)
        self.obj.check_health(self.now)
        self.now += 6.
        self.fresh_feed()
        self.obj.check_health(self.now)
        self.assertEqual('fault', self.obj.alarm)
        self.assertEqual('ai', self.obj._fault_kind)

    def test_motion_only_coverage_does_not_require_ai_availability(self):
        self.obj.zones[0].trigger = 'motion'
        self.obj.semantic = None
        self.now = 130.
        self.fresh_feed()
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual('', self.obj.fault_reason)

    def test_ai_recovery_clears_continuous_unhealthy_deadline(self):
        self.now = 130.
        self.fresh_feed()
        self.semantic._status.update(available=False, state='retrying', updated_at=None)
        self.obj.check_health(self.now)
        self.now += 4.
        self.fresh_feed()
        self.semantic._status.update(available=True, state='ready', updated_at=self.now,
                                     _epoch=self.semantic._epoch)
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj._ai_bad_since)
        self.now += 4.
        self.fresh_feed()
        self.semantic._status.update(available=False, state='retrying', updated_at=None)
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj.alarm)

    def test_old_generation_cannot_appear_ready_just_because_its_timestamp_is_fresh(self):
        self.now = 130.
        for index in range(2):
            self.fresh_feed()
            self.semantic._status.update(available=True, state='ready', updated_at=self.now,
                                         _epoch=self.semantic._epoch-1)
            self.obj.check_health(self.now)
            if not index:
                self.assertIsNone(self.obj.alarm)
                self.now += 6.
        self.assertEqual('fault', self.obj.alarm)
        self.assertEqual('ai', self.obj._fault_kind)

    def test_disarmed_health_clears_old_ai_failure_deadline(self):
        self.now = 130.
        self.fresh_feed()
        self.semantic._status.update(available=False, state='retrying', updated_at=None)
        self.obj.check_health(self.now)
        self.assertIsNotNone(self.obj._ai_bad_since)
        self.obj.armed = False
        self.now += 6.
        self.fresh_feed()
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj._ai_bad_since)
        self.assertIsNone(self.obj.alarm)


if __name__ == '__main__':
    unittest.main()
