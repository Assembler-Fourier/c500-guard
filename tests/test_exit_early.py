"""Earlier exit alarms on PTZ 2 (16 Nov 2026 audit): tail-light corner, AI burst, bottom-edge entry,
moving-only weak detections and the "already opening" chirp. Synthetic frames and boxes; no camera or model."""
import time
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher
from exit_entry import measure_taillights
from object_tracking import ObjectVisitTracker
from semantic_monitor import SemanticMonitor
from zone_editor import validate_config

CORNER = dict(name='Exit corner - tail lights', rect=(.12, .72, .38, .2),
              polygon=[[.12, .72], [.5, .72], [.5, .92], [.12, .92]], trigger='taillight', classes=[],
              motion=False, light=True, severity='main', light_severity='main', on_frames=2, off_frames=15,
              priority=True, route_group='campus-exit', route_role='exit', alert_stage='advance')
EXIT_APPROACH = dict(name='Exit approach', rect=(.12, .555, .478, .365),
                     polygon=[[.12, .7], [.3, .69], [.37, .645], [.41, .6], [.44, .555], [.585, .555], [.598, .62],
                              [.59, .665], [.555, .745], [.51, .84], [.47, .92], [.19, .92], [.2, .85], [.12, .82]],
                     trigger='ai', classes=['car', 'motorcycle', 'bus', 'truck'], motion=False, light=False,
                     ai_confidence=.5, arrival_direction=[.35, -1.], arrival_min_progress=.012,
                     stationary_confirm_s=3, ai_confirm_observations=2, route_group='campus-exit',
                     route_role='exit', alert_stage='advance')


def blob(frame, colour, x0=.2, y0=.75, x1=.4, y1=.88):
    out = frame.copy()
    out[int(y0 * 360):int(y1 * 360), int(x0 * 640):int(x1 * 640)] = colour
    return out


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((360, 640, 3), 80., np.float32)
        self.bg = self.base.mean(axis=2)
        self.mask = watch.Zone(**CORNER).mask_for(640, 360)

    def test_red_tail_lights_entering_hit(self):
        m = measure_taillights(blob(self.base, (255, 90, 80)), self.bg, self.mask)
        self.assertTrue(m['hit'] and m['wake'] and m['lane_seen'])
        self.assertGreater(m['red_pct'], 2)
        self.assertIsNotNone(m['support_bounds'])

    def test_white_headlights_only_wake_the_ai(self):
        m = measure_taillights(blob(self.base, (245, 245, 240)), self.bg, self.mask)
        self.assertFalse(m['hit'])
        self.assertTrue(m['wake'])
        self.assertEqual(m['red_pct'], 0)

    def test_empty_road_and_a_small_red_light_stay_quiet(self):
        self.assertFalse(measure_taillights(self.base, self.bg, self.mask)['wake'])
        small = measure_taillights(blob(self.base, (255, 90, 80), .2, .8, .22, .82), self.bg, self.mask)
        self.assertFalse(small['hit'])

    def test_bad_input_is_harmless(self):
        self.assertFalse(measure_taillights(self.base, self.bg[:10], self.mask)['hit'])
        self.assertFalse(measure_taillights(self.base[..., 0], self.bg, self.mask)['hit'])


class CornerAlarmTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher()
        self.obj.loudness = mock.Mock()
        self.obj.args.warmup = 0.0
        self.obj.args.adapt = 0.0
        self.obj.alignment_config = {}
        self.obj._alignment_blocked = False
        self.corner = watch.Zone(**CORNER)
        self.obj.zones = [self.corner]
        self.obj.semantic = mock.Mock()
        self.obj.semantic.update.return_value = []
        self.obj.semantic.status.return_value = {"available": True, "state": "ready", "age_s": 0., "detections": []}
        self.base = np.full((360, 640, 3), 80., np.float32)
        self.obj.process(self.base)

    def deliver(self, frame, times=1):
        for _ in range(times):
            self.now += .1
            self.obj.process(frame)

    def test_tail_lights_raise_the_exit_advance_alarm_within_two_frames(self):
        self.deliver(blob(self.base, (255, 90, 80)), 2)
        self.assertEqual(self.obj.alarm, 'main')
        details = self.obj.alarm_details
        self.assertEqual((details['source'], details['trigger'], details['route_group'], details['alert_stage']),
                         ('light', 'taillight', 'campus-exit', 'advance'))
        self.assertIn('tail lights entering', self.obj.alarm_reason)
        self.obj.semantic.boost.assert_called()

    def test_headlights_of_an_entering_car_wake_the_ai_but_never_alarm(self):
        self.deliver(blob(self.base, (245, 245, 240)), 5)
        self.assertIsNone(self.obj.alarm)
        self.obj.semantic.boost.assert_called()

    def test_one_red_frame_is_not_enough(self):
        self.deliver(blob(self.base, (255, 90, 80)))
        self.deliver(self.base)
        self.assertIsNone(self.obj.alarm)

    def test_lane_evidence_is_remembered_for_the_already_opening_check(self):
        self.deliver(blob(self.base, (255, 90, 80)))
        self.assertEqual(self.obj._lane_seen['exit'][0], self.now)


class AlreadyOpeningTests(unittest.TestCase):
    def setUp(self):
        self.now = 500.0
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher()
        self.obj.loudness = mock.Mock()
        self.obj.semantic = None
        self.obj._alignment_blocked = False
        self.obj.alignment = {'state': 'tracking'}
        self.obj.zones = [watch.Zone(**EXIT_APPROACH)]
        self.obj._pending_rearm = False
        self.obj.barrier_workflow = None

    def alarm(self, lane='exit', **extra):
        group = 'campus-exit' if lane == 'exit' else 'campus-entrance'
        details = {'source': 'ai', 'route_group': group, 'route_role': lane, 'zone_name': 'Exit approach', **extra}
        self.assertTrue(self.obj.raise_alarm('main', 'Exit approach: AI confirmed car arrival', details=details))

    def events(self, kind):
        return [call for call in self.obj.log_event.call_args_list if call.args and call.args[0] == kind]

    def test_barrier_already_rising_for_this_vehicle_chirps_then_stops(self):
        self.obj._lane_seen = {'exit': [self.now - 3., self.now - .3]}       # tail lights seen 3 s ago
        self.obj._recent_rise = {'exit': self.now - 1.}                      # guard began raising 1 s ago
        self.alarm()
        self.assertEqual(self.obj.alarm, 'main')                             # it is still raised and logged
        self.assertEqual(self.obj.alarm_details['already_opening']['arm'], 'exit')
        self.now += .5
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.alarm, 'main')                             # still chirping
        self.now += self.obj.SERVED_CHIRP_S
        # A scheduled chirp is not sufficient proof after the delay: the
        # current frame must still show this lane's same rising movement.
        self.obj.last_frame_at = self.now
        self.obj._arm_live = {'exit': {'ok': True, 'motion': 'rising',
            'motion_started': 499., 'at': self.now, 'at_wall': time.time(),
            'shown': 45., 'position_conflict': False}}
        self.obj._update_alarm_audio()
        self.assertIsNone(self.obj.alarm)
        done = self.events('auto-barrier-up')
        self.assertEqual(len(done), 1)
        self.assertIs(done[0].kwargs['already_opening'], True)

    def test_ai_track_seen_before_the_rise_also_counts(self):
        self.obj._recent_rise = {'exit': self.now - 1.}
        self.alarm(first_observed_at=self.now - 2.5)
        self.assertIn('already_opening', self.obj.alarm_details)

    def test_rise_before_the_vehicle_was_seen_keeps_ringing(self):
        # The arm was going up for the car before; this car only appeared afterwards.
        self.obj._lane_seen = {'exit': [self.now - .5, self.now - .2]}
        self.obj._recent_rise = {'exit': self.now - 1.}
        self.alarm()
        self.assertNotIn('already_opening', self.obj.alarm_details)
        self.now += 3.
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.alarm, 'main')

    def test_old_rise_other_lane_or_switched_off_rule_keep_ringing(self):
        self.obj._lane_seen = {'exit': [self.now - 8., self.now - .3]}
        self.obj._recent_rise = {'exit': self.now - 5.}                      # older than PRE_RISE_S
        self.alarm()
        self.assertNotIn('already_opening', self.obj.alarm_details)
        self.obj.acknowledge(via='test')
        self.now += 20.
        self.obj._recent_rise = {'exit': self.now - 1.}
        self.obj._lane_seen = {'entrance': [self.now - 3., self.now - .3]}
        self.alarm(lane='entrance')                                          # exit arm rising, entrance alarm
        self.assertNotIn('already_opening', self.obj.alarm_details)
        self.obj.acknowledge(via='test')
        self.now += 20.
        self.obj.barrier_rules = {**self.obj.barrier_rules, 'ack_already_opening': False}
        self.obj._recent_rise = {'exit': self.now - 1.}
        self.obj._lane_seen = {'exit': [self.now - 3., self.now - .3]}
        self.alarm()
        self.assertNotIn('already_opening', self.obj.alarm_details)

    def test_rise_recorded_only_from_a_trusted_rising_reading(self):
        self.obj._note_arm_rise('exit', {'motion': 'rising', 'ok': True, 'motion_started': 12.5})
        self.obj._note_arm_rise('entrance', {'motion': 'rising', 'ok': False, 'motion_started': 13.})
        self.obj._note_arm_rise('entrance', {'motion': 'lowering', 'ok': True, 'motion_started': 13.})
        self.assertEqual(self.obj._recent_rise, {'exit': 12.5})


class TrackerTests(unittest.TestCase):
    def zone(self, **extra):
        return watch.Zone(**{**EXIT_APPROACH, **extra})

    def run_track(self, zone, boxes, confidences, step=.25):
        tracker, events, now = ObjectVisitTracker(), [], 10.
        for box, conf in zip(boxes, confidences):
            now += step
            found, _ = tracker.update([{'label': 'car', 'bbox': box, 'confidence': conf}], [zone], 640, 360, now)
            events += found
        return events

    def clipped_entry(self, n=6):
        # Cut off by the bottom edge, driving up-right: x moves right, the top of the box rises.
        return [[120 + 6 * i, 250 - 8 * i, 240 + 6 * i, 360] for i in range(n)]

    def test_vehicle_cut_off_by_the_bottom_edge_is_followed_in(self):
        self.assertTrue(self.run_track(self.zone(bottom_edge_entry=True), self.clipped_entry(), [.7] * 6))

    def test_without_the_option_nothing_changes(self):
        self.assertFalse(self.run_track(self.zone(), self.clipped_entry(), [.7] * 6))

    def moving_in(self, n=6):
        return [[180 + 5 * i, 230 - 9 * i, 300 + 5 * i, 310 - 9 * i] for i in range(n)]

    def test_weak_detections_count_for_a_vehicle_moving_in(self):
        events = self.run_track(self.zone(ai_confidence_moving=.3), self.moving_in(), [.38] * 6)
        self.assertTrue(events)

    def test_weak_detections_need_one_extra_observation(self):
        zone = self.zone(ai_confidence_moving=.3)
        self.assertFalse(self.run_track(zone, self.moving_in(2), [.38] * 2))
        self.assertTrue(self.run_track(self.zone(), self.moving_in(3), [.7] * 3))

    def test_weak_parked_or_too_weak_objects_never_alarm(self):
        parked = [[200, 220, 320, 300]] * 30                                 # 7.5 s, past stationary_confirm_s
        self.assertFalse(self.run_track(self.zone(ai_confidence_moving=.3), parked, [.4] * 30))
        self.assertFalse(self.run_track(self.zone(ai_confidence_moving=.3), self.moving_in(), [.25] * 6))
        self.assertFalse(self.run_track(self.zone(), self.moving_in(), [.38] * 6))


class SemanticTests(unittest.TestCase):
    def monitor(self, zones):
        m = object.__new__(SemanticMonitor)
        m.zones, m.confidence, m.interval = zones, .6, .5
        return m

    def test_moving_floor_lowers_the_detector_floor_but_not_the_display(self):
        m = self.monitor([watch.Zone(**{**EXIT_APPROACH, 'ai_confidence_moving': .3}), watch.Zone(**CORNER)])
        self.assertAlmostEqual(m._candidate_confidence(), .3)
        self.assertAlmostEqual(m._display_floor(), .5)
        m._status = {}
        m._tracker = mock.Mock()
        m._tracker.update.return_value = ([], [{'confidence': .35}, {'confidence': .35, 'confirmed_regions': ['x']},
                                               {'confidence': .8}])
        m._confirm([], 640, 360, 1.)
        self.assertEqual([d['confidence'] for d in m._status['detections']], [.35, .8])

    def test_boost_shortens_the_interval_for_a_few_seconds(self):
        m = self.monitor([])
        with mock.patch('semantic_monitor.time.monotonic', return_value=50.):
            self.assertEqual(m._current_interval(50.), .5)
            m.boost(4.)
        self.assertEqual(m._current_interval(52.), m.BOOST_INTERVAL)
        self.assertEqual(m._current_interval(54.5), .5)


class ConfigTests(unittest.TestCase):
    def config(self, **extra):
        corner = {**CORNER, 'rect': list(CORNER['rect']), **extra}
        return {'zones': [corner]}

    def test_corner_zone_is_valid_and_keeps_its_settings(self):
        result = validate_config(self.config(taillight_red_pct=2.0, taillight_change_pct=10.0))
        self.assertEqual(result['zones'][0]['trigger'], 'taillight')
        self.assertEqual(result['zones'][0]['taillight_red_pct'], 2.0)

    def test_bad_values_are_refused(self):
        for extra in ({'taillight_red_pct': 0}, {'taillight_change_pct': 500}, {'ai_confidence_moving': 2},
                      {'bottom_edge_entry': 'yes'}, {'trigger': 'sparkle'}):
            with self.assertRaises(ValueError, msg=str(extra)):
                validate_config(self.config(**extra))


if __name__ == '__main__':
    unittest.main()
