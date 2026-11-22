"""Entrance lane headlights: fallback alarm when an arriving car's headlights blind the AI (21 Nov 2026)."""
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher
from exit_entry import measure_headlights
from zone_editor import validate_config

LANE = dict(name="Entrance lane - headlights", rect=(.66, .36, .24, .14),
            polygon=[[.66, .36], [.90, .36], [.90, .50], [.66, .50]], trigger="headlights", classes=[],
            motion=False, light=True, severity="main", light_severity="main", on_frames=5, off_frames=30,
            priority=True, route_group="campus-entrance", route_role="entrance", alert_stage="barrier")


def lamp_pair(frame, value=255., y=.44, xs=(.74, .80), r=.02):
    out = frame.copy()
    for x in xs:
        out[int((y - r) * 360):int((y + r) * 360), int((x - r) * 640):int((x + r) * 640)] = value
    return out


class MeasureHeadlightTests(unittest.TestCase):
    def setUp(self):
        self.road = np.full((360, 640, 3), 120., np.float32)          # a lit, but not white, road
        self.mask = watch.Zone(**LANE).mask_for(640, 360)

    def run_frames(self, frames, usual=None, dt=.1):
        metrics = None
        for frame in frames:
            metrics, usual = measure_headlights(frame, usual, self.mask, dt=dt)
        return metrics, usual

    def test_new_headlights_hit_after_the_baseline_exists(self):
        first, base = measure_headlights(self.road, None, self.mask)
        self.assertTrue(first.get("starting"))
        metrics, _ = measure_headlights(lamp_pair(self.road), base, self.mask)
        self.assertTrue(metrics["hit"])
        self.assertGreater(metrics["bright_pct"], 2)
        self.assertIsNotNone(metrics["support_bounds"])

    def test_a_waiting_car_is_not_learned_into_the_empty_scene(self):
        lit = lamp_pair(self.road)
        metrics, usual = self.run_frames([self.road] + [lit] * 600)             # a minute at 10 fps
        self.assertTrue(metrics["hit"])
        metrics, _ = self.run_frames([lit] * 300, usual, dt=1.)                 # five more minutes: still the same car
        self.assertTrue(metrics["hit"])

    def test_a_dark_object_crossing_the_floodlit_road_leaves_no_trail(self):
        # 20 Nov 23:47: the arm crossed the near-white far lane; the white road reappearing is not new light.
        white = np.full((360, 640, 3), 250., np.float32)
        arm = white.copy(); arm[150:158, 420:580] = 40.
        sequence = [white] * 50 + [arm] * 30 + [white] * 40
        self.assertFalse(any(measure_headlights(f, u, self.mask)[0]["hit"]
                             for f, u in self._walk(sequence)))

    def test_a_thin_floodlit_arm_is_not_a_bloom(self):
        streak = self.road.copy()
        for x in range(430, 570):                                               # a 2-pixel diagonal white line
            y = 140 + (x - 430) // 5
            streak[y:y + 2, x] = 255.
        self.assertFalse(self.run_frames([self.road, streak])[0]["hit"])

    def _walk(self, frames):
        usual = None
        for frame in frames:
            yield frame, usual
            _, usual = measure_headlights(frame, usual, self.mask)

    def test_already_white_road_dim_lamps_and_floods_do_not_hit(self):
        white = np.full((360, 640, 3), 250., np.float32)
        self.assertFalse(self.run_frames([white, lamp_pair(white)])[0]["hit"])     # no new light on a white road
        self.assertFalse(self.run_frames([self.road, lamp_pair(self.road, value=200.)])[0]["hit"])
        self.assertFalse(self.run_frames([self.road, np.full_like(self.road, 252.)])[0]["hit"])  # floodlight on
        self.assertFalse(self.run_frames([self.road, lamp_pair(self.road, r=.003)])[0]["hit"])   # tiny glint

    def test_red_tail_lights_are_not_headlights(self):
        tail = self.road.copy(); tail[150:170, 470:500] = (255, 40, 40)
        self.assertFalse(self.run_frames([self.road, tail])[0]["hit"])

    def test_bad_input_is_harmless(self):
        metrics, usual = measure_headlights(self.road[..., 0], None, self.mask)
        self.assertFalse(metrics["hit"])
        self.assertIsNone(usual)


class HeadlightAlarmTests(unittest.TestCase):
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
        self.lane = watch.Zone(**LANE)
        self.obj.zones = [self.lane]
        self.obj.semantic = mock.Mock()
        self.obj.semantic.update.return_value = []
        self.obj.semantic.status.return_value = {"available": True, "state": "ready", "age_s": 0., "detections": []}
        self.road = np.full((360, 640, 3), 120., np.float32)
        self.obj.process(self.road)

    def deliver(self, frame, times=1):
        for _ in range(times):
            self.now += .1
            self.obj.process(frame)

    def test_headlights_held_half_a_second_raise_the_entrance_alarm(self):
        self.deliver(self.road, 2)
        self.deliver(lamp_pair(self.road), 4)
        self.assertIsNone(self.obj.alarm)                                       # not yet: needs five frames
        self.deliver(lamp_pair(self.road))
        self.assertEqual(self.obj.alarm, 'main')
        details = self.obj.alarm_details
        self.assertEqual((details['source'], details['trigger'], details['route_group'], details['alert_stage']),
                         ('light', 'headlights', 'campus-entrance', 'barrier'))
        self.assertIn('headlights facing the entrance barrier', self.obj.alarm_reason)

    def test_the_same_waiting_car_is_announced_once_and_a_later_car_again(self):
        self.deliver(self.road, 2)
        lit = lamp_pair(self.road)
        self.deliver(lit, 5)
        self.assertEqual(self.obj.alarm, 'main')
        self.assertTrue(self.obj.acknowledge(self.obj.alarm_id, via='api'))
        self.deliver(lit, 300)                       # 30 s more, well past the 10 s quiet: the same car waits
        self.assertIsNone(self.obj.alarm)
        self.deliver(self.road, 200)                 # it drives in; the lane stays dark for 20 s
        self.deliver(lit, 5)                         # the next car's headlights
        self.assertEqual(self.obj.alarm, 'main')

    def test_a_flash_does_not_alarm(self):
        self.deliver(self.road, 2)
        self.deliver(lamp_pair(self.road), 2)
        self.deliver(self.road, 5)
        self.assertIsNone(self.obj.alarm)

    def test_rearm_restarts_the_baseline(self):
        self.deliver(self.road, 2)
        self.assertIsNotNone(getattr(self.lane, '_headlight_usual', None))
        with self.obj._state_lock:
            self.obj._pending_rearm = True
        self.deliver(self.road)                     # the re-arm frame rebuilds the watcher background first
        self.assertIsNone(getattr(self.lane, '_headlight_usual', None))
        self.deliver(self.road)
        self.assertEqual(getattr(self.lane, '_headlight_usual').shape, (51, 154))  # rebuilt from the zone's box


    def _shifted_white_road(self, correction):
        lit_road = self.road.copy(); lit_road[150:170, 430:570] = 250.          # the floodlit far lane
        moved = self.road.copy(); moved[155:175, 430:570] = 250.                # the same scene 5 px lower
        self.deliver(lit_road, 3)
        if correction:
            with self.obj._state_lock:
                self.obj._pending_background_reset = True                    # verified correction > 2 px
        self.deliver(moved, 10)
        return self.obj.alarm

    def test_a_camera_correction_over_two_pixels_restarts_the_lane_map(self):
        # A shifted picture moves the white road's edge into "usually dark" pixels; it must not look like new light.
        self.assertIsNone(self._shifted_white_road(correction=True))

    def test_without_the_restart_a_shifted_picture_would_alarm(self):
        self.assertEqual(self._shifted_white_road(correction=False), 'main')


class HeadlightConfigTests(unittest.TestCase):
    def test_zone_is_valid_and_bad_values_are_refused(self):
        lane = {**LANE, "rect": list(LANE["rect"])}
        validate_config({"zones": [{**lane, "headlight_pct": 2.0}]})
        for bad in ({"headlight_pct": 0}, {"headlight_white": 100}, {"headlight_learn_s": 1}, {"headlight_max_pct": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_config({"zones": [{**lane, **bad}]})


if __name__ == "__main__":
    unittest.main()
