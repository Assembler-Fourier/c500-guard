"""A long waiting vehicle must not blind the fallback to a later arrival."""
import unittest

import numpy as np

import test_entrance_headlights as fixtures
from exit_entry import measure_headlights


class HeadlightBaselineTests(unittest.TestCase):
    def setUp(self):
        self.road = np.full((51, 154, 3), 120., dtype=np.float32)
        self.mask = np.ones(self.road.shape[:2], dtype=bool)
        self.lamps = self.road.copy()
        self.lamps[24:38, 40:65] = 255.
        self.lamps[24:38, 95:120] = 255.

    def step(self, image, usual, count, dt=.1):
        metrics = None
        for _ in range(count):
            metrics, usual = measure_headlights(image, usual, self.mask, dt=dt)
        return metrics, usual

    def test_stationary_lamps_remain_evidence_for_ten_minutes(self):
        _, usual = self.step(self.road, None, 1)
        metrics, usual = self.step(self.lamps, usual, 600, dt=1.)
        self.assertTrue(metrics['hit'])
        self.assertEqual(0., usual[28, 50])

    def test_long_wait_then_clear_does_not_hide_the_next_car(self):
        _, usual = self.step(self.road, None, 1)
        _, usual = self.step(self.lamps, usual, 120, dt=1.)
        cleared, usual = self.step(self.road, usual, 20, dt=1.)
        self.assertFalse(cleared['hit'])
        following, _ = self.step(self.lamps, usual, 5)
        self.assertTrue(following['hit'])

    def test_unrelated_scene_pixels_keep_adapting_while_lamps_are_present(self):
        _, usual = self.step(self.road, None, 1)
        usual[:10, :] = 1.  # a previously bright patch has become dark
        before = usual.copy()
        metrics, learned = self.step(self.lamps, usual, 10, dt=1.)
        self.assertTrue(metrics['hit'])
        self.assertLess(float(learned[0, 0]), float(before[0, 0]))
        self.assertEqual(0., learned[28, 50])
        np.testing.assert_array_equal(usual, before, err_msg='caller baseline mutated')

    def test_broad_flood_is_rejected_and_still_becomes_usual_illumination(self):
        _, usual = self.step(self.road, None, 1)
        flood = np.full_like(self.road, 252.)
        metrics, learned = self.step(flood, usual, 120, dt=1.)
        self.assertFalse(metrics['hit'])
        self.assertGreater(float(learned.mean()), .3)
        self.assertAlmostEqual(float(learned.mean()), 1.-np.exp(-120./300.), places=5)

    def test_mask_holes_do_not_hold_unrelated_white_pixels(self):
        self.mask[24:38, 40:65] = False
        _, usual = self.step(self.road, None, 1)
        metrics, learned = self.step(self.lamps, usual, 100, dt=1.)
        self.assertTrue(metrics['hit'])
        self.assertGreater(float(learned[28, 50]), .25)
        self.assertEqual(0., learned[28, 100])


class WaitingVehicleAlarmTests(unittest.TestCase):
    setUp = fixtures.HeadlightAlarmTests.setUp
    deliver = fixtures.HeadlightAlarmTests.deliver

    def test_acknowledged_long_wait_stays_quiet_but_next_arrival_alarms(self):
        self.deliver(self.road, 2)
        lamps = fixtures.lamp_pair(self.road)
        self.deliver(lamps, 5)
        self.assertEqual(self.obj.alarm, 'main')
        self.assertTrue(self.obj.acknowledge(self.obj.alarm_id, via='api'))
        # The map used to absorb the lamp pixels at 86.5 s. Distinct samples
        # remain present across a two-minute wait, without another alarm.
        for _ in range(120):
            self.now += 1.
            self.obj.process(lamps)
        self.assertTrue(self.lane.light_active)
        self.assertTrue(self.lane.light_metrics['hit'])
        self.assertIsNone(self.obj.alarm)
        self.deliver(self.road, 200)
        self.assertFalse(self.lane.light_active)
        self.deliver(lamps, 5)
        self.assertEqual(self.obj.alarm, 'main')


if __name__ == '__main__':
    unittest.main()
