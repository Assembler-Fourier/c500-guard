"""Time-based scene diagnostics with real crop, exposure and persistence cases."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from illumination import LightingMonitor


def solid(value, size=(100, 100)):
    return np.full((*size, 3), value, dtype=np.uint8)


class LightingMonitorTests(unittest.TestCase):
    def test_scene_crop_excludes_monitor_borders_without_discarding_dark_scene(self):
        frame = solid(0)
        frame[25:75, 25:75] = 100
        monitor = LightingMonitor()
        result = monitor.update(frame, (.25, .25, .5, .5), now=0)
        self.assertEqual(result["mode"], "normal")
        self.assertEqual(result["median_luma"], 100.0)
        self.assertEqual(result["dark_fraction"], 0.0)
        for now in range(1, 6):
            result = monitor.update(solid(0), (.25, .25, .5, .5), now=now)
        self.assertEqual(result["state"], "near_black")

    def test_local_headlights_do_not_classify_the_scene_as_glare(self):
        frame = solid(80)
        frame[20:40, 20:40] = 255
        monitor = LightingMonitor()
        for now in range(8):
            result = monitor.update(frame, now=now)
        self.assertEqual(result["mode"], "normal")
        self.assertEqual(result["state"], "stable")
        self.assertAlmostEqual(result["bright_fraction"], .04)

    def test_transient_exposure_change_settles_and_is_not_immediate_glare(self):
        monitor = LightingMonitor()
        monitor.update(solid(100), now=0)
        result = monitor.update(solid(245), now=.1)
        self.assertEqual(result["state"], "adjusting")
        self.assertEqual(result["condition"], "glare")
        result = monitor.update(solid(100), now=.5)
        self.assertEqual(result["state"], "adjusting")
        monitor.update(solid(100), now=1.5)
        result = monitor.update(solid(100), now=2.5)
        self.assertEqual(result["state"], "stable")
        self.assertIsNone(result["condition"])

    def test_sustained_glare_requires_continuous_four_seconds(self):
        monitor = LightingMonitor()
        for now in (0, 1, 2, 3, 3.99):
            result = monitor.update(solid(245), now=now)
            self.assertNotEqual(result["state"], "glare")
        self.assertEqual(monitor.update(solid(245), now=4)["state"], "glare")
        self.assertNotEqual(monitor.update(solid(100), now=4.1)["state"], "glare")

    def test_interrupted_darkness_does_not_accumulate_to_a_fault(self):
        monitor = LightingMonitor()
        for now, value in ((0, 0), (1, 0), (2, 100), (3, 0), (4, 0), (5, 0), (6, 0)):
            result = monitor.update(solid(value), now=now)
            self.assertNotEqual(result["state"], "near_black")
        self.assertEqual(monitor.update(solid(0), now=7)["state"], "near_black")

    def test_feed_gap_resets_dark_persistence_and_exposes_staleness(self):
        monitor = LightingMonitor()
        monitor.update(solid(0), now=0)
        monitor.update(solid(0), now=1)
        self.assertEqual(monitor.status(now=4)["state"], "stale")
        self.assertFalse(monitor.status(now=4)["available"])
        result = monitor.update(solid(0), now=6)
        self.assertNotEqual(result["state"], "near_black")
        self.assertEqual(result["condition_duration_s"], 0)

    def test_changed_crop_resets_persistence(self):
        monitor = LightingMonitor()
        for now in range(4):
            monitor.update(solid(0), now=now)
        result = monitor.update(solid(0), (.1, .1, .8, .8), now=4)
        self.assertNotEqual(result["state"], "near_black")
        self.assertEqual(result["condition_duration_s"], 0)

    def test_dim_mode_hysteresis_prevents_boundary_flicker(self):
        monitor = LightingMonitor(smoothing_s=.01)
        self.assertEqual(monitor.update(solid(40), now=0)["mode"], "dim")
        for now, value in enumerate((47, 53, 48, 57), 1):
            self.assertEqual(monitor.update(solid(value), now=now)["mode"], "dim")
        self.assertEqual(monitor.update(solid(60), now=5)["mode"], "normal")
        self.assertEqual(monitor.update(solid(48), now=6)["mode"], "normal")
        self.assertEqual(monitor.update(solid(40), now=7)["mode"], "dim")

    def test_bright_mode_hysteresis_prevents_boundary_flicker(self):
        monitor = LightingMonitor(smoothing_s=.01)
        self.assertEqual(monitor.update(solid(200), now=0)["mode"], "bright")
        self.assertEqual(monitor.update(solid(180), now=1)["mode"], "bright")
        self.assertEqual(monitor.update(solid(170), now=2)["mode"], "normal")
        self.assertEqual(monitor.update(solid(185), now=3)["mode"], "normal")

    def test_time_based_smoothing_is_independent_of_frame_rate(self):
        endpoints = []
        for rate in (2, 10):
            monitor = LightingMonitor()
            monitor.update(solid(40), now=0)
            for index in range(1, 4 * rate + 1):
                result = monitor.update(solid(100), now=index / rate)
            endpoints.append(result["smoothed_luma"])
        self.assertEqual(endpoints[0], endpoints[1])

    def test_status_does_not_expose_mutable_state_and_is_json_serializable(self):
        monitor = LightingMonitor()
        result = monitor.update(solid(100), now=0)
        result["scene_rect"][0] = .5
        self.assertEqual(monitor.status(now=0)["scene_rect"][0], 0)
        json.dumps(monitor.status(now=0), allow_nan=False)
        monitor.reset()
        self.assertEqual(monitor.status(now=0)["state"], "unavailable")

    def test_input_not_mutated_and_rgb_luma_is_used(self):
        frame = solid(0)
        frame[:, :, 1] = 100
        before = frame.copy()
        result = LightingMonitor().update(frame, now=0)
        self.assertAlmostEqual(result["median_luma"], 58.7)
        np.testing.assert_array_equal(frame, before)

    def test_invalid_input_rejected_without_changing_previous_status(self):
        monitor = LightingMonitor()
        monitor.update(solid(100), now=1)
        for frame, crop, now in ((solid(100), None, 0), (solid(100), None, float("nan")),
                                 (np.zeros((2, 2)), None, 2), (solid(1, (0, 2)), None, 2),
                                 (solid(100), (.9, 0, .2, 1), 2),
                                 (solid(100), (0, 0, 0, 1), 2),
                                 (solid(100), (0, 0, float("nan"), 1), 2)):
            with self.subTest(crop=crop, now=now), self.assertRaises(ValueError):
                monitor.update(frame, crop, now)
        self.assertEqual(monitor.status(now=1)["median_luma"], 100.0)

    def test_invalid_durations_rejected(self):
        for value in (0, -1, True, float("inf"), "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LightingMonitor(sustained_s=value)


if __name__ == "__main__":
    unittest.main()
