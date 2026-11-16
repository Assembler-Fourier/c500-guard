"""Localized lights stay sensitive without alarming on scene exposure/noise."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from approach_light import measure_local_brightness


class ApproachLightTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((100, 120), dtype=np.float32)
        self.background = 30 + xx * .7 + yy * .3
        self.region = np.zeros(self.background.shape, dtype=bool)
        self.region[50:95, 5:45] = True

    def measure(self, current, **kwargs):
        return measure_local_brightness(current, self.background, self.region, **kwargs)

    def test_local_headlight_pool_alerts(self):
        current = self.background.copy()
        current[60:80, 15:35] += 28
        report = self.measure(current)
        self.assertTrue(report["hit"])
        self.assertGreater(report["largest_patch_pct"], 20)
        self.assertEqual(report["global_offset"], 0)

    def test_entire_region_wash_is_not_vetoed(self):
        current = self.background.copy()
        current[self.region] += 4
        report = self.measure(current)
        self.assertTrue(report["hit"])
        self.assertEqual(report["bright_pct"], 100)

    def test_shadows_do_not_cancel_a_headlight(self):
        current = self.background.copy()
        current[50:70, 5:45] -= 18
        current[75:95, 5:45] += 18
        report = self.measure(current)
        self.assertAlmostEqual(report["raw_delta"], 0, places=4)
        self.assertTrue(report["hit"])

    def test_small_lamp_and_sensitive_threshold(self):
        current = self.background.copy()
        current[55:61, 15:21] += 140
        report = self.measure(current, min_delta=2.5)
        self.assertTrue(report["hit"])

    def test_global_additive_changes_are_rejected(self):
        for offset in (3., 12., -12., 30.):
            with self.subTest(offset=offset):
                report = self.measure(self.background + offset)
                self.assertFalse(report["hit"])
                self.assertAlmostEqual(report["global_offset"], offset)
                self.assertLess(report["positive_delta"], .001)

    def test_global_gain_is_rejected_when_every_tile_agrees(self):
        report = self.measure(self.background * 1.17 + 4)
        self.assertFalse(report["hit"])
        self.assertAlmostEqual(report["global_gain"], 1.17, places=4)

    def test_global_gain_plus_local_lamp_keeps_local_residual(self):
        current = self.background * 1.1 + 4
        current[62:71, 15:24] += 80
        report = self.measure(current)
        self.assertTrue(report["hit"])
        self.assertGreater(report["positive_delta"], 3)

    def test_local_gradient_is_not_mistaken_for_global_gain(self):
        current = self.background.copy()
        current[self.region] += np.linspace(3, 24, np.count_nonzero(self.region))
        report = self.measure(current)
        self.assertTrue(report["hit"])
        self.assertEqual(report["global_gain"], 1)

    def test_one_hot_pixel_is_rejected(self):
        current = self.background.copy()
        current[65, 15] = 255
        self.assertFalse(self.measure(current)["hit"])

    def test_sparse_rain_speckles_are_rejected_even_with_high_mean(self):
        current = self.background.copy()
        current[50:95:3, 5:45:3] = 255
        report = self.measure(current)
        self.assertGreater(report["positive_delta"], 3)
        self.assertFalse(report["hit"])
        self.assertEqual(report["largest_patch_pct"], 0)

    def test_small_camera_noise_is_rejected(self):
        rng = np.random.default_rng(22)
        current = self.background + rng.uniform(-1.5, 1.5, self.background.shape)
        self.assertFalse(self.measure(current)["hit"])

    def test_blackout_does_not_look_like_an_approaching_light(self):
        self.assertFalse(self.measure(np.zeros_like(self.background))["hit"])

    def test_changes_outside_polygon_are_ignored(self):
        current = self.background.copy()
        current[0:35, 0:40] += 80
        self.assertFalse(self.measure(current)["hit"])

    def test_exclusion_hole_is_ignored(self):
        self.region[60:80, 15:35] = False
        current = self.background.copy()
        current[60:80, 15:35] += 80
        self.assertFalse(self.measure(current)["hit"])

    def test_bezel_exclusion_allows_real_scene_exposure_compensation(self):
        scene = np.zeros_like(self.region)
        scene[15:95, 5:115] = True
        current = self.background.copy()
        current[scene] += 18
        report = self.measure(current, scene_mask=scene)
        self.assertFalse(report["hit"])
        self.assertEqual(report["global_offset"], 18)

    def test_empty_region_is_safe(self):
        self.region[:] = False
        self.assertFalse(self.measure(self.background)["hit"])

    def test_rgb_inputs_and_masks_are_not_modified(self):
        background = np.repeat(np.full((40, 40, 1), 30, dtype=np.uint8), 3, axis=2)
        current = background.copy()
        current[25:35, 5:15, :] = 65
        region = np.zeros((40, 40), dtype=np.uint8)
        region[20:, :20] = 255
        original = current.copy()
        report = measure_local_brightness(current, background, region)
        self.assertTrue(report["hit"])
        np.testing.assert_array_equal(current, original)
        self.assertEqual(int(background.max()), 30)
        self.assertEqual(int(region.max()), 255)

    def test_restless_pixels_stop_alarming_while_the_same_change_alerts_when_steady(self):
        flicker = np.zeros(self.background.shape, dtype=np.float32)
        flicker[50:95, 5:45] = 9.0            # learned foliage flicker in the region
        current = self.background.copy()
        current[60:85, 10:40] += 22           # a gust brightening those leaves
        self.assertTrue(self.measure(current)["hit"], "unchanged without a noise map")
        report = self.measure(current, noise=flicker)
        self.assertFalse(report["hit"])
        self.assertGreater(report["noise_band"], 20)
        self.assertEqual(report["positive_delta"], 0)

    def test_headlight_on_quiet_pixels_still_alerts(self):
        quiet = np.full(self.background.shape, .6, dtype=np.float32)
        current = self.background.copy()
        current[60:80, 15:35] += 28
        self.assertTrue(self.measure(current, noise=quiet)["hit"])

    def test_noise_map_and_factor_are_validated(self):
        for invalid in (np.ones((2, 2)), np.full(self.background.shape, -1.),
                        np.full(self.background.shape, np.nan)):
            with self.subTest(invalid=invalid.shape), self.assertRaises(ValueError):
                self.measure(self.background, noise=invalid)
        for value in (0, -1, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.measure(self.background, noise_k=value)

    def test_invalid_inputs_are_rejected(self):
        for invalid in (np.zeros((0, 2)), np.array([[np.nan]]), np.array([[256]])):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                measure_local_brightness(invalid, invalid, np.ones(invalid.shape))
        with self.assertRaises(ValueError):
            self.measure(self.background, scene_mask=np.ones((2, 2)))
        for name in ("min_delta", "min_bright_pct"):
            for value in (0, -1, True, float("nan")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.measure(self.background, **{name: value})


if __name__ == "__main__":
    unittest.main()
