"""Large scene-wide steps are exposure; local road changes remain immediate.

These synthetic sequences establish the specific additive-exposure correction.
They do not pretend to classify the origin of a localized headlight reflection.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from approach_light import measure_urgent_change


class LargeExposureTests(unittest.TestCase):
    def setUp(self):
        self.region = np.zeros((180, 320), dtype=bool)
        self.region[80:120, 40:120] = True
        self.scene = np.ones(self.region.shape, dtype=bool)

    def measure(self, current, background):
        return measure_urgent_change(current, background, self.region,
                                     scene_mask=self.scene)

    def test_large_bright_and_dark_global_steps_do_not_sound(self):
        for base, step in ((60, 80), (60, 130), (180, -80), (180, -130)):
            with self.subTest(base=base, step=step):
                background = np.full(self.region.shape, base, dtype=np.float32)
                result = self.measure(background + step, background)
                self.assertFalse(result["hit"])
                self.assertFalse(result["weak_hit"])
                self.assertEqual(result["exposure_fit"], "additive")
                self.assertAlmostEqual(result["global_offset"], step)

    def test_local_bright_and_dark_arrivals_survive_large_transition(self):
        for base, step in ((60, 100), (180, -100)):
            for local in (-25, 25):
                with self.subTest(base=base, step=step, local=local):
                    background = np.full(self.region.shape, base, dtype=np.float32)
                    current = background + step
                    current[94:102, 75:83] += local
                    result = self.measure(current, background)
                    self.assertTrue(result["hit"])
                    self.assertAlmostEqual(result["global_offset"], step)
                    self.assertGreaterEqual(result["largest_patch_pixels"], 48)

    def test_outside_vehicle_during_large_transition_does_not_spoil_fit(self):
        background = np.full(self.region.shape, 60, dtype=np.float32)
        current = background + 100
        current[20:40, 220:250] += 50
        self.assertFalse(self.measure(current, background)["hit"])
        current[94:102, 75:83] -= 25
        self.assertTrue(self.measure(current, background)["hit"])

    def test_gain_free_steps_with_partial_saturation_still_correct(self):
        _yy, xx = np.mgrid[:180, :320]
        background = (40 + 150 * xx / 320).astype(np.float32)
        for step in (-90, 90):
            with self.subTest(step=step):
                current = np.clip(background + step, 0, 255)
                self.assertFalse(self.measure(current, background)["hit"])

    def test_night_headlight_patch_is_eligible_on_first_touched_frame(self):
        background = np.full(self.region.shape, 35, dtype=np.float32)
        for left in (38, 40, 43, 48):
            with self.subTest(left=left):
                current = background.copy()
                current[94:104, left:left + 10] += 25
                self.assertTrue(self.measure(current, background)["hit"])

    def test_dark_car_on_textured_road_remains_immediate(self):
        random = np.random.default_rng(20260912)
        background = random.uniform(77, 83, self.region.shape).astype(np.float32)
        current = background.copy()
        current[94:104, 70:84] -= 25
        self.assertTrue(self.measure(current, background)["hit"])

    def test_smooth_local_headlight_pool_is_not_blanket_vetoed(self):
        yy, xx = np.mgrid[:180, :320]
        background = np.full(self.region.shape, 35, dtype=np.float32)
        local = 45 * np.exp(-((xx - 80) ** 2 / (2 * 14 ** 2)
                             + (yy - 100) ** 2 / (2 * 9 ** 2)))
        result = self.measure((background + local).astype(np.float32), background)
        self.assertTrue(result["hit"])

    def test_whole_road_headlight_wash_is_not_erased_by_its_own_pixels(self):
        background = np.full(self.region.shape, 50, dtype=np.float32)
        current = background.copy()
        current[self.region] += 35
        result = self.measure(current, background)
        self.assertTrue(result["hit"])
        self.assertEqual(result["global_offset"], 0)

    def test_partial_scene_spill_cannot_establish_a_global_step(self):
        background = np.full(self.region.shape, 50, dtype=np.float32)
        current = background.copy()
        current[70:140, 20:160] += 90
        result = self.measure(current, background)
        # It could be the very headlight pool the user needs to hear. A single
        # intensity pair cannot establish whether it came from exit or entry.
        self.assertTrue(result["hit"])
        self.assertEqual(result["global_offset"], 0)

    def test_urgent_pixels_cannot_change_large_exposure_estimate(self):
        background = np.full(self.region.shape, 60, dtype=np.float32)
        current = background + 100
        before = self.measure(current, background)
        current[self.region] += 25
        after = self.measure(current, background)
        self.assertEqual(before["global_offset"], after["global_offset"])
        self.assertTrue(after["hit"])


if __name__ == "__main__":
    unittest.main()
