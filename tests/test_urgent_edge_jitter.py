"""Two-pixel registration tolerance must preserve real new road material."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from approach_light import measure_urgent_change


class UrgentEdgeJitterTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((120, 160), 80., dtype=np.float32)
        self.region = np.zeros(self.background.shape, dtype=bool)
        self.region[30:100, 30:130] = True

    def measure(self, current):
        return measure_urgent_change(current, self.background, self.region)

    def shift(self, dx, dy):
        return cv2.warpAffine(self.background, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (160, 120), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT_101)

    def test_diagonal_edge_one_and_two_pixel_translations_stay_quiet(self):
        cv2.line(self.background, (25, 100), (130, 35), 145., 2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2)):
            with self.subTest(dx=dx, dy=dy):
                report = self.measure(self.shift(dx, dy))
                self.assertFalse(report["hit"])
                self.assertFalse(report["weak_hit"])
                self.assertGreater(report["raw_changed_pixels"], 48)
                self.assertEqual(report["edge_rejected_pixels"], report["raw_changed_pixels"])

    def test_textured_scene_one_and_two_pixel_translations_stay_quiet(self):
        rng = np.random.default_rng(593)
        self.background = rng.uniform(45, 155, self.background.shape).astype(np.float32)
        for dx, dy in ((1, -1), (2, 0), (0, 2), (-2, -2)):
            with self.subTest(dx=dx, dy=dy):
                report = self.measure(self.shift(dx, dy))
                self.assertFalse(report["hit"])
                self.assertFalse(report["weak_hit"])
                self.assertGreater(report["edge_rejected_pixels"], 100)

    def test_bright_and_dark_new_material_remain_immediate(self):
        cv2.line(self.background, (25, 100), (130, 35), 145., 2)
        for delta in (20., -20.):
            current = self.shift(2, 1)
            current[40:48, 45:53] += delta
            report = self.measure(current)
            self.assertTrue(report["hit"])
            self.assertGreaterEqual(report["largest_patch_pixels"], 48)
            self.assertGreater(report["edge_rejected_pixels"], 48)

    def test_new_object_on_textured_road_remains_immediate(self):
        rng = np.random.default_rng(941)
        self.background += rng.uniform(-3, 3, self.background.shape).astype(np.float32)
        for delta in (20., -20.):
            current = self.shift(2, -1)
            current[50:58, 60:68] += delta
            self.assertTrue(self.measure(current)["hit"])

    def test_nine_level_small_lamp_on_flat_road_still_uses_weak_path(self):
        for delta in (9., -9.):
            current = self.background.copy()
            current[50:58, 60:68] += delta
            report = self.measure(current)
            self.assertFalse(report["hit"])
            self.assertTrue(report["weak_hit"])

    def test_uniform_exposure_and_local_object_behaviour_is_preserved(self):
        for offset in (20., -20., 60.):
            current = self.background + offset
            report = self.measure(current)
            self.assertFalse(report["weak_hit"])
            self.assertAlmostEqual(report["global_offset"], offset)
            for delta in (20., -20.):
                local = current.copy()
                local[50:58, 60:68] += delta
                self.assertTrue(self.measure(local)["hit"])

    def test_input_images_and_mask_are_unchanged(self):
        current = self.background.copy()
        current[50:58, 60:68] += 20
        originals = [a.copy() for a in (current, self.background, self.region)]
        self.measure(current)
        for actual, expected in zip((current, self.background, self.region), originals):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
