"""Synthetic checks for conservative whole-picture exposure compensation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from illumination import compensated_motion_diff


class IlluminationTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((100, 100), dtype=np.float32)
        self.background = 70.0 + (xx + yy) / 4.0

    def test_uniform_plus_and_minus_30_are_suppressed(self):
        for offset in (30., -30.):
            with self.subTest(offset=offset):
                result = compensated_motion_diff(self.background + offset, self.background)
                np.testing.assert_allclose(result, 0.)

    def test_localized_block_is_preserved(self):
        current = self.background.copy()
        current[30:45, 30:45] += 70.
        expected = np.abs(current - self.background)
        np.testing.assert_array_equal(compensated_motion_diff(current, self.background), expected)

    def test_broad_gradient_is_not_removed(self):
        gradient = np.linspace(15., 45., 100, dtype=np.float32)[None, :]
        current = self.background + gradient
        expected = np.abs(current - self.background)
        np.testing.assert_array_equal(compensated_motion_diff(current, self.background), expected)

    def test_headlight_pool_is_not_removed(self):
        yy, xx = np.indices(self.background.shape, dtype=np.float32)
        light = 65. * np.exp(-((xx - 70.) ** 2 + (yy - 50.) ** 2) / (2. * 18. ** 2))
        current = self.background + light
        expected = np.abs(current - self.background)
        np.testing.assert_array_equal(compensated_motion_diff(current, self.background), expected)

    def test_offset_plus_object_keeps_object_and_original_light_measurement(self):
        for offset in (30., -30.):
            with self.subTest(offset=offset):
                current = self.background + offset
                current[35:45, 35:45] += 60.
                untouched = current.copy()
                result = compensated_motion_diff(current, self.background)
                expected = np.zeros_like(current)
                expected[35:45, 35:45] = 60.
                np.testing.assert_allclose(result, expected, atol=1e-5)
                np.testing.assert_array_equal(current, untouched)
                self.assertAlmostEqual(float(current.mean() - self.background.mean()), offset + .6, places=4)

    def test_globally_small_but_tile_dominating_light_change_is_ambiguous(self):
        # Only 2.5% of the picture differs from +30, but one tile has a
        # substantial local change. Conservative fallback retains raw motion.
        current = self.background + 30.
        current[:25, :10] += 20.
        expected = np.abs(current - self.background)
        np.testing.assert_array_equal(compensated_motion_diff(current, self.background), expected)

    def test_uint8_input_does_not_wrap_negative_changes(self):
        background = np.full((12, 12), 100, dtype=np.uint8)
        current = background.copy()
        current[:2, :2] = 40
        result = compensated_motion_diff(current, background)
        self.assertEqual(float(result[0, 0]), 60.)

    def test_substantial_clipping_falls_back_to_raw_difference(self):
        background = np.linspace(150., 250., 10000, dtype=np.float32).reshape(100, 100)
        current = np.minimum(background + 30., 255.)
        np.testing.assert_array_equal(compensated_motion_diff(current, background), np.abs(current - background))

    def test_invalid_arrays_fail_explicitly(self):
        for current, reference in ((np.zeros((2, 2)), np.zeros((2, 3))),
                                   (np.array([[np.nan]]), np.zeros((1, 1))),
                                   (np.zeros((0, 0)), np.zeros((0, 0)))):
            with self.subTest(shape=current.shape):
                with self.assertRaises(ValueError):
                    compensated_motion_diff(current, reference)


if __name__ == "__main__":
    unittest.main()
