"""The urgent channel measures inside a box now. It must see exactly what it saw before.

The slow, obviously-correct full-frame computation lives here in the test. Every
case asserts the shipped function agrees with it, so the speed-up can never
quietly change what counts as a change at the barrier.
"""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import approach_light as al  # noqa: E402


def reference_measurement(current, predicted, region, band):
    """The original whole-frame version, kept as the yardstick."""
    neighborhood = np.ones((5, 5), dtype=np.uint8)
    lower = cv2.erode(predicted, neighborhood, borderType=cv2.BORDER_REPLICATE)
    upper = cv2.dilate(predicted, neighborhood, borderType=cv2.BORDER_REPLICATE)
    difference = np.maximum(np.maximum(current - upper, lower - current), 0)
    changed = ((difference > band) & region).astype(np.uint8)
    neighbors = cv2.filter2D(changed, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    supported = changed & (neighbors >= 5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(supported, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    bounds = None
    if largest:
        component = stats[1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())]
        bounds = [int(v) for v in component[:4]]
    return {"changed_pixels": int(np.count_nonzero(changed)),
            "largest_patch_pixels": largest, "support_bounds": bounds}


class RegionBoxTests(unittest.TestCase):
    def test_box_covers_the_region_with_room_for_the_kernels(self):
        region = np.zeros((100, 120), dtype=bool)
        region[40:60, 30:50] = True
        rows, columns = al._region_box(region, pad=8)
        self.assertEqual((rows.start, rows.stop), (32, 68))
        self.assertEqual((columns.start, columns.stop), (22, 58))
        self.assertTrue(region[rows, columns].sum() == region.sum())

    def test_a_region_at_the_frame_edge_keeps_the_edge(self):
        region = np.zeros((60, 60), dtype=bool)
        region[0:5, 55:60] = True
        rows, columns = al._region_box(region, pad=8)
        self.assertEqual((rows.start, columns.stop), (0, 60))

    def test_an_empty_region_gives_the_whole_frame_and_never_raises(self):
        region = np.zeros((30, 40), dtype=bool)
        rows, columns = al._region_box(region)
        self.assertEqual((rows.start, rows.stop, columns.start, columns.stop), (0, 30, 0, 40))


class UrgentChangeAgreesWithTheWholeFrameTests(unittest.TestCase):
    def setUp(self):
        self.random = np.random.default_rng(20260912)

    def scene(self, *, shape=(180, 320)):
        background = self.random.uniform(40, 200, shape).astype(np.float32)
        background = cv2.GaussianBlur(background, (7, 7), 0)
        return background

    def region_at(self, shape, top, left, height, width):
        region = np.zeros(shape, dtype=bool)
        region[top:top + height, left:left + width] = True
        return region

    def check(self, current, background, region, threshold=6.0, min_patch_pct=0.6, noise=None):
        scene = np.ones(current.shape, dtype=bool)
        report = al.measure_urgent_change(current, background, region, scene_mask=scene,
                                          threshold=threshold, min_patch_pct=min_patch_pct,
                                          noise=noise)
        predicted, offset, gain, _ = al._urgent_exposure_reference(
            current.astype(np.float32), background.astype(np.float32), scene, region)
        band = np.full(current.shape, float(threshold), dtype=np.float32)
        if noise is not None:
            band = np.maximum(band, np.minimum(np.asarray(noise, np.float32) * 2.0, 12.0))
        expected = reference_measurement(current.astype(np.float32), predicted, region, band)
        self.assertEqual(report["changed_pixels"], expected["changed_pixels"])
        self.assertEqual(report["largest_patch_pixels"], expected["largest_patch_pixels"])
        self.assertEqual(report["support_bounds"], expected["support_bounds"])
        return report

    def test_a_car_sized_patch_reads_the_same(self):
        background = self.scene()
        current = background.copy()
        current[90:110, 140:180] += 60          # a bright arrival inside the region
        region = self.region_at(background.shape, 80, 130, 45, 70)
        report = self.check(current, background, region)
        self.assertTrue(report["hit"])

    def test_a_quiet_scene_reads_the_same(self):
        background = self.scene()
        current = background + self.random.normal(0, .8, background.shape).astype(np.float32)
        region = self.region_at(background.shape, 80, 130, 45, 70)
        report = self.check(np.clip(current, 0, 255), background, region)
        self.assertFalse(report["hit"])

    def test_a_change_that_straddles_the_region_edge_reads_the_same(self):
        background = self.scene()
        current = background.copy()
        current[70:95, 120:150] -= 50           # darker, half in and half out
        region = self.region_at(background.shape, 80, 130, 45, 70)
        self.check(np.clip(current, 0, 255), background, region)

    def test_a_region_touching_every_frame_edge_reads_the_same(self):
        background = self.scene(shape=(120, 140))
        current = background.copy()
        current[0:12, 0:12] += 70
        region = np.ones(background.shape, dtype=bool)
        self.check(current, background, region)

    def test_a_learned_flicker_map_still_applies_inside_the_box(self):
        """Flicker raises the bar for small changes, and is capped for big ones."""
        background = self.scene()
        region = self.region_at(background.shape, 80, 130, 45, 70)
        noise = np.full(background.shape, 3.0, dtype=np.float32)
        noise[95:105, 150:165] = 25.0           # this patch is known to flicker

        rustling = background.copy()
        rustling[95:105, 150:165] += 10         # leaves moving: under the 12-level band
        self.assertFalse(self.check(rustling, background, region, noise=noise)["hit"])

        headlights = background.copy()
        headlights[95:105, 150:165] += 60       # a real arrival clears the capped band
        self.assertTrue(self.check(headlights, background, region, noise=noise)["hit"],
                        "the flicker band is capped so activity cannot hide a vehicle")

    def test_support_bounds_are_reported_in_frame_coordinates(self):
        background = self.scene()
        current = background.copy()
        current[100:115, 200:230] += 80
        region = self.region_at(background.shape, 90, 190, 40, 60)
        report = self.check(current, background, region)
        left, top, width, height = report["support_bounds"]
        self.assertGreater(left, 150, "a box-local coordinate would be far smaller")
        self.assertGreater(top, 60)
        self.assertTrue(region[top:top + height, left:left + width].any())

    def test_many_random_scenes_agree(self):
        for _ in range(12):
            background = self.scene()
            current = background.copy()
            top, left = int(self.random.integers(20, 120)), int(self.random.integers(20, 240))
            current[top:top + 12, left:left + 18] += float(self.random.uniform(-80, 80))
            region = self.region_at(background.shape, max(0, top - 15), max(0, left - 20), 50, 70)
            self.check(np.clip(current, 0, 255), background, region)


def reference_light(current, predicted, reference, region, band, min_delta, min_bright_pct):
    """The original whole-frame light measurement, kept as the yardstick."""
    difference = current - predicted
    excess = np.maximum(difference - band, 0.0)
    bright = ((excess >= max(2.0, float(min_delta))) & region).astype(np.uint8)
    neighbors = cv2.filter2D(bright, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    supported = bright & (neighbors >= 4).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(supported, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    area = int(np.count_nonzero(region))
    return {"delta": float(difference[region].mean()),
            "positive_delta": float(excess[region].mean()),
            "raw_delta": float((current - reference)[region].mean()),
            "bright_pct": 100.0 * int(np.count_nonzero(bright)) / area,
            "largest_patch_pct": 100.0 * largest / area}


class LightMeasurementAgreesWithTheWholeFrameTests(unittest.TestCase):
    def setUp(self):
        self.random = np.random.default_rng(12092026)

    def scene(self, shape=(180, 320)):
        return cv2.GaussianBlur(self.random.uniform(40, 200, shape).astype(np.float32), (7, 7), 0)

    def check(self, current, background, region, *, min_delta=3.0, min_bright_pct=2.0, noise=None):
        scene = np.ones(current.shape, dtype=bool)
        report = al.measure_local_brightness(current, background, region, scene_mask=scene,
                                             min_delta=min_delta, min_bright_pct=min_bright_pct,
                                             noise=noise, noise_k=3.0)
        predicted, _, _ = al._exposure_reference(current.astype(np.float32),
                                                 background.astype(np.float32), scene)
        band = (np.zeros(current.shape, np.float32) if noise is None
                else np.asarray(noise, np.float32) * 3.0)
        expected = reference_light(current.astype(np.float32), predicted,
                                   background.astype(np.float32), region, band,
                                   min_delta, min_bright_pct)
        for key, value in expected.items():
            self.assertAlmostEqual(report[key], value, places=5, msg=key)
        return report

    def region_at(self, shape, top, left, height, width):
        region = np.zeros(shape, dtype=bool)
        region[top:top + height, left:left + width] = True
        return region

    def test_a_headlight_wash_reads_the_same(self):
        background = self.scene()
        current = background.copy()
        current[95:115, 150:190] += 45
        region = self.region_at(background.shape, 85, 140, 45, 70)
        self.assertTrue(self.check(np.clip(current, 0, 255), background, region)["hit"])

    def test_a_quiet_region_reads_the_same(self):
        background = self.scene()
        current = np.clip(background + self.random.normal(0, .6, background.shape), 0, 255).astype(np.float32)
        region = self.region_at(background.shape, 85, 140, 45, 70)
        self.assertFalse(self.check(current, background, region)["hit"])

    def test_the_learned_flicker_band_reads_the_same(self):
        background = self.scene()
        current = background.copy()
        current[95:115, 150:190] += 12
        region = self.region_at(background.shape, 85, 140, 45, 70)
        noise = np.full(background.shape, 1.0, dtype=np.float32)
        noise[95:115, 150:190] = 8.0
        report = self.check(np.clip(current, 0, 255), background, region, noise=noise)
        self.assertFalse(report["hit"], "a change inside the flicker band is not a headlight")

    def test_a_region_at_the_frame_corner_reads_the_same(self):
        background = self.scene((120, 160))
        current = background.copy()
        current[0:20, 0:20] += 50
        self.check(current, background, self.region_at(background.shape, 0, 0, 30, 30))

    def test_a_wrong_sized_flicker_map_is_still_refused(self):
        background = self.scene()
        region = self.region_at(background.shape, 85, 140, 45, 70)
        with self.assertRaises(ValueError):
            al.measure_local_brightness(background, background, region,
                                        noise=np.zeros((10, 10), dtype=np.float32))

    def test_many_random_scenes_agree(self):
        for _ in range(10):
            background = self.scene()
            current = background.copy()
            top, left = int(self.random.integers(10, 130)), int(self.random.integers(10, 250))
            current[top:top + 15, left:left + 20] += float(self.random.uniform(-60, 60))
            region = self.region_at(background.shape, max(0, top - 10), max(0, left - 15), 45, 60)
            self.check(np.clip(current, 0, 255), background, region)


if __name__ == "__main__":
    unittest.main()
