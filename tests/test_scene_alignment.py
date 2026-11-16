"""Deterministic synthetic alignment checks; no camera/network access."""
import dataclasses
import os
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scene_alignment import SceneAligner, AlignmentConfig, build_feature_mask


def textured_scene(seed=902, width=640, height=360):
    rng = np.random.default_rng(seed)
    scene = np.full((height, width, 3), (35, 45, 55), np.uint8)
    for _ in range(340):
        point = tuple(rng.integers([20, 20], [width - 20, height - 20]).tolist())
        color = tuple(rng.integers(60, 250, size=3).tolist())
        cv2.circle(scene, point, int(rng.integers(2, 9)), color, -1)
    for row in range(3):
        for col in range(5):
            cv2.putText(scene, f"{row}x{col}", (30 + col * 117, 55 + row * 110),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    return scene


def shifted(scene, dx=0., dy=0., angle=0., scale=1.):
    height, width = scene.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2., height / 2.), angle, scale)
    transform[:, 2] += [dx, dy]
    return cv2.warpAffine(scene, transform, (width, height)), transform


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.reference = textured_scene()
        self.mask = build_feature_mask(self.reference.shape, border_px=15)
        self.aligner = SceneAligner(self.reference, self.mask)
        self.assertTrue(self.aligner.reference_ready, self.aligner.status())

    def test_feature_polygon_intersects_existing_motion_exclusions(self):
        mask = build_feature_mask(self.reference.shape,
            include_polygon=[[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
            exclude_rects=[[.3, .3, .2, .2]], border_px=0)
        self.assertEqual(mask[180, 10], 0)
        self.assertEqual(mask[150, 250], 0)
        self.assertEqual(mask[80, 320], 255)

    def test_feature_polygon_rejects_unbounded_geometry(self):
        with self.assertRaises(ValueError):
            build_feature_mask(self.reference.shape,
                               include_polygon=[[0, 0], [1.1, 0], [1, 1]])

    def assert_transform(self, dx=0., dy=0., angle=0., scale=1.):
        current, forward = shifted(self.reference, dx, dy, angle, scale)
        original_current = current.copy()
        aligned, status = self.aligner.update(current)
        self.assertTrue(status["accepted"], status)
        expected = cv2.invertAffineTransform(forward)
        probes = np.float64([[60, 60, 1], [550, 65, 1], [120, 290, 1], [560, 295, 1]])
        estimated = np.array(status["current_to_reference"])
        errors = np.linalg.norm(probes @ estimated.T - probes @ expected.T, axis=1)
        self.assertLess(errors.max(), .7, status)
        np.testing.assert_array_equal(current, original_current)
        self.assertEqual(aligned.shape, current.shape)
        self.assertEqual(aligned.dtype, np.uint8)
        return aligned, status

    def test_identity_preserves_reference_and_input(self):
        aligned, _ = self.assert_transform()
        np.testing.assert_array_equal(aligned, self.reference)

    def test_small_translation_recovered(self):
        self.assert_transform(dx=7, dy=-5)

    def test_small_rotation_scale_and_translation_recovered(self):
        self.assert_transform(dx=4, dy=-3, angle=1.4, scale=1.015)

    def test_large_translation_rotation_and_scale_rejected(self):
        for settings in ({"dx": 42}, {"angle": 8}, {"scale": 1.13}):
            current, _ = shifted(self.reference, **settings)
            aligned, status = self.aligner.update(current)
            self.assertIsNone(aligned)
            self.assertFalse(status["accepted"])
            self.assertIsNone(self.aligner.valid_mask)

    def test_changed_scene_rejected(self):
        aligned, status = self.aligner.update(textured_scene(seed=999))
        self.assertIsNone(aligned)
        self.assertFalse(status["accepted"])

    def test_gray_current_frame_fails_without_reusing_old_transform(self):
        self.assert_transform(dx=4, dy=2)
        aligned, status = self.aligner.update(np.full_like(self.reference, 128))
        self.assertIsNone(aligned)
        self.assertFalse(status["accepted"])
        self.assertIsNone(status["current_to_reference"])
        self.assertIsNone(self.aligner.valid_mask)
        self.assertTrue(self.aligner.update(self.reference)[1]["accepted"])

    def test_moderate_global_exposure_changes_are_accepted(self):
        for delta in (-20, 20):
            exposed = np.clip(self.reference.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            current, _ = shifted(exposed, dx=3, dy=-2)
            aligned, status = self.aligner.update(current)
            self.assertIsNotNone(aligned, status)
            self.assertTrue(status["accepted"], status)

    def test_distributed_matching_patches_do_not_hide_changed_static_scene(self):
        # A few surviving landmarks must not authorize an otherwise changed
        # display. Four original corner patches still fit identity perfectly.
        current = textured_scene(seed=1212)
        for y1, y2 in ((30, 125), (235, 330)):
            for x1, x2 in ((35, 145), (495, 605)):
                current[y1:y2, x1:x2] = self.reference[y1:y2, x1:x2]
        aligned, status = self.aligner.update(current)
        self.assertIsNone(aligned)
        self.assertFalse(status["accepted"])
        self.assertIn("Static scene appearance changed", status["reason"], status)

    def test_blank_and_repetitive_ambiguous_reference_not_trusted(self):
        blank = np.full_like(self.reference, 100)
        aligner = SceneAligner(blank)
        self.assertFalse(aligner.reference_ready)
        self.assertIsNone(aligner.update(blank)[0])
        # Repeated fine tiles provide indistinguishable ORB neighborhoods.
        tile = np.zeros((16, 16, 3), np.uint8)
        tile[:8, :8] = tile[8:, 8:] = 255
        repetitive = np.tile(tile, (23, 40, 1))[:360]
        aligner = SceneAligner(repetitive)
        result, status = aligner.update(shifted(repetitive, dx=8)[0])
        self.assertIsNone(result)
        self.assertFalse(status["accepted"])

    def test_local_moving_patch_cannot_define_scene_transform(self):
        blank = np.full_like(self.reference, 40)
        blank[100:180, 230:360] = self.reference[100:180, 230:360]
        aligner = SceneAligner(blank)
        self.assertFalse(aligner.reference_ready, aligner.status())

    @staticmethod
    def anchor_points(centers):
        offsets = np.float32([[-3,-2],[0,-2],[3,-2],[-3,0],[3,0],[-3,2],[0,2],[3,2]])
        return np.concatenate([offsets + np.float32(center) for center in centers]).astype(np.float32)

    def test_three_supported_noncollinear_anchor_areas_are_valid(self):
        points = self.anchor_points([(100, 70), (580, 70), (335, 305)])
        distribution = self.aligner._distribution(points)
        self.assertEqual(distribution["occupied_cells"], 3)
        self.assertEqual(distribution["supported_cells"], 3)
        self.assertTrue(self.aligner._distributed(distribution))

    def test_collinear_anchor_areas_fail_hull_support(self):
        points = self.anchor_points([(100, 70), (340, 185), (580, 300)])
        distribution = self.aligner._distribution(points)
        self.assertEqual(distribution["supported_cells"], 3)
        self.assertFalse(self.aligner._distributed(distribution))

    def test_supported_triangle_crossing_grid_boundary_remains_valid(self):
        # The third anchor straddles four bins, leaving only one feature per
        # bin. Spatial evidence is unchanged by that arbitrary bin boundary.
        x1, y1, x2, y2 = self.aligner._bounds
        origin, extent = np.float32([x1, y1]), np.float32([x2-x1, y2-y1])
        offsets = np.float32([[-3,-2],[3,-2],[-3,2],[3,2]])
        points = np.concatenate([origin + extent * center + offsets
                                 for center in [( .08, .08), (.92, .92), (1/3, 2/3)]]).astype(np.float32)
        distribution = self.aligner._distribution(points)
        self.assertEqual(distribution["supported_cells"], 2)
        self.assertEqual(distribution["supported_points"], 12)
        self.assertTrue(self.aligner._distributed(distribution))
        for delta in (-5., 5.):
            moved = self.aligner._distribution(points + np.float32([delta, delta]))
            self.assertTrue(self.aligner._distributed(moved))
            self.assertAlmostEqual(moved["supported_hull_fraction"],
                                   distribution["supported_hull_fraction"], places=4)

    def test_insufficient_local_neighbors_cannot_extend_supported_hull(self):
        points = np.concatenate([self.anchor_points([(100, 70), (580, 300)]),
                                 np.float32([[578, 70], [580, 70], [582, 70]])])
        distribution = self.aligner._distribution(points)
        self.assertGreater(distribution["hull_fraction"], .2)
        self.assertEqual(distribution["supported_points"], 16)
        self.assertFalse(self.aligner._distributed(distribution))

    def test_singleton_outliers_cannot_supply_missing_anchor_area(self):
        points = np.concatenate([self.anchor_points([(100, 70), (580, 300)]),
                                 np.float32([[580, 70], [100, 300]])])
        distribution = self.aligner._distribution(points)
        self.assertEqual(distribution["occupied_cells"], 4)
        self.assertEqual(distribution["supported_cells"], 2)
        self.assertGreater(distribution["hull_fraction"], .5)
        self.assertFalse(self.aligner._distributed(distribution))

    def test_excluded_moving_zone_does_not_bias_alignment_or_erase_object(self):
        mask = build_feature_mask(self.reference.shape, exclude_rects=[(.35, .3, .25, .3)])
        aligner = SceneAligner(self.reference, mask)
        self.assertTrue(aligner.reference_ready, aligner.status())
        changed = self.reference.copy()
        changed[135:175, 265:305] = [255, 0, 0]
        current, _ = shifted(changed, dx=6, dy=3)
        aligned, status = aligner.update(current)
        self.assertTrue(status["accepted"], status)
        self.assertGreater(np.mean(aligned[140:170, 270:300, 0]), 250)
        self.assertLess(np.mean(aligned[140:170, 270:300, 1]), 5)

    def test_required_region_cropped_by_shift_is_rejected(self):
        required = np.zeros(self.reference.shape[:2], np.uint8)
        required[100:200, -12:-2] = 255
        aligner = SceneAligner(self.reference, self.mask, required_mask=required)
        aligned, status = aligner.update(shifted(self.reference, dx=15)[0])
        self.assertIsNone(aligned)
        self.assertIn("no longer fully visible", status["reason"])

    def test_reset_dimension_change_and_recalibration_are_explicit(self):
        self.assertIsNone(self.aligner.update(self.reference[:-1])[0])
        self.aligner.reset()
        self.assertFalse(self.aligner.reference_ready)
        self.assertIsNone(self.aligner.update(self.reference)[0])
        self.aligner.calibrate(self.reference, self.mask)
        self.assertTrue(self.aligner.update(self.reference)[1]["accepted"])
        self.aligner.calibrate(np.zeros_like(self.reference))
        self.assertFalse(self.aligner.reference_ready)

    def test_held_transform_rewarps_and_still_requires_alarm_regions(self):
        current, _ = shifted(self.reference, 6., -4.)
        aligned, status = self.aligner.update(current)
        self.assertTrue(status["accepted"], status)
        held = self.aligner.warp(current, status["current_to_reference"])
        # warpAffine quantizes subpixel positions, so the reported (rounded)
        # matrix may move a few edge pixels by one interpolation step.
        difference = np.abs(held.astype(int) - aligned.astype(int))
        self.assertLessEqual(np.percentile(difference, 99.9), 2)
        self.assertLess(difference.mean(), .05)
        self.assertIsNotNone(self.aligner.valid_mask)
        self.assertIsNone(self.aligner.warp(current, [[1, 0], [0, 1]]))
        self.assertIsNone(self.aligner.warp(current[:100], status["current_to_reference"]))
        self.assertIsNone(self.aligner.warp(current, [[1, 0, float("nan")], [0, 1, 0]]))
        required = np.zeros(self.reference.shape[:2], bool)
        required[:, :4] = True
        strict = SceneAligner(self.reference, self.mask, required_mask=required)
        self.assertIsNone(strict.warp(current, [[1, 0, 25.], [0, 1, 0]]))
        self.assertIsNone(strict.valid_mask)
        self.assertIsNone(SceneAligner().warp(current, [[1, 0, 0], [0, 1, 0]]))

    def test_weak_evidence_confirms_prior_but_never_moves_zones(self):
        aligner = SceneAligner(self.reference, self.mask)
        aligner.config = dataclasses.replace(AlignmentConfig(), min_hull_fraction=.99)  # Spread gate now fails.
        current, _ = shifted(self.reference, 1.5, 0.)
        self.assertIsNone(aligner.update(current)[0], "without a prior the strict rule is unchanged")
        prior = [[1., 0., 0.], [0., 1., 0.]]
        aligned, status = aligner.update(current, prior=prior)
        self.assertTrue(status["accepted"], status)
        self.assertTrue(status["weak"])
        self.assertEqual(status["current_to_reference"], prior)
        self.assertLess(status["pose_gap_px"], 2.5)
        np.testing.assert_array_equal(aligned, current)   # Warped with the prior, not the estimate.
        far = [[1., 0., -12.], [0., 1., 0.]]
        self.assertIsNone(aligner.update(current, prior=far)[0])
        self.assertIn("disagree", aligner.status()["reason"])
        for bad in ([[1, 0], [0, 1]], [[1, 0, float("nan")], [0, 1, 0]], "junk"):
            self.assertIsNone(aligner.update(current, prior=bad)[0])

    def test_lighting_change_is_confirmed_but_a_different_scene_is_not(self):
        aligner = SceneAligner(self.reference, self.mask)
        aligner.config = dataclasses.replace(AlignmentConfig(), min_good_tile_fraction=1.01)  # Strict gate fails.
        prior = [[1., 0., 0.], [0., 1., 0.]]
        darker = np.clip(self.reference.astype(np.float32) * .6 + 10, 0, 255).astype(np.uint8)
        aligned, status = aligner.update(darker, prior=prior)
        self.assertTrue(status["accepted"], status)
        self.assertTrue(status["weak"])
        for seed in (77, 5150):
            other = textured_scene(seed=seed)
            result, status = SceneAligner(self.reference, self.mask).update(other, prior=prior)
            self.assertIsNone(result, status)

    def test_feature_mask_excludes_clocks_motion_and_border(self):
        mask = build_feature_mask((100, 200, 3), include_rects=[(.1, .1, .8, .8)],
                                  exclude_rects=[(.4, .4, .2, .2)], border_px=5, exclusion_padding_px=2)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[20, 30], 255)
        self.assertEqual(mask[50, 100], 0)
        self.assertEqual(mask[1, 1], 0)
        with self.assertRaises(ValueError):
            build_feature_mask((100, 200), exclude_rects=[(.8, .8, .5, .5)])

    @unittest.skipUnless(os.environ.get("C500_ALIGNMENT_REFERENCE"), "Optional private reference fixture is not bundled")
    def test_actual_monitor_foreground_occlusion_and_exposure(self):
        reference_bgr = cv2.imread(os.environ["C500_ALIGNMENT_REFERENCE"])
        self.assertIsNotNone(reference_bgr, "Reference fixture could not be read")
        reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        height, width = reference.shape[:2]
        zones = [(.64, .17, .17, .25), (.47, .35, .30, .28), (.23, .31, .24, .18)]
        mask = build_feature_mask(reference.shape, include_rects=[(.12, .10, .79, .82)],
                                  exclude_rects=zones + [(0, 0, 1, .14), (.68, .7, .22, .2)])
        required = build_feature_mask(reference.shape, include_rects=zones, border_px=0)
        aligner = SceneAligner(reference, mask, required_mask=required)
        self.assertTrue(aligner.reference_ready, aligner.status())
        for delta in (-20, 20):
            exposed = np.clip(reference.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            self.assertTrue(aligner.update(shifted(exposed, dx=4, dy=-3)[0])[1]["accepted"])
        for rect, expected in [((.2, .55, .3, .25), True), ((.15, .5, .55, .3), False)]:
            x, y, rw, rh = rect
            x1, y1, x2, y2 = int(x * width), int(y * height), int((x + rw) * width), int((y + rh) * height)
            occluded = reference.copy()
            occluded[y1:y2, x1:x2] = [42, 48, 56]
            cv2.rectangle(occluded, (x1 + 7, y1 + 6), (x2 - 7, y1 + (y2 - y1) // 3), (95, 115, 130), -1)
            aligned, status = aligner.update(shifted(occluded, dx=5, dy=-3)[0])
            self.assertEqual(status["accepted"], expected, status)
            if expected:
                np.testing.assert_allclose(status["translation_px"], [-5, 3], atol=.5)
                # Alignment must retain the foreground object, not reconstruct
                # it away using the clear reference image underneath it.
                np.testing.assert_allclose(aligned[y2 - 12, x1 + 12], [42, 48, 56], atol=3)


if __name__ == "__main__":
    unittest.main()
