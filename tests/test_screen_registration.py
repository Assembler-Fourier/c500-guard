"""Finding the saved scene again after the camera is physically moved."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from screen_registration import coverage, estimate_transform, movement_px, warp, registration_mask  # noqa: E402
from test_scene_alignment import textured_scene  # noqa: E402


def moved_view(scene, corners):
    """Simulate the same scene filmed from a different camera position."""
    height, width = scene.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    forward = cv2.getPerspectiveTransform(source, np.float32(corners))
    return cv2.warpPerspective(scene, forward, (width, height)), forward


class ReacquireTests(unittest.TestCase):
    def setUp(self):
        self.reference = textured_scene()
        self.height, self.width = self.reference.shape[:2]

    def zone_mask(self, rect=(.3, .3, .3, .3)):
        mask = np.zeros((self.height, self.width), np.uint8)
        x, y, w, h = rect
        mask[int(y * self.height):int((y + h) * self.height),
             int(x * self.width):int((x + w) * self.width)] = 255
        return mask

    def test_moved_camera_is_found_and_zone_points_land_back(self):
        cases = {
            "shifted and turned": [[40, 18], [self.width - 25, 52],
                                   [self.width - 60, self.height - 12], [12, self.height - 40]],
            "closer and angled": [[95, 60], [self.width - 70, 30],
                                  [self.width - 40, self.height - 70], [60, self.height - 35]],
        }
        for name, corners in cases.items():
            with self.subTest(move=name):
                moved, forward = moved_view(self.reference, corners)
                matrix, report = estimate_transform(moved, self.reference, required_mask=self.zone_mask())
                self.assertIsNotNone(matrix, report)
                probes = np.float32([[200, 120], [320, 200], [420, 150]]).reshape(-1, 1, 2)
                seen_at = cv2.perspectiveTransform(probes, forward)
                back = cv2.perspectiveTransform(seen_at, matrix).reshape(-1, 2)
                self.assertLess(float(np.linalg.norm(back - probes.reshape(-1, 2), axis=1).max()), 3.)
                self.assertGreater(movement_px(matrix, self.reference.shape[:2]), 5)
                self.assertGreaterEqual(report["zone_coverage"], 1.)

    def test_a_different_scene_is_refused(self):
        matrix, report = estimate_transform(textured_scene(seed=31), self.reference)
        self.assertIsNone(matrix)
        self.assertTrue(report["reason"])

    def test_a_zone_outside_the_new_view_is_refused(self):
        shift = 120          # the camera now misses the right edge of the scene
        moved, _ = moved_view(self.reference, [[-shift, 0], [self.width - 1 - shift, 0],
                                               [self.width - 1 - shift, self.height - 1], [-shift, self.height - 1]])
        matrix, report = estimate_transform(moved, self.reference,   # left edge now unseen
                                            required_mask=self.zone_mask((.02, .3, .12, .3)))
        self.assertIsNone(matrix)
        self.assertIn("outside", report["reason"])
        still_filmed, _ = estimate_transform(moved, self.reference, required_mask=self.zone_mask())
        self.assertIsNotNone(still_filmed, "a zone that is still filmed must be accepted")

    def test_blank_view_is_refused_without_raising(self):
        matrix, report = estimate_transform(np.zeros_like(self.reference), self.reference)
        self.assertIsNone(matrix)
        self.assertTrue(report["reason"])

    def test_many_matches_on_one_small_patch_cannot_certify_a_new_pose(self):
        # A repeated object or surviving screen corner must not position every
        # alarm zone, even when it contains hundreds of matching descriptors.
        current = textured_scene(seed=31)
        current[110:240, 230:400] = self.reference[110:240, 230:400]
        matrix, report = estimate_transform(current, self.reference)
        self.assertIsNone(matrix, report)
        self.assertIn("concentrated", report["reason"])

    def test_registration_mask_excludes_other_planes_and_timestamp(self):
        mask = registration_mask(self.reference.shape,
            polygon=[[.15, .1], [.85, .1], [.85, .9], [.15, .9]],
            exclude_rects=[[.2, .15, .2, .1]])
        self.assertEqual(mask[180, 30], 0)
        self.assertEqual(mask[70, 190], 0)
        self.assertEqual(mask[180, 320], 255)

    def test_even_one_unseen_required_pixel_is_not_rounded_to_full_coverage(self):
        required = np.ones((self.height, self.width), np.uint8)
        matrix = np.eye(3)
        observed = coverage(matrix, self.reference.shape[:2], required)
        self.assertLess(observed, 1.)

    def test_masked_plane_can_move_while_surroundings_change(self):
        mask = registration_mask(self.reference.shape,
            polygon=[[.12, .1], [.88, .1], [.88, .9], [.12, .9]])
        current, forward = moved_view(self.reference,
            [[25, 10], [self.width - 15, 25], [self.width - 35, self.height - 5], [10, self.height - 15]])
        visible = cv2.warpPerspective(mask, forward, (self.width, self.height)) != 0
        current[~visible] = textured_scene(seed=31)[~visible]
        matrix, report = estimate_transform(current, self.reference,
            required_mask=self.zone_mask(), reference_mask=mask)
        self.assertIsNotNone(matrix, report)

    def test_warp_and_coverage_report_the_unseen_part(self):
        corners = [[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]]
        moved, _ = moved_view(self.reference, corners)
        matrix, _ = estimate_transform(moved, self.reference)
        self.assertIsNotNone(matrix)
        self.assertGreater(coverage(matrix, self.reference.shape[:2]), .95)
        restored = warp(moved, matrix, self.reference.shape[:2])
        self.assertEqual(restored.shape, self.reference.shape)


if __name__ == "__main__":
    unittest.main()
