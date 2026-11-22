"""A bounded, revalidated pose hold must not hide actual camera displacement."""
import dataclasses
from unittest import mock
import unittest

import cv2
import numpy as np

from test_scene_alignment import textured_scene, shifted
from scene_alignment import SceneAligner, AlignmentConfig, build_feature_mask, pose_gap_px


IDENTITY = np.float64([[1., 0., 0.], [0., 1., 0.]])


class StableAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.reference = textured_scene()
        self.mask = build_feature_mask(self.reference.shape,
            exclude_rects=[(.32, .32, .34, .34)], border_px=15)
        self.aligner = SceneAligner(self.reference, self.mask)
        self.assertTrue(self.aligner.reference_ready)

    def test_stationary_scene_lighting_and_excluded_moving_vehicle_retain_pose(self):
        prior = IDENTITY.copy()
        for step in range(6):
            frame = np.clip(self.reference.astype(np.int16) + step * 3, 0, 255).astype(np.uint8)
            cv2.rectangle(frame, (230 + step * 10, 140), (265 + step * 10, 205), (250, 250, 250), -1)
            frame = cv2.imdecode(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1], 1)
            aligned, status = self.aligner.update(frame, prior=prior)
            self.assertIsNotNone(aligned, status)
            self.assertFalse(status.get('weak'), status)
            self.assertEqual('held', status['stabilization']['action'])
            self.assertFalse(status['stabilization']['supported_motion'])
            np.testing.assert_array_equal(status['current_to_reference'], prior)
            self.assertEqual([0., 0.], status['translation_px'])

    def test_actual_two_pixel_translation_fails_prior_fit_and_corrects_immediately(self):
        frame, forward = shifted(self.reference, dx=2.)
        aligned, status = self.aligner.update(frame, prior=IDENTITY)
        self.assertIsNotNone(aligned, status)
        self.assertEqual('accepted', status['stabilization']['action'])
        self.assertTrue(status['stabilization']['supported_motion'])
        self.assertIn('no longer fits', status['stabilization']['reason'])
        self.assertLess(pose_gap_px(status['current_to_reference'], cv2.invertAffineTransform(forward)), .7)

    def test_real_translation_rotation_and_zoom_still_follow_current_landmarks(self):
        for movement in ({'dx':7.,'dy':-4.}, {'angle':1.3}, {'scale':1.015}):
            with self.subTest(movement=movement):
                frame, forward = shifted(self.reference, **movement)
                aligned, status = self.aligner.update(frame, prior=IDENTITY)
                self.assertIsNotNone(aligned, status)
                self.assertEqual('accepted', status['stabilization']['action'])
                self.assertTrue(status['stabilization']['supported_motion'])
                self.assertLess(pose_gap_px(status['current_to_reference'], cv2.invertAffineTransform(forward)), .9)

    def test_incremental_drift_is_compared_with_applied_pose_not_last_estimate(self):
        prior = IDENTITY.copy()
        corrections = 0
        for dx in np.arange(.25, 5.01, .25):
            frame, forward = shifted(self.reference, dx=float(dx))
            aligned, status = self.aligner.update(frame, prior=prior)
            self.assertIsNotNone(aligned, status)
            corrections += status['stabilization']['action'] == 'accepted'
            prior = np.asarray(status['current_to_reference'])
            self.assertLess(pose_gap_px(prior, cv2.invertAffineTransform(forward)), 1.7)
        self.assertGreaterEqual(corrections, 2)
        self.assertLess(prior[0, 2], -3.5)

    def test_new_reference_in_same_coordinates_does_not_shift_applied_pose(self):
        bright = np.clip(self.reference.astype(np.int16)+15, 0, 255).astype(np.uint8)
        other = SceneAligner(bright, self.mask)
        for aligner in (self.aligner, other, self.aligner):
            _, status = aligner.update(bright, prior=IDENTITY)
            self.assertTrue(status['accepted'], status)
            self.assertEqual('held', status['stabilization']['action'])
            np.testing.assert_array_equal(status['current_to_reference'], IDENTITY)

    def test_prior_cannot_hide_an_alarm_pixel_outside_the_source(self):
        required = np.zeros(self.reference.shape[:2], bool)
        required[180, 1] = True
        aligner = SceneAligner(self.reference, self.mask, required_mask=required)
        prior = IDENTITY.copy(); prior[0, 2] = 1.
        _, status = aligner.update(self.reference, prior=prior)
        self.assertTrue(status['accepted'], status)
        self.assertEqual('accepted', status['stabilization']['action'])
        self.assertIn('visibility', status['stabilization']['reason'])
        self.assertEqual(255, aligner.valid_mask[180, 1])
        self.assertLess(pose_gap_px(status['current_to_reference'], IDENTITY), .01)

    def test_prior_must_repass_static_tile_agreement(self):
        good = dict(tiles=8, good_tiles=8, good_fraction=1., median_correlation=.99)
        bad = dict(tiles=8, good_tiles=2, good_fraction=.25, median_correlation=.4)
        with mock.patch.object(self.aligner, '_static_agreement', side_effect=[good, bad]):
            _, status = self.aligner.update(self.reference, prior=IDENTITY)
        self.assertTrue(status['accepted'], status)
        self.assertEqual('accepted', status['stabilization']['action'])
        self.assertIn('static scene', status['stabilization']['reason'])

    def test_invalid_old_similarity_is_not_held(self):
        prior = IDENTITY.copy(); prior[0, 1] = .002
        _, status = self.aligner.update(self.reference, prior=prior)
        self.assertTrue(status['accepted'], status)
        self.assertEqual('accepted', status['stabilization']['action'])
        self.assertIn('similarity', status['stabilization']['reason'])

    def test_wrong_scene_large_movement_and_missing_landmarks_are_not_blindly_held(self):
        for frame in (textured_scene(seed=999), shifted(self.reference, dx=45.)[0],
                      np.full_like(self.reference, 100)):
            aligned, status = self.aligner.update(frame, prior=IDENTITY)
            self.assertIsNone(aligned, status)
            self.assertFalse(status['accepted'])
            self.assertIsNone(self.aligner.valid_mask)

    def test_initial_pose_does_not_claim_motion_and_hold_metadata_matches_geometry(self):
        _, status = self.aligner.update(self.reference)
        self.assertEqual('accepted', status['stabilization']['action'])
        self.assertFalse(status['stabilization']['supported_motion'])
        self.assertIsNone(status['stabilization']['candidate_delta_px'])
        prior = np.float64([[1., 0., -.2], [0., 1., .1]])
        _, status = self.aligner.update(self.reference, prior=prior)
        self.assertEqual('held', status['stabilization']['action'])
        self.assertEqual([-.2, .1], status['translation_px'])
        self.assertEqual(1., status['scale'])
        self.assertEqual(0., status['rotation_deg'])
        self.assertAlmostEqual(.224, status['median_error_px'], places=3)
        self.assertIn('estimate_median_error_px', status)


if __name__ == '__main__':
    unittest.main()
