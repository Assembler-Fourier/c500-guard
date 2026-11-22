"""Reference-only work is cached without reusing current-frame evidence."""
import dataclasses
import unittest
from unittest import mock

from test_scene_alignment import textured_scene
from scene_alignment import SceneAligner, build_feature_mask, cv2


@unittest.skipIf(cv2 is None, 'OpenCV unavailable')
class AlignmentCacheTests(unittest.TestCase):
    def setUp(self):
        self.reference = textured_scene()
        self.mask = build_feature_mask(self.reference.shape)
        self.aligner = SceneAligner(self.reference, self.mask)
        self.assertTrue(self.aligner.reference_ready)

    def test_only_reference_work_is_reused_and_changed_scene_still_rejected(self):
        with mock.patch.object(cv2, 'dilate', wraps=cv2.dilate) as dilate:
            for _ in range(3):
                result, info = self.aligner.update(self.reference)
                self.assertIsNotNone(result)
                self.assertTrue(info['accepted'])
            self.assertEqual(1, dilate.call_count)
            result, info = self.aligner.update(self.reference[::-1].copy())
            self.assertIsNone(result)
            self.assertFalse(info['accepted'])
            self.assertIsNone(self.aligner.valid_mask)
            self.assertEqual(1, dilate.call_count)

    def test_new_search_limit_and_recalibration_invalidate_cache(self):
        with mock.patch.object(cv2, 'dilate', wraps=cv2.dilate) as dilate:
            self.aligner.update(self.reference)
            old_search = self.aligner._search_mask
            self.aligner.config = dataclasses.replace(self.aligner.config, max_corner_displacement_px=35)
            self.aligner.update(self.reference)
            self.assertEqual(2, dilate.call_count)
            self.assertIsNot(old_search, self.aligner._search_mask)
            old_blur = self.aligner._reference_blur
            self.aligner.calibrate(self.reference, self.mask)
            self.assertIsNot(old_blur, self.aligner._reference_blur)
            self.assertIsNone(self.aligner._search_mask)
            self.aligner.update(self.reference)
            self.assertEqual(3, dilate.call_count)


if __name__ == '__main__':
    unittest.main()
