"""Night alignment cost: shared frame features and a once-a-second probe while weakly confirmed (21 Nov 2026)."""
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from test_watch import watch, watcher
from test_scene_alignment import shifted, textured_scene
import scene_alignment
from scene_alignment import SceneAligner


def fingerprint(result):
    aligned, info = result
    info = {k: v for k, v in info.items() if k != "processing_ms"}
    return (None if aligned is None else aligned.tobytes(), repr(sorted(info.items())))


class SharedFeatureTests(unittest.TestCase):
    def setUp(self):
        scene_alignment._FEATURE_CACHE[0] = None
        self.reference = textured_scene(seed=77)
        self.frame, _ = shifted(self.reference, dx=3., dy=-2.)

    def test_results_are_identical_to_computing_features_again(self):
        a, b = SceneAligner(self.reference), SceneAligner(self.reference)
        shared = [fingerprint(a.update(self.frame)), fingerprint(b.update(self.frame))]
        scene_alignment._FEATURE_CACHE[0] = None
        alone = fingerprint(SceneAligner(self.reference).update(self.frame))
        self.assertEqual(shared[0], alone)
        self.assertEqual(shared[1], alone)
        self.assertIsNotNone(a.update(self.frame)[0])

    def test_features_are_computed_once_per_frame_and_mask(self):
        a, b = SceneAligner(self.reference), SceneAligner(self.reference)
        calls = []
        real = scene_alignment._shared_frame_features

        def spy(orb, orb_key, gray, mask, mask_key):
            before = scene_alignment._FEATURE_CACHE[0]
            out = real(orb, orb_key, gray, mask, mask_key)
            calls.append(scene_alignment._FEATURE_CACHE[0] is not before)
            return out
        with mock.patch.object(scene_alignment, "_shared_frame_features", spy):
            a.update(self.frame)
            b.update(self.frame)                      # same frame, same mask: reused
            other, _ = shifted(self.reference, dx=-4.)
            b.update(other)                           # a new frame: computed
        self.assertEqual(calls, [True, False, True])

    def test_a_different_mask_never_reuses_features(self):
        a = SceneAligner(self.reference)
        mask = np.zeros(self.reference.shape[:2], np.uint8); mask[:, :400] = 255
        b = SceneAligner(self.reference, feature_mask=mask)
        a.update(self.frame)
        key_a = scene_alignment._FEATURE_CACHE[0][0]
        b.update(self.frame)
        self.assertNotEqual(key_a, scene_alignment._FEATURE_CACHE[0][0])


class WeakProbeTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.frame = np.zeros((360, 640, 3), np.uint8)
        self.now = 100.
        clock = mock.patch.object(watch.time, "monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.main, self.extra = mock.Mock(), mock.Mock()
        self.extra.update.return_value = (self.frame, {"weak": True, "accepted": True})
        self.obj.aligner, self.obj._extra_aligners, self.obj._extra_used = self.main, [self.extra], [0.]
        self.obj._preferred_aligner = 0

    def run_frames(self, count):
        for _ in range(count):
            self.obj._update_alignment(self.frame, None)
            self.now += .1

    def test_weakly_confirmed_preferred_reference_probes_once_a_second(self):
        self.main.update.return_value = (self.frame, {"weak": True, "accepted": True})
        self.run_frames(30)                                       # three seconds at 10 fps
        self.assertEqual(self.main.update.call_count, 30)
        self.assertEqual(self.extra.update.call_count, 3)

    def test_a_stronger_reference_found_by_the_probe_becomes_preferred(self):
        self.main.update.return_value = (self.frame, {"weak": True, "accepted": True})
        self.extra.update.return_value = (self.frame, {"weak": False, "accepted": True})
        self.run_frames(1)
        self.assertEqual(self.obj._preferred_aligner, 1)

    def test_rejected_preferred_reference_still_probes_every_frame(self):
        self.main.update.return_value = (None, {"reason": "Scene changed"})
        self.extra.update.return_value = (None, {"reason": "Scene changed"})
        self.run_frames(10)
        self.assertEqual(self.extra.update.call_count, 10)

    def test_strong_preferred_reference_needs_no_probe(self):
        self.main.update.return_value = (self.frame, {"weak": False, "accepted": True})
        self.run_frames(10)
        self.extra.update.assert_not_called()

    def test_every_lighting_reference_is_probed_at_regular_camera_cadences(self):
        # The throttle must not alias with the number of saved references.
        # Previously, seven references at 10 fps visited only indices 1,3,5;
        # three references at any even frame rate never visited index 2.
        for count in (3, 5, 7):
            for fps in (6, 8, 10, 12):
                with self.subTest(references=count, fps=fps):
                    self.obj = watcher()
                    aligners = [mock.Mock() for _ in range(count)]
                    for aligner in aligners:
                        aligner.update.return_value = (self.frame, {"weak": True, "accepted": True})
                    self.obj.aligner = aligners[0]
                    self.obj._extra_aligners = aligners[1:]
                    self.obj._extra_used = [0.] * (count - 1)
                    self.obj._preferred_aligner = self.obj._probe_index = 0
                    for index in range(8 * fps):
                        self.now = 100. + index / fps
                        self.obj._update_alignment(self.frame, None)
                    calls = [aligner.update.call_count for aligner in aligners[1:]]
                    self.assertTrue(all(calls), calls)
                    self.assertEqual(sum(calls), 8)
                    self.assertLessEqual(max(calls) - min(calls), 1)

    def test_stronger_reference_previously_starved_by_cadence_is_recovered(self):
        self.main.update.return_value = (self.frame, {"weak": True, "accepted": True})
        strong = mock.Mock()
        strong.update.return_value = (self.frame, {"weak": False, "accepted": True})
        self.obj._extra_aligners.append(strong)
        self.obj._extra_used.append(0.)
        for index in range(21):
            self.now = 100. + index / 10.
            result = self.obj._update_alignment(self.frame, None)
        self.assertEqual(self.obj._preferred_aligner, 2)
        self.assertEqual(result[1]["reference_index"], 2)
        self.assertFalse(result[1]["weak"])

    def test_rejected_preferred_rotates_without_waiting_for_weak_probe_timer(self):
        self.main.update.return_value = (None, {"reason": "Scene changed"})
        second = mock.Mock()
        second.update.return_value = self.extra.update.return_value = (None, {"reason": "Scene changed"})
        self.obj._extra_aligners.append(second)
        self.obj._extra_used.append(0.)
        self.obj._weak_probe_at = self.now
        self.run_frames(6)
        self.assertEqual(self.extra.update.call_count, 3)
        self.assertEqual(second.update.call_count, 3)


if __name__ == "__main__":
    unittest.main()
