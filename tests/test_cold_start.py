"""Start-up in unseen lighting, and lighting references that stay diverse."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from test_watch import watch, watcher
from test_scene_alignment import textured_scene
from scene_alignment import structural_match

H, W = watch.PROC_H, watch.PROC_W


def relit(scene, gamma=.45, gain=1.15):
    """Very different lighting: brighter, compressed, with a light falloff across the view."""
    ramp = np.linspace(.6, 1.2, W, dtype=np.float32)[None, :, None]
    out = ((scene.astype(np.float32) / 255.) ** gamma) * 255. * gain * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def moved(scene, dx, dy):
    return cv2.warpAffine(scene, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)


class StructuralMatchTests(unittest.TestCase):
    def test_same_scene_matches_under_very_different_light_and_reports_shift(self):
        scene = textured_scene(seed=31)
        same = structural_match(relit(scene), scene)
        self.assertGreater(same['response'], .25)
        self.assertLess(np.hypot(same['dx'], same['dy']), .5)
        shifted = structural_match(moved(relit(scene), 4, -3), scene)
        self.assertAlmostEqual(shifted['dx'], 4, delta=.5)
        self.assertAlmostEqual(shifted['dy'], -3, delta=.5)

    def test_other_views_do_not_match(self):
        scene = textured_scene(seed=31)
        for name, other in (('flipped', cv2.flip(scene, 1)), ('other scene', textured_scene(seed=99)),
                            ('black', np.zeros_like(scene)),
                            ('noise', np.random.default_rng(1).integers(0, 255, scene.shape, dtype=np.uint8))):
            with self.subTest(name):
                self.assertLess(structural_match(other, scene)['response'], .25)
        self.assertEqual(0., structural_match(scene[:100], scene)['response'])


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = mock.Mock()
        self.scene = textured_scene(seed=4242)
        self.main = mock.Mock()
        self.main.reference = self.scene
        self.main.update.return_value = (None, {'reason': 'Too few unambiguous mutual landmark matches',
                                                'reference_ready': True})
        self.main.warp.side_effect = lambda frame, matrix: frame
        self.obj.aligner = self.main
        self.obj._alignment_reference = self.scene
        self.obj._alignment_pose = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.folder = tempfile.TemporaryDirectory()
        base = Path(self.folder.name)
        for name, path in (('_extra_path', base / 'lighting.npz'), ('_reacquire_path', base / 'r.json'),
                           ('_pose_path', base / 'pose.json')):
            mock.patch.object(self.obj, name, return_value=path).start()
        mock.patch.object(watch, 'BOOTSTRAP_AFTER_S', 0.).start()
        mock.patch.object(watch, 'REACQUIRE_FIRST_S', 1e9).start()   # keep the moved-camera search out of it

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()
        if self.obj.semantic:
            self.obj.semantic.stop()

    def run_frames(self, frame, count):
        results = []
        for _ in range(count):
            results.append(self.obj.align_frame(frame))
        return results

    def test_unseen_lighting_at_start_learns_a_reference_and_locks(self):
        frame = relit(self.scene)
        results = self.run_frames(frame, 7)
        self.assertIsNone(results[0], 'nothing may be watched before the check has passed')
        self.assertEqual(1, len(self.obj._extra_aligners))
        self.assertTrue(Path(self.folder.name, 'lighting.npz').is_file())
        self.assertIsNotNone(results[-1])
        self.assertEqual('tracking', self.obj.alignment['state'])
        self.assertEqual(1, self.obj.alignment['reference_index'])
        self.assertGreater(self.obj.alignment['bootstrap']['response'], .25)
        logged = [call.args[1] for call in self.obj.log_event.call_args_list if call.args[0] == 'alignment']
        self.assertTrue(any('structure check matched' in line for line in logged))

    def test_a_different_or_moved_view_is_never_learned(self):
        for name, frame, reason in (('other scene', textured_scene(seed=7), 'does not match'),
                                    ('moved 6 px', moved(relit(self.scene), 6, 0), 'shifted')):
            with self.subTest(name):
                self.obj._last_bootstrap_try, self.obj._learning = 0., None
                results = self.run_frames(frame, 4)
                self.assertTrue(all(result is None for result in results))
                self.assertIsNone(self.obj._learning)
                self.assertEqual([], self.obj._extra_aligners)
                self.assertIn(reason, self.obj.alignment['bootstrap']['reason'])
                self.assertEqual('calibration_required', self.obj.alignment['state'])

    def test_not_used_once_alignment_has_worked_this_run(self):
        self.obj._alignment_used = ([[1., 0., 0.], [0., 1., 0.]], 0.)
        self.obj.alignment_hold_s = 0.
        with mock.patch('scene_alignment.structural_match') as check:
            self.run_frames(relit(self.scene), 3)
        check.assert_not_called()


class DiverseReferenceTests(unittest.TestCase):
    def test_full_set_replaces_the_most_similar_reference_not_the_oldest(self):
        obj = watcher()
        base = textured_scene(seed=5)
        levels = [.3, .5, .7, .9, 1.1, 1.3]
        stored = [mock.Mock(reference=np.clip(base * level, 0, 255).astype(np.uint8)) for level in levels]
        obj._extra_aligners = list(stored)
        obj._extra_used = [0., 100., 100., 100., 100., 100.]   # the evening one (slot 0) is the least used
        candidate = mock.Mock(reference=np.clip(base * .92, 0, 255).astype(np.uint8))
        with mock.patch.object(obj, '_save_lighting_references'):
            obj._commit_lighting_reference(candidate, 200.)
        self.assertIs(obj._extra_aligners[3], candidate)
        self.assertIs(obj._extra_aligners[0], stored[0], 'the rarely used, different reference survives')
        self.assertEqual(6, watch.MAX_LIGHTING_REFERENCES)


if __name__ == '__main__':
    unittest.main()
