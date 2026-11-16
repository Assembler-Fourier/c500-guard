"""Watcher must fail visibly without using old zones or partially saved references."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from test_watch import watch, watcher


class AlignmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = mock.Mock()
        self.frame = watch.np.zeros((watch.PROC_H, watch.PROC_W, 3), watch.np.uint8)

    def tearDown(self):
        if self.obj.semantic:
            self.obj.semantic.stop()

    def test_rejected_match_with_null_translation_is_visible_and_not_a_crash(self):
        self.obj.aligner = mock.Mock()
        self.obj.aligner.update.return_value = (None, {'translation_px':None, 'reason':'Scene changed', 'reference_ready':True})
        self.assertIsNone(self.obj.align_frame(self.frame))
        self.assertEqual(self.obj.alignment['state'], 'calibration_required')
        self.assertEqual(self.obj.alignment['dx'], 0)
        self.assertTrue(self.obj._alignment_bad_since)

    def accepted(self):
        return (self.frame, {'translation_px': [0., 0.], 'reference_ready': True,
                             'current_to_reference': [[1., 0., 0.], [0., 1., 0.]]})

    def test_brief_rejection_holds_verified_transform_without_pause_reset_or_fault(self):
        aligner = mock.Mock()
        aligner.update.side_effect = [self.accepted(),
                                      (None, {'reason': 'Inliers are concentrated', 'reference_ready': True}),
                                      self.accepted()]
        aligner.warp.return_value = self.frame
        self.obj.aligner = aligner
        self.assertIsNotNone(self.obj.align_frame(self.frame))
        self.obj._pending_rearm = False
        self.assertIs(self.obj.align_frame(self.frame), self.frame)
        self.assertEqual(aligner.warp.call_args.args[1], [[1., 0., 0.], [0., 1., 0.]])
        self.assertEqual(self.obj.alignment['state'], 'holding')
        self.assertIn('Holding', self.obj.alignment['message'])
        self.assertFalse(self.obj._alignment_blocked)
        now = time.monotonic(); self.obj.first_frame_seen = True; self.obj.last_frame_at = now + 3
        self.obj.check_health(now + 3)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNotNone(self.obj.align_frame(self.frame))
        self.assertFalse(self.obj._pending_rearm, 'recovery after a hold must not reset detection')
        self.assertEqual(self.obj._alignment_bad_since, 0.)

    def test_rejection_longer_than_hold_pauses_scene_alerts_and_faults(self):
        aligner = mock.Mock()
        aligner.reference = None        # no moved-camera search with a fake aligner (hold can exceed 20 s)
        aligner.update.side_effect = [self.accepted()] + [(None, {'reason': 'Scene changed', 'reference_ready': True})] * 3
        aligner.warp.return_value = self.frame
        self.obj.aligner = aligner
        with mock.patch.object(watch.time, 'monotonic', return_value=100.) as clock:
            self.obj.align_frame(self.frame)
            self.assertIsNotNone(self.obj.align_frame(self.frame))
            clock.return_value = 100. + self.obj.alignment_hold_s + .1
            self.assertIsNone(self.obj.align_frame(self.frame))
            self.assertTrue(self.obj._alignment_blocked)
            self.assertEqual(self.obj.alignment['state'], 'calibration_required')
            self.assertIsNone(self.obj.align_frame(self.frame), 'an expired hold must not resume')
            self.obj.first_frame_seen = True; self.obj.last_frame_at = clock.return_value
            self.obj.check_health(clock.return_value)
        self.assertEqual(self.obj.alarm, 'fault')

    def test_no_verified_transform_means_no_hold(self):
        aligner = mock.Mock()
        aligner.update.return_value = (None, {'reason': 'Scene changed', 'reference_ready': True})
        self.obj.aligner = aligner
        self.obj._alignment_blocked = False
        self.assertIsNone(self.obj.align_frame(self.frame))
        aligner.warp.assert_not_called()

    def test_weak_confirmation_uses_verified_prior_and_keeps_it(self):
        moved = [[1., 0., 1.], [0., 1., 0.]]
        aligner = mock.Mock()
        aligner.update.side_effect = [
            (self.frame, {'translation_px': [1., 0.], 'reference_ready': True, 'current_to_reference': moved}),
            (self.frame, {'translation_px': [1., 0.], 'reference_ready': True, 'current_to_reference': moved,
                          'weak': True})]
        self.obj.aligner = aligner
        with mock.patch.object(self.obj, '_maybe_save_pose') as save:
            self.obj.align_frame(self.frame)
            self.assertIsNotNone(self.obj.align_frame(self.frame))
        self.assertEqual(aligner.update.call_args_list[0].kwargs['prior'], [[1., 0., 0.], [0., 1., 0.]])
        self.assertEqual(aligner.update.call_args_list[1].kwargs['prior'], moved)
        save.assert_called_once()   # Only the strong estimate is persisted.
        self.assertEqual(self.obj.alignment['state'], 'tracking')
        self.assertIn('Low light', self.obj.alignment['message'])

    def test_verified_pose_jump_relearns_background_but_small_drift_does_not(self):
        aligner = mock.Mock()
        self.obj.aligner = aligner
        self.obj._alignment_blocked = False
        with mock.patch.object(self.obj, '_maybe_save_pose'):
            for shift, expected in ((0.3, False), (1.5, False), (3.0, True)):
                self.obj._alignment_used = ([[1., 0., 0.], [0., 1., 0.]], 0.)
                self.obj._pending_rearm = False
                aligner.update.return_value = (self.frame, {
                    'reference_ready': True, 'translation_px': [shift, 0.],
                    'current_to_reference': [[1., 0., shift], [0., 1., 0.]]})
                self.obj.align_frame(self.frame)
                self.assertEqual(self.obj._pending_rearm, expected, shift)

    def test_verified_pose_survives_restart_and_bad_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(self.obj, '_pose_path', return_value=Path(folder) / 'pose.json'):
            self.obj._maybe_save_pose([[1., 0., 2.], [0., 1., -1.]], self.frame.shape)
            self.assertEqual(self.obj._read_pose().tolist(), [[1., 0., 2.], [0., 1., -1.]])
            self.obj._alignment_good = None
            self.obj._alignment_pose = self.obj._read_pose()
            self.assertEqual(self.obj._alignment_prior().tolist(), [[1., 0., 2.], [0., 1., -1.]])
            (Path(folder) / 'pose.json').write_text('{"current_to_reference": [[1, 0], [0, 1]]}')
            self.assertIsNone(self.obj._read_pose())

    def test_lighting_reference_is_learned_validated_persisted_and_preferred(self):
        from test_scene_alignment import textured_scene
        night = textured_scene(seed=4242)
        identity = [[1., 0., 0.], [0., 1., 0.]]
        main = mock.Mock()
        main.update.return_value = (night, {'reference_ready': True, 'translation_px': [0., 0.], 'weak': True,
                                            'current_to_reference': identity, 'pose_gap_px': .4,
                                            'static_agreement': {'median_correlation': .7}})
        self.obj.aligner = main
        self.obj._alignment_blocked = False
        self.obj._last_strong_at = self.obj._last_learn_at = -10000.
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(self.obj, '_extra_path', return_value=Path(folder) / 'lighting.npz'), \
                mock.patch.object(self.obj, '_maybe_save_pose'):
            for _ in range(6):   # One trigger frame, then five validation frames.
                self.assertIsNotNone(self.obj.align_frame(night))
            self.assertEqual(len(self.obj._extra_aligners), 1)
            self.assertEqual(self.obj._preferred_aligner, 1)
            self.assertTrue((Path(folder) / 'lighting.npz').is_file())
            self.obj.align_frame(night)
            self.assertEqual(self.obj.alignment['reference_index'], 1)
            self.assertEqual(self.obj.alignment['state'], 'tracking')
            self.assertIn('learned lighting', self.obj.alignment['message'])
            self.obj._extra_aligners, self.obj._extra_used = [], []
            self.obj._load_lighting_references()
            self.assertEqual(len(self.obj._extra_aligners), 1)

    def test_learning_needs_weak_consistent_evidence_and_no_recent_strong_match(self):
        frame = self.frame
        good = {'weak': True, 'pose_gap_px': .4, 'static_agreement': {'median_correlation': .7}}
        cases = [({**good, 'pose_gap_px': 2.}, -1e4), ({**good, 'static_agreement': {'median_correlation': .55}}, -1e4),
                 ({**good, 'weak': False}, -1e4), (good, time.monotonic())]
        for info, strong_at in cases:
            with self.subTest(info=info), mock.patch.object(self.obj, '_build_aligner') as build:
                self.obj._learning, self.obj._last_learn_at, self.obj._last_strong_at = None, -1e4, strong_at
                self.obj._consider_learning(frame, frame, info, time.monotonic())
                build.assert_not_called()

    def test_explicit_lighting_capture_uses_verified_pose_and_needs_off(self):
        from test_scene_alignment import textured_scene
        night = textured_scene(seed=4242)
        main = mock.Mock()
        main.update.return_value = (night, {'translation_px': [0.,0.], 'current_to_reference': [[1.,0.,0.],[0.,1.,0.]],
                                           'reference_ready': True})
        main.warp.return_value = night
        self.obj.aligner = main
        self.obj.first_frame_seen, self.obj.last_frame_at = True, time.monotonic()
        with self.assertRaises(PermissionError):
            self.obj.request_lighting_reference()
        self.obj.armed = False
        self.obj.alignment = {'state': 'tracking'}
        self.obj._alignment_pose = watch.np.array([[1., 0., 0.], [0., 1., 0.]])
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(self.obj, '_extra_path', return_value=Path(folder) / 'lighting.npz'):
            self.obj.request_lighting_reference()
            self.assertIsNotNone(self.obj.align_frame(night))
            with self.assertRaises(ValueError):              # One check at a time.
                self.obj.request_lighting_reference()
            for _ in range(5):
                self.obj.align_frame(night)
            main.warp.assert_not_called()
            self.assertEqual(len(self.obj._extra_aligners), 1)
            self.assertIn('Saved a lighting reference', self.obj._lighting_status)
            self.assertIsNotNone(self.obj.align_frame(night))
            self.assertEqual(self.obj.alignment['reference_index'], 1)
            self.assertEqual(self.obj.alignment['state'], 'tracking')

    def test_lighting_capture_without_landmarks_reports_failure(self):
        main = mock.Mock()
        main.update.return_value = (None, {'reason': 'Scene changed', 'reference_ready': True})
        main.warp.return_value = self.frame   # Blank frame: no landmarks to learn.
        self.obj.aligner = main
        self.obj.armed, self.obj.first_frame_seen, self.obj.last_frame_at = False, True, time.monotonic()
        with self.assertRaisesRegex(ValueError, 'verified alignment'):
            self.obj.request_lighting_reference()
        # A request accepted just before the view moved is checked again at capture.
        self.obj._pending_lighting = True
        self.obj.align_frame(self.frame)
        self.assertIsNone(self.obj._learning)
        self.assertIn('no lighting reference was saved', self.obj._lighting_status)

    def test_lighting_save_failure_restores_previous_references(self):
        prior, candidate = mock.Mock(), mock.Mock()
        self.obj._extra_aligners, self.obj._extra_used, self.obj._preferred_aligner = [prior], [1.], 1
        with mock.patch.object(self.obj, '_save_lighting_references', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.obj._commit_lighting_reference(candidate, 2.)
        self.assertEqual(self.obj._extra_aligners, [prior])
        self.assertEqual(self.obj._extra_used, [1.])
        self.assertEqual(self.obj._preferred_aligner, 1)

    def test_moved_camera_is_re_registered_and_zones_keep_working(self):
        import cv2
        from scene_alignment import SceneAligner, build_feature_mask
        from test_scene_alignment import textured_scene
        scene = textured_scene(seed=77)
        height, width = scene.shape[:2]
        self.obj.aligner = SceneAligner(scene, build_feature_mask(scene.shape, border_px=15))
        self.obj._alignment_reference = scene
        self.obj._alignment_blocked = True
        moved = cv2.warpPerspective(scene, cv2.getPerspectiveTransform(
            watch.np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
            watch.np.float32([[35, 20], [width - 30, 45], [width - 55, height - 15], [15, height - 35]])),
            (width, height))
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(self.obj, '_reacquire_path', return_value=Path(folder) / 'r.json'):
            self.assertIsNone(self.obj.align_frame(moved), 'the moved view must not align yet')
            self.obj._alignment_bad_since = time.monotonic() - 60   # lost for a while
            self.obj._last_reacquire_try = 0.
            self.obj.align_frame(moved)                            # triggers re-registration
            self.assertIsNotNone(self.obj._reacquire, self.obj.alignment.get('reacquire'))
            self.assertTrue((Path(folder) / 'r.json').is_file())
            aligned = self.obj.align_frame(moved)
        self.assertIsNotNone(aligned)
        self.assertEqual(self.obj.alignment['state'], 'tracking')
        self.assertTrue(self.obj.alignment['reacquired'])
        self.assertGreater(self.obj.alignment['reacquire']['moved_px'], 5)

    def test_a_strange_view_is_not_re_registered(self):
        from scene_alignment import SceneAligner, build_feature_mask
        from test_scene_alignment import textured_scene
        scene = textured_scene(seed=77)
        self.obj.aligner = SceneAligner(scene, build_feature_mask(scene.shape, border_px=15))
        self.obj._alignment_reference = scene
        self.obj._alignment_blocked = True
        self.obj._alignment_bad_since = time.monotonic() - 60
        self.obj._last_reacquire_try = 0.
        self.assertIsNone(self.obj.align_frame(textured_scene(seed=404)))
        self.assertIsNone(self.obj._reacquire)
        self.assertTrue(self.obj.alignment['reacquire'].get('reason'))

    def test_inverse_affine_cannot_restore_pixels_cropped_by_remount(self):
        self.obj.zones = [watch.Zone(name='Right edge', rect=(.87, .4, .05, .1))]
        self.obj._reacquire = watch.np.array([[1., 0., 100.], [0., 1., 0.], [0., 0., 1.]])
        self.assertFalse(self.obj._reacquired_zones_visible(
            self.frame, [[1., 0., -100.], [0., 1., 0.]]),
            'The second warp cannot recover source pixels discarded by the first')

    def test_accepted_affine_cannot_hide_an_unseen_remounted_alarm_zone(self):
        self.obj.zones = [watch.Zone(name='Left edge', rect=(.1, .4, .05, .1))]
        self.obj._reacquire = watch.np.array([[1., 0., 150.], [0., 1., 0.], [0., 0., 1.]])
        self.obj._last_reacquire_try = time.monotonic()
        aligner = mock.Mock()
        aligner.update.return_value = self.accepted()
        aligner.warp.return_value = self.frame
        self.obj.aligner = aligner
        self.obj._alignment_blocked = False
        self.obj._alignment_used = ([[1., 0., 0.], [0., 1., 0.]], time.monotonic())
        self.assertIsNone(self.obj.align_frame(self.frame))
        self.assertTrue(self.obj._alignment_blocked)
        self.assertEqual(self.obj.alignment['state'], 'calibration_required')
        self.assertIn('outside', self.obj.alignment['reason'])
        aligner.warp.assert_not_called()

    def test_calibration_preserves_remount_and_next_raw_frame_alignment(self):
        import cv2
        import screen_registration
        from scene_alignment import SceneAligner, build_feature_mask
        from test_scene_alignment import textured_scene
        reference = textured_scene(seed=77)
        height, width = reference.shape[:2]
        forward = cv2.getPerspectiveTransform(
            watch.np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
            watch.np.float32([[35, 20], [width - 30, 45], [width - 55, height - 15], [15, height - 35]]))
        raw = cv2.warpPerspective(reference, forward, (width, height))
        self.obj.zones = [watch.Zone(name='Center', rect=(.35, .35, .1, .1))]
        self.obj.alignment_config = {'enabled': True, 'include_rects': [[0, 0, 1, 1]], 'border_px': 15}
        self.obj.aligner = SceneAligner(reference, self.obj._feature_mask(reference),
                                        required_mask=self.obj._required_mask(reference))
        self.obj._alignment_reference = reference.copy()
        self.obj._reacquire = watch.np.linalg.inv(forward)
        original_remount = self.obj._reacquire.copy()
        self.obj.armed = False
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(self.obj, '_alignment_path', return_value=Path(folder) / 'reference.npz'):
            self.obj._save_reacquire()
            before = self.obj.align_frame(raw)
            self.assertIsNotNone(before, self.obj.alignment)
            self.obj._pending_calibration = True
            self.assertIsNotNone(self.obj.align_frame(raw), self.obj.alignment)
            self.assertEqual(self.obj._calibration_error, '')
            watch.np.testing.assert_allclose(self.obj._reacquire, original_remount)
            self.assertTrue(self.obj._reacquire_path().is_file())
            self.assertTrue(self.obj._alignment_path().is_file())
            after = self.obj.align_frame(raw)
            self.assertIsNotNone(after, self.obj.alignment)
            self.assertEqual(self.obj.alignment['state'], 'tracking')
            monitored = self.obj._required_mask(raw) != 0
            self.assertLess(float(watch.np.abs(before.astype(float) - after)[monitored].mean()), 3.)

    def test_unverified_remount_cannot_certify_itself_during_calibration(self):
        self.obj.armed = False
        self.obj._pending_calibration = True
        self.obj._reacquire = watch.np.eye(3)
        self.obj._last_reacquire_try = time.monotonic()
        self.obj.aligner = mock.Mock()
        self.obj.aligner.update.return_value = (None, {'reason': 'Scene changed', 'reference_ready': True})
        previous = self.obj.aligner
        with mock.patch('zone_editor._atomic_write') as save:
            self.assertIsNone(self.obj.align_frame(self.frame))
        save.assert_not_called()
        self.assertIs(self.obj.aligner, previous)
        watch.np.testing.assert_array_equal(self.obj._reacquire, watch.np.eye(3))
        self.assertIn('Verify the recovered camera position', self.obj._calibration_error)

    def test_arming_waits_for_reference_operation(self):
        entered, release, armed = threading.Event(),threading.Event(),threading.Event()
        def operation(frame):
            entered.set(); self.assertTrue(release.wait(2)); return frame
        self.obj.armed=False
        with mock.patch.object(self.obj, '_align_frame_locked', side_effect=operation):
            worker=threading.Thread(target=self.obj.align_frame,args=(self.frame,));worker.start()
            self.assertTrue(entered.wait(2))
            switch=threading.Thread(target=lambda:(self.obj.set_armed(True),armed.set()));switch.start()
            self.assertFalse(armed.wait(.05))
            release.set();worker.join(2);switch.join(2)
            self.assertTrue(armed.is_set())

    def test_failed_reference_save_keeps_previous_aligner(self):
        old=mock.Mock(); old.update.return_value=(self.frame,{'translation_px':[0,0], 'reference_ready':True})
        self.obj.aligner=old;self.obj.armed=False;self.obj._pending_calibration=True
        candidate=mock.Mock();candidate.update.return_value=(self.frame,{'translation_px':[0,0],'reference_ready':True})
        with mock.patch.object(self.obj,'_feature_mask',return_value=None), \
             mock.patch.object(self.obj,'_required_mask',return_value=None), \
             mock.patch('scene_alignment.SceneAligner',return_value=candidate), \
             mock.patch('zone_editor._atomic_write',side_effect=OSError('Disk unavailable')):
            result=self.obj.align_frame(self.frame)
        self.assertIs(self.obj.aligner,old)
        self.assertIsNotNone(result)
        self.assertIn('Disk unavailable',self.obj.alignment['message'])

    def test_alignment_fault_does_not_repeat_after_ack_and_recovers(self):
        now=time.monotonic();self.obj.first_frame_seen=True;self.obj.last_frame_at=now
        self.obj._alignment_blocked=True;self.obj._alignment_bad_since=now-3;self.obj._alignment_used=([[1.,0.,0.],[0.,1.,0.]],now-10)
        self.obj.check_health(now)
        self.assertEqual(self.obj.alarm,'fault')
        self.obj.acknowledge();self.obj.check_health(now+.05)
        self.assertIsNone(self.obj.alarm)
        self.obj._alignment_blocked=False;self.obj.check_health(now+.06)
        self.assertEqual(self.obj.fault_reason,'')

    def test_feed_fault_resets_confirmation_once_per_episode(self):
        now=time.monotonic();self.obj.first_frame_seen=True;self.obj.last_frame_at=now-2
        self.obj.check_health(now);self.obj.check_health(now+.01)
        self.obj.semantic.reset.assert_called_once()


if __name__=='__main__': unittest.main()
