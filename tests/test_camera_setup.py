import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from camera_setup import CameraSetupAssessment, zone_coverage, framing_guidance
from watch import Watcher


IDENTITY = np.eye(3)
ZONES = [{'name': 'Exit', 'polygon': [[.1, .2], [.45, .2], [.45, .9], [.1, .9]]},
         {'name': 'Footpath', 'polygon': [[.05, .1], [.4, .1], [.3, .3], [.05, .4]]}]


class CameraSetupTests(unittest.TestCase):
    def setUp(self):
        self.assessment = CameraSetupAssessment()

    def update(self, stamp, **kwargs):
        values = dict(matrix=IDENTITY, shape=(360, 640, 3), zones=ZONES,
                      alignment={'state': 'tracking', 'accepted': True},
                      stamp=stamp, view='PTZ 2', reference_id='saved-a')
        values.update(kwargs)
        return self.assessment.update(**values)

    def settle(self):
        for t in np.arange(10., 15.01, .25):
            result = self.update(float(t))
        return result

    def test_requires_real_time_and_fresh_picture(self):
        first = self.update(10.)
        self.assertFalse(first['ready_to_save'])
        result = self.settle()
        self.assertEqual(result['stable_s'], 5.)
        self.assertTrue(result['ready_to_save'])
        self.assertFalse(self.assessment.status(15.9, frame_age=.1)['ready_to_save'])
        self.assertIsNone(self.assessment.status(15.2, frame_age=2.)['score_pct'])

    def test_gap_resets_stability_but_only_geometry_invalidates_review(self):
        old = self.settle()
        gap = self.update(16.)
        self.assertFalse(gap['ready_to_save'])
        self.assertEqual(old['reference_id'], gap['reference_id'])
        new = self.update(16.25, reference_id='saved-b')
        self.assertNotEqual(new['reference_id'], gap['reference_id'])
        zones = [dict(ZONES[0], name='Entrance')]
        changed = self.update(16.5, zones=zones)
        self.assertEqual(changed['stable_s'], 0.)

    def test_drift_measured_from_anchor_not_only_previous_frame(self):
        self.settle()
        first = self.assessment.status(15., frame_age=0.)
        for index in range(1, 10):
            moved = IDENTITY.copy(); moved[0, 2] = index*.3
            result = self.update(15.+index*.25, matrix=moved)
        self.assertNotEqual(first['reference_id'], result['reference_id'])
        self.assertFalse(result['ready_to_save'])

    def test_verified_weak_landmarks_can_hold_but_unverified_cannot(self):
        result = self.settle()
        retained = self.update(15.25, alignment={'state':'tracking','accepted':True,'weak':True,
            'pose_gap_px':1.,'static_agreement':{'good_tiles':8,'good_fraction':1.}})
        self.assertTrue(retained['ready_to_save'])
        self.assertEqual(result['reference_id'], retained['reference_id'])
        lost = self.update(15.5, alignment={'state':'holding','accepted':False})
        self.assertIsNone(lost['score_pct'])
        self.assertFalse(lost['ready_to_save'])
        self.assertEqual(lost['guidance']['action'], 'wait')
        recovered = self.update(15.75)
        self.assertEqual(result['reference_id'], recovered['reference_id'])
        self.assertFalse(recovered['ready_to_save'])

    def test_weak_only_or_stale_landmark_proof_cannot_certify_capture(self):
        weak={'state':'tracking','accepted':True,'weak':True,'pose_gap_px':1.,
              'static_agreement':{'good_tiles':8,'good_fraction':1.}}
        for t in np.arange(10., 16., .25):result=self.update(float(t),alignment=weak)
        self.assertFalse(result['ready_to_save'])
        self.assertLess(result['score_pct'],100)
        self.update(16.)
        self.assertTrue(self.update(16.25,alignment=weak)['ready_to_save'])
        for t in np.arange(16.5,27.,.25):result=self.update(float(t),alignment=weak)
        self.assertFalse(result['ready_to_save'])

    def test_clipped_small_zone_cannot_hide_in_total_coverage(self):
        zones = [ZONES[0], {'name':'Small path', 'polygon':[[.98,.5],[1.08,.5],[1.08,.55],[.98,.55]]}]
        coverage, points = zone_coverage(IDENTITY, (360,640), zones)
        self.assertEqual(coverage['required_visible'], 1)
        self.assertEqual(coverage['clipped_names'], ['Small path'])
        self.assertAlmostEqual(coverage['pct'], 20.)
        self.assertNotEqual(framing_guidance(points,coverage)['action'], 'good')

    def test_nonconvex_polygon_area_clip(self):
        zones=[{'name':'L', 'polygon':[[.5,.1],[1.1,.1],[1.1,.3],[.7,.3],[.7,.7],[.5,.7]]}]
        coverage, _ = zone_coverage(IDENTITY,(360,640),zones)
        self.assertAlmostEqual(coverage['pct'], 90.)

    def test_raw_geometry_respects_saved_remount(self):
        matrix=IDENTITY.copy();matrix[0,2]=80.
        result=self.update(10.,matrix=matrix)
        self.assertIn('Footpath',result['coverage']['clipped_names'])
        self.assertFalse(result['ready_to_save'])

    def test_invalid_geometry_and_empty_zones_never_score(self):
        for matrix,zones in ((np.zeros((3,3)),ZONES),(np.full((3,3),np.nan),ZONES),(IDENTITY,[])):
            result=self.update(10.,matrix=matrix,zones=zones)
            self.assertIsNone(result['score_pct'])
            self.assertFalse(result['ready_to_save'])

    def test_pending_capture_does_not_report_ready_or_success(self):
        self.settle()
        result=self.update(15.25,calibration={'pending':True,'revision':0,'error':''})
        self.assertFalse(result['ready_to_save'])
        self.assertEqual(result['calibration']['revision'],0)

    def test_returned_report_cannot_mutate_internal_state(self):
        result=self.settle()
        result['coverage']['clipped_names'].append('Fake')
        self.assertEqual(self.assessment.status(15.,frame_age=0.)['coverage']['clipped_names'],[])


class GuidedCalibrationTests(unittest.TestCase):
    def watcher(self):
        watcher=Watcher.__new__(Watcher)
        watcher._state_lock=threading.RLock()
        watcher._camera_setup_status=lambda: {'ready_to_save':True,'reference_id':'current'}
        return watcher

    def test_rejects_old_review_or_not_ready(self):
        watcher=self.watcher()
        with patch.object(watcher,'request_calibration') as capture:
            with self.assertRaises(ValueError): watcher.request_guided_calibration('old')
            watcher._camera_setup_status=lambda: {'ready_to_save':False,'reference_id':'current'}
            with self.assertRaises(ValueError): watcher.request_guided_calibration('current')
            capture.assert_not_called()

    def test_fresh_review_uses_existing_paused_calibration_guard(self):
        watcher=self.watcher()
        with patch.object(watcher,'request_calibration',side_effect=PermissionError('Pause first')):
            with self.assertRaises(PermissionError): watcher.request_guided_calibration('current')

    def test_fresh_review_requests_once_and_does_not_claim_saved(self):
        watcher=self.watcher()
        with patch.object(watcher,'request_calibration') as capture:
            watcher.request_guided_calibration('current')
            capture.assert_called_once_with()
        self.assertEqual(watcher._camera_setup_calibration()['revision'],0)


if __name__ == '__main__':
    unittest.main()
