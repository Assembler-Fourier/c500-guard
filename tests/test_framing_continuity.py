"""Framing advice must not turn transient landmark loss into a camera instruction."""
import unittest
from unittest import mock
import numpy as np
from test_watch import watcher, watch


class FramingContinuityTests(unittest.TestCase):
    def build(self):
        obj = watcher()
        obj.view_name = 'ptz2'
        obj.alignment_config = {'zoom_tolerance': .2}
        obj.zones = [watch.Zone(name='Lanes', rect=(.08, .33, .83, .59))]
        obj._reacquire = None
        return obj

    def test_missing_landmarks_never_invent_a_direction_from_old_advice(self):
        for old in (None, {'primary': 'good'}, {'primary':'zoom_in', 'zoom_factor':1.9},
                    {'primary':'zoom_out', 'cut_off':True}):
            with self.subTest(old=old):
                obj = self.build()
                if old:
                    obj._framing_last = (old, 99.)
                guide = obj._framing_status(False, {}, 100., (360,640,3))
                self.assertEqual(guide['state'], 'unknown')
                self.assertEqual(guide['headline'], 'View alignment uncertain')
                self.assertNotIn('zoom', guide['text'].lower())
                self.assertNotIn('barriers are not in', guide['text'])

    def test_transient_hold_keeps_verified_transform_and_detection_running(self):
        obj = self.build()
        frame = np.zeros((360,640,3), np.uint8)
        identity = [[1.,0.,0.],[0.,1.,0.]]
        obj.aligner = mock.Mock()
        obj.aligner.warp.return_value = frame
        obj.aligner.update.side_effect = [
            (frame, {'reference_ready':True, 'current_to_reference':identity}),
            (None, {'reference_ready':True, 'reason':'Vehicle glare obscures landmarks',
                    'current_to_reference':[[1.2,0.,30.],[0.,1.2,20.]]}),
            (frame, {'reference_ready':True, 'current_to_reference':identity})]
        with mock.patch.object(obj, '_maybe_save_pose'):
            self.assertIs(obj.align_frame(frame), frame)
            obj._pending_rearm = False
            self.assertIs(obj.align_frame(frame), frame)
            self.assertFalse(obj._alignment_blocked)
            self.assertFalse(obj._pending_rearm)
            guide = obj.alignment['framing']
            self.assertEqual(guide['primary'], 'held')
            self.assertTrue(guide['retained'])
            self.assertAlmostEqual(guide['scale'], 1.)
            self.assertIs(obj.align_frame(frame), frame)
            self.assertFalse(obj._pending_rearm)
            self.assertEqual(obj.alignment['framing']['primary'], 'good')

    def test_weak_current_confirmation_is_labelled_retained(self):
        obj = self.build()
        frame = np.zeros((360,640,3), np.uint8)
        obj.aligner = mock.Mock()
        obj.aligner.update.return_value = (frame, {'reference_ready':True, 'weak':True,
            'current_to_reference':[[1.,0.,0.],[0.,1.,0.]]})
        self.assertIs(obj.align_frame(frame), frame)
        self.assertEqual(obj.alignment['state'], 'tracking')
        self.assertEqual(obj.alignment['framing']['primary'], 'held')
        self.assertFalse(obj._alignment_blocked)

    def test_real_rejection_without_valid_prior_remains_blocked(self):
        obj = self.build()
        frame = np.zeros((360,640,3), np.uint8)
        obj.aligner = mock.Mock()
        obj.aligner.update.return_value = (None, {'reference_ready':True, 'reason':'Different scene'})
        self.assertIsNone(obj.align_frame(frame))
        self.assertTrue(obj._alignment_blocked)
        self.assertEqual(obj.alignment['framing']['state'], 'unknown')
        obj.aligner.warp.assert_not_called()


if __name__ == '__main__':
    unittest.main()
