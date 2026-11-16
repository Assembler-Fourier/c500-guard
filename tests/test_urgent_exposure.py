"""Urgent road arrivals must survive exposure correction without self-fitting."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from approach_light import measure_urgent_change

class UrgentExposureTests(unittest.TestCase):
    def setUp(self):
        yy,xx=np.mgrid[:360,:640]
        self.base=(50 + 140*xx/640 + 20*yy/360).astype(np.float32)
        self.mask=np.zeros((360,640),bool)
        self.mask[160:200,80:160]=True
        self.scene=np.ones_like(self.mask)

    def measure(self, current):
        return measure_urgent_change(current,self.base,self.mask,scene_mask=self.scene)

    def test_unrelated_car_does_not_create_a_road_change(self):
        frame=self.base.copy()
        frame[85:175,360:455]=240
        result=self.measure(frame)
        self.assertFalse(result['hit'])
        self.assertFalse(result['weak_hit'])

    def test_gain_change_with_an_unrelated_car_stays_quiet(self):
        frame=self.base*.86+6
        frame[85:175,360:455]=240
        result=self.measure(frame)
        self.assertFalse(result['hit'])
        self.assertFalse(result['weak_hit'])

    def test_local_arrivals_survive_the_same_exposure_and_other_car(self):
        for delta in (-25,25):
            with self.subTest(delta=delta):
                frame=self.base*.86+6
                frame[85:175,360:455]=240
                frame[180:188,110:118]+=delta
                self.assertTrue(self.measure(frame)['hit'])

    def test_urgent_pixels_cannot_change_the_fitted_exposure(self):
        frame=self.base*.86+6
        frame[85:175,360:455]=240
        original=self.measure(frame)
        frame[self.mask]+=25
        changed=self.measure(frame)
        self.assertEqual(original['global_offset'],changed['global_offset'])
        self.assertEqual(original['global_gain'],changed['global_gain'])
        self.assertTrue(changed['hit'])

    def test_large_flat_exposure_step_stays_quiet(self):
        self.base.fill(80)
        result=self.measure(self.base+60)
        self.assertFalse(result['hit'])
        self.assertFalse(result['weak_hit'])

    def test_insufficient_exterior_never_erases_a_local_arrival(self):
        self.scene=self.mask.copy()
        frame=self.base.copy()
        frame[180:188,110:118]+=25
        self.assertTrue(self.measure(frame)['hit'])

if __name__=='__main__':
    unittest.main()
