"""Independent bounded fusion API and exclusion checks; no live services."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from approach_motion import ApproachMotion, corroborate_motion
import test_approach_motion as motion_cases
POLYGON = motion_cases.POLYGON


class FusionContractTests(unittest.TestCase):
    def setUp(self):
        self.motion={'hit':True,'qualified':True,'pairs':2,'support_bbox':[.40,.40,.48,.47],'observed_at':5.75}
        self.candidate={'label':'truck','confidence':.418,'bbox':[110,40,170,96],
                        'observed_at':5.,'view':'whole'}
        self.next_candidate={**self.candidate,'bbox':[116,41,176,97],'observed_at':5.5}

    def fuse(self,motion=None,candidates=None,**kwargs):
        return corroborate_motion(self.motion if motion is None else motion,
            [self.candidate,self.next_candidate] if candidates is None else candidates,5.75,POLYGON,320,180,
            direction=[1,0],**kwargs)

    def test_original_valid_offer_is_preserved(self):
        self.assertIsNotNone(self.fuse())

    def test_context_pair_after_focused_distractors_is_not_dropped_by_older_prefix(self):
        distractor={**self.candidate,'label':'person'}
        candidates=[distractor]*100+[self.candidate]+[distractor]*149+[self.next_candidate]
        self.assertIsNotNone(self.fuse(candidates=candidates))

    def test_malformed_proof_types_are_rejected_without_exception(self):
        for field, values in (('pairs',('2',None,True,float('nan'))),
                              ('hit',('true',1,None)),('qualified',('true',1,None)),
                              ('observed_at',('5.75',True,float('inf')))):
            for value in values:
                with self.subTest(field=field,value=value):
                    self.assertIsNone(self.fuse(motion={**self.motion,field:value}))

    def test_boolean_or_string_model_scores_and_timestamps_are_not_numbers(self):
        for field, values in (('confidence',(True,False,'0.8',float('nan'))),
                              ('observed_at',(True,'5.0',float('inf')))):
            for value in values:
                with self.subTest(field=field,value=value):
                    self.assertIsNone(self.fuse(candidates=[{**self.candidate,field:value}]))

    def test_current_ignored_ground_and_ignored_motion_centre_cannot_corrobate(self):
        for hole in ([[.42,.52],[.46,.52],[.46,.56],[.42,.56]],
                     [[.40,.40],[.48,.40],[.48,.47],[.40,.47]]):
            with self.subTest(hole=hole):
                self.assertIsNone(self.fuse(exclude_polygons=[hole]))
        self.assertIsNotNone(self.fuse(exclude_polygons=[[[.8,.5],[.85,.5],[.85,.6],[.8,.6]]]))

    def test_invalid_ignored_polygons_fail_closed(self):
        for exclusions in (None,'bad',[[[0,0],[1,1]]],[POLYGON]*17):
            with self.subTest(exclusions=exclusions):
                self.assertIsNone(self.fuse(exclude_polygons=exclusions))

    def test_motion_inside_ignored_area_does_not_offer_a_warning(self):
        fixture=motion_cases.ApproachMotionTests();fixture.setUp()
        hole=[[.23,.40],[.55,.40],[.55,.66],[.23,.66]]
        detector=ApproachMotion(POLYGON,[1,0],exclude_polygons=[hole])
        for i,x in enumerate((80,86,92,98,104,110,116,122)):
            self.assertFalse(detector.update(fixture.frame(x),i*.25)['hit'])


if __name__=='__main__':unittest.main()
