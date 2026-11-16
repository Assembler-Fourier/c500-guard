"""Synthetic image-policy controls, not real-world vehicle classification proof."""
from pathlib import Path
import sys,unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from approach_motion import ApproachMotion,corroborate_motion

POLYGON=[[.2,.3],[.9,.3],[.9,.7],[.2,.7]]


class ApproachMotionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(27)
        self.background=cv2.GaussianBlur(rng.integers(40,170,(180,320),dtype=np.uint8),(3,3),0)
        self.object=rng.integers(25,230,(30,45),dtype=np.uint8)
        self.object=cv2.GaussianBlur(self.object,(3,3),0)

    def detector(self):return ApproachMotion(POLYGON,[1,0])

    def frame(self,x,y=80):
        image=self.background.copy();image[y:y+30,x:x+45]=self.object
        return image

    def replay(self,frames):
        detector=self.detector()
        results=[]
        for i,frame in enumerate(frames):
            result=detector.update(frame,i*.25);results.append(result)
            if result['hit']:detector.mark_emitted(i*.25)
        return results

    def test_coherent_forward_body_warns_once_after_two_pairs(self):
        result=self.replay([self.frame(x) for x in (80,86,92,98,104,110,116,122)])
        self.assertEqual([2],[i for i,r in enumerate(result) if r['hit']])

    def test_reversed_and_sideways_objects_do_not_warn(self):
        for frames in ([self.frame(x) for x in (140,134,128,122,116,110)],
                       [self.frame(100,y) for y in (60,65,70,75,80,85)]):
            self.assertFalse(any(r['hit'] for r in self.replay(frames)))

    def test_parked_jitter_and_an_object_outside_road_do_not_warn(self):
        for frames in ([self.frame(x) for x in (100,101,100,101,99,100,101)],
                       [self.frame(x,12) for x in (80,86,92,98,104,110)]):
            self.assertFalse(any(r['hit'] for r in self.replay(frames)))

    def test_static_empty_scene_and_global_exposure_steps_do_not_warn(self):
        frames=[np.clip(self.background.astype(float)*gain+offset,0,255).astype(np.uint8)
                for gain,offset in ((1,0),(1,30),(.85,0),(1.2,10),(1,0),(1,8),(1,0))]
        self.assertFalse(any(r['hit'] for r in self.replay(frames)))
        self.assertFalse(any(r['hit'] for r in self.replay([self.background]*8)))

    def test_global_camera_translation_and_zoom_do_not_warn(self):
        for zoom in (False,True):
            frames=[]
            for i in range(8):
                matrix=cv2.getRotationMatrix2D((160,90),0,1+i*.01) if zoom else np.float32([[1,0,i*3],[0,1,i]])
                frames.append(cv2.warpAffine(self.background,matrix,(320,180),borderMode=cv2.BORDER_REFLECT))
            result=self.replay(frames)
            self.assertFalse(any(r['hit'] for r in result))
            if not zoom:self.assertTrue(any(r['reason']=='camera or widespread motion' for r in result))

    def test_moving_soft_glare_and_shadow_on_static_texture_do_not_warn(self):
        yy,xx=np.mgrid[:180,:320]
        for sign in (-1,1):
            frames=[]
            for x in (80,88,96,104,112,120,128,136):
                light=sign*95*np.exp(-((xx-x)**2/800+(yy-95)**2/450))
                frames.append(np.clip(self.background.astype(float)+light,0,255).astype(np.uint8))
            self.assertFalse(any(r['hit'] for r in self.replay(frames)))

    def test_moving_sharp_shadow_and_glare_preserve_static_road_texture(self):
        for amount in (-.6,.6):
            frames=[]
            for x in (75,83,91,99,107,115,123,131):
                mask=np.zeros((180,320),np.float32);mask[74:115,x:x+55]=1
                mask=cv2.GaussianBlur(mask,(0,0),1.5)
                frames.append(np.clip(self.background.astype(float)*(1+amount*mask),0,255).astype(np.uint8))
            self.assertFalse(any(r['hit'] for r in self.replay(frames)))

    def test_one_global_scene_cut_cannot_complete_two_pairs(self):
        frames=[self.background,self.frame(80),self.frame(86),self.background,self.background]
        # A birth/cut is not translational evidence; only one true motion pair.
        self.assertFalse(any(r['hit'] for r in self.replay(frames)))

    def test_no_warning_from_one_pair_or_repeated_timestamp(self):
        detector=self.detector();detector.update(self.frame(80),0)
        self.assertFalse(detector.update(self.frame(86),.25)['hit'])
        self.assertFalse(detector.update(self.frame(92),.25)['hit'])
        self.assertEqual(1,detector.hits)

    def test_gap_clears_confirmation_but_does_not_rearm_a_latched_warning(self):
        detector=self.detector()
        for at,x in ((0,80),(.25,86),(.5,92)):
            result=detector.update(self.frame(x),at)
        self.assertTrue(result['hit'])
        self.assertTrue(detector.mark_emitted(.5))
        self.assertFalse(detector.update(self.frame(100),3)['hit'])
        self.assertFalse(detector.update(self.frame(106),3.25)['hit'])
        self.assertFalse(detector.update(self.frame(112),3.5)['hit'])
        self.assertTrue(detector.latched)

    def test_fresh_quiet_dwell_allows_a_later_approach(self):
        detector=self.detector();hits=[]
        for i,x in enumerate((80,86,92)):
            if detector.update(self.frame(x),i*.25)['hit']:
                hits.append(i*.25);detector.mark_emitted(i*.25)
        for i in range(3,21):detector.update(self.background,i*.25)
        for i,x in enumerate((80,86,92),21):
            if detector.update(self.frame(x),i*.25)['hit']:
                hits.append(i*.25);detector.mark_emitted(i*.25)
        self.assertEqual(2,len(hits))

    def test_bad_frame_time_and_shape_cannot_emit_or_use_old_confirmation(self):
        detector=self.detector();detector.update(self.frame(80),0);detector.update(self.frame(86),.25)
        self.assertFalse(detector.update(self.frame(92),float('nan'))['hit'])
        self.assertFalse(detector.update(self.frame(98),.5)['hit'])
        self.assertFalse(detector.update(np.zeros((20,20),np.uint8),.75)['hit'])

    def test_flat_scene_has_insufficient_static_context(self):
        result=self.replay([np.full((180,320),80,np.uint8)]*3)
        self.assertFalse(any(r['hit'] for r in result))
        self.assertEqual('insufficient static context',result[-1]['reason'])

    def test_configuration_is_bounded(self):
        for polygon,direction in (([],[1,0]),(POLYGON,[0,0]),([[0,0],[1,0],[2,1]],[1,0]),(POLYGON,[float('nan'),0])):
            with self.assertRaises(ValueError):ApproachMotion(polygon,direction)
        with self.assertRaises(ValueError):ApproachMotion(POLYGON,[1,0],max_gap=2)

    def test_rejected_delivery_retries_only_on_fresh_qualified_movement(self):
        detector=self.detector()
        for i,x in enumerate((80,86,92)):
            result=detector.update(self.frame(x),i*.25)
        self.assertTrue(result['hit'])
        self.assertFalse(detector.mark_emitted(.25))
        self.assertTrue(detector.update(self.frame(98),.75)['hit'])
        self.assertTrue(detector.mark_emitted(.75))
        self.assertFalse(detector.update(self.frame(104),1.)['hit'])


class CorroboratedMotionTests(unittest.TestCase):
    def proof(self):return {'hit':True,'qualified':True,'pairs':2,'support_bbox':[.40,.40,.48,.47],'observed_at':5.75}
    def candidate(self,**changes):
        return {**{'label':'truck','confidence':.418,'bbox':[110,40,170,96],'observed_at':5.,'view':'whole'},**changes}
    def fuse(self,candidates,proof=None,now=5.75):
        return corroborate_motion(proof or self.proof(),candidates,now,POLYGON,320,180,direction=[1,0])

    def pair(self):return [self.candidate(),self.candidate(bbox=[116,41,176,97],observed_at=5.5)]

    def test_fresh_weak_candidate_overlapping_directional_motion_can_warn(self):
        result=self.fuse(self.pair())
        self.assertIsNotNone(result);self.assertEqual(.418,result['confidence'])
        self.assertEqual(.25,result['age_s'])
        self.assertEqual(2,result['candidate_observations'])

    def test_motion_without_vehicle_or_without_two_pairs_cannot_warn(self):
        self.assertIsNone(self.fuse([]))
        for change in ({'hit':False},{'qualified':False},{'pairs':1},{'observed_at':5.5}):
            self.assertIsNone(self.fuse([self.candidate()],{**self.proof(),**change}))

    def test_stale_future_low_confidence_or_nonvehicle_candidates_do_not_corrobate(self):
        for change in ({'observed_at':4.749},{'observed_at':5.751},{'confidence':.299},
                       {'confidence':float('nan')},{'label':'person'},{'bbox':[0,0,float('inf'),1]}):
            with self.subTest(change=change):self.assertIsNone(self.fuse([self.candidate(**change)]))

    def test_waiting_outline_and_nonoverlapping_vehicle_do_not_corrobate(self):
        for box in ([110,40,170,155],[220,65,265,110],[130,70,136,90]):
            self.assertIsNone(self.fuse([self.candidate(bbox=box)]))

    def test_patterned_light_motion_without_spatial_ai_support_is_not_a_warning(self):
        # Independent image probe proves patterned light can pass optical flow.
        # Its motion-only proof cannot borrow a static model phantom elsewhere.
        self.assertIsNone(self.fuse([self.candidate(bbox=[220,60,270,115])]))

    def test_missing_or_malformed_candidate_collection_does_not_raise(self):
        self.assertIsNone(self.fuse(None))
        self.assertIsNone(self.fuse([None,{},'bad']))

    def test_stationary_car_under_moving_patterned_light_cannot_corrobate(self):
        self.assertIsNone(self.fuse([self.candidate(),self.candidate(observed_at=5.5)]))
        # Bottom-only shape jitter also does not prove movement of the body.
        self.assertIsNone(self.fuse([self.candidate(),self.candidate(observed_at=5.5,bbox=[110,40,170,103])]))

    def test_two_views_of_one_frame_cannot_replace_two_observations(self):
        self.assertIsNone(self.fuse([self.candidate(),self.candidate(bbox=[116,42,176,99])]))

    def test_reverse_and_unrelated_boxes_cannot_corrobate_forward_flow(self):
        for box in ([104,38,164,93],[180,40,240,96]):
            self.assertIsNone(self.fuse([self.candidate(),self.candidate(observed_at=5.5,bbox=box)]))


if __name__=='__main__':unittest.main()
