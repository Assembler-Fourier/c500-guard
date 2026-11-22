import copy
import unittest
from watch import Watcher


class DisplayContinuityTests(unittest.TestCase):
    def setUp(self):
        self.watcher = Watcher.__new__(Watcher)
        self.watcher._alignment_blocked = False
        self.watcher.view_name = 'ptz2'

    def report(self, stamp, *, region=True, ident=1, generation=1, confidence=.8):
        return {'observed_at': stamp, 'age_s': .1, 'tracking_generation':[generation,0],
                'detections':[{'label':'car','track_id':ident,'confidence':confidence,
                               'regions':['Entrance'] if region else [], 'confirmed_regions':[],
                               'box_n':[.6,.4,.8,.7],'trail':[[.7,.65],[.7,.7]]}]}

    def test_current_box_continues_across_boundary_without_claiming_zone_or_alert(self):
        self.watcher._display_objects(self.report(10.))
        report=self.report(10.25,region=False)
        before=copy.deepcopy(report)
        objects=self.watcher._display_objects(report)
        self.assertEqual(objects[0]['id'],1)
        self.assertEqual(objects[0]['state'],'seen')
        self.assertEqual(objects[0]['zone'],'')
        self.assertEqual(report,before,'display must not alter semantic evidence')

    def test_new_unscoped_guess_and_other_car_do_not_inherit(self):
        self.assertEqual(self.watcher._display_objects(self.report(10.,region=False)),[])
        self.watcher._display_objects(self.report(10.25))
        self.assertEqual(self.watcher._display_objects(self.report(10.5,region=False,ident=2)),[])

    def test_no_box_is_synthesized_for_missing_detection(self):
        self.watcher._display_objects(self.report(10.))
        report=self.report(10.25);report['detections']=[]
        self.assertEqual(self.watcher._display_objects(report),[])

    def test_weak_candidate_cannot_create_or_extend_bridge(self):
        self.watcher._display_objects(self.report(10.))
        self.assertEqual(self.watcher._display_objects(self.report(10.25,region=False,confidence=.3)),[])
        self.assertEqual(self.watcher._display_objects(self.report(11.01,region=False)),[])

    def test_repeated_ui_polls_do_not_refresh_source_history(self):
        self.watcher._display_objects(self.report(10.))
        for _ in range(10):self.watcher._display_objects(self.report(10.5,region=False))
        self.assertEqual(self.watcher._display_objects(self.report(11.01,region=False)),[])

    def test_reset_view_or_generation_clears_history(self):
        for changed in ('view','epoch','blocked'):
            self.setUp();self.watcher._display_objects(self.report(10.))
            report=self.report(10.25,region=False)
            if changed=='view':self.watcher.view_name='anpr'
            elif changed=='epoch':report['tracking_generation']=[2,0]
            else:self.watcher._alignment_blocked=True
            self.assertEqual(self.watcher._display_objects(report),[])

    def test_stale_source_cannot_continue_outside_zone(self):
        self.watcher._display_objects(self.report(10.))
        report=self.report(10.25,region=False);report['age_s']=1.1
        self.assertEqual(self.watcher._display_objects(report),[])


if __name__=='__main__':unittest.main()
