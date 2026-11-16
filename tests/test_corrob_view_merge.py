"""Same-source supplemental model merging; no model, camera or live state."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from regional_inference import add_context_detections, merge_corrob_view
from object_tracking import ObjectVisitTracker
from region_geometry import region_contains


def obj(box, confidence=.7, label='car', **extra):
    return dict(label=label, confidence=confidence, bbox=list(box), **extra)


class CorroboratingMergeTests(unittest.TestCase):
    def test_empty_primary_accepts_same_frame_supplement_inside_focused_view(self):
        extra = obj((100,100,150,160))
        self.assertEqual([extra], merge_corrob_view([], [extra], primary_rects=[(0,0,200,200)]))
        # The old whole-picture policy is intentionally unchanged.
        self.assertEqual([], add_context_detections([], [extra], [(0,0,200,200)]))

    def test_unique_same_family_match_prefers_higher_score_without_summing(self):
        primary = obj((100,100,150,160), .50, model='tiny')
        extra = obj((101,100,151,160), .68, label='bus', model='s')
        self.assertEqual([extra], merge_corrob_view([primary], [extra]))
        self.assertEqual([primary], merge_corrob_view([primary], [{**extra,'confidence':.49}]))
        self.assertEqual([primary], merge_corrob_view([primary], [{**extra,'confidence':.50}]))

    def test_exact_confidence_breaks_rounded_score_tie(self):
        primary = obj((100,100,150,160), .6, raw_confidence=.59996)
        extra = obj((100,100,150,160), .6, raw_confidence=.60004)
        self.assertEqual([extra], merge_corrob_view([primary], [extra]))

    def test_distinct_adjacent_car_and_overlapping_person_bicycle_survive(self):
        primary = obj((100,100,150,160))
        nearby = obj((155,100,205,160), .8)
        person = obj((110,100,140,160), .9, 'person')
        bicycle = obj((100,100,150,160), .75, 'bicycle')
        result = merge_corrob_view([primary], [nearby,person,bicycle])
        self.assertEqual(4,len(result))
        self.assertEqual({'car','person','bicycle'},{x['label'] for x in result})
        self.assertIn(nearby,result)

    def test_high_confidence_primary_crop_fragment_yields_to_complete_supplement(self):
        part = obj((70,20,90,80), .95)
        whole = obj((30,20,90,80), .65)
        self.assertEqual([whole],merge_corrob_view([part],[whole],
            primary_rects=[(70,0,150,100)],supplemental_rect=(0,0,100,100)))

    def test_high_confidence_supplement_fragment_cannot_replace_complete_primary(self):
        whole = obj((30,20,90,80), .65)
        part = obj((70,20,90,80), .95)
        self.assertEqual([whole],merge_corrob_view([whole],[part],
            primary_rects=[(0,0,100,100)],supplemental_rect=(70,0,150,100)))

    def test_small_complete_box_or_unknown_crop_provenance_is_not_erased(self):
        part = obj((70,20,90,80), .95)
        whole = obj((30,20,90,80), .65)
        for rects in ([],[(0,0,100,100)],[(70,0,150,100),(0,0,100,100)]):
            with self.subTest(rects=rects):
                self.assertEqual(2,len(merge_corrob_view([part],[whole],
                    primary_rects=rects,supplemental_rect=(0,0,100,100))))

    def test_two_clipped_boxes_do_not_supply_complete_fragment_proof(self):
        whole = obj((30,20,90,80), .65)
        part = obj((70,20,90,80), .95)
        self.assertEqual(2,len(merge_corrob_view([part],[whole],
            primary_rects=[(70,0,150,100)],supplemental_rect=(0,0,90,100))))

    def test_supplement_spanning_two_cars_cannot_merge_their_identities(self):
        first, second = obj((0,0,20,40)), obj((20,0,40,40))
        spanning = obj((0,0,40,40), .95)
        self.assertEqual([first,second],merge_corrob_view([first,second],[spanning]))

    def test_two_supplement_cars_cannot_split_one_primary_identity(self):
        primary = obj((0,0,40,40))
        first, second = obj((0,0,20,40), .95),obj((20,0,40,40), .9)
        self.assertEqual([primary],merge_corrob_view([primary],[first,second]))

    def test_supplement_duplicates_produce_one_object(self):
        first, duplicate = obj((10,10,30,40), .7),obj((11,10,31,40), .8, 'truck')
        self.assertEqual([duplicate],merge_corrob_view([],[first,duplicate]))

    def test_inputs_are_unchanged_and_returned_boxes_are_independent(self):
        primary=[obj((10,10,30,40))];extra=[obj((100,100,120,140))]
        before=copy.deepcopy((primary,extra))
        result=merge_corrob_view(primary,extra)
        result[0]['bbox'][0]=999
        self.assertEqual(before,(primary,extra))

    def test_malformed_candidates_do_not_reach_tracker(self):
        bad=[None,{},obj((1,1,2,2),True),obj((1,1,2,2),float('nan')),
             obj((1,1,2,2),raw_confidence=float('inf')),obj((1,1,2,2),'0.7'),
             obj((1,1,2,2),label='dog'),obj((1,1,1,2)),obj((1,1,float('nan'),2)),
             obj((1,1,2,2),raw_confidence=True),obj((True,1,2,2))]
        good=obj((10,10,30,40))
        self.assertEqual([good],merge_corrob_view(bad,bad+[good]))

    def test_each_input_consumes_at_most_fifty_and_output_is_bounded(self):
        def bounded(offset):
            for i in range(50):yield obj((offset+i*10,0,offset+i*10+5,5))
            raise AssertionError('consumed more than50')
        self.assertEqual(50,len(merge_corrob_view(bounded(0),bounded(1000))))

    def test_full_primary_capacity_still_allows_matching_score_upgrade(self):
        primary=[obj((i*10,0,i*10+5,5),.6) for i in range(50)]
        upgraded={**primary[-1],'confidence':.9}
        result=merge_corrob_view(primary,[obj((1000,0,1005,5)),upgraded])
        self.assertEqual(50,len(result))
        self.assertEqual(upgraded,result[-1])

    def test_bad_thresholds_fail_explicitly(self):
        for threshold in (0,1.1,True,float('nan'),'0.45'):
            with self.subTest(threshold=threshold),self.assertRaises(ValueError):
                merge_corrob_view([],[],iou_threshold=threshold)


class CorroboratingTrackerTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'config/views/ptz2/zones.json'
        row=next(z for z in json.loads(path.read_text())['zones'] if z['name']=='Entrance waiting')
        self.zone=SimpleNamespace(**row,ai_enabled=True)
        self.zone.contains_point=lambda x,y:region_contains(x,y,row['polygon'],row.get('exclude_polygons',[]))
        self.tracker=ObjectVisitTracker()

    def test_two_models_on_one_source_frame_cannot_satisfy_two_observations(self):
        primary=obj((450,150,520,199),.5362)
        supplement=obj((454,149,517,200),.7694)
        merged=merge_corrob_view([primary],[supplement])
        self.assertEqual(1,len(merged))
        events,tracked=self.tracker.update(merged,[self.zone],640,360,100.)
        self.assertEqual([],events)
        self.assertEqual('confirming',tracked[0]['region_status'][self.zone.name])
        events,_=self.tracker.update(merged,[self.zone],640,360,100.5)
        self.assertEqual(1,len(events))
        proof=self.tracker.event_context(events[0][2],[self.zone])
        self.assertEqual(2,proof['observations'])
        self.assertEqual(.7694,proof['confidence'])
        self.assertEqual([100.,100.5],[s['observed_at'] for s in proof['strong_confirmation']])

    def test_saved_cpu_whole_frame_boxes_confirm_only_on_second_source(self):
        # Exact clean-frame CPU S results, not a GPU/scheduling measurement.
        # First: source offset9.9886835; second:10.4963805. The second frame
        # also contains the saved Tiny detection, which must count only once.
        first=obj((454.61,147.39,519.89,198.71),.638,raw_confidence=.6380431056022644)
        second=obj((454.,148.92,517.27,199.56),.7694,raw_confidence=.769395649433136)
        primary=obj((463.93,149.3,518.31,198.67),.5362,raw_confidence=.5362275838851929)
        events,_=self.tracker.update(merge_corrob_view([],[first]),[self.zone],640,360,100.)
        self.assertEqual([],events)
        merged=merge_corrob_view([primary],[second])
        self.assertEqual([second],merged)
        events,_=self.tracker.update(merged,[self.zone],640,360,100.507697)
        self.assertEqual(1,len(events))
        proof=self.tracker.event_context(events[0][2],[self.zone])
        self.assertEqual('strong_pair',proof['confirmation_mode'])
        self.assertEqual(2,proof['observations'])
        self.assertEqual([.638,.7694],[s['confidence'] for s in proof['strong_confirmation']])

    def test_supplemental_score_is_not_added_to_weak_primary_score(self):
        primary=obj((450,150,520,199),.28)
        supplement=obj((454,149,517,200),.32)
        for at in (100.,100.5,101.):
            merged=merge_corrob_view([primary],[supplement])
            self.assertEqual(.32,merged[0]['confidence'])
            events,tracked=self.tracker.update(merged,[self.zone],640,360,at)
            self.assertEqual([],events)
            self.assertEqual('below confidence threshold',tracked[0]['region_status'][self.zone.name])


if __name__=='__main__':unittest.main()
