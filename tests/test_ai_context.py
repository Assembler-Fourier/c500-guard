"""Opt-in whole-picture recovery; fake inference and isolated configurations."""
import copy
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np

from test_watch import watch
import semantic_monitor as sm
from regional_inference import add_context_detections, inference_rects
from zone_editor import validate_config
from visit_memory import VisitMemory
import test_semantic_entrance_verifier as verifier_tests
from test_semantic_entrance_verifier import detection

STAGE = Path(__file__).resolve().parents[1]


def context_zone(**changes):
    fields = dict(name='First sighting', rect=(.4, .2, .5, .6), trigger='ai',
                  classes=['car', 'truck', 'bus'], ai_context=True, ai_confidence=.35,
                  motion=False, arrival_direction=None, ai_confirmation_window_s=1.5,
                  route_role='entrance', route_group='campus-entrance', alert_stage='advance')
    fields.update(changes)
    return watch.Zone(**fields)


class ContextMergeTests(unittest.TestCase):
    def setUp(self):
        self.monitor = sm.SemanticMonitor.__new__(sm.SemanticMonitor)
        self.monitor.confidence = .6
        self.monitor.zones = [context_zone()]
        self.car = detection((50, 25, 70, 60), .4)
        self.focused = [(0, 0, 100, 100)]

    def merge(self, primary=(), extra=None):
        return self.monitor._merge_context_detections(list(primary),
            [self.car] if extra is None else extra, (0, 0, 100, 100), self.focused)

    def test_default_false_retains_exact_original_mask_and_result(self):
        self.monitor.zones[0].ai_context = False
        primary = [detection((5, 5, 15, 15), .8)]
        self.assertEqual(add_context_detections(primary, [self.car], self.focused), self.merge(primary))
        self.assertEqual([], self.merge())
        self.assertFalse(watch.Zone('Old config', (0, 0, 1, 1)).ai_context)

    def test_missing_focused_vehicle_is_recovered_only_in_opted_region(self):
        self.assertEqual([self.car], self.merge())
        outside = detection((5, 20, 15, 60), .9)
        self.assertEqual([], self.merge(extra=[outside]))

    def test_disabled_empty_classes_false_and_truthy_invalid_flags_do_not_opt_in(self):
        for changes in ({'trigger': 'motion'}, {'trigger': 'hybrid', 'motion': False},
                        {'classes': []}, {'ai_context': False}, {'ai_context': 1}, {'ai_context': 'true'}):
            with self.subTest(changes=changes):
                self.monitor.zones = [context_zone(**changes)]
                self.assertEqual([], self.merge())

    def test_ground_point_exclusion_class_and_exact_floor_are_required(self):
        hole = [[.55,.5],[.65,.5],[.65,.7],[.55,.7]]
        self.monitor.zones[0].exclude_polygons = [hole]
        self.assertEqual([], self.merge())
        self.monitor.zones[0].exclude_polygons = []
        for obj in (detection((50,25,70,85),.9), detection((50,25,70,60),.9,'person'),
                    detection((50,25,70,60),.3499),
                    {**self.car, 'confidence': .35, 'raw_confidence': .349999}):
            self.assertEqual([], self.merge(extra=[obj]))
        exact = {**self.car, 'confidence': .35, 'raw_confidence': float(np.float32(.35))}
        self.assertEqual([exact], self.merge(extra=[exact]))

    def test_malformed_boxes_and_scores_are_not_rescued(self):
        for box in ([50,25,70,float('nan')], [-1,25,70,60], [70,25,50,60], [50,25,70,101]):
            self.assertEqual([], self.merge(extra=[{**self.car,'bbox':box}]))
        for score in (True, float('nan'), float('inf'), 1.1):
            self.assertEqual([], self.merge(extra=[{**self.car,'raw_confidence':score}]))

    def test_same_family_duplicate_upgrades_once_and_person_stays_separate(self):
        weak = detection((50,25,70,60), .36, 'truck')
        person = detection((50,25,70,60), .8, 'person')
        merged = self.merge([weak, person])
        self.assertEqual(2, len(merged))
        self.assertEqual([self.car, person], merged)

    def test_distinct_vehicle_is_not_erased_and_inputs_are_unchanged(self):
        first = detection((42,25,52,60), .8)
        before = copy.deepcopy([first, self.car])
        result = self.merge([first])
        self.assertEqual(2, len(result))
        self.assertEqual(before, [first, self.car])

    def test_crop_boundary_fragment_uses_provenance_and_keeps_one_complete_vehicle(self):
        fragment = detection((280,59,334.4,104),.75,'truck')
        whole = detection((280,3,334.4,104),.65,'truck')
        self.assertEqual([whole],self.monitor._merge_context_detections(
            [fragment],[whole],(0,0,640,360),[(0,77,556,352),(236,59,630,232)]))

    def test_own_floor_does_not_use_another_zones_lower_floor(self):
        self.monitor.zones[0].ai_confidence = .5
        self.monitor.zones.append(context_zone(name='Other', rect=(0,0,.2,.2), ai_confidence=.2))
        self.assertEqual([], self.merge())

    def test_bounds_of_whole_context_are_not_invented_for_a_crop(self):
        self.assertEqual([], self.monitor._merge_context_detections([], [self.car],
            (10, 0, 100, 100), self.focused))

    def test_schema_rejects_nonboolean_and_preserves_explicit_flag(self):
        raw = json.loads((STAGE/'config/views/ptz2/zones.json').read_text())
        for value in (1, 0, 'true', None, [], {}):
            bad = copy.deepcopy(raw);bad['zones'][0]['ai_context'] = value
            with self.assertRaises(ValueError):validate_config(bad)
        for value in (True,False):
            good = copy.deepcopy(raw);good['zones'][0]['ai_context'] = value
            self.assertIs(value, validate_config(good)['zones'][0]['ai_context'])
        self.assertEqual(raw['zones'], validate_config(raw)['zones'])

    def test_overlapping_zone_adds_no_crop_or_extra_inference_view(self):
        raw = json.loads((STAGE/'config/views/ptz2/zones.json').read_text())
        zones = [watch.Zone(**row) for row in raw['zones']]
        old_top = next(z for z in zones if z.name == 'Top road arrival')
        new_top = copy.deepcopy(old_top)
        new_top.name = 'Top road first sighting';new_top.ai_context = True
        new_top.ai_confidence = .35;new_top.arrival_direction = None;new_top.approach_motion = False
        self.assertEqual(inference_rects(zones,640,360), inference_rects(zones+[new_top],640,360))
        self.assertEqual(.55,old_top.ai_confidence)
        self.assertEqual([1.,.5],old_top.arrival_direction)
        self.assertTrue(old_top.approach_motion)
        self.monitor.zones = zones
        floor = self.monitor._candidate_confidence()
        self.monitor.zones = zones + [new_top]
        self.assertEqual(floor,self.monitor._candidate_confidence())

    def test_lower_context_floor_only_publishes_objects_in_its_own_region(self):
        self.monitor._status = {}
        self.monitor.zones.append(context_zone(name='Exit',rect=(0,.2,.3,.6),
            ai_context=False,ai_confidence=.5,route_role='exit',route_group='campus-exit'))
        self.assertEqual(.5,self.monitor._display_floor())
        low_exit = detection((10,25,30,60),.4)
        self.monitor._confirm([low_exit,self.car],100,100,100.)
        visible = self.monitor._status['detections']
        self.assertEqual(1,len(visible))
        self.assertEqual(['First sighting'],visible[0]['regions'])

    def test_weak_exit_boxes_cannot_extend_a_previously_acknowledged_exit_visit(self):
        raw = json.loads((STAGE/'config/views/ptz2/zones.json').read_text())
        base = [watch.Zone(**z) for z in raw['zones'] if not z.get('ai_context')]
        early = copy.deepcopy(next(z for z in base if z.name=='Top road arrival'))
        early.name='Top first sighting';early.ai_context=True;early.ai_confidence=.35
        early.arrival_direction=None;early.approach_motion=False
        weak_exit=detection((.46*640,.45*360,.54*640,.65*360),.4)
        for zones in (base,base+[early]):
            with self.subTest(context_enabled=len(zones)>len(base)):
                monitor=sm.SemanticMonitor.__new__(sm.SemanticMonitor)
                monitor.zones=zones;monitor.confidence=.6;monitor._status={}
                memory=VisitMemory()
                decision=memory.consider({'source':'light','route_group':'campus-exit',
                    'route_role':'exit','measurement':{'hit':True}},99.)
                self.assertTrue(memory.emitted(decision.visit_id,'prior',99.))
                self.assertEqual(1,memory.acknowledge('prior',99.5))
                for n in range(15):
                    at=100+n*.5
                    self.assertEqual([],monitor._confirm([weak_exit],640,360,at))
                    self.assertEqual([],monitor._status['detections'])
                    memory.observe(at,zones,{'available':True,'state':'ready','age_s':0,
                        'detections':monitor._status['detections']})
                self.assertEqual(0,memory.status(107.)['active_visits'])

    def test_same_track_keeps_weak_presence_into_waiting_but_new_waiting_object_does_not(self):
        early=context_zone(rect=(.4,.2,.5,.3))
        waiting=context_zone(name='Waiting',rect=(.4,.5,.5,.3),ai_context=False,
                             ai_confidence=.5,alert_stage='barrier')
        first=detection((50,20,70,40),.4)
        later=detection((50,30,70,55),.4)
        self.monitor.zones=[early,waiting];self.monitor._status={}
        self.monitor._confirm([first],100,100,100.)
        track=self.monitor._status['detections'][0]['track_id']
        self.monitor._confirm([later],100,100,100.5)
        visible=self.monitor._status['detections']
        self.assertEqual(1,len(visible))
        self.assertEqual(track,visible[0]['track_id'])
        self.assertEqual(['Waiting'],visible[0]['regions'])
        self.assertIn('First sighting',visible[0]['direction'])
        self.monitor._tracker.reset()
        self.monitor._confirm([later],100,100,101.)
        self.assertEqual([],self.monitor._status['detections'],
                         'a new weak object in Waiting has no first-sighting history')

    def test_history_never_crosses_route_role_class_or_normal_floor(self):
        early=context_zone()
        target=context_zone(name='Waiting',ai_context=False,ai_confidence=.5)
        self.monitor.zones=[early,target]
        item={**self.car,'regions':['Waiting'],'direction':{'First sighting':'unrestricted'}}
        self.assertTrue(self.monitor._context_presence(item))
        for updates in ({'route_group':'campus-exit','route_role':'exit'},
                        {'route_group':'campus-entrance','route_role':'pedestrian'},
                        {'route_group':'other','route_role':'entrance'}, {'classes':['person']}):
            with self.subTest(updates=updates):
                self.monitor.zones=[early,context_zone(name='Waiting',ai_context=False,
                    ai_confidence=.5,**updates)]
                self.assertFalse(self.monitor._context_presence(item))
        self.monitor.zones=[early,target]
        self.assertFalse(self.monitor._context_presence({**item,'label':'person'}))
        self.assertFalse(self.monitor._context_presence({**item,'raw_confidence':.34}))
        self.assertFalse(self.monitor._context_presence({**item,'direction':{}}))


class ContextWorkerTests(unittest.TestCase):
    start = verifier_tests.EntranceVerifierTests.start
    wait_for = staticmethod(verifier_tests.EntranceVerifierTests.wait_for)
    frame = staticmethod(verifier_tests.EntranceVerifierTests.frame)
    submit = verifier_tests.EntranceVerifierTests.submit
    deliver = verifier_tests.EntranceVerifierTests.deliver

    def start_context(self, **kwargs):
        output = lambda call: [detection((50,20,70,60),.55)] if call % 3 == 0 else []
        monitor = self.start(tiny_output=output, **kwargs)
        monitor.zones[0].ai_context = True
        return monitor

    def test_existing_three_passes_and_two_distinct_source_frames_confirm(self):
        monitor = self.start_context(optional=False)
        first, report, stamp = self.deliver(monitor)
        self.assertEqual([], first)
        self.assertEqual(3,self.instances[0].calls)
        self.assertEqual(1,len(report['detections']))
        second, _, second_stamp = self.deliver(monitor)
        self.assertEqual(1,len(second))
        proof = monitor.event_context(second[0][2])
        self.assertEqual(2,proof['observations'])
        self.assertGreater(second_stamp,stamp)
        self.assertEqual(6,self.instances[0].calls)

    def test_s_and_rescued_tiny_merge_without_double_counting(self):
        monitor = self.start_context()
        first,report,_ = self.deliver(monitor)
        self.assertEqual([],first)
        self.assertEqual(1,len(report['detections']))
        self.assertEqual(.8,report['detections'][0]['confidence'])
        self.assertEqual(3,self.instances[0].calls)
        self.assertEqual(1,self.instances[1].calls)
        second,_,_ = self.deliver(monitor)
        self.assertEqual(2,monitor.event_context(second[0][2])['observations'])

    def test_opt_in_does_not_create_a_whole_picture_pass_when_disabled(self):
        monitor = self.start_context(optional=False,context=False)
        events,report,_ = self.deliver(monitor)
        self.assertEqual([],events)
        self.assertEqual([],report['detections'])
        self.assertEqual(2,self.instances[0].calls)
        self.assertEqual(2,report['inference_views'])

    def test_expired_context_source_is_not_published(self):
        def expire(detector,_frame):
            if detector.calls==3:self.offset += 3.1
        monitor = self.start_context(optional=False,tiny_callback=expire)
        self.submit(monitor)
        self.assertTrue(self.finished.wait(1.))
        self.assertEqual(0,monitor._pipeline['results_published'])
        self.assertTrue(monitor._output.empty())

    def test_reset_invalidates_inflight_context_source(self):
        holder = {}
        def reset(detector,_frame):
            if detector.calls==3:holder['monitor'].reset()
        monitor = self.start_context(optional=False,tiny_callback=reset);holder['monitor']=monitor
        self.submit(monitor)
        self.assertTrue(self.finished.wait(1.))
        self.assertEqual(0,monitor._pipeline['results_published'])
        self.assertTrue(monitor._output.empty())


if __name__=='__main__':unittest.main()
