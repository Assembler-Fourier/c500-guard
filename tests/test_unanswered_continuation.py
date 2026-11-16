"""Bounded unacknowledged approach continuation and explicit new-car controls."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from object_tracking import ObjectVisitTracker
from visit_memory import VisitMemory
from test_watch import watch,watcher

ROUTE='campus-entrance'
OLD=[.56,.25,.66,.40]
NEW=[.62,.32,.74,.50]


def zone(name,stage):
    return SimpleNamespace(name=name,route_group=ROUTE,route_role='entrance',
        alert_stage=stage,ai_confirmation_window_s=1.5 if stage=='advance' else 0,
        arrival_direction=[1,.5],arrival_min_progress=.012,ai_confidence=.55,
        ai_confirm_observations=2,motion_active=False,light_active=False,
        contains_point=lambda x,y:.45<x<.9 and .2<y<.65,confirmation_polygon=None)


def detection(track,box,name):
    return {'track_id':track,'label':'car','bbox_normalized':box,'regions':[name],'confirmed_regions':[name]}


class UnansweredContinuationTests(unittest.TestCase):
    def setUp(self):
        self.memory=VisitMemory();self.zones=[zone('Approach','advance'),zone('Barrier','barrier')]
        self.previous={'source':'ai','route_group':ROUTE,'route_role':'entrance','zone_name':'Approach',
            'alert_stage':'advance','track_id':1,'label':'car','confidence':.8,'observations':2,
            'observed_at':100.,'first_observed_at':99.,'bbox_normalized':OLD,'direction_state':'toward'}
        self.ident=self.memory.consider(self.previous,100).visit_id
        self.memory.emitted(self.ident,'a',100)
        self.observe(100,detection(1,OLD,'Approach'))
        for at in (100.5,103,105.5):self.observe(at)
        self.details={**self.previous,'zone_name':'Barrier','alert_stage':'barrier','track_id':2,
            'observed_at':106.8,'first_observed_at':105.7,'bbox_normalized':NEW}
        self.observe(105.7,detection(2,NEW,'Barrier'))
        self.observe(106.8,detection(2,NEW,'Barrier'))

    def observe(self,now,*objects):
        self.memory.observe(now,self.zones,{'available':True,'state':'ready','age_s':0.,'detections':list(objects)})

    def attempt(self,details=None,now=106.8):
        return self.memory.continue_unanswered(self.ident,'a',self.previous,
            self.details if details is None else details,self.zones,now)

    def test_forward_barrier_continuation_keeps_the_sounding_visit(self):
        result=self.attempt()
        self.assertTrue(result.suppress)
        self.assertEqual(self.ident,result.visit_id)
        self.assertEqual({'1','2'},self.memory._visits[self.ident].track_ids)
        self.assertTrue(self.memory.evidence_for_alarm('a',106.8)['present'])

    def test_acknowledged_or_no_longer_current_alarm_never_inherits(self):
        self.memory.acknowledge('a',106.7)
        self.assertIsNone(self.attempt())
        self.assertFalse(self.memory.consider(self.details,106.8).suppress)

    def test_previously_visible_second_car_cannot_be_joined(self):
        # Even if the follower later acquires another tracker ID, any observed
        # queue makes continuation ambiguous and keeps the new visit separate.
        self.memory._visits[self.ident].separate_tracks.add('different-old-id')
        self.assertIsNone(self.attempt())
        self.assertEqual({'1'},self.memory._visits[self.ident].track_ids)

    def test_separate_car_memory_is_collected_when_both_are_visible(self):
        other=VisitMemory();ident=other.consider(self.previous,100).visit_id;other.emitted(ident,'a',100)
        other.observe(100.2,self.zones,{'available':True,'state':'ready','age_s':0.,
            'detections':[detection(1,OLD,'Approach'),detection(2,NEW,'Barrier')]})
        self.assertIn('2',other._visits[ident].separate_tracks)

    def test_six_second_clear_boundary_proves_a_new_arrival(self):
        self.assertIsNone(self.attempt({**self.details,'first_observed_at':106.5}))
        self.assertIsNotNone(self.attempt({**self.details,'first_observed_at':106.499}))

    def test_eight_second_identity_horizon_is_not_extended(self):
        visit=self.memory._visits[self.ident];visit.clear_since=visit.retired_at=None
        self.assertIsNone(self.attempt({**self.details,'observed_at':108.001,'first_observed_at':107.8},108.001))

    def test_stale_future_nonfinite_or_unqualified_observations_are_rejected(self):
        for changes in ({'observed_at':103.}, {'observed_at':107.}, {'observed_at':float('nan')},
                        {'first_observed_at':float('inf')},{'first_observed_at':107.},
                        {'confidence':.54},{'confidence':float('nan')},{'observations':1}):
            with self.subTest(changes=changes):self.assertIsNone(self.attempt({**self.details,**changes}))

    def test_wrong_route_stage_class_and_opposed_or_oversized_boxes_are_rejected(self):
        for changes in ({'route_group':'campus-exit'},{'alert_stage':'advance'},{'label':'person'},
                        {'direction_state':'away'},{'bbox_normalized':[0,0,1,1]},
                        {'bbox_normalized':[.50,.24,.62,.38]},
                        {'bbox_normalized':[.6,.3,float('nan'),.5]}):
            with self.subTest(changes=changes):self.assertIsNone(self.attempt({**self.details,**changes}))

    def test_policy_is_opt_in_and_rejection_does_not_mutate_aliases(self):
        self.zones[0].ai_confirmation_window_s=0
        self.assertIsNone(self.attempt())
        self.assertEqual({'1'},self.memory._visits[self.ident].track_ids)


class FirstSightingContinuationTests(UnansweredContinuationTests):
    """Repeat every identity/ACK/queue/clear guard with unrestricted sighting."""
    def setUp(self):
        super().setUp()
        self.first=self.zones[0]
        self.first.ai_enabled=True
        self.first.polygon=[[.45,.2],[.9,.2],[.9,.65],[.45,.65]]
        self.first.rect=(.45,.2,.45,.45)
        self.first.exclude_polygons=[]
        self.first.classes=['car','truck','bus']
        self.directional=copy.deepcopy(self.first)
        self.directional.name='Configured directional approach'
        self.first.ai_context=True
        self.first.arrival_direction=None
        self.previous['direction_state']='unrestricted'
        self.zones.append(self.directional)

    def test_unrestricted_sighting_borrows_only_identity_direction_and_keeps_its_policy(self):
        self.assertTrue(self.attempt().suppress)
        self.assertIsNone(self.first.arrival_direction)
        self.assertEqual('unrestricted',self.previous['direction_state'])

    def test_missing_ambiguous_or_nonopted_direction_does_not_extend_identity(self):
        for action in ('missing','ambiguous','not-opted','disabled-first'):
            with self.subTest(action=action):
                self.setUp()
                if action=='missing':self.zones.remove(self.directional)
                elif action=='ambiguous':self.zones.append(copy.deepcopy(self.directional))
                elif action=='not-opted':self.first.ai_context=False
                else:self.first.ai_enabled=False
                self.assertIsNone(self.attempt())
                self.assertEqual({'1'},self.memory._visits[self.ident].track_ids)

    def test_different_or_disabled_geometry_role_class_and_invalid_direction_are_rejected(self):
        for name,value in (('route_group','other'),('route_role','exit'),('alert_stage','barrier'),
                ('ai_enabled',False),('polygon',[[.4,.2],[.9,.2],[.9,.65],[.4,.65]]),
                ('rect',(.4,.2,.5,.45)),('exclude_polygons',[[[.5,.3],[.6,.3],[.6,.4],[.5,.4]]]),
                ('classes',['truck']),('arrival_direction',[0,0]),
                ('arrival_direction',[True,.5]),('arrival_direction',[float('nan'),.5]),
                ('arrival_direction',[-1,-.5]),('arrival_min_progress',.5),
                ('arrival_min_progress',float('nan')),('arrival_min_progress',True)):
            with self.subTest(field=name,value=value):
                self.setUp();setattr(self.directional,name,value)
                self.assertIsNone(self.attempt())
                self.assertEqual({'1'},self.memory._visits[self.ident].track_ids)


class WatcherContinuationTests(unittest.TestCase):
    def setUp(self):
        self.now=100.
        clock=mock.patch.object(watch.time,'monotonic',side_effect=lambda:self.now);clock.start();self.addCleanup(clock.stop)
        notice=mock.patch.object(watch,'notify');notice.start();self.addCleanup(notice.stop)
        self.obj=watcher(zones_path=Path(__file__).resolve().parents[1]/'config/zones.json')
        self.obj._pending_rearm=False;self.obj._queue_snapshot=mock.Mock(return_value=None)
        next(z for z in self.obj.zones if z.name=='Top road arrival').ai_confirmation_window_s=1.5
        self.tracker=ObjectVisitTracker()

    def step(self,at,objects):
        self.now=100+at
        events,seen=self.tracker.update(objects,self.obj.zones,640,360,self.now)
        self.obj.visits.observe(self.now,self.obj.zones,{'available':True,'state':'ready','age_s':0.,'detections':seen})
        for kind,reason,event in events:
            if self.obj.raise_alarm(kind,reason,details={'source':'ai',**self.tracker.event_context(event,self.obj.zones)}):
                self.tracker.mark_emitted(event)

    @staticmethod
    def car(box,confidence=.83):return {'label':'car','confidence':confidence,'bbox':box}

    def approach(self):
        # Approximate original video overlay boxes/rounded confidence; synthetic
        # sensitivity fixture, not exact original model-output provenance.
        for at,objects in [(1.5,[self.car([360,100,425,134],.70)]),
                           (2,[self.car([390,102,453,139],.548)]),(2.5,[]),
                           (3,[self.car([434,112,487,149],.57)])]:self.step(at,objects)

    def stop(self):
        for at in (3.5,4,4.5,5,5.5,6,6.5,7,7.5,8):self.step(at,[])
        # Exact normalized stop-line detections, relative timing shifted only.
        boxes=[(.72869,.40811,.82363,.55706),(.72828,.40756,.82425,.55797),
               (.72830,.40942,.82378,.55808),(.72655,.41133,.82383,.55711)]
        for at,box in zip((8.724,9.294,9.9,10.451),boxes):
            self.step(at,[self.car([v*d for v,d in zip(box,[640,360,640,360])])])

    def test_same_unanswered_approach_keeps_alarm_id_age_and_escalation(self):
        self.approach();ident=self.obj.alarm_id;since=self.obj.alarm_since
        self.stop()
        self.assertEqual(ident,self.obj.alarm_id)
        self.assertEqual(since,self.obj.alarm_since)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertEqual('Entrance waiting',self.obj.alarm_details['zone_name'])

    def test_ack_during_gap_does_not_extend_identity_or_consume_a_later_car(self):
        self.approach();old=self.obj.alarm_details['visit_id']
        self.now=103.2;self.assertTrue(self.obj.acknowledge())
        self.stop()
        for at in (11,11.5,12,12.5,13,13.5,14):
            self.step(at,[self.car([.7283*640,.40942*360,.82378*640,.55808*360])])
        self.assertEqual(2,self.obj._alarm_sequence)
        self.assertNotEqual(old,self.obj.alarm_details['visit_id'])

    def test_co_visible_follower_cannot_be_consumed_after_both_tracks_expire(self):
        self.approach();old=self.obj.alarm_details['visit_id']
        self.step(3.2,[self.car([434,112,487,149]),self.car([360,90,414,131])])
        self.stop()
        self.assertEqual(2,self.obj._alarm_sequence)
        self.assertNotEqual(old,self.obj.alarm_details['visit_id'])


if __name__=='__main__':unittest.main()
