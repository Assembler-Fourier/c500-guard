"""Short same-car ID break versus a separate follower; isolated fake alarms."""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

from visit_memory import VisitMemory
from test_watch import watch, watcher

OLD = [.456078125,.253305556,.56284375,.344638889]
NEW = [.558016,.266917,.652578,.372333]


def fixtures():
    first = SimpleNamespace(name='First sighting', route_group='entrance', route_role='entrance',
        alert_stage='advance', ai_confirmation_window_s=1.5, ai_context=True,
        arrival_direction=None, arrival_min_progress=.012, ai_confidence=.35,
        ai_confirm_observations=2, ai_enabled=True, motion_active=False, light_active=False,
        polygon=[[.4,.2],[.85,.2],[.85,.5],[.4,.5]], rect=(.4,.2,.45,.3),
        exclude_polygons=[], classes=['car','truck','bus'], confirmation_polygon=None,
        contains_point=lambda x,y:.4<=x<=.85 and .2<=y<=.5)
    directional = copy.deepcopy(first)
    directional.name = 'Directional approach'
    directional.arrival_direction = [1.,.5]
    previous = dict(source='ai', route_group='entrance', route_role='entrance',
        zone_name='First sighting', alert_stage='advance', track_id=2, label='car',
        confidence=.7466, observations=2, observed_at=100., first_observed_at=99.3,
        first_ground=[.4678,.3262], bbox_normalized=OLD, direction_state='unrestricted')
    details = {**previous, 'track_id':5, 'confidence':.8451, 'observed_at':100.9074,
               'first_observed_at':100.4058, 'first_ground':[.5779,.3561],
               'bbox_normalized':NEW}
    return [first,directional], previous, details


def detection(track, box):
    return dict(track_id=track,label='car',bbox_normalized=box,
                regions=['First sighting'],confirmed_regions=['First sighting'])


def observe(memory, zones, now, *objects):
    memory.observe(now,zones,{'available':True,'state':'ready','age_s':0.,'detections':list(objects)})


class SameAdvanceContinuationTests(unittest.TestCase):
    def setUp(self):
        self.zones,self.previous,self.details=fixtures()
        self.memory=VisitMemory()
        self.ident=self.memory.consider(self.previous,100.).visit_id
        self.memory.emitted(self.ident,'alarm-1',100.)
        observe(self.memory,self.zones,100.,detection(2,OLD))
        observe(self.memory,self.zones,100.4058,detection(5,NEW))
        observe(self.memory,self.zones,100.9074,detection(5,NEW))

    def attempt(self,details=None):
        return self.memory.continue_unanswered(self.ident,'alarm-1',self.previous,
            self.details if details is None else details,self.zones,100.9074)

    def test_measured_forward_fragment_keeps_original_unanswered_visit(self):
        result=self.attempt()
        self.assertTrue(result.suppress)
        self.assertEqual(self.ident,result.visit_id)
        self.assertEqual({'2','5'},self.memory._visits[self.ident].track_ids)

    def test_follower_born_behind_old_car_is_distinct_even_after_moving_forward(self):
        details={**self.details,'first_ground':[.48,.33]}
        self.assertIsNone(self.attempt(details))
        decision=self.memory.consider(details,100.9074)
        self.assertFalse(decision.suppress)
        self.assertNotEqual(self.ident,decision.visit_id)

    def test_concurrently_observed_second_car_never_inherits(self):
        # Preserve the queue even if its second car later gets another ID.
        # The later first_seen still follows the old sighting, so this checks
        # concurrent identity memory rather than failing the time-order guard.
        observe(self.memory,self.zones,100.1,detection(2,OLD),detection(9,NEW))
        observe(self.memory,self.zones,100.9074,detection(5,NEW))
        self.assertIn('9',self.memory._visits[self.ident].separate_tracks)
        self.assertIsNone(self.attempt())
        self.assertEqual({'2'},self.memory._visits[self.ident].track_ids)

    def test_acknowledged_or_noncurrent_alarm_does_not_continue(self):
        self.memory.acknowledge('alarm-1',100.8)
        self.assertIsNone(self.attempt())
        self.assertFalse(self.memory.consider(self.details,100.9074).suppress)

    def test_birth_gap_must_be_at_most_three_quarters_second(self):
        self.assertIsNone(self.attempt({**self.details,'first_observed_at':100.75001}))
        self.assertIsNotNone(self.attempt({**self.details,'first_observed_at':100.75}))

    def test_different_advance_zone_does_not_gain_general_route_cooldown(self):
        details={**self.details,'zone_name':'Directional approach'}
        self.assertIsNone(self.attempt(details))

    def test_parallel_vehicle_offset_from_old_trajectory_is_distinct(self):
        details={**self.details,'first_ground':[.59,.27],
                 'bbox_normalized':[.56,.21,.66,.31]}
        self.assertIsNone(self.attempt(details))
        self.assertFalse(self.memory.consider(details,100.9074).suppress)

    def test_follower_after_observed_retirement_cannot_inherit(self):
        self.memory._visits[self.ident].retired_at=100.2
        self.assertIsNone(self.attempt())
        decision=self.memory.consider(self.details,100.9074)
        self.assertFalse(decision.suppress)
        self.assertNotEqual(self.ident,decision.visit_id)

    def test_missing_invalid_stationary_or_outside_origin_rejects_alias(self):
        for origin in (None,[],[True,.35],[float('nan'),.35],[.95,.35],
                       [(OLD[0]+OLD[2])/2,OLD[3]],[.48,.33]):
            with self.subTest(origin=origin):
                self.assertIsNone(self.attempt({**self.details,'first_ground':origin}))
                self.assertEqual({'2'},self.memory._visits[self.ident].track_ids)

    def test_existing_opt_in_direction_and_confirmation_guards_still_apply(self):
        for action in ('context','window','direction','opposed','confidence','count'):
            with self.subTest(action=action):
                self.setUp()
                if action=='context':self.zones[0].ai_context=False
                elif action=='window':self.zones[0].ai_confirmation_window_s=0
                elif action=='direction':self.zones.pop()
                elif action=='opposed':self.zones[1].arrival_direction=[-1.,-.5]
                elif action=='confidence':self.details['confidence']=.34
                else:self.details['observations']=1
                self.assertIsNone(self.attempt())


class SameAdvanceWatcherTests(unittest.TestCase):
    def test_unanswered_fragment_does_not_restart_audio_or_ten_second_ack(self):
        zones,previous,details=fixtures()
        now=100.
        with mock.patch.object(watch.time,'monotonic',side_effect=lambda:now), \
             mock.patch.object(watch,'notify'):
            obj=watcher()
            obj.zones=zones
            obj._pending_rearm=False
            obj._queue_snapshot=mock.Mock(return_value=None)
            obj._consider_already_opening=mock.Mock()
            obj.loudness=mock.Mock(kind=None)
            obj.learning=obj.recorder=None
            observe(obj.visits,zones,now,detection(2,OLD))
            self.assertTrue(obj.raise_alarm('main','first',details=previous))
            ident,since=obj.alarm_id,obj.alarm_since
            now=100.4058
            observe(obj.visits,zones,now,detection(5,NEW))
            now=100.9074
            observe(obj.visits,zones,now,detection(5,NEW))
            self.assertTrue(obj.raise_alarm('main','same car fragment',details=details))
            self.assertEqual(ident,obj.alarm_id)
            self.assertEqual(since,obj.alarm_since)
            self.assertEqual(1,obj._alarm_sequence)
            self.assertEqual(1,obj.loudness.begin.call_count)
            self.assertTrue(obj.acknowledge())
            self.assertAlmostEqual(10.,obj._ack_until-now)


if __name__=='__main__':
    unittest.main()
