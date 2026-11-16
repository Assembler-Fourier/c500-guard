"""Strong stop-line policy replay; no model inference, alarm outputs or live state."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from object_tracking import ObjectVisitTracker
from region_geometry import region_contains
from test_entrance_waiting_core import BUS, DEPARTURES, observations
from test_tracking_reversal import PARKING
import test_visit_rearm_incidents as visit_cases
import test_unanswered_continuation as continuation_cases

# Exact normalized boxes/confidences and relative source timestamps from the
# 23:58 truck. The absent result used in the baseline test is explicitly a
# synthetic missing observation; retained metadata does not save empty results.
TRUCK = [
    (0., .6446, (.70173, .17322, .84081, .52567)),
    (.472, .6390, (.70109, .17556, .84098, .53356)),
    (2.521, .6069, (.71142, .21514, .83989, .51622)),
    (3.089, .7149, (.70988, .21475, .83947, .50686)),
    (3.676, .7059, (.70897, .21536, .83731, .50547)),
]


def object_at(x=.77, y=.52, confidence=.7, label='car', height=.13):
    return dict(label=label, confidence=confidence,
                bbox=[(x-.05)*640, (y-height)*360, (x+.05)*640, y*360])


class StrongStoplineTests(unittest.TestCase):
    def setUp(self):
        self.zones = []
        for row in json.loads((ROOT/'config/views/ptz2/zones.json').read_text())['zones']:
            # Fast confirmation belongs to the narrow stop line. Top first
            # sighting independently warns on any upper-road vehicle now.
            if row['name'] != 'Entrance waiting':
                continue
            z = SimpleNamespace(**row, ai_enabled=row.get('trigger') == 'ai')
            z.contains_point = lambda x,y,p=row['polygon'],e=row.get('exclude_polygons',[]): region_contains(x,y,p,e)
            self.zones.append(z)
        self.waiting = next(z for z in self.zones if z.name == 'Entrance waiting')
        self.waiting.ai_fast_confirm_confidence = .60
        self.tracker = ObjectVisitTracker()
        self.events = []

    def step(self, at, objects):
        found, observed = self.tracker.update(objects, self.zones, 640, 360, 100+at)
        accepted = []
        for event in found:
            proof = self.tracker.event_context(event[2], self.zones)
            accepted.append((at, proof, event[1]))
            self.assertTrue(self.tracker.mark_emitted(event[2]))
        self.events.extend(accepted)
        return accepted, observed

    def recorded(self, rows, label='car', dx=0, dy=0):
        for at, confidence, box in rows:
            self.step(at, [dict(label=label, confidence=confidence,
                bbox=[v*d+shift for v,d,shift in zip(box, (640,360,640,360), (dx,dy,dx,dy))])])
        return self.events

    def test_actual_truck_confirms_second_strong_observation_and_does_not_repeat(self):
        self.recorded(TRUCK[:2], 'bus')
        self.assertEqual(1, len(self.events))
        at, proof, reason = self.events[0]
        self.assertEqual(.472, at)
        self.assertEqual('Entrance waiting', proof['zone_name'])
        self.assertEqual(2, proof['observations'])
        self.assertEqual('strong_pair', proof['confirmation_mode'])
        self.assertEqual([100., 100.472], [sample['observed_at'] for sample in proof['strong_confirmation']])
        self.assertEqual([.6446, .6390], [sample['confidence'] for sample in proof['strong_confirmation']])
        self.assertIn('in 2 observations', reason)
        self.assertEqual('unrestricted', proof['direction_state'])
        self.assertTrue(proof['confirmation_required'])
        self.step(1., [])
        self.recorded(TRUCK[2:], 'bus')
        for i in range(30):
            self.step(4+i*.5, [object_at(label='bus', height=.35)])
        self.assertEqual(1, len(self.events))

    def test_disabled_policy_retains_three_reading_confirmation_after_gap(self):
        self.waiting.ai_fast_confirm_confidence = 0
        self.recorded(TRUCK[:2], 'bus')
        self.assertEqual([], self.events)
        self.step(1., [])
        self.recorded(TRUCK[2:], 'bus')
        self.assertEqual([3.676], [event[0] for event in self.events])
        self.assertEqual(3, self.events[0][1]['observations'])

    def test_weaker_recorded_ambulance_keeps_original_third_reading(self):
        self.recorded(BUS[:2], 'bus')
        self.assertEqual([], self.events)
        self.recorded(BUS[2:], 'bus')
        self.assertEqual([1.143], [event[0] for event in self.events])
        self.assertEqual(3, self.events[0][1]['observations'])

    def test_strong_pair_window_is_source_time_bounded_and_distinct(self):
        for delta, expected in ((0., 0), (.472, 1), (1., 1), (1.001, 0), (4., 0)):
            with self.subTest(delta=delta):
                self.setUp()
                self.step(0, [object_at()])
                self.step(delta, [object_at()])
                self.assertEqual(expected, len(self.events))

    def test_shorter_ordinary_window_does_not_undercount_valid_fast_proof(self):
        self.waiting.ai_confirmation_window_s = .4
        self.step(0, [object_at()])
        self.step(.5, [object_at()])
        self.assertEqual(1, len(self.events))
        self.assertEqual(2, self.events[0][1]['observations'])
        self.assertEqual(2, len(self.events[0][1]['strong_confirmation']))

    def test_moving_weak_prefix_does_not_hide_the_accepted_strong_pair_proof(self):
        # Both optional policies can coexist: a preceding weak frame needs an
        # extra ordinary observation, but two later strong eligible frames
        # supply the independent fast proof and must report that exact proof.
        self.waiting.ai_confidence_moving = .3
        self.waiting.arrival_direction = [1,0]
        self.waiting.stationary_confirm_s = 0
        self.step(0, [object_at(x=.74,confidence=.4)])
        self.step(.3, [object_at(x=.76,confidence=.7)])
        self.assertEqual([],self.events)
        self.step(.6, [object_at(x=.78,confidence=.7)])
        self.assertEqual(1,len(self.events))
        self.assertEqual(2,self.events[0][1]['observations'])
        self.assertEqual('strong_pair',self.events[0][1]['confirmation_mode'])
        self.assertEqual([100.3,100.6],
                         [sample['observed_at'] for sample in self.events[0][1]['strong_confirmation']])

    def test_missing_result_or_same_track_absent_clears_fast_candidate(self):
        for intervening in ([], [object_at(.2, .2)]):
            with self.subTest(intervening=intervening):
                self.setUp()
                self.step(0, [object_at()])
                self.step(.2, intervening)
                self.step(.4, [object_at()])
                self.assertEqual([], self.events)
                self.step(.6, [object_at()])
                self.assertEqual([.6], [event[0] for event in self.events])

    def test_weak_observation_keeps_only_ordinary_three_reading_path(self):
        for weak in (.59, .499):
            with self.subTest(weak=weak):
                self.setUp()
                self.step(0, [object_at()])
                self.step(.2, [object_at(confidence=weak)])
                self.assertEqual([], self.events)
                self.step(.4, [object_at()])
                if weak >= .5:
                    self.assertEqual(1, len(self.events))
                    self.assertEqual(3, self.events[0][1]['observations'])
                    self.assertIn('in 3 observations', self.events[0][2])
                else:
                    self.assertEqual([], self.events)
                    self.step(.6, [object_at()])
                    self.assertEqual([.6], [event[0] for event in self.events])

    def test_boundary_straddles_and_visible_outside_do_not_accumulate(self):
        for outside in ((.75, .479), (.77, .63)):
            with self.subTest(outside=outside):
                self.setUp()
                for i in range(20):
                    self.step(i*.2, [object_at(.75, .481) if i%2 == 0 else object_at(*outside)])
                self.assertEqual([], self.events)

    def test_static_vehicle_in_narrow_waiting_core_is_intentionally_eligible(self):
        # Existing unrestricted stopping policy also accepts an already stopped
        # vehicle. Height is not a special exception or a reason to delay it.
        for label, height in (('car', .10), ('truck', .35), ('bus', .30)):
            with self.subTest(label=label):
                self.setUp()
                self.step(0, [object_at(label=label, height=height)])
                self.step(.5, [object_at(label=label, height=height)])
                self.assertEqual([.5], [event[0] for event in self.events])

    def test_parked_jitter_and_broad_tracking_only_points_stay_silent(self):
        for point in ((.77, .63), (.70, .44), (.75, .46), (.8373, .4735)):
            with self.subTest(point=point):
                self.setUp()
                for i in range(30):
                    self.step(i*.3, [object_at(point[0]+(.001 if i%2 else -.001), point[1])])
                self.assertEqual([], self.events)

    def test_recorded_departures_and_parking_remain_silent_with_alignment_error(self):
        for rows in [PARKING] + [observations(path) for path in DEPARTURES]:
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    with self.subTest(first=rows[0], dx=dx, dy=dy):
                        self.setUp()
                        self.assertEqual([], self.recorded(rows, dx=dx, dy=dy))

    def test_direction_and_reversal_eligibility_are_not_bypassed(self):
        self.waiting.arrival_direction = [1, 0]
        self.waiting.stationary_confirm_s = 0
        for i, x in enumerate([.80, .78, .76, .78, .78, .78, .78]):
            self.step(i*.15, [object_at(x=x)])
        self.assertEqual([], self.events)

    def test_invalid_optin_advance_stage_and_missing_core_cannot_enable_fast_path(self):
        for value in (0, .59, True, float('nan'), float('inf'), '0.60'):
            with self.subTest(value=value):
                self.setUp()
                self.waiting.ai_fast_confirm_confidence = value
                self.recorded(TRUCK[:2], 'bus')
                self.assertEqual([], self.events)
        for field, value in (('alert_stage', 'advance'), ('confirmation_polygon', None)):
            with self.subTest(field=field):
                self.setUp()
                setattr(self.waiting, field, value)
                self.recorded(TRUCK[:2], 'bus')
                self.assertEqual([], self.events)

    def test_different_tracks_cannot_pool_strong_observations(self):
        # Width/height ratio rejects association, although both ground points
        # lie in the core. Two cars must not complete each other's proof.
        self.step(0, [object_at(x=.70, height=.04)])
        self.step(.4, [object_at(x=.82, height=.30)])
        self.assertEqual([], self.events)
        self.assertEqual(2, self.tracker.track_count)


class StrongStoplineVisitTests(visit_cases.ConfiguredVisitRearmTests):
    """Run real Watcher/VisitMemory/ACK scenarios with the fast option enabled."""
    def setUp(self):
        super().setUp()
        next(z for z in self.obj.zones if z.name == 'Entrance waiting').ai_fast_confirm_confidence = .60

    def test_actual_truck_fast_proof_reaches_fake_alarm_then_ack_stays_quiet(self):
        for index, (_, confidence, box) in enumerate(TRUCK[:2]):
            self.deliver([dict(label='bus', confidence=confidence,
                bbox=[v*d for v,d in zip(box, (640,360,640,360))])],
                after=0. if index == 0 else .472)
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, self.obj.alarm_details['observations'])
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertTrue(self.obj.acknowledge())
        _, confidence, box = TRUCK[1]
        vehicle = dict(label='bus', confidence=confidence,
                       bbox=[v*d for v,d in zip(box, (640,360,640,360))])
        self.hold([vehicle], seconds=15.)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)


class StrongStoplineContinuationTests(continuation_cases.WatcherContinuationTests):
    """Fast proof must retain the same unanswered approach, never merge a queue."""
    def setUp(self):
        super().setUp()
        next(z for z in self.obj.zones if z.name == 'Entrance waiting').ai_fast_confirm_confidence = .60


class StrongProofValidationTests(unittest.TestCase):
    def setUp(self):
        self.case = continuation_cases.UnansweredContinuationTests()
        self.case.setUp()
        barrier = self.case.zones[1]
        barrier.ai_confirm_observations = 3
        barrier.ai_fast_confirm_confidence = .60
        barrier.arrival_direction = None
        barrier.confirmation_polygon = [[.60,.45],[.80,.45],[.80,.60],[.60,.60]]
        barrier.classes = ['car', 'truck', 'bus']
        self.details = {**self.case.details, 'direction_state': 'unrestricted',
            'confirmation_mode': 'strong_pair', 'strong_confirmation': [
                {'observed_at':106.3,'confidence':.8,'ground_point':[.675,.495],'direction_state':'unrestricted'},
                {'observed_at':106.8,'confidence':.8,'ground_point':[.68,.50],'direction_state':'unrestricted'}]}

    def test_complete_strong_pair_can_continue_the_same_unanswered_visit(self):
        decision = self.case.attempt(self.details)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.suppress)
        self.assertEqual(self.case.ident, decision.visit_id)

    def test_generic_count_two_or_partial_proof_cannot_continue(self):
        for update in ({'strong_confirmation':None}, {'strong_confirmation':[]},
                       {'strong_confirmation':[self.details['strong_confirmation'][0]]},
                       {'confirmation_mode':'ordinary'}, {'observations':True}, {'observations':1}):
            with self.subTest(update=update):
                self.assertIsNone(self.case.attempt({**self.details, **update}))
                self.assertEqual({'1'}, self.case.memory._visits[self.case.ident].track_ids)

    def test_each_frame_requires_finite_fresh_strong_confidence_and_current_geometry(self):
        for index in (0, 1):
            for field, value in (
                ('observed_at', float('nan')), ('observed_at', float('inf')),
                ('observed_at', True), ('observed_at', 103.8), ('observed_at', 107.),
                ('confidence', .5999), ('confidence', 1.01), ('confidence', float('nan')),
                ('confidence', True), ('ground_point', [.59,.50]),
                ('ground_point', [.68,.44]), ('ground_point', [.68,float('nan')]),
                ('ground_point', [.68,True]), ('ground_point', [.68]),
                ('direction_state', 'away'), ('direction_state', 'toward')):
                with self.subTest(index=index, field=field, value=value):
                    details = copy.deepcopy(self.details)
                    details['strong_confirmation'][index][field] = value
                    self.assertIsNone(self.case.attempt(details))

    def test_pair_must_match_current_track_observation_and_not_bridge_stale_interval(self):
        for field, value in (('observed_at', 106.7), ('confidence', .81),
                             ('first_observed_at', 106.4), ('direction_state', 'toward'),
                             ('bbox_normalized', [.62,.32,.74,.501])):
            with self.subTest(field=field):
                self.assertIsNone(self.case.attempt({**self.details, field:value}))
        for first, last in ((106.8,106.8), (106.9,106.8), (105.799,106.8)):
            with self.subTest(first=first,last=last):
                details = copy.deepcopy(self.details)
                details['strong_confirmation'][0]['observed_at'] = first
                details['strong_confirmation'][1]['observed_at'] = last
                self.assertIsNone(self.case.attempt(details))

    def test_live_configuration_must_still_allow_both_strong_samples(self):
        for field, value in (('ai_fast_confirm_confidence',0), ('ai_fast_confirm_confidence',.9),
                             ('confirmation_polygon',None), ('classes',['truck']),
                             ('contains_point',lambda x,y:False)):
            with self.subTest(field=field):
                self.setUp()
                setattr(self.case.zones[1], field, value)
                self.assertIsNone(self.case.attempt(self.details))

    def test_ack_queue_and_proven_clear_still_block_strong_continuation(self):
        for state in ('acknowledged','queue','clear'):
            with self.subTest(state=state):
                self.setUp()
                if state == 'acknowledged':
                    self.case.memory.acknowledge('a',106.7)
                elif state == 'queue':
                    self.case.memory._visits[self.case.ident].separate_tracks.add('other-car')
                else:
                    self.details['first_observed_at'] = 106.5
                    self.details['strong_confirmation'][0]['observed_at'] = 106.5
                self.assertIsNone(self.case.attempt(self.details))


if __name__ == '__main__':
    unittest.main()
