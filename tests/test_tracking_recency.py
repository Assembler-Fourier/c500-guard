"""Recorded reflection/vehicle identity competition and two-car controls.

The fixture preserves saved relative detector timestamps, boxes and confidence;
it tests association and route policy, not image detector accuracy. The recording
independently shows one incoming vehicle and a transient empty-road reflection.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from object_tracking import ObjectVisitTracker
from region_geometry import region_contains

RECORDED = [
    (0.0, [(0.7756, (0.44808, 0.24275, 0.53941, 0.33942))]),
    (0.503, [(0.8246, (0.47341, 0.25564, 0.60525, 0.35911))]),
    (1.095, [(0.8583, (0.561, 0.27103, 0.66189, 0.38011))]),
    (1.69, [(0.8477, (0.61122, 0.28358, 0.70755, 0.398))]),
    (2.198, [(0.7866, (0.64327, 0.29656, 0.73767, 0.41767)), (0.6424, (0.67702, 0.42981, 0.86836, 0.62611))]),
    (2.805, [(0.5366, (0.67506, 0.33003, 0.76825, 0.4375))]),
    (3.897, [(0.709, (0.71073, 0.34475, 0.803, 0.47733))]),
    (4.505, [(0.7656, (0.72669, 0.35503, 0.80933, 0.49239))]),
    (5.695, [(0.6274, (0.67573, 0.39619, 0.81184, 0.5485))]),
    (6.186, [(0.8017, (0.67202, 0.39231, 0.81028, 0.56206))]),
    (6.69, [(0.8648, (0.66317, 0.37914, 0.80723, 0.56558))]),
    (7.307, [(0.728, (0.70692, 0.43786, 0.80683, 0.57578))]),
    (7.802, [(0.6685, (0.70073, 0.43886, 0.80209, 0.61522))]),
    (8.4, [(0.5743, (0.72078, 0.47486, 0.79664, 0.63283))]),
    (8.896, [(0.7894, (0.68291, 0.48942, 0.78953, 0.65428))]),
    (9.491, [(0.8154, (0.67275, 0.52497, 0.77722, 0.69975))]),
    (10.098, [(0.8852, (0.63619, 0.59167, 0.76181, 0.75781))]),
    (10.688, [(0.8374, (0.58916, 0.61325, 0.74111, 0.83986))]),
    (11.186, [(0.8667, (0.55425, 0.66967, 0.72492, 0.92594))]),
    (11.709, [(0.8854, (0.44717, 0.72625, 0.69503, 0.97586))]),
]


def road():
    return SimpleNamespace(name='Road',rect=(0,0,1,1),ai_enabled=True,
        classes=['car','truck','bus'],severity='main',route_group='road',
        ai_confirm_observations=2,ai_confidence=.5)


def car(x, *, confidence=.9):
    return {'label':'car','confidence':confidence,
            'bbox':[(x-.05)*640,.4*360,(x+.05)*640,.5*360]}


class RecencyAssociationTests(unittest.TestCase):
    def test_recorded_reflection_cannot_take_recent_vehicle_identity(self):
        zones=[]
        for row in json.loads((ROOT/'config/views/ptz2/zones.json').read_text())['zones']:
            z=SimpleNamespace(**row,ai_enabled=True)
            z.contains_point=lambda x,y,p=row['polygon']:region_contains(x,y,p,[])
            zones.append(z)
        tracker=ObjectVisitTracker()
        alarms=[]
        actual_id=None
        phantom_id=None
        self.assertEqual(20,len(RECORDED))
        for at,objects in RECORDED:
            detections=[{'label':'car','confidence':confidence,
                'bbox':[v*d for v,d in zip(box,[640,360,640,360])]}
                for confidence,box in objects]
            events,observed=tracker.update(detections,zones,640,360,100+at)
            if actual_id is None:
                actual_id=observed[0]['track_id']
            if len(observed)==2:
                self.assertEqual(actual_id,observed[0]['track_id'])
                phantom_id=observed[1]['track_id']
                self.assertNotEqual(actual_id,phantom_id)
            if at>=5.695:
                self.assertEqual(actual_id,observed[0]['track_id'])
            for event in events:
                alarms.append(tracker.event_context(event[2],zones))
                self.assertTrue(tracker.mark_emitted(event[2]))
        self.assertIsNotNone(phantom_id)
        self.assertEqual(1,len(alarms))
        self.assertEqual('Top road arrival',alarms[0]['zone_name'])

    def test_two_nearby_visible_cars_keep_separate_visits(self):
        tracker=ObjectVisitTracker()
        zones=[road()]
        _,first=tracker.update([car(.4),car(.53)],zones,640,360,100)
        ids=[o['track_id'] for o in first]
        events,_=tracker.update([car(.53),car(.4)],zones,640,360,100.5)
        self.assertEqual(2,len(events))
        for event in events:self.assertTrue(tracker.mark_emitted(event[2]))
        for index in range(10):
            events,observed=tracker.update([car(.4),car(.53)],zones,640,360,101+index*.5)
            self.assertEqual([],events)
            self.assertEqual(ids,[o['track_id'] for o in observed])

    def test_occluded_car_recovers_beside_another_recent_car(self):
        tracker=ObjectVisitTracker()
        zones=[road()]
        tracker.update([car(.4),car(.58)],zones,640,360,100)
        events,first=tracker.update([car(.4),car(.58)],zones,640,360,100.5)
        original=[o['track_id'] for o in first]
        for event in events:tracker.mark_emitted(event[2])
        for index in range(1,7):
            tracker.update([car(.58)],zones,640,360,100.5+index*.5)
        events,observed=tracker.update([car(.4),car(.58)],zones,640,360,104)
        self.assertEqual([],events)
        self.assertEqual(original,[o['track_id'] for o in observed])
        self.assertEqual(2,len(set(o['track_id'] for o in observed)))

    def test_new_nearby_vehicle_is_not_muted_by_occluded_acknowledged_vehicle(self):
        tracker=ObjectVisitTracker()
        zones=[road()]
        tracker.update([car(.4)],zones,640,360,100)
        events,first=tracker.update([car(.4)],zones,640,360,100.5)
        original=first[0]['track_id']
        tracker.mark_emitted(events[0][2])
        tracker.update([],zones,640,360,101)
        tracker.update([],zones,640,360,102)
        # Both cars are now observed, so an old visit cannot absorb the new one.
        tracker.update([car(.55),car(.4)],zones,640,360,103)
        events,observed=tracker.update([car(.55),car(.4)],zones,640,360,103.5)
        self.assertEqual(1,len(events))
        evidence=tracker.event_context(events[0][2],zones)
        self.assertNotEqual(original,evidence['track_id'])
        self.assertEqual(original,observed[1]['track_id'])

    def test_strong_old_match_still_beats_a_weak_recent_candidate(self):
        tracker=ObjectVisitTracker()
        zones=[road()]
        _,objects=tracker.update([car(.4),car(.6)],zones,640,360,100)
        original=objects[0]['track_id']
        tracker.update([car(.4),car(.6)],zones,640,360,100.5)
        for index in range(1,7):
            tracker.update([car(.55)],zones,640,360,100.5+index*.5)
        # Only the previously hidden car returns. Recency is a weight, not a
        # rule forcing it onto the weakly compatible nearby fresh identity.
        _,observed=tracker.update([car(.4)],zones,640,360,104)
        self.assertEqual(original,observed[0]['track_id'])


if __name__=='__main__':
    unittest.main()
