"""Weak reflected road blobs need an entry origin before creating a new visit."""
import copy
import unittest

from test_watch import watch
from object_tracking import ObjectVisitTracker
from zone_editor import validate_config


ENTRY = [[0, .63], [.35, .63], [.35, .82], [1, .82], [1, 1], [0, 1]]
BASE = dict(name='Exit', rect=[.1, .3, .8, .7], trigger='ai', classes=['car', 'truck', 'bus'],
            motion=False, light=False, ai_confidence=.5, ai_confidence_moving=.3,
            arrival_direction=[.35, -1.], arrival_min_progress=.012,
            stationary_confirm_s=0, ai_confirm_observations=2,
            route_group='exit', route_role='exit', ai_moving_entry_polygon=ENTRY)


def car(x, y, confidence=.35, box=None):
    box = box or [x-.07, y-.16, x+.07, y]
    return {'label': 'car', 'confidence': confidence,
            'bbox': [v*scale for v, scale in zip(box, [640, 360, 640, 360])]}


class WeakEntryTests(unittest.TestCase):
    def setUp(self):
        self.zone = watch.Zone(**BASE)
        self.tracker = ObjectVisitTracker()
        self.now = 100.

    def step(self, *objects):
        self.now += .3
        return self.tracker.update(list(objects), [self.zone], 640, 360, self.now)

    def ghost(self):
        # Known incident origin and final two measured boxes/confidences;
        # intermediate cadence/first confidence are synthetic, not a full replay.
        return [car(.5608, .7002),
                car(0, 0, .5342, [.42597, .47811, .61184, .59708]),
                car(0, 0, .3597, [.42925, .470417, .614766, .597306])]

    def test_ghost_single_strong_observation_does_not_bypass_entry_requirement(self):
        for obj in self.ghost():
            events, observed = self.step(obj)
            self.assertEqual([], events)
        self.assertEqual('weak arrival has no entry evidence', observed[0]['region_status']['Exit'])
        self.assertEqual(1, self.tracker.track_count, 'still tracked; identity was not thrown away')
        # Two subsequent normal observations can still establish a real car.
        self.assertEqual([], self.step(car(.52, .58, .7))[0])
        self.assertEqual(1, len(self.step(car(.52, .56, .75))[0]))

    def test_without_opt_in_existing_weak_behavior_is_unchanged(self):
        self.zone.ai_moving_entry_polygon = None
        events=[]
        for obj in self.ghost():
            events += self.step(obj)[0]
        self.assertEqual(1, len(events))

    def test_two_normal_observations_first_seen_mid_lane_still_confirm(self):
        self.assertEqual([], self.step(car(.54, .70, .7))[0])
        self.assertEqual(1, len(self.step(car(.54, .66, .7))[0]))

    def test_weak_true_bottom_and_side_entries_still_confirm(self):
        for path in ([(.55,.96),(.55,.92),(.55,.88)],
                     [(.20,.72),(.23,.68),(.26,.64)]):
            with self.subTest(path=path):
                self.tracker = ObjectVisitTracker()
                events=[]
                for x,y in path:
                    events += self.step(car(x,y))[0]
                self.assertEqual(1, len(events))

    def test_distinct_weak_followers_at_entry_are_not_merged_or_suppressed(self):
        events=[]
        for y in (.96,.92,.88):
            events += self.step(car(.25,y), car(.65,y))[0]
        self.assertEqual(2, len(events))
        self.assertEqual(2, len({self.tracker.event_context(event[2],[self.zone])['track_id'] for event in events}))

    def test_confirmed_unemitted_mid_lane_visit_survives_later_weak_observation(self):
        self.step(car(.54,.70,.7))
        first = self.step(car(.54,.66,.7))[0]
        self.assertEqual(1, len(first))
        following = self.step(car(.54,.63,.35))[0]
        self.assertEqual(first[0][2], following[0][2])
        self.tracker.mark_emitted(following[0][2])
        self.assertEqual([], self.step(car(.54,.60,.35))[0])

    def test_weak_first_seen_mid_lane_deliberately_waits_for_normal_evidence(self):
        for y in (.77,.73,.69,.65,.61):
            self.assertEqual([], self.step(car(.55,y,.4))[0])

    def test_below_moving_floor_does_not_gain_permission_from_entry(self):
        for y in (.96,.92,.88,.84):
            self.assertEqual([], self.step(car(.55,y,.25))[0])


class WeakEntryConfigTests(unittest.TestCase):
    def test_valid_optional_origin_can_extend_outside_alert_rectangle(self):
        result = validate_config({'zones':[copy.deepcopy(BASE)]})
        self.assertEqual(ENTRY, result['zones'][0]['ai_moving_entry_polygon'])
        self.assertEqual(ENTRY, watch.Zone(**result['zones'][0]).ai_moving_entry_polygon)
        legacy=copy.deepcopy(BASE)
        legacy.pop('ai_moving_entry_polygon')
        self.assertIsNone(watch.Zone(**validate_config({'zones':[legacy]})['zones'][0]).ai_moving_entry_polygon)

    def test_invalid_origin_and_missing_weak_rule_are_rejected(self):
        for extra in ({'ai_moving_entry_polygon':[]},
                      {'ai_moving_entry_polygon':[[0,0],[2,0],[1,1]]},
                      {'ai_moving_entry_polygon':[[0,0],[1,1],[0,1],[1,0]]},
                      {'ai_confidence_moving':None}, {'arrival_direction':None}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_config({'zones':[{**copy.deepcopy(BASE),**extra}]})


if __name__ == '__main__':
    unittest.main()
