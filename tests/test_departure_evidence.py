"""Departure history needs real visit evidence and a turn needs fresh motion.

These use the PTZ 2 polygons with synthetic boxes; they test policy, not model
recall. The one-frame shift is 9.6 by 1.8 pixels in the 640x360 source.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from object_tracking import ObjectVisitTracker
from region_geometry import point_in_polygon
from visit_memory import VisitMemory


def zones():
    data = json.loads((ROOT / 'config/views/ptz2/zones.json').read_text(encoding='utf-8-sig'))
    result = []
    for row in data['zones']:
        if row['name'] not in ('Exit approach', 'Exit barrier', 'Top road first sighting'):
            continue
        z = SimpleNamespace(**row, ai_enabled=True)
        z.contains_point = lambda x, y, polygon=row['polygon']: point_in_polygon(x, y, polygon)
        result.append(z)
    return result


def car(x, y, confidence=.7):
    return dict(label='car', confidence=confidence,
                bbox=[(x-.04)*640, (y-.06)*360, (x+.04)*640, y*360])


def path(points):
    result = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        result.extend((x0+(x1-x0)*i/6, y0+(y1-y0)*i/6) for i in range(6))
    return result + [points[-1]]


EXIT_PATH = path([(.30,.86),(.40,.70),(.50,.55),(.56,.45),(.55,.36)])
ARRIVAL_PATH = path([(.45,.32),(.50,.33),(.60,.36),(.70,.40),(.78,.43)])


class DepartureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.zones = zones()
        self.tracker = ObjectVisitTracker()
        self.now = 100.
        self.events = []

    def observe(self, point, *, confidence=.7, step=.25):
        self.now += step
        objects = [] if point is None else [car(*point, confidence)]
        events, observed = self.tracker.update(objects, self.zones, 640, 360, self.now)
        self.events.extend((self.now, event) for event in events
                           if event[1].startswith('Top road first sighting:'))
        return observed

    def exit_and_stop(self):
        for point in EXIT_PATH + [(.55,.36)]*10:
            self.observe(point)
        self.assertEqual([], self.events)
        self.assertIn('campus-exit', self.tracker._tracks[1].groups_seen)

    def test_single_box_jitter_after_exit_does_not_announce_a_new_arrival(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.observe((.55,.36))
        self.assertEqual([], self.events)

    def test_shift_then_stationary_readings_do_not_prove_a_return(self):
        self.exit_and_stop()
        for _ in range(5):
            self.observe((.565,.365))
        self.assertEqual([], self.events)

    def test_real_return_still_alarms_on_second_independent_advance(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.assertEqual([], self.events)
        self.observe((.58,.37))
        self.assertEqual(1, len(self.events))
        self.assertEqual(self.now, self.events[0][0])

    def test_return_then_stop_keeps_confirmed_alarm_evidence(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.observe((.58,.37))
        event = self.events[-1][1]
        details = {'source':'ai', **self.tracker.event_context(event[2], self.zones)}
        memory = VisitMemory()
        decision = memory.consider(details, self.now)
        self.assertFalse(decision.suppress)
        self.assertTrue(memory.emitted(decision.visit_id, 'return-alarm', self.now))
        self.assertTrue(self.tracker.mark_emitted(event[2]))
        self.events.clear()
        for _ in range(40):
            observed = self.observe((.58,.37))
            memory.observe(self.now,self.zones,dict(available=True,state='ready',age_s=0.,detections=observed))
            evidence = memory.evidence_for_alarm('return-alarm', self.now)
            self.assertTrue(evidence['fresh'])
            self.assertTrue(evidence['present'])
            self.assertTrue(evidence['confirmed'])
        self.assertEqual([], self.events)

    def test_later_observed_departure_cancels_return_release(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.observe((.58,.37))
        for _ in range(8):
            self.observe((.58,.37))
        self.events.clear()
        self.observe((.565,.365))
        observed = self.observe((.55,.36))
        self.assertNotIn('Top road first sighting', observed[0]['confirmed_regions'])
        self.assertEqual('departing (came from the exit lane)', observed[0]['region_status']['Top road first sighting'])

    def test_slow_return_at_fast_checks_accumulates_real_progress(self):
        self.exit_and_stop()
        for i in range(1, 11):
            self.observe((.55 + i*.003,.36 + i*.001), step=.1)
        self.assertTrue(self.events, 'Checking more often must not make slow motion impossible to prove')

    def test_new_arrival_keeps_unchanged_first_sighting_timing(self):
        control = ObjectVisitTracker()
        unrestricted = zones()
        for zone in unrestricted:
            if zone.name == 'Top road first sighting':
                zone.ignore_from_routes = []
        for point in ARRIVAL_PATH:
            self.observe(point)
            expected, _ = control.update([car(*point)], unrestricted, 640, 360, self.now)
            expected = [e for e in expected if e[1].startswith('Top road first sighting:')]
            self.assertEqual(bool(expected), any(at == self.now for at, _ in self.events))
        self.assertTrue(self.events)

    def test_low_confidence_exit_touch_cannot_silence_a_stationary_top_arrival(self):
        self.observe((.55,.46), confidence=.3)
        self.observe((.55,.36))
        self.observe((.55,.36))
        self.assertTrue(self.events)
        self.assertEqual(1, len(self.tracker._tracks), 'same identity was followed')

    def test_single_strong_unconfirmed_exit_touch_does_not_prove_departure(self):
        self.observe((.55,.46), confidence=.9)
        self.observe((.55,.36))
        self.observe((.55,.36))
        self.assertTrue(self.events)

    def test_motion_before_an_observation_gap_cannot_complete_a_return(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.observe(None, step=1.6)
        self.observe((.58,.37))
        self.observe((.58,.37))
        self.assertEqual([], self.events)

    def test_repeated_timestamp_cannot_count_as_second_advance(self):
        self.exit_and_stop()
        self.observe((.565,.365))
        self.observe((.58,.37), step=0)
        self.assertEqual([], self.events)

    def test_two_visible_vehicles_keep_separate_route_history(self):
        self.exit_and_stop()
        exit_id = 1
        for point in [(.45,.32),(.47,.325),(.49,.33)]:
            self.now += .25
            events, observed = self.tracker.update([car(.55,.36),car(*point)], self.zones,640,360,self.now)
            self.assertEqual(exit_id, observed[0]['track_id'])
            self.assertNotEqual(exit_id, observed[1]['track_id'])
            self.assertNotIn('campus-exit', self.tracker._tracks[observed[1]['track_id']].groups_seen)
            self.events.extend((self.now,event) for event in events if event[1].startswith('Top road first sighting:'))
        self.assertTrue(self.events)


if __name__ == '__main__':
    unittest.main()
