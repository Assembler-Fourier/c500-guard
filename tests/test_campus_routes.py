"""Synthetic regression trajectories against the exact configured site polygons.

These points test application rules and region geometry. They are invented
detector outputs, not recorded traffic, and do not validate real-world model
accuracy, camera coverage, parking intent, or physical route calibration.
"""
# Coordinates below are the September site positions mapped into the keyboard
# capture picture (13 Nov 2026) with the same lens-corrected fit as the zones.
import json
from pathlib import Path
import unittest

from test_watch import watch
from object_tracking import ObjectVisitTracker
from region_geometry import point_in_polygon
from zone_editor import validate_config


EXIT_ROUTE = [(46,321),(63,305),(82,286),(104,263),(127,242),
              (148,225),(172,211),(193,196),(214,178),(236,165)]
ENTRANCE_ROUTE = [(483,56),(474,63),(465,71),(453,80),
                  (441,87),(427,92),(413,95)]
CHARGING_ROUTE = [(514,108),(509,122),(496,138),
                  (482,153),(468,164)]


def car_at(x,y):
    return {'label':'car','confidence':.9,'bbox':[x-14,y-20,x+14,y]}


class ConfiguredCampusRoutesTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'config' / 'views' / 'anpr' / 'zones.json'
        config = validate_config(json.loads(source.read_text(encoding='utf-8')))
        self.zones = [watch.Zone(**item) for item in config['zones']]
        self.by_name = {zone.name:zone for zone in self.zones}
        self.tracker = ObjectVisitTracker()
        self.now = 100.

    def step(self, point):
        self.now += .5
        return self.tracker.update([car_at(*point)], self.zones, 640, 360, self.now)

    def accepted_events(self, points):
        accepted, evidence, tracks = [], [], set()
        for point in points:
            events, objects = self.step(point)
            self.assertEqual(1,len(objects))
            tracks.add(objects[0]['track_id'])
            for event in events:
                evidence.append(self.tracker.event_context(event[2],self.zones))
                self.assertTrue(self.tracker.mark_emitted(event[2]))
                accepted.append(event)
        self.assertEqual(1,len(tracks), 'Synthetic adjacent points must retain one track')
        return accepted,evidence

    def test_charging_only_curve_is_tracked_but_silent(self):
        approach = self.by_name['Top road arrival']
        for x,y in CHARGING_ROUTE:
            self.assertTrue(approach.contains_point(x/640,y/360))
            self.assertFalse(point_in_polygon(x/640,y/360,approach.confirmation_polygon))
        for point in CHARGING_ROUTE + [CHARGING_ROUTE[-1]]*12:
            events,objects = self.step(point)
            self.assertEqual([],events)
            self.assertEqual('tracking approach',objects[0]['region_status']['Top road arrival'])

    def test_top_arrival_alerts_once_without_realert_at_entrance_waiting(self):
        accepted,evidence = self.accepted_events(ENTRANCE_ROUTE+[ENTRANCE_ROUTE[-1]]*14)
        self.assertEqual(1,len(accepted))
        self.assertEqual('campus-entrance',evidence[0]['route_group'])
        self.assertEqual('entrance',evidence[0]['route_role'])
        self.assertEqual('Top road arrival',evidence[0]['zone_name'])
        self.assertEqual('advance',evidence[0]['alert_stage'])
        self.assertTrue(evidence[0]['confirmation_required'])
        self.assertGreaterEqual(evidence[0]['observations'],2)

    def test_lower_left_exit_to_barrier_keeps_one_alert_for_visit(self):
        accepted,evidence = self.accepted_events(EXIT_ROUTE+[EXIT_ROUTE[-1]]*14)
        self.assertEqual(1,len(accepted))
        self.assertEqual('Exit approach',evidence[0]['zone_name'])
        self.assertEqual('campus-exit',evidence[0]['route_group'])
        self.assertEqual('exit',evidence[0]['route_role'])
        # This regression covers AI route deduplication; raw light episode
        # correlation is exercised separately by the watcher runtime tests.

    def test_reverse_departures_are_not_ai_arrivals(self):
        for name,path in [('exit',EXIT_ROUTE),('entrance',ENTRANCE_ROUTE)]:
            with self.subTest(route=name):
                self.tracker = ObjectVisitTracker()
                for point in list(reversed(path))+[path[0]]*12:
                    self.assertEqual([],self.step(point)[0])

    def test_first_seen_waiting_at_either_barrier_has_stationary_fallback(self):
        for role,point in [('entrance',(422,85)),('exit',(233,166))]:
            with self.subTest(route=role):
                self.tracker = ObjectVisitTracker()
                accepted,evidence = self.accepted_events([point]*20)
                self.assertEqual(1,len(accepted))
                self.assertEqual(role,evidence[0]['route_role'])
                self.assertEqual('stationary',evidence[0]['direction_state'])

    def test_foreground_parking_at_y180_is_outside_vehicle_regions(self):
        points = [(328,213),(349,213),(371,214),(393,214),(417,214),(442,214),(469,214)]
        for point in points+[points[-1]]*12:
            events,objects = self.step(point)
            self.assertEqual([],events)
            self.assertEqual([],objects[0]['regions'])

    def test_single_gate_observation_then_charging_route_stays_silent(self):
        approach = self.by_name['Top road arrival']
        crossing = [(540,61),(538,76),(532,106)]
        self.assertTrue(point_in_polygon(538/640,76/360,approach.confirmation_polygon))
        self.assertEqual(1,sum(point_in_polygon(x/640,y/360,approach.confirmation_polygon)
                               for x,y in crossing))
        for point in crossing+CHARGING_ROUTE+[CHARGING_ROUTE[-1]]*12:
            self.assertEqual([],self.step(point)[0])

    def test_top_advance_can_alert_before_vehicle_later_chooses_charging(self):
        # This is the explicit sensitivity tradeoff: the future parking turn
        # is unknown when the early top-road approach first becomes visible.
        path = [(483,56),(488,64),(501,75),(516,86),(528,96),(519,113)]
        accepted,evidence = self.accepted_events(path+CHARGING_ROUTE+[CHARGING_ROUTE[-1]]*12)
        self.assertEqual(1,len(accepted))
        self.assertEqual('main',accepted[0][0])
        self.assertEqual('Top road arrival',evidence[0]['zone_name'])
        self.assertEqual('advance',evidence[0]['alert_stage'])

    def test_stationary_top_road_object_has_no_advance_fallback(self):
        for _ in range(30):
            self.assertEqual([],self.step((530,92))[0])
        events,_ = self.step((535,99))
        self.assertEqual(1,len(events))
        self.assertEqual('advance',self.tracker.event_context(events[0][2],self.zones)['alert_stage'])


if __name__ == '__main__':
    unittest.main()
