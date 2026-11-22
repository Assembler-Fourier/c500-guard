"""Person-only recovery from an already-running whole-picture view.

Synthetic frames/observations exercise routing, not a substitute for site recall.
"""
import unittest
from pathlib import Path

from test_watch import watch
from semantic_monitor import SemanticMonitor
from object_tracking import ObjectVisitTracker


def person(box=(20,20,30,60), confidence=.75, label='person'):
    return {'label':label, 'confidence':confidence, 'bbox':list(box)}


class PedestrianContextTests(unittest.TestCase):
    def setUp(self):
        self.zones=[watch.Zone(name='Paved path',rect=(.1,.4,.35,.4),classes=['person'],
            trigger='ai',motion=False,ai_confidence=.6,ai_context=True,
            ai_confirm_observations=2,route_role='pedestrian',route_group='walking-route'),
            watch.Zone(name='Crossing',rect=(.4,.4,.5,.4),classes=['person'],
            trigger='ai',motion=False,ai_confidence=.6,ai_context=True,
            ai_confirm_observations=2,route_role='pedestrian',route_group='walking-route')]
        self.monitor=SemanticMonitor.__new__(SemanticMonitor)
        self.monitor.zones=self.zones;self.monitor.confidence=.6
        self.tracker=ObjectVisitTracker()

    def merged(self, supplemental, primary=()):
        return self.monitor._merge_context_detections(list(primary),supplemental,
                                                      (0,0,100,100),[(0,0,100,100)])

    def step(self, supplemental, now, primary=()):
        return self.tracker.update(self.merged(supplemental,primary),self.zones,100,100,now)

    def test_context_recovers_person_missed_by_focused_view(self):
        p=person()
        self.assertEqual([p],self.merged([p]))
        self.assertEqual([],self.step([p],10.)[0])
        events,_=self.step([p],10.3)
        self.assertEqual(1,len(events))
        evidence=self.tracker.event_context(events[0][2],self.zones)
        self.assertEqual(('person','pedestrian','walking-route'),
                         (evidence['label'],evidence['route_role'],evidence['route_group']))
        self.assertEqual(2,evidence['observations'])

    def test_two_views_of_one_frame_do_not_confirm_twice(self):
        p=person()
        objects=self.merged([p],[p])
        self.assertEqual(1,len(objects))
        self.assertEqual([],self.step([p],10.,[p])[0])

    def test_single_flash_or_intervening_empty_frame_does_not_alert(self):
        self.assertEqual([],self.step([person()],10.)[0])
        self.assertEqual([],self.step([],10.3)[0])
        self.assertEqual([],self.step([person()],10.6)[0])

    def test_original_confidence_floor_retained(self):
        self.assertEqual([],self.merged([person(confidence=.599)]))
        self.assertEqual([person(confidence=.6)],self.merged([person(confidence=.6)]))

    def test_road_vehicle_labels_do_not_become_pedestrians(self):
        for label in ('car','truck','bus','motorcycle','bicycle'):
            with self.subTest(label=label):
                self.assertEqual([],self.merged([person(label=label)]))
                self.assertEqual([],self.step([person(label=label)],10.)[0])

    def test_ground_point_not_box_overlap_defines_path(self):
        # The head overlaps the path but the feet are on unconfigured road.
        self.assertEqual([],self.merged([person((20,20,30,95))]))
        self.assertEqual([],self.merged([person((1,1,8,60))]))

    def test_path_to_crossing_same_visit_does_not_repeat(self):
        p=person((28,20,38,60))
        self.step([p],10.);events,_=self.step([p],10.3)
        self.assertTrue(self.tracker.mark_emitted(events[0][2]))
        for i,x in enumerate((33,38,43,48,53),1):
            events,_=self.step([person((x,20,x+10,60))],10.3+i*.3)
            self.assertEqual([],events)

    def test_two_separate_people_are_not_merged(self):
        a,b=person(),person((60,20,70,60))
        self.step([a,b],10.)
        events,_=self.step([a,b],10.3)
        self.assertEqual(2,len(events))
        self.assertEqual(2,len({self.tracker.event_context(e[2],self.zones)['track_id'] for e in events}))

    def test_rider_and_vehicle_remain_distinct_classes(self):
        p=person();bike=person(label='motorcycle')
        merged=self.merged([p],[bike])
        self.assertEqual({'person','motorcycle'},{o['label'] for o in merged})
        self.step([p],10.,[bike]);events,_=self.step([p],10.3,[bike])
        self.assertEqual(['person'],[self.tracker.event_context(e[2],self.zones)['label'] for e in events])

    def test_default_non_opted_region_remains_unchanged(self):
        for zone in self.zones:zone.ai_context=False
        self.assertEqual([],self.merged([person()]))


class SitePedestrianCoverageTests(unittest.TestCase):
    def setUp(self):
        zones=watch.load_zones(Path(__file__).resolve().parents[1]/'config/views/ptz2/zones.json')
        self.zones=[z for z in zones if z.route_role=='pedestrian']
        self.crossing=next(z for z in self.zones if z.name=='Pedestrian crossing')

    def test_known_crossing_points_include_foreground_road_not_vehicle_arrival(self):
        for point in ((282,220),(288,230),(107,249),(107,252),(276.49,191.16),(329.57,185.22)):
            with self.subTest(point=point):
                self.assertTrue(self.crossing.contains_point(point[0]/640,point[1]/360))
        for point in ((173,231),(282,147),(150,280),(375,135),(470,225)):
            with self.subTest(point=point):
                self.assertFalse(self.crossing.contains_point(point[0]/640,point[1]/360))

    def test_site_policy_is_two_person_observations_and_existing_floor(self):
        self.assertEqual(2,len(self.zones))
        for zone in self.zones:
            self.assertEqual(['person'],zone.classes)
            self.assertEqual(.6,zone.ai_confidence)
            self.assertEqual(2,zone.ai_confirm_observations)
            self.assertTrue(zone.ai_context)
            self.assertEqual('campus-pedestrian',zone.route_group)
            self.assertIsNone(zone.arrival_direction)


if __name__=='__main__':unittest.main()
