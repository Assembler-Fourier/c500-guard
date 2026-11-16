"""PTZ geometry regressions; trajectories from saved sights, synthetic boxes.

These replay the route layer, not detector precision. No live state or services.
The full 20:02 recording independently confirms the entrance turn; 20:06 saved
grounds supply a second complete incoming route. Confidence and box size below
are synthetic so they do not claim original per-frame detector provenance.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from object_tracking import ObjectVisitTracker
from region_geometry import region_contains
from zone_editor import validate_config


INCOMING = [(.5101,.3446),(.5685,.3634),(.6205,.3807),(.6468,.3942),
            (.6749,.4119),(.7026,.4207),(.7188,.4391),(.7469,.468),
            (.748,.4901),(.7539,.5113),(.7429,.5368)]
MISSED_INCOMING = [(.4784,.3276),(.5108,.3456),(.5763,.3643),(.6305,.3845),
                   (.6656,.4023),(.7038,.4238),(.7256,.4441),(.7517,.4853),
                   (.7555,.5039),(.7433,.5333),(.7581,.5497),(.7415,.5549),
                   (.7564,.5571),(.7597,.5601),(.7598,.5604)]
EXIT = [(.2573,.9662),(.3094,.9169),(.3602,.8413),(.395,.7788),(.4298,.7142),
        (.4617,.6647),(.4815,.6289),(.5051,.5933),(.5316,.5522),(.5413,.5388),
        (.5525,.5074),(.5532,.4833),(.5543,.4574),(.5452,.4375),(.5216,.4144),
        (.4963,.3905),(.4557,.3616),(.3986,.3605)]


class PtzArrivalRoutesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.config = validate_config(json.loads((self.root/'config/views/ptz2/zones.json').read_text()))
        self.zones = []
        for item in self.config['zones']:
            z = SimpleNamespace(**item,ai_enabled=True)
            z.contains_point = lambda x,y,polygon=z.polygon,exclusions=item.get('exclude_polygons',[]): region_contains(x,y,polygon,exclusions)
            self.zones.append(z)
        self.tracker = ObjectVisitTracker()
        self.now = 100.

    def step(self, point, confidence=.9, label='car'):
        self.now += .55
        x, y = point
        return self.tracker.update([{'label':label,'confidence':confidence,
                                    'bbox':[(x-.04)*640,(y-.08)*360,(x+.04)*640,y*360]}],
                                   self.zones,640,360,self.now)

    def run_path(self, path, confidence=.9, label='car'):
        accepted = []
        for index, point in enumerate(path):
            events, _ = self.step(point,confidence,label)
            for event in events:
                evidence = self.tracker.event_context(event[2],self.zones)
                self.assertTrue(self.tracker.mark_emitted(event[2]))
                accepted.append((index,evidence))
        return accepted

    def test_recorded_incoming_paths_warn_before_barrier(self):
        for path in (INCOMING,MISSED_INCOMING):
            with self.subTest(first=path[0]):
                self.tracker = ObjectVisitTracker()
                accepted = self.run_path(path)
                self.assertEqual(1,len(accepted))
                index,evidence = accepted[0]
                self.assertLessEqual(index,2)
                self.assertEqual('Top road arrival',evidence['zone_name'])
                self.assertEqual('campus-entrance',evidence['route_group'])
                self.assertEqual('toward',evidence['direction_state'])

    def test_recorded_charging_road_departure_is_not_an_entrance_arrival(self):
        # 20:11:55: independently viewed tail lights moving right-to-left/up.
        # Former [-.3,1] waiting direction incorrectly rewarded the leftward leg.
        path = [(.8382,.4804),(.8254,.4734),(.7865,.472),(.7116,.4591),
                (.6706,.4515),(.6207,.443),(.615,.4259),(.6018,.4163)]
        self.assertEqual([], self.run_path(path, confidence=.8))

    def test_same_arrival_does_not_repeat_at_waiting_area_or_while_stopped(self):
        accepted = self.run_path(INCOMING[:9]+[INCOMING[8]]*24+INCOMING[9:])
        self.assertEqual(1,len(accepted))

    def test_reverse_top_road_warns_once_without_claiming_direction(self):
        accepted = self.run_path(list(reversed(INCOMING[:6]))+[INCOMING[0]]*16)
        self.assertEqual(1,len(accepted))
        self.assertEqual(1,accepted[0][0])
        self.assertEqual('Top road first sighting',accepted[0][1]['zone_name'])
        self.assertEqual('unrestricted',accepted[0][1]['direction_state'])
        self.assertEqual(2,accepted[0][1]['observations'])

    def test_stationary_top_road_warns_once_after_two_sightings(self):
        accepted = self.run_path([INCOMING[2]]*20)
        self.assertEqual(1,len(accepted))
        self.assertEqual(1,accepted[0][0])
        self.assertEqual('Top road first sighting',accepted[0][1]['zone_name'])
        self.assertEqual('unrestricted',accepted[0][1]['direction_state'])

    def test_foreground_parking_and_known_bright_road_phantom_remain_outside(self):
        for path in [[(.68+i*.01,.63) for i in range(12)],[(.77,.63)]*20,
                     [(.38,.31),(.40,.31),(.43,.31),(.40,.31),(.38,.31)]]:
            with self.subTest(first=path[0]):
                self.tracker = ObjectVisitTracker()
                self.assertEqual([],self.run_path(path))

    def test_recorded_exit_alerts_once_and_does_not_become_entrance_arrival(self):
        accepted = self.run_path(EXIT)
        self.assertEqual(1,len(accepted))
        self.assertEqual('campus-exit',accepted[0][1]['route_group'])

    def test_low_confidence_and_single_observation_stay_silent(self):
        self.assertEqual([],self.run_path(INCOMING,.349))
        self.tracker = ObjectVisitTracker()
        self.assertEqual([],self.run_path([INCOMING[2]]))

    def test_moderate_top_confidence_only_uses_explicit_first_sighting_policy(self):
        for confidence in (.35,.49,.54):
            with self.subTest(confidence=confidence):
                self.tracker = ObjectVisitTracker()
                accepted = self.run_path(INCOMING[:6],confidence)
                self.assertEqual(1,len(accepted))
                self.assertEqual('Top road first sighting',accepted[0][1]['zone_name'])
                self.assertEqual(2,accepted[0][1]['observations'])
                self.assertEqual('unrestricted',accepted[0][1]['direction_state'])

    def test_second_distinct_arrival_gets_its_own_event(self):
        self.assertEqual(1,len(self.run_path(INCOMING)))
        self.now += 6.
        self.tracker.update([],self.zones,640,360,self.now)
        self.assertEqual(1,len(self.run_path(INCOMING)))

    def test_waiting_fallback_and_thresholds_are_preserved(self):
        for ground in ((.75,.49),(.75,.56),(.7428,.5728)):
            with self.subTest(ground=ground):
                self.tracker = ObjectVisitTracker()
                accepted = self.run_path([ground]*16)
                self.assertEqual(1,len(accepted))
                self.assertEqual('Entrance waiting',accepted[0][1]['zone_name'])
        road = next(z for z in self.zones if z.name=='Top road arrival')
        self.assertEqual(.55,road.ai_confidence)
        self.assertEqual(2,road.ai_confirm_observations)
        self.assertEqual(0,road.stationary_confirm_s)

    def test_waiting_band_extends_past_old_cutoff_but_not_downstream_phantom(self):
        # Three fresh readings after the old .515 cutoff can still confirm;
        # confidence flicker before these must not make a waiting car invisible.
        self.step((.7256,.4441),.59)
        self.step((.7517,.4853),.59)
        self.step((.7555,.5039),.59)
        accepted = self.run_path([(.7433,.5333),(.7581,.5497),(.7415,.5549)])
        self.assertEqual(1,len(accepted))
        self.assertEqual('Entrance waiting',accepted[0][1]['zone_name'])
        waiting = next(z for z in self.zones if z.name=='Entrance waiting')
        self.assertFalse(waiting.contains_point(.771,.627))
        self.assertFalse(waiting.contains_point(.7443,.6122))

    def test_observed_left_side_road_starts_before_old_x_cutoff(self):
        # Saved 20:21:56 side-road grounds; first point was outside x=.13.
        # Boxes/confidence remain synthetic, as for the other route fixtures.
        road = next(z for z in self.zones if z.name=='Exit approach')
        path = [(.1281,.7336),(.1834,.7386),(.2609,.7345),(.3041,.7306),
                (.4339,.7037),(.4553,.6872),(.4845,.6537)]
        self.assertTrue(road.contains_point(*path[0]))
        for label in ('car','truck','bus'):
            with self.subTest(label=label):
                self.tracker = ObjectVisitTracker()
                accepted = self.run_path(path,label=label)
                self.assertEqual(1,len(accepted))
                self.assertEqual(1,accepted[0][0])
                self.assertEqual('campus-exit',accepted[0][1]['route_group'])
        # The change does not pull the icon/tree margin or the cabin footpath
        # into the vehicle zone, or change which classes use the footpath.
        self.assertFalse(road.contains_point(.10,.74))
        self.assertFalse(road.contains_point(.20,.58))
        footpath = next(z for z in self.zones if z.name=='Pedestrian footpath')
        self.assertTrue(footpath.contains_point(.20,.58))
        self.assertEqual(['person'],footpath.classes)


if __name__=='__main__':
    unittest.main()
