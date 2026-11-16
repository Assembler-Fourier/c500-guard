"""Stop-line policy replay, without image inference or live state.

The bus fixture preserves the three recorded detector boxes, confidence and
relative timestamps. Departure fixtures preserve observed grounds but use
synthetic confidence/boxes/cadence; they are route controls, not recall labels.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from object_tracking import ObjectVisitTracker
from region_geometry import region_contains
from zone_editor import validate_config


BUS = [
    (0., .5460, (.72733, .29622, .84434, .50203)),
    (.576, .5876, (.72539, .29667, .84378, .51083)),
    (1.143, .5379, (.7292, .29544, .8445, .48744)),
]
DEPARTURES = [
    [(.8373,.4735),(.8447,.4762),(.8451,.4754),(.8239,.4659),
     (.8016,.4676),(.7556,.4546),(.7092,.4489),(.6253,.4203),
     (.5711,.409),(.5372,.3917),(.5072,.373),(.4791,.357)],
    [(.8382,.4804),(.8254,.4734),(.7865,.472),(.7116,.4591),
     (.6706,.4515),(.6207,.443),(.615,.4259),(.6018,.4163)],
]


def observations(path, confidence=.9):
    return [(i*.55, confidence, (x-.04,y-.08,x+.04,y))
            for i,(x,y) in enumerate(path)]


class EntranceWaitingCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = validate_config(json.loads(
            (ROOT/'config/views/ptz2/zones.json').read_text()))
        self.zones = []
        for row in self.config['zones']:
            # These regressions exercise stop-line confirmation, independently
            # of the intentionally unrestricted Top first-sighting warning.
            if row['name'] != 'Entrance waiting':
                continue
            z = SimpleNamespace(**row, ai_enabled=True)
            z.contains_point = lambda x,y,p=row['polygon'],e=row.get('exclude_polygons',[]): region_contains(x,y,p,e)
            self.zones.append(z)

    def replay(self, rows, label='car', dx=0, dy=0):
        tracker = ObjectVisitTracker()
        alarms = []
        for at, confidence, box in rows:
            shifted = [v*d+delta for v,d,delta in zip(
                box, (640,360,640,360), (dx,dy,dx,dy))]
            events, _ = tracker.update([{'label':label,'confidence':confidence,
                                        'bbox':shifted}], self.zones,640,360,100+at)
            for event in events:
                alarms.append((at,tracker.event_context(event[2],self.zones)))
                self.assertTrue(tracker.mark_emitted(event[2]))
        return alarms

    def test_recorded_bus_confirms_on_third_reading_despite_box_height_jitter(self):
        self.assertEqual([], self.replay(BUS[:2], 'bus'))
        alarms = self.replay(BUS, 'bus')
        self.assertEqual(1,len(alarms))
        self.assertAlmostEqual(1.143,alarms[0][0])
        self.assertEqual('Entrance waiting',alarms[0][1]['zone_name'])
        self.assertEqual('campus-entrance',alarms[0][1]['route_group'])
        self.assertTrue(alarms[0][1]['confirmation_required'])

    def test_recorded_bus_tolerates_two_pixel_alignment_error(self):
        for dx in range(-2,3):
            for dy in range(-2,3):
                with self.subTest(dx=dx,dy=dy):
                    self.assertEqual(1,len(self.replay(BUS,'bus',dx,dy)))

    def test_all_recorded_departure_grounds_stay_silent_with_two_pixel_error(self):
        for path in DEPARTURES:
            for dx in range(-2,3):
                for dy in range(-2,3):
                    with self.subTest(first=path[0],dx=dx,dy=dy):
                        self.assertEqual([],self.replay(observations(path),dx=dx,dy=dy))

    def test_parking_branch_and_downstream_reflection_stay_silent(self):
        for path in [[(.68+i*.01,.63) for i in range(12)],[(.77,.63)]*20]:
            self.assertEqual([],self.replay(observations(path)))

    def test_low_confidence_single_detection_and_pedestrian_cannot_confirm(self):
        self.assertEqual([],self.replay(BUS[:1],'bus'))
        self.assertEqual([],self.replay([(t,.49,b) for t,_,b in BUS],'bus'))
        self.assertEqual([],self.replay(BUS,'person'))

    def test_waiting_vehicle_does_not_repeat_while_occupied(self):
        rows=BUS+[(1.743+i*.6,.6,BUS[-1][2]) for i in range(30)]
        self.assertEqual(1,len(self.replay(rows,'bus')))

    def test_broad_waiting_area_is_not_an_occupancy_trigger(self):
        # All three points are inside the tracking polygon, outside the core.
        for point in ((.8373,.4735),(.75,.46),(.70,.44)):
            self.assertEqual([],self.replay(observations([point]*15)))

    def test_active_and_saved_ptz_waiting_policy_match(self):
        active=json.loads((ROOT/'config/zones.json').read_text())
        self.assertEqual('ptz2',active['view'])
        live=next(z for z in active['zones'] if z['name']=='Entrance waiting')
        saved=next(z for z in self.config['zones'] if z['name']=='Entrance waiting')
        for key in ('polygon','confirmation_polygon','ai_confidence',
                    'arrival_direction','ai_confirm_observations'):
            self.assertEqual(live[key],saved[key])
        self.assertEqual(3,saved['ai_confirm_observations'])
        self.assertIsNone(saved['arrival_direction'])


if __name__=='__main__':
    unittest.main()
