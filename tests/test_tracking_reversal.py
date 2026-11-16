"""Recorded detection replay and synthetic reversal controls; no live services.

Recorded fixtures preserve relative timestamps, normalized boxes and confidence.
They test route policy, not detector recall. The parking recording independently
shows a car crossing right to left, then stopping behind a tree. It does not turn
back toward the barrier. U-turn controls below are explicitly synthetic.
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

PARKING = [
    (0.0, 0.8452, (0.59413, 0.48475, 0.78584, 0.67689)),
    (0.505, 0.9093, (0.50667, 0.49267, 0.73323, 0.69481)),
    (1.11, 0.8934, (0.38386, 0.50769, 0.64616, 0.72097)),
    (1.713, 0.9003, (0.24292, 0.5275, 0.54366, 0.73606)),
    (2.317, 0.876, (0.12659, 0.52869, 0.43028, 0.73931)),
    (2.807, 0.8359, (0.0723, 0.53028, 0.37538, 0.73439)),
    (3.41, 0.652, (0.02653, 0.53956, 0.26092, 0.73089)),
    (4.029, 0.7596, (0.10952, 0.54694, 0.20987, 0.71872)),
    (5.73, 0.5511, (0.03259, 0.5445, 0.18589, 0.72183)),
    (6.318, 0.5996, (0.03442, 0.54481, 0.18564, 0.72197)),
    (6.838, 0.6275, (0.02322, 0.54511, 0.18419, 0.72211)),
    (7.396, 0.6777, (0.00806, 0.54414, 0.18642, 0.72011)),
    (8.0, 0.6601, (0.01487, 0.54456, 0.18691, 0.72053)),
    (8.598, 0.6899, (0.03009, 0.54525, 0.18703, 0.72053)),
    (9.104, 0.6578, (0.03291, 0.54569, 0.185, 0.72167)),
    (9.703, 0.7238, (0.03463, 0.54822, 0.18511, 0.72178)),
    (10.3, 0.7172, (0.03127, 0.54831, 0.18683, 0.72086)),
    (10.804, 0.6713, (0.03644, 0.54889, 0.18708, 0.72044)),
    (11.304, 0.6696, (0.03086, 0.54869, 0.18477, 0.72042)),
    (11.898, 0.6637, (0.03719, 0.55031, 0.18678, 0.72053)),
    (12.41, 0.6705, (0.03572, 0.55067, 0.18694, 0.72039)),
    (12.999, 0.6495, (0.0363, 0.55314, 0.18605, 0.72089)),
    (13.501, 0.6834, (0.03042, 0.55275, 0.18781, 0.72067)),
    (14.002, 0.6511, (0.03386, 0.55511, 0.18777, 0.72156)),
    (14.598, 0.6898, (0.036, 0.55508, 0.18634, 0.72153)),
    (15.204, 0.6829, (0.03155, 0.55219, 0.18636, 0.72108)),
    (15.726, 0.6594, (0.03033, 0.55253, 0.18392, 0.72044)),
    (16.323, 0.6451, (0.02675, 0.55122, 0.18292, 0.72044)),
    (16.902, 0.6462, (0.03428, 0.54989, 0.18461, 0.72044)),
    (17.409, 0.6196, (0.03986, 0.55161, 0.187, 0.72217)),
    (18.011, 0.6384, (0.03966, 0.55058, 0.18752, 0.72197)),
    (18.498, 0.6557, (0.03922, 0.55044, 0.18723, 0.72131)),
    (19.104, 0.6352, (0.03825, 0.55178, 0.18697, 0.72133)),
    (19.715, 0.6673, (0.04058, 0.552, 0.187, 0.72156)),
    (20.305, 0.6514, (0.03983, 0.55292, 0.18669, 0.72164)),
    (20.903, 0.6233, (0.03886, 0.55381, 0.18714, 0.72139)),
    (21.403, 0.6577, (0.02058, 0.551, 0.1865, 0.72156)),
    (21.903, 0.6566, (0.02284, 0.55139, 0.18508, 0.72161)),
    (22.514, 0.6609, (0.02623, 0.55133, 0.18502, 0.72153)),
    (22.991, 0.6773, (0.02264, 0.55056, 0.18423, 0.72144)),
    (23.495, 0.7051, (0.03747, 0.55231, 0.18533, 0.72219)),
    (24.009, 0.693, (0.03787, 0.55239, 0.18545, 0.72194)),
    (24.521, 0.6966, (0.03722, 0.55111, 0.18552, 0.723)),
    (25.111, 0.6902, (0.03123, 0.55006, 0.18544, 0.72294)),
    (25.694, 0.6725, (0.02528, 0.54822, 0.18283, 0.72231)),
    (26.308, 0.666, (0.02536, 0.54833, 0.184, 0.72242)),
    (26.788, 0.6894, (0.01911, 0.54747, 0.18294, 0.72208)),
    (27.397, 0.6835, (0.02841, 0.55275, 0.18813, 0.72297)),
    (27.99, 0.7061, (0.02869, 0.55075, 0.18655, 0.72247)),
    (28.528, 0.7098, (0.02928, 0.54928, 0.18425, 0.72319)),
    (29.104, 0.6737, (0.02094, 0.54867, 0.18444, 0.72258)),
    (31.321, 0.5611, (0.04648, 0.55106, 0.18323, 0.72278)),
    (33.083, 0.5273, (0.04713, 0.55453, 0.18644, 0.72342)),
    (33.592, 0.5752, (0.04922, 0.54672, 0.18664, 0.72303)),
    (34.192, 0.5874, (0.04778, 0.55939, 0.18547, 0.72086)),
    (34.796, 0.6387, (0.0437, 0.55742, 0.18302, 0.72144)),
    (35.29, 0.6763, (0.04556, 0.55281, 0.18272, 0.72236)),
    (35.881, 0.6339, (0.04566, 0.55683, 0.18361, 0.72181)),
]
ARRIVAL = [
    (0.0, 0.7794, (0.5643, 0.26675, 0.67625, 0.384)),
    (0.486, 0.7245, (0.59705, 0.271, 0.69425, 0.38725)),
    (1.022, 0.7078, (0.6247, 0.28117, 0.72486, 0.40708)),
    (1.592, 0.8163, (0.64173, 0.29239, 0.73164, 0.41761)),
    (2.113, 0.5609, (0.66186, 0.30378, 0.74058, 0.41592)),
    (2.691, 0.6076, (0.67394, 0.30831, 0.75995, 0.44339)),
    (3.203, 0.7167, (0.67609, 0.31433, 0.78675, 0.44875)),
    (3.79, 0.6837, (0.6942, 0.32222, 0.81269, 0.45994)),
]


def zones():
    result = []
    for row in json.loads((ROOT/'config/views/ptz2/zones.json').read_text())['zones']:
        item = SimpleNamespace(**row, ai_enabled=True)
        item.contains_point = lambda x,y,p=row['polygon']: region_contains(x,y,p,[])
        result.append(item)
    return result


class ReversalTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectVisitTracker()
        self.zones = [SimpleNamespace(name='road',rect=(0,0,1,1),ai_enabled=True,
            classes=['car'],severity='main',ai_confirm_observations=2,
            ai_confidence=.6,arrival_direction=[1,0],arrival_min_progress=.012,
            stationary_confirm_s=3)]
        self.now = 100.

    def step(self, x, confidence=.9):
        self.now += .5
        objects = [] if x is None else [{'label':'car','confidence':confidence,
            'bbox':[(x-.05)*640,.4*360,(x+.05)*640,.5*360]}]
        return self.tracker.update(objects,self.zones,640,360,self.now)

    def test_single_reverse_box_jump_then_stationary_cannot_confirm(self):
        # One apparent recovery exceeds the direction minimum, but no car
        # actually travels again; a count reset alone would alert next frame.
        for x in [.6,.56,.52,.48,.50]+[.50,.501,.499,.50]*20:
            events, observed = self.step(x)
            self.assertEqual([],events)
        self.assertEqual('away',observed[0]['direction']['road'])
        self.assertEqual(1,observed[0]['track_id'])

    def test_synthetic_uturn_needs_two_forward_observations(self):
        for x in [.6,.56,.52,.48,.485]:
            self.assertEqual([],self.step(x)[0])
        events, observed = self.step(.515)
        self.assertEqual(1,len(events))
        evidence = self.tracker.event_context(events[0][2],self.zones)
        self.assertEqual(2,evidence['observations'])
        self.assertEqual('toward',evidence['direction_state'])
        self.assertTrue(self.tracker.mark_emitted(events[0][2]))
        for x in [.53,.55,.58]+[.58]*12:
            self.assertEqual([],self.step(x)[0])

    def test_old_box_shift_cannot_combine_with_shift_after_ten_seconds_parked(self):
        for x in [.6,.56,.52,.48,.50]+[.50]*20+[.505]+[.505]*20:
            self.assertEqual([],self.step(x)[0])
        # Actual fresh continuous motion still recovers after that long stop.
        self.assertEqual([],self.step(.51)[0])
        self.assertEqual(1,len(self.step(.53)[0]))

    def test_reversal_evidence_does_not_survive_missing_or_low_confidence_frame(self):
        for gap in (None, .50):
            with self.subTest(gap=gap):
                self.setUp()
                for x in [.6,.56,.52,.48,.50]:
                    self.assertEqual([],self.step(x)[0])
                self.step(gap,confidence=.3)
                for x in [.52]+[.52]*12:
                    self.assertEqual([],self.step(x)[0])

    def test_recorded_lateral_parking_sequence_remains_silent_through_stop(self):
        self.zones = zones()
        self.assertEqual(58,len(PARKING))
        rows = list(PARKING)
        last_t, confidence, box = rows[-1]
        # All 58 real observations plus 20 explicitly synthetic parked controls.
        rows.extend((last_t+(i+1)*.5,confidence,box) for i in range(20))
        for at, confidence, box in rows:
            events, _ = self.tracker.update([{'label':'car','confidence':confidence,
                'bbox':[v*d for v,d in zip(box,[640,360,640,360])]}],
                self.zones,640,360,100+at)
            self.assertEqual([],events,msg=f'Unexpected parking alarm at {at}s')
        self.assertGreaterEqual(len(rows),60)

    def test_recorded_fast_initial_arrival_still_confirms_second_observation(self):
        self.zones = zones()
        accepted = []
        for index,(at,confidence,box) in enumerate(ARRIVAL):
            events,_ = self.tracker.update([{'label':'car','confidence':confidence,
                'bbox':[v*d for v,d in zip(box,[640,360,640,360])]}],
                self.zones,640,360,100+at)
            for event in events:
                accepted.append((index,self.tracker.event_context(event[2],self.zones)))
                self.assertTrue(self.tracker.mark_emitted(event[2]))
        self.assertEqual(1,len(accepted))
        self.assertEqual(1,accepted[0][0])
        self.assertEqual('Top road arrival',accepted[0][1]['zone_name'])
        self.assertAlmostEqual(.486,ARRIVAL[accepted[0][0]][0])


if __name__=='__main__':
    unittest.main()
