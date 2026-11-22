"""Measured clipped-box regression; no camera, model inference or real audio."""
import copy
import unittest

from test_watch import watch
from object_tracking import ObjectVisitTracker, zone_accepts_box, _bottom_edge_y
from zone_editor import validate_config


class BottomEdgeCalibrationTests(unittest.TestCase):
    def setUp(self):
        # Self-contained lane fixture: no private camera calibration required.
        self.data = dict(name='Exit approach', rect=[.12,.555,.478,.365],
            classes=['car','motorcycle','bus','truck'], trigger='ai',
            route_group='exit', route_role='exit', alert_stage='advance',
            arrival_direction=[.35,-1.], arrival_min_progress=.012,
            stationary_confirm_s=3., ai_confidence=.5, ai_confirm_observations=2,
            bottom_edge_entry=True, ai_confidence_moving=.3,
            ai_moving_entry_polygon=[[0,.63],[.35,.63],[.35,.82],[1,.82],[1,1],[0,1]])
        self.zone = watch.Zone(**self.data)

    def sequence(self, zone, samples):
        tracker = ObjectVisitTracker()
        events = []
        for at, score, box in samples:
            found, _ = tracker.update([{'label':'car','confidence':score,'bbox':box}], [zone], 640, 360, 100+at)
            events.extend(found)
            for _, _, event in found:
                tracker.mark_emitted(event)
        return events

    def test_known_clipped_car_enters_early_only_with_its_configured_edge(self):
        # Real clean-frame NN boxes, source times from clip003. The car is
        # physically clipped by the bottom, but boxes stop around y=.974-.977.
        samples = [(7.998,.4593,[55.8,317.88,188.94,351.89]),
                   (8.403,.7658,[89.58,291.66,223.75,351.49]),
                   (8.603,.8963,[109.25,278.26,235.94,350.76])]
        self.assertEqual([], self.sequence(self.zone, samples))
        self.zone.bottom_edge_y = .97
        self.assertEqual(1, len(self.sequence(self.zone, samples)))

    def test_one_sighting_or_no_motion_cannot_confirm(self):
        self.zone.bottom_edge_y = .97
        box = [89.58,291.66,223.75,351.49]
        self.assertEqual([], self.sequence(self.zone, [(1,.9,box)]))
        self.assertEqual([], self.sequence(self.zone, [(1,.9,box),(1.2,.9,box),(1.4,.9,box)]))

    def test_clipped_motion_away_does_not_become_an_arrival(self):
        self.zone.bottom_edge_y = .97
        self.assertEqual([], self.sequence(self.zone, [
            (1,.9,[109.25,278.26,235.94,350.76]),
            (1.2,.9,[89.58,291.66,223.75,351.49]),
            (1.4,.9,[55.8,317.88,188.94,351.89])]))

    def test_edge_rescue_still_requires_lane_x_and_vehicle_class(self):
        self.zone.bottom_edge_y = .97
        self.assertTrue(zone_accepts_box(self.zone,'truck',(.25,.976),(.15,.8,.35,.976)))
        self.assertFalse(zone_accepts_box(self.zone,'truck',(.80,.976),(.7,.8,.9,.976)))
        self.assertFalse(zone_accepts_box(self.zone,'person',(.25,.976),(.15,.8,.35,.976)))
        self.assertFalse(zone_accepts_box(self.zone,'car',(.25,.96),(.15,.8,.35,.96)))
        self.zone.bottom_edge_entry = False
        self.assertFalse(zone_accepts_box(self.zone,'car',(.25,.976),(.15,.8,.35,.976)))

    def test_config_accepts_measured_value_and_preserves_legacy_default(self):
        changed = copy.deepcopy(self.data)
        changed['bottom_edge_y'] = .97
        result = validate_config({'zones':[changed]})
        self.assertEqual(.97,watch.Zone(**result['zones'][0]).bottom_edge_y)
        self.assertEqual(.985,self.zone.bottom_edge_y)
        self.assertEqual(.985,_bottom_edge_y(self.zone))

    def test_invalid_runtime_values_fall_back_and_invalid_config_is_rejected(self):
        for value in (True,None,'0.97',.949,1.001,float('nan'),float('inf')):
            with self.subTest(value=value):
                self.zone.bottom_edge_y = value
                self.assertEqual(.985,_bottom_edge_y(self.zone))
                changed = {**self.data,'bottom_edge_y':value}
                with self.assertRaises(ValueError):
                    validate_config({'zones':[changed]})


if __name__ == '__main__':
    unittest.main()
