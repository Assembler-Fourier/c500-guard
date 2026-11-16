import copy
# Coordinates below are the September site positions mapped into the keyboard
# capture picture (13 Nov 2026) with the same lens-corrected fit as the zones.
import json
from pathlib import Path
import unittest

from test_watch import watch, watcher
from region_geometry import point_in_polygon
from zone_editor import validate_config


class CampusConfigurationTests(unittest.TestCase):
    def config(self):
        return {'zones':[dict(name='Entrance', rect=[.1,.1,.8,.8], trigger='ai',
            motion=False, classes=['car'], route_role='entrance',
            operator_hint='Check the entrance lane', alert_stage='barrier',
            confirmation_polygon=[[.3,.3],[.6,.3],[.6,.6],[.3,.6]])]}

    def test_site_metadata_and_gate_round_trip(self):
        config=self.config()
        self.assertEqual(config,validate_config(config))
        zone=watch.Zone(**config['zones'][0])
        self.assertEqual('entrance',zone.route_role)
        self.assertEqual(config['zones'][0]['confirmation_polygon'],zone.confirmation_polygon)

    def test_invalid_site_metadata_or_gate_rejected(self):
        for key,value in [('route_role','charging'),('route_role',True),
            ('operator_hint',None),('operator_hint','a'*241),
            ('confirmation_polygon',[[0,0],[.5,.2],[.2,.5]]),
            ('confirmation_polygon',[[.2,.2],[.5,.5]]),
            ('confirmation_polygon',[[.2,.2],[float('nan'),.3],[.4,.5]])]:
            config=self.config();config['zones'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_config(config)

    def test_anpr_profile_distinguishes_shared_approach_from_gate(self):
        path=Path(__file__).parents[1]/'config/views/anpr/zones.json'
        config=validate_config(json.loads(path.read_text()))
        zones={z.name:z for z in (watch.Zone(**item) for item in config['zones'])}
        exit_zone=zones['Exit approach']
        self.assertTrue(exit_zone.contains_point(62/640,216/360))
        self.assertFalse(exit_zone.contains_point(54/640,259/360))
        self.assertEqual('ai_light',exit_zone.trigger)
        self.assertEqual('main',exit_zone.light_severity)
        approach=zones['Top road arrival'];waiting=zones['Entrance waiting']
        self.assertTrue(approach.contains_point(500/640,94/360))
        self.assertFalse(point_in_polygon(491/640,139/360,approach.confirmation_polygon))
        self.assertTrue(point_in_polygon(466/640,67/360,approach.confirmation_polygon))
        self.assertEqual('advance',approach.alert_stage)
        self.assertEqual(0,approach.stationary_confirm_s)
        self.assertEqual('ai_light',approach.trigger)
        self.assertEqual(approach.confirmation_polygon,approach.light_polygon)
        self.assertFalse(approach.light_mask_for(640,360)[139,491])
        self.assertFalse(waiting.contains_point(393/640,214/360))
        self.assertEqual(approach.route_group,waiting.route_group)
        self.assertEqual(['person'],zones['Pedestrian footpath'].classes)

    def test_dashboard_state_exposes_gate_and_role(self):
        obj=watcher();obj.zones=[watch.Zone(**self.config()['zones'][0])]
        zone=obj.state_dict()['zones'][0]
        self.assertEqual('entrance',zone['route_role'])
        self.assertEqual('Check the entrance lane',zone['operator_hint'])
        self.assertEqual(self.config()['zones'][0]['confirmation_polygon'],zone['confirmation_polygon'])


if __name__ == '__main__':
    unittest.main()
