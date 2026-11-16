"""Configuration gates and exact clean-preview handoff; all local scratch/fakes."""
import copy
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from test_watch import watch, watcher
from zone_editor import validate_config

ROOT = Path(__file__).resolve().parents[1]


class TruckConfigTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'config/views/ptz2/zones.json').read_text())
        self.top = next(z for z in self.data['zones'] if z['name'] == 'Top road arrival')
        self.wait = next(z for z in self.data['zones'] if z['name'] == 'Entrance waiting')
        self.top['approach_motion'] = True
        self.wait['ai_fast_confirm_confidence'] = .60

    def test_optin_roundtrips_and_other_profiles_stay_off(self):
        result = validate_config(self.data)
        self.assertTrue(next(z for z in result['zones'] if z['name'] == self.top['name'])['approach_motion'])
        zones = watch.load_zones(ROOT/'config/views/anpr/zones.json')
        self.assertFalse(any(z.approach_motion or z.ai_fast_confirm_confidence for z in zones))

    def test_invalid_fast_config_is_rejected(self):
        for field, value in [('ai_fast_confirm_confidence', .59), ('ai_fast_confirm_confidence', True),
                             ('ai_fast_confirm_confidence', float('nan')), ('ai_confidence', .8),
                             ('alert_stage', 'advance'), ('confirmation_polygon', None),
                             ('ai_confirm_observations', 2)]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                next(z for z in data['zones'] if z['name'] == self.wait['name'])[field] = value
                with self.assertRaises(ValueError): validate_config(data)

    def test_approach_requires_explicit_directed_entrance_geometry(self):
        for field, value in [('approach_motion', 1), ('alert_stage', 'barrier'), ('arrival_direction', None),
                             ('arrival_direction', [0,0]), ('route_group', ''), ('route_role', 'exit'),
                             ('trigger', 'motion'), ('polygon', None)]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                next(z for z in data['zones'] if z['name'] == self.top['name'])[field] = value
                with self.assertRaises(ValueError): validate_config(data)


class CleanPreviewHandoffTests(unittest.TestCase):
    def test_rendered_and_clean_are_paired_with_original_source_time(self):
        obj = watcher()
        obj.recorder = mock.Mock()
        frame = np.full((360,640,3), 70, np.uint8)
        obj.render_preview(frame, raw=np.full_like(frame, 120), received_at=123.456)
        args, kwargs = obj.recorder.add_frame.call_args
        self.assertEqual(123.456, kwargs['source_at'])
        self.assertEqual(obj.preview_jpeg, args[0])
        self.assertEqual(obj.scene_jpeg, kwargs['clean_jpeg'])
        clean = np.asarray(Image.open(BytesIO(kwargs['clean_jpeg'])))
        self.assertLess(np.abs(clean.astype(float)-70).max(), 2)
        self.assertNotEqual(obj.preview_jpeg, obj.scene_jpeg)
        self.assertNotEqual(obj.raw_jpeg, obj.scene_jpeg)

    def test_missing_source_timestamp_remains_explicitly_unknown(self):
        obj = watcher(); obj.recorder = mock.Mock()
        obj.render_preview(np.full((360,640,3),70,np.uint8))
        self.assertIsNone(obj.recorder.add_frame.call_args.kwargs['source_at'])


if __name__ == '__main__': unittest.main()
