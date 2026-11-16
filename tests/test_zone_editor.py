"""Zone editing validation and atomic-save regression coverage."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
spec = importlib.util.spec_from_file_location(
    "zone_editor", Path(__file__).resolve().parents[1] / "src" / "zone_editor.py")
editor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)


class ZoneEditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "zones.json"
        self.original = {"_comment": ["Keep me"], "zones": [{
            "name": "Approach", "rect": [0.2, 0.3, 0.4, 0.2],
            "motion": True, "light": False, "severity": "main",
            "light_severity": "early", "motion_pct": 1.2, "light_delta": 10,
            "on_frames": 2, "off_frames": 8}]}
        self.path.write_text(json.dumps(self.original), encoding="utf-8")
        self.payload = copy.deepcopy(self.original)

    def test_invalid_rectangles_and_types_are_rejected(self):
        for rect in ([0, 0, 0, 1], [-0.1, 0, 1, 1], [0.9, 0, 0.2, 1],
                     [0, 0.9, 1, 0.2], [0, 0, float("nan"), 1],
                     [0, 0, float("inf"), 1], [0, 0, True, 1], [0, 0, 1], "bad"):
            with self.subTest(rect=rect):
                self.payload["zones"][0]["rect"] = rect
                with self.assertRaises(ValueError):
                    editor.validate_config(self.payload)

    def test_invalid_config_and_thresholds_are_rejected(self):
        for payload in ({}, {"zones": []}, {"zones": "bad"}, {"zones": [None]},
                        {"zones": [self.original["zones"][0]] * 33}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                editor.validate_config(payload)
        for field, value in (("motion_pct", 0), ("motion_pct", 101),
                             ("light_delta", 256), ("motion", "false"),
                             ("on_frames", True), ("off_frames", 2.5),
                             ("severity", "unknown"), ("name", ""),
                             ("unexpected", True)):
            payload = copy.deepcopy(self.original)
            payload["zones"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                editor.validate_config(payload)

    def test_saved_update_preserves_settings_and_exact_backup(self):
        original_bytes = self.path.read_bytes()
        self.payload["zones"][0]["name"] = "New approach"
        self.payload["zones"][0]["rect"] = [0.1, 0.1, 0.7, 0.8]
        result = editor.save_config(self.path, {"zones": self.payload["zones"]}, armed=False)
        self.assertEqual(result["_comment"], ["Keep me"])
        self.assertEqual(result["zones"][0]["on_frames"], 2)
        self.assertEqual(result["zones"][0]["light_severity"], "early")
        self.assertEqual(json.loads(self.path.read_text()), result)
        backups = list(self.path.parent.glob("zones.json.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_bytes)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_armed_and_invalid_saves_leave_file_unchanged(self):
        before = self.path.read_bytes()
        with self.assertRaises(PermissionError):
            editor.save_config(self.path, self.payload, armed=True)
        with self.assertRaises(ValueError):
            editor.save_config(self.path, {"zones": []}, armed=False)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(list(self.path.parent.iterdir())), 1)

    def test_replace_failure_leaves_original_and_backup(self):
        before = self.path.read_bytes()
        actual_replace = editor.os.replace
        def fail_config_replace(source, destination):
            if Path(destination) == self.path:
                raise OSError("simulated replace failure")
            return actual_replace(source, destination)
        with patch.object(editor.os, "replace", side_effect=fail_config_replace), \
             self.assertRaises(OSError):
            editor.save_config(self.path, self.payload, armed=False)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(list(self.path.parent.glob("zones.json.bak-*"))), 1)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_embedded_labels_cannot_close_script(self):
        self.original["zones"][0]["name"] = "</script><script>alert(1)</script>"
        self.path.write_text(json.dumps(self.original), encoding="utf-8")
        html = editor.render_page(self.path)
        self.assertNotIn("</script><script>alert(1)</script>", html)
        script_data = html.split('<script id="initial-config" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn('<', script_data)
        self.assertEqual(json.loads(script_data)['zones'][0]['name'],
                         self.original['zones'][0]['name'])

    def test_polygon_class_filters_and_ignored_parking_round_trip(self):
        zone = self.payload['zones'][0]
        zone.update(polygon=[[.2, .3], [.6, .3], [.5, .5], [.3, .5]],
                    classes=['car', 'truck'], trigger='ai',
                    exclude_polygons=[[[.4, .4], [.6, .4], [.6, .5], [.4, .5]]])
        result = editor.save_config(self.path, self.payload, armed=False)
        self.assertEqual(result['zones'][0], zone)
        self.assertEqual(editor.read_config(self.path), result)
        self.assertEqual(result['_comment'], ['Keep me'])

    def test_detailed_polygons_render_and_round_trip_without_simplification(self):
        for count in (18, 21, 32):
            with self.subTest(count=count):
                points = [[.4 + .08 * math.cos(2 * math.pi * i / count),
                           .4 + .08 * math.sin(2 * math.pi * i / count)]
                          for i in range(count)]
                original = copy.deepcopy(self.original)
                zone = original['zones'][0]
                zone.update(polygon=points, confirmation_polygon=points,
                            light_polygon=points, exclude_polygons=[points])
                self.path.write_text(json.dumps(original), encoding='utf-8')
                before = self.path.read_bytes()
                loaded = editor.read_config(self.path)
                page = editor.render_page(self.path)
                script_data = page.split(
                    '<script id="initial-config" type="application/json">', 1
                )[1].split('</script>', 1)[0]
                self.assertEqual(json.loads(script_data)['zones'], original['zones'])
                self.assertEqual(editor.save_config(
                    self.path, {'zones': loaded['zones']}, armed=False), original)
                self.assertEqual(editor.read_config(self.path), original)
                self.assertTrue(any(p.read_bytes() == before
                                    for p in self.path.parent.glob('zones.json.bak-*')))

    def test_oversized_polygon_saves_leave_original_and_backups_unchanged(self):
        points = [[.4 + .08 * math.cos(2 * math.pi * i / 33),
                   .4 + .08 * math.sin(2 * math.pi * i / 33)] for i in range(33)]
        before = self.path.read_bytes()
        for field in ('polygon', 'confirmation_polygon', 'light_polygon', 'exclude_polygons'):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.original)
                payload['zones'][0][field] = [points] if field == 'exclude_polygons' else points
                with self.assertRaisesRegex(ValueError, 'needs 3–32 corners'):
                    editor.save_config(self.path, payload, armed=False)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_bad_polygon_shapes_coordinates_and_bounds_are_rejected(self):
        polygons = (
            [], [[.2, .3], [.6, .3]],
            [[.2, .3], [.6, .5], [.6, .3], [.2, .5]],
            [[.2, .3], [.4, .3], [.6, .3]],
            [[.2, .3], [.6, .3], [.6, float('nan')]],
            [[.2, .3], [.6, .3], [.6, True]],
            [[.2, .3], [.6, .3], [.7, .5]],  # Outside the bounding rectangle.
            [[-.1, .3], [.6, .3], [.6, .5]],
            'bad', None)
        for polygon in polygons:
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                payload = copy.deepcopy(self.original)
                payload['zones'][0]['polygon'] = polygon
                editor.validate_config(payload)

    def test_invalid_class_filters_triggers_and_exclusions_are_rejected(self):
        pairs = [('classes', 'person'), ('classes', ['person', 'person']),
                 ('classes', ['unknown']), ('classes', [True]),
                 ('classes', [[]]), ('classes', []), ('trigger', 'all'),
                 ('exclude_polygons', None), ('exclude_polygons', ['bad']),
                 ('exclude_polygons', [[]] * 17)]
        for field, value in pairs:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                payload = copy.deepcopy(self.original)
                payload['zones'][0][field] = value
                editor.validate_config(payload)
        self.payload['zones'][0].update(trigger='motion', classes=[])
        self.assertEqual(editor.validate_config(self.payload)['zones'][0]['classes'], [])

    def test_rectangle_only_configuration_remains_unchanged(self):
        result = editor.validate_config(self.original)
        self.assertEqual(result['zones'], self.original['zones'])
        self.assertNotIn('polygon', result['zones'][0])
        self.assertNotIn('classes', result['zones'][0])
        self.assertNotIn('trigger', result['zones'][0])


if __name__ == "__main__":
    unittest.main()
