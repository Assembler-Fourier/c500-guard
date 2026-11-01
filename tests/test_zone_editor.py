"""Zone editing validation and atomic-save regression coverage."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertIn(r"\u003c/script>", html)


if __name__ == "__main__":
    unittest.main()
