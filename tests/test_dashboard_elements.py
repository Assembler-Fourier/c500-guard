"""The page must contain every element its script touches.

The browser tests build elements on demand, so a script that reaches for an
element the page does not have still passes there while the real dashboard
fails on load. This checks the markup itself.
"""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dashboard_ui


def page_ids(html):
    return set(re.findall(r'id="([^"]+)"', html))


def referenced_ids(*scripts):
    return set(re.findall(r"""\$\(['"]([A-Za-z0-9_-]+)['"]\)""", "".join(scripts)))


class DashboardElementTests(unittest.TestCase):
    def test_dashboard_has_every_element_its_script_uses(self):
        # Obsidian embeds its own script; legacy constants serve the old page.
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', dashboard_ui.PAGE, re.S)
        missing = referenced_ids(*scripts) - page_ids(dashboard_ui.PAGE)
        self.assertEqual(missing, set(), f"dashboard script uses missing elements: {sorted(missing)}")

    def test_legacy_dashboard_script_matches_its_own_markup(self):
        missing = referenced_ids(dashboard_ui.CORE_SCRIPT, dashboard_ui.DASHBOARD_SCRIPT) - page_ids(dashboard_ui.DASHBOARD_BODY)
        self.assertEqual(missing, set())

    def test_editor_has_every_element_its_script_uses(self):
        page = dashboard_ui.zone_editor_page({"zones": [{"name": "Z", "rect": [.2, .2, .2, .2],
                                                         "motion": True, "light": False}]})
        missing = referenced_ids(dashboard_ui.CORE_SCRIPT, dashboard_ui.EDITOR_SCRIPT) - page_ids(page)
        self.assertEqual(missing, set(), f"editor script uses missing elements: {sorted(missing)}")

    def test_no_duplicate_element_ids(self):
        for name, html in (("dashboard", dashboard_ui.PAGE),
                           ("editor", dashboard_ui.zone_editor_page({"zones": [
                               {"name": "Z", "rect": [.2, .2, .2, .2], "motion": True, "light": False}]}))):
            with self.subTest(page=name):
                found = re.findall(r'id="([^"]+)"', html)
                duplicates = sorted({value for value in found if found.count(value) > 1})
                self.assertEqual(duplicates, [], f"{name} repeats ids: {duplicates}")


if __name__ == "__main__":
    unittest.main()
