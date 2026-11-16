"""Two saved camera views (ANPR, PTZ 2), zoom tolerance and the framing guide (15 Nov 2026)."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import screen_registration  # noqa: E402
from screen_registration import framing_guide  # noqa: E402
from test_scene_alignment import textured_scene  # noqa: E402
from test_watch import watcher  # noqa: E402

SHAPE = (360, 640, 3)


def about_centre(scale, dx=0., dy=0.):
    """live -> saved matrix for a picture zoomed out by `scale` (>1) and shifted by (dx, dy) pixels."""
    cx, cy = 320., 180.
    return [[scale, 0., cx - scale * cx + dx], [0., scale, cy - scale * cy + dy]]


class FramingGuideTests(unittest.TestCase):
    def test_same_framing_is_ok(self):
        guide = framing_guide([[1., 0., 0.], [0., 1., 0.]], SHAPE, "the saved PTZ 2 view")
        self.assertEqual(guide["state"], "ok")
        self.assertIn("matches the saved PTZ 2 view", guide["text"])
        self.assertEqual(guide["outline"], [[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

    def test_wider_picture_asks_to_zoom_in(self):
        guide = framing_guide(about_centre(1.4), SHAPE)
        self.assertEqual((guide["state"], guide["zoom"], guide["pan"], guide["tilt"]), ("adjust", "in", None, None))
        self.assertAlmostEqual(guide["zoom_pct"], 40, delta=1)
        self.assertIn("zoom in (about 40 %)", guide["text"])
        # The saved view is a smaller box in the middle of the live picture.
        self.assertAlmostEqual(guide["outline"][0][0], .5 - .5 / 1.4, places=3)

    def test_closer_picture_asks_to_zoom_out_a_little(self):
        guide = framing_guide(about_centre(1 / 1.15), SHAPE)
        self.assertEqual(guide["zoom"], "out")
        self.assertIn("zoom out a little", guide["text"])

    def test_small_differences_are_ok(self):
        self.assertEqual(framing_guide(about_centre(1.03, dx=8, dy=-6), SHAPE)["state"], "ok")

    def test_turn_and_tilt_directions(self):
        # The saved view's centre lies 64 px right and 40 px below the live centre: turn right, tilt down.
        guide = framing_guide([[1., 0., -64.], [0., 1., -40.]], SHAPE)
        self.assertEqual((guide["zoom"], guide["pan"], guide["tilt"]), (None, "right", "down"))
        guide = framing_guide([[1., 0., 64.], [0., 1., 40.]], SHAPE)
        self.assertEqual((guide["pan"], guide["tilt"]), ("left", "up"))

    def test_estimate_reports_its_matrix_even_when_refused(self):
        reference = textured_scene()
        zone = np.zeros(reference.shape[:2], np.uint8)
        zone[:, :40] = 255                                   # a zone at the very left edge
        moved = cv2.warpAffine(reference, np.float32([[1.2, 0, -64], [0, 1.2, -36]]), (640, 360))   # zoomed in
        matrix, report = screen_registration.estimate_transform(moved, reference, required_mask=zone)
        self.assertIsNone(matrix)
        self.assertIn("outside", report["reason"])
        self.assertEqual(framing_guide(report["matrix"], SHAPE)["zoom"], "out")


class ZoomToleranceTests(unittest.TestCase):
    def build(self, alignment):
        obj = watcher()
        obj.alignment_config = alignment
        return obj

    def test_zoom_tolerance_widens_the_limits_together(self):
        config = self.build({"zoom_tolerance": .2})._aligner_config()
        self.assertAlmostEqual(config.max_scale_change, .2)
        self.assertAlmostEqual(config.max_translation_px, 76.)
        self.assertAlmostEqual(config.max_corner_displacement_px, 96.)
        self.assertAlmostEqual(config.min_valid_fraction, (1 / 1.2) ** 2 - .04)

    def test_fixed_camera_keeps_the_strict_limits(self):
        for alignment in ({}, {"zoom_tolerance": 0}, {"zoom_tolerance": .9}, {"zoom_tolerance": True},
                          {"zoom_tolerance": "big"}):
            obj = self.build(alignment)
            config = obj._aligner_config()
            self.assertEqual((config.max_scale_change, config.min_valid_fraction), (.04, .85), alignment)
            self.assertEqual(obj._zoom_tolerance(), 0., alignment)

    def test_each_view_keeps_its_own_reference_file(self):
        obj = watcher()
        source = obj.args.source
        self.assertEqual(obj._view_key(None), hashlib.sha256(source.encode()).hexdigest()[:16])
        self.assertEqual(obj._view_key("ptz2"), hashlib.sha256(f"{source}|ptz2".encode()).hexdigest()[:16])
        obj.view_name = "ptz2"
        self.assertTrue(obj._alignment_path().name.startswith(f"alignment-{obj._view_key('ptz2')}"))
        self.assertEqual(obj._view_title(), "PTZ 2")
        self.assertEqual(obj._view_title("anpr"), "ANPR Entrance")


class FramingStatusTests(unittest.TestCase):
    def build(self, zoom=.2):
        obj = watcher()
        obj.alignment_config = {"zoom_tolerance": zoom} if zoom else {}
        obj.view_name = "ptz2"
        obj._reacquire = None
        # PTZ-like alarm areas: inside the picture with a margin, like the real profile.
        obj.zones = [sys.modules[type(obj).__module__].Zone(name="Lanes", rect=(.08, .33, .83, .59))]
        return obj

    def test_fixed_views_have_no_guide(self):
        self.assertIsNone(self.build(zoom=0)._framing_status(True, {}, 10., SHAPE))

    def test_locked_on_the_saved_framing_is_ok(self):
        info = {"current_to_reference": [[1., 0., 0.], [0., 1., 0.]]}
        self.assertEqual(self.build()._framing_status(True, info, 10., SHAPE)["state"], "ok")

    def test_locked_through_a_remap_reports_the_whole_zoom(self):
        obj = self.build()
        obj._reacquire = np.vstack([np.asarray(about_centre(1.4)), [0., 0., 1.]])
        info = {"current_to_reference": [[1.02, 0., -3.2], [0., 1.02, -1.8]]}
        guide = obj._framing_status(True, info, 10., SHAPE)
        self.assertEqual(guide["zoom"], "in")
        self.assertAlmostEqual(guide["scale"], 1.428, delta=.01)

    def test_lost_view_uses_the_recent_estimate_then_says_not_found(self):
        obj = self.build()
        self.assertEqual(obj._framing_status(False, {}, 10., SHAPE)["state"], "unknown")
        obj._framing_seen = (about_centre(1.3), 100.)
        self.assertEqual(obj._framing_status(False, {}, 105., SHAPE)["zoom"], "in")
        guide = obj._framing_status(False, {}, 100. + obj.FRAMING_ESTIMATE_S + 1, SHAPE)
        self.assertEqual((guide["state"], guide["headline"]), ("unknown", "View alignment uncertain"))
        self.assertNotIn("zoom in slowly", guide["text"])

    def test_preview_draws_the_guide_without_failing(self):
        obj = self.build()
        obj._alignment_blocked = True
        obj.alignment = {"framing": framing_guide(about_centre(1.4), SHAPE)}
        obj.render_preview(np.zeros(SHAPE, dtype=np.uint8))
        self.assertTrue(obj.snapshot_preview().startswith(b"\xff\xd8"))


class ReacquireTimingTests(unittest.TestCase):
    def run_once(self, zoom, lost_for):
        obj = watcher()
        obj.alignment_config = {"zoom_tolerance": zoom} if zoom else {}
        obj.aligner = types.SimpleNamespace(reference=np.zeros(SHAPE, np.uint8))
        obj._extra_aligners = []
        obj._alignment_good = ("seen before", 1.)
        obj._alignment_bad_since, obj._last_reacquire_try = 1000. - lost_for, 0.
        refused = (None, {"reason": "Part of an alarm zone is outside the new camera view",
                          "matrix": [[1.3, 0., -96.], [0., 1.3, -54.], [0., 0., 1.]]})
        with mock.patch.object(screen_registration, "estimate_transform", return_value=refused) as estimate:
            obj._try_reacquire(np.zeros(SHAPE, np.uint8), 1000.)
        return obj, estimate

    def test_zoomable_view_looks_again_within_seconds(self):
        obj, estimate = self.run_once(.2, lost_for=4.)
        estimate.assert_called_once()
        self.assertEqual(obj._framing_seen[1], 1000.)
        self.assertIsNone(obj._reacquire)

    def test_fixed_view_still_waits(self):
        _, estimate = self.run_once(0, lost_for=4.)
        estimate.assert_not_called()


class OtherViewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.state = root / "state"
        self.state.mkdir()
        config = root / "config"
        (config / "views" / "ptz2").mkdir(parents=True)
        (config / "zones.json").write_text(json.dumps({"zones": []}), encoding="utf-8")
        self.profile = config / "views" / "ptz2" / "zones.json"
        self.profile.write_text(json.dumps({
            "view": "ptz2",
            "zones": [{"name": "Lane", "polygon": [[.35, .4], [.65, .4], [.65, .7], [.35, .7]]}],
            "alignment": {"include_rects": [[0, 0, 1, 1]], "border_px": 0, "zoom_tolerance": .2}}), encoding="utf-8")
        self.reference = textured_scene()
        self.obj = watcher()
        # test_watch loads watch.py under its own module name: patch that module.
        self.module = sys.modules[type(self.obj).__module__]
        self.obj.args.zones = str(config / "zones.json")
        self.obj.view_name = None
        self.obj.alignment_config = {}
        # The active view's zones cover everything: the other view must be checked with its own zones.
        self.obj.zones = [self.module.Zone(name="Everything", rect=(0., 0., 1., 1.))]
        np.savez_compressed(self.state / f"alignment-{self.obj._view_key('ptz2')}.npz", reference=self.reference)

    def tearDown(self):
        self.folder.cleanup()

    def attempt(self, frame):
        with mock.patch.object(self.module, "STATE_DIR", self.state), \
                mock.patch.object(self.obj, "_switch_view") as switch:
            self.obj._alignment_bad_since, self.obj._view_tried_at = 1.0, None
            found = self.obj._try_other_views(frame, 1.0 + self.obj.VIEW_SWITCH_AFTER_S + 1)
        return found, switch

    def test_saved_view_at_its_own_framing_is_switched_to(self):
        found, switch = self.attempt(self.reference.copy())
        self.assertTrue(found, getattr(self.obj, "_calibration_error", ""))
        switch.assert_called_once_with(self.profile, "PTZ 2")

    def test_saved_view_zoomed_out_is_still_recognised(self):
        zoomed_out = cv2.warpAffine(self.reference, np.float32([[.8, 0, 64], [0, .8, 36]]), (640, 360))
        found, switch = self.attempt(zoomed_out)
        self.assertTrue(found, getattr(self.obj, "_calibration_error", ""))

    def test_a_different_scene_is_not_switched_to(self):
        found, switch = self.attempt(textured_scene(seed=31))
        self.assertFalse(found)
        switch.assert_not_called()

    def test_waits_before_looking_and_between_looks(self):
        with mock.patch.object(self.module, "STATE_DIR", self.state), mock.patch.object(self.obj, "_switch_view") as switch:
            self.obj._alignment_bad_since, self.obj._view_tried_at = 100., 0.
            self.assertFalse(self.obj._try_other_views(self.reference, 105.))
            self.obj._view_tried_at = 125.
            self.assertFalse(self.obj._try_other_views(self.reference, 130.))
        switch.assert_not_called()


PTZ_ZONES = [[[0.13, 0.70], [0.44, 0.555], [0.585, 0.555], [0.47, 0.92], [0.19, 0.92]],      # exit approach
             [[0.66, 0.335], [0.905, 0.44], [0.905, 0.49], [0.67, 0.415]],                 # top road arrival
             [[0.08, 0.56], [0.31, 0.525], [0.40, 0.335], [0.79, 0.425], [0.08, 0.675]]]   # footpath


class FramingCoachTests(unittest.TestCase):
    def coach(self, matrix):
        return framing_guide(matrix, SHAPE, "the saved PTZ 2 view", zones=PTZ_ZONES)

    def test_home_framing_is_good(self):
        guide = self.coach([[1., 0., 0.], [0., 1., 0.]])
        self.assertEqual((guide["primary"], guide["state"]), ("good", "ok"))
        self.assertGreaterEqual(guide["fit_pct"], 90)

    def test_much_too_wide_says_zoom_in_a_lot(self):
        guide = self.coach(about_centre(1.95))
        self.assertEqual((guide["primary"], guide["headline"]), ("zoom_in", "ZOOM IN"))
        self.assertIn("2.0x closer", guide["text"])
        self.assertLess(guide["fit_pct"], 60)

    def test_a_little_closer_is_still_good_but_too_close_is_not(self):
        self.assertEqual(self.coach(about_centre(1 / 1.1))["primary"], "good")
        guide = self.coach(about_centre(1 / 1.3))
        self.assertEqual(guide["primary"], "zoom_out")
        self.assertTrue(guide["cut_off"])

    def test_off_centre_says_which_way_to_turn(self):
        self.assertEqual(self.coach([[1., 0., 60.], [0., 1., 0.]])["primary"], "turn_left")
        self.assertEqual(self.coach([[1., 0., -60.], [0., 1., 0.]])["primary"], "turn_right")

    def test_wide_and_off_centre_zooms_first_then_turns(self):
        guide = self.coach(about_centre(1.6, dx=-90.))
        self.assertEqual(guide["primary"], "zoom_in")
        self.assertIn("turn", guide["text"])

    def test_watcher_uses_its_zones_and_remembers_the_last_advice(self):
        obj = watcher()
        obj.alignment_config = {"zoom_tolerance": .2}
        obj.view_name, obj._reacquire = "ptz2", None
        obj.zones = [sys.modules[type(obj).__module__].Zone(name=f"z{i}", rect=(.1, .3, .8, .6)) for i in range(2)]
        self.assertEqual(len(obj._framing_zones()), 2)
        obj._framing_seen = (about_centre(1.9), 100.)
        self.assertEqual(obj._framing_status(False, {}, 101., SHAPE)["primary"], "zoom_in")
        lost = obj._framing_status(False, {}, 100. + obj.FRAMING_ESTIMATE_S + 5, SHAPE)
        self.assertEqual((lost["primary"], lost["headline"]), ("lost", "View alignment uncertain"))
        self.assertEqual(obj._framing_status(False, {}, 400., SHAPE)["headline"], "View alignment uncertain")

    def test_app_page_has_the_banner(self):
        import dashboard_ui
        page = dashboard_ui.dashboard_page()
        self.assertIn('id="framing-coach"', page)
        self.assertIn("renderFraming(s,status)", page)


if __name__ == "__main__":
    unittest.main()
