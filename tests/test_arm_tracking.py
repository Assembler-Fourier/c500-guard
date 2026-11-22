"""Barrier arms followed at every angle; alarms stop as soon as an arm starts moving (15 Nov 2026)."""
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from test_watch import watch, watcher
import arm_angle
import dashboard_ui

LINES = {"down": [[0.3596, 0.2045], [0.5408, 0.2281]], "up": [[0.541, 0.01], [0.541, 0.205]]}


def picture_with_arm(theta, geometry, size=(360, 640)):
    """Dark scene with a bright, red-striped bar at angle theta."""
    frame = np.full(size + (3,), 55, np.uint8)
    pivot = geometry["pivot"]
    _points, tip = arm_angle.segment(geometry, theta, start=0., end=1.)
    cv2.line(frame, tuple(int(v) for v in pivot), tuple(int(v) for v in tip), (235, 235, 235), 3)
    for f in (.3, .55, .8):
        p = pivot + f * (tip - pivot)
        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (220, 30, 30), -1)
    return frame


class GeometryAndScoreTests(unittest.TestCase):
    def setUp(self):
        self.geometry = arm_angle.geometry_from_lines(LINES)

    def test_pivot_is_the_shared_end(self):
        self.assertTrue(np.allclose(self.geometry["pivot"], [346.2, 78.0], atol=.5))
        self.assertTrue(np.allclose(self.geometry["up_tip"], [346.2, 3.6], atol=.5))

    def test_the_arm_is_found_at_every_angle(self):
        for theta in (0., 20., 45., 70., 90.):
            with self.subTest(theta=theta):
                scores = arm_angle.arm_scores(picture_with_arm(theta, self.geometry), self.geometry)
                self.assertAlmostEqual(theta, float(arm_angle.ANGLES[int(np.argmax(scores))]), delta=5)

    def test_refinement_moves_lines_onto_the_real_arm(self):
        true_geometry = {k: v + (3, 2) for k, v in self.geometry.items()}           # the real arm is offset
        refined = arm_angle.refine_geometry(self.geometry, picture_with_arm(0., true_geometry),
                                            picture_with_arm(90., true_geometry))
        self.assertIsNotNone(refined)
        down, up = picture_with_arm(0., true_geometry), picture_with_arm(90., true_geometry)

        def fit(g):
            return float(arm_angle.arm_scores(down, g, [0.])[0] + arm_angle.arm_scores(up, g, [90.])[0])
        self.assertGreaterEqual(fit(refined), fit(self.geometry))            # the lines sit on the real arm


class MotionRuleTests(unittest.TestCase):
    def tracker(self):
        return arm_angle.ArmAngleTracker({"exit": arm_angle.geometry_from_lines(LINES)})

    def motion(self, thetas, step=.17):
        tracker = self.tracker()
        history = [(i * step, float(v)) for i, v in enumerate(thetas)]
        return tracker._motion(history)

    def test_a_real_lift_is_rising_and_knows_when_it_started(self):
        kind, started = self.motion([0, 0, 0, 5, 15, 25, 35, 75])        # measured 14 Nov 22:07
        self.assertEqual("rising", kind)
        self.assertAlmostEqual(3 * .17, started, places=2)                  # last reading still at rest (5 deg)

    def test_a_real_drop_is_lowering(self):
        self.assertEqual("lowering", self.motion([90, 90, 85, 75, 65, 60, 55], step=.4)[0])

    def test_a_single_odd_frame_or_a_jump_is_never_movement(self):
        for thetas in ([0, 0, 0, 90, 0, 0], [0, 0, 90, 90, 90, 90], [90, 90, 0, 0, 0, 0], [0, 0, 0, 0, 0, 90]):
            with self.subTest(thetas=thetas):
                self.assertIsNone(self.motion(thetas)[0])

    def test_back_and_forth_or_too_slow_is_not_a_lift(self):
        self.assertIsNone(self.motion([0, 0, 20, 5, 30, 40, 50])[0])
        self.assertIsNone(self.motion([0, 0, 15, 30, 40, 50], step=1.)[0])            # 5 s to lift: not this barrier

    def test_tracker_end_to_end_on_pictures(self):
        tracker = self.tracker()
        geometry = tracker.geometry["exit"]
        result = None
        for i, theta in enumerate([0, 0, 0, 20, 35, 60, 90, 90]):
            result = tracker.update(picture_with_arm(theta, geometry), i * .17)["exit"]
        self.assertEqual("rising", result["motion"])
        self.assertEqual(90., result["theta"])
        self.assertEqual(2, len(result["line"]))

    def test_unreadable_arm_cannot_repeat_or_bridge_a_movement(self):
        tracker = self.tracker()
        geometry = tracker.geometry["exit"]
        for i, theta in enumerate([0, 0, 0, 20, 35, 60]):
            reading = tracker.update(picture_with_arm(theta, geometry), i * .17)["exit"]
        self.assertEqual("rising", reading["motion"])
        unreadable = tracker.update(np.zeros((360, 640, 3), np.uint8), 1.02)["exit"]
        self.assertFalse(unreadable["ok"])
        self.assertIsNone(unreadable["motion"])
        self.assertIsNone(unreadable["motion_started"])
        self.assertIsNone(unreadable["shown"])
        recovered = tracker.update(picture_with_arm(90, geometry), 1.19)["exit"]
        self.assertTrue(recovered["ok"])
        self.assertIsNone(recovered["motion"])

    def test_gap_or_nonincreasing_timestamp_cannot_complete_a_partial_lift(self):
        for stamp in (2., .1, .68):
            with self.subTest(stamp=stamp):
                tracker = self.tracker()
                geometry = tracker.geometry["exit"]
                for i, theta in enumerate([0, 0, 0, 20, 35]):
                    tracker.update(picture_with_arm(theta, geometry), i * .17)
                result = tracker.update(picture_with_arm(60, geometry), stamp)["exit"]
                self.assertIsNone(result["motion"])
                self.assertEqual(1, len(tracker._history["exit"]))

    def test_reset_discards_the_previous_views_trajectory(self):
        tracker = self.tracker()
        geometry = tracker.geometry["exit"]
        for i, theta in enumerate([0, 0, 0, 20, 35]):
            tracker.update(picture_with_arm(theta, geometry), i * .17)
        tracker.reset()
        self.assertEqual({}, tracker.latest)
        self.assertIsNone(tracker.update(picture_with_arm(60, geometry), .85)["exit"]["motion"])

    def test_invalid_timestamps_are_not_motion_observations(self):
        tracker = self.tracker()
        geometry = tracker.geometry["exit"]
        for stamp in (None, True, "1", float("nan"), float("inf")):
            with self.subTest(stamp=stamp):
                self.assertEqual({}, tracker.update(picture_with_arm(60, geometry), stamp))
                self.assertEqual([], tracker._history["exit"])


class RecorderTests(unittest.TestCase):
    PTZ_GEOMETRY = {
        "exit": {"pivot": [.635, .452], "down_tip": [.497, .439], "up_tip": [.655, .13]},
        "entrance": {"pivot": [.656, .474], "down_tip": [.84, .492], "up_tip": [.656, .04]},
    }

    def test_ptz_crop_contains_both_arms_at_every_angle(self):
        geometry = {arm: arm_angle.geometry_from_config(g) for arm, g in self.PTZ_GEOMETRY.items()}
        roi = arm_angle.ArmMotionRecorder.roi_from_geometry(geometry)
        low, high = np.array(roi[:2]) * (640, 360), np.array(roi[2:]) * (640, 360)
        self.assertGreater(roi[3], .492)         # old ANPR crop stopped at .36 and lost both down arms
        for arm in geometry.values():
            for theta in np.linspace(0, 90, 181):
                points, tip = arm_angle.segment(arm, theta, start=0, end=1)
                self.assertTrue(np.all(points >= low))
                self.assertTrue(np.all(points <= high))

    def test_crop_includes_extrema_between_the_up_and_down_endpoints(self):
        geometry = {"test": {"pivot": np.array([100, 100]), "down_tip": np.array([200, 0]),
                             "up_tip": np.array([200, 0])}}
        roi = arm_angle.ArmMotionRecorder.roi_from_geometry(geometry, margin_px=0)
        self.assertAlmostEqual(roi[2] * 640, 100 + math.sqrt(2) * 100)
        self.assertEqual(0., roi[1])             # sweep reaches above the top of the image

    def test_missing_geometry_retains_legacy_crop_and_malformed_geometry_is_bounded(self):
        self.assertEqual(arm_angle.ArmMotionRecorder.ROI, arm_angle.ArmMotionRecorder.roi_from_geometry(None))
        for geometry in ({"exit": {}}, {"exit": {"pivot": [float("nan"), 0], "down_tip": [1, 1], "up_tip": [2, 2]}}):
            with self.subTest(geometry=geometry):
                self.assertEqual((0., 0., 1., 1.), arm_angle.ArmMotionRecorder.roi_from_geometry(geometry))

    def test_saved_ptz_frames_and_metadata_use_the_active_geometry(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            geometry = {arm: arm_angle.geometry_from_config(g) for arm, g in self.PTZ_GEOMETRY.items()}
            recorder = arm_angle.ArmMotionRecorder(folder, geometry=geometry, after_s=.3)
            frame = np.zeros((360, 640, 3), np.uint8)
            frame[170:185, 420:540] = 255          # entrance down arm area below the old crop
            angles = {"entrance": {"theta": 40., "score": 30., "motion": "rising"}}
            recorder.add(frame, 1., angles)
            recorder.add(frame, 1.4, angles)
            target = next(Path(folder).iterdir())
            meta = json.loads((target / "angles.json").read_text(encoding="utf-8"))
            self.assertEqual(list(recorder.roi), meta["roi"])
            pixels = np.asarray(Image.open(target / "frame_000.jpg"))
            self.assertGreater(int(pixels.max()), 250)
            self.assertEqual(pixels.shape[:2],
                             (math.ceil(recorder.roi[3] * 360) - math.floor(recorder.roi[1] * 360),
                              math.ceil(recorder.roi[2] * 640) - math.floor(recorder.roi[0] * 640)))

    def test_a_movement_is_kept_as_clean_frames_with_angles(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = arm_angle.ArmMotionRecorder(folder, keep=2, before_s=.5, after_s=1.)
            frame = np.full((360, 640, 3), 120., np.float32)      # the watcher hands over a float frame
            (Path(folder) / "20000101-000000-exit-rising").mkdir()   # an empty folder left by an old failure
            for i in range(20):
                motion = "rising" if 5 <= i < 8 else None
                recorder.add(frame, i * .2, {"exit": {"theta": 10. * i, "score": 9., "motion": motion}})
            folders = list(Path(folder).iterdir())
            self.assertEqual(1, len(folders))                       # the empty one was tidied away
            self.assertTrue(list(folders[0].glob("frame_*.jpg")))
            meta = json.loads((folders[0] / "angles.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(meta["frames"]), 6)
            self.assertTrue(folders[0].name.endswith("-exit-rising"))


class EarlyAckTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj._alignment_blocked = False
        self.obj.alignment = {"state": "tracking"}
        self.obj._arm_memory = {"exit": {"state": "down"}, "entrance": {"state": "down"}}
        self.assertTrue(self.obj.raise_alarm("main", "exit corner", details={"route_role": "exit"}))
        self.obj.alarm_since -= 1.
        self.started = self.obj.alarm_since + .3

    def reading(self, **changes):
        return {"motion": "rising", "motion_started": self.started, "strict": None,
                "ok": True, "at": time.monotonic(), **changes}

    def test_rising_arm_stops_the_alarm_at_once(self):
        alarm_id = self.obj.alarm_id
        self.assertTrue(self.obj._ack_on_arm_motion("exit", self.reading()))
        self.assertIsNone(self.obj.alarm)
        acked = {a["id"]: a["acked"] for a in self.obj.alerts()["alerts"]}
        self.assertEqual("barrier-raised", acked[alarm_id]["via"])
        kinds = [c.args[0] for c in self.obj.log_event.call_args_list]
        self.assertIn("auto-barrier-up", kinds)

    def test_safety_gates(self):
        cases = {
            "held open (memory up)": lambda: self.obj._arm_memory.update({"exit": {"state": "up"}}),
            "strict still sees down": None,
            "movement began before the alarm": None,
            "switched off in settings": lambda: self.obj.barrier_rules.update({"ack_on_movement": False}),
        }
        for name, prepare in cases.items():
            with self.subTest(name):
                self.setUp()
                if prepare:
                    prepare()
                reading = self.reading()
                if name == "strict still sees down":
                    reading["strict"] = "down"
                if name == "movement began before the alarm":
                    reading["motion_started"] = self.obj.alarm_since - 5
                self.assertFalse(self.obj._ack_on_arm_motion("exit", reading))
                self.assertEqual("main", self.obj.alarm)

    def test_live_line_is_published_while_fresh(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {"configured": True, "barriers": [
            {"id": "exit", "state": "unknown", "fresh": True, "calibrated": True},
            {"id": "entrance", "state": "down", "fresh": True, "calibrated": True}]}
        self.obj._arm_live = {"exit": {"line": [[.4, .1], [.54, .22]], "theta": 45., "motion": "rising",
                                       "ok": True, "at_wall": time.time()}}
        rows = {r["id"]: r for r in self.obj.barrier_status()["barriers"]}
        self.assertEqual([[.4, .1], [.54, .22]], rows["exit"]["live_line"])
        geometry = self.obj._arm_tracker.geometry['entrance']
        expected = [[round(float(p[0])/watch.PROC_W, 4), round(float(p[1])/watch.PROC_H, 4)]
                    for p in (geometry['down_tip'], geometry['pivot'])]
        self.assertEqual(expected, rows["entrance"]["live_line"])
        self.obj._arm_live["exit"]["at_wall"] -= 5
        self.assertIsNone({r["id"]: r for r in self.obj.barrier_status()["barriers"]}["exit"]["live_line"])


class FirstLoadGraceTests(unittest.TestCase):
    def test_slow_first_ai_load_is_not_a_fault_but_a_failure_is(self):
        obj = watcher()
        obj.semantic = mock.Mock()
        obj.semantic.status.return_value = {"available": False, "state": "loading"}
        obj.first_frame_seen = True
        obj.last_frame_at = time.monotonic()
        obj.started = time.monotonic() - 30
        obj._ai_bad_since = time.monotonic() - 20
        with mock.patch.object(obj, "raise_alarm", return_value=True) as raised:
            obj.check_health(time.monotonic())
            self.assertFalse(any(c.args[0] == "fault" for c in raised.call_args_list))
            obj.semantic.status.return_value = {"available": False, "state": "retrying"}
            obj.check_health(time.monotonic())
            self.assertTrue(any(c.args[0] == "fault" for c in raised.call_args_list))


class PageTests(unittest.TestCase):
    def test_dashboard_draws_the_live_arm(self):
        page = dashboard_ui.dashboard_page("BarrierWatcher")
        self.assertIn("Array.isArray(raw.live_line)", page)
        self.assertIn("const validLine=(known||!!live)", page)


class CrossCheckTests(unittest.TestCase):
    """15 Nov 18:20: a car under the raised exit arm made the strict reading say "down" for a moment."""

    def setUp(self):
        self.obj = watcher()
        self.obj._alignment_blocked = False
        self.obj.alignment = {"state": "tracking"}

    def live(self, arm, shown):
        self.obj._arm_live = {arm: {"shown": shown, "theta": shown, "at_wall": time.time()}}

    def status(self, state):
        return {"trusted": True, "barriers": [{"id": "exit", "state": state, "fresh": True, "calibrated": True}]}

    def test_screen_smoothing_follows_movement_but_not_jumps(self):
        tracker = arm_angle.ArmAngleTracker({"exit": arm_angle.geometry_from_lines(LINES)})
        history = []
        shown = None
        for i, theta in enumerate([90, 90, 90, 0, 0]):          # a car passing: two odd readings
            history.append((i * .17, float(theta)))
            shown = tracker._smooth("exit", history, float(theta))
        self.assertEqual(90., shown)
        for i, theta in enumerate([80, 70, 55, 40, 25], start=5):   # a real drop: small steps, shown at once
            history.append((i * .17, float(theta)))
            shown = tracker._smooth("exit", history, float(theta))
        self.assertEqual(25., shown)

    def test_memory_ignores_a_false_down_while_the_tracker_sees_the_arm_up(self):
        self.obj._note_arm_memory(self.status("up"))
        self.live("exit", 90.)
        self.obj._note_arm_memory(self.status("down"))
        self.assertEqual("up", self.obj._arm_memory["exit"]["state"])
        self.live("exit", 5.)                                   # the arm really came down
        self.obj._note_arm_memory(self.status("down"))
        self.assertEqual("down", self.obj._arm_memory["exit"]["state"])

    def test_without_a_tracker_reading_a_change_must_hold_a_second(self):
        self.obj._note_arm_memory(self.status("up"))
        self.obj._arm_live = {}
        self.obj._note_arm_memory(self.status("down"))
        self.assertEqual("up", self.obj._arm_memory["exit"]["state"])
        self.obj._arm_pending["exit"] = ("down", time.monotonic() - 1.5)
        self.obj._note_arm_memory(self.status("down"))
        self.assertEqual("down", self.obj._arm_memory["exit"]["state"])

    def test_a_car_under_a_raised_arm_never_acknowledges_an_alarm(self):
        self.obj._arm_last_known = {"exit": ("up", time.monotonic())}
        self.assertTrue(self.obj.raise_alarm("main", "exit corner", details={"route_role": "exit"}))
        self.live("exit", 90.)
        stamp = time.monotonic()
        with mock.patch.object(self.obj, "_local_frame_fresh", return_value=True):
            self.assertFalse(self.obj._ack_on_any_arm(self.status("down"), stamp))   # vetoed
            self.assertFalse(self.obj._ack_on_any_arm(self.status("up"), stamp))     # and not a "lift" either
        self.assertEqual("main", self.obj.alarm)
        self.assertEqual("up", self.obj._arm_last_known["exit"][0])

    def test_service_silencing_needs_the_tracker_to_agree(self):
        self.assertTrue(self.obj.raise_alarm("main", "exit corner", details={"route_role": "exit"}))
        self.live("exit", 90.)
        intent = {"type": "lowered", "alarm_id": self.obj.alarm_id, "route": "campus-exit"}
        self.assertFalse(self.obj._silence_for_barrier_change(intent))
        self.assertEqual("main", self.obj.alarm)


if __name__ == "__main__":
    unittest.main()
