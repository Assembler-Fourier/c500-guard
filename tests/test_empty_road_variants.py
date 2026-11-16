""""The car has gone and the road looks normal" must work at every hour.

The empty-road reference was frozen once, at night. In the evening the same
empty road reported "Road illumination is outside the empty reference", so the
road could never be called clear and the automatic acknowledgement the guard
asked for could never complete. Several frozen references fix that; a road with
something on it must still never be called clear.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import barrier_monitor as bm  # noqa: E402
from barrier_monitor import BarrierMonitor, create_seed  # noqa: E402

def _calibration_dir():
    """Use only explicitly supplied private calibration evidence."""
    import os
    supplied = os.environ.get("BARRIERWATCHER_CALIBRATION_FIXTURES")
    return Path(supplied) if supplied else Path(__file__).parent / "calibration-fixtures-unavailable"


CALIBRATION = _calibration_dir()
SOURCE = "regression-fixture-source"


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def relit(frame, gain, offset):
    return np.clip(frame.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


@unittest.skipUnless((CALIBRATION / "both-down-check.png").is_file(),
                     "Labelled campus scene fixtures unavailable")
class EmptyRoadVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = {label: [rgb(CALIBRATION / f"{label}-{index}.jpg") for index in range(7)]
                      for label in ("both-down", "exit-up", "entrance-up", "both-down-check")}
        cls.temporary = tempfile.TemporaryDirectory()
        cls.seed = Path(cls.temporary.name) / "seed.npz"
        create_seed(cls.seed, cls.groups["both-down"], cls.groups["exit-up"],
                    cls.groups["entrance-up"], source_key=SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.now = 100.0
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.monitor = BarrierMonitor(Path(folder.name) / "capture.npz", self.seed, SOURCE,
                                      clock=lambda: self.now)
        # The exit lane only: the same road polygon the watcher uses, so the
        # bezel and the far verge cannot decide whether the road is clear.
        self.mask = np.zeros((bm.HEIGHT, bm.WIDTH), np.uint8)
        cv2.fillPoly(self.mask, [np.array([[75, 275], [260, 122], [319, 130], [130, 295]])], 1)

    def feed(self, frames):
        for frame in frames:
            self.now += 0.12
            self.monitor.update(frame, self.now)

    def clear(self, frame):
        return self.monitor.road_clear(frame, self.mask)

    def test_the_seeded_night_road_is_recognised(self):
        self.feed(self.groups["both-down-check"])
        self.assertTrue(self.clear(self.groups["both-down-check"][-1])["clear"])

    def recalibrate_arms_for(self, frames):
        """The operating order: the arms must be readable before the road."""
        for identity in bm.IDS:
            self.feed(frames)
            saved = self.monitor.capture(identity, "down")
            self.assertTrue(saved["ok"], saved.get("reason"))

    def test_a_differently_lit_empty_road_is_not_recognised_until_it_is_frozen(self):
        # Freeze from the still calibration set; verify on the held-out set.
        evening = [relit(frame, 1.4, -22) for frame in self.groups["both-down"]]
        check = [relit(frame, 1.4, -22) for frame in self.groups["both-down-check"]]
        self.feed(check)
        self.assertFalse(self.clear(check[-1])["clear"])

        self.recalibrate_arms_for(evening)
        self.feed(evening)
        saved = self.monitor.capture_road()
        self.assertTrue(saved["ok"], saved.get("reason"))
        self.feed(check)
        self.assertTrue(self.clear(check[-1])["clear"])

    def test_the_road_cannot_be_frozen_while_an_arm_is_unreadable(self):
        """An obscured arm is not a settled arm, so the road cannot be frozen."""
        obscured = []
        for frame in self.groups["both-down"]:
            copy = frame.copy()
            copy[55:105, 250:500] = 15          # something blocking both arm lines
            obscured.append(copy)
        self.feed(obscured)
        refused = self.monitor.capture_road()
        self.assertFalse(refused["ok"])
        self.assertIn("settled and readable", refused["reason"])

    def test_freezing_the_evening_road_keeps_the_night_one(self):
        evening = [relit(frame, 1.4, -22) for frame in self.groups["both-down"]]
        self.recalibrate_arms_for(evening)
        self.feed(evening)
        self.assertTrue(self.monitor.capture_road()["ok"])
        self.feed(self.groups["both-down-check"])
        self.assertTrue(self.clear(self.groups["both-down-check"][-1])["clear"],
                        "the original night reference must keep working")

    def test_something_standing_on_the_road_is_never_clear(self):
        """The arms themselves are masked out; anything else on the road is not."""
        blocked = self.groups["both-down-check"][-1].copy()
        blocked[150:210, 140:210] = 30          # a dark shape in the middle of the lane
        self.feed(self.groups["both-down-check"])
        reading = self.clear(blocked)
        self.assertFalse(reading["clear"])
        self.assertLess(reading["matched_fraction"], 0.97)

    def test_a_clear_reading_needs_a_fresh_view(self):
        self.feed(self.groups["both-down-check"])
        self.assertTrue(self.clear(self.groups["both-down-check"][-1])["clear"])
        self.now += 30                          # the camera stopped delivering
        self.assertFalse(self.clear(self.groups["both-down-check"][-1])["clear"])

    def test_references_are_bounded(self):
        for index in range(bm.MAX_VARIANTS + 3):
            frames = [relit(frame, 1.0 + index * .12, -6 * index)
                      for frame in self.groups["both-down-check"]]
            self.feed(frames)
            self.monitor.capture_road()
        self.assertLessEqual(len(self.monitor._road_reference), bm.MAX_VARIANTS)

    def test_a_moving_scene_is_refused(self):
        mixed = [self.groups["both-down-check"][0], self.groups["exit-up"][0]] * 3
        self.feed(mixed)
        refused = self.monitor.capture_road()
        self.assertFalse(refused["ok"])

    def test_a_stale_view_is_refused(self):
        self.feed(self.groups["both-down-check"])
        self.now += 30
        self.assertFalse(self.monitor.capture_road()["ok"])
        self.assertFalse(self.clear(self.groups["both-down-check"][-1])["clear"])


if __name__ == "__main__":
    unittest.main()
