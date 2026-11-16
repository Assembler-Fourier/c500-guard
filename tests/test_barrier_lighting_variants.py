"""A barrier must read correctly by floodlight and by daylight.

Measured on the real gate: templates calibrated at 01:52 at night recognised
their own captures with an error of 0.05-0.23, and scored 0.47-0.90 on the same
barriers in the afternoon - three times the 0.30 gate, so every barrier feature
went silent. Several lighting variants per position fix that, and calibrating in
one light must never destroy the calibration taken in another.

The fixtures are the real labelled captures, re-lit to stand for another hour of
the day: a synthetic drawn arm does not carry the texture the matcher measures.
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
    """The same scene under different light, as the camera would report it."""
    return np.clip(frame.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


@unittest.skipUnless((CALIBRATION / "both-down-check.png").is_file(),
                     "Labelled campus scene fixtures unavailable")
class LightingVariantTests(unittest.TestCase):
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
        self.storage = Path(folder.name) / "capture.npz"
        self.monitor = BarrierMonitor(self.storage, self.seed, SOURCE, clock=lambda: self.now)

    def feed(self, frames):
        for frame in frames:
            self.now += 0.12
            self.monitor.update(frame, self.now)
        return {row["id"]: row for row in self.monitor.status()["barriers"]}

    def calibrate(self, frames, identity, position):
        self.feed(frames)
        return self.monitor.capture(identity, position)

    def variants(self, identity, position):
        row = {r["id"]: r for r in self.monitor.status()["barriers"]}[identity]
        return row["lighting_variants"][position]

    # -- the fault this fixes ---------------------------------------------


    def test_calibrating_in_the_new_light_makes_it_readable(self):
        other_light = [relit(frame, 1.45, -28) for frame in self.groups["both-down"]]
        self.assertTrue(self.calibrate(other_light, "exit", "down")["ok"])
        check = [relit(frame, 1.45, -28) for frame in self.groups["both-down-check"]]
        self.assertEqual(self.feed(check)["exit"]["state"], "down")

    def test_the_original_calibration_still_works_afterwards(self):
        other_light = [relit(frame, 1.45, -28) for frame in self.groups["both-down"]]
        self.assertTrue(self.calibrate(other_light, "exit", "down")["ok"])
        self.assertEqual(self.variants("exit", "down"), 2)
        # The night pair must still be readable after the second era is stored.
        self.assertEqual(self.feed(self.groups["both-down-check"])["exit"]["state"], "down")
        self.assertEqual(self.feed(self.groups["exit-up"])["exit"]["state"], "up")

    # -- bounds and safety -------------------------------------------------

    def test_variants_are_bounded(self):
        for index in range(bm.MAX_VARIANTS + 2):
            gain = 1.0 + index * 0.22
            self.calibrate([relit(f, gain, -12 * index) for f in self.groups["both-down"]],
                           "exit", "down")
        self.assertLessEqual(self.variants("exit", "down"), bm.MAX_VARIANTS)

    def test_up_and_down_must_differ_in_every_stored_light(self):
        """A new reference that matches the opposite position anywhere is refused."""
        refused = self.calibrate(self.groups["both-down"], "exit", "up")
        self.assertFalse(refused["ok"])
        self.assertIn("distinctly different", refused["reason"])

    def test_an_old_single_variant_seed_is_migrated_not_rejected(self):
        legacy = Path(tempfile.mkdtemp()) / "legacy.npz"
        with np.load(self.seed, allow_pickle=False) as saved:
            np.savez_compressed(legacy, version=np.array(1), source_key=saved["source_key"],
                                templates=saved["templates"][:, :, 0],
                                counts=saved["counts"][:, :, 0],
                                reference=saved["reference"], road_reference=saved["road_reference"])
        monitor = BarrierMonitor(seed_path=legacy, source_key=SOURCE, clock=lambda: self.now)
        self.assertTrue(monitor.status()["configured"])
        self.monitor = monitor
        self.assertEqual(self.feed(self.groups["entrance-up"])["entrance"]["state"], "up")


if __name__ == "__main__":
    unittest.main()
