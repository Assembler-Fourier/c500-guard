"""Labelled calibration/held-out scenes and adverse-image regression tests."""
from pathlib import Path
import tempfile
import unittest
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from barrier_monitor import BarrierMonitor, LINES, create_seed


def _calibration_dir():
    """Use only explicitly supplied private calibration evidence."""
    import os
    supplied = os.environ.get("BARRIERWATCHER_CALIBRATION_FIXTURES")
    return Path(supplied) if supplied else Path(__file__).parent / "calibration-fixtures-unavailable"


CALIBRATION = _calibration_dir()
SOURCE = "regression-fixture-source"


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


@unittest.skipUnless((CALIBRATION/"both-down-check.png").is_file(), "Labelled campus scene fixtures unavailable")
class BarrierMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.photos = {label: rgb(CALIBRATION/f"{label}.png") for label in
                      ("both-down", "exit-up", "entrance-up", "both-down-check")}
        cls.groups = {label: [rgb(CALIBRATION/f"{label}-{index}.jpg") for index in range(7)]
                      for label in cls.photos}
        cls.temporary = tempfile.TemporaryDirectory()
        cls.seed = Path(cls.temporary.name)/"seed.npz"
        create_seed(cls.seed, cls.groups["both-down"], cls.groups["exit-up"],
                    cls.groups["entrance-up"], source_key=SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.now = 100.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.storage = Path(temporary.name)/"capture.npz"
        self.monitor = BarrierMonitor(self.storage, self.seed, SOURCE, clock=lambda: self.now)
        self.mask = np.zeros((360, 640), np.uint8)
        cv2.fillPoly(self.mask, [np.array([[75, 275], [260, 122], [319, 130], [130, 295]])], 1)

    def feed(self, photo, count=3, step=.2):
        for _ in range(count):
            self.now += step
            self.monitor.update(photo, self.now)
        return {item["id"]: item for item in self.monitor.status()["barriers"]}

    def test_all_labelled_jpegs_and_separate_pngs_match_user_positions(self):
        expected = {"both-down": ("down", "down"), "exit-up": ("up", "down"),
                    "entrance-up": ("down", "up"), "both-down-check": ("down", "down")}
        for group, states in expected.items():
            with self.subTest(group=group):
                for photo in self.groups[group]:
                    result = self.feed(photo, 1)
                self.assertEqual((result["exit"]["state"], result["entrance"]["state"]), states)
                result = self.feed(self.photos[group])
                self.assertEqual((result["exit"]["state"], result["entrance"]["state"]), states)

    def test_held_out_photos_were_not_used_in_seed(self):
        with np.load(self.seed, allow_pickle=False) as saved:
            # counts is (barrier, position, lighting variant); a seed fills the
            # first variant of each position and leaves the rest for other light.
            counts = saved["counts"]
            self.assertEqual(counts.shape, (2, 2, 4))
            self.assertTrue(np.array_equal(counts[:, :, 0], [[7, 7], [7, 7]]))
            self.assertTrue(np.array_equal(counts[:, :, 1:], np.zeros((2, 2, 3))))
            expected = np.median(np.stack([cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
                                           for photo in self.groups["both-down"]]), axis=0)
            # The empty road is stored as a stack of lighting variants; a seed
            # provides the first one.
            road = saved["road_reference"]
            self.assertEqual(road.shape, (1,) + expected.shape)
            self.assertTrue(np.array_equal(road[0], expected))
        result = self.feed(self.photos["both-down-check"])
        self.assertTrue(all(row["state"] == "down" for row in result.values()))

    def test_one_observation_or_repeated_stamp_cannot_confirm_up(self):
        self.monitor.update(self.photos["exit-up"], self.now)
        self.now += .5
        self.monitor.update(self.photos["exit-up"], self.now-.5)
        self.assertEqual(self.monitor.status()["barriers"][0]["state"], "unknown")
        self.monitor.update(self.photos["exit-up"], self.now)
        self.assertEqual(self.monitor.status()["barriers"][0]["state"], "up")

    def test_distinct_frames_need_minimum_stable_span(self):
        result = self.feed(self.photos["exit-up"], 2, .1)
        self.assertEqual(result["exit"]["state"], "unknown")
        result = self.feed(self.photos["exit-up"], 1, .2)
        self.assertEqual(result["exit"]["state"], "up")

    def test_state_transition_becomes_unknown_before_new_position_is_stable(self):
        self.feed(self.photos["exit-up"])
        result = self.feed(self.photos["both-down"], 1)
        self.assertEqual(result["exit"]["state"], "unknown")
        position_started = self.now
        self.assertEqual(position_started, result["exit"]["position_since"])
        result = self.feed(self.photos["both-down"], 2)
        self.assertEqual(result["exit"]["state"], "down")
        self.assertEqual(position_started, result["exit"]["position_since"])
        self.assertGreater(result["exit"]["observed_at"], position_started)

    def test_up_duration_advances_only_while_current_state_is_known(self):
        first = self.feed(self.photos["exit-up"])["exit"]["up_for_s"]
        later = self.feed(self.photos["exit-up"])["exit"]["up_for_s"]
        self.assertGreater(later, first)
        self.now += 3.01
        result = self.monitor.status()["barriers"][0]
        self.assertEqual(result["state"], "unknown")
        self.assertFalse(result["fresh"])
        self.assertEqual(result["up_for_s"], 0)

    def test_alignment_loss_and_stale_future_frames_never_preserve_up(self):
        for changes in (dict(aligned=False), dict(stamp_offset=-4), dict(stamp_offset=1)):
            with self.subTest(changes=changes):
                self.feed(self.photos["exit-up"])
                self.now += .2
                self.monitor.update(self.photos["exit-up"], self.now+changes.get("stamp_offset", 0),
                                    aligned=changes.get("aligned", True))
                self.assertTrue(all(item["state"] == "unknown" for item in self.monitor.status()["barriers"]))

    def test_invalid_frame_never_preserves_previous_state(self):
        for photo in (None, np.full((360, 640, 3), np.nan), np.zeros((20, 30, 3))):
            self.feed(self.photos["exit-up"])
            self.now += .2
            self.monitor.update(photo, self.now)
            self.assertTrue(all(item["state"] == "unknown" for item in self.monitor.status()["barriers"]))

    def test_adverse_exposure_black_white_and_obscured_view_are_unknown(self):
        for photo in (np.zeros_like(self.photos["exit-up"]), np.full_like(self.photos["exit-up"], 255),
                      (self.photos["exit-up"]*.20).astype(np.uint8)):
            self.feed(self.photos["exit-up"])
            result = self.feed(photo)
            self.assertTrue(all(row["state"] == "unknown" for row in result.values()))
            self.assertFalse(self.monitor.road_clear(photo, self.mask)["clear"])

    def test_barrier_occlusion_is_unknown_not_assumed_down(self):
        obscured = self.photos["exit-up"].copy()
        obscured[8:81, 350:375] = 90
        result = self.feed(obscured)
        self.assertEqual(result["exit"]["state"], "unknown")

    def test_conflicting_up_and_down_arm_evidence_is_unknown(self):
        conflict = self.photos["both-down"].copy()
        strip = np.zeros((360, 640), np.uint8)
        cv2.line(strip, (361, 14), (360, 67), 1, 15)
        conflict[strip != 0] = self.photos["exit-up"][strip != 0]
        result = self.feed(conflict)
        self.assertEqual(result["exit"]["state"], "unknown")

    def test_modest_gain_and_offset_do_not_invent_position_changes(self):
        photo = np.clip(self.photos["exit-up"].astype(float)*1.08+6, 0, 255).astype(np.uint8)
        result = self.feed(photo)
        self.assertEqual(result["exit"]["state"], "up")
        self.assertEqual(result["entrance"]["state"], "down")

    def test_exact_status_contract_and_geometry(self):
        result = self.feed(self.photos["exit-up"])
        expected = {"id", "name", "state", "confidence", "up_for_s", "fresh", "calibrated", "samples",
                    "lighting_variants", "lines", "active_line", "route_group", "observed_at",
                    "position_since", "candidate_position", "confirmation_s",
                    "match", "opposite_match", "background_match", "discriminative_match",
                    "opposite_discriminative_match", "reason"}
        for identity, row in result.items():
            self.assertEqual(set(row), expected)
            self.assertEqual(row["lines"], LINES[identity])
            self.assertEqual(row["active_line"], LINES[identity][row["state"]])
            self.assertEqual(row["route_group"], identity)

    def test_source_mismatch_never_uses_other_camera_seed(self):
        monitor = BarrierMonitor(seed_path=self.seed, source_key="another-view", clock=lambda: self.now)
        for _ in range(3):
            self.now += .2
            monitor.update(self.photos["exit-up"], self.now)
        self.assertFalse(monitor.status()["configured"])
        self.assertTrue(all(row["state"] == "unknown" for row in monitor.status()["barriers"]))
        self.assertFalse(monitor.road_clear(self.photos["both-down"], self.mask)["clear"])

    def test_capture_requires_five_fresh_stable_frames(self):
        self.feed(self.photos["both-down"], 4)
        self.assertFalse(self.monitor.capture("exit", "down")["ok"])
        self.feed(self.photos["both-down"], 1)
        self.assertTrue(self.monitor.capture("exit", "down")["ok"])
        self.now += 3.01
        self.assertFalse(self.monitor.capture("exit", "down")["ok"])

    def test_capture_rejects_identical_up_down_and_moving_labels(self):
        self.feed(self.photos["both-down"], 5)
        self.assertFalse(self.monitor.capture("exit", "up")["ok"])
        for index in range(9):
            self.feed(self.photos["exit-up" if index % 2 else "both-down"], 1)
        self.assertFalse(self.monitor.capture("exit", "down")["ok"])

    def test_capture_poor_contrast_is_rejected(self):
        self.feed(np.full_like(self.photos["both-down"], 90), 6)
        self.assertFalse(self.monitor.capture("exit", "down")["ok"])

    def test_capture_is_atomic_source_bound_and_does_not_modify_empty_reference(self):
        original = self.monitor._road_reference.copy()
        self.feed(self.photos["exit-up"], 7)
        self.assertTrue(self.monitor.capture("exit", "up")["ok"])
        self.assertTrue(self.storage.is_file())
        self.assertFalse(list(self.storage.parent.glob("*.tmp")))
        with np.load(self.storage, allow_pickle=False) as saved:
            self.assertEqual(saved["source_key"].item(), SOURCE)
            self.assertTrue(np.array_equal(saved["road_reference"], original))
        restored = BarrierMonitor(self.storage, source_key=SOURCE)
        self.assertTrue(restored.status()["configured"])
        self.assertFalse(BarrierMonitor(self.storage, source_key="wrong").status()["configured"])

    def test_reset_persists_and_does_not_reload_seed_over_user_reset(self):
        self.assertTrue(self.monitor.reset("exit")["ok"])
        restored = BarrierMonitor(self.storage, self.seed, SOURCE)
        self.assertFalse(restored.status()["barriers"][0]["calibrated"])
        self.assertTrue(restored.status()["barriers"][1]["calibrated"])

    def test_damaged_storage_falls_back_to_bound_seed(self):
        self.storage.write_bytes(b"broken reference")
        self.assertTrue(BarrierMonitor(self.storage, self.seed, SOURCE).status()["configured"])

    def test_invalid_capture_or_reset_ids_do_not_mutate_references(self):
        self.assertFalse(self.monitor.capture("other", "up")["ok"])
        self.assertFalse(self.monitor.capture("exit", "unknown")["ok"])
        self.assertFalse(self.monitor.reset("other")["ok"])
        self.assertTrue(self.monitor.status()["configured"])

    def test_frozen_empty_road_matches_independent_held_out_image(self):
        self.feed(self.photos["both-down-check"])
        result = self.monitor.road_clear(self.photos["both-down-check"], self.mask)
        self.assertTrue(result["clear"], result)
        self.assertGreaterEqual(result["matched_fraction"], .97)

    def test_parked_vehicle_never_becomes_a_new_empty_background(self):
        original = self.monitor._road_reference.copy()
        parked = self.photos["both-down"].copy()
        parked[205:230, 170:215] = 30
        for _ in range(12):
            self.feed(parked, 1)
            self.assertFalse(self.monitor.road_clear(parked, self.mask)["clear"])
        self.assertTrue(np.array_equal(self.monitor._road_reference, original))

    def test_small_coherent_patch_blocks_clear_even_with_over_97_percent_match(self):
        image = self.photos["both-down"].copy()
        image[210:216, 200:206] = 0
        self.feed(image)
        result = self.monitor.road_clear(image, self.mask)
        self.assertGreater(result["matched_fraction"], .97)
        self.assertGreater(result["largest_change_px"], 20)
        self.assertFalse(result["clear"])

    def test_road_matching_accepts_small_exposure_offset_but_not_adverse_exposure(self):
        for offset, expected in ((8, True), (90, False)):
            photo = np.clip(self.photos["both-down"].astype(float)+offset, 0, 255).astype(np.uint8)
            self.feed(photo)
            result = self.monitor.road_clear(photo, self.mask)
            self.assertEqual(result["clear"], expected, result)

    def test_stale_or_unaligned_road_cannot_release_a_visit(self):
        self.feed(self.photos["both-down"])
        self.now += 3.01
        self.assertFalse(self.monitor.road_clear(self.photos["both-down"], self.mask)["clear"])
        self.monitor.update(self.photos["both-down"], self.now, aligned=False)
        self.assertFalse(self.monitor.road_clear(self.photos["both-down"], self.mask)["clear"])

    def test_invalid_or_tiny_road_mask_is_not_clear(self):
        self.feed(self.photos["both-down"])
        for mask in (None, np.zeros((360, 640)), np.full((360, 640), np.nan)):
            self.assertFalse(self.monitor.road_clear(self.photos["both-down"], mask)["clear"])

    def test_seed_rejects_unlabelled_equivalent_positions(self):
        with self.assertRaises(ValueError):
            create_seed(self.storage, self.groups["both-down"], self.groups["both-down"],
                        self.groups["entrance-up"], source_key=SOURCE)


if __name__ == "__main__":
    unittest.main()
