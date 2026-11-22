"""The shared-grey / cached-mask barrier path returns exactly what the original path returns (20 Nov 2026)."""
from pathlib import Path
import json
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import barrier_monitor as bm  # noqa: E402
from barrier_monitor import BarrierMonitor  # noqa: E402
from barrier_workflow import BarrierWorkflow  # noqa: E402

H, W = bm.HEIGHT, bm.WIDTH


def textured(seed=3):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.uniform(30, 220, (H, W)).astype(np.float32), (7, 7), 0)
    return image


def as_rgb(gray):
    return np.repeat(np.asarray(gray, np.float32)[:, :, None], 3, axis=2)


class SpeedPathTests(unittest.TestCase):
    def setUp(self):
        self.now = 50.
        self.reference = textured()
        self.monitor = self.make()
        rng = np.random.default_rng(11)
        self.frames = []
        for k in range(24):
            frame = np.clip(self.reference * rng.uniform(.88, 1.12) + rng.uniform(-10, 10)
                            + rng.normal(0, 1.5, self.reference.shape), 0, 255)
            if k % 3 == 1:
                y, x = int(rng.integers(60, 300)), int(rng.integers(40, 580))
                frame[y:y + 30, x:x + 45] = rng.uniform(0, 255)
            self.frames.append(as_rgb(frame))
        self.route = np.zeros((H, W), bool)
        self.route[150:330, 100:560] = True

    def make(self):
        monitor = BarrierMonitor(clock=lambda: self.now)
        monitor._reference = self.reference
        monitor._road_reference = [self.reference]
        return monitor

    def feed(self, monitor, frame, **extra):
        self.now += .1
        return monitor.update(frame, self.now, aligned=True, **extra)

    def test_photometric_with_precomputed_indices_is_identical(self):
        mask = self.route
        indices = bm._photometric_indices(self.reference, mask)
        for frame in self.frames[:8]:
            current = bm._gray(frame)
            a = bm._photometric(self.reference, current, mask)
            b = bm._photometric(self.reference, current, mask, indices)
            self.assertEqual(a[1:], b[1:])
            if a[0] is None:
                self.assertIsNone(b[0])
            else:
                np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(bm._photometric(self.reference, current, np.zeros_like(mask))[1],
                         "Insufficient reference detail")

    def test_update_with_shared_grey_is_identical(self):
        fast, slow = self.make(), self.make()
        for frame in self.frames:
            a = self.feed(fast, frame, gray=BarrierMonitor.prepare_gray(frame))
            self.now -= .1
            b = self.feed(slow, frame)
            self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_road_check_fast_path_is_identical_and_covers_clear_and_changed(self):
        outcomes = set()
        for frame in self.frames:
            self.feed(self.monitor, frame)
            slow = self.monitor.road_clear(frame, self.route)
            fast = self.monitor.road_clear(frame, self.route, gray=BarrierMonitor.prepare_gray(frame), cache_key="exit")
            self.assertEqual(slow, fast)
            outcomes.add(slow["clear"])
        self.assertEqual(outcomes, {True, False})

    def test_caches_rebuild_for_new_masks_lines_and_references(self):
        frame = self.frames[0]
        self.feed(self.monitor, frame)
        self.monitor.road_clear(frame, self.route, cache_key="exit")
        first = self.monitor._road_mask_cache["exit"][2]
        other = self.route.copy(); other[:, :300] = False                  # a zone edit: a new mask object
        self.assertEqual(self.monitor.road_clear(frame, other, cache_key="exit"), self.monitor.road_clear(frame, other))
        self.assertIsNot(self.monitor._road_mask_cache["exit"][2], first)
        self.monitor.lines = dict(self.monitor.lines)                       # new barrier lines object
        self.monitor.road_clear(frame, other, cache_key="exit")
        self.assertIs(self.monitor._road_mask_cache["exit"][1], self.monitor.lines)
        self.monitor._road_reference = [textured(9)]                       # a new empty-road reference
        self.assertEqual(self.monitor.road_clear(frame, other, cache_key="exit"), self.monitor.road_clear(frame, other))

    def test_unusable_shared_grey_falls_back_to_the_frame(self):
        frame = self.frames[2]
        self.feed(self.monitor, frame)
        expected = self.monitor.road_clear(frame, self.route)
        for bad in (None, np.zeros((10, 10), np.float32), bm._gray(frame).astype(np.float64)):
            self.assertEqual(self.monitor.road_clear(frame, self.route, gray=bad, cache_key="exit"), expected)
        self.assertIsNone(BarrierMonitor.prepare_gray(np.zeros((5, 5, 3), np.uint8)))


class WorkflowCompatibilityTests(unittest.TestCase):
    def test_stand_in_monitor_without_the_new_hooks_keeps_the_old_calls(self):
        monitor = mock.Mock(spec=["update", "status", "road_clear", "reset"])
        monitor.status.return_value = {"barriers": []}
        monitor.road_clear.return_value = {"clear": False}
        workflow = BarrierWorkflow.__new__(BarrierWorkflow)
        workflow.monitor = monitor
        frame = np.zeros((H, W, 3), np.float32)
        gray = getattr(workflow.monitor, "SHARED_GRAY", False) is True
        self.assertFalse(gray)
        monitor.update(frame, 1., aligned=True)
        monitor.update.assert_called_once_with(frame, 1., aligned=True)


if __name__ == "__main__":
    unittest.main()
