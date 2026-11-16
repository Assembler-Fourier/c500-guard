"""Pose and missing eye detections must never invent a sleeping operator."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guard_presence import (DeskBaseline, FaceLook, GuardPresence,
                            PresenceTracker, Reading, YuNetLook)


class PresenceEvidenceTests(unittest.TestCase):
    def test_yunet_still_downward_head_cannot_become_asleep(self):
        looker = YuNetLook.__new__(YuNetLook)
        looker.cv2 = cv2
        looker.baseline = DeskBaseline()
        # Valid face; mouth close to eye level indicates a downward head pose.
        row = np.array([80, 50, 120, 120, 100, 80, 140, 80,
                        120, 84, 106, 88, 134, 88, .95], dtype=np.float32)
        looker.detector = mock.Mock()
        looker.detector.detect.return_value = (1, np.array([row]))
        frame = np.full((240, 320, 3), 100, np.uint8)
        tracker = PresenceTracker()
        for at in range(121):
            reading = looker.look(frame, float(at))
            self.assertIsNone(reading.eyes)
            self.assertFalse(reading.attention)
            tracker.observe(reading)
        self.assertEqual(tracker.state, "present")
        self.assertIsNone(tracker.open_eye_ratio())
        self.assertIsNone(tracker.closed_since)

    def test_yunet_screen_pose_preserves_attention_without_eye_claim(self):
        looker = YuNetLook.__new__(YuNetLook)
        looker.cv2 = cv2
        looker.baseline = DeskBaseline()
        row = np.array([80, 50, 120, 120, 100, 80, 140, 80,
                        120, 102, 106, 120, 134, 120, .95], dtype=np.float32)
        looker.detector = mock.Mock()
        looker.detector.detect.return_value = (1, np.array([row]))
        reading = looker.look(np.full((240, 320, 3), 100, np.uint8), 1.0)
        self.assertTrue(reading.attention)
        self.assertIsNone(reading.eyes)
        tracker = PresenceTracker()
        self.assertEqual(tracker.observe(reading), "watching")
        self.assertIsNone(tracker.status(1.0)["eyes_age_s"])
        self.assertEqual(tracker.status(1.0)["attention_age_s"], 0.0)

    def test_haar_missing_eye_region_does_not_claim_closed_eyes(self):
        looker = FaceLook.__new__(FaceLook)
        looker.cv2 = cv2
        looker.face, looker.profile, looker.eyes = mock.Mock(), mock.Mock(), mock.Mock()
        looker.face.detectMultiScale.return_value = [(80, 50, 120, 120)]
        looker.eyes.detectMultiScale.return_value = []
        frame = np.full((240, 320, 3), 100, np.uint8)
        tracker = PresenceTracker()
        for at in range(121):
            reading = looker.look(frame, float(at))
            self.assertIsNone(reading.eyes)
            tracker.observe(reading)
        self.assertEqual(tracker.state, "present")

    def test_measured_closure_is_interrupted_by_unknown_eyes(self):
        tracker = PresenceTracker()
        for at in range(81):
            tracker.observe(Reading(at, face=True, eyes=False, box=(100, 100, 100, 100)))
        self.assertEqual(tracker.state, "asleep")
        self.assertEqual(tracker.observe(Reading(81, face=True, eyes=None, attention=False)), "present")
        self.assertIsNone(tracker.closed_since)
        self.assertIsNone(tracker.open_eye_ratio())
        self.assertEqual(tracker.observe(Reading(82, face=True, eyes=False)), "present")

    def test_lost_face_cannot_supply_closed_eyes(self):
        tracker = PresenceTracker()
        tracker.observe(Reading(0, face=True, attention=True))
        tracker.observe(Reading(1, face=False, eyes=False))
        self.assertIsNone(tracker.closed_since)
        self.assertIsNone(tracker.open_eye_ratio())

    def test_camera_failure_resets_attention(self):
        tracker = PresenceTracker()
        tracker.observe(Reading(1, face=True, attention=True))
        tracker.camera_failed(2, "unavailable")
        tracker.observe(Reading(3, face=True, attention=False))
        self.assertEqual(tracker.state, "present")
        self.assertIsNone(tracker.last_attention_at)

    def test_live_status_reports_actual_model_capabilities(self):
        presence = GuardPresence()
        presence.detector_name = "YuNet"
        state = presence.status()
        self.assertEqual(state["attention_method"], "head pose")
        self.assertFalse(state["sleep_detection_supported"])
        self.assertFalse(state["eye_openness_supported"])


if __name__ == "__main__":
    unittest.main()
