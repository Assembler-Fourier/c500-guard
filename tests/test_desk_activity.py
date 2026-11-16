from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from desk_activity import DeskActivity
from guard_presence import GuardPresence, Reading


def scene():
    y, x = np.indices((120, 160))
    gray = (60 + (x//8 % 2)*35 + (y//8 % 2)*30).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def moved(frame=None):
    frame = scene() if frame is None else frame.copy()
    frame[80:98, 60:83] = 210
    return frame


class DeskActivityTests(unittest.TestCase):
    def setUp(self):
        self.activity = DeskActivity()
        self.frame = scene()

    def test_initial_view_unknown_then_static_scene_quiet(self):
        self.assertEqual('unknown', self.activity.update(self.frame, 0, now=0)['state'])
        report = self.activity.update(self.frame, .25, now=.25)
        self.assertEqual('quiet', report['state'])
        self.assertTrue(report['ready'])
        self.assertFalse(report['acknowledges'])
        self.assertEqual(.25, report['observed_at'])

    def test_local_desk_movement_has_measured_coverage_and_two_second_hold(self):
        self.activity.update(self.frame, 0, now=0)
        shifted = moved()
        first = self.activity.update(shifted, .25, now=.25)
        self.assertEqual('movement', first['state'])
        self.assertGreater(first['coverage'], .004)
        for at in (.5, 1., 1.5, 2., 2.25):
            report = self.activity.update(shifted, at, now=at)
            self.assertEqual('movement', report['state'])
        self.assertEqual('quiet', self.activity.status(2.26)['state'])
        self.assertFalse(self.activity.status(2.26)['acknowledges'])

    def test_activity_cannot_stay_fresh_without_new_source_frames(self):
        self.activity.update(self.frame, 0, now=0)
        self.activity.update(moved(), .25, now=.25)
        for at in (.5, 1, 2):
            report = self.activity.update(moved(), .25, now=at)
            self.assertEqual(.25, report['observed_at'])
        stale = self.activity.status(3.25)
        self.assertEqual('unknown', stale['state'])
        self.assertFalse(stale['ready'])
        self.assertEqual(0., stale['hold_remaining_s'])

    def test_covered_black_white_and_featureless_cameras_report_unknown(self):
        for level in (0, 100, 255):
            with self.subTest(level=level):
                self.activity.update(self.frame, 0, now=0)
                covered = np.full_like(self.frame, level)
                report = self.activity.update(covered, 1, now=1)
                self.assertEqual('unknown', report['state'])
                self.assertFalse(report['ready'])
                self.assertFalse(report['acknowledges'])
                self.activity = DeskActivity()

    def test_uniform_exposure_step_does_not_become_desk_movement(self):
        self.activity.update(self.frame, 0, now=0)
        bright = self.frame + 30
        report = self.activity.update(bright, .25, now=.25)
        self.assertEqual('quiet', report['state'])
        self.assertEqual(30., report['exposure_shift'])
        self.assertEqual(0., report['coverage'])

    def test_upper_view_motion_is_not_desk_motion(self):
        self.activity.update(self.frame, 0, now=0)
        upper = self.frame.copy()
        upper[10:30, 60:83] = 210
        self.assertEqual('quiet', self.activity.update(upper, .25, now=.25)['state'])

    def test_current_and_previous_face_boxes_are_excluded(self):
        self.activity.update(self.frame, 0, now=0, face_box=(55, 75, 30, 25))
        report = self.activity.update(moved(), .25, now=.25, face_box=(90, 75, 30, 25))
        self.assertEqual('quiet', report['state'])
        self.assertTrue(report['face_excluded'])

    def test_large_view_shift_is_unknown(self):
        self.activity.update(self.frame, 0, now=0)
        shifted = np.roll(self.frame, 8, axis=1)
        report = self.activity.update(shifted, .25, now=.25)
        self.assertEqual('unknown', report['state'])
        self.assertEqual('broad_view_change', report['reason'])

    def test_sustained_identical_pixels_are_static_or_frozen_unknown(self):
        for at in (0, .5, 1, 1.5, 2, 2.5):
            self.activity.update(self.frame, at, now=at)
        report = self.activity.update(self.frame, 3, now=3)
        self.assertEqual('unknown', report['state'])
        self.assertEqual('static_or_frozen_view', report['reason'])
        self.assertFalse(report['ready'])
        self.assertEqual('movement', self.activity.update(moved(), 3.25, now=3.25)['state'])

    def test_source_gap_reestablishes_baseline_without_inventing_gesture(self):
        self.activity.update(self.frame, 0, now=0)
        report = self.activity.update(moved(), 4, now=4)
        self.assertEqual('unknown', report['state'])
        self.assertEqual('establishing_fresh_view', report['reason'])

    def test_stale_future_and_failed_camera_never_provide_activity(self):
        for stamp, ready in ((0, True), (5, True), (float('nan'), True), (True, True), (4, False)):
            report = self.activity.update(moved(), stamp, now=4, camera_ready=ready)
            self.assertEqual('unknown', report['state'])
            self.assertFalse(report['acknowledges'])

    def test_invalidation_discards_samples_and_recovery_starts_unknown(self):
        self.activity.update(self.frame, 0, now=0)
        self.activity.update(moved(), .25, now=.25)
        self.activity.invalidate()
        self.assertIsNone(self.activity._previous)
        self.assertEqual('unknown', self.activity.status(.5)['state'])
        self.assertEqual('unknown', self.activity.update(self.frame, 1, now=1)['state'])

    def test_input_is_not_mutated_and_status_snapshot_is_detached(self):
        original = self.frame.copy()
        self.activity.update(self.frame, 0, now=0)
        self.assertTrue(np.array_equal(self.frame, original))
        status = self.activity.status(0)
        status['roi'][0] = 999
        self.assertEqual(.05, self.activity.status(0)['roi'][0])


class PresenceDeskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.now = 10.
        self.presence = GuardPresence(clock=lambda: self.now,
                                      baseline_path=Path(self.folder.name)/'no-baseline.json')

    def observe(self, frame, at):
        self.now = at
        reading = Reading(at, face=False)
        self.presence._observe_desk(frame, reading)
        self.presence.tracker.observe(reading)

    def test_status_publishes_advisory_signal_without_starting_thread_or_files(self):
        self.observe(scene(), 10)
        self.observe(moved(), 10.25)
        report = self.presence.status()
        self.assertEqual('movement', report['desk_activity']['state'])
        self.assertFalse(report['desk_activity']['acknowledges'])
        self.assertIsNone(self.presence._thread)
        self.assertEqual([], list(Path(self.folder.name).iterdir()))
        self.assertFalse(report['sleep_detection_supported'])

    def test_camera_failure_stop_and_stale_presence_reset_activity(self):
        self.observe(scene(), 10)
        self.observe(moved(), 10.25)
        self.presence._camera_failed('test camera lost')
        self.assertEqual('unknown', self.presence.status()['desk_activity']['state'])
        self.observe(scene(), 11)
        self.observe(moved(), 11.25)
        self.now = 14.25
        self.assertEqual('unknown', self.presence.status()['desk_activity']['state'])
        self.presence.stop()
        self.assertIsNone(self.presence._desk_activity._previous)
        self.assertFalse(self.presence.status()['running'])

    def test_optional_analysis_failure_does_not_change_presence_or_acknowledge(self):
        reading = Reading(10, face=True, attention=True)
        self.presence.tracker.observe(reading)
        self.presence._observe_desk(None, reading)
        self.assertEqual('watching', self.presence.status()['state'])
        self.assertEqual('unknown', self.presence.status()['desk_activity']['state'])
        self.assertFalse(self.presence.status()['desk_activity']['acknowledges'])


if __name__ == '__main__':
    unittest.main()
