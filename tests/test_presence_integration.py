"""YuNet/baseline lifecycle integration with synthetic camera and detector calls."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from guard_presence import DeskBaseline, GuardPresence, Reading, looking_at_screen


class PresenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.now = 10.0
        self.obj = GuardPresence(clock=lambda: self.now,
            model_path=Path(self.folder.name) / 'no-model.onnx',
            baseline_path=Path(self.folder.name) / 'baseline.json')

    def test_learned_baseline_survives_stale_camera_but_attention_does_not(self):
        baseline = DeskBaseline(eye_gap=40, sideways=.5, pitch=1, samples=40)
        geometry = {'eye_gap': 40, 'sideways': .5, 'pitch': 1.0}
        self.obj.baseline = baseline
        self.obj.detector_name = 'YuNet'
        self.obj._looker = types.SimpleNamespace(baseline=baseline, last_geometry=geometry, last_score=.95)
        self.assertTrue(looking_at_screen(geometry, .95, baseline))
        self.obj.tracker.observe(Reading(at=self.now, face=True, eyes=True))
        self.assertIn('face_geometry', self.obj.status())
        self.now += self.obj.MAX_READING_AGE
        expired = self.obj.status()
        self.assertEqual(expired['state'], 'unknown')
        self.assertNotIn('face_geometry', expired)
        self.assertEqual(expired['detector'], 'YuNet')
        self.assertTrue(expired['baseline']['learned'])
        self.assertIs(self.obj._looker.baseline, baseline)
        self.obj.tracker.observe(Reading(at=self.now, face=True, eyes=None))
        self.assertEqual(self.obj.plan().state, 'present')
        self.obj.tracker.observe(Reading(at=self.now, face=True,
            eyes=looking_at_screen(geometry, .95, self.obj.baseline)))
        self.assertEqual(self.obj.plan().state, 'watching')

    def test_late_detector_result_after_stop_cannot_publish_or_calibrate(self):
        self.obj._running = True
        self.obj.start_calibration()
        self.obj._collect_calibration = mock.Mock()
        self.obj._finish_calibration = mock.Mock()
        self.obj._store_preview = mock.Mock()
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, object())
        detector = mock.Mock()

        def finish_after_stop(frame, at):
            self.obj.stop()
            return Reading(at=at, face=True, eyes=True)

        detector.look.side_effect = finish_after_stop
        self.obj._build_looker = mock.Mock(return_value=detector)
        fake_cv2 = types.SimpleNamespace(VideoCapture=mock.Mock(return_value=capture),
            CAP_DSHOW=700, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
        with mock.patch.dict(sys.modules, {'cv2': fake_cv2}):
            self.obj._loop()
        self.assertFalse(self.obj.tracker.camera_ok)
        self.assertEqual(self.obj.plan().state, 'unknown')
        self.obj._store_preview.assert_not_called()
        self.obj._collect_calibration.assert_not_called()
        self.obj._finish_calibration.assert_not_called()
        capture.release.assert_called_once_with()

    def test_finishing_calibration_updates_the_active_yunet_baseline(self):
        geometry = {'eye_gap': 40, 'sideways': .5, 'pitch': 1.0}
        self.obj._looker = types.SimpleNamespace(baseline=self.obj.baseline)
        self.assertFalse(looking_at_screen(geometry, .95, self.obj._looker.baseline))
        self.obj.start_calibration()
        for _ in range(12):
            self.obj._collect_calibration(dict(geometry), .95)
        self.obj._finish_calibration()
        self.assertIs(self.obj._looker.baseline, self.obj.baseline)
        self.assertTrue(looking_at_screen(geometry, .95, self.obj._looker.baseline))
        reloaded = GuardPresence(baseline_path=self.obj.baseline_path)
        self.assertTrue(reloaded.baseline.learned)
        self.assertEqual(reloaded.baseline.samples, 12)


if __name__ == '__main__':
    unittest.main()
