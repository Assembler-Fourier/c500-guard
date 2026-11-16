"""AI on the Intel graphics when it works, the processor whenever it does not - never no AI."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import test_watch  # noqa: F401 - puts src on the path
import object_detector as od

MODEL = Path(r"C:\c500-guard\models\yolox_tiny.onnx")
OPENVINO = Path(r"C:\c500-guard\tools\openvino")


@unittest.skipUnless(MODEL.is_file(), "installed model not available")
class EngineTests(unittest.TestCase):
    def frame(self):
        return np.zeros((360, 640, 3), np.uint8)

    def test_missing_graphics_runtime_falls_back_to_the_processor(self):
        with mock.patch.object(od, 'OPENVINO_DIR', Path(tempfile.gettempdir()) / 'no-openvino-here'), \
                mock.patch.dict('sys.modules', {'openvino': None}):
            detector = od.ObjectDetector(MODEL, engine='graphics')
        self.assertTrue(detector.available)
        self.assertEqual('CPU', detector.status()['provider'])
        self.assertIn('using the processor', detector.status()['engine_note'])
        self.assertEqual([], detector.detect(self.frame()))

    def test_processor_engine_never_touches_the_graphics(self):
        with mock.patch.object(od.ObjectDetector, '_open_graphics') as graphics:
            detector = od.ObjectDetector(MODEL, engine='processor')
        graphics.assert_not_called()
        self.assertEqual('CPU', detector.provider)

    @unittest.skipUnless(OPENVINO.is_dir(), "OpenVINO not installed in the tool folder")
    def test_graphics_failure_mid_shift_switches_to_the_processor_and_keeps_detecting(self):
        with mock.patch.object(od, 'OPENVINO_DIR', OPENVINO):
            detector = od.ObjectDetector(MODEL, engine='graphics')
        if detector.provider == 'CPU':
            self.skipTest('no usable Intel graphics on this machine: ' + detector.engine_note)
        detector._request = mock.Mock()
        detector._request.infer.side_effect = RuntimeError('GPU device lost')
        self.assertEqual([], detector.detect(self.frame()))
        self.assertTrue(detector.available)
        self.assertEqual('CPU', detector.provider)
        self.assertIn('switched to the processor', detector.engine_note)
        detector.detect(self.frame())           # keeps working on the processor
        self.assertTrue(detector.available)


class SettingTests(unittest.TestCase):
    def test_engine_setting_loads_with_a_safe_default(self):
        from test_watch import watch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            for stored, expected in ((None, 'graphics'), ('processor', 'processor'), ('gpu-turbo', 'graphics')):
                path.write_text(json.dumps({'ai': {} if stored is None else {'engine': stored}}))
                self.assertEqual(expected, watch.Watcher._load_ai_engine(path))


if __name__ == '__main__':
    unittest.main()
