"""Explicit Tiny/S adapter profiles with fake runtimes; no model or GPU work."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import object_detector as od


class DetectorProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = self.folder / 'explicit-artifact.onnx'
        self.model.write_bytes(b'isolated detector profile fixture')
        self.sha = hashlib.sha256(self.model.read_bytes()).hexdigest()

    def runtimes(self, size=640, *, gpu_shape=None, cpu_shape=None, fail_gpu_after=None):
        count = sum((size // stride) ** 2 for stride in (8, 16, 32))
        raw = np.zeros((1, count, 85), np.float32)
        raw[0, 0, :5] = [2, 3, np.log(2), np.log(4), .9]
        raw[0, 0, 7] = .8  # car, score .72
        cpu_inputs, gpu_inputs = [], []

        def cpu_run(_outputs, inputs):
            cpu_inputs.append(next(iter(inputs.values())).copy())
            return [raw.copy()]

        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(
                name='images', type='tensor(float)',
                shape=[1, 3, size, size] if cpu_shape is None else cpu_shape)],
            run=Mock(side_effect=cpu_run))
        ort = SimpleNamespace(
            SessionOptions=Mock(side_effect=lambda: SimpleNamespace(add_session_config_entry=Mock())),
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL='all'),
            InferenceSession=Mock(return_value=session))

        def gpu_infer(inputs):
            gpu_inputs.append(next(iter(inputs.values())).copy())
            if fail_gpu_after is not None and len(gpu_inputs) > fail_gpu_after:
                raise RuntimeError('fake graphics failure')
            return {'output': raw.copy()}

        request = SimpleNamespace(infer=Mock(side_effect=gpu_infer))
        compiled = SimpleNamespace(create_infer_request=lambda: request, output=lambda _index: 'output')
        graph = SimpleNamespace(input=lambda _index: SimpleNamespace(
            get_shape=lambda: [1, 3, size, size] if gpu_shape is None else gpu_shape))
        core = SimpleNamespace(available_devices=['GPU'], set_property=Mock(),
                               read_model=Mock(return_value=graph), compile_model=Mock(return_value=compiled))
        ov = SimpleNamespace(Core=Mock(return_value=core))
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(sys.modules, {'onnxruntime': ort, 'openvino': ov}))
        stack.enter_context(patch.object(od, 'OPENVINO_DIR', self.folder / 'fake-openvino'))
        return SimpleNamespace(ort=ort, core=core, cpu_inputs=cpu_inputs, gpu_inputs=gpu_inputs)

    def detector(self, **kwargs):
        defaults = dict(model_path=self.model, expected_sha256=self.sha,
                        input_size=(640, 640), model_label='YOLOX-S COCO pretrained')
        return od.ObjectDetector(**{**defaults, **kwargs})

    def test_legacy_default_size_label_and_checksum_remain_tiny(self):
        with patch.object(od.ObjectDetector, '_open_processor') as processor:
            rejected = od.ObjectDetector(self.model, engine='processor')
        self.assertFalse(rejected.available)
        self.assertIn('checksum', rejected.error)
        processor.assert_not_called()
        self.assertEqual('427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7',
                         od.MODEL_SHA256)
        self.runtimes(416)
        tiny = od.ObjectDetector(self.model, expected_sha256=self.sha, engine='processor')
        self.assertTrue(tiny.available, tiny.error)
        self.assertEqual([416, 416], tiny.status()['input_size'])
        self.assertEqual('YOLOX-Tiny COCO pretrained', tiny.status()['model'])

    def test_explicit_processor_profile_preprocesses_and_decodes_640(self):
        runtime = self.runtimes()
        detector = self.detector(engine='processor')
        frame = np.full((360, 640, 3), [255, 20, 3], np.uint8)
        objects = detector.detect(frame)
        self.assertTrue(detector.available, detector.error)
        self.assertEqual((1, 3, 640, 640), runtime.cpu_inputs[0].shape)
        np.testing.assert_equal(runtime.cpu_inputs[0][0, :, 0, 0], [3, 20, 255])
        np.testing.assert_equal(runtime.cpu_inputs[0][0, :, 400, 0], [114, 114, 114])
        self.assertEqual(1, len(objects))
        self.assertEqual('car', objects[0]['label'])
        self.assertEqual([8., 8., 24., 40.], objects[0]['bbox'])
        self.assertEqual('YOLOX-S COCO pretrained', detector.status()['model'])
        self.assertEqual([640, 640], detector.status()['input_size'])
        self.assertEqual('CPU', detector.status()['provider'])
        self.assertEqual('f32', detector.status()['precision'])
        self.assertEqual(1, detector.status()['frames_processed'])

    def test_graphics_warmup_and_detection_use_requested_640_profile(self):
        runtime = self.runtimes()
        detector = self.detector(engine='graphics')
        self.assertTrue(detector.available, detector.error)
        self.assertEqual((1, 3, 640, 640), runtime.gpu_inputs[0].shape)
        self.assertEqual(np.float32, runtime.gpu_inputs[0].dtype)
        self.assertFalse(runtime.gpu_inputs[0].any())
        self.assertEqual(1, len(detector.detect(np.zeros((360, 640, 3), np.uint8))))
        self.assertEqual((1, 3, 640, 640), runtime.gpu_inputs[1].shape)
        runtime.ort.InferenceSession.assert_not_called()
        self.assertEqual('Intel graphics (OpenVINO)', detector.status()['provider'])
        self.assertEqual('f16', detector.status()['precision'])
        self.assertEqual('f16', runtime.core.compile_model.call_args.args[2]['INFERENCE_PRECISION_HINT'])

    def test_explicit_f32_graphics_precision_reaches_runtime_and_status(self):
        runtime = self.runtimes()
        detector = self.detector(engine='graphics', gpu_precision='f32')
        self.assertTrue(detector.available, detector.error)
        self.assertEqual('f32', runtime.core.compile_model.call_args.args[2]['INFERENCE_PRECISION_HINT'])
        self.assertEqual('f32', detector.status()['precision'])
        self.assertEqual('Intel graphics (OpenVINO)', detector.status()['provider'])
        self.assertEqual((1, 3, 640, 640), runtime.gpu_inputs[0].shape)

    def test_graphics_initial_failure_falls_back_without_changing_profile(self):
        runtime = self.runtimes(fail_gpu_after=0)
        detector = self.detector(engine='graphics')
        self.assertTrue(detector.available, detector.error)
        self.assertIn('fake graphics failure', detector.engine_note)
        self.assertEqual('CPU', detector.provider)
        self.assertEqual('f32', detector.status()['precision'])
        self.assertEqual(1, len(detector.detect(np.zeros((360, 640, 3), np.uint8))))
        self.assertEqual((1, 3, 640, 640), runtime.cpu_inputs[0].shape)
        self.assertEqual('YOLOX-S COCO pretrained', detector.status()['model'])

    def test_mid_inference_graphics_failure_reuses_640_tensor_on_processor(self):
        runtime = self.runtimes(fail_gpu_after=1)
        detector = self.detector(engine='graphics')
        self.assertEqual(1, len(detector.detect(np.zeros((360, 640, 3), np.uint8))))
        self.assertTrue(detector.available, detector.error)
        self.assertEqual('CPU', detector.provider)
        self.assertEqual('f32', detector.status()['precision'])
        self.assertEqual([640, 640], detector.status()['input_size'])
        self.assertEqual('YOLOX-S COCO pretrained', detector.status()['model'])
        np.testing.assert_array_equal(runtime.gpu_inputs[1], runtime.cpu_inputs[0])

    def test_processor_rejects_actual_shape_mismatch_or_dynamic_shape(self):
        for shape in ([1, 3, 416, 416], [1, 640, 640, 3], ['batch', 3, 640, 640]):
            with self.subTest(shape=shape):
                runtime = self.runtimes(cpu_shape=shape)
                detector = self.detector(engine='processor')
                self.assertFalse(detector.available)
                self.assertIn('expected [1, 3, 640, 640]', detector.error)
                self.assertEqual([], detector.detect(np.zeros((10, 10, 3), np.uint8)))
                self.assertEqual([], runtime.cpu_inputs)

    def test_optional_graphics_initial_failure_never_opens_processor(self):
        runtime = self.runtimes(fail_gpu_after=0)
        detector = self.detector(engine='graphics', allow_cpu_fallback=False)
        self.assertFalse(detector.available)
        self.assertIn('CPU fallback disabled', detector.error)
        runtime.ort.InferenceSession.assert_not_called()
        self.assertEqual([], runtime.cpu_inputs)

    def test_optional_graphics_runtime_failure_never_opens_or_runs_processor(self):
        runtime = self.runtimes(fail_gpu_after=1)
        detector = self.detector(engine='graphics', allow_cpu_fallback=False)
        self.assertTrue(detector.available, detector.error)
        self.assertEqual([], detector.detect(np.zeros((360, 640, 3), np.uint8)))
        self.assertFalse(detector.available)
        self.assertIn('CPU fallback disabled', detector.error)
        self.assertIsNone(detector.status()['precision'])
        runtime.ort.InferenceSession.assert_not_called()
        self.assertEqual([], runtime.cpu_inputs)

    def test_fallback_setting_requires_explicit_boolean(self):
        for value in (None, 'false', 0, 1, []):
            with self.subTest(value=value), patch.object(od.ObjectDetector, '_open_graphics') as graphics:
                detector = self.detector(engine='graphics', allow_cpu_fallback=value)
                self.assertFalse(detector.available)
                self.assertIn('CPU fallback setting', detector.error)
                graphics.assert_not_called()

    def test_graphics_rejects_shape_before_compile_and_cpu_revalidates(self):
        runtime = self.runtimes(gpu_shape=[1, 3, 416, 416], cpu_shape=[1, 3, 416, 416])
        detector = self.detector(engine='graphics')
        self.assertFalse(detector.available)
        self.assertIn('expected [1, 3, 640, 640]', detector.engine_note)
        self.assertIn('expected [1, 3, 640, 640]', detector.error)
        runtime.core.compile_model.assert_not_called()
        self.assertEqual([], runtime.gpu_inputs)

    def test_invalid_requested_profile_fails_before_runtime_loading(self):
        for size in (None, 640, (640,), (416, 640), (320, 320), (640., 640.), (True, True)):
            with self.subTest(size=size), patch.object(od.ObjectDetector, '_open_processor') as processor:
                detector = self.detector(input_size=size, engine='processor')
                self.assertFalse(detector.available)
                self.assertIn('input size', detector.error)
                processor.assert_not_called()
        for label in ('', ' ', None, 'wrong\nlabel', 'x' * 121):
            with self.subTest(label=label):
                detector = self.detector(model_label=label, engine='processor')
                self.assertFalse(detector.available)
                self.assertIn('model label', detector.error)

    def test_invalid_graphics_precision_fails_before_runtime_loading(self):
        for precision in (None, 'auto', 'FP32', 32, True, [], {}):
            with self.subTest(precision=precision), patch.object(od.ObjectDetector, '_open_graphics') as graphics:
                detector = self.detector(gpu_precision=precision, engine='graphics')
                self.assertFalse(detector.available)
                self.assertIn('graphics precision', detector.error)
                self.assertIsNone(detector.status()['precision'])
                graphics.assert_not_called()

    def test_explicit_secondary_profile_still_requires_supplied_checksum(self):
        with patch.object(od.ObjectDetector, '_open_processor') as processor:
            detector = self.detector(expected_sha256='0' * 64, engine='processor')
        self.assertFalse(detector.available)
        self.assertIn('checksum', detector.error)
        processor.assert_not_called()
        self.assertEqual('YOLOX-S COCO pretrained', detector.status()['model'])


if __name__ == '__main__':
    unittest.main()
