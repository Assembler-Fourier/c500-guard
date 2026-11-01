"""Meaningful ML contract tests; real-model smoke test is opt-in via env var."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_detector import ObjectDetector, preprocess_rgb, decode_detections


class DetectorTests(unittest.TestCase):
    def raw_prediction(self, class_id=0, objectness=.9, class_score=.8):
        raw = np.zeros((1, 3549, 85), np.float32)
        # First grid coordinate has stride 8; center (16,24), size (16,32).
        raw[0, 0, :5] = [2, 3, np.log(2), np.log(4), objectness]
        raw[0, 0, 5 + class_id] = class_score
        return raw

    def test_rgb_conversion_padding_and_no_normalization(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[:] = [255, 20, 3]
        tensor, ratio = preprocess_rgb(frame, (416, 416))
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(ratio, 2.08)
        np.testing.assert_equal(tensor[0, :, 0, 0], [3, 20, 255])
        np.testing.assert_equal(tensor[0, :, 300, 0], [114, 114, 114])

    def test_decode_coordinates_confidence_and_target_class(self):
        found = decode_detections(self.raw_prediction(), (416, 416), 2, (208, 208))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "person")
        self.assertAlmostEqual(found[0]["confidence"], .72)
        np.testing.assert_allclose(found[0]["bbox"], [4, 4, 12, 20], atol=.01)

    def test_non_target_class_and_low_confidence_are_excluded(self):
        self.assertEqual(decode_detections(self.raw_prediction(class_id=16), (416, 416), 1, (416, 416)), [])
        self.assertEqual(decode_detections(self.raw_prediction(objectness=.5, class_score=.5), (416, 416), 1, (416, 416)), [])

    def test_same_class_duplicates_are_suppressed(self):
        raw = self.raw_prediction()
        raw[0, 1] = raw[0, 0]
        raw[0, 1, 0] -= 1  # Next grid's x offset is one stride.
        raw[0, 1, 4] = .8
        self.assertEqual(len(decode_detections(raw, (416, 416), 1, (416, 416))), 1)

    def test_different_classes_are_not_suppressed(self):
        raw = self.raw_prediction()
        raw[0, 1] = raw[0, 0]
        raw[0, 1, 0] -= 1
        raw[0, 1, 5] = 0
        raw[0, 1, 7] = .8
        self.assertEqual(len(decode_detections(raw, (416, 416), 1, (416, 416))), 2)

    def test_padding_only_box_is_rejected_and_boxes_are_clipped(self):
        raw = self.raw_prediction()
        raw[0, 0, 1] = 45  # Center y=360, below original scene height 208.
        self.assertEqual(decode_detections(raw, (416, 416), 1, (208, 416)), [])
        raw[0, 0, 1] = 0
        self.assertEqual(decode_detections(raw, (416, 416), 1, (208, 416))[0]["bbox"][1], 0)

    def test_competing_vehicle_labels_do_not_double_count(self):
        raw = self.raw_prediction(class_id=2)
        raw[0, 1] = raw[0, 0]
        raw[0, 1, 0] -= 1
        raw[0, 1, 7] = 0
        raw[0, 1, 12] = .7
        found = decode_detections(raw, (416, 416), 1, (416, 416))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "car")

    def test_malformed_outputs_are_explicit_errors(self):
        for output in (np.zeros((1, 8400, 85)), np.zeros((1, 3549, 6)), np.full((1, 3549, 85), np.nan)):
            with self.assertRaises(ValueError):
                decode_detections(output, (416, 416), 1, (416, 416))

    def test_missing_model_is_visible_and_nonfatal(self):
        detector = ObjectDetector("definitely-missing-model.onnx")
        self.assertFalse(detector.available)
        self.assertIn("missing", detector.status()["error"])
        self.assertEqual(detector.detect(np.zeros((10, 10, 3), np.uint8)), [])

    def test_corrupted_model_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "invalid.onnx"
            model.write_bytes(b"bad model")
            detector = ObjectDetector(model)
            self.assertFalse(detector.available)
            self.assertIn("checksum", detector.error)

    def test_inference_failure_becomes_unavailable_without_raising(self):
        detector = ObjectDetector("definitely-missing-model.onnx")
        detector.available = True
        detector._input_name = "images"
        with patch.object(detector, "_session") as session:
            session.run.side_effect = RuntimeError("test provider failure")
            self.assertEqual(detector.detect(np.zeros((10, 10, 3), np.uint8)), [])
        self.assertFalse(detector.available)
        self.assertIn("test provider failure", detector.status()["error"])

    @unittest.skipUnless(os.environ.get("C500_ML_SMOKE_IMAGE"), "Set C500_ML_SMOKE_IMAGE to official YOLOX assets/dog.jpg")
    def test_real_model_public_reference_image(self):
        detector = ObjectDetector()
        self.assertTrue(detector.available, detector.error)
        frame = np.asarray(Image.open(os.environ["C500_ML_SMOKE_IMAGE"]).convert("RGB"))
        found = detector.detect(frame)
        self.assertTrue(detector.available, detector.error)
        # Public reference includes a bicycle, dog and a pickup in the distance.
        self.assertIn("bicycle", {item["label"] for item in found})
        self.assertGreater(detector.status()["frames_processed"], 0)


if __name__ == "__main__":
    unittest.main()
