"""Offline YOLOX detections, independent of the motion/fault alarm.

This is a COCO-pretrained object detector, not a site-trained security model.
Pre/postprocessing follows the official YOLOX ONNXRuntime demonstration:
https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
YOLOX Copyright (c) 2021-2022 Megvii Inc. Apache-2.0; see models/YOLOX-LICENSE.txt.
The adapter is independently integrated for RGB frames, bounded CPU work, safe
failure reporting, target-class filtering and plain JSON-compatible results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
# Locally calculated after HTTPS download from the official release. The GitHub
# release asset has no publisher digest; this pins the tested artifact only.
MODEL_SHA256 = "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolox_tiny.onnx"
OPENVINO_DIR = Path(__file__).resolve().parent.parent / "tools" / "openvino"
# "graphics": OpenVINO on the Intel graphics, else the processor. "processor": ONNX Runtime CPU only.
ENGINE = "graphics"
TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Suppress competing road-vehicle labels for the same object, while retaining
# overlapping people/cycles (for example a person riding a bicycle).
NMS_GROUPS = ((0,), (1,), (2, 3, 5, 7))


def preprocess_rgb(frame: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """RGB uint8 HWC -> BGR float32 NCHW, 0..255, top-left 114 padding.

    Pillow bilinear resize replaces OpenCV INTER_LINEAR; Pillow's downsampling
    filter can differ slightly. No ImageNet normalization or /255 is applied.
    input_size is (height, width), matching the official ONNX model.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("ML requires an HxWx3 RGB uint8 frame")
    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("ML received an empty frame")
    target_h, target_w = input_size
    ratio = min(target_h / height, target_w / width)
    resized = Image.fromarray(frame).resize(
        (max(1, int(width * ratio)), max(1, int(height * ratio))), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized)[:, :, ::-1]
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:pixels.shape[0], :pixels.shape[1]] = pixels
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Bounded greedy IoU suppression; all boxes belong to one class."""
    order = scores.argsort()[::-1][:300]
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < 50:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        low = np.maximum(boxes[current, :2], boxes[remaining, :2])
        high = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        overlap = np.maximum(0, high - low).prod(axis=1)
        union = area[current] + area[remaining] - overlap
        iou = overlap / np.maximum(union, 1e-9)
        order = remaining[iou <= threshold]
    return keep


def decode_detections(output: np.ndarray, input_size: tuple[int, int], ratio: float,
                      frame_size: tuple[int, int], confidence: float = .45,
                      nms_iou: float = .45, normal_confidence: float | None = None) -> list[dict]:
    """Decode official YOLOX raw stride-8/16/32 output to image-pixel xyxy boxes.

    frame_size is (height, width). Only person and road vehicle classes return.
    """
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[0] != 1 or predictions.shape[2] != 85:
        raise ValueError(f"Unsupported YOLOX output shape: {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("ML model returned non-finite values")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid image scaling ratio")
    height, width = input_size
    grids, strides = [], []
    for stride in (8, 16, 32):
        yy, xx = np.meshgrid(np.arange(height // stride), np.arange(width // stride), indexing="ij")
        grid = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    grid = np.concatenate(grids)
    stride_values = np.concatenate(strides)
    raw = predictions[0]
    if len(raw) != len(grid):
        raise ValueError(f"Unsupported YOLOX grid size: {len(raw)}; expected {len(grid)}")
    class_ids = raw[:, 5:].argmax(axis=1)
    scores = raw[:, 4] * raw[np.arange(len(raw)), class_ids + 5]
    selected = np.flatnonzero((scores >= confidence) & np.isin(class_ids, list(TARGET_CLASSES)))
    if not len(selected):
        return []
    centers = (raw[selected, :2] + grid[selected]) * stride_values[selected]
    sizes = np.exp(np.clip(raw[selected, 2:4], -20, 20)) * stride_values[selected]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
    frame_h, frame_w = frame_size
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, frame_w)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, frame_h)
    results = []
    for group in NMS_GROUPS:
        local_ids = np.flatnonzero(np.isin(class_ids[selected], group)
                                  & (boxes[:, 2] > boxes[:, 0])
                                  & (boxes[:, 3] > boxes[:, 1]))
        for kept in _nms(boxes[local_ids], scores[selected[local_ids]], nms_iou):
            i = local_ids[kept]
            class_id = int(class_ids[selected[i]])
            label = TARGET_CLASSES[class_id]
            results.append({"class_id": class_id, "label": label,
                            "confidence": round(float(scores[selected[i]]), 4),
                            "raw_confidence": float(scores[selected[i]]),
                            "bbox": [round(float(value), 2) for value in boxes[i]]})
    if normal_confidence is not None:
        # Optional weak diagnostic candidates must not consume the existing
        # 50-result budget ahead of a normal candidate merely because both
        # display confidences round to the same value. Preserve normal order.
        floor = float(np.float32(normal_confidence))
        return sorted(results, key=lambda item: (item['raw_confidence'] >= floor,
                                                 item['confidence']), reverse=True)[:50]
    return sorted(results, key=lambda item: item["confidence"], reverse=True)[:50]


class ObjectDetector:
    """Best-effort local ML. Failure never disables the main motion/fault path.

    detect() accepts an RGB NumPy frame and returns JSON-compatible detections.
    Always include status() in the dashboard: an unavailable detector must not
    be presented as an empty scene. Caller controls rate limiting and requires
    consecutive-frame confirmation before using results as an extra alarm.
    This synchronous class must be called from one worker at a time.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, confidence: float = .45,
                 nms_iou: float = .45, threads: int = 2,
                 expected_sha256: str | None = MODEL_SHA256, engine: str | None = None,
                 input_size: tuple[int, int] = (416, 416),
                 model_label: str = "YOLOX-Tiny COCO pretrained",
                 gpu_precision: str = "f16", allow_cpu_fallback: bool = True):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.available = False
        self.error = ""
        self.inference_ms = None
        self.frames_processed = 0
        self.input_size = (416, 416)
        self.model_label = "YOLOX-Tiny COCO pretrained"
        self.gpu_precision = "f16"
        self.allow_cpu_fallback = True
        self.precision = None
        self._session = None
        self._request = None
        self._output = None
        self._threads = threads
        self.provider = "CPU"
        self.engine_note = ""
        try:
            # A caller selects a tested profile explicitly. Neither a filename
            # nor a graph shape identifies the model or overrides its checksum.
            if (not isinstance(input_size, (tuple, list)) or len(input_size) != 2
                    or any(type(value) is not int for value in input_size)
                    or tuple(input_size) not in ((416, 416), (640, 640))):
                raise ValueError("ML input size must be (416, 416) or (640, 640)")
            if (not isinstance(model_label, str) or not model_label.strip()
                    or len(model_label) > 120 or any(ord(char) < 32 for char in model_label)):
                raise ValueError("ML model label must be nonempty plain text (at most 120 characters)")
            if gpu_precision not in ("f16", "f32"):
                raise ValueError("ML graphics precision must be f16 or f32")
            if not isinstance(allow_cpu_fallback, bool):
                raise ValueError("ML CPU fallback setting must be boolean")
            self.input_size = tuple(input_size)
            self.model_label = model_label.strip()
            self.gpu_precision = gpu_precision
            self.allow_cpu_fallback = allow_cpu_fallback
            if not 0 < confidence <= 1 or not 0 < nms_iou <= 1:
                raise ValueError("ML confidence and NMS IoU must be in (0, 1]")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ML model missing: {self.model_path.name}")
            if expected_sha256:
                digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
                if digest != expected_sha256.lower():
                    raise ValueError("ML model checksum differs from tested official release")
            if (engine or ENGINE) == "graphics":
                try:
                    self._open_graphics()
                except Exception as exc:                          # noqa: BLE001 - fall back below
                    self._request = None
                    if not self.allow_cpu_fallback:
                        raise RuntimeError(f"Graphics initialization failed; CPU fallback disabled: {exc}") from exc
                    self.engine_note = (f"Intel graphics unavailable, using the processor: "
                                        f"{type(exc).__name__}: {exc}")[:300]
            if self._request is not None:
                self.available = True
                return
            self._open_processor()
        except Exception as exc:
            self._fail(exc)

    def _open_graphics(self) -> None:
        import sys
        if OPENVINO_DIR.is_dir() and str(OPENVINO_DIR) not in sys.path:
            sys.path.append(str(OPENVINO_DIR))      # after the tool's own packages, never shadowing them
        import openvino as ov
        core = ov.Core()
        if "GPU" not in core.available_devices:
            raise RuntimeError("no Intel graphics device")
        cache = OPENVINO_DIR / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        model = core.read_model(str(self.model_path))
        shape = [int(value) for value in model.input(0).get_shape()]
        expected = [1, 3, *self.input_size]
        if shape != expected:
            raise ValueError(f"Unsupported ML input shape: {shape}; expected {expected}")
        compiled = core.compile_model(model, "GPU", {"PERFORMANCE_HINT": "LATENCY",
                                                     "INFERENCE_PRECISION_HINT": self.gpu_precision})
        request = compiled.create_infer_request()
        request.infer({0: np.zeros(expected, np.float32)})   # a broken driver fails here, not mid-shift
        self._request, self._output = request, compiled.output(0)
        self.provider = "Intel graphics (OpenVINO)"
        self.precision = self.gpu_precision

    def _open_processor(self) -> None:
        if True:
            threads = self._threads
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, int(threads)))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self._session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                 providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise ValueError("Unsupported ML input type")
            shape = inputs[0].shape
            expected = [1, 3, *self.input_size]
            if shape != expected:
                raise ValueError(f"Unsupported ML input shape: {shape}; expected {expected}")
            self._input_name = inputs[0].name
            self.provider = "CPU"
            self.precision = "f32"
            self.available = True

    def _fail(self, exc: Exception) -> None:
        self.available = False
        self.error = f"{type(exc).__name__}: {exc}"
        self._session = None
        self.precision = None

    def detect(self, rgb_frame: np.ndarray) -> list[dict]:
        if not self.available:
            return []
        started = time.perf_counter()
        try:
            prepared, ratio = preprocess_rgb(rgb_frame, self.input_size)
            if self._request is not None:
                try:
                    output = self._request.infer({0: prepared})[self._output].copy()
                except Exception as exc:                          # noqa: BLE001
                    # Graphics trouble mid-shift: carry on with the processor at once.
                    self._request = None
                    if not self.allow_cpu_fallback:
                        raise RuntimeError(f"Graphics inference failed; CPU fallback disabled: {exc}") from exc
                    self.engine_note = (f"Intel graphics failed, switched to the processor: "
                                        f"{type(exc).__name__}: {exc}")[:300]
                    self._open_processor()
                    output = self._session.run(None, {self._input_name: prepared})[0]
            else:
                output = self._session.run(None, {self._input_name: prepared})[0]
            detections = decode_detections(output, self.input_size, ratio,
                                           rgb_frame.shape[:2], self.confidence, self.nms_iou,
                                           normal_confidence=getattr(self, 'normal_confidence', None))
            self.inference_ms = round((time.perf_counter() - started) * 1000, 2)
            self.frames_processed += 1
            return detections
        except Exception as exc:
            self._fail(exc)
            return []

    def status(self) -> dict:
        return {"available": self.available, "error": self.error,
                "model": self.model_label, "provider": self.provider,
                "precision": self.precision,
                "engine_note": self.engine_note,
                "confidence_threshold": self.confidence,
                "classes": list(TARGET_CLASSES.values()), "inference_ms": self.inference_ms,
                "frames_processed": self.frames_processed,
                "input_size": list(self.input_size), "offline": True}
