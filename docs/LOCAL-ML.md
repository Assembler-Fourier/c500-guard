# Local object detection

The bundled YOLOX-Tiny model detects people, bicycles, cars, motorcycles, buses
and trucks on this PC. ONNX Runtime runs on the CPU using two worker threads.
Frames stay local; inference uses no account, cloud service, API key or GPU.

The model is COCO-pretrained by Megvii. No custom training was performed on the
user's camera. It identifies object categories, not names, faces, registration
numbers or intent. A camera filming another monitor may lose substantial detail;
lighting, glare, monitor scaling and small objects can all reduce detection.
The motion and feed-fault detectors must continue independently of ML.

Install the Python dependencies into the watcher's existing Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe src\install_ml.py
.\.venv\Scripts\python.exe src\ml_selftest.py
```

Run these from the application directory using its `.venv`, rather than a
different global Python. `install.ps1` performs these steps automatically.
Python package installation and a missing model need internet access. Later
launches work offline. The deployment bundle contains
`models/yolox_tiny.onnx`; `install_ml.py` verifies it or downloads the tested
official model if absent. The model is SHA256-checked before loading. The
hash pins the copy tested here: the publisher's old release exposes no checksum.
Full provenance and the publisher's Apache-2.0 license are beside the model.

The self-test exits with code 0 only after a real model inference succeeds.
Without `--image`, it uses a blank synthetic image to test loading and CPU
inference. To check a scene locally, supply `--image path\to\snapshot.jpg`.
That does not prove that site zones, camera position or real intruders have been
validated. Walk/drive tests in the actual scene remain necessary for calibration.

Integration API:

```python
from object_detector import ObjectDetector
detector = ObjectDetector()  # Optional: model_path, confidence, nms_iou, threads
detections = detector.detect(rgb_frame)  # NumPy uint8 [height,width,3]
health = detector.status()  # Always surface available and error to the operator
```

Each detection is a JSON-compatible dictionary with `class_id`, `label`,
`confidence` and `bbox=[left,top,right,bottom]` in original frame pixels. The
confidence is objectness multiplied by class probability. The default threshold
is 0.45 in the low-level API. The integrated watcher uses **0.60**, samples at
most once every **0.5 seconds**, crops around the configured zones, and requires
two matching observations for an additional alert. Competing motor-vehicle
labels are suppressed for an overlapping object;
people and bicycles can remain separate when their boxes overlap.

Sample at approximately 1 frame per second in a separate worker, or within an
existing bounded frame loop. Require at least two consecutive detection samples
inside a configured zone before an additional ML alarm. Clear the confirmation
streak on source loss, a missed sample, changed framing or zone change. An old
result must not be reused as fresh evidence. ML must never suppress motion or
fault alarms. Startup/import/model/inference failure makes `available=false`
with a visible `error`; `detect()` returns an empty list without raising.

Verification on this PC, 2026-09-11: unit checks cover RGB/BGR conversion,
letterboxing, coordinate scaling, confidence multiplication, target filtering,
duplicate suppression, malformed output, missing/corrupt models and inference
failure. A real inference on the official repository's `assets/dog.jpg` detects
the bicycle and car. Initial warm CPU latency measured about 42 ms per frame at
two threads. This verifies installation and one public reference image; it is
not a measured site detection rate or a guarantee against missed events.

Official references:

- https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
- https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
- https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
- https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE
