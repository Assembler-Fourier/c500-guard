# Models and optional components

Weights, vendor runtimes, compiled DLLs and camera bridge binaries are not in this repository. Obtain them separately and review their upstream terms. The project's MIT license does not replace those terms.

| Component | Use | Configuration |
| --- | --- | --- |
| YOLOX-Tiny ONNX | Primary COCO object detector | 416 × 416; pinned URL/checksum in `src/object_detector.py`; install with `src/install_ml.py` |
| YOLOX-S ONNX | Optional entrance verifier | 640 × 640, GPU f32, explicit checksum, no CPU fallback; enabled only for supported PTZ whole-picture configuration |
| ONNX Runtime | Primary CPU execution | Pinned in `requirements-ml.txt` |
| OpenVINO | Optional Intel graphics execution and vehicle appearance models | Install a compatible runtime separately; the application checks availability at startup |
| Intel vehicle attributes / vehicle-reid models | Optional vehicle appearance suggestions | `vehicle-attributes-recognition-barrier-0042` and `vehicle-reid-0001`; without these files, the report can use its classical appearance path |
| FFmpeg | Capture and local incident encoding | Install separately or supply a local tools directory |
| WebView2 | Optional native Windows dashboard window | Source and build instructions in `app/` |
| go2rtc | Optional network-camera bridge | Source patch/provenance in `patches/`; camera credentials remain local |

YOLOX-S is expected at `models/yolox_s.onnx`. Its pinned SHA-256 is `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. YOLOX-Tiny's checksum is `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`. Check the source constants when updating weights. Do not rename an arbitrary model to satisfy a path: model shape and checksum are deliberately checked.

The optional entrance verifier is enabled through `ai.entrance_verifier: true`, a supported PTZ view and whole-picture processing. Missing GPU support, missing whole-picture input, invalid model bytes or a slow optional pass are reported as unavailable. The primary detector continues. If whole-picture processing was disabled when the worker started, enabling that setting requires restarting the worker/application in this release.

The software does not train on local footage. Fleet brands, ambulance liveries and individual vehicles are not separate trained classes. COCO car/truck/bus/motorcycle observations and route evidence remain subject to lighting, occlusion, glare and model errors. Recorded positive examples and selected negative controls are bounded regression evidence, not a measured field recall or false-alarm rate.

The optional vehicle-report software is in `car-memory/`. Its Windows integration defaults to an installation at `C:\c500-guard` and stores its private database under `car-memory/data/`; review those paths if installing elsewhere. Its authorization note and model directory must be supplied locally. Do not copy a previous installation's vehicle data as application software.

Upstream projects: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [OpenVINO](https://github.com/openvinotoolkit/openvino), [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), [go2rtc](https://github.com/AlexxIT/go2rtc), [WebView2](https://learn.microsoft.com/microsoft-edge/webview2/).
