"""Run offline model health/latency verification; optionally inspect a local image."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

from object_detector import DEFAULT_MODEL, ObjectDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, help="Optional local image; never uploaded")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    detector = ObjectDetector(args.model)
    if not detector.available:
        print(json.dumps(detector.status(), indent=2))
        return 1
    try:
        frame = (np.asarray(Image.open(args.image).convert("RGB")) if args.image
                 else np.zeros((270, 480, 3), dtype=np.uint8))
    except Exception as exc:
        print(json.dumps({"available": False, "error": str(exc)}, indent=2))
        return 1
    times, detections = [], []
    for _ in range(max(1, min(20, args.runs))):
        detections = detector.detect(frame)
        if not detector.available:
            print(json.dumps(detector.status(), indent=2))
            return 1
        times.append(detector.inference_ms)
    print(json.dumps({"status": detector.status(), "detections": detections,
                      "median_ms": round(statistics.median(times), 2),
                      "max_ms": max(times), "image_supplied": bool(args.image)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
