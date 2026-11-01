"""Local CPU object recognition alongside independent motion/fault detection."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np


class SemanticMonitor:
    RETRY_INITIAL = 2.0
    RETRY_MAX = 60.0
    MAX_RESULT_AGE = 3.0

    def __init__(self, zones, model_path: Path, interval=0.75, confidence=0.60):
        self.zones = zones
        self.model_path = model_path
        self.interval = interval
        self.confidence = confidence
        self._input = queue.Queue(maxsize=1)
        self._output = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._result_lock = threading.RLock()
        self._epoch = 0
        self._worker_generation = 0
        self._confirmation_generation = 0
        self._enabled = False
        self._last_submit = 0.0
        self._last_confirmed_at = None
        self._counts = {}
        self._misses = {}
        self._active = set()
        self._status = {'available': False, 'state': 'loading', 'error': '',
                        'inferences': 0, 'detections': [], 'model': 'YOLOX tiny',
                        'mode': 'additional confirmed person/vehicle alerts',
                        'confidence_threshold': confidence, 'recovery_attempts': 0}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        detector = None
        backoff = self.RETRY_INITIAL
        while not self._stop.is_set():
            if detector is None:
                try:
                    from object_detector import ObjectDetector
                    detector = ObjectDetector(self.model_path, confidence=self.confidence, threads=2)
                    if not detector.available:
                        raise RuntimeError(detector.status().get('error') or 'Local AI initialization failed')
                    with self._result_lock:
                        # Do not process a frame retained throughout model loading/recovery.
                        self._clear_queue(self._input)
                        self._status = {**self._status, **detector.status(), 'state': 'ready',
                                        'detections': [], 'updated_at': None, '_retry_at': None}
                except Exception as exc:
                    self._retry_failure(exc, backoff)
                    detector = None
                    if self._stop.wait(backoff):
                        break
                    backoff = min(backoff * 2, self.RETRY_MAX)
                    continue
            try:
                epoch, generation, stamp, frame, offset = self._input.get(timeout=0.25)
            except queue.Empty:
                continue
            if (epoch != self._epoch or generation != self._worker_generation
                    or time.monotonic() - stamp >= self.MAX_RESULT_AGE):
                continue
            start = time.monotonic()
            try:
                objects = detector.detect(frame)
                if not detector.available:
                    raise RuntimeError(detector.status().get('error') or 'Local AI inference failed')
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    obj['bbox'] = [x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]]
            except Exception as exc:
                self._retry_failure(exc, backoff)
                detector = None
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self.RETRY_MAX)
                continue
            backoff = self.RETRY_INITIAL
            with self._result_lock:
                if self._stop.is_set():
                    break
                self._status = {**self._status, **detector.status(), 'state': 'ready',
                                'inferences': self._status['inferences'] + 1,
                                'inference_ms': round((time.monotonic() - start) * 1000),
                                'detections': objects, 'updated_at': stamp, '_epoch': epoch}
                self._clear_queue(self._output)
                self._output.put_nowait((epoch, generation, stamp, objects))

    @staticmethod
    def _clear_queue(target):
        try:
            while True:
                target.get_nowait()
        except queue.Empty:
            pass

    def _retry_failure(self, exc, delay):
        with self._result_lock:
            # Results before a model failure must never complete a post-recovery pair.
            self._worker_generation += 1
            self._clear_queue(self._input)
            self._clear_queue(self._output)
            self._status = {**self._status, 'available': False, 'state': 'retrying',
                            'error': str(exc), 'detections': [], 'updated_at': None,
                            'recovery_attempts': self._status['recovery_attempts'] + 1,
                            '_retry_at': time.monotonic() + delay}

    def reset(self):
        with self._result_lock:
            self._epoch += 1
            self._counts.clear()
            self._misses.clear()
            self._active.clear()
            self._last_confirmed_at = None

    def _confirm(self, objects, width, height):
        hits = set()
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            for index, zone in enumerate(self.zones):
                x, y, w, h = zone.rect
                if zone.motion and x <= cx <= x+w and y <= cy <= y+h:
                    hits.add((index, obj['label']))
        events = []
        for key in hits:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._misses[key] = 0
            if self._counts[key] >= 2 and key not in self._active:
                self._active.add(key)
                zone = self.zones[key[0]]
                events.append((zone.severity, f'{zone.name}: AI confirmed {key[1]} in two observations'))
        for key in set(self._counts) - hits:
            self._counts[key] = 0
            self._misses[key] = self._misses.get(key, 0) + 1
            if self._misses[key] >= 3:
                self._active.discard(key)
        return events

    def update(self, frame, enabled):
        now = time.monotonic()
        events = []
        with self._result_lock:
            if (enabled != self._enabled
                    or self._worker_generation != self._confirmation_generation
                    or (self._last_confirmed_at is not None
                        and now - self._last_confirmed_at >= self.MAX_RESULT_AGE)):
                self.reset()
                self._enabled = enabled
                self._confirmation_generation = self._worker_generation
            try:
                epoch, generation, stamp, objects = self._output.get_nowait()
                if (enabled and epoch == self._epoch and generation == self._worker_generation
                        and now - stamp < self.MAX_RESULT_AGE):
                    events = self._confirm(objects, frame.shape[1], frame.shape[0])
                    self._last_confirmed_at = stamp
            except queue.Empty:
                pass
        if now - self._last_submit >= self.interval and not self._input.full():
            # The subject may occupy only a monitor within the camera picture.
            # Focus the model on the union of configured zones with a small margin.
            h, w = frame.shape[:2]
            x1 = max(0, int((min(z.rect[0] for z in self.zones) - .04) * w))
            y1 = max(0, int((min(z.rect[1] for z in self.zones) - .04) * h))
            x2 = min(w, int((max(z.rect[0]+z.rect[2] for z in self.zones) + .04) * w))
            y2 = min(h, int((max(z.rect[1]+z.rect[3] for z in self.zones) + .04) * h))
            crop = np.asarray(frame[y1:y2, x1:x2], dtype=np.uint8).copy()
            self._input.put_nowait((self._epoch, self._worker_generation, now, crop, (x1, y1)))
            self._last_submit = now
        return events

    def status(self):
        report = dict(self._status)
        updated = report.pop('updated_at', None)
        epoch = report.pop('_epoch', self._epoch)
        retry_at = report.pop('_retry_at', None)
        report['retry_in_s'] = round(max(0, retry_at - time.monotonic()), 1) if retry_at else None
        report['age_s'] = round(time.monotonic()-updated, 1) if updated else None
        if report.get('available') and (epoch != self._epoch or updated is None or report['age_s'] >= self.MAX_RESULT_AGE):
            report['state'] = 'waiting_for_frames'
            report['detections'] = []
        return report

    def stop(self):
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
