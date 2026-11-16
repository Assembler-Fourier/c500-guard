"""Conservative additive exposure correction for the motion channel only.

This does not classify motion or change the background/light measurement. A
large brightness change is removed only when it is nearly identical across
the whole picture. Ambiguous changes retain the original motion difference.
"""
from __future__ import annotations

import math
import time

import numpy as np


def compensated_motion_diff(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return absolute motion differences, correcting a clearly uniform shift.

    The median signed change must be at least 8 luma levels. At least 95% of
    all pixels and 80% of every tile in a 4x4 grid must agree with it within
    3 levels. Every tile's median must also agree. These gates deliberately
    reject localized illumination, gradients, substantial clipping, and
    contrast/gamma changes. Residual local changes survive the subtraction.

    Inputs are finite 2-D luma arrays in the same 0..255 scale. Neither array
    is mutated. Continue computing the light channel from the original arrays.
    """
    current = np.asarray(gray, dtype=np.float32)
    reference = np.asarray(background, dtype=np.float32)
    if current.ndim != 2 or current.shape != reference.shape or not current.size:
        raise ValueError("gray and background must be nonempty matching 2-D arrays")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("gray and background must contain finite luma values")

    signed = current - reference
    original = np.abs(signed)
    offset = float(np.median(signed))
    tolerance = 3.0
    if abs(offset) < 8.0:
        return original

    residual = signed - offset
    if float(np.mean(np.abs(residual) <= tolerance)) < 0.95:
        return original

    height, width = signed.shape
    for row in np.array_split(signed, min(4, height), axis=0):
        for tile in np.array_split(row, min(4, width), axis=1):
            if abs(float(np.median(tile)) - offset) > tolerance:
                return original
            if float(np.mean(np.abs(tile - offset) <= tolerance)) < 0.80:
                return original

    return np.abs(residual)


class LightingMonitor:
    """Describe scene exposure without changing detection or alignment decisions.

    update() accepts RGB uint8 frames and an optional normalized (x, y, w, h)
    scene crop. The caller should crop to the monitored image, excluding the
    monitor's bezel/black borders. Dark pixels *inside* that crop are retained:
    dropping them would conceal a genuinely dark or covered view.

    Modes describe measured image brightness, not the time of day or a claim
    that night vision is active. No image, object coordinates or key input is
    retained. A frame gap resets sustained-condition timers.
    """

    def __init__(self, smoothing_s=2.0, settling_s=2.0, sustained_s=4.0,
                 max_gap_s=2.0):
        values = (smoothing_s, settling_s, sustained_s, max_gap_s)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Lighting durations must be finite positive seconds")
        self.smoothing_s, self.settling_s, self.sustained_s, self.max_gap_s = map(float, values)
        self.reset()

    def reset(self):
        self._last_time = None
        self._last_median = None
        self._smoothed = None
        self._mode = "unknown"
        self._settling_until = 0.0
        self._condition = None
        self._condition_since = None
        self._report = {"available": False, "state": "unavailable", "mode": "unknown",
                        "message": "Waiting for fresh scene exposure measurements",
                        "median_luma": None, "smoothed_luma": None, "p10_luma": None,
                        "p90_luma": None, "dark_fraction": None, "bright_fraction": None,
                        "condition": None, "condition_duration_s": 0.0,
                        "settling_remaining_s": 0.0, "scene_rect": None}

    @staticmethod
    def _sample_luma(frame, scene_rect):
        frame = np.asarray(frame)
        if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
                or not frame.shape[0] or not frame.shape[1]):
            raise ValueError("Lighting requires a nonempty RGB uint8 frame")
        rect = (0.0, 0.0, 1.0, 1.0) if scene_rect is None else scene_rect
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            raise ValueError("scene_rect must be a normalized (x, y, w, h)")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in rect):
            raise ValueError("scene_rect must contain finite numbers")
        x, y, w, h = map(float, rect)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1.00000001 or y+h > 1.00000001:
            raise ValueError("scene_rect must be inside the frame")
        height, width = frame.shape[:2]
        x1, y1 = int(x * width), int(y * height)
        x2, y2 = min(width, math.ceil((x+w) * width)), min(height, math.ceil((y+h) * height))
        scene = frame[y1:y2, x1:x2]
        # Uniform spatial sampling bounds work and keeps dark/highlight fractions
        # meaningful. This is not a histogram that removes shadows or highlights.
        stride = max(1, math.ceil(max(scene.shape[:2]) / 128))
        sample = scene[::stride, ::stride].astype(np.float32)
        luma = sample[:, :, 0] * .299 + sample[:, :, 1] * .587 + sample[:, :, 2] * .114
        return luma, [x, y, w, h]

    def update(self, frame, scene_rect=None, now=None):
        """Return current JSON-safe diagnostics; ``now`` uses monotonic seconds."""
        now = time.monotonic() if now is None else now
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("Lighting timestamp must be finite monotonic seconds")
        if self._last_time is not None and now < self._last_time:
            raise ValueError("Lighting timestamps cannot move backwards")
        luma, rect = self._sample_luma(frame, scene_rect)
        p10, median, p90 = map(float, np.quantile(luma, [.1, .5, .9]))
        dark_fraction = float(np.mean(luma <= 12.0))
        bright_fraction = float(np.mean(luma >= 235.0))
        gap = None if self._last_time is None else now - self._last_time
        if gap is None or gap > self.max_gap_s or rect != self._report.get("scene_rect"):
            # A stale interval or changed crop is not evidence of persistence.
            self._smoothed = median
            self._last_median = None
            self._condition = self._condition_since = None
            self._settling_until = 0.0
            self._mode = "dim" if median <= 45 else "bright" if median >= 190 else "normal"
        else:
            alpha = -math.expm1(-gap / self.smoothing_s)
            self._smoothed += alpha * (median - self._smoothed)
        if self._last_median is not None and abs(median - self._last_median) >= 18:
            self._settling_until = now + self.settling_s
        level = self._smoothed
        if self._mode == "dim":
            self._mode = "bright" if level >= 190 else "normal" if level >= 58 else "dim"
        elif self._mode == "bright":
            self._mode = "dim" if level <= 45 else "normal" if level <= 175 else "bright"
        else:
            self._mode = "dim" if level <= 45 else "bright" if level >= 190 else "normal"
        condition = ("near_black" if p90 <= 12 and dark_fraction >= .85 else
                     "glare" if p10 >= 235 and bright_fraction >= .85 else None)
        if condition != self._condition:
            self._condition, self._condition_since = condition, now if condition else None
        duration = 0.0 if self._condition_since is None else max(0.0, now - self._condition_since)
        remaining = max(0.0, self._settling_until - now)
        state = condition if condition and duration >= self.sustained_s else "adjusting" if remaining else "stable"
        messages = {"near_black": "Scene remains almost black; check camera visibility",
                    "glare": "Most of the scene remains washed out; check glare or exposure",
                    "adjusting": "A brightness change is settling; detection settings are unchanged"}
        message = messages.get(state, {"dim": "Dim scene; automatic alignment checks still apply",
                                       "normal": "Scene exposure is steady",
                                       "bright": "Bright scene; automatic alignment checks still apply"}[self._mode])
        self._last_time, self._last_median = float(now), median
        self._report = {"available": True, "state": state, "mode": self._mode, "message": message,
                        "median_luma": round(median, 1), "smoothed_luma": round(level, 1),
                        "p10_luma": round(p10, 1), "p90_luma": round(p90, 1),
                        "dark_fraction": round(dark_fraction, 4),
                        "bright_fraction": round(bright_fraction, 4),
                        "condition": condition, "condition_duration_s": round(duration, 1),
                        "settling_remaining_s": round(remaining, 1), "scene_rect": rect}
        return self.status(now=now)

    def status(self, now=None):
        report = dict(self._report)
        if report["scene_rect"] is not None:
            report["scene_rect"] = list(report["scene_rect"])
        now = time.monotonic() if now is None else now
        age = None if self._last_time is None else max(0.0, now - self._last_time)
        report["frame_age_s"] = None if age is None else round(age, 2)
        if age is not None and age > self.max_gap_s:
            report.update(available=False, state="stale", message="Waiting for a fresh scene exposure measurement")
        return report
