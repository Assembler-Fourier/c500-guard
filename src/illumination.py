"""Conservative additive exposure correction for the motion channel only.

This does not classify motion or change the background/light measurement. A
large brightness change is removed only when it is nearly identical across
the whole picture. Ambiguous changes retain the original motion difference.
"""
from __future__ import annotations

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
