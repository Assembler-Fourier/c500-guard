"""Exit-corner tail-light measurement for an aligned camera view.

Combines red-light area and image change; the caller owns source freshness,
zone geometry and alarm policy. Thresholds require validation for each camera.
Pure measurement: no state, camera ownership or sound output.
"""
from __future__ import annotations

import numpy as np

RED_MIN_LEVEL = 150.     # a lit tail/brake lamp is bright...
RED_OVER_GREEN = 60.     # ...and clearly red
CHANGE_LEVEL = 30.       # grey levels that count as "this pixel changed"


def measure_taillights(frame_rgb, background_gray, mask, *, red_pct=2.0, change_pct=10.0, wake_pct=5.0) -> dict:
    """Share of red-lamp pixels and of changed pixels inside ``mask``.

    hit: red >= red_pct AND change >= change_pct (a vehicle with lit tail lights entering).
    wake: change >= wake_pct (something is moving in the corner: worth looking at more often).
    lane_seen: red >= red_pct / 2 (early, weaker evidence that an exit vehicle is in the corner).
    """
    frame = np.asarray(frame_rgb, dtype=np.float32)
    region = np.asarray(mask, dtype=bool)
    empty = {"red_pct": 0.0, "changed_pct": 0.0, "hit": False, "wake": False, "lane_seen": False,
             "support_bounds": None}
    if frame.ndim != 3 or frame.shape[2] != 3 or region.shape != frame.shape[:2] or not region.any():
        return empty
    background = np.asarray(background_gray, dtype=np.float32)
    if background.shape != region.shape:
        return empty
    pixels = frame[region]
    red_channel, green_channel = pixels[:, 0], pixels[:, 1]
    red = (red_channel - green_channel > RED_OVER_GREEN) & (red_channel > RED_MIN_LEVEL)
    gray = pixels[:, 0] * 0.299 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.114
    changed = np.abs(gray - background[region]) > CHANGE_LEVEL
    red_share = float(red.mean()) * 100.
    change_share = float(changed.mean()) * 100.
    bounds = None
    if red.any():
        ys, xs = np.nonzero(region)
        ys, xs = ys[red], xs[red]
        x0, y0 = int(xs.min()), int(ys.min())
        bounds = (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)
    return {"red_pct": round(red_share, 3), "changed_pct": round(change_share, 3),
            "hit": bool(red_share >= red_pct and change_share >= change_pct),
            "wake": bool(change_share >= wake_pct),
            "lane_seen": bool(red_share >= red_pct / 2.),
            "support_bounds": bounds}
