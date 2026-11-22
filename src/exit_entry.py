"""Exit-corner tail-light measurement for an aligned camera view.

Combines red-light area and image change; the caller owns source freshness,
zone geometry and alarm policy. Thresholds require validation for each camera.
Pure measurement: no state, camera ownership or sound output.
"""
from __future__ import annotations

import math

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


# --- Entrance lane headlights (21 Nov 2026) -------------------------------------------------------------------------
# The AI loses an arriving car whose headlights face the camera (the 20 Nov 20:51 car had no box at all while it waited
# at the entrance barrier; the guard raised the arm 28 s into the clip). Only an ARRIVING car faces the camera with lit
# headlights in the entrance lane beyond the barrier; exit cars there drive away.
# "New light" = near-white now (min channel >= 245) where the lane is USUALLY not white. The far lane is already almost
# white under the floodlight, so a darkest-value baseline was fooled by anything dark passing over it (the barrier arm,
# a car body): the white road reappearing looked new. The per-pixel "usually white" share is learned slowly over time,
# and a 3x3 opening drops thin streaks such as the floodlit arm itself.
# Replayed on the clean frames of the kept clips (20 Nov 20:28 - 21 Nov 00:40; 85 clips, 151 in an earlier pass):
# 3 % of x .66-.90 / y .36-.50 held five frames fired on the 20:51 car at 19.0 s, on 14 of 24 clips with an AI entrance
# alarm (blooms peak 6-36 %, a fallback there), and on one other clip - a real car waiting at the entrance (20:57). The
# highest peak on any clip without such a car was 1.5 % (hence 3 %); the darkest-value baseline it replaced had fired on
# 3 barrier-arm movements (peaks up to 11 %). First live alarm, 21 Nov 01:05: a real car's headlights lighting the lane
# 3-5 s before the arm was raised, while the AI was still warming up after a restart.
# A verified camera correction over 2 px restarts the map (watch.py), so a shifted picture is not "new light".
HEADLIGHT_USUAL_MAX = .25     # a pixel white more than a quarter of the (recent) time is part of the normal scene


def _open3(mask):
    """3x3 morphological opening (border treated like OpenCV: erosion does not eat the edge)."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=True)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + h, dx:dx + w]
    padded = np.pad(eroded, 1, constant_values=False)
    opened = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            opened |= padded[dy:dy + h, dx:dx + w]
    return opened


def measure_headlights(frame_rgb, usual, mask, *, white=245., pct=3., max_pct=60., learn_s=300., dt=.1):
    """Share of the lane showing new near-white light. Returns (metrics, new_usual).

    ``usual`` is the lane's per-pixel share of time spent near-white (float32 array the shape of the frame, or None to
    start from this frame). It moves toward the current frame with time constant ``learn_s`` seconds (``dt`` is the
    time since the previous frame). Valid new headlight support is excluded from learning while present: a waiting
    vehicle must not become the empty-road baseline and hide the next car. Other pixels still adapt, including broad
    illumination changes rejected by ``max_pct``. Dark things passing over the floodlit road do not make it "new"
    when they leave. A bloom over more than ``max_pct`` of the lane is general brightening, not headlights.
    """
    frame = np.asarray(frame_rgb, dtype=np.float32)
    region = np.asarray(mask, dtype=bool)
    empty = {"bright_pct": 0.0, "hit": False, "support_bounds": None}
    if frame.ndim != 3 or frame.shape[2] != 3 or region.shape != frame.shape[:2] or not region.any():
        return empty, None
    is_white = frame.min(axis=2) >= white
    if usual is None or np.shape(usual) != is_white.shape:
        return {**empty, "starting": True}, is_white.astype(np.float32)
    usual = np.asarray(usual, dtype=np.float32)
    bright = _open3(is_white & (usual < HEADLIGHT_USUAL_MAX) & region) & region
    share = float(np.count_nonzero(bright)) * 100. / float(np.count_nonzero(region))
    hit = bool(pct <= share <= max_pct)
    rate = 1. - math.exp(-min(2., max(0., float(dt))) / max(1., float(learn_s)))
    new_usual = (usual + rate * (is_white - usual)).astype(np.float32)
    if hit:
        # Learning a waiting car's lamps into the usual scene also blinds the
        # next arrival at those pixels for several minutes after it leaves.
        # Freeze only measured new support; a broad flood still adapts normally.
        new_usual[bright] = usual[bright]
    bounds = None
    if share > 0:
        ys, xs = np.nonzero(bright)
        x0, y0 = int(xs.min()), int(ys.min())
        bounds = (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)
    return {"bright_pct": round(share, 3), "hit": hit, "support_bounds": bounds}, new_usual
