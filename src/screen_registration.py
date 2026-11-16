"""Re-find the scene after the camera is physically moved, and re-map the zones.

The camera films a monitor, so every zone is really a region of that screen.
When the camera is put back in a different place the whole view shifts, turns
and skews. Matching landmarks between the new view and a saved reference gives
the perspective transform between them, so the saved zones keep pointing at the
same parts of the screen without anyone redrawing them.

Geometry only: nothing is learned or stored here, and an estimate that cannot
be verified is returned as None rather than a guess. The caller must still
confirm the warped view against its reference before trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ReacquireConfig:
    """Deliberately generous about movement, strict about evidence."""
    max_features: int = 4000
    descriptor_ratio: float = .75
    ransac_error_px: float = 3.0
    min_matches: int = 40
    min_inliers: int = 60
    min_inlier_ratio: float = .55
    max_median_error_px: float = 2.0
    min_zone_coverage: float = 1.0      # every alarm zone must stay visible
    min_view_coverage: float = .60
    min_supported_hull: float = .12
    min_supported_span: float = .45
    support_radius_fraction: float = .10


def _gray(frame, name):
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(np.ascontiguousarray(image.astype(np.uint8)), cv2.COLOR_RGB2GRAY)
    if image.ndim == 2:
        return np.ascontiguousarray(image.astype(np.uint8))
    raise ValueError(f"{name} must be an RGB or grayscale image")


def warp(frame, matrix, shape):
    """Apply a transform into reference coordinates; unseen pixels stay black."""
    height, width = int(shape[0]), int(shape[1])
    return cv2.warpPerspective(np.asarray(frame), np.asarray(matrix, dtype=np.float64),
                               (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def visibility_mask(matrix, shape):
    """Real source pixels, including the interpolation footprint at edges."""
    height, width = int(shape[0]), int(shape[1])
    seen = cv2.warpPerspective(np.full((height, width), 255, np.uint8),
        np.asarray(matrix, dtype=np.float64), (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.erode(seen, np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def coverage(matrix, shape, region_mask=None):
    """How much of the reference view (or of the alarm zones) the new view still sees."""
    height, width = int(shape[0]), int(shape[1])
    seen = visibility_mask(matrix, (height, width)) != 0
    if region_mask is None:
        return float(seen.mean())
    region = np.asarray(region_mask) != 0
    total = int(np.count_nonzero(region))
    return float(np.count_nonzero(seen & region) / total) if total else 0.0


def registration_mask(shape, polygon=None, exclude_rects=()):
    """Monitor plane for remount matching, with mutable overlays excluded.

    This includes road detail: robust matching rejects independently moving
    objects. The caller separately verifies the result with its static mask.
    """
    from scene_alignment import build_feature_mask
    mask = build_feature_mask(shape, include_polygon=polygon,
                              exclude_rects=exclude_rects, border_px=0)
    return cv2.erode(mask, np.ones((17, 17), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _distribution(points, reference_mask, shape, cfg):
    if reference_mask is None:
        low, span = np.float32([0, 0]), np.float32([shape[1] - 1, shape[0] - 1])
    else:
        ys, xs = np.nonzero(reference_mask)
        if not len(xs):
            return {"supported_points": 0, "supported_hull": 0., "supported_span": [0., 0.]}
        low = np.float32([xs.min(), ys.min()])
        span = np.float32([xs.max() - xs.min(), ys.max() - ys.min()])
    span = np.maximum(span, 1)
    normalized = np.asarray((points - low) / span, dtype=np.float32)
    if len(points) < 4:
        supported = points[:0]
    else:
        distances, _ = cv2.batchDistance(normalized, normalized, cv2.CV_32F,
                                         normType=cv2.NORM_L2SQR, K=4)
        supported = points[distances[:, -1] <= cfg.support_radius_fraction ** 2]
    if len(supported) < 3:
        return {"supported_points": len(supported), "supported_hull": 0., "supported_span": [0., 0.]}
    return {"supported_points": len(supported),
            "supported_hull": float(cv2.contourArea(cv2.convexHull(supported)) / np.prod(span)),
            "supported_span": (np.ptp(supported, axis=0) / span).tolist()}


def estimate_transform(current, reference, config=None, required_mask=None, reference_mask=None):
    """Estimate current->reference perspective transform. Returns (matrix, report).

    The matrix is None whenever the evidence is too weak, the fit too loose, or
    an alarm zone would fall outside the new view.
    """
    cfg = config or ReacquireConfig()
    report = {"matches": 0, "inliers": 0, "inlier_ratio": 0.0, "median_error_px": None,
              "zone_coverage": 0.0, "view_coverage": 0.0, "reason": ""}
    try:
        current_gray, reference_gray = _gray(current, "current"), _gray(reference, "reference")
        orb = cv2.ORB_create(nfeatures=cfg.max_features, fastThreshold=7, nlevels=10)
        keypoints_a, descriptors_a = orb.detectAndCompute(current_gray, None)
        if reference_mask is not None:
            reference_mask = np.where(reference_mask, 255, 0).astype(np.uint8)
            if reference_mask.shape != reference_gray.shape:
                raise ValueError("Registration mask must match reference dimensions")
        keypoints_b, descriptors_b = orb.detectAndCompute(reference_gray, reference_mask)
        if descriptors_a is None or descriptors_b is None:
            report["reason"] = "No landmarks in one of the views"
            return None, report
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        def distinct_matches(first, second):
            result = {}
            for neighbours in matcher.knnMatch(first, second, k=2):
                if (len(neighbours) == 2
                        and neighbours[0].distance < cfg.descriptor_ratio * neighbours[1].distance
                        and neighbours[0].distance <= 64):
                    result[neighbours[0].queryIdx] = neighbours[0].trainIdx
            return result
        forward = distinct_matches(descriptors_a, descriptors_b)
        reverse = distinct_matches(descriptors_b, descriptors_a)
        pairs = {a: b for a, b in forward.items() if reverse.get(b) == a}
        report["matches"] = len(pairs)
        if len(pairs) < cfg.min_matches:
            report["reason"] = "Too few landmarks match the saved view"
            return None, report
        source = np.float32([keypoints_a[a].pt for a in pairs])
        target = np.float32([keypoints_b[b].pt for b in pairs.values()])
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, cfg.ransac_error_px,
                                          maxIters=5000, confidence=.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            report["reason"] = "No consistent camera position fits the landmarks"
            return None, report
        inliers = mask.ravel().astype(bool)
        report["inliers"] = int(inliers.sum())
        report["inlier_ratio"] = round(report["inliers"] / len(pairs), 3)
        if report["inliers"] < cfg.min_inliers or report["inlier_ratio"] < cfg.min_inlier_ratio:
            report["reason"] = "Landmark agreement is too weak to move the zones"
            return None, report
        distribution = _distribution(target[inliers], reference_mask, reference_gray.shape, cfg)
        report["distribution"] = distribution
        if (distribution["supported_hull"] < cfg.min_supported_hull
                or min(distribution["supported_span"]) < cfg.min_supported_span):
            report["reason"] = "Matching landmarks are concentrated in too little of the screen"
            return None, report
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        residual = np.linalg.norm(projected - target, axis=1)[inliers]
        report["median_error_px"] = round(float(np.median(residual)), 3)
        if report["median_error_px"] > cfg.max_median_error_px:
            report["reason"] = "Landmarks do not fit one camera position precisely"
            return None, report
        # Kept even when refused below: the framing guide uses it to say how to zoom/turn back.
        report["matrix"] = [[round(float(v), 8) for v in row] for row in matrix]
        shape = reference_gray.shape
        view_coverage = coverage(matrix, shape)
        zone_coverage = coverage(matrix, shape, required_mask) if required_mask is not None else 1.0
        report["view_coverage"] = round(view_coverage, 6)
        report["zone_coverage"] = round(zone_coverage, 6)
        if zone_coverage < cfg.min_zone_coverage:
            report["reason"] = "Part of an alarm zone is outside the new camera view"
            return None, report
        if view_coverage < cfg.min_view_coverage:
            report["reason"] = "The new view covers too little of the saved scene"
            return None, report
        return matrix, report
    except cv2.error as exc:                       # pragma: no cover - OpenCV guard
        report["reason"] = f"Re-registration failed: {exc}"
        return None, report


def movement_px(matrix, shape):
    """How far the view moved, measured at the corners of the reference view."""
    height, width = int(shape[0]), int(shape[1])
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)
    return float(np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max())


def framing_guide(matrix, shape, label="the saved view", ok_scale=.06, ok_shift=.04, zones=None) -> dict:
    """How the live picture differs from the saved view, in words for the operator of a zoomable camera.

    matrix maps the live picture to the saved (reference) view (2x3 or 3x3). Returns zoom / turn / tilt advice and
    the saved view's outline in the live picture (normalised corners; may reach outside 0..1 when zoomed in).
    """
    import math
    height, width = int(shape[0]), int(shape[1])
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0., 0., 1.]])
    inverse = np.linalg.inv(m)
    corners = np.float64([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    outline = cv2.perspectiveTransform(corners, inverse).reshape(-1, 2) / (width, height)
    centre = cv2.perspectiveTransform(np.float64([[[width / 2., height / 2.]]]), inverse).reshape(2) / (width, height)
    size = math.sqrt(max(abs(float(cv2.contourArea(outline.astype(np.float32)))), 1e-9))
    advice, zoom, zoom_pct, pan, tilt = [], None, 0, None, None
    if size < 1 - ok_scale:                      # the saved view looks smaller: the camera is zoomed out
        zoom, zoom_pct = "in", round((1 / size - 1) * 100)
    elif size > 1 + ok_scale:
        zoom, zoom_pct = "out", round((size - 1) * 100)
    if zoom:
        advice.append(f"zoom {zoom}{' a little' if zoom_pct < 30 else ''} (about {zoom_pct} %)")
    dx, dy = float(centre[0] - .5), float(centre[1] - .5)
    if abs(dx) > ok_shift:
        pan = "right" if dx > 0 else "left"
        advice.append(f"turn {pan}{' a little' if abs(dx) < .15 else ''}")
    if abs(dy) > ok_shift:
        tilt = "down" if dy > 0 else "up"
        advice.append(f"tilt {tilt}{' a little' if abs(dy) < .15 else ''}")
    result = {"state": "adjust" if advice else "ok", "zoom": zoom, "zoom_pct": zoom_pct, "pan": pan, "tilt": tilt,
              "scale": round(1 / size, 3), "centre": [round(float(v), 4) for v in centre],
              "outline": [[round(float(x), 4), round(float(y), 4)] for x, y in outline]}
    if advice:
        result["text"] = f"To match {label}: " + ", ".join(advice)
        result["short"] = "Saved view (yellow): " + ", ".join(a.split(" (")[0] for a in advice)
    else:
        result["text"] = result["short"] = f"Framing matches {label}"
    if zones:
        fit = _zone_fit(inverse, width, height, zones)
        if fit:
            result.update(fit)
            result["state"] = "ok" if fit["primary"] == "good" else "adjust"
            result["short"] = fit["text"]
            result["text"] = fit["text"]
    return result


FIT_MARGIN = .07          # keep every alarm area this far inside the picture (room for a small zoom/turn)
ZOOM_OK = (.93, 1.12)     # within this the framing is good (a 7 % zoom-in still keeps every zone inside)


def _zone_fit(inverse, width, height, zones) -> dict | None:
    """The best framing for these alarm areas, relative to the live picture (live = inverse(matrix) applied)."""
    import math
    points = [p for polygon in zones for p in polygon if len(p) == 2]
    if len(points) < 3:
        return None
    reference = np.float64([[x * width, y * height] for x, y in points]).reshape(-1, 1, 2)
    live = cv2.perspectiveTransform(reference, inverse).reshape(-1, 2) / (width, height)
    x0, y0 = live.min(axis=0)
    x1, y1 = live.max(axis=0)
    m = FIT_MARGIN
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    factor = min((1 - 2 * m) / span_x, (1 - 2 * m) / span_y)        # >1: zoom in by this, <1: zoom out
    half = .5 - m
    target_x = min(max(.5, x1 - half / factor), x0 + half / factor) if factor > 0 else .5
    target_y = min(max(.5, y1 - half / factor), y0 + half / factor) if factor > 0 else .5
    dx, dy = float(target_x - .5), float(target_y - .5)
    cut = x0 < 0 or y0 < 0 or x1 > 1 or y1 > 1
    zoom_pct = round(abs(factor - 1) * 100)

    def amount(value, small, big):
        return " a little" if value < small else "" if value < big else " a lot"
    zoom_words = (f"zoom in a lot (about {factor:.1f}x closer)" if factor >= 1.5 else
                  f"zoom in{amount(factor - 1, .25, .5)} (about {zoom_pct} %)" if factor > ZOOM_OK[1] else
                  f"zoom out a lot (about {1 / factor:.1f}x wider)" if factor <= 2 / 3 else
                  f"zoom out{amount(1 - factor, .15, .35)} (about {zoom_pct} %)" if factor < ZOOM_OK[0] else "")
    turn_words = (f"turn {'right' if dx > 0 else 'left'}{amount(abs(dx), .12, .3)}" if abs(dx) > .05 else "")
    tilt_words = (f"tilt {'down' if dy > 0 else 'up'}{amount(abs(dy), .12, .3)}" if abs(dy) > .05 else "")
    # One headline: a cut-off alarm area first, then too wide, then centring.
    if cut and factor < ZOOM_OK[0]:
        primary = "zoom_out"
    elif cut and turn_words and abs(dx) >= abs(dy):
        primary = "turn_right" if dx > 0 else "turn_left"
    elif cut and tilt_words:
        primary = "tilt_down" if dy > 0 else "tilt_up"
    elif factor > ZOOM_OK[1]:
        primary = "zoom_in"
    elif factor < ZOOM_OK[0]:
        primary = "zoom_out"
    elif turn_words and abs(dx) >= abs(dy):
        primary = "turn_right" if dx > 0 else "turn_left"
    elif tilt_words:
        primary = "tilt_down" if dy > 0 else "tilt_up"
    else:
        primary = "good"
    headline = {"zoom_in": "ZOOM IN", "zoom_out": "ZOOM OUT", "turn_left": "TURN LEFT", "turn_right": "TURN RIGHT",
                "tilt_up": "TILT UP", "tilt_down": "TILT DOWN", "good": "GOOD - leave it here"}[primary]
    steps = [w for w in (turn_words, tilt_words) if w]
    if zoom_words:
        steps = ([zoom_words] + steps) if primary.startswith("zoom") else (steps + ["then " + zoom_words])
    text = ("Best framing: every lane, barrier and footpath fills the picture" if primary == "good"
            else "For the best view: " + ", ".join(steps))
    closeness = min(factor, 1 / factor) if factor > 0 else 0.
    fit_pct = max(0, min(100, round(100 * (closeness - 2 * max(abs(dx), abs(dy)) - (.15 if cut else 0)))))
    return {"primary": primary, "headline": headline, "zoom_factor": round(float(factor), 3),
            "target_centre": [round(float(target_x), 4), round(float(target_y), 4)], "cut_off": bool(cut),
            "fit_pct": fit_pct, "zones_box": [round(float(v), 4) for v in (x0, y0, x1, y1)], "text": text}
