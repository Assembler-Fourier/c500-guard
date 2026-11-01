"""Conservative alignment of small camera shifts to an explicit fixed reference.

ORB descriptors and partial-affine RANSAC come from OpenCV:
https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

No image is learned automatically, no file is read/written, and a rejected
estimate returns None instead of an untrustworthy scene. This estimates only a
2-D similarity transform; perspective changes or a moved/changed monitor view
need operator calibration. Masks must exclude moving scenes and time overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

try:
    import cv2
    _CV_ERROR = ""
except (ImportError, OSError) as exc:
    cv2 = None
    _CV_ERROR = f"OpenCV unavailable: {exc}"


@dataclass(frozen=True)
class AlignmentConfig:
    """Conservative pixel limits verified at 640x360 (also usable at 480x270)."""
    max_translation_px: float = 18.0
    max_rotation_deg: float = 3.0
    max_scale_change: float = .04
    max_corner_displacement_px: float = 30.0
    min_matches: int = 24
    min_inliers: int = 20
    min_inlier_ratio: float = .65
    descriptor_ratio: float = .72
    ransac_error_px: float = 2.5
    max_median_error_px: float = 1.2
    max_p90_error_px: float = 2.5
    min_hull_fraction: float = .12
    min_span_fraction: float = .45
    # Use local support before measuring spatial coverage. Four nearby matches
    # must support a point; isolated outliers cannot expand the accepted hull.
    # Distance is normalized by the static mask width/height, so support does
    # not change when an anchor crosses an arbitrary grid boundary.
    min_support_neighbors: int = 4
    support_radius_fraction: float = .08
    # Coarse cell counts remain diagnostics, not an acceptance condition.
    min_features_per_cell: int = 4
    min_static_correlation: float = .75
    # Require a majority plus four independent static tiles, so a foreground
    # vehicle can occlude part of the scene without forcing a false fault.
    min_good_tile_fraction: float = .55
    min_good_tiles: int = 4
    min_valid_fraction: float = .85
    descriptor_guard_px: int = 16
    max_features: int = 1800
    # Confirmation tier, used only when the caller supplies a verified prior
    # pose. Evidence too weak to estimate a new transform (for example at
    # night, when the scene is lit differently from the reference) may still
    # CONFIRM that the camera has not moved. Zones then keep the prior
    # transform; a weak estimate never moves them. Residual, movement,
    # coverage and zone-visibility checks stay strict.
    weak_min_matches: int = 12
    weak_min_inliers: int = 10
    weak_min_correlation: float = .5
    weak_min_good_tiles: int = 3
    pose_tolerance_px: float = 5.0


def build_feature_mask(shape, include_rects=None, exclude_rects=(), border_px=10,
                       exclusion_padding_px=8) -> np.ndarray:
    """Build uint8 static-landmark mask from normalized (x,y,w,h) rectangles.

    include_rects=None means the image interior. An empty include list means no
    landmarks. Explicitly supply a monitor/content interior if only one plane
    should be tracked. Exclude every motion zone, clock/timestamp and unstable
    overlay. This helper cannot infer which monitor or regions are static.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1 or border_px < 0 or exclusion_padding_px < 0:
        raise ValueError("Invalid mask dimensions or padding")
    mask = np.full((height, width), 255 if include_rects is None else 0, np.uint8)

    def bounds(rect, padding=0):
        if len(rect) != 4 or not np.isfinite(rect).all():
            raise ValueError("Rectangles must contain four finite normalized values")
        x, y, w, h = map(float, rect)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("Rectangle must be inside the normalized frame")
        return (max(0, int(math.floor(x * width)) - padding),
                max(0, int(math.floor(y * height)) - padding),
                min(width, int(math.ceil((x + w) * width)) + padding),
                min(height, int(math.ceil((y + h) * height)) + padding))

    for rect in include_rects or ():
        x1, y1, x2, y2 = bounds(rect)
        mask[y1:y2, x1:x2] = 255
    for rect in exclude_rects:
        x1, y1, x2, y2 = bounds(rect, int(exclusion_padding_px))
        mask[y1:y2, x1:x2] = 0
    margin = min(int(border_px), height, width)
    if margin:
        mask[:margin] = mask[-margin:] = 0
        mask[:, :margin] = mask[:, -margin:] = 0
    return mask


def pose_gap_px(first, second, shape=(360, 640)) -> float:
    """Largest distance an image corner lands apart under two current->reference matrices."""
    height, width = int(shape[0]), int(shape[1])
    corners = np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    return float(np.linalg.norm(corners @ a[:, :2].T + a[:, 2] - (corners @ b[:, :2].T + b[:, 2]), axis=1).max())


def _rgb(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Alignment requires RGB uint8 [height,width,3]")
    if min(frame.shape[:2]) < 64:
        raise ValueError("Alignment image is too small")
    return np.ascontiguousarray(frame)


class SceneAligner:
    """One fixed reference per instance. Call only from one worker at a time.

    update(frame) -> (aligned_rgb_or_None, JSON-compatible status).
    The accepted matrix maps CURRENT pixels into REFERENCE coordinates.
    valid_mask is a fresh 0/255 mask of real observed pixels after successful
    alignment; border pixels are black, never filled with stale reference data.
    Optional required_mask marks alarm pixels which must all remain visible.
    """

    def __init__(self, reference_rgb=None, feature_mask=None, config=None, required_mask=None):
        self.config = config or AlignmentConfig()
        self.reset()
        if reference_rgb is not None:
            self.calibrate(reference_rgb, feature_mask, required_mask)

    def reset(self):
        self.reference = None
        self.feature_mask = None
        self.required_mask = None
        self.valid_mask = None
        self._gray = self._keypoints = self._descriptors = self._orb = None
        self._bounds = None
        self._last = self._state("uncalibrated", "Explicit reference calibration required")

    @property
    def reference_ready(self):
        return self.reference is not None and self._descriptors is not None

    def _state(self, state, reason="", **values):
        status = {"accepted": False, "state": state, "reason": reason,
                  "reference_ready": self.reference is not None and self._descriptors is not None,
                  "current_to_reference": None, "reference_to_current": None,
                  "translation_px": None, "rotation_deg": None, "scale": None,
                  "matches": 0, "inliers": 0, "inlier_ratio": 0.0}
        status.update(values)
        return status

    def status(self):
        import copy
        return copy.deepcopy(self._last)

    def calibrate(self, reference_rgb, feature_mask=None, required_mask=None):
        """Replace reference explicitly. Failure clears old calibration."""
        self.reset()
        if cv2 is None:
            self._last = self._state("unavailable", _CV_ERROR)
            return self.status()
        try:
            reference = _rgb(reference_rgb).copy()
            height, width = reference.shape[:2]
            if feature_mask is None:
                feature_mask = build_feature_mask(reference.shape)
            mask = np.asarray(feature_mask)
            if mask.shape != (height, width):
                raise ValueError("Feature mask must match the reference image dimensions")
            mask = np.where(mask, 255, 0).astype(np.uint8)
            if np.count_nonzero(mask) < height * width * .03:
                raise ValueError("Too little static landmark area in the feature mask")
            if required_mask is not None:
                required_mask = np.asarray(required_mask)
                if required_mask.shape != (height, width):
                    raise ValueError("Required visibility mask must match reference dimensions")
                required_mask = (required_mask != 0).copy()
            # An ORB descriptor samples a 31x31 neighborhood. Excluded motion
            # or timestamp pixels must not leak into a permitted keypoint.
            guard = max(0, int(self.config.descriptor_guard_px))
            kernel = np.ones((2 * guard + 1, 2 * guard + 1), np.uint8)
            safe_mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            ys, xs = np.nonzero(safe_mask)
            if len(xs) < 256:
                raise ValueError("Static landmark mask is too narrow after descriptor exclusion")
            self._bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            orb = cv2.ORB_create(nfeatures=self.config.max_features, edgeThreshold=16,
                                 patchSize=31, fastThreshold=10, nlevels=8)
            gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, safe_mask)
            if descriptors is None or len(keypoints) < self.config.min_matches:
                raise ValueError("Insufficient distinct static landmarks in reference")
            distribution = self._distribution(np.float32([p.pt for p in keypoints]))
            if not self._distributed(distribution):
                raise ValueError("Reference landmarks are concentrated in too small an area")
            self.reference, self.feature_mask = reference, mask.copy()
            self.required_mask, self._safe_mask = required_mask, safe_mask
            self._gray, self._keypoints, self._descriptors, self._orb = gray, keypoints, descriptors, orb
            self._last = self._state("ready", "Reference calibrated; awaiting current-frame validation",
                                     reference_features=len(keypoints), distribution=distribution)
        except Exception as exc:
            self.reference = self._descriptors = None
            self._last = self._state("uncalibrated", f"{type(exc).__name__}: {exc}")
        return self.status()

    def _distribution(self, points):
        x1, y1, x2, y2 = self._bounds
        width, height = max(1, x2 - x1), max(1, y2 - y1)
        empty = {"hull_fraction": 0., "span_x": 0., "span_y": 0., "occupied_cells": 0,
                 "supported_cells": 0, "cell_counts": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 "supported_points": 0, "supported_hull_fraction": 0.,
                 "supported_span_x": 0., "supported_span_y": 0.}
        if len(points) < 3:
            return empty
        cells = np.clip(((points - [x1, y1]) / [width, height] * 3).astype(int), 0, 2)
        counts = np.bincount(cells[:, 1] * 3 + cells[:, 0], minlength=9)
        result = {**empty,
                  "hull_fraction": round(float(cv2.contourArea(cv2.convexHull(points))) / (width * height), 4),
                  "span_x": round(float(np.ptp(points[:, 0])) / width, 4),
                  "span_y": round(float(np.ptp(points[:, 1])) / height, 4),
                  "occupied_cells": int(np.count_nonzero(counts)),
                  "supported_cells": int(np.count_nonzero(counts >= self.config.min_features_per_cell)),
                  "cell_counts": counts.reshape(3, 3).tolist()}
        neighbors = max(4, int(self.config.min_support_neighbors))
        if len(points) >= neighbors:
            normalized = np.float32((points - [x1, y1]) / [width, height])
            # K includes the point itself. OpenCV returns only the nearest K
            # distances; no full NxN Python/NumPy distance array is retained.
            distances, _ = cv2.batchDistance(normalized, normalized, cv2.CV_32F,
                                             normType=cv2.NORM_L2SQR, K=neighbors)
            supported = points[distances[:, -1] <= self.config.support_radius_fraction ** 2]
            result["supported_points"] = len(supported)
            if len(supported) >= 3:
                result.update(
                    supported_hull_fraction=round(float(cv2.contourArea(cv2.convexHull(supported))) / (width * height), 4),
                    supported_span_x=round(float(np.ptp(supported[:, 0])) / width, 4),
                    supported_span_y=round(float(np.ptp(supported[:, 1])) / height, 4))
        return result

    def _distributed(self, value):
        cfg = self.config
        return (value["supported_hull_fraction"] >= cfg.min_hull_fraction
                and value["supported_span_x"] >= cfg.min_span_fraction
                and value["supported_span_y"] >= cfg.min_span_fraction)

    def _static_agreement(self, aligned_gray, valid):
        """Compare separate textured static tiles; affine brightness tolerated."""
        x1, y1, x2, y2 = self._bounds
        reference = cv2.GaussianBlur(self._gray, (5, 5), 0).astype(np.float32)
        current = cv2.GaussianBlur(aligned_gray, (5, 5), 0).astype(np.float32)
        # Keep the comparison filter away from moving regions and warp borders.
        mask = (self._safe_mask != 0) & (valid != 0)
        scores = []
        for row in range(3):
            for col in range(3):
                xa, xb = x1 + (x2 - x1) * col // 3, x1 + (x2 - x1) * (col + 1) // 3
                ya, yb = y1 + (y2 - y1) * row // 3, y1 + (y2 - y1) * (row + 1) // 3
                selected = mask[ya:yb, xa:xb]
                if np.count_nonzero(selected) < 96:
                    continue
                a, b = reference[ya:yb, xa:xb][selected], current[ya:yb, xa:xb][selected]
                if a.std() < 6:
                    continue
                a, b = a - a.mean(), b - b.mean()
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                scores.append(float(np.dot(a, b) / denominator) if denominator > 1e-6 else 0.)
        good = sum(score >= self.config.min_static_correlation for score in scores)
        return {"tiles": len(scores), "good_tiles": good,
                "good_fraction": round(good / max(1, len(scores)), 4),
                "median_correlation": round(float(np.median(scores)), 4) if scores else 0.}

    @staticmethod
    def _warp(frame, matrix):
        height, width = frame.shape[:2]
        aligned = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), matrix,
                               (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Interpolation needs neighboring source pixels, so erode the valid
        # footprint by one pixel before claiming a monitored pixel is real.
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return aligned, valid

    def warp(self, frame, current_to_reference):
        """Re-apply a transform that this reference accepted earlier.

        Bridges brief rejections on a fixed mount (for example a vehicle hiding
        many landmarks) without trusting any new estimate. The caller bounds
        how long a held transform may be used. Returns None when a required
        alarm region would not be fully visible.
        """
        self.valid_mask = None
        if cv2 is None or not self.reference_ready:
            return None
        frame = _rgb(frame)
        matrix = np.asarray(current_to_reference, dtype=np.float64)
        if frame.shape != self.reference.shape or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return None
        aligned, valid = self._warp(frame, matrix)
        if self.required_mask is not None and np.any(self.required_mask & (valid == 0)):
            return None
        self.valid_mask = valid
        return aligned

    def update(self, frame, prior=None):
        """Estimate this frame's transform to the reference.

        prior: optional current->reference matrix verified earlier. With it,
        evidence too weak for a new estimate may still confirm that the camera
        has not moved (within pose_tolerance_px). The frame is then warped
        with the prior, never with the weak estimate, and status["weak"] is True.
        """
        started = time.perf_counter()
        self.valid_mask = None  # Never retain a mask from a previous good frame.
        if not self.reference_ready:
            self._last = self._state("uncalibrated" if cv2 is not None else "unavailable",
                                     "Explicit reference calibration required" if cv2 is not None else _CV_ERROR)
            return None, self.status()
        values, weak = {}, []
        if prior is not None:
            try:
                prior = np.asarray(prior, dtype=np.float64)
            except (TypeError, ValueError):
                prior = None
            if prior is not None and (prior.shape != (2, 3) or not np.isfinite(prior).all()):
                prior = None

        def reject(reason):
            self._last = self._state("rejected", reason,
                                     processing_ms=round((time.perf_counter() - started) * 1000, 2), **values)
            return None, self.status()

        try:
            frame = _rgb(frame)
            if frame.shape != self.reference.shape:
                return reject("Frame dimensions changed; recalibration required")
            cfg = self.config
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Search only near the known landmark mask; its shifted locations
            # may lie outside the exact reference mask before alignment.
            radius = int(math.ceil(cfg.max_corner_displacement_px))
            search = cv2.dilate(self._safe_mask, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
            keypoints, descriptors = self._orb.detectAndCompute(gray, search)
            if descriptors is None or len(keypoints) < cfg.min_matches:
                if prior is None or descriptors is None or len(keypoints) < cfg.weak_min_matches:
                    return reject("Insufficient current-frame landmarks")
                weak.append("few landmarks")
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

            def distinct_matches(first, second):
                result = {}
                for neighbors in matcher.knnMatch(first, second, k=2):
                    if len(neighbors) == 2:
                        a, b = neighbors
                        if a.distance < cfg.descriptor_ratio * b.distance and a.distance <= 64:
                            result[a.queryIdx] = a.trainIdx
                return result

            forward = distinct_matches(self._descriptors, descriptors)
            reverse = distinct_matches(descriptors, self._descriptors)
            pairs = [(ref, cur) for ref, cur in forward.items() if reverse.get(cur) == ref]
            values["matches"] = len(pairs)
            if len(pairs) < cfg.min_matches:
                if prior is None or len(pairs) < cfg.weak_min_matches:
                    return reject("Too few unambiguous mutual landmark matches")
                weak.append("few matches")
            reference_points = np.float32([self._keypoints[r].pt for r, _ in pairs])
            current_points = np.float32([keypoints[c].pt for _, c in pairs])
            matrix, inlier_mask = cv2.estimateAffinePartial2D(
                current_points, reference_points, method=cv2.RANSAC,
                ransacReprojThreshold=cfg.ransac_error_px, maxIters=2000,
                confidence=.995, refineIters=10)
            if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
                return reject("No stable similarity transform")
            inliers = inlier_mask.ravel().astype(bool)
            count = int(inliers.sum())
            ratio = count / len(pairs)
            values.update(inliers=count, inlier_ratio=round(ratio, 4))
            if count < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
                if prior is None or count < cfg.weak_min_inliers or ratio < cfg.min_inlier_ratio:
                    return reject("Landmark consensus is too weak")
                weak.append("few consistent landmarks")
            distribution = self._distribution(reference_points[inliers])
            values["distribution"] = distribution
            if not self._distributed(distribution):
                if prior is None:
                    return reject("Inliers are concentrated; cannot distinguish a moving object from camera movement")
                weak.append("landmarks concentrated")
            projected = current_points[inliers] @ matrix[:, :2].T + matrix[:, 2]
            residuals = np.linalg.norm(projected - reference_points[inliers], axis=1)
            median, p90 = float(np.median(residuals)), float(np.percentile(residuals, 90))
            values.update(median_error_px=round(median, 3), p90_error_px=round(p90, 3))
            if median > cfg.max_median_error_px or p90 > cfg.max_p90_error_px:
                return reject("Landmarks do not fit a precise similarity transform")
            height, width = frame.shape[:2]
            center = np.array([width / 2., height / 2.])
            displacement = center @ matrix[:, :2].T + matrix[:, 2] - center
            scale = math.hypot(matrix[0, 0], matrix[1, 0])
            rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            corners = np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            corner_shift = float(np.linalg.norm(corners @ matrix[:, :2].T + matrix[:, 2] - corners, axis=1).max())
            inverse = cv2.invertAffineTransform(matrix)
            values.update(current_to_reference=matrix.round(6).tolist(),
                          reference_to_current=inverse.round(6).tolist(),
                          translation_px=displacement.round(3).tolist(), rotation_deg=round(rotation, 4),
                          scale=round(scale, 6), max_corner_displacement_px=round(corner_shift, 3))
            if (np.linalg.norm(displacement) > cfg.max_translation_px or abs(rotation) > cfg.max_rotation_deg
                    or abs(scale - 1) > cfg.max_scale_change or corner_shift > cfg.max_corner_displacement_px):
                return reject("Movement exceeds the allowed small framing correction; recalibration required")
            aligned, valid = self._warp(frame, matrix)
            values["valid_fraction"] = round(float(np.count_nonzero(valid)) / valid.size, 4)
            if values["valid_fraction"] < cfg.min_valid_fraction:
                return reject("Too much of the reference view is outside the current image")
            if self.required_mask is not None and np.any(self.required_mask & (valid == 0)):
                return reject("A required alarm region is no longer fully visible")
            agreement = self._static_agreement(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), valid)
            values["static_agreement"] = agreement
            if agreement["good_tiles"] < cfg.min_good_tiles or agreement["good_fraction"] < cfg.min_good_tile_fraction:
                # Lighting changes a static scene's appearance only partly; a
                # different scene (another camera view, a dark monitor) fails this.
                if (prior is None or agreement["median_correlation"] < cfg.weak_min_correlation
                        or agreement["good_tiles"] < cfg.weak_min_good_tiles):
                    return reject("Static scene appearance changed or alignment is ambiguous")
                weak.append("scene lit differently from the reference")
            if not weak:
                self.valid_mask = valid
                self._last = self._state("aligned", "Supported small framing correction", accepted=True,
                                         processing_ms=round((time.perf_counter() - started) * 1000, 2), **values)
                return aligned, self.status()
            gap = pose_gap_px(matrix, prior, frame.shape)
            values.update(pose_gap_px=round(gap, 3), weak_reasons=weak,
                          estimate_to_reference=values["current_to_reference"])
            if gap > cfg.pose_tolerance_px:
                return reject("Weak landmarks disagree with the last verified camera position")
            held = self.warp(frame, prior)
            if held is None:
                return reject("A required alarm region is no longer fully visible")
            values.update(current_to_reference=prior.round(6).tolist(),
                          reference_to_current=cv2.invertAffineTransform(prior).round(6).tolist())
            self._last = self._state("confirmed", "Weak landmarks confirm the last verified camera position",
                                     accepted=True, weak=True,
                                     processing_ms=round((time.perf_counter() - started) * 1000, 2), **values)
            return held, self.status()
        except Exception as exc:
            return reject(f"Alignment failed: {type(exc).__name__}: {exc}")
