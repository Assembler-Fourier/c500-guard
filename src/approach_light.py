"""Localized approach-light measurements with conservative exposure correction.

This is a change detector, not a vehicle classifier. The caller owns the
background, warmup, persistence and one-episode alarm latch. No images or other
state are retained here. A localized wash is deliberately eligible even when
no vehicle can yet be recognized.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _luma(image, name):
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        array = (array[:, :, 0].astype(np.float32) * .299
                 + array[:, :, 1].astype(np.float32) * .587
                 + array[:, :, 2].astype(np.float32) * .114)
    else:
        array = array.astype(np.float32)
    if (array.ndim != 2 or not array.size or not np.isfinite(array).all()
            or float(array.min()) < -.001 or float(array.max()) > 255.001):
        raise ValueError(f"{name} must be a nonempty finite luma or RGB image in 0..255")
    return np.clip(array, 0, 255)


def _mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite mask matching the image")
    return array != 0


def _uniform_residual(residual, scene):
    """Require agreement across the scene, including every populated tile."""
    if np.count_nonzero(scene) < 64:
        return False
    agrees = np.abs(residual) <= 2.0
    if float(np.mean(agrees[scene])) < .97:
        return False
    # A localized pool of light must not be declared a global exposure shift
    # just because a large dark border dominates the frame's histogram.
    height, width = scene.shape
    for rows in np.array_split(np.arange(height), min(4, height)):
        for columns in np.array_split(np.arange(width), min(4, width)):
            tile = np.ix_(rows, columns)
            selected = scene[tile]
            if np.count_nonzero(selected) >= 16 and float(np.mean(agrees[tile][selected])) < .90:
                return False
    return True


def _exposure_reference(current, reference, scene):
    """Remove only a small, demonstrably scene-wide additive/gain change."""
    if np.count_nonzero(scene) < 64:
        return reference, 0.0, 1.0
    signed = current - reference
    # Sampling bounds the photometric fit's work. Every candidate is still
    # validated against all usable scene pixels before it can suppress light.
    stride = max(1, math.ceil(math.sqrt(current.size / 16384)))
    sampled_scene = scene[::stride, ::stride]
    before = reference[::stride, ::stride][sampled_scene]
    after = current[::stride, ::stride][sampled_scene]
    if before.size < 64:
        return reference, 0.0, 1.0
    sample_change = after - before
    offset = float(np.median(sample_change))
    if .75 <= abs(offset) <= 48:
        predicted = np.clip(reference + offset, 0, 255)
        if _uniform_residual(current - predicted, scene):
            return predicted, offset, 1.0

    # Robust intensity-bin medians estimate a camera gain change. Only accept
    # the fit when virtually every scene pixel AND each tile agrees. This is
    # intentionally much stricter than simply subtracting a scene average.
    # An unchanged majority cannot establish a gain change. Keeping the raw
    # differences here also preserves a small local lamp that sampling misses.
    if abs(offset) < .75 and np.mean(np.abs(sample_change) < .75) >= .90:
        return reference, 0.0, 1.0
    edges = np.quantile(before, [0, .2, .4, .6, .8, 1])
    if float(edges[-1] - edges[0]) < 40:
        return reference, 0.0, 1.0
    x, y = [], []
    for index in range(5):
        selected = (before >= edges[index]) & (before <= edges[index + 1])
        if np.count_nonzero(selected) >= 16:
            x.append(float(np.median(before[selected])))
            y.append(float(np.median(after[selected])))
    if len(x) < 4 or max(x) - min(x) < 30:
        return reference, 0.0, 1.0
    values = np.asarray(x, dtype=np.float64)
    centered = values - values.mean()
    gain = float(np.dot(centered, np.asarray(y) - np.mean(y)) / np.dot(centered, centered))
    offset = float(np.mean(y) - gain * values.mean())
    if not (.75 <= gain <= 1.30 and abs(offset) <= 32 and abs(gain - 1) >= .015):
        return reference, 0.0, 1.0
    predicted = np.clip(reference * gain + offset, 0, 255)
    if not _uniform_residual(current - predicted, scene):
        return reference, 0.0, 1.0
    return predicted, offset, gain


def _region_box(region, pad=8):
    """Smallest padded box containing the region, as array slices.

    Padding must exceed every kernel radius used inside the box, so that pixels
    belonging to the region see exactly the neighbourhood they would have seen
    in the whole frame. A region touching the frame edge gives a box touching
    the same edge, so replicated borders behave identically too.
    """
    rows = np.flatnonzero(region.any(axis=1))
    columns = np.flatnonzero(region.any(axis=0))
    if not rows.size or not columns.size:
        return (slice(0, region.shape[0]), slice(0, region.shape[1]))
    return (slice(max(0, int(rows[0]) - pad), min(region.shape[0], int(rows[-1]) + 1 + pad)),
            slice(max(0, int(columns[0]) - pad), min(region.shape[1], int(columns[-1]) + 1 + pad)))


def _urgent_exposure_reference(current, reference, scene, region):
    """Fit exterior exposure without letting an arrival explain itself away.

    The ordinary light channel deliberately retains its stricter fitter. This
    fallback balances small intensity groups across spatial tiles, so a car or
    headlight elsewhere cannot dominate a least-squares fit of all pixels.
    """
    outside = scene & ~cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    report = {"exposure_fit": "none", "exposure_support": 0.0,
              "exposure_tiles": 0, "exposure_uncertainty": 0.0}
    predicted, offset, gain = _exposure_reference(current, reference, outside)
    if offset != 0.0 or gain != 1.0:
        report.update(exposure_fit="uniform", exposure_support=1.0)
        return predicted, offset, gain, report

    if np.count_nonzero(outside) < 256:
        return reference, 0.0, 1.0, report
    exterior_change = (current - reference)[outside]
    if float(np.median(np.abs(exterior_change))) < 3:
        return reference, 0.0, 1.0, report
    # Retain additive evidence on flat scenes where gain is unidentifiable.
    # A moving object may spoil one tile without spoiling a global step.
    shift = float(np.median(exterior_change))
    candidate = np.clip(reference + shift, 0, 255)
    agrees = np.abs(current - candidate) <= 2
    # A day/night or monitor-exposure step can exceed 64 luma. Magnitude alone
    # is not evidence of an arrival: accept any representable additive step
    # only after the same exterior-pixel and distributed-tile agreement checks.
    # The urgent road is excluded from the fit, so an independent bright/dark
    # arrival there remains a residual even during a large exposure transition.
    if .75 <= abs(shift) <= 255 and float(np.mean(agrees[outside])) >= .90:
        supported = populated = 0
        for rows in np.array_split(np.arange(current.shape[0]), 4):
            for columns in np.array_split(np.arange(current.shape[1]), 4):
                tile = np.ix_(rows, columns)
                selected = outside[tile]
                if np.count_nonzero(selected) >= 32:
                    populated += 1
                    supported += float(np.mean(agrees[tile][selected])) >= .75
        if populated >= 3 and supported >= math.ceil(.75 * populated):
            report.update(exposure_fit="additive", exposure_support=supported / populated,
                          exposure_tiles=supported)
            return candidate, shift, 1.0, report

    before = cv2.GaussianBlur(reference, (3, 3), 0)
    after = cv2.GaussianBlur(current, (3, 3), 0)
    gradient = np.maximum(np.abs(cv2.Sobel(before, cv2.CV_32F, 1, 0, ksize=3)),
                          np.abs(cv2.Sobel(before, cv2.CV_32F, 0, 1, ksize=3))) / 8.0
    usable = (outside & (before >= 40) & (before <= 230)
              & (after >= 10) & (after <= 245) & (gradient <= 6))
    cells, tile_ids = [], []
    for row_index, rows in enumerate(np.array_split(np.arange(current.shape[0]), 6)):
        for col_index, columns in enumerate(np.array_split(np.arange(current.shape[1]), 8)):
            tile = np.ix_(rows, columns)
            selected, x, y = usable[tile], before[tile], after[tile]
            for low in (40, 80, 120, 160, 200):
                group = selected & (x >= low) & (x < low + 40)
                if np.count_nonzero(group) >= 32:
                    cells.append((float(np.median(x[group])), float(np.median(y[group]))))
                    tile_ids.append(row_index * 8 + col_index)
    if len(cells) < 12 or len(set(tile_ids)) < 6:
        return reference, 0.0, 1.0, report
    x, y = np.asarray(cells, dtype=np.float64).T
    # Never extrapolate a gain fit into road intensities absent from its
    # exterior evidence. Flat scenes are handled by the uniform additive fit.
    road_low, road_high = np.percentile(reference[region], [10, 90])
    if np.ptp(x) < 30 or road_low < x.min() - 8 or road_high > x.max() + 8:
        return reference, 0.0, 1.0, report
    gain, offset = 1.0, float(np.median(y - x))
    for _ in range(8):
        residual = y - (gain * x + offset)
        scale = max(2.0, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-6))
        mx, my = np.average(x, weights=weights), np.average(y, weights=weights)
        denominator = float(np.sum(weights * (x - mx) ** 2))
        if denominator < 1e-6:
            return reference, 0.0, 1.0, report
        gain = float(np.sum(weights * (x - mx) * (y - my)) / denominator)
        offset = float(my - gain * mx)
    if not (.70 <= gain <= 1.35 and abs(offset) <= 40):
        return reference, 0.0, 1.0, report
    residual = y - (gain * x + offset)
    raw_error, fitted_error = float(np.median(np.abs(y - x))), float(np.median(np.abs(residual)))
    report.update(exposure_median_residual=round(fitted_error, 4),
                  exposure_reference_change=round(raw_error, 4))
    if raw_error < 3 or fitted_error > 12 or fitted_error > .65 * raw_error:
        return reference, 0.0, 1.0, report
    agrees = np.abs(residual) <= 18
    ids = np.asarray(tile_ids)
    supported = sum(float(np.mean(agrees[ids == key])) >= .60 for key in set(tile_ids))
    coverage = supported / len(set(tile_ids))
    if supported < 6 or coverage < .60 or float(np.mean(agrees)) < .60:
        return reference, 0.0, 1.0, report
    # Only accepted, imperfect exposure fits get a small uncertainty allowance.
    # It is capped at TWO luma levels; no frame delay or unbounded noise floor
    # can grow here. An exact fit keeps the original small-change threshold.
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    uncertainty = min(2.0, 1.96 * sigma / math.sqrt(supported))
    report.update(exposure_fit="robust", exposure_support=round(coverage, 4),
                  exposure_tiles=supported, exposure_uncertainty=round(uncertainty, 4))
    return np.clip(reference * gain + offset, 0, 255), offset, gain, report


def measure_urgent_change(gray, background, region_mask, *, scene_mask=None,
                          threshold=8.0, min_patch_pct=0.6, noise=None, min_connected_pct=None):
    """One-frame road change with spatial support, not scattered pixel counts.

    The noise threshold is capped: prolonged activity cannot teach this urgent
    route to ignore a new clear vehicle/headlight patch. Exposure is estimated
    outside the urgent region, with spatially distributed support. Both light
    and dark residuals remain eligible; no classifier or extra frame is used.
    """
    current, reference = _luma(gray, "gray"), _luma(background, "background")
    if current.shape != reference.shape:
        raise ValueError("gray and background must have matching image dimensions")
    region = _mask(region_mask, current.shape, "region_mask")
    scene = np.ones(current.shape, dtype=bool) if scene_mask is None else _mask(
        scene_mask, current.shape, "scene_mask")
    for name, value, limit in (("threshold", threshold, 255),
                                ("min_patch_pct", min_patch_pct, 100)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError(f"{name} must be finite and in (0, {limit}]")
    area = int(np.count_nonzero(region))
    required = max(12, math.ceil(area * min_patch_pct / 100.0))
    if min_connected_pct is not None and (isinstance(min_connected_pct, bool)
                                          or not isinstance(min_connected_pct, (int, float))
                                          or not math.isfinite(min_connected_pct)
                                          or not 0 < min_connected_pct <= 100):
        raise ValueError("min_connected_pct must be finite and in (0, 100]")
    connected = math.ceil(area * min_connected_pct / 100.0) if min_connected_pct else 0
    report = {"hit": False, "weak_hit": False, "support_bounds": None,
              "changed_pct": 0.0, "largest_patch_pixels": 0,
              "required_patch_pixels": required, "global_offset": 0.0,
              "global_gain": 1.0, "noise_threshold": float(threshold),
              "raw_changed_pixels": 0, "edge_rejected_pixels": 0,
              "jitter_radius_pixels": 2}
    if not area:
        return report
    predicted, offset, gain, exposure = _urgent_exposure_reference(current, reference, scene, region)
    report.update(exposure)
    band = np.full(current.shape, float(threshold) + exposure["exposure_uncertainty"], dtype=np.float32)
    if noise is not None:
        flicker = np.asarray(noise, dtype=np.float32)
        if flicker.shape != current.shape or not np.isfinite(flicker).all() or float(flicker.min()) < 0:
            raise ValueError("noise must be a finite non-negative map matching the image")
        band = np.maximum(band, np.minimum(flicker * 2.0, 12.0))
    # Everything below is confined to the region, so it is computed on the
    # region's own box instead of the whole frame. The padding is wider than
    # both kernel radii (5x5 and 3x3), which makes the result inside the region
    # identical to the full-frame computation - measured at 7.7 ms -> 0.2 ms on
    # the real corner zone, and the saving is what stops frames being dropped.
    # The exposure fit above stays global: it must see the whole scene.
    box = _region_box(region, pad=8)
    current, predicted, region = current[box], predicted[box], region[box]
    band = band[box] if isinstance(band, np.ndarray) and band.ndim == 2 else band
    raw_difference = np.abs(current - predicted)
    # A small registration/monitor-refresh shift must not turn a stationary
    # road edge into an arrival. Permit only values already present within two
    # pixels of the predicted background; new bright/dark material still has
    # to exceed that local range by the original threshold. No frames are held.
    neighborhood = np.ones((5, 5), dtype=np.uint8)
    lower = cv2.erode(predicted, neighborhood, borderType=cv2.BORDER_REPLICATE)
    upper = cv2.dilate(predicted, neighborhood, borderType=cv2.BORDER_REPLICATE)
    difference = np.maximum(np.maximum(current - upper, lower - current), 0)
    changed = ((difference > band) & region).astype(np.uint8)
    neighbors = cv2.filter2D(changed, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    supported = changed & (neighbors >= 5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(supported, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    total = int(np.count_nonzero(changed))
    if largest:
        component = stats[1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())]
        # Report the patch where the caller can find it: full-frame coordinates.
        report["support_bounds"] = [int(component[0]) + box[1].start,
                                    int(component[1]) + box[0].start,
                                    int(component[2]), int(component[3])]
    strong = ((difference > np.maximum(band, 10.0)) & region).astype(np.uint8)
    strong_neighbors = cv2.filter2D(strong, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8),
                                    borderType=cv2.BORDER_CONSTANT)
    strong_supported = strong & (strong_neighbors >= 5).astype(np.uint8)
    n, _, strong_stats, _ = cv2.connectedComponentsWithStats(strong_supported, connectivity=8)
    strong_largest = int(strong_stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    raw_total = int(np.count_nonzero((raw_difference > band) & region))
    report.update(hit=bool(np.count_nonzero(strong) >= max(48, required)
                           and strong_largest >= max(24, connected)),
                  weak_hit=bool(total >= required and largest >= max(9, connected)), changed_pixels=total,
                  required_connected_pixels=max(24, connected),
                  raw_changed_pixels=raw_total, edge_rejected_pixels=raw_total-total,
                  jitter_radius_pixels=2,
                  changed_pct=100.0 * largest / area,
                  largest_patch_pixels=largest, global_offset=offset,
                  global_gain=gain, noise_threshold=float(band[region].mean()))
    return report


def measure_local_brightness(gray, background, region_mask, *, scene_mask=None,
                             min_delta=3.0, min_bright_pct=2.0, noise=None, noise_k=3.0):
    """Return JSON-safe measurements and an unlatched localized-light ``hit``.

    Inputs are matching full-frame luma or RGB arrays in 0..255, and a polygon
    mask where nonzero pixels belong to the region. ``scene_mask`` optionally
    excludes the monitor bezel from exposure estimation; it does not alter the
    region. Exposure correction only applies when 97% of the scene and 90% of
    every populated tile agree with the same additive/gain change within two
    luma levels. Any local residual remains eligible.

    A hit needs the specified average positive increase, enough bright pixels,
    and a connected supported patch. Neighbor support rejects isolated hot
    pixels and sparse rain/compression speckles. The whole region may be bright:
    there is no upper-area veto that would hide a headlight wash. The caller
    should require consecutive frames and latch a hit until the episode clears.

    ``delta`` is the signed mean after correction; ``positive_delta`` measures
    positive change only, so simultaneous shadows do not cancel headlights.
    Neither input is modified. An empty region safely returns no hit.

    ``noise`` is an optional per-pixel map of how much each pixel normally
    flickers (wind in foliage, rain, decoder speckle). Where it is supplied, a
    change counts only above ``noise_k`` times that pixel's own flicker, so
    restless pixels stop raising alarms while steady tarmac keeps full
    sensitivity. Without it the measurement is exactly as before.
    """
    for name, value, upper in (("min_delta", min_delta, 255),
                               ("min_bright_pct", min_bright_pct, 100),
                               ("noise_k", noise_k, 20)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= upper):
            raise ValueError(f"{name} must be finite and in (0, {upper}]")
    current, reference = _luma(gray, "gray"), _luma(background, "background")
    if current.shape != reference.shape:
        raise ValueError("gray and background must have matching image dimensions")
    region = _mask(region_mask, current.shape, "region_mask")
    scene = np.ones(current.shape, dtype=bool) if scene_mask is None else _mask(
        scene_mask, current.shape, "scene_mask")
    area = int(np.count_nonzero(region))
    report = {"delta": 0.0, "positive_delta": 0.0, "raw_delta": 0.0,
              "global_offset": 0.0, "global_gain": 1.0, "bright_pct": 0.0,
              "largest_patch_pct": 0.0, "hit": False, "support_bounds": None}
    if not area:
        return report
    predicted, offset, gain = _exposure_reference(current, reference, scene)
    # The flicker map is checked against the whole frame, exactly as before, and
    # only then cropped: a caller passing a wrong-sized map must still be told.
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != current.shape or not np.isfinite(noise).all() or float(noise.min()) < 0:
            raise ValueError("noise must be a finite non-negative map matching the image")
    # The exposure fit above needs the whole scene; everything below is confined
    # to the region, so it runs on the region's own box. Measured at 10 ms ->
    # under 1 ms per light zone, which is frame budget the detector gets back.
    box = _region_box(region, pad=8)
    current, reference, predicted, region = current[box], reference[box], predicted[box], region[box]
    if noise is not None:
        noise = noise[box]
    difference = current - predicted
    # Restless pixels (wind in foliage, rain, decoder speckle) raise their own
    # bar: a change counts only where it also exceeds this pixel's learned
    # flicker band. Steady pixels keep the plain threshold, so a headlight
    # wash on quiet tarmac is unaffected.
    if noise is None:
        band = np.zeros(current.shape, dtype=np.float32)
    else:
        band = noise * float(noise_k)
    values = difference[region]                       # signed change, meaning unchanged
    excess = np.maximum(difference - band, 0.0)       # change beyond this pixel's own flicker
    bright = ((excess >= max(2.0, float(min_delta))) & region).astype(np.uint8)
    # A pixel must have at least four bright members in its 3x3 neighborhood.
    # This retains a small 2x2 lamp while rejecting single pixels/thin streaks.
    neighbors = cv2.filter2D(bright, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    supported = (bright & (neighbors >= 4).astype(np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(supported, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    bounds = None
    if largest:
        component = stats[1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())]
        # Where the light appeared, in full-frame pixels (x, y, w, h), for the on-screen outline.
        bounds = [int(component[0]) + box[1].start, int(component[1]) + box[0].start,
                  int(component[2]), int(component[3])]
    positive_delta = float(excess[region].mean())
    bright_pct = 100.0 * int(np.count_nonzero(bright)) / area
    required_patch = max(4, math.ceil(area * .001))
    report.update(delta=float(values.mean()), positive_delta=positive_delta,
                  raw_delta=float((current - reference)[region].mean()),
                  noise_band=round(float(band[region].mean()), 3),
                  global_offset=offset, global_gain=gain, bright_pct=bright_pct,
                  largest_patch_pct=100.0 * largest / area, support_bounds=bounds,
                  hit=bool(positive_delta >= min_delta and bright_pct >= min_bright_pct
                           and largest >= required_patch))
    return report
