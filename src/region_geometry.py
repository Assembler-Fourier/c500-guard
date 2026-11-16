"""Normalized simple polygons shared by configuration, tracking and masks."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9
# Shared with the editor so saved detailed outlines remain editable without
# simplification. Keep validation and rendering work bounded.
MAX_POLYGON_CORNERS = 32


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _on_segment(p, a, b):
    return (abs(_cross(a, b, p)) <= EPS
            and min(a[0], b[0])-EPS <= p[0] <= max(a[0], b[0])+EPS
            and min(a[1], b[1])-EPS <= p[1] <= max(a[1], b[1])+EPS)


def _intersects(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    return ((ab_c * ab_d < -EPS and cd_a * cd_b < -EPS)
            or _on_segment(c, a, b) or _on_segment(d, a, b)
            or _on_segment(a, c, d) or _on_segment(b, c, d))


def validate_polygon(points, label='polygon'):
    if not isinstance(points, (list, tuple)) or not 3 <= len(points) <= MAX_POLYGON_CORNERS:
        raise ValueError(f'{label} needs 3–{MAX_POLYGON_CORNERS} corners')
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f'{label} corners must contain x and y')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError(f'{label} coordinates must be finite numbers from 0 to 1')
        normalized.append([float(v) for v in point])
    count = len(normalized)
    if len({tuple(p) for p in normalized}) != count:
        raise ValueError(f'{label} has duplicate corners')
    area = abs(sum(normalized[i][0]*normalized[(i+1)%count][1]
                   - normalized[(i+1)%count][0]*normalized[i][1] for i in range(count))) / 2
    if area < 1e-6:
        raise ValueError(f'{label} has no usable area')
    for i in range(count):
        for j in range(i+1, count):
            if j == i+1 or (i == 0 and j == count-1):
                continue
            if _intersects(normalized[i], normalized[(i+1)%count],
                           normalized[j], normalized[(j+1)%count]):
                raise ValueError(f'{label} edges must not cross or touch')
    return normalized


def polygon_bounds(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)


def rect_polygon(rect):
    x, y, w, h = rect
    return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]


def point_in_polygon(x, y, points):
    inside = False
    for i, a in enumerate(points):
        b = points[(i+1) % len(points)]
        if _on_segment((x, y), a, b):
            return True
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def region_contains(x, y, polygon, exclusions=()):
    return (math.isfinite(x) and math.isfinite(y)
            and point_in_polygon(x, y, polygon)
            and not any(point_in_polygon(x, y, hole) for hole in exclusions))


def polygon_mask(width, height, polygon, exclusions=()):
    """Sample at pixel centers; exact same boundary rules as ground anchors."""
    xx, yy = np.meshgrid((np.arange(width)+.5)/width, (np.arange(height)+.5)/height)

    def raster(points):
        inside = np.zeros((height, width), dtype=bool)
        boundary = inside.copy()
        for i, a in enumerate(points):
            b = points[(i+1) % len(points)]
            cross = (b[0]-a[0])*(yy-a[1]) - (b[1]-a[1])*(xx-a[0])
            boundary |= ((np.abs(cross) <= EPS) & (xx >= min(a[0], b[0])-EPS)
                         & (xx <= max(a[0], b[0])+EPS) & (yy >= min(a[1], b[1])-EPS)
                         & (yy <= max(a[1], b[1])+EPS))
            if abs(b[1]-a[1]) > EPS:
                inside ^= ((a[1] > yy) != (b[1] > yy)) & (xx < (b[0]-a[0])*(yy-a[1])/(b[1]-a[1])+a[0])
        return inside | boundary

    mask = raster(polygon)
    for hole in exclusions:
        mask &= ~raster(hole)
    return mask
