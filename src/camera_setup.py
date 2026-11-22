"""Measured setup guidance, separate from detector trust and alarm decisions.

The score is a transparent setup progress measure, never detection accuracy.
No image, reference, zone, camera control or alarm is changed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np


def _area(points):
    return abs(sum(a[0]*b[1]-a[1]*b[0]
                   for a, b in zip(points, points[1:]+points[:1])))*.5


def _clip(points):
    result = list(points)
    for axis, boundary, lower in ((0, 0., True), (0, 1., False),
                                  (1, 0., True), (1, 1., False)):
        source, result = result, []
        if not source:
            break
        previous = source[-1]
        inside_previous = previous[axis] >= boundary if lower else previous[axis] <= boundary
        for point in source:
            inside = point[axis] >= boundary if lower else point[axis] <= boundary
            if inside != inside_previous:
                t = (boundary-previous[axis])/(point[axis]-previous[axis])
                result.append(tuple(previous[k]+t*(point[k]-previous[k]) for k in (0, 1)))
            if inside:
                result.append(tuple(point))
            previous, inside_previous = point, inside
    return result


def _matrix(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0., 0., 1.]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid camera geometry')
    if abs(np.linalg.det(matrix)) < 1e-9:
        raise ValueError('Camera geometry is singular')
    return matrix


def _project(matrix, points):
    projected = np.c_[np.asarray(points, dtype=np.float64), np.ones(len(points))] @ matrix.T
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 1e-7):
        raise ValueError('Camera geometry crosses the viewing plane')
    return projected[:, :2]/projected[:, 2:]


def zone_coverage(matrix, shape, zones):
    """Visible polygon area in the original image; each required zone counts."""
    height, width = shape[:2]
    inverse = np.linalg.inv(_matrix(matrix))
    results, all_points = [], []
    for index, zone in enumerate(zones):
        polygon = zone.get('polygon')
        if not polygon or len(polygon) < 3:
            raise ValueError('A required zone has no valid outline')
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid required zone outline')
        live = (_project(inverse, points*(width, height))/(width, height)).tolist()
        area = _area(live)
        if area < 1e-9:
            raise ValueError('A required zone has no area')
        fraction = min(1., _area(_clip(live))/area)
        # A rounded area percentage must not conceal a clipped corner.
        inside = all(-1e-5 <= x <= 1.+1e-5 and -1e-5 <= y <= 1.+1e-5 for x, y in live)
        results.append({'name': str(zone.get('name', f'Zone {index+1}')),
                        'pct': round(fraction*100, 2), 'visible': inside})
        all_points.extend(live)
    if not results:
        raise ValueError('No required alarm zones configured')
    return {'pct': round(min(item['pct'] for item in results), 2),
            'required_visible': sum(item['visible'] for item in results),
            'total_required': len(results),
            'clipped_names': [item['name'] for item in results if not item['visible']],
            'zones': results}, all_points


def framing_guidance(points, coverage):
    xs, ys = zip(*points)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1-x0, y1-y0)
    factor = .9/max(span, 1e-6)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    clipped = bool(coverage['clipped_names'])
    if clipped and span > .94:
        action, headline = 'zoom_out', 'Zoom out slightly'
        detail = 'Show the entire highlighted road and footpath before saving.'
    elif clipped:
        dx = x0 if x0 < 0 else x1-1 if x1 > 1 else 0
        dy = y0 if y0 < 0 else y1-1 if y1 > 1 else 0
        action = ('turn_right' if dx > 0 else 'turn_left') if abs(dx) >= abs(dy) else ('tilt_down' if dy > 0 else 'tilt_up')
        headline = action.replace('_', ' ').capitalize()
        detail = 'Bring the clipped alarm area back into the original camera picture.'
    elif factor > 1.25:
        action, headline = 'zoom_in', 'A closer view is possible'
        detail = 'Optional: zoom in a little, keeping every road and footpath visible.'
    else:
        action, headline = 'good', 'All alarm areas are in view'
        detail = 'Keep this position and review the routes below. No zoom adjustment is needed.'
    fit = 100. if action == 'good' else min(99., 100.*min(factor, 1./factor))
    return {'action': action, 'headline': headline, 'detail': detail,
            'zoom_factor': round(factor, 3), 'target_centre': [round(cx, 4), round(cy, 4)],
            'fit_pct': round(max(0., fit), 1)}


class CameraSetupAssessment:
    REQUIRED_STABLE_S = 5.
    MAX_GAP_S = .8
    STABLE_DELTA_PX = 2.

    def __init__(self):
        self._basis = None
        self._anchor = None
        self._since = None
        self._last = None
        self._revision = 0
        self._review_basis = None
        self._review_anchor = None
        self._last_strong = None
        self._report = None

    def update(self, *, matrix, shape, zones, alignment, stamp, view, reference_id,
               reference_saved=True, calibration=None):
        report = {'version': 1, 'view': view, 'view_id': view,
                  'state': 'waiting', 'score_pct': None, 'stable_s': 0.,
                  'required_stable_s': self.REQUIRED_STABLE_S, 'ready_to_save': False,
                  'blockers': [], 'reference': {'saved': bool(reference_saved), 'at': None},
                  'calibration': dict(calibration or {}),
                  'alignment': {'state': alignment.get('state'),
                                'landmarks': alignment.get('matches'), 'inliers': alignment.get('inliers'),
                                'reprojection_error_px': alignment.get('median_error_px')},
                  'guidance': {'action': 'wait', 'headline': 'Checking camera position',
                               'detail': 'Wait for a fresh, verified picture. Do not move the camera on an uncertain reading.'},
                  'coverage': {'pct': None, 'required_visible': 0, 'total_required': len(zones),
                               'clipped_names': []}}
        basis = hashlib.sha256(json.dumps([view, reference_id, zones], sort_keys=True).encode()).hexdigest()[:16]
        valid = False
        try:
            if not math.isfinite(stamp):
                raise ValueError('Camera timestamp unavailable')
            if alignment.get('state') != 'tracking' or not alignment.get('accepted'):
                raise ValueError('Wait for verified scene landmarks')
            matrix = _matrix(matrix)
            coverage, points = zone_coverage(matrix, shape, zones)
            report['coverage'] = coverage
            report['guidance'] = framing_guidance(points, coverage)
            height, width = shape[:2]
            anchors = _project(matrix, [(0, 0), (width, 0), (width, height), (0, height), (width/2, height/2)])
            if (basis != self._review_basis or self._review_anchor is None
                    or float(np.linalg.norm(anchors-self._review_anchor, axis=1).max()) > self.STABLE_DELTA_PX):
                self._review_basis, self._review_anchor = basis, anchors.copy()
                self._revision += 1
                self._last_strong = None
            if not alignment.get('weak'):
                self._last_strong = stamp
            if (basis != self._basis or self._anchor is None or self._last is None
                    or not 0 < stamp-self._last <= self.MAX_GAP_S
                    or float(np.linalg.norm(anchors-self._anchor, axis=1).max()) > self.STABLE_DELTA_PX):
                self._anchor, self._since = anchors, stamp
            self._basis, self._last = basis, stamp
            stable = max(0., min(self.REQUIRED_STABLE_S, stamp-self._since))
            report['stable_s'] = round(stable, 2)
            stable_pct = stable/self.REQUIRED_STABLE_S*100.
            landmark_proof = self._last_strong is not None and 0 <= stamp-self._last_strong <= 10.
            if alignment.get('weak'):
                agreement = alignment.get('static_agreement') or {}
                landmark_proof = (landmark_proof and alignment.get('pose_gap_px', 99.) <= 2.
                                  and agreement.get('good_tiles', 0) >= 4
                                  and agreement.get('good_fraction', 0.) >= .8)
            report['components'] = {'coverage_pct': coverage['pct'], 'steady_pct': round(stable_pct, 1),
                                    'framing_pct': report['guidance']['fit_pct'],
                                    'landmark_pct': 100. if landmark_proof else 0.,
                                    'weights': {'coverage': 50, 'steady': 25, 'framing': 15, 'landmarks': 10}}
            report['score_pct'] = round(.5*coverage['pct']+.25*stable_pct+.15*report['guidance']['fit_pct']+10.*landmark_proof)
            if not landmark_proof:
                report['blockers'].append('Wait for a recent distributed landmark check before saving')
            if coverage['clipped_names']:
                report['blockers'].append('Clipped alarm areas: '+', '.join(coverage['clipped_names']))
            if stable < self.REQUIRED_STABLE_S:
                report['blockers'].append('Keep the camera still for five seconds')
            if calibration and calibration.get('pending'):
                report['blockers'].append('Reference capture is still processing')
            report['ready_to_save'] = not report['blockers']
            report['state'] = ('adjust' if coverage['clipped_names'] else
                               'ready' if report['ready_to_save'] else 'stabilizing')
            valid = True
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            report['blockers'] = [str(exc)]
        if not valid:
            self._anchor, self._since, self._last = None, None, None
        report['reference_id'] = f'{basis}:{self._revision}'
        self._report = report
        return copy.deepcopy(report)

    def status(self, now, *, frame_age=None, calibration=None):
        report = copy.deepcopy(self._report) if self._report else None
        if report is None:
            return None
        if calibration is not None:
            report['calibration'] = dict(calibration)
        if (self._last is None or not 0 <= now-self._last <= self.MAX_GAP_S
                or frame_age is None or not 0 <= frame_age < self.MAX_GAP_S):
            report.update(state='waiting', score_pct=None, stable_s=0., ready_to_save=False)
            report['blockers'] = ['Wait for a fresh, verified camera picture']
            report['guidance'] = {'action': 'wait', 'headline': 'Picture not verified',
                                  'detail': 'Check the source feed; no zoom direction is inferred from missing frames.'}
        return report
