"""Local second-view diagnostics, never an alarm veto or self-trained model.

The existing, pinned detector sees a contextual route crop and (in darkness) a
CLAHE version of that crop. Both passes belong to ONE observation. The worker
also measures image quality, structural features and bounded clear-scene noise
baselines. It does not infer road semantics, move zones or learn model weights.
All inference runs outside the detector loop. Only the newest pending frame is
retained; stale or stopped-generation work cannot publish a current result.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from object_detector import ObjectDetector
from regional_inference import merge_view_detections, _family, _iou

try:
    import cv2
except (ImportError, OSError):
    cv2 = None


VIEWS = ('enhanced', 'crop', 'rectified', 'edges', 'change', 'landmarks')
MAX_AGE = 4.0
BUILD = 'second-view-1'


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _zones(values):
    """Snapshot only small, JSON-compatible configuration; never copy masks."""
    result = []
    for value in list(values or ())[:32]:
        polygon = _get(value, 'polygon')
        if polygon is None:
            rect = _get(value, 'rect', ())
            if len(rect) != 4:
                continue
            x, y, w, h = rect
            polygon = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        try:
            points = np.asarray(polygon, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 16
                    or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()):
                continue
            role = str(_get(value, 'route_role', ''))
            if role in ('parking', 'charging', 'ignore'):
                continue
            result.append({'name': str(_get(value, 'name', 'Region'))[:100],
                           'polygon': points.tolist(), 'role': role,
                           'route': str(_get(value, 'route_group', '') or _get(value, 'name', 'Region'))[:100],
                           'active': bool(_get(value, 'motion_active', False) or _get(value, 'light_active', False)),
                           'classes': copy.deepcopy(_get(value, 'classes', None))})
        except (TypeError, ValueError, OverflowError):
            continue
    return result


def _encode(rgb):
    okay, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
    return bytes(encoded) if okay else None


def _enhance(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _finite(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


class SecondaryAnalyzer:
    def __init__(self, model_path, interval=1.0, learning_path=None):
        self.model_path = Path(model_path)
        self.interval = max(.5, min(10., _finite(interval, 1.)))
        self.learning_path = Path(learning_path) if learning_path else None
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._generation = 0
        self._last_submit = -math.inf
        self._published_at = None
        self._images = {}
        self._previous = None
        self._previous_stamp = None
        self._prior_objects = {}
        self._route_cursor = 0
        self._baselines = {}
        self._last_save = -math.inf
        self._state = {'enabled': False, 'state': 'off', 'available': False, 'age_s': None,
                       'verdict': 'watching', 'reason': 'Second view is off', 'route': '',
                       'views': list(VIEWS), 'quality': {}, 'landmarks': {}, 'learning': {},
                       'detections': [], 'comparison': {}, 'inference_ms': None,
                       'processed': 0, 'dropped': 0, 'error': '', 'build': BUILD,
                       'advisory_only': True, 'model_independent': False,
                       'model': 'YOLOX-Tiny: focused second view', 'offline': True}
        self._load_learning()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self._generation += 1
            self._clear_queue()
            self._previous = None
            self._previous_stamp = None
            self._prior_objects = {}
            self._images = {}
            self._published_at = None
            self._last_submit = -math.inf
            self._state.update(enabled=True, state='loading', available=False,
                               reason='Loading the local second-view model', error='',
                               detections=[], quality={}, landmarks={}, comparison={})
            self._thread = threading.Thread(target=self._run, args=(self._generation,),
                                            name='c500-secondary-analysis', daemon=True)
            self._thread.start()
            return True

    def submit(self, frame, zones, primary, alignment, alarm, stamp=None):
        now = time.monotonic()
        stamp = now if stamp is None else _finite(stamp, -math.inf)
        # The detector loop may call this while holding its own state lock.
        # Never make it wait behind diagnostics publication or calibration I/O;
        # the next camera frame can retry without advancing the submit throttle.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if (not self._state['enabled'] or self._stop.is_set()
                    or now-self._last_submit < self.interval):
                return False
            if stamp > now+.25 or now-stamp >= MAX_AGE:
                self._state['dropped'] += 1
                return False
            rgb = np.asarray(frame)
            if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                    or min(rgb.shape[:2]) < 16 or max(rgb.shape[:2]) > 1920):
                return False
            # A bounded copy is the only image work performed on the caller.
            packet = (self._generation, stamp, rgb.copy(), _zones(zones),
                      copy.deepcopy(primary or {}), copy.deepcopy(alignment or {}),
                      copy.deepcopy(alarm or {}))
            if self._clear_queue():
                self._state['dropped'] += 1
            self._queue.put_nowait(packet)
            self._last_submit = now
            return True
        finally:
            self._lock.release()

    def _clear_queue(self):
        removed = False
        try:
            while True:
                self._queue.get_nowait()
                removed = True
        except queue.Empty:
            return removed

    def stop(self, timeout=3.0):
        with self._lock:
            self._stop.set()
            self._generation += 1
            self._clear_queue()
            self._state.update(enabled=False, state='off', available=False,
                               verdict='watching', reason='Second view is off', detections=[])
            self._images = {}
            self._published_at = None
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0., min(5., _finite(timeout, 3.))))
        stopped = not bool(thread and thread.is_alive())
        if stopped:
            self._save_learning(force=True)
        return stopped

    def status(self):
        with self._lock:
            result = copy.deepcopy(self._state)
            age = time.monotonic()-self._published_at if self._published_at is not None else None
            result['age_s'] = round(max(0., age), 2) if age is not None else None
            result['fresh'] = bool(result['enabled'] and age is not None and age < MAX_AGE)
            if result['enabled'] and age is not None and age >= MAX_AGE:
                result.update(state='stale', verdict='watching', reason='Waiting for a fresh camera analysis',
                              detections=[])
            return result

    def image(self, view):
        if view not in VIEWS:
            return None
        with self._lock:
            if (not self._state['enabled'] or self._published_at is None
                    or time.monotonic()-self._published_at >= MAX_AGE):
                return None
            return self._images.get(view)

    def _current(self, generation, stamp):
        return (not self._stop.is_set() and generation == self._generation
                and 0 <= time.monotonic()-stamp < MAX_AGE)

    def _run(self, generation):
        detector = None
        retry_at = 0.0
        while not self._stop.is_set() and generation == self._generation:
            if cv2 is None:
                with self._lock:
                    self._state.update(state='unavailable', error='OpenCV is unavailable', available=False)
                return
            if detector is None and time.monotonic() >= retry_at:
                try:
                    detector = ObjectDetector(self.model_path, confidence=.45, threads=1, engine="processor")
                    if not detector.available:
                        raise RuntimeError(detector.status().get('error', 'Model is unavailable'))
                except Exception as exc:
                    detector = None
                    retry_at = time.monotonic()+15.
                    with self._lock:
                        if generation == self._generation and not self._stop.is_set():
                            self._state.update(state='unavailable', error=str(exc)[:240], available=False)
            try:
                packet = self._queue.get(timeout=.2)
            except queue.Empty:
                continue
            packet_generation, stamp, frame, zones, primary, alignment, alarm = packet
            if packet_generation != generation or not self._current(generation, stamp):
                with self._lock:
                    self._state['dropped'] += 1
                continue
            started = time.monotonic()
            try:
                report, images = self._analyze(frame, zones, primary, alignment, alarm,
                                               stamp, detector, generation)
            except Exception as exc:
                with self._lock:
                    if self._current(generation, stamp):
                        self._state.update(state='error', available=False, error=str(exc)[:240],
                                           verdict='view_unreliable', reason='Second-view processing failed',
                                           detections=[])
                        self._images = {}
                        self._published_at = None
                continue
            with self._lock:
                if not self._current(generation, stamp):
                    self._state['dropped'] += 1
                    continue
                report['inference_ms'] = round((time.monotonic()-started)*1000, 1)
                report['processed'] = self._state['processed']+1
                self._state.update(report)
                self._published_at = stamp
                self._images = images
            if detector is not None and not detector.available:
                detector = None
                retry_at = time.monotonic()+15.

    def _select_route(self, zones, alarm):
        route = _get(alarm, 'route_group', '')
        name = _get(alarm, 'zone_name', '')
        details = _get(alarm, 'details', _get(alarm, 'alarm_details', {})) or {}
        route = route or _get(details, 'route_group', '')
        name = name or _get(details, 'zone_name', '')
        matching = [z for z in zones if (route and z['route'] == route) or (name and z['name'] == name)]
        if not matching:
            keys = list(dict.fromkeys(z['route'] for z in zones))
            if not keys:
                return [], 'Full view'
            route = keys[self._route_cursor % len(keys)]
            self._route_cursor += 1
            matching = [z for z in zones if z['route'] == route]
        return matching, matching[0]['route']

    def _analyze(self, frame, zones, primary, alignment, alarm, stamp, detector, generation):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        brightness = float(np.mean(gray))
        enhanced = _enhance(frame)
        edges = cv2.Canny(gray, 45, 130)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=180, qualityLevel=.025, minDistance=10)
        points = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2))
        cell_ids = {(min(7, int(x/w*8)), min(4, int(y/h*5))) for x, y in points}
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=max(18, w//15), maxLineGap=8)
        # OpenCV 4 returns lines as N x 1 x 4, OpenCV 5 as N x 4: one flat list of (x1, y1, x2, y2) for both.
        lines = np.asarray(lines).reshape(-1, 4)[:60] if lines is not None else []
        drift, exposure, changed, raw_changed, noise = 0., 0., 0., 0., 0.
        lighting = 'dark' if brightness < 65 else 'dim' if brightness < 120 else 'bright'
        baseline = self._baselines.get(lighting, {})
        baseline_used = baseline.get('samples', 0) >= 10
        threshold = max(14., min(28., 6.*baseline.get('noise', 0.)+8.)) if baseline_used else 14.
        adaptive_changed = 0.
        residual = np.zeros_like(gray)
        comparable = (self._previous is not None and self._previous.shape == gray.shape
                      and self._previous_stamp is not None and 0 < stamp-self._previous_stamp < MAX_AGE)
        if comparable:
            difference = gray.astype(np.float32)-self._previous.astype(np.float32)
            exposure = float(np.median(difference))
            residual_float = np.abs(difference-exposure)
            residual = np.clip(residual_float*4, 0, 255).astype(np.uint8)
            changed = float(np.mean(residual_float > 14))
            adaptive_changed = float(np.mean(residual_float > threshold))
            raw_changed = float(np.mean(np.abs(difference) > 14))
            noise = float(np.median(residual_float))
            # Phase correlation is only a diagnostic; never moves any zone.
            if float(gray.std()) > 8 and float(self._previous.std()) > 8:
                delta, response = cv2.phaseCorrelate(self._previous.astype(np.float32), gray.astype(np.float32))
                if response >= .15:
                    drift = min(float(math.hypot(*delta)), max(h, w))
        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        dark, clipped = float(np.mean(gray < 12)), float(np.mean(gray > 248))
        reasons = []
        blocked = (bool(alignment.get('blocked')) or alignment.get('accepted') is False
                   or alignment.get('frame_aligned') is False)
        if blocked:
            reasons.append('Main alignment is not trusted')
        if dark > .92:
            reasons.append('Most of the image is dark or covered')
        if clipped > .75:
            reasons.append('Most of the image is overexposed')
        if blur < 4 and float(gray.std()) < 10:
            reasons.append('Very little usable image detail; possible obstruction')
        if comparable and (changed > .30 or drift > 8):
            reasons.append('Large scene change or residual movement')
        quality = {'brightness': round(brightness, 1), 'blur_variance': round(blur, 1),
                   'dark_fraction': round(dark, 4), 'clipped_fraction': round(clipped, 4),
                   'change_fraction': round(changed, 4), 'raw_change_fraction': round(raw_changed, 4),
                   'exposure_shift': round(exposure, 2), 'drift_px': round(drift, 2),
                   'noise_median': round(noise, 2), 'reliable': not reasons,
                   'reasons': reasons, 'compared_previous': comparable,
                   'adaptive_change_fraction': round(adaptive_changed, 4),
                   'change_threshold': round(threshold, 2),
                   'noise_baseline_used': baseline_used,
                   'noise_baseline_samples': int(baseline.get('samples', 0))}
        chosen, route = self._select_route(zones, alarm)
        if chosen:
            extent = np.concatenate([np.asarray(z['polygon']) for z in chosen])
            x1, y1 = np.maximum(0, np.floor((extent.min(0)-[.06, .12])*[w, h])).astype(int)
            x2, y2 = np.minimum([w, h], np.ceil((extent.max(0)+[.06, .05])*[w, h])).astype(int)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
        rect = (int(x1), int(y1), int(max(x1+1, x2)), int(max(y1+1, y2)))
        x1, y1, x2, y2 = rect
        crop = frame[y1:y2, x1:x2].copy()
        passes = []
        available = bool(detector is not None and detector.available)
        if available and not blocked and self._current(generation, stamp):
            passes.append((detector.detect(crop), rect))
            if brightness < 85 and detector.available and self._current(generation, stamp):
                passes.append((detector.detect(enhanced[y1:y2, x1:x2]), rect))
            available = detector.available
        objects = merge_view_detections(passes) if available else []
        main_age = primary.get('age_s')
        primary_fresh = (primary.get('available') is True and primary.get('state') == 'ready'
                         and isinstance(main_age, (int, float)) and not isinstance(main_age, bool)
                         and math.isfinite(main_age) and 0 <= main_age < 3.)
        all_main_objects = primary.get('detections', []) if primary_fresh else []
        if not isinstance(all_main_objects, list):
            all_main_objects = []
        valid_main_objects = []
        for obj in all_main_objects[:50]:
            try:
                box = obj['bbox']
                if (len(box) == 4 and all(math.isfinite(float(v)) for v in box)
                        and box[2] > box[0] and box[3] > box[1]):
                    valid_main_objects.append(obj)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        all_main_objects = valid_main_objects
        # Compare only primary objects overlapping this crop, while ANY primary
        # object still freezes baseline calibration elsewhere in the image.
        main_objects = [o for o in all_main_objects if _iou(o['bbox'], rect) > 0]
        selected_names = {z['name'] for z in chosen}
        matched = 0
        for obj in objects:
            prior = [p for p in self._prior_objects.get(route, [])
                     if 0 < stamp-p['_stamp'] < MAX_AGE and _family(p['label']) == _family(obj['label'])
                     and _iou(p['bbox'], obj['bbox']) >= .25]
            obs = min(99, max((p['observations'] for p in prior), default=0)+1)
            main_matches = [p for p in main_objects if _family(p.get('label')) == _family(obj['label'])
                            and _iou(p['bbox'], obj['bbox']) >= .25]
            same = bool(main_matches)
            route_confirmed = any(selected_names.intersection(p.get('confirmed_regions', []))
                                  for p in main_matches if isinstance(p.get('confirmed_regions'), list))
            if same:
                matched += 1
            obj.update(route=route, matched_primary=same, observations=obs,
                       confirmed=bool(obs >= 2 and route_confirmed and not reasons),
                       observed_twice=bool(obs >= 2), route_confirmed=route_confirmed, _stamp=stamp)
        repeated = sum(o['matched_primary'] and o['confirmed'] for o in objects)
        comparison = {'primary_count': len(main_objects), 'secondary_count': len(objects),
                      'matched_count': matched, 'repeated_matches': repeated,
                      'repeated_object_matches': sum(o['matched_primary'] and o['observed_twice'] for o in objects),
                      'unmatched_count': len(objects)-matched, 'primary_fresh': primary_fresh,
                      'agreement': ('unavailable' if not available else 'matched' if matched else
                                    'not_seen_again' if main_objects else 'no_primary_object'),
                      'note': 'A missed second-view detection does not establish a false alarm.'}
        if reasons:
            verdict, reason = 'view_unreliable', reasons[0]
        elif not available:
            verdict, reason = 'watching', 'Image diagnostics only; the second-view model is unavailable'
        elif repeated:
            verdict, reason = 'corroborated', 'A primary object matched two recent second-view observations'
        elif alarm or main_objects:
            verdict, reason = 'unconfirmed', ('Object seen again; waiting for repeated route confirmation' if matched
                                             else 'No repeated object match yet; check the live view')
        else:
            verdict, reason = 'watching', 'Scanning the configured route in a focused second view'
        landmark_image = frame.copy()
        for x, y in points:
            cv2.circle(landmark_image, (int(x), int(y)), 2, (45, 230, 215), -1)
        edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        for line in lines:
            ax, ay, bx, by = map(int, line)
            cv2.line(edge_image, (ax, ay), (bx, by), (45, 230, 215), 1)
        zone_reports = []
        for zone in zones:
            polygon = np.rint(np.asarray(zone['polygon'])*[w-1, h-1]).astype(np.int32)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [polygon], 255)
            count = sum(cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) >= 0 for p in points)
            selected = gray[mask > 0]
            visible = float(np.mean((selected > 4) & (selected < 253))) if selected.size else 0.
            # This is photometric visibility, not proof of the intended road boundary.
            zone_reports.append({'name': zone['name'], 'visible_fraction': round(visible, 3),
                                 'landmarks': count,
                                 'status': 'check_view' if visible < .6 or blocked else 'observed'})
            cv2.polylines(landmark_image, [polygon], True, (220, 185, 80), 1)
        overview = enhanced.copy()
        cv2.rectangle(overview, (x1, y1), (x2-1, y2-1), (45, 230, 215), 1)
        for obj in objects:
            ax, ay, bx, by = [int(v) for v in obj['bbox']]
            color = (90, 230, 155) if obj['matched_primary'] else (245, 185, 80)
            cv2.rectangle(overview, (ax, ay), (bx, by), color, 2)
            cv2.putText(overview, obj['label'], (ax, max(12, ay-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        zoom = cv2.resize(enhanced[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
        rectified, rectification = self._rectify(frame, alignment.get('scene_polygon'), (w, h))
        with self._lock:
            learning = self._learn(quality, zones, objects, all_main_objects, primary_fresh,
                                   alignment, alarm, stamp, available,
                                   self._current(generation, stamp))
        landmarks = {'count': len(points), 'coverage': round(len(cell_ids)/40., 3),
                     'lines': len(lines), 'zones': zone_reports,
                     'method': 'Shi-Tomasi corners and Hough line segments',
                     'zone_boundaries_verified': False}
        report = {'state': 'ready' if available else 'diagnostics_only', 'available': available,
                  'error': '' if available else detector.status().get('error', '') if detector else 'Model unavailable',
                  'verdict': verdict, 'reason': reason, 'route': route, 'quality': quality,
                  'landmarks': landmarks, 'learning': learning, 'comparison': comparison,
                  'detections': [{k: v for k, v in o.items() if not k.startswith('_')} for o in objects],
                  'crop': list(rect), 'passes': len(passes), 'dark_enhancement': len(passes) == 2,
                  'rectification': rectification, 'observed_at': stamp,
                  'calibration_verified': False}
        images = {'enhanced': overview, 'crop': zoom, 'rectified': rectified,
                  'edges': edge_image, 'change': cv2.cvtColor(cv2.applyColorMap(residual, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB),
                  'landmarks': landmark_image}
        encoded = {key: _encode(image) for key, image in images.items()}
        if self._current(generation, stamp):
            self._previous, self._previous_stamp = gray.copy(), stamp
            self._prior_objects = {key: [o for o in values if 0 <= stamp-o['_stamp'] < MAX_AGE]
                                   for key, values in self._prior_objects.items()}
            self._prior_objects = {key: values for key, values in self._prior_objects.items() if values}
            if len(self._prior_objects) >= 32 and route not in self._prior_objects:
                self._prior_objects.pop(next(iter(self._prior_objects)))
            self._prior_objects[route] = objects
        return report, encoded

    @staticmethod
    def _rectify(frame, scene_polygon, size):
        """Saved screen-boundary approximation, never used for decisions."""
        h, w = frame.shape[:2]
        unavailable = {'applied': False, 'method': 'No usable saved screen boundary',
                       'calibration_verified': False, 'used_for_inference': False}
        try:
            points = np.asarray(scene_polygon, np.float32)
            if (points.ndim != 2 or points.shape[1] != 2 or not 4 <= len(points) <= 16
                    or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()):
                return frame.copy(), unavailable
        except (TypeError, ValueError):
            return frame.copy(), unavailable
        points = points*[w-1, h-1]
        # The saved visible-screen boundary may include clipped-edge vertices.
        # Distinct extrema define a diagnostic trapezoid, not a newly measured
        # calibration. Degenerate/concave selections are rejected.
        sums, differences = points.sum(1), points[:, 0]-points[:, 1]
        indices = [int(np.argmin(sums)), int(np.argmax(differences)),
                   int(np.argmax(sums)), int(np.argmin(differences))]
        if len(set(indices)) != 4:
            return frame.copy(), unavailable
        box = points[indices].astype(np.float32)
        target = np.float32([[0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1], [0, size[1]-1]])
        if cv2.contourArea(box) < 16 or not cv2.isContourConvex(box):
            return frame.copy(), unavailable
        matrix = cv2.getPerspectiveTransform(box, target)
        return cv2.warpPerspective(frame, matrix, size), {
            'applied': True, 'method': 'Saved screen-boundary rectification; approximate',
            'calibration_verified': False, 'used_for_inference': False}

    def _learn(self, q, zones, objects, main_objects, primary_fresh, alignment, alarm, stamp, available,
               current=True):
        lighting = 'dark' if q['brightness'] < 65 else 'dim' if q['brightness'] < 120 else 'bright'
        freeze = ''
        if not current:
            freeze = 'Frame is stale or the analyzer stopped'
        elif alarm or any(z['active'] for z in zones):
            freeze = 'Alarm or zone activity'
        elif objects or main_objects:
            freeze = 'An object is present'
        elif not primary_fresh or not available:
            freeze = 'Object checks are not fresh and available'
        elif not q['reliable'] or alignment.get('accepted') is not True:
            freeze = 'View or alignment is uncertain'
        elif not q['compared_previous']:
            freeze = 'Waiting for a comparison frame'
        elif q['change_fraction'] > .003 or abs(q['exposure_shift']) > 3 or q['drift_px'] > 1.5:
            freeze = 'Image change; calibration is frozen'
        if not freeze:
            entry = self._baselines.setdefault(lighting, {'samples': 0, 'noise': 0., 'brightness': 0., 'blur': 0.})
            count = min(10000, entry['samples']+1)
            alpha = 1./min(count, 120)
            for name, value in [('noise', q['noise_median']), ('brightness', q['brightness']), ('blur', q['blur_variance'])]:
                entry[name] += alpha*(float(value)-entry[name])
            entry['samples'] = count
            self._save_learning()
        entry = self._baselines.get(lighting, {})
        return {'samples': int(entry.get('samples', 0)),
                'total_samples': sum(int(v['samples']) for v in self._baselines.values()),
                'lighting': lighting, 'updated': not bool(freeze), 'frozen_reason': freeze,
                'noise_baseline': round(entry.get('noise', 0.), 3) if entry else None,
                'brightness_baseline': round(entry.get('brightness', 0.), 2) if entry else None,
                'scope': 'Clear-scene noise and exposure statistics; model weights and zones stay fixed',
                'controls_alarms': False}

    def _load_learning(self):
        if self.learning_path is None:
            return
        try:
            if self.learning_path.stat().st_size > 10000:
                return
            data = json.loads(self.learning_path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('lighting'), dict):
                return
            for key, value in data.get('lighting', {}).items():
                if key not in ('dark', 'dim', 'bright') or not isinstance(value, dict):
                    continue
                samples = value.get('samples')
                if not isinstance(samples, int) or isinstance(samples, bool) or not 0 <= samples <= 10000:
                    continue
                numbers = {k: float(value[k]) for k in ('noise', 'brightness', 'blur')}
                if all(math.isfinite(v) and 0 <= v <= 1000000 for v in numbers.values()):
                    self._baselines[key] = {'samples': samples, **numbers}
        except (OSError, ValueError, TypeError, KeyError):
            return

    def _save_learning(self, force=False):
        if self.learning_path is None or not self._baselines:
            return
        now = time.monotonic()
        if not force and now-self._last_save < 60.:
            return
        try:
            self.learning_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.learning_path.with_suffix(self.learning_path.suffix+'.tmp')
            temporary.write_text(json.dumps({'version': 1, 'lighting': self._baselines}, allow_nan=False), encoding='utf-8')
            temporary.replace(self.learning_path)
            self._last_save = now
        except (OSError, ValueError, TypeError):
            # Calibration persistence must never interrupt camera analysis.
            pass
