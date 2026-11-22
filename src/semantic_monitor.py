"""Local CPU object recognition alongside independent motion/fault detection."""
from __future__ import annotations

from collections import deque
import queue
import copy
import math
import threading
import time
from pathlib import Path

import numpy as np

from object_tracking import ObjectVisitTracker
from regional_inference import add_context_detections, inference_rects, merge_view_detections, merge_corrob_view


class _EntranceBudgetExceeded(RuntimeError):
    """A completed, healthy optional inference took too long to use."""


class SemanticMonitor:
    RETRY_INITIAL = 2.0
    RETRY_MAX = 60.0
    MAX_RESULT_AGE = 3.0
    MAX_DIAGNOSTIC_VIEWS = 4
    MAX_DIAGNOSTIC_DETECTIONS = 50
    WEAK_CANDIDATE_CONFIDENCE = .30
    ENTRANCE_BUDGET_MS = 500.
    ENTRANCE_RETRY_INITIAL = 5.
    ENTRANCE_RETRY_MAX = 60.

    def __init__(self, zones, model_path: Path, interval=0.75, confidence=0.60, context_view=False,
                 entrance_verifier_path=None):
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 < confidence <= 1):
            raise ValueError('Default AI confidence must be finite and in (0, 1]')
        self.zones = zones
        self.model_path = model_path
        self.entrance_verifier_path = entrance_verifier_path
        self.interval = interval
        self.confidence = confidence
        # A third, whole-picture view: objects where the focused crops do not look (e.g. a person
        # crossing the main road near the camera). Off unless the caller asks for it.
        self.context_view = bool(context_view)
        # The last few analysed frames, so the caller can crop exactly what the AI's boxes refer to.
        self.keep_frames = False
        self._frames = deque(maxlen=6)
        candidate_confidence = self._candidate_confidence()
        self._input = queue.Queue(maxsize=1)
        self._output = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._result_lock = threading.RLock()
        self._epoch = 0
        self._worker_generation = 0
        self._confirmation_generation = 0
        self._enabled = False
        self._last_submit = 0.0
        self._last_source_submitted = None
        # At most the dequeued packet and the one pending packet. Keep the
        # existing four-field work/result packets and source stamps unchanged.
        self._input_times = {}
        self._published_result = None
        self._pipeline = {'inputs_submitted': 0, 'pending_replaced': 0,
                          'results_published': 0, 'undelivered_replaced': 0,
                          'results_consumed': 0, 'last_inference': None,
                          'last_delivery': None}
        self._last_confirmed_at = None
        self._tracker = ObjectVisitTracker()
        self._status = {'available': False, 'state': 'loading', 'error': '',
                        'inferences': 0, 'detections': [], 'model': 'YOLOX tiny',
                        'mode': 'tracked person/vehicle arrivals in class-specific regions',
                        'confidence_threshold': candidate_confidence, 'recovery_attempts': 0}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _zone_confidence(self, zone):
        """Zero/omitted keeps the old global default as that zone's alarm floor."""
        value = getattr(zone, 'ai_confidence', 0.)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Zone AI confidence must be finite and in [0, 1]')
        return value if value > 0 else self.confidence

    def _candidate_confidence(self):
        thresholds = [self._zone_confidence(zone) for zone in self.zones
                      if getattr(zone, 'ai_enabled', getattr(zone, 'motion', True))
                      and getattr(zone, 'classes', None) != []]
        # Moving-only floors (ai_confidence_moving) admit weaker candidates to the tracker.
        thresholds += [self._moving_floor(zone) for zone in self.zones
                       if self._moving_floor(zone) is not None
                       and getattr(zone, 'ai_enabled', getattr(zone, 'motion', True))]
        return min(thresholds, default=self.confidence)

    @staticmethod
    def _moving_floor(zone):
        value = getattr(zone, 'ai_confidence_moving', None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return float(value) if 0 < value < 1 else None

    def _display_floor(self):
        """Keep the legacy presence floor; opt-in context floors apply by region."""
        thresholds = [self._zone_confidence(zone) for zone in self.zones
                      if getattr(zone, 'ai_enabled', getattr(zone, 'motion', True))
                      and getattr(zone, 'classes', None) != []
                      and getattr(zone, 'ai_context', False) is not True]
        return min(thresholds, default=self.confidence)

    def _context_presence(self, item):
        """A weak first sighting can remain visible along its own tracked route."""
        regions = item.get('regions', [])
        history = item.get('direction', {})
        for zone in self.zones:
            if (getattr(zone, 'ai_context', False) is not True
                    or not getattr(zone, 'ai_enabled', False)
                    or getattr(zone, 'classes', None) == []
                    or (zone.classes is not None and item.get('label') not in zone.classes)
                    or not self._normal_detections([item], self._zone_confidence(zone))):
                continue
            if zone.name in regions:
                return True
            route = getattr(zone, 'route_group', '')
            # Tracker annotations already retain recent region visits. Reuse
            # that identity evidence, without promoting unrelated weak objects
            # elsewhere on the lane or introducing another persistence cache.
            if route and zone.name in history and any(
                    target.name in regions and getattr(target, 'ai_enabled', False)
                    and getattr(target, 'route_group', '') == route
                    and getattr(target, 'route_role', '') == getattr(zone, 'route_role', '')
                    and (target.classes is None or item.get('label') in target.classes)
                    for target in self.zones):
                return True
        return False

    BOOST_INTERVAL = .12      # AI cadence while the exit corner shows something moving in

    def boost(self, seconds=4.):
        """Look every frame (about) for a few seconds: something is entering a lane."""
        try:
            seconds = max(0., min(10., float(seconds)))
        except (TypeError, ValueError):
            return
        self._boost_until = max(getattr(self, '_boost_until', 0.), time.monotonic() + seconds)

    def _current_interval(self, now):
        if now < getattr(self, '_boost_until', 0.):
            return min(self.interval, self.BOOST_INTERVAL)
        return self.interval

    def _inference_confidence(self, normal_confidence):
        """Opt-in motion evidence shares the existing passes, never their alarm gate."""
        enabled = any(getattr(zone, 'approach_motion', False) is True
                      and getattr(zone, 'ai_enabled', getattr(zone, 'motion', True))
                      and getattr(zone, 'classes', None) != [] for zone in self.zones)
        return min(normal_confidence, self.WEAK_CANDIDATE_CONFIDENCE) if enabled else normal_confidence

    @staticmethod
    def _normal_detections(detections, threshold):
        """Use the decoder's unrounded score to preserve its original boundary."""
        normal = []
        for detection in detections:
            try:
                value = detection.get('raw_confidence', detection.get('confidence', 0.))
                # The decoder compares float32 score arrays with this scalar;
                # match that boundary even for .7 (whose float32 value is a
                # fraction below the Python float). Rounded adapters retain
                # their existing comparison when no exact score is supplied.
                floor = float(np.float32(threshold)) if 'raw_confidence' in detection else threshold
                if (not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and floor <= value <= 1.):
                    normal.append(detection)
            except (AttributeError, TypeError, ValueError):
                continue
        return normal

    def _tracking_zones(self):
        # The tracker must keep each zone's own floor, including default zones,
        # after the detector admits weaker candidates for an early approach.
        zones = []
        for zone in self.zones:
            effective = copy.copy(zone)
            effective.ai_confidence = self._zone_confidence(zone)
            zones.append(effective)
        return zones

    def _merge_context_detections(self, objects, detections, rect, focused_rects):
        """Opt-in regions may use the existing whole view inside a focused crop.

        All boxes still belong to one source observation. Other regions retain
        the original focused-crop exclusion and every zone's tracking gates.
        """
        objects = add_context_detections(objects, detections, focused_rects)
        zones = [z for z in self.zones if getattr(z, 'ai_context', False) is True
                 and getattr(z, 'ai_enabled', False) and getattr(z, 'classes', None) != []]
        if not zones or rect[0] != 0 or rect[1] != 0:
            return objects
        width, height = rect[2], rect[3]
        candidates = []
        for obj in detections[:50]:
            try:
                box = [float(v) for v in obj['bbox']]
                if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                        or not 0 <= box[0] < box[2] <= width
                        or not 0 <= box[1] < box[3] <= height):
                    continue
                x, y = (box[0] + box[2]) / (2 * width), box[3] / height
                if any((getattr(z, 'classes', None) is None or obj['label'] in z.classes)
                       and z.contains_point(x, y)
                       and self._normal_detections([obj], self._zone_confidence(z))
                       for z in zones):
                    candidates.append(obj)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                continue
        return (merge_corrob_view(objects, candidates, primary_rects=focused_rects,
                                  supplemental_rect=rect) if candidates else objects)

    def _make_entrance_verifier(self):
        """The optional entrance pass shares this worker and must stay on GPU.

        Tiny remains the primary detector, including its existing CPU fallback.
        A failed optional GPU pass cannot replace it with a slow CPU workload.
        """
        if getattr(self, 'entrance_verifier_path', None) is None:
            return None, {'state': 'disabled'}
        try:
            from object_detector import ObjectDetector
            detector = ObjectDetector(self.entrance_verifier_path, confidence=self._candidate_confidence(),
                expected_sha256='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063',
                engine='graphics', threads=1, input_size=(640, 640),
                model_label='YOLOX-S entrance check', gpu_precision='f32', allow_cpu_fallback=False)
            if not detector.available or detector.provider != 'Intel graphics (OpenVINO)':
                raise RuntimeError(detector.status().get('error') or 'Fast graphics inference unavailable')
            return detector, {**detector.status(), 'state': 'ready'}
        except Exception as exc:
            return None, {'state': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'[:240]}

    def _entrance_verifier_view(self, views):
        """Reuse the full decoded view; never create a camera or new queue."""
        entrance = [z for z in self.zones if getattr(z, 'route_role', '') == 'entrance'
                    and getattr(z, 'ai_enabled', False) and getattr(z, 'classes', None) != []]
        if not entrance:
            return None
        whole = [view for view in views if len(view) > 2 and view[2] in {'context', 'focused whole'}
                 and view[1][0] == 0 and view[1][1] == 0]
        # Wider visual context proved more reliable than the narrow entrance
        # crop in the clean missed-car recording. Only entrance ground points
        # may enter tracking from this second model.
        return whole[-1][:2] if whole else None

    def _entrance_candidates(self, detections, rect, width, height):
        candidates = []
        for obj in detections[:50]:
            try:
                box = [float(v) for v in obj['bbox']]
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    continue
                full = [box[0]+rect[0], box[1]+rect[1], box[2]+rect[0], box[3]+rect[1]]
                if not 0 <= full[0] < full[2] <= width or not 0 <= full[1] < full[3] <= height:
                    continue
                x, y = (full[0]+full[2])/2/width, full[3]/height
                if not any(getattr(z, 'route_role', '') == 'entrance' and getattr(z, 'ai_enabled', False)
                           and (z.classes is None or obj['label'] in z.classes) and z.contains_point(x, y)
                           for z in self.zones):
                    continue
                candidates.append({**obj, 'bbox': full, 'view': 'entrance verifier'})
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return candidates

    def _run(self):
        detector = None
        verifier = None
        verifier_status = {'state': 'disabled'}
        verifier_retry_at = 0.
        verifier_backoff = self.ENTRANCE_RETRY_INITIAL
        backoff = self.RETRY_INITIAL
        while not self._stop.is_set():
            if detector is None:
                try:
                    from object_detector import ObjectDetector
                    normal_confidence = self._candidate_confidence()
                    inference_confidence = self._inference_confidence(normal_confidence)
                    detector = ObjectDetector(self.model_path,
                        confidence=inference_confidence, threads=2)
                    if not detector.available:
                        raise RuntimeError(detector.status().get('error') or 'Local AI initialization failed')
                    verifier, verifier_status = self._make_entrance_verifier()
                    verifier_retry_at = 0.
                    verifier_backoff = self.ENTRANCE_RETRY_INITIAL
                    with self._result_lock:
                        # Do not process a frame retained throughout model loading/recovery.
                        self._clear_queue(self._input)
                        self._input_times.clear()
                        self._status = {**self._status, **detector.status(), 'state': 'ready',
                                        'confidence_threshold': normal_confidence,
                                        'diagnostic_confidence_threshold': inference_confidence,
                                        'entrance_verifier': verifier_status,
                                        'detections': [], 'view_diagnostics': [],
                                        'updated_at': None, '_retry_at': None}
                except Exception as exc:
                    self._retry_failure(exc, backoff)
                    detector = None
                    if self._stop.wait(backoff):
                        break
                    backoff = min(backoff * 2, self.RETRY_MAX)
                    continue
            try:
                epoch, generation, stamp, views = self._input.get(timeout=0.25)
            except queue.Empty:
                continue
            start = time.monotonic()
            with self._result_lock:
                queued_at = self._input_times.pop((epoch, generation, stamp), None)
                if (epoch != self._epoch or generation != self._worker_generation
                        or not 0 <= start - stamp < self.MAX_RESULT_AGE):
                    continue
            try:
                # Confidence is a postprocessing gate. Refresh it after zone
                # edits without changing the ONNX graph or adding inference work.
                normal_confidence = self._candidate_confidence()
                detector.confidence = self._inference_confidence(normal_confidence)
                detector.normal_confidence = (normal_confidence
                                              if detector.confidence < normal_confidence else None)
                view_results, context_results, view_diagnostics = [], [], []
                for item in views:
                    crop, rect = item[0], item[1]
                    if (self._stop.is_set() or epoch != self._epoch
                            or generation != self._worker_generation
                            or time.monotonic() - stamp >= self.MAX_RESULT_AGE):
                        break
                    view_started = time.monotonic()
                    detections = detector.detect(crop)
                    view_ms = round((time.monotonic() - view_started) * 1000, 3)
                    if not detector.available:
                        raise RuntimeError(detector.status().get('error') or 'Local AI inference failed')
                    normal = (self._normal_detections(detections, normal_confidence)
                              if detector.confidence < normal_confidence else detections)
                    is_context = len(item) > 2 and item[2] == 'context'
                    (context_results if is_context else view_results).append((normal, rect))
                    if len(view_diagnostics) < self.MAX_DIAGNOSTIC_VIEWS:
                        diagnostic = self._view_diagnostics(
                            detections, rect, 'context' if is_context else 'focused', view_ms)
                        diagnostic.update(observed_at=stamp, epoch=epoch, generation=generation,
                                          normal_detection_count=len(normal))
                        view_diagnostics.append(diagnostic)
                if len(view_results) + len(context_results) != len(views):
                    continue
                objects = merge_view_detections(view_results)
                for detections, rect in context_results:
                    objects = self._merge_context_detections(
                        objects, detections, rect, [r for _, r in view_results])
                extra_checked = False
                extra_view = self._entrance_verifier_view(views) if verifier is not None else None
                if verifier is not None and extra_view is None:
                    verifier_status = {'state': 'unavailable',
                        'error': 'Entrance check requires the whole picture and an enabled entrance region'}
                extra_now = time.monotonic()
                if extra_view is not None and extra_now < verifier_retry_at:
                    verifier_status = {'state': 'cooldown', '_retry_at': verifier_retry_at,
                        'error': 'Entrance check exceeded its 500 ms processing budget; retrying after cooldown'}
                elif (extra_view is not None
                        and extra_now-stamp >= self.MAX_RESULT_AGE-self.ENTRANCE_BUDGET_MS/1000):
                    # Never spend the last part of a fresh primary result's
                    # lifetime on an optional pass. This is still a post-return
                    # budget, not cancellation of a hung native GPU call.
                    verifier_status = {'state': 'deferred',
                        'error': 'Entrance check deferred to keep the primary result fresh; retrying on the next fresh frame'}
                if (extra_view is not None and not self._stop.is_set() and epoch == self._epoch
                        and generation == self._worker_generation and extra_now >= verifier_retry_at
                        and 0 <= extra_now-stamp < self.MAX_RESULT_AGE-self.ENTRANCE_BUDGET_MS/1000):
                    crop, rect = extra_view
                    try:
                        verifier.confidence = normal_confidence
                        extra_checked = True
                        view_started = time.monotonic()
                        extra = verifier.detect(crop)
                        elapsed_ms = (time.monotonic()-view_started)*1000
                        if not verifier.available or verifier.provider != 'Intel graphics (OpenVINO)':
                            raise RuntimeError(verifier.status().get('error') or 'Entrance graphics check unavailable')
                        if elapsed_ms > self.ENTRANCE_BUDGET_MS:
                            raise _EntranceBudgetExceeded('Entrance check exceeded its 500 ms processing budget')
                        # All boxes belong to ONE source frame. Combining model
                        # views cannot count as a second temporal observation.
                        width, height = rect[2], rect[3]
                        accepted = self._entrance_candidates(extra, rect, width, height)
                        objects = merge_corrob_view(objects, accepted,
                            primary_rects=[r for _, r in view_results], supplemental_rect=rect)
                        diagnostic = self._view_diagnostics(extra, rect, 'entrance verifier', round(elapsed_ms, 3))
                        diagnostic.update(observed_at=stamp, epoch=epoch, generation=generation,
                                          normal_detection_count=len(accepted))
                        view_diagnostics.append(diagnostic)
                        verifier_status = {**verifier.status(), 'state': 'ready', 'accepted': len(accepted)}
                        verifier_retry_at = 0.
                        verifier_backoff = self.ENTRANCE_RETRY_INITIAL
                    except _EntranceBudgetExceeded as exc:
                        # A transient contention spike is not a broken model.
                        # Keep the already loaded GPU request, discard this
                        # result and probe later on the existing worker. The
                        # primary keeps running throughout the cooldown.
                        verifier_retry_at = time.monotonic() + verifier_backoff
                        verifier_backoff = min(verifier_backoff * 2, self.ENTRANCE_RETRY_MAX)
                        verifier_status = {'state': 'cooldown', '_retry_at': verifier_retry_at,
                            'error': f'{exc}; retrying after cooldown',
                            'inference_ms': round(elapsed_ms, 3)}
                    except Exception as exc:
                        verifier_status = {'state': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'[:240]}
                        verifier = None
            except Exception as exc:
                self._retry_failure(exc, backoff)
                detector = None
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self.RETRY_MAX)
                continue
            backoff = self.RETRY_INITIAL
            with self._result_lock:
                if self._stop.is_set():
                    break
                published_at = time.monotonic()
                if (epoch != self._epoch or generation != self._worker_generation
                        or not 0 <= published_at - stamp < self.MAX_RESULT_AGE):
                    continue
                self._status = {**self._status, **detector.status(), 'state': 'ready',
                                'model': (detector.status().get('model', 'YOLOX-Tiny')
                                          + (' + YOLOX-S entrance' if verifier_status.get('state') == 'ready' else '')),
                                'confidence_threshold': normal_confidence,
                                'diagnostic_confidence_threshold': detector.confidence,
                                'inferences': self._status['inferences'] + 1,
                                'inference_ms': round((published_at - start) * 1000),
                                'inference_views': len(views) + int(extra_checked),
                                'inference_mode': ('focused routes' if len(views) - len(context_results) > 1
                                                   else 'region union')
                                                  + (' + whole picture' if context_results else '')
                                                  + (' + entrance check' if extra_checked else ''),
                                'inference_crops': [list(item[1]) for item in views],
                                'view_diagnostics': view_diagnostics,
                                'entrance_verifier': verifier_status,
                                'detections': objects, 'updated_at': stamp, '_epoch': epoch,
                                '_generation': generation}
                self._pipeline['last_inference'] = {
                    'observed_at': stamp, 'queued_at': queued_at,
                    'started_at': start, 'published_at': published_at,
                    'source_to_enqueue_ms': (round((queued_at-stamp)*1000, 3)
                                             if queued_at is not None else None),
                    'queue_wait_ms': (round(max(0., start-queued_at)*1000, 3)
                                      if queued_at is not None else None),
                    'source_to_result_ms': round((published_at-stamp)*1000, 3)}
                self._pipeline['results_published'] += 1
                if not self._output.empty():
                    self._pipeline['undelivered_replaced'] += 1
                self._clear_queue(self._output)
                self._output.put_nowait((epoch, generation, stamp, objects))
                self._published_result = ((epoch, generation, stamp), published_at)

    @classmethod
    def _view_diagnostics(cls, detections, rect, kind, inference_ms):
        """Bounded pre-merge evidence; never retain pixels or adapter extras."""
        records = []
        for detection in detections[:cls.MAX_DIAGNOSTIC_DETECTIONS]:
            try:
                label = detection['label']
                score = float(detection['confidence'])
                bbox = [float(value) for value in detection['bbox']]
                if (not isinstance(label, str) or len(bbox) != 4
                        or not math.isfinite(score) or not 0 <= score <= 1
                        or not all(math.isfinite(value) for value in bbox)):
                    continue
                records.append({'label': label[:32], 'confidence': score, 'bbox': bbox,
                                'bbox_frame': [bbox[0]+rect[0], bbox[1]+rect[1],
                                               bbox[2]+rect[0], bbox[3]+rect[1]]})
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return {'kind': kind, 'rect': list(rect), 'inference_ms': inference_ms,
                'detection_count': len(detections), 'detections': records}

    @staticmethod
    def _clear_queue(target):
        try:
            while True:
                target.get_nowait()
        except queue.Empty:
            pass

    def _retry_failure(self, exc, delay):
        with self._result_lock:
            # Results before a model failure must never complete a post-recovery pair.
            self._worker_generation += 1
            self._clear_queue(self._input)
            self._clear_queue(self._output)
            self._input_times.clear()
            self._published_result = None
            self._pipeline['last_inference'] = self._pipeline['last_delivery'] = None
            self._status = {**self._status, 'available': False, 'state': 'retrying',
                            'error': str(exc), 'detections': [], 'view_diagnostics': [],
                            'updated_at': None,
                            'recovery_attempts': self._status['recovery_attempts'] + 1,
                            '_retry_at': time.monotonic() + delay}

    def reset(self):
        with self._result_lock:
            self._epoch += 1
            if hasattr(self, '_tracker'):
                self._tracker.reset()
            self._last_confirmed_at = None
            self._last_source_submitted = None
            if hasattr(self, '_input_times'):
                self._clear_queue(self._input)
                self._clear_queue(self._output)
                self._input_times.clear()
                self._published_result = None
                self._pipeline['last_inference'] = self._pipeline['last_delivery'] = None

    def _confirm(self, objects, width, height, stamp=None):
        if not hasattr(self, '_tracker'):
            self._tracker = ObjectVisitTracker()
        events, observed = self._tracker.update(
            objects, self._tracking_zones(), width, height,
            time.monotonic() if stamp is None else stamp)
        if hasattr(self, '_status'):
            floor = self._display_floor()
            # A lower first-sighting floor must not turn an unrelated lane's
            # weak box into presence that prolongs an acknowledged visit.
            self._status['detections'] = [item for item in observed
                                          if float(item.get('confidence', 1.)) >= floor
                                          or item.get('confirmed_regions')
                                          or self._context_presence(item)]
        return events

    def mark_emitted(self, event_id):
        """Consume a region visit only after the caller actually raises its alarm."""
        with self._result_lock:
            return self._tracker.mark_emitted(event_id)

    def event_context(self, event_id):
        with self._result_lock:
            return self._tracker.event_context(event_id, self.zones)

    def update(self, frame, enabled, stamp=None):
        """Consume recent results and queue pixels with their acquisition time.

        ``stamp`` is the local decoder receipt time, not camera wall time. Old
        pixels must not acquire a fresh inference window after slow alignment.
        Omission preserves callers which provide a newly acquired frame here.
        """
        now = time.monotonic()
        submitted_at = now if stamp is None else stamp
        events = []
        with self._result_lock:
            if (enabled != self._enabled
                    or self._worker_generation != self._confirmation_generation):
                self.reset()
                self._enabled = enabled
                self._confirmation_generation = self._worker_generation
            now = time.monotonic()
            if (isinstance(submitted_at, bool) or not isinstance(submitted_at, (int, float))
                    or not math.isfinite(submitted_at)
                    or not 0 <= now-submitted_at < self.MAX_RESULT_AGE):
                # An expired input cannot clear tracks, consume another fresh
                # result, or restart the source timestamp. Explicit OFF/recovery
                # changes above still invalidate work as required.
                return []
            gap_expired = (self._last_confirmed_at is not None
                           and now - self._last_confirmed_at >= self.MAX_RESULT_AGE)
            consumed = False
            try:
                epoch, generation, stamp, objects = self._output.get_nowait()
                if (enabled and epoch == self._epoch and generation == self._worker_generation
                        and 0 <= now - stamp < self.MAX_RESULT_AGE):
                    # A busy main loop can leave a NEW result waiting while the
                    # last consumed observation ages out. Reset only temporal
                    # confirmation here: changing the worker epoch first would
                    # discard that fresh result and prolong recovery. An old
                    # observation must still never complete its first pair.
                    if gap_expired:
                        self._tracker.reset()
                        self._last_confirmed_at = None
                    events = self._confirm(objects, frame.shape[1], frame.shape[0], stamp)
                    self._last_confirmed_at = stamp
                    publication = self._published_result
                    published_at = (publication[1] if publication is not None
                                    and publication[0] == (epoch, generation, stamp) else None)
                    self._pipeline['results_consumed'] += 1
                    self._pipeline['last_delivery'] = {
                        'observed_at': stamp, 'delivered_at': now,
                        'result_wait_ms': (round(max(0., now-published_at)*1000, 3)
                                           if published_at is not None else None),
                        'source_to_delivery_ms': round((now-stamp)*1000, 3)}
                    self._published_result = None
                    consumed = True
            except queue.Empty:
                pass
            if gap_expired and not consumed:
                # With no usable fresh result, keep the existing hard boundary
                # so stale queued/in-flight work cannot bridge a missing view.
                self.reset()
            submission_epoch, submission_generation = self._epoch, self._worker_generation
        now = time.monotonic()
        if (0 <= now-submitted_at < self.MAX_RESULT_AGE
                and now - self._last_submit >= self._current_interval(now)):
            h, w = frame.shape[:2]
            rects = inference_rects(self.zones, w, h)
            if rects:
                # All views belong to this same aligned frame. Merge them before
                # tracking so two model passes cannot count as two observations.
                views = [(np.asarray(frame[y1:y2, x1:x2], dtype=np.uint8).copy(), rect)
                         for rect in rects for x1, y1, x2, y2 in [rect]]
                if self.context_view:
                    if (0, 0, w, h) in rects:
                        # This focused crop already is the complete picture.
                        # Preserve its primary merge semantics and reuse these
                        # pixels for the optional verifier, without another
                        # primary pass or another temporal observation.
                        index = rects.index((0, 0, w, h))
                        views[index] = (*views[index], 'focused whole')
                    else:
                        views.append((np.asarray(frame, dtype=np.uint8).copy(), (0, 0, w, h), 'context'))
                # Cropping can itself be delayed under contention. Check again
                # instead of handing already expired pixels to the worker.
                whole = None
                if self.keep_frames:
                    whole = views[-1][0] if len(views[-1]) > 2 else np.asarray(frame, dtype=np.uint8).copy()
                with self._result_lock:
                    queued_at = time.monotonic()
                    if (submission_epoch != self._epoch
                            or submission_generation != self._worker_generation
                            or not 0 <= queued_at-submitted_at < self.MAX_RESULT_AGE
                            or queued_at-self._last_submit < self._current_interval(queued_at)
                            or (self._last_source_submitted is not None
                                and submitted_at <= self._last_source_submitted)):
                        return events
                    # One producer under this lock; the worker may dequeue at
                    # any time. Replace pending work, never the in-flight frame.
                    try:
                        previous = self._input.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        self._input_times.pop(previous[:3], None)
                        self._pipeline['pending_replaced'] += 1
                    key = (submission_epoch, submission_generation, submitted_at)
                    self._input.put_nowait((*key, views))
                    self._input_times[key] = queued_at
                    while len(self._input_times) > 2:
                        self._input_times.pop(next(iter(self._input_times)))
                    self._pipeline['inputs_submitted'] += 1
                    self._last_submit, self._last_source_submitted = queued_at, submitted_at
                    if whole is not None:
                        self._frames.append((submitted_at, whole))
        return events

    def frame_at(self, stamp):
        """The analysed frame taken at ``stamp`` (None if it is no longer kept)."""
        for taken, frame in reversed(self._frames):
            if taken == stamp:
                return frame
        return None

    def status(self):
        with self._result_lock:
            report = dict(self._status)
            report['view_diagnostics'] = copy.deepcopy(report.get('view_diagnostics', []))
            report['pipeline'] = copy.deepcopy(getattr(self, '_pipeline', {}))
            report['pipeline']['pending_inputs'] = self._input.qsize() if hasattr(self, '_input') else 0
            report['tracked_objects'] = self._tracker.track_count if hasattr(self, '_tracker') else 0
            report['active_visits'] = self._tracker.active_visits if hasattr(self, '_tracker') else 0
        updated = report.pop('updated_at', None)
        # Consumers accumulating clear-road evidence need the original pixel
        # timestamp. Reconstructing it from rounded age_s creates a different
        # apparent observation every time the same inference result is polled.
        report['observed_at'] = updated
        epoch = report.pop('_epoch', self._epoch)
        generation = report.pop('_generation', getattr(self, '_worker_generation', 0))
        report['tracking_generation'] = [epoch, generation]
        retry_at = report.pop('_retry_at', None)
        report['retry_in_s'] = round(max(0, retry_at - time.monotonic()), 1) if retry_at else None
        if isinstance(report.get('entrance_verifier'), dict):
            verifier_report = dict(report['entrance_verifier'])
            verifier_retry_at = verifier_report.pop('_retry_at', None)
            if verifier_retry_at is not None:
                verifier_report['retry_in_s'] = round(max(0., verifier_retry_at-time.monotonic()), 1)
            report['entrance_verifier'] = verifier_report
        report['age_s'] = round(time.monotonic()-updated, 1) if updated else None
        if report.get('available') and (epoch != self._epoch
                or generation != getattr(self, '_worker_generation', 0)
                or updated is None or report['age_s'] >= self.MAX_RESULT_AGE):
            report['state'] = 'waiting_for_frames'
            report['detections'] = []
            report['view_diagnostics'] = []
            report['pipeline']['last_inference'] = None
            report['pipeline']['last_delivery'] = None
        return report

    def stop(self):
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
