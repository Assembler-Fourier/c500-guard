"""Correlate accepted image warnings and object sightings into bounded visits.

This layer does not classify images or override detector confidence/direction
gates. A route without an object identity is one *unidentified episode*. Its
first confirmed object may inherit that warning; distinct confirmed objects
remain separate even when they use the same road at the same time. A consumed
visit ends after observed clear time, never just because acknowledgement time
elapsed. Missing camera/AI evidence is unknown, not proof that the road cleared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading

from region_geometry import point_in_polygon


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(x) for x in value):
        return None
    a, b, c, d = value
    return tuple(float(x) for x in value) if 0 <= a < c <= 1 and 0 <= b < d <= 1 else None


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap)


def _family(label):
    return 'vehicle' if label in ('car', 'truck', 'bus') else label


def _strong_stopline_proof(details, zone, now, freshness_s):
    """Validate the tracker's explicit two-frame proof without weakening visits."""
    threshold = _get(zone, 'ai_fast_confirm_confidence', 0.)
    gate = _get(zone, 'confirmation_polygon')
    contains = _get(zone, 'contains_point')
    proof = details.get('strong_confirmation')
    label = details.get('label')
    if (not _number(threshold) or not .60 <= threshold <= 1.
            or _get(zone, 'alert_stage') != 'barrier' or not gate or not callable(contains)
            or _get(zone, 'ai_confirm_observations', 2) < 3
            or label not in ('car', 'motorcycle', 'bus', 'truck')
            or (_get(zone, 'classes') is not None and label not in _get(zone, 'classes'))
            or details.get('confirmation_mode') != 'strong_pair'
            or type(details.get('observations')) is not int or details['observations'] != 2
            or not isinstance(proof, (list, tuple)) or len(proof) != 2):
        return False
    direction = _get(zone, 'arrival_direction')
    permitted = ('unrestricted',) if direction is None else ('toward', 'stationary')
    for sample in proof:
        if not isinstance(sample, dict):
            return False
        stamp, confidence, point = (sample.get('observed_at'), sample.get('confidence'),
                                    sample.get('ground_point'))
        if (not _number(stamp) or not 0 <= now-stamp < freshness_s
                or not _number(confidence) or not max(threshold, _get(zone, 'ai_confidence', 0.)) <= confidence <= 1.
                or not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(_number(value) and 0 <= value <= 1 for value in point)
                or sample.get('direction_state') not in permitted
                or (sample.get('direction_state') == 'stationary' and _get(zone, 'stationary_confirm_s', 0.) <= 0)
                or not contains(*point) or not point_in_polygon(*point, gate)):
            return False
    first, last = proof
    box = _box(details.get('bbox_normalized'))
    return bool(box is not None
                and 0 < last['observed_at']-first['observed_at'] <= 1.
                and last['observed_at'] == details.get('observed_at')
                and last['confidence'] == details.get('confidence')
                and last['direction_state'] == details.get('direction_state')
                and _number(details.get('first_observed_at'))
                and details['first_observed_at'] <= first['observed_at']
                and math.dist(last['ground_point'], ((box[0]+box[2])/2, box[3])) <= 1e-6)


@dataclass(frozen=True)
class VisitDecision:
    visit_id: str | None
    suppress: bool
    reason: str
    matched: bool = False


@dataclass
class _Visit:
    id: str
    route: str
    created_at: float
    last_seen: float
    source: str
    label: str = ''
    track_ids: set = field(default_factory=set)
    bbox: tuple | None = None
    confirmed: bool = False
    last_confirmed: float | None = None
    emitted: bool = False
    acknowledged: bool = False
    alarm_ids: set = field(default_factory=set)
    clear_since: float | None = None
    retired_at: float | None = None
    separate_tracks: set = field(default_factory=set)


class VisitMemory:
    """Thread-safe association with a six-second continuously observed clear gate.

    Call observe after image-channel measurements and again after fresh semantic
    results. consider returns a stable visit ID without consuming it. Only call
    emitted after the alarm owner accepts a real alarm; quiet-time/rank rejection
    therefore leaves a different car eligible for its next current observation.
    """
    def __init__(self, *, clear_dwell_s=6., reidentify_s=8., freshness_s=3.,
                 max_visits=128, retired_keep_s=60.):
        for name, value in (('clear_dwell_s', clear_dwell_s), ('reidentify_s', reidentify_s),
                            ('freshness_s', freshness_s), ('retired_keep_s', retired_keep_s)):
            if not _number(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(max_visits, bool) or not isinstance(max_visits, int) or max_visits < 1:
            raise ValueError('max_visits must be a positive integer')
        self.clear_dwell_s = float(clear_dwell_s)
        self.reidentify_s = float(reidentify_s)
        self.freshness_s = float(freshness_s)
        self.max_visits = max_visits
        self.retired_keep_s = float(retired_keep_s)
        self._lock = threading.RLock()
        self._visits = {}
        self._routes = {}
        self._visible_tracks = set()
        self._next_id = 1
        self._last_now = None
        self._last_observation = None
        self._suppressed = 0
        self._current_alarm_id = None

    def _time(self, now):
        if not _number(now):
            raise ValueError('Visit time must be finite')
        # Caller threads can acquire this lock in a different order from their
        # timestamp reads. An acknowledgement followed by a marginally older
        # detector timestamp must never erase the acknowledged visit.
        self._last_now = max(now, self._last_now) if self._last_now is not None else now

    def reset(self):
        """Explicit rearm/camera-layout reset; IDs stay unique for diagnostics."""
        with self._lock:
            self._visits.clear()
            self._routes.clear()
            self._visible_tracks.clear()
            self._last_now = self._last_observation = None
            self._current_alarm_id = None

    @staticmethod
    def route_key(details):
        return str(_get(details, 'route_group') or _get(details, 'zone_name') or _get(details, 'name') or '')[:160]

    def observe(self, now, zones, ml=None, *, frame_fresh=True, width=640, height=360):
        """Use only already-validated channel activity and fresh AI observations.

        Fresh detections without tracker annotations may update known identities,
        but cannot declare a road clear. SemanticMonitor briefly publishes raw
        worker detections before its tracking update; that interval is unknown.
        """
        with self._lock:
            self._time(now)
            previous_observation = self._last_observation
            if previous_observation is None or now-previous_observation >= self.freshness_s:
                for visit in self._visits.values():
                    visit.clear_since = None
            self._last_observation = now if frame_fresh else None
            routes, by_name = {}, {}
            for zone in list(zones or [])[:100]:
                route = self.route_key(zone)
                if not route or _get(zone, 'route_role') in ('parking', 'charging', 'ignore'):
                    continue
                by_name[str(_get(zone, 'name', ''))] = route
                state = routes.setdefault(route, {'signal': False, 'occupied': False,
                    'ml_fresh': False, 'frame_fresh': bool(frame_fresh), 'observed_at': now})
                state['signal'] |= bool(_get(zone, 'motion_active', False) or _get(zone, 'light_active', False)
                                        or _get(zone, 'approach_motion_active', False))
            ml = ml if isinstance(ml, dict) else {}
            age = ml.get('age_s')
            fresh = (frame_fresh and ml.get('available') is True and ml.get('state') == 'ready'
                     and _number(age) and 0 <= age < self.freshness_s)
            detections = ml.get('detections', []) if fresh else []
            if not isinstance(detections, list):
                detections, fresh = [], False
            annotated = fresh and all(isinstance(obj, dict) and 'regions' in obj for obj in detections)
            stamp = now - age if fresh else now
            self._visible_tracks = {str(obj['track_id']) for obj in detections[:100]
                                    if isinstance(obj, dict) and obj.get('track_id') is not None}
            for state in routes.values():
                state['ml_fresh'] = annotated
            seen_visits = set()
            for obj in detections[:100]:
                if not isinstance(obj, dict):
                    continue
                regions = obj.get('regions', [])
                object_routes = {by_name[name] for name in regions if isinstance(name, str) and name in by_name} if isinstance(regions, list) else set()
                for route in object_routes:
                    routes[route]['occupied'] = True
                confirmed_regions = obj.get('confirmed_regions', [])
                confirmed_routes = ({by_name[name] for name in confirmed_regions
                                     if isinstance(name, str) and name in by_name}
                                    if isinstance(confirmed_regions, list) else set())
                track = str(obj['track_id']) if obj.get('track_id') is not None else None
                bbox = _box(obj.get('bbox_normalized'))
                raw_box = obj.get('bbox')
                if bbox is None and isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 and width > 0 and height > 0:
                    try:
                        bbox = _box([raw_box[0]/width, raw_box[1]/height, raw_box[2]/width, raw_box[3]/height])
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                for visit in self._visits.values():
                    if (visit.retired_at is None and visit.route in object_routes
                            and track is not None and track not in visit.track_ids
                            and visit.track_ids & self._visible_tracks):
                        # An independently visible follower must never inherit
                        # the first car's unanswered warning after occlusion.
                        if len(visit.separate_tracks) < 100:
                            visit.separate_tracks.add(track)
                    if visit.retired_at is None and track in visit.track_ids and visit.route in object_routes:
                        visit.last_seen = max(visit.last_seen, stamp)
                        if visit.route in confirmed_routes:
                            visit.last_confirmed = max(visit.last_confirmed or stamp, stamp)
                        visit.clear_since = None
                        if bbox:
                            visit.bbox = bbox
                        seen_visits.add(visit.id)
            self._routes = routes
            for visit in self._visits.values():
                if visit.retired_at is not None:
                    continue
                state = routes.get(visit.route)
                if not state or not frame_fresh:
                    visit.clear_since = None
                    continue
                if visit.id in seen_visits or state['signal'] or (not visit.confirmed and state['occupied']):
                    if not visit.confirmed and state['signal']:
                        visit.last_seen = now
                    visit.clear_since = None
                elif not state['ml_fresh']:
                    # A model gap or unannotated worker result never proves exit.
                    visit.clear_since = None
                else:
                    if visit.clear_since is None:
                        visit.clear_since = now
                    if now - visit.clear_since >= self.clear_dwell_s:
                        visit.retired_at = now
            self._prune(now)

    def _prune(self, now):
        def pinned(visit):
            return self._current_alarm_id is not None and self._current_alarm_id in visit.alarm_ids
        self._visits = {key: visit for key, visit in self._visits.items()
                        if visit.retired_at is None or pinned(visit)
                        or now - visit.retired_at < self.retired_keep_s}
        while len(self._visits) >= self.max_visits:
            retired = [visit for visit in self._visits.values() if visit.retired_at is not None and not pinned(visit)]
            if not retired:
                break
            del self._visits[min(retired, key=lambda visit: visit.retired_at).id]

    @staticmethod
    def _valid_candidate(details):
        if _get(details, 'route_role') in ('parking', 'charging', 'ignore'):
            return False
        if _get(details, 'direction_state') in ('away', 'moving away'):
            return False
        measurement = _get(details, 'measurement', {})
        # The image detector remains the authority. Respect an explicit rejection
        # if a caller accidentally submits its rejected measurement as an alarm.
        if isinstance(measurement, dict):
            if measurement.get('exposure_only') is True or measurement.get('rejected') is True:
                return False
            if measurement.get('hit') is False and not measurement.get('confirmed'):
                return False
        return True

    def consider(self, details, now):
        with self._lock:
            self._time(now)
            details = details if isinstance(details, dict) else {}
            route = self.route_key(details)
            if not route:
                return VisitDecision(None, False, 'unscoped event')
            if not self._valid_candidate(details):
                return VisitDecision(None, True, 'detector rejected or excluded route')
            active = [visit for visit in self._visits.values() if visit.route == route and visit.retired_at is None]
            source = str(details.get('source') or 'image')
            stamp = now
            if source == 'ai' and 'observed_at' in details:
                stamp = details['observed_at']
                if not _number(stamp) or not 0 <= now-stamp < self.freshness_s:
                    return VisitDecision(None, True, 'stale AI observation')
            track = str(details['track_id']) if details.get('track_id') is not None and source == 'ai' else None
            bbox = _box(details.get('bbox_normalized'))
            label = str(details.get('label') or '')
            visit, matched = None, False
            if track is not None:
                visit = next((item for item in active if track in item.track_ids), None)
                if visit is None and bbox:
                    # Re-identify only after the old track disappears. Two visible
                    # cars, even overlapping cars, must retain separate visits.
                    candidates = [item for item in active if item.confirmed and item.bbox
                        and _family(item.label) == _family(label)
                        and not (item.track_ids & self._visible_tracks)
                        and 0 <= now-item.last_seen <= self.reidentify_s
                        and _iou(item.bbox, bbox) >= .35]
                    if candidates:
                        visit = max(candidates, key=lambda item: _iou(item.bbox, bbox))
                if visit is None:
                    # A single image warning can credit exactly one subsequently
                    # recognized identity, including AI confirmation after 5 sec.
                    provisional = [item for item in active if not item.confirmed]
                    if provisional:
                        visit = min(provisional, key=lambda item: item.created_at)
                if visit:
                    visit.track_ids.add(track)
                    # Keep bounded aliases despite repeated tracker recovery.
                    if len(visit.track_ids) > 16:
                        visit.track_ids = {track}
                    visit.confirmed = True
                    visit.last_confirmed = stamp
                    visit.label = label or visit.label
                    visit.bbox = bbox or visit.bbox
            elif active:
                # Anonymous lights cannot reliably identify a second car. They
                # belong to the current observed road episode; a distinct AI
                # track is always eligible independently of this association.
                visit = max(active, key=lambda item: item.last_seen)
            if visit is not None:
                matched = True
                visit.clear_since = None
                visit.last_seen = max(visit.last_seen, stamp)
            else:
                self._prune(now)
                if len(self._visits) >= self.max_visits:
                    # Capacity exhaustion may permit extra alarms, never mute a
                    # new approach by evicting/borrowing another object's ID.
                    return VisitDecision(None, False, 'visit capacity reached')
                ident = f'visit-{self._next_id}'
                self._next_id += 1
                visit = _Visit(ident, route, now, stamp, source, label=label,
                               track_ids={track} if track is not None else set(),
                               bbox=bbox, confirmed=track is not None,
                               last_confirmed=stamp if track is not None else None)
                self._visits[ident] = visit
            if visit.emitted:
                self._suppressed += 1
            return VisitDecision(visit.id, visit.emitted,
                'acknowledged visit' if visit.acknowledged else 'already announced visit' if visit.emitted else 'new visit', matched)

    def continue_unanswered(self, visit_id, alarm_id, previous, details, zones, now):
        """Continue an unanswered approach through a proven short identity break.

        This narrow association applies only to the alarm still sounding. It
        never extends acknowledged-visit identity or joins a known visible
        second car. All checks and alias mutation share the memory lock.
        """
        with self._lock:
            self._time(now)
            visit = self._visits.get(visit_id)
            same_advance = (previous.get('alert_stage') == details.get('alert_stage') == 'advance'
                            and previous.get('zone_name') == details.get('zone_name'))
            if (not visit or not visit.emitted or visit.acknowledged
                    or str(alarm_id) != self._current_alarm_id or str(alarm_id) not in visit.alarm_ids
                    or previous.get('source') != 'ai' or details.get('source') != 'ai'
                    or previous.get('alert_stage') != 'advance'
                    or (details.get('alert_stage') != 'barrier' and not same_advance)
                    or self.route_key(previous) != visit.route or self.route_key(details) != visit.route
                    or not self._valid_candidate(details)):
                return None
            stamp, first = details.get('observed_at'), details.get('first_observed_at')
            if (not _number(stamp) or not 0 <= now-stamp < self.freshness_s
                    or not _number(first) or not visit.last_seen < first <= stamp
                    or not 0 <= now-visit.last_seen <= min(8.,self.reidentify_s)):
                return None
            # Same-region fragments must start just after the last real sighting.
            # The longer existing barrier handoff remains separately bounded.
            if same_advance and first-visit.last_seen > .75:
                return None
            # A fresh road-clear interval completed before this new track first
            # appeared proves a new arrival, not an occluded continuation.
            if (visit.clear_since is not None and first-visit.clear_since >= self.clear_dwell_s
                    or visit.retired_at is not None and first >= visit.retired_at):
                return None
            track = str(details['track_id']) if details.get('track_id') is not None else None
            if (track is None or visit.separate_tracks or track in visit.track_ids
                    or visit.track_ids & self._visible_tracks
                    or _family(visit.label) != _family(details.get('label'))):
                return None
            approach = next((z for z in zones if _get(z,'name') == previous.get('zone_name')),None)
            barrier = next((z for z in zones if _get(z,'name') == details.get('zone_name')),None)
            window = _get(approach,'ai_confirmation_window_s',0)
            if (approach is None or barrier is None or not _number(window) or not 0 < window <= 1.5
                    or self.route_key(approach) != visit.route or self.route_key(barrier) != visit.route
                    or _get(approach,'alert_stage') != 'advance'
                    or (_get(barrier,'alert_stage') != 'barrier' and not same_advance)):
                return None
            if same_advance and (approach is not barrier or _get(approach,'ai_context') is not True):
                return None
            confidence = details.get('confidence')
            observations = details.get('observations')
            if (not _number(confidence) or not _get(barrier,'ai_confidence',0) <= confidence <= 1
                    or type(observations) is not int
                    or (observations < _get(barrier,'ai_confirm_observations',2)
                        and not _strong_stopline_proof(details, barrier, now, self.freshness_s))):
                return None
            old, new = visit.bbox, _box(details.get('bbox_normalized'))
            if old is None or new is None:
                return None
            aw,ah,bw,bh = old[2]-old[0],old[3]-old[1],new[2]-new[0],new[3]-new[1]
            if max(aw/bw,bw/aw,ah/bh,bh/ah) > 2.5:
                return None
            a,b = ((old[0]+old[2])/2,old[3]),((new[0]+new[2])/2,new[3])
            if math.dist(a,b) > min(.25,.55*(math.hypot(aw,ah)+math.hypot(bw,bh))):
                return None
            direction_region = approach
            direction = _get(approach,'arrival_direction')
            if direction is None and _get(approach,'ai_context') is True:
                # First sighting intentionally has no arrival-intent gate. Its
                # later identity handoff may use the independently configured
                # direction of the exact same physical advance region. Never
                # borrow a direction from another lane or a different polygon.
                polygon = _get(approach,'polygon')
                candidates = [candidate for candidate in zones
                    if candidate is not approach and polygon
                    and _get(approach,'ai_enabled') is True
                    and _get(candidate,'ai_enabled') is True
                    and self.route_key(candidate) == visit.route
                    and _get(candidate,'route_role') == _get(approach,'route_role')
                    and _get(candidate,'alert_stage') == 'advance'
                    and _get(candidate,'polygon') == polygon
                    and _get(candidate,'rect') == _get(approach,'rect')
                    and _get(candidate,'exclude_polygons',[]) == _get(approach,'exclude_polygons',[])
                    and _get(candidate,'arrival_direction') is not None
                    and (_get(candidate,'classes') is None
                         or (previous.get('label') in _get(candidate,'classes')
                             and details.get('label') in _get(candidate,'classes')))]
                if len(candidates) != 1:
                    return None
                direction_region = candidates[0]
                direction = _get(direction_region,'arrival_direction')
            if (not isinstance(direction,(list,tuple)) or len(direction)!=2
                    or not all(_number(v) for v in direction) or math.hypot(*direction)<.01):
                return None
            progress = ((b[0]-a[0])*direction[0]+(b[1]-a[1])*direction[1])/math.hypot(*direction)
            minimum = _get(direction_region,'arrival_min_progress',.012)
            if not _number(minimum) or minimum <= 0 or progress < minimum:
                return None
            for zone,point in ((approach,a),(barrier,b)):
                contains = _get(zone,'contains_point')
                if not callable(contains) or not contains(*point):
                    return None
            if same_advance:
                # A real follower enters behind the old car. Its later forward
                # motion must not disguise that separate birth as continuation.
                origin = details.get('first_ground')
                lateral_limit = max(.012,.25*min(math.hypot(aw,ah),math.hypot(bw,bh)))
                if (not isinstance(origin,(list,tuple)) or len(origin)!=2
                        or not all(_number(v) and 0 <= v <= 1 for v in origin)
                        or not _get(approach,'contains_point')(*origin)
                        or math.dist(a,origin) > min(.15,.55*(math.hypot(aw,ah)+math.hypot(bw,bh)))
                        or ((origin[0]-a[0])*direction[0]+(origin[1]-a[1])*direction[1])/math.hypot(*direction) < minimum
                        or any(abs((point[0]-a[0])*direction[1]-(point[1]-a[1])*direction[0])
                               /math.hypot(*direction) > lateral_limit for point in (origin,b))):
                    return None
            gate = _get(barrier,'confirmation_polygon')
            if gate and not point_in_polygon(*b,gate):
                return None
            visit.track_ids.add(track)
            if len(visit.track_ids)>16:
                visit.track_ids={track}
            visit.bbox,visit.last_seen,visit.last_confirmed = new,stamp,stamp
            visit.confirmed = True
            visit.clear_since = visit.retired_at = None
            self._suppressed += 1
            return VisitDecision(visit.id,True,'unanswered approach continuation',True)

    def emitted(self, visit_id, alarm_id, now):
        with self._lock:
            self._time(now)
            visit = self._visits.get(visit_id)
            if not visit or visit.retired_at is not None:
                return False
            visit.emitted = True
            if alarm_id is not None:
                visit.alarm_ids.add(str(alarm_id))
                self._current_alarm_id = str(alarm_id)
            return True

    def acknowledge(self, alarm_id, now):
        with self._lock:
            self._time(now)
            count = 0
            for visit in self._visits.values():
                if str(alarm_id) in visit.alarm_ids:
                    visit.acknowledged = True
                    count += 1
            if str(alarm_id) == self._current_alarm_id:
                self._current_alarm_id = None
            return count

    def active_evidence(self, visit_id, now):
        with self._lock:
            visit = self._visits.get(visit_id)
            if not visit:
                return {'fresh': False, 'present': None, 'confirmed': False}
            route = self._routes.get(visit.route, {})
            age = max(0., now - visit.last_seen)
            frame_fresh = (_number(self._last_observation) and 0 <= now-self._last_observation < self.freshness_s)
            fresh = frame_fresh and (age < self.freshness_s or route.get('ml_fresh', False))
            confirmed_fresh = (visit.confirmed and frame_fresh and visit.last_confirmed is not None
                               and 0 <= now-visit.last_confirmed < self.freshness_s)
            present = None
            if visit.retired_at is not None:
                present = False if fresh else None
            elif fresh:
                present = age < self.freshness_s or bool(route.get('signal'))
            return {'fresh': bool(fresh), 'present': present, 'confirmed': bool(confirmed_fresh),
                    'recognized': visit.confirmed,
                    'retired': visit.retired_at is not None,
                    'visit_id': visit.id, 'route': visit.route, 'acknowledged': visit.acknowledged,
                    'last_seen_age_s': round(age, 2), 'bbox_normalized': list(visit.bbox) if visit.bbox else None,
                    'label': visit.label or None}

    def evidence_for_alarm(self, alarm_id, now):
        with self._lock:
            visit = next((visit for visit in self._visits.values() if str(alarm_id) in visit.alarm_ids), None)
            return self.active_evidence(visit.id if visit else None, now)

    def status(self, now):
        with self._lock:
            active = [visit for visit in self._visits.values() if visit.retired_at is None]
            return {'policy': 'one warning per observed visit', 'clear_dwell_s': self.clear_dwell_s,
                    'active_visits': len(active), 'acknowledged_visits': sum(visit.acknowledged for visit in active),
                    'held_alarms': self._suppressed,
                    'routes': [{'visit_id': visit.id, 'route': visit.route, 'acknowledged': visit.acknowledged,
                                'confirmed': visit.confirmed, 'sightings': len(visit.track_ids)} for visit in active]}
