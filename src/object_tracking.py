"""Small object identities and region-visit confirmation.

This is an application association layer, not an implementation of SORT or
ByteTrack. Geometry-only identities can change after long occlusion; the camera
feed and model still determine which objects can be observed reliably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct

from region_geometry import point_in_polygon


TARGET_LABELS = frozenset({'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'})
VEHICLE_LABELS = frozenset({'car', 'bus', 'truck'})


def _family(label):
    return 'large_vehicle' if label in VEHICLE_LABELS else label


def _confidence_threshold(value, raw_confidence):
    # Decoder scores are float32. Match its boundary (not display rounding),
    # while adapters without raw scores keep their existing float comparison.
    return struct.unpack('f', struct.pack('f', value))[0] if raw_confidence is not None else value


def _iou(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(area_a + area_b - overlap, 1e-12)


def _association_score(a, b):
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if max(aw / bw, bw / aw, ah / bh, bh / ah) > 2.5:
        return None
    distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                          (a[1] + a[3] - b[1] - b[3]) / 2)
    gate = .55 * (math.hypot(aw, ah) + math.hypot(bw, bh))
    overlap = _iou(a, b)
    if overlap < .12 and distance > gate:
        return None
    return overlap + .2 * max(0., 1. - distance / max(gate, 1e-12))


def _ground(bbox):
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def _direction(zone):
    value = getattr(zone, 'arrival_direction', None)
    if value is None:
        return None
    dx, dy = value
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-9 else None


def _confirmation_window(zone):
    """Opt-in brief-glare tolerance; never exceed the motion association horizon."""
    value = getattr(zone, 'ai_confirmation_window_s', 0.)
    return min(1.5, max(0., float(value)))


def _strong_confirmation(visit, zone, label, confidence, ground, now, eligible, count,
                         raw_confidence=None):
    """Optional two-frame stop-line proof; retain the ordinary weaker path.

    Both observations must belong to this visit, pass its existing direction
    and narrow confirmation area, and be consecutive strong model results.
    The one-second bound uses source timestamps, never polling/receipt time.
    """
    threshold = getattr(zone, 'ai_fast_confirm_confidence', 0.)
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not .60 <= threshold <= 1.
            or count < 3 or getattr(zone, 'alert_stage', '') != 'barrier'
            or not getattr(zone, 'confirmation_polygon', None)
            or label not in {'car', 'motorcycle', 'bus', 'truck'}
            or not eligible
            or (raw_confidence if raw_confidence is not None else confidence)
                < _confidence_threshold(threshold, raw_confidence)):
        visit.strong_confirmation_samples.clear()
        return False
    samples = visit.strong_confirmation_samples
    if samples and now-samples[0]['observed_at'] > 1.:
        samples.clear()
    if not samples or now > samples[-1]['observed_at']:
        samples.append({'observed_at': now, 'confidence': confidence,
                        **({'raw_confidence': raw_confidence} if raw_confidence is not None else {}),
                        'ground_point': tuple(ground), 'direction_state': visit.direction_state})
    del samples[:-2]
    return len(samples) == 2 and 0 < samples[-1]['observed_at']-samples[0]['observed_at'] <= 1.


def _track_score(track, bbox, now):
    """Use short motion prediction without allowing implausible identity jumps."""
    elapsed = max(0., now - track.last_seen)
    previous, current = _ground(track.bbox), _ground(bbox)
    distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
    if distance > min(.4, .04 + .7 * elapsed):
        return None
    raw = _association_score(track.bbox, bbox)
    # A long absence is not a reliable velocity forecast or equally strong
    # association evidence. Age its geometric score beyond the existing motion
    # horizon: a dormant one-frame reflection must not outrank a plausible
    # recent vehicle merely because the later box overlaps it slightly more.
    # Keep the spatial gates and lifetime unchanged, so an occluded stationary
    # vehicle can still recover when it is the best remaining match.
    if elapsed > 1.5:
        return None if raw is None else raw * (1.5 / elapsed)
    if track.velocity is None:
        return raw
    vx, vy = track.velocity
    shift_x, shift_y = vx * min(elapsed, 1.), vy * min(elapsed, 1.)
    predicted = (track.bbox[0] + shift_x, track.bbox[1] + shift_y,
                 track.bbox[2] + shift_x, track.bbox[3] + shift_y)
    forecast = _association_score(predicted, bbox)
    if forecast is None:
        return raw
    if (elapsed <= .75 and track.label in VEHICLE_LABELS
            and len(track.trail) >= 3 and track.confidence >= .5):
        # With recent established motion, old-position glare must not win just
        # because it overlaps the previous box. Keep this independent of zones;
        # a car retains its identity as it crosses approach/barrier boundaries.
        # Uncertain, short or stale trajectories retain the legacy fallback.
        return forecast + .05
    return max(raw or 0., forecast + .05)


def _recent_motion_supports(track, bbox, now, previous_update):
    """Narrow evidence for a moving follower over a missing co-visible track."""
    elapsed = now - track.last_seen
    confidence = track.confidence if track.raw_confidence is None else track.raw_confidence
    if (previous_update is None or track.last_seen != previous_update
            or not .05 <= elapsed <= .75 or track.label not in VEHICLE_LABELS
            or len(track.trail) < 3 or confidence < .5 or track.velocity is None):
        return False
    vx, vy = track.velocity
    speed = math.hypot(vx, vy)
    old, new = _ground(track.bbox), _ground(bbox)
    dx, dy = new[0]-old[0], new[1]-old[1]
    distance = math.hypot(dx, dy)
    if speed < .02 or distance < .005 or vx*dx+vy*dy < .8*speed*distance:
        return False
    predicted = (track.bbox[0]+vx*elapsed, track.bbox[1]+vy*elapsed,
                 track.bbox[2]+vx*elapsed, track.bbox[3]+vy*elapsed)
    return _iou(predicted, bbox) >= .5


def zone_accepts(zone, label, point):
    """Use the object's ground point, class allowlist and exact region shape."""
    if not getattr(zone, 'ai_enabled', getattr(zone, 'motion', True)):
        return False
    classes = getattr(zone, 'classes', TARGET_LABELS)
    if classes is not None and label not in classes:
        return False
    contains = getattr(zone, 'contains_point', None)
    if contains is not None:
        return bool(contains(*point))
    x, y, width, height = zone.rect
    return x <= point[0] <= x + width and y <= point[1] <= y + height


EDGE_Y = .985     # a box reaching this far down is cut off by the bottom of the picture


def _bottom_edge_y(zone):
    """A calibrated detector-box edge, bounded to the last five percent of the image.

    Some clipped boxes stop a few pixels short of the picture boundary. Keep
    the legacy threshold unless this lane explicitly opts into a measured one.
    """
    value = getattr(zone, 'bottom_edge_y', EDGE_Y)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not .95 <= value <= 1.):
        return EDGE_Y
    return float(value)


def zone_accepts_box(zone, label, ground, bbox):
    """zone_accepts, plus (opt-in bottom_edge_entry) vehicles cut off by the bottom edge of the picture.

    Their real ground point is below the frame; they count when the lane reaches the edge right above them.
    """
    if zone_accepts(zone, label, ground):
        return True
    if not getattr(zone, 'bottom_edge_entry', False) or bbox[3] < _bottom_edge_y(zone):
        return False
    return any(zone_accepts(zone, label, (ground[0], y)) for y in (.97, .95, .93, .91, .89))


DEPARTING_TOWARD_SPEED = .02   # normalised picture widths/s along the arrival way that still count as arriving


def _directed_motion_start(samples, dx, dy, minimum):
    """Latest start of two separately observed advances, including slow returns."""
    if len(samples) < 3:
        return None
    end_at, end = samples[-1]
    for index in range(len(samples)-3, -1, -1):
        start_at, start = samples[index]
        elapsed = end_at - start_at
        total = (end[0]-start[0])*dx + (end[1]-start[1])*dy
        if not 0 < elapsed <= 1.5 or total < minimum or total/elapsed <= DEPARTING_TOWARD_SPEED:
            continue
        for middle_at, middle in samples[index+1:-1]:
            first = (middle[0]-start[0])*dx + (middle[1]-start[1])*dy
            second = (end[0]-middle[0])*dx + (end[1]-middle[1])*dy
            if start_at < middle_at < end_at and min(first, second) >= minimum/4:
                return start_at
    return None


def _departing_from(zone, track, visit):
    """A confirmed exit visit is leaving unless two fresh advances show a return.

    An isolated box-size/position correction is not a turn. New identities have
    no confirmed departure history, so their first-sighting policy is unchanged.
    """
    groups = getattr(zone, 'ignore_from_routes', None)
    if not groups or not track.groups_seen.intersection(groups):
        return False
    toward = getattr(zone, 'toward_direction', None) or getattr(zone, 'arrival_direction', None)
    samples = track.ground_samples
    if toward is not None and len(samples) >= 3:
        try:
            dx, dy = (float(v) for v in toward)
        except (TypeError, ValueError):
            return True
        length = math.hypot(dx, dy)
        if length > 1e-9:
            dx, dy = dx / length, dy / length
            minimum = max(.012, float(getattr(zone, 'arrival_min_progress', .012)))
            toward_at = _directed_motion_start(samples, dx, dy, minimum)
            away_at = _directed_motion_start(samples, -dx, -dy, minimum)
            if away_at is not None and (toward_at is None or away_at > toward_at):
                visit.departure_released = False
            elif toward_at is not None:
                visit.departure_released = True
    # A proven return stays eligible when it stops waiting. A new departure
    # or expiry of this zone visit resets that release; stationary readings do not.
    return not visit.departure_released


def _weak_floor(zone):
    value = getattr(zone, 'ai_confidence_moving', None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        return None
    return float(value)


@dataclass
class _Visit:
    event_id: str = ''
    hits: int = 0
    outside: int = 0
    active: bool = False
    emitted: bool = False
    first_seen: float = 0.
    origin: tuple = ()
    origin_bbox: tuple = ()
    max_motion: float = 0.
    direction_state: str = 'unrestricted'
    group: str = ''
    confirmation_hits: int = 0
    in_confirmation: bool = True
    allowed: bool = False
    peak_progress: float = 0.
    trough_progress: float = 0.
    reversal_peak: float | None = None
    reversal_hits: int = 0
    reversal_started: float | None = None
    confirmation_samples: list = field(default_factory=list)
    strong_confirmation_samples: list = field(default_factory=list)
    strong_confirmation_used: bool = False
    weak_hits: int = 0              # observations below the zone floor (moving-only evidence)
    departure_released: bool = False


@dataclass
class _Route:
    emitted: bool = False
    outside: int = 0


@dataclass
class _Track:
    id: int
    label: str
    bbox: tuple
    last_seen: float
    confidence: float = 0.
    raw_confidence: float | None = None
    visits: dict = field(default_factory=dict)
    routes: dict = field(default_factory=dict)
    velocity: tuple | None = None
    previous_ground: tuple | None = None
    trail: list = field(default_factory=list)    # recent ground points, for display
    first_seen: float | None = None               # where and when this object was first followed
    first_ground: tuple | None = None
    groups_seen: set = field(default_factory=set)  # route groups with confirmed, eligible visits
    ground_samples: list = field(default_factory=list)  # bounded source-timed motion proof
    co_visible_tracks: set = field(default_factory=set)  # independent IDs actually seen together


class ObjectVisitTracker:
    """Confirm configurable observations once per object/region or route visit.

    Three *visible* outside observations end a visit. Missing detections retain
    the visit for five seconds, preventing ordinary detector flicker from
    reporting the same stationary vehicle as another arrival. This lifetime is
    separate from the caller's camera/model failure generation handling.
    """
    def __init__(self, *, confirm_observations=2, exit_observations=3,
                 lost_after=5., max_tracks=100, max_detections=50):
        self.confirm_observations = max(2, int(confirm_observations))
        self.exit_observations = max(2, int(exit_observations))
        self.lost_after = max(0., float(lost_after))
        self.max_tracks = max(1, int(max_tracks))
        self.max_detections = max(1, int(max_detections))
        self._tracks = {}
        self._next_id = 1
        self._next_visit = 1
        self._last_update = None

    def reset(self):
        self._tracks.clear()
        self._last_update = None
        # IDs remain increasing so a reset cannot make a new arrival look like
        # a previous object in an on-screen diagnostic or an alert history.

    @property
    def track_count(self):
        return len(self._tracks)

    @property
    def active_visits(self):
        return sum(visit.active for track in self._tracks.values()
                   for visit in track.visits.values())

    def mark_emitted(self, event_id):
        """Only the alarm owner can consume an event after accepting it."""
        for track in self._tracks.values():
            for visit in track.visits.values():
                if visit.event_id == event_id and visit.active and visit.allowed:
                    if visit.emitted:
                        return False
                    visit.emitted = True
                    if visit.group:
                        route = track.routes.setdefault(visit.group, _Route())
                        route.emitted = True
                        for related in track.visits.values():
                            if related.group == visit.group:
                                related.emitted = True
                    return True
        return False

    def event_context(self, event_id, zones):
        """Bounded evidence from the observed frame, not the later preview JPEG.

        ``observed_at`` is the caller's monotonic frame timestamp in seconds;
        the alarm logger supplies its separate wall-clock timestamp.
        """
        for track in self._tracks.values():
            for index, visit in track.visits.items():
                if (visit.event_id == event_id and visit.active and visit.allowed and not visit.emitted
                        and index < len(zones)):
                    strong = visit.strong_confirmation_samples if visit.strong_confirmation_used else []
                    fast_proof = ({'confirmation_mode': 'strong_pair',
                                   'strong_confirmation': [{**sample, 'ground_point': list(sample['ground_point'])}
                                                           for sample in strong]} if len(strong) == 2 else {})
                    return {'zone_name': zones[index].name, 'route_group': visit.group,
                            'route_role': getattr(zones[index], 'route_role', ''),
                            'alert_stage': getattr(zones[index], 'alert_stage', ''),
                            'operator_hint': getattr(zones[index], 'operator_hint', ''),
                            'track_id': track.id, 'label': track.label,
                            'bbox_normalized': [round(value, 6) for value in track.bbox],
                            'confidence': track.confidence,
                            **({'raw_confidence': track.raw_confidence}
                               if track.raw_confidence is not None else {}),
                            'direction_state': visit.direction_state,
                            'observations': 2 if fast_proof else visit.confirmation_hits,
                            'confirmation_required': bool(getattr(zones[index], 'confirmation_polygon', None)),
                            'observed_at': track.last_seen,
                            **self._origin(track, zones), **fast_proof}
        return None

    @staticmethod
    def _origin(track, zones):
        """Where the object was first followed: the zone it started in (or 'outside the zones')."""
        if track.first_ground is None or track.first_seen is None:
            return {}
        came_from = 'outside the zones'
        for zone in zones:
            contains = getattr(zone, 'contains_point', None)
            try:
                if contains is not None and contains(*track.first_ground):
                    came_from = zone.name
                    break
            except (TypeError, ValueError):
                continue
        return {'first_ground': [round(v, 4) for v in track.first_ground], 'came_from': came_from,
                'first_observed_at': track.first_seen,
                'tracked_s': round(max(0., track.last_seen - track.first_seen), 1)}

    @staticmethod
    def _arrival_allowed(visit, zone, bbox, now):
        direction = _direction(zone)
        if direction is None:
            return True
        ground = _ground(bbox)
        edge_y = _bottom_edge_y(zone)
        if (getattr(zone, 'bottom_edge_entry', False) and bbox[3] >= edge_y and visit.origin_bbox
                and visit.origin_bbox[3] >= edge_y):
            # Pinned to the bottom edge: follow the top of the box (it rises as the vehicle drives on).
            ground = (ground[0], visit.origin[1] + (bbox[1] - visit.origin_bbox[1]))
        progress = ((ground[0] - visit.origin[0]) * direction[0]
                    + (ground[1] - visit.origin[1]) * direction[1])
        minimum = getattr(zone, 'arrival_min_progress', .012)
        visit.max_motion = max(visit.max_motion,
                               math.dist(ground, visit.origin),
                               max(abs(a - b) for a, b in zip(bbox, visit.origin_bbox)))
        # Re-evaluate a pending arrival when the observed car turns around.
        # Total displacement from the visit origin alone stays positive long
        # after a reversal; high/low watermarks require meaningful *recent*
        # travel in the new direction and tolerate detector position jitter.
        if visit.direction_state == 'toward':
            visit.peak_progress = max(visit.peak_progress, progress)
            if visit.peak_progress - progress >= minimum:
                visit.direction_state = 'away'
                visit.trough_progress = progress
                visit.reversal_peak, visit.reversal_hits = progress, 0
                visit.reversal_started = None
        elif visit.direction_state == 'away':
            # A clipped/occluded box can shift its ground point once while a
            # departing car parks. Require two separately observed advances,
            # not another stationary reading of that shifted box. The existing
            # minimum still bounds total recovery; quarter-minimum increments
            # reject subpixel jitter without delaying ordinary initial arrivals.
            if (visit.reversal_started is not None
                    and now - visit.reversal_started > 1.5):
                # Use the same horizon as consecutive motion association.
                # A later parked-box shift cannot complete an old reversal.
                visit.reversal_peak, visit.reversal_hits = progress, 0
                visit.reversal_started = None
            if progress < visit.trough_progress:
                visit.trough_progress = progress
                visit.reversal_peak, visit.reversal_hits = progress, 0
                visit.reversal_started = None
            elif visit.reversal_peak is None:
                visit.reversal_peak = progress
            elif visit.reversal_peak - progress >= minimum:
                visit.reversal_peak, visit.reversal_hits = progress, 0
                visit.reversal_started = None
            elif progress - visit.reversal_peak >= minimum / 4:
                visit.reversal_peak = progress
                if not visit.reversal_hits:
                    visit.reversal_started = now
                visit.reversal_hits += 1
            if progress - visit.trough_progress >= minimum and visit.reversal_hits >= 2:
                visit.direction_state = 'toward'
                visit.peak_progress = progress
        elif progress >= minimum:
            visit.direction_state = 'toward'
            visit.peak_progress = progress
        elif progress <= -minimum:
            visit.direction_state = 'away'
            visit.trough_progress = progress
            visit.reversal_peak, visit.reversal_hits = progress, 0
            visit.reversal_started = None
        if visit.direction_state == 'toward':
            return True
        # Waiting vehicles can first become visible after stopping at a barrier.
        # Only an object without observed departure motion gets this fallback.
        stationary_after = getattr(zone, 'stationary_confirm_s', 3.)
        # Advance road warnings require observed approach movement. A zero
        # timeout explicitly disables the waiting-at-barrier fallback there.
        if (stationary_after > 0 and visit.in_confirmation
                and visit.direction_state != 'away' and visit.max_motion < minimum
                and now - visit.first_seen >= stationary_after):
            visit.direction_state = 'stationary'
            return True
        return False

    def update(self, objects, zones, width, height, now):
        if width <= 0 or height <= 0 or not math.isfinite(now):
            raise ValueError('Tracking requires a finite time and positive image dimensions')
        previous_update = self._last_update
        if previous_update is not None and now < previous_update:
            self.reset()
            previous_update = None
        self._last_update = now
        self._tracks = {key: value for key, value in self._tracks.items()
                        if now - value.last_seen <= self.lost_after}

        detections = []
        # The detector already returns confidence-sorted, suppressed detections.
        # Bound all work, including validation, if a different adapter misbehaves.
        for obj in objects[:self.max_detections]:
            try:
                label = obj['label']
                if label not in TARGET_LABELS:
                    continue
                raw = tuple(float(value) for value in obj['bbox'])
                if len(raw) != 4 or not all(math.isfinite(value) for value in raw):
                    continue
                bbox = (max(0., min(1., raw[0] / width)),
                        max(0., min(1., raw[1] / height)),
                        max(0., min(1., raw[2] / width)),
                        max(0., min(1., raw[3] / height)))
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                confidence = float(obj.get('confidence', 0.))
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    continue
                raw_confidence = obj.get('raw_confidence')
                if 'raw_confidence' in obj and (isinstance(raw_confidence, bool)
                        or not isinstance(raw_confidence, (int, float))
                        or not math.isfinite(raw_confidence) or not 0 <= raw_confidence <= 1):
                    continue
                detections.append((obj, label, bbox, confidence, raw_confidence))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue

        candidates = []
        for track in self._tracks.values():
            for index, (_, label, bbox, _, _) in enumerate(detections):
                if _family(track.label) == _family(label):
                    score = _track_score(track, bbox, now)
                    if score is not None:
                        candidates.append((-score, track.id, index))
        matched_tracks, assignments = set(), {}
        ordered = sorted(candidates)
        best_by_track, by_detection = {}, {}
        for candidate in ordered:
            _, track_id, index = candidate
            best_by_track.setdefault(track_id, index)
            by_detection.setdefault(index, []).append(candidate)
        preferred = []
        for candidate in ordered:
            _, track_id, index = candidate
            track = self._tracks[track_id]
            if (best_by_track[track_id] != index
                    or not _recent_motion_supports(track, detections[index][2], now, previous_update)):
                continue
            stronger = [other for other in by_detection[index] if other < candidate]
            if stronger and all(other_id in track.co_visible_tracks
                                and self._tracks[other_id].last_seen < previous_update
                                for _, other_id, _ in stronger):
                # A stale lead's last box can have higher IoU than the moving
                # follower behind it. Reserve only that follower's best motion
                # match; all stronger rivals must be missing, proven separate.
                # Other boxes remain available if the old car reappears too.
                preferred.append(candidate)
        for _, track_id, index in preferred + ordered:
            if track_id not in matched_tracks and index not in assignments:
                assignments[index] = self._tracks[track_id]
                matched_tracks.add(track_id)
        for track in self._tracks.values():
            if track.id not in matched_tracks:
                for zone_index, visit in track.visits.items():
                    visit.hits = visit.outside = 0
                    visit.weak_hits = 0
                    visit.strong_confirmation_samples.clear()
                    window = _confirmation_window(zones[zone_index])
                    visit.confirmation_samples = [sample for sample in visit.confirmation_samples
                                                  if window and 0 <= now-sample[0] <= window]
                    visit.confirmation_hits = len(visit.confirmation_samples)
                    visit.reversal_peak, visit.reversal_hits = None, 0
                    visit.reversal_started = None
                    visit.allowed = False

        events, observed = [], []
        for index, (obj, label, bbox, confidence, raw_confidence) in enumerate(detections):
            track = assignments.get(index)
            if track is None:
                if len(self._tracks) >= self.max_tracks:
                    continue
                track = _Track(self._next_id, label, bbox, now)
                track.first_seen, track.first_ground = now, _ground(bbox)
                self._next_id += 1
                self._tracks[track.id] = track
            else:
                elapsed = now - track.last_seen
                previous = _ground(track.bbox)
                track.previous_ground = previous
                if .05 <= elapsed <= 1.5:
                    ground_now = _ground(bbox)
                    velocity = ((ground_now[0] - previous[0]) / elapsed,
                                (ground_now[1] - previous[1]) / elapsed)
                    track.velocity = velocity if track.velocity is None else tuple(
                        .5 * old + .5 * new for old, new in zip(track.velocity, velocity))
                elif elapsed > 1.5:
                    track.previous_ground = None
                    track.velocity = None
            track.label, track.bbox, track.last_seen = label, bbox, now
            track.confidence = confidence
            track.raw_confidence = raw_confidence
            score = confidence if raw_confidence is None else raw_confidence
            ground = _ground(bbox)
            track.trail.append((round(ground[0], 4), round(ground[1], 4)))
            del track.trail[:-10]
            if not track.ground_samples or now > track.ground_samples[-1][0]:
                track.ground_samples.append((now, ground))
                track.ground_samples = [sample for sample in track.ground_samples[-16:]
                                        if 0 <= now - sample[0] <= 1.5]
            current_zones, region_status = [], {}
            inside_zones = [zone_accepts_box(zone, label, ground, bbox) for zone in zones]
            inside_groups = {getattr(zone, 'route_group', '') for index, zone in enumerate(zones)
                             if inside_zones[index] and getattr(zone, 'route_group', '')}
            for group in inside_groups:
                track.routes.setdefault(group, _Route()).outside = 0
            for group, route in list(track.routes.items()):
                if group not in inside_groups:
                    route.outside += 1
                    if route.outside >= self.exit_observations:
                        del track.routes[group]
            proposed_groups = set()
            for zone_index, zone in enumerate(zones):
                inside = inside_zones[zone_index]
                visit = track.visits.get(zone_index)
                if inside:
                    current_zones.append(zone.name)
                    threshold = _confidence_threshold(getattr(zone, 'ai_confidence', 0.), raw_confidence)
                    floor = _weak_floor(zone)
                    weak = (score < threshold and floor is not None
                            and score >= _confidence_threshold(floor, raw_confidence)
                            and _direction(zone) is not None)
                    entry = getattr(zone, 'ai_moving_entry_polygon', None)
                    entry_missing = bool(weak and entry and (visit is None or not visit.active)
                                         and (track.first_ground is None
                                              or not point_in_polygon(*track.first_ground, entry)))
                    if entry_missing:
                        # A reflection can move up the road beneath a real car.
                        # Weak NEW visits require an observed lane-entry origin;
                        # a single stronger flash must not legitimise that ghost.
                        # Normal observations can still confirm a vehicle first
                        # visible mid-lane; confirmed identities remain intact.
                        weak = False
                    if score < threshold and not weak:
                        if visit is not None:
                            visit.hits = 0
                            visit.weak_hits = 0
                            visit.strong_confirmation_samples.clear()
                            visit.outside = 0
                            window = _confirmation_window(zone)
                            visit.confirmation_samples = [sample for sample in visit.confirmation_samples
                                                          if window and 0 <= now-sample[0] <= window]
                            visit.confirmation_hits = len(visit.confirmation_samples)
                            visit.reversal_peak, visit.reversal_hits = None, 0
                            visit.reversal_started = None
                            visit.allowed = False
                        region_status[zone.name] = ('weak arrival has no entry evidence' if entry_missing else
                                                    'below confidence threshold')
                        continue
                    if visit is None:
                        visit = track.visits[zone_index] = _Visit(
                            event_id=f'{track.id}:{zone_index}:{self._next_visit}',
                            first_seen=now, origin=track.previous_ground or ground,
                            origin_bbox=bbox,
                            direction_state='waiting' if _direction(zone) else 'unrestricted',
                            group=getattr(zone, 'route_group', ''))
                        self._next_visit += 1
                    visit.hits += 1
                    visit.outside = 0
                    if weak:
                        visit.weak_hits += 1
                    gate = getattr(zone, 'confirmation_polygon', None)
                    visit.in_confirmation = not gate or point_in_polygon(*ground, gate)
                    window = _confirmation_window(zone)
                    if window:
                        visit.confirmation_samples = [sample for sample in visit.confirmation_samples
                                                      if visit.in_confirmation and 0 <= now-sample[0] <= window]
                        if visit.in_confirmation and (not visit.confirmation_samples
                                                      or now > visit.confirmation_samples[-1][0]):
                            visit.confirmation_samples.append((now, ground))
                        del visit.confirmation_samples[:-8]
                        visit.confirmation_hits = len(visit.confirmation_samples)
                    else:
                        visit.confirmation_hits = visit.confirmation_hits + 1 if visit.in_confirmation else 0
                    count = getattr(zone, 'ai_confirm_observations', self.confirm_observations)
                    was_away = visit.direction_state == 'away'
                    direction_allowed = self._arrival_allowed(visit, zone, bbox, now)
                    if weak and visit.direction_state != 'toward':
                        # Weak evidence only counts for a vehicle seen moving in, never parked or reversing.
                        direction_allowed = False
                    if was_away or visit.direction_state == 'away':
                        # Departure observations cannot confirm the reversal.
                        # Retain the independent confirmation-polygon limit.
                        visit.confirmation_hits = min(visit.confirmation_hits, visit.reversal_hits)
                        if window:
                            visit.confirmation_samples = [(now, ground)] if visit.in_confirmation else []
                            visit.confirmation_hits = min(visit.confirmation_hits, len(visit.confirmation_samples))
                    # The broad approach still contributes trajectory and route
                    # identity, but cannot sound until the ground point is seen
                    # repeatedly in the barrier commitment area. Parking turns
                    # and cars waiting on the shared curve remain track-only.
                    allowed = visit.in_confirmation and direction_allowed
                    departing = _departing_from(zone, track, visit)
                    if departing:
                        allowed = False      # came from the exit lane: leaving, not arriving
                    if window and _direction(zone) and visit.direction_state != 'stationary':
                        # Both confirmation and travel must be fresh. An old
                        # approach followed by parked-box jitter cannot qualify
                        # merely by accumulating fresh stationary observations.
                        origin = visit.confirmation_samples[0][1] if visit.confirmation_samples else ground
                        dx, dy = _direction(zone)
                        recent_progress = (ground[0]-origin[0])*dx + (ground[1]-origin[1])*dy
                        allowed = allowed and recent_progress >= getattr(zone, 'arrival_min_progress', .012)
                    strong_confirmed = _strong_confirmation(
                        visit, zone, label, confidence, ground, now,
                        allowed and not was_away, count, raw_confidence)
                    # One extra observation only when the strong ones alone are not enough.
                    needed = count + (1 if visit.weak_hits and visit.hits - visit.weak_hits < count else 0)
                    visit.strong_confirmation_used = strong_confirmed and visit.confirmation_hits < needed
                    visit.allowed = allowed and (visit.confirmation_hits >= needed or strong_confirmed)
                    if visit.group and visit.allowed:
                        # A single weak/shifted box touching another lane must
                        # not permanently label a real arrival as departing.
                        track.groups_seen.add(visit.group)
                    if not visit.active and visit.allowed:
                        visit.active = True
                    if visit.group and track.routes[visit.group].emitted:
                        visit.emitted = True
                    region_status[zone.name] = ('already alerted' if visit.emitted else
                                               'departing (came from the exit lane)' if departing else
                                               'tracking approach' if not visit.in_confirmation else
                                               'moving away' if visit.direction_state == 'away' else
                                               'confirmed' if visit.active and visit.allowed else
                                               'checking direction' if not allowed else 'confirming')
                    if (visit.active and not visit.emitted and visit.allowed
                            and (not visit.group or visit.group not in proposed_groups)):
                        confirmed_count = needed if visit.confirmation_hits >= needed else 2
                        events.append((zone.severity,
                                       f'{zone.name}: AI confirmed {label} arrival '
                                       f'in {confirmed_count} observations', visit.event_id))
                        if visit.group:
                            proposed_groups.add(visit.group)
                elif visit is not None:
                    visit.hits = 0
                    visit.weak_hits = 0
                    visit.confirmation_hits = 0
                    visit.confirmation_samples.clear()
                    visit.strong_confirmation_samples.clear()
                    visit.reversal_peak, visit.reversal_hits = None, 0
                    visit.reversal_started = None
                    visit.allowed = False
                    visit.outside += 1
                    if visit.outside >= self.exit_observations:
                        del track.visits[zone_index]
            observed.append({**obj, 'track_id': track.id,
                             'box_n': [round(value, 5) for value in bbox],
                             'trail': [list(point) for point in track.trail],
                             'tracked_s': round(max(0., now - (track.first_seen or now)), 1),
                             'view': obj.get('view', 'focused'),
                             'ground_point': [round(ground[0], 5), round(ground[1], 5)],
                             'regions': current_zones,
                             'region_status': region_status,
                             'direction': {zones[key].name: visit.direction_state
                                           for key, visit in track.visits.items()
                                           if key < len(zones)},
                             'confirmed_regions': [zones[key].name for key, visit in track.visits.items()
                                                   if visit.active and visit.allowed and key < len(zones)]})
        visible_ids = {obj['track_id'] for obj in observed}
        for track_id in visible_ids:
            track = self._tracks[track_id]
            track.co_visible_tracks.intersection_update(self._tracks)
            track.co_visible_tracks.update(visible_ids - {track_id})
        return events, observed
