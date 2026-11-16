"""Bounded fusion of forward road motion and fresh model candidates.

This owns no threads, device, storage or audio. The watcher owns delivery and
VisitMemory owns acknowledgement; anonymous evidence never becomes an AI ID.
"""
from collections import deque
import math

import numpy as np

from approach_motion import ApproachMotion, corroborate_motion


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class ApproachWarnings:
    def __init__(self):
        self.detectors = {}
        self.candidates = deque(maxlen=300)
        self.last_source = None
        self.generation = None

    def reset(self):
        self.detectors.clear()
        self.candidates.clear()
        self.last_source = self.generation = None

    def _collect(self, primary, at):
        # Raw results are diagnostic candidates only. Expired/failed inference
        # cannot resurrect one from the last healthy model generation.
        if not (isinstance(primary, dict) and primary.get('available') is True
                and primary.get('state') == 'ready' and _finite(primary.get('age_s'))
                and 0 <= primary['age_s'] < 3):
            self.candidates.clear()
            self.last_source = None
            return
        rows = primary.get('view_diagnostics', [])
        if not isinstance(rows, list):
            return
        for view in rows[:4]:
            if not isinstance(view, dict):
                continue
            stamp = view.get('observed_at')
            if not _finite(stamp) or not 0 <= at-stamp <= 1:
                continue
            generation = (view.get('epoch'), view.get('generation'))
            if generation != self.generation:
                self.reset()
                self.generation = generation
            # Model views of one frame are collected together below.
            if self.last_source is not None and stamp <= self.last_source:
                continue
            objects = view.get('detections', [])
            if not isinstance(objects, list):
                continue
            for obj in objects[:50]:
                if not isinstance(obj, dict):
                    continue
                self.candidates.append({'label': obj.get('label'),
                    'confidence': obj.get('confidence'), 'bbox': obj.get('bbox_frame'),
                    'observed_at': stamp, 'view': view.get('kind', '')})
        stamps = [v.get('observed_at') for v in rows[:4] if isinstance(v, dict)
                  and _finite(v.get('observed_at')) and 0 <= at-v['observed_at'] <= 1]
        if stamps:
            self.last_source = max(stamps + ([self.last_source] if self.last_source is not None else []))
        self.candidates = deque((c for c in self.candidates if 0 <= at-c['observed_at'] <= 1), maxlen=300)

    def update(self, frame, zones, primary, at, *, enabled):
        for zone in zones:
            zone.approach_motion_active = False
        if not enabled or not _finite(at):
            self.reset()
            return []
        self._collect(primary, at)
        height, width = frame.shape[:2]
        warnings = []
        enabled_names = set()
        for zone in zones:
            if not getattr(zone, 'approach_motion', False):
                continue
            enabled_names.add(zone.name)
            geometry = (repr(zone.polygon), repr(zone.arrival_direction), repr(zone.exclude_polygons))
            owned = self.detectors.get(zone.name)
            if owned is None or owned[0] != geometry:
                detector = ApproachMotion(zone.polygon, zone.arrival_direction,
                                          exclude_polygons=zone.exclude_polygons)
                self.detectors[zone.name] = (geometry, detector)
            else:
                detector = owned[1]
            motion = detector.update(np.asarray(frame, dtype=np.uint8), at)
            allowed = zone.classes if zone.classes is not None else ('car', 'truck', 'bus', 'motorcycle')
            zone_candidates = [c for c in self.candidates if c.get('label') in allowed]
            # A weak, current object still occupying this route prevents a
            # stopped vehicle's no-flow interval from proving the road empty.
            for candidate in zone_candidates:
                try:
                    box = candidate['bbox']
                    score = candidate['confidence']
                    if (candidate['label'] in ('car', 'truck', 'bus', 'motorcycle')
                            and _finite(score) and .30 <= score <= 1
                            and isinstance(box, (list, tuple)) and len(box) == 4
                            and all(_finite(v) for v in box) and box[2] > box[0] and box[3] > box[1]
                            and zone.contains_point((box[0]+box[2])/2/width, box[3]/height)):
                        zone.approach_motion_active = True
                        break
                except (TypeError, ValueError):
                    continue
            proof = corroborate_motion(motion, zone_candidates, at,
                zone.polygon, width, height, exclude_polygons=zone.exclude_polygons,
                direction=zone.arrival_direction)
            zone.approach_metrics = {**motion, 'corroborated': proof is not None}
            if proof is not None:
                zone.approach_motion_active = True
                warnings.append((zone, detector, motion, proof))
        self.detectors = {name: value for name, value in self.detectors.items() if name in enabled_names}
        return warnings
