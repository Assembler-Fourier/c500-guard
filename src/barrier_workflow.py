"""Combine calibrated arm/empty-road observations into targeted service intents.

No hardware controls, audio writes, input hooks or extra worker live here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import copy
import json
import math
import threading
import os
import shutil
import time
from pathlib import Path

import numpy as np

from service_cycle import ServiceCycleMonitor
from visit_memory import VisitMemory


def fresh(stamp, now):
    return (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            and math.isfinite(stamp) and 0 <= now-stamp < 3.)


class BarrierWorkflow:
    def __init__(self, source, state_dir, seed_path, *, monitor=None, clock=time.monotonic, lines=None):
        self.clock = clock
        identity = source
        if lines is not None:
            # Arm templates are sampled along these lines: calibration taken on
            # other lines belongs to a different picture and must not load.
            from barrier_monitor import validate_lines
            lines = validate_lines(lines)
            identity = source + '|' + json.dumps(lines, sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        if monitor is None:
            from barrier_monitor import BarrierMonitor
            monitor = BarrierMonitor(storage_path=Path(state_dir)/f'barriers-{key}.npz',
                                     seed_path=seed_path, source_key=key, clock=clock, lines=lines)
        self.monitor = monitor
        self.cycles = ServiceCycleMonitor()
        self._lock = threading.RLock()
        self._route_masks = {}
        self._readings = {}
        self._last_resolution = None
        self._trusted = False
        self._last_stamp = None
        self._enabled = False
        self._observation_interrupted = threading.Event()
        self._operation = None
        self._snapshot_valid = False
        self._status_snapshot = (None, {'configured': False, 'barriers': []})
        # Self-learning: a confidently read arm whose match has drifted with
        # the light is saved as another lighting example (max 4 per position;
        # the closest is replaced after that). Never with anything in the lanes.
        self._learned_at = {}
        self.learned = []
        self.on_learned = None          # callback(entry) after each self-learned example
        self.learning_skips = {}        # (arm, position) -> last reason a capture was refused

    @staticmethod
    def _groups(zones):
        groups = {}
        for zone in zones:
            if zone.route_role in ('parking', 'charging', 'ignore'):
                continue
            route = VisitMemory.route_key(zone)
            if route:
                groups.setdefault(route, []).append(zone)
        return groups

    @staticmethod
    def _route_for(identity, routes):
        """Which approach does this barrier control?

        An exact name wins. Otherwise the one approach whose name ends with the
        barrier's name ("exit" controls "campus-exit"). An ambiguous or absent
        match returns None, so a barrier is never attached to the wrong road.
        """
        if not isinstance(identity, str) or not identity:
            return None
        if identity in routes:
            return identity
        lowered = identity.lower()
        matches = [route for route in routes
                   if isinstance(route, str) and route.lower().endswith(lowered)]
        return matches[0] if len(matches) == 1 else None

    def _mask(self, route, zones, shape):
        key = (route, shape, tuple((id(z), repr(z.polygon), repr(z.rect), repr(z.exclude_polygons)) for z in zones))
        cached = self._route_masks.get(route)
        if cached is None or cached[0] != key:
            mask = np.zeros(shape, dtype=bool)
            for zone in zones:
                mask |= zone.mask_for(shape[1], shape[0])
            self._route_masks[route] = (key, mask)
        return self._route_masks[route][1]

    def observe(self, frame, stamp, primary, zones, alignment_ok, alarm=None,
                pending_routes=(), enabled=True):
        # Manual calibration saves compressed references while holding this
        # lock. A slow disk must not hold the camera/alarm loop behind that
        # user action. Missing evidence is unknown until a new observation.
        if not self._lock.acquire(blocking=False):
            self._observation_interrupted.set()
            return []
        try:
            now = self.clock()
            if self._observation_interrupted.is_set():
                self._snapshot_valid = False
                self.cycles.reset()
                self._observation_interrupted.clear()
            self._enabled = bool(enabled)
            self._trusted = bool(alignment_ok and fresh(stamp, now))
            if not self._trusted:
                self._snapshot_valid = False
            self._last_stamp = stamp
            self._operation = 'observe'
            # One grey conversion per frame, shared by the arm reading and every lane's road check.
            shared = getattr(self.monitor, 'SHARED_GRAY', False) is True
            gray = self.monitor.prepare_gray(frame) if shared else None
            if shared:
                self.monitor.update(frame, stamp, aligned=self._trusted, gray=gray)
            else:
                self.monitor.update(frame, stamp, aligned=self._trusted)
            status = self.monitor.status()
            primary = primary if isinstance(primary, dict) else {}
            model_stamp = primary.get('observed_at')
            detections = primary.get('detections')
            ready = (self._trusted and primary.get('state') == 'ready'
                     and primary.get('available') is True and fresh(model_stamp, now)
                     and isinstance(detections, list)
                     and all(isinstance(d, dict) and isinstance(d.get('regions'), list) for d in detections))
            groups = self._groups(zones)
            # A barrier is named for the movement it controls ("exit"); a zone
            # names the approach it belongs to ("campus-exit"). Without this
            # mapping every arm reading is dropped on the floor and the barrier
            # features go silent while the monitor is reading them perfectly.
            arms = {}
            for row in status.get('barriers', []):
                identity = row.get('id')
                if identity not in ('exit', 'entrance'):
                    continue
                # Role and route name must agree. A footpath called "exit",
                # or a group mixing opposing lane roles, owns neither arm.
                compatible = {route: members for route, members in groups.items()
                              if all(z.route_role in ('', identity) for z in members)}
                route = self._route_for(identity, compatible)
                if route is not None:
                    arms[route] = row
            observations = {}
            self._readings = {}
            for route, members in groups.items():
                # Footpath occupancy is useful, but adjacent vehicle arms do
                # not prove that a pedestrian visit was served.
                arm = arms.get(route, {})
                names = {z.name for z in members}
                object_present = ready and any(names.intersection(d['regions']) for d in detections)
                image_present = self._trusted and any(z.motion_active or z.light_active for z in members)
                road = {'clear': False, 'reason': 'Current model or alignment unavailable'}
                occupancy, observed_at = 'unknown', None
                if image_present or object_present:
                    occupancy = 'occupied'
                    observed_at = stamp if image_present else model_stamp
                elif ready:
                    route_mask = self._mask(route, members, frame.shape[:2])
                    road = (self.monitor.road_clear(frame, route_mask, gray=gray, cache_key=route) if shared
                            else self.monitor.road_clear(frame, route_mask))
                    if road.get('clear') is True:
                        occupancy, observed_at = 'clear', min(stamp, model_stamp)
                observations[route] = {
                    'barrier': arm.get('state', 'unknown') if self._trusted else 'unknown',
                    'barrier_observed_at': arm.get('observed_at'),
                    # The first image supporting this position can precede its
                    # stable confirmation; do not credit that old transition to
                    # an alarm which only started while confirmation was pending.
                    'barrier_position_since': arm.get('position_since', arm.get('observed_at')),
                    'occupancy': occupancy, 'occupancy_observed_at': observed_at,
                    'model_ready': ready, 'new_arrival': route in pending_routes,
                }
                self._readings[route] = {'occupancy': occupancy, 'empty_road': road,
                                        'observed_at': observed_at,
                                        'barrier_id': arm.get('id'),
                                        'barrier_reason': arm.get('reason') or (
                                            'Physical barrier mapped to this lane' if arm else
                                            'No physical barrier controls this route')}
            # The model can miss a vehicle whose lights/motion already prove
            # activity. Never learn its image as a quiet lighting reference.
            learning_now = self.clock()
            if (ready and fresh(stamp, learning_now) and fresh(model_stamp, learning_now)
                    and not detections and not alarm and not pending_routes
                    and not any(z.motion_active or z.light_active for z in zones)):
                self._learn_arms(status, now)
            # Monitor work or a self-learned save can outlive the image. Use
            # the completion clock, not the time before those operations, for
            # every service intent and its model freshness checks.
            now = self.clock()
            if not fresh(stamp, now) or self._observation_interrupted.is_set():
                self._trusted = False
                self._snapshot_valid = False
                self.cycles.reset()
                return []
            if not fresh(model_stamp, now):
                for observation in observations.values():
                    observation['model_ready'] = False
            if not enabled:
                self.cycles.reset()
                self._publish_snapshot(self._build_status())
                return []
            intents = self.cycles.update(observations, alarm=alarm, now=now)
            self._publish_snapshot(self._build_status())
            return intents
        except Exception:
            self._trusted = self._snapshot_valid = False
            self._observation_interrupted.set()
            raise
        finally:
            self._operation = None
            self._lock.release()

    LEARN_EVERY_S = 1800.
    # A reading this sure, unchanged for a full minute with both strips
    # agreeing, is trusted enough to learn from - exactly the drifting case
    # (night light on 14 Nov: the entrance arm read down at 0.42).
    # Raised from 0.35 on 14 Nov: a raised arm can silence an alarm, so a
    # learned "up" example must come from a clear reading.
    LEARN_MIN_CONFIDENCE = .6
    LEARN_MAX_GLARE = .02           # share of near-white pixels on the arm strips
    LEARN_MAX_LIGHT_RANGE = 12.     # brightness swing on the strips over the last frames
    LEARN_BACKUPS = 10
    LEARN_STABLE_S = 60.
    LEARN_DRIFT_BELOW = .92

    def _learn_arms(self, status, now):
        observations = getattr(self.monitor, '_observations', {})
        for row in status.get('barriers', []):
            identity, state = row.get('id'), row.get('state')
            if state not in ('up', 'down') or not row.get('fresh') or not row.get('calibrated'):
                continue
            observation = observations.get(identity, {})
            best = observation.get('best')
            if (not isinstance(best, float) or best >= self.LEARN_DRIFT_BELOW
                    or row.get('confidence', 0.) < self.LEARN_MIN_CONFIDENCE):
                continue
            since = observation.get('candidate_at')
            if (not isinstance(since, (int, float)) or isinstance(since, bool)
                    or now - since < self.LEARN_STABLE_S):
                continue
            last = self._learned_at.get((identity, state))
            if last is not None and now - last < self.LEARN_EVERY_S:
                continue
            conditions = getattr(self.monitor, 'learning_conditions', None)
            light = conditions(identity) if conditions is not None else {}
            if (isinstance(light, dict) and (light.get('glare', 0.) > self.LEARN_MAX_GLARE
                                             or light.get('light_range', 0.) > self.LEARN_MAX_LIGHT_RANGE)):
                self.learning_skips[(identity, state)] = {
                    'reason': 'glare or changing light on the arm', 'at': time.strftime('%H:%M:%S'), **light}
                continue
            self._learned_at[(identity, state)] = now
            backup = self._backup_references()
            if backup is None:
                self.learning_skips[(identity, state)] = {
                    'reason': 'Could not preserve the current barrier references before learning',
                    'at': time.strftime('%H:%M:%S')}
                continue
            try:
                result = self.monitor.capture(identity, state)
            except (ValueError, RuntimeError, OSError):
                result = {'ok': False}
            if not result.get('ok'):
                self._drop_backup(backup)
                continue
            entry = {'arm': identity, 'position': state, 'match': round(best, 2),
                     'confidence': round(float(row.get('confidence', 0.)), 2), 'at': time.strftime('%H:%M:%S'),
                     'backup': backup.name if backup else None, **({'light': light} if light else {})}
            self.learned.append(entry)
            self.learned = self.learned[-20:]
            if self.on_learned is not None:
                try:
                    self.on_learned(dict(entry))
                except Exception:                                  # noqa: BLE001 - logging only
                    pass

    def _backup_references(self):
        """Copy the reference file before a self-learned change, so it can be undone."""
        path = getattr(self.monitor, 'storage_path', None)
        if not isinstance(path, (str, os.PathLike)) or not Path(path).is_file():
            return None
        path = Path(path)
        # Both arms can learn during the same frame. Each needs its own undo
        # point; second-resolution names silently overwrite the first backup.
        backup = path.with_name(f"{path.name}.before-learn-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns():020d}")
        try:
            shutil.copyfile(path, backup)
        except OSError:
            return None
        for old in sorted(path.parent.glob(path.name + '.before-learn-*'))[:-self.LEARN_BACKUPS]:
            old.unlink(missing_ok=True)
        return backup

    @staticmethod
    def _drop_backup(backup):
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass

    def undo_last_learning(self):
        """Put back the references as they were before the latest self-learned example."""
        with self._operation_lock('undo'):
            path = getattr(self.monitor, 'storage_path', None)
            if not isinstance(path, (str, os.PathLike)):
                return {'ok': False, 'reason': 'Barrier references are not stored'}
            path = Path(path)
            backups = sorted(path.parent.glob(path.name + '.before-learn-*'))
            if not backups:
                return {'ok': False, 'reason': 'No self-learned barrier example to undo'}
            latest = backups[-1]
            try:
                os.replace(latest, path)
            except OSError as exc:
                return {'ok': False, 'reason': f'Could not restore the earlier references: {exc}'}
            reload = getattr(self.monitor, 'reload', None)
            if reload is not None:
                reload()
            undone = next((item for item in reversed(self.learned) if item.get('backup') == latest.name), None)
            if undone is not None:
                self.learned.remove(undone)
            return {'ok': True, 'undone': undone, 'restored': latest.name,
                    'reason': 'The latest learned barrier example was removed'}

    def status(self):
        if not self._lock.acquire(blocking=False):
            return self._contended_status()
        try:
            if self._observation_interrupted.is_set():
                return self._unavailable_status()
            result = self._build_status()
            if self._operation is None:
                self._publish_snapshot(result)
            return result
        finally:
            self._lock.release()

    def _build_status(self):
        now = self.clock()
        result = self.monitor.status()
        current = self._trusted and fresh(self._last_stamp, now)
        cycle = self.cycles.status(now=now)
        result['service'] = cycle
        result['road_readings'] = {route: dict(value) for route, value in self._readings.items()}
        result['last_resolution'] = self._last_resolution
        result['reminder_active'] = bool(self._enabled and current and any(
            state.get('reminder_active') for state in cycle.get('routes', {}).values()))
        result['trusted'] = current
        return result

    def _publish_snapshot(self, result):
        # Publish source time and data together, after all service work. Readers
        # must never pair an old reading with the in-flight frame's timestamp.
        self._status_snapshot = (self._last_stamp, copy.deepcopy(result))
        self._snapshot_valid = result.get('trusted') is True

    def _contended_status(self):
        now = self.clock()
        stamp, saved = self._status_snapshot
        if (self._operation != 'observe' or not self._snapshot_valid or not self._trusted
                or self._observation_interrupted.is_set() or not fresh(stamp, now)):
            return self._unavailable_status()
        # Ordinary image analysis may use the last completed position for the
        # display. It cannot replay service evidence or authorize an ACK while
        # the newer observation is still being assembled.
        result = copy.deepcopy(saved)
        for row in result.get('barriers', []):
            if not fresh(row.get('observed_at'), now):
                self._unknown_row(row)
        result.update(trusted=False, reminder_active=False, road_readings={},
                      service={'routes': {}}, busy=True, cached=True)
        return result

    @staticmethod
    def _unknown_row(row):
        row.update(state='unknown', confidence=0., fresh=False, up_for_s=0.,
                   active_line=None, candidate_position=None, confirmation_s=0.,
                   reason='Waiting for fresh completed barrier observations')

    def _unavailable_status(self):
        """Keep setup geometry visible without lending cached state to an ACK."""
        result = copy.deepcopy(self._status_snapshot[1])
        for row in result.get('barriers', []):
            self._unknown_row(row)
        result.update(trusted=False, reminder_active=False, road_readings={},
                      service={'routes': {}}, busy=True)
        return result

    @contextmanager
    def _operation_lock(self, operation):
        with self._lock:
            previous = self._operation
            self._operation = operation
            self._snapshot_valid = False
            try:
                yield
            finally:
                self._operation = previous

    def resolved(self, intent):
        with self._lock:
            self._last_resolution = {**intent, 'at': time.strftime('%H:%M:%S')}

    def mute_reminders(self):
        """Quiet the left-up reminder until each reminding arm closes."""
        with self._lock:
            return self.cycles.mute_reminders(self.clock())

    def reset(self):
        with self._operation_lock('reset'):
            self.cycles.reset()
            self._trusted = False
            self._enabled = False
            self._readings.clear()

    def calibrate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Choose the exit or entrance barrier')
        if payload.get('action') == 'capture-road':
            # The empty road in this hour's light, so "the car has gone" can be
            # recognised outside the conditions of the first calibration.
            with self._operation_lock('calibration'):
                if not self._trusted or not fresh(self._last_stamp, self.clock()):
                    raise ValueError('A fresh, verified camera view is required')
                result = self.monitor.capture_road()
                if result.get('ok'):
                    self.cycles.reset()
                return result
        if payload.get('id') not in ('exit', 'entrance'):
            raise ValueError('Choose the exit or entrance barrier')
        action, ident = payload.get('action'), payload['id']
        with self._operation_lock('calibration'):
            if action == 'capture':
                if payload.get('position') not in ('up', 'down'):
                    raise ValueError('Choose UP or DOWN')
                if not self._trusted or not fresh(self._last_stamp, self.clock()):
                    raise ValueError('A fresh, verified camera view is required')
                result = self.monitor.capture(ident, payload['position'])
                if result.get('ok') is not True:
                    return result
            elif action == 'reset':
                result = self.monitor.reset(ident)
                result = result if isinstance(result, dict) else {'ok': True}
            else:
                raise ValueError('Unknown calibration action')
            self.cycles.reset()
            return result
