"""Route-scoped service completion intents; no audio, input hooks or gate control.

The caller must label a route ``clear`` only when BOTH a fresh, ready primary
model sees no vehicle and an independently calibrated empty-road view matches.
An adaptive background, missing detection, desk gesture, or secondary-model
absence alone is not clear-road evidence. All observation stamps are exact
monotonic source receipt times, never polling/completion times.
"""
from dataclasses import dataclass
import math
import threading
import time


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class _Event:
    alarm_id: str
    started_at: float
    occupied_seen: bool = False
    occupied_at: float | None = None
    clear_since: float | None = None
    clear_count: int = 0
    last_stamp: float | None = None
    cycle: str = 'waiting_down'
    up_seen: bool = False
    new_arrival: bool = False
    resolution_offered: bool = False
    # Last confirmed arm position observed during this specific alarm.
    # Kept through 'unknown' readings: a moving arm reads unknown for a second
    # or two, and an occluding car can hide it, but the guard did move it.
    anchor: str | None = None
    anchor_at: float | None = None


@dataclass
class _Route:
    barrier: str = 'unknown'
    barrier_stamp: float | None = None
    up_since: float | None = None
    last_reminder: float | None = None
    occupancy: str = 'unknown'
    occupancy_stamp: float | None = None
    event: _Event | None = None
    # The arm was raised (down -> up) while an alarm on this lane was current:
    # the guard is serving a car. Only such an arm earns a "left up" reminder;
    # an arm already up beforehand is being held open on purpose.
    served: bool = False
    muted: bool = False         # the guard dismissed the reminder until the arm closes
    last_known: str | None = None       # last confirmed up/down, never reset by 'unknown'
    last_known_at: float | None = None


class ServiceCycleMonitor:
    """Small, thread-safe state machine with explicit source freshness.

    ``update(observations, alarm, now)`` accepts a mapping from route name to::

        {barrier: 'up'|'down'|'unknown', barrier_observed_at: float,
         occupancy: 'occupied'|'clear'|'unknown', occupancy_observed_at: float,
         model_ready: bool, new_arrival: bool}

    ``alarm`` is the CURRENT event ``{'id': str, 'route': str,
    'started_at': float}``, or None. Missing/invalid start times conservatively
    bind at the current update, never borrowing a pre-alarm arm position.
    A new_arrival pulse means a DISTINCT arrival after this event, not repeated
    confirmation of its original vehicle. It blocks automatic resolution of
    that old event until the current alarm identity changes or reset is called.

    Returned dictionaries are intents only. The caller must compare alarm ID,
    route, freshness and any pending new arrival again under its own state lock
    before resolving. Resolving must not invoke a global acknowledgement/quiet
    API. Replacing the current alarm discards its old completion evidence.
    A qualifying fresh reading re-offers the same targeted resolution while
    that alarm remains current, so a rejected/raced application can retry.
    Clear-road completion also requires a fresh known barrier: down, or up
    witnessed during this event. An unknown barrier cannot complete service.
    """

    ARM_CHANGE_WINDOW_S = 30.

    def __init__(self, *, clear_dwell_s=2., freshness_s=3., open_after_s=15.,
                 reminder_interval_s=15., max_routes=32):
        for name, value, minimum in (
                ('clear_dwell_s', clear_dwell_s, 2.),
                ('freshness_s', freshness_s, .1),
                ('open_after_s', open_after_s, 15.),
                ('reminder_interval_s', reminder_interval_s, 15.)):
            if not _number(value) or value < minimum:
                raise ValueError(f'{name} must be finite and >= {minimum}')
        if isinstance(max_routes, bool) or not isinstance(max_routes, int) or not 1 <= max_routes <= 128:
            raise ValueError('max_routes must be an integer in 1..128')
        self.clear_dwell_s = float(clear_dwell_s)
        self.freshness_s = float(freshness_s)
        self.open_after_s = float(open_after_s)
        self.reminder_interval_s = float(reminder_interval_s)
        self.max_routes = max_routes
        self._routes = {}
        self._lock = threading.RLock()
        self._last_now = None

    def reset(self):
        """Discard all evidence, including open reminders, on OFF/recalibration."""
        with self._lock:
            self._routes.clear()
            self._last_now = None

    def _fresh(self, stamp, now):
        return _number(stamp) and 0 <= now-stamp < self.freshness_s

    @staticmethod
    def _alarm(alarm):
        if not isinstance(alarm, dict):
            return None, None
        ident, route = alarm.get('id'), alarm.get('route')
        if not isinstance(ident, str) or not ident or len(ident) > 200:
            return None, None
        if not isinstance(route, str) or not route or len(route) > 160:
            return None, None
        return ident, route

    @staticmethod
    def _reset_clear(event):
        if event:
            event.clear_since = None
            event.clear_count = 0

    def _barrier(self, state, obs, now):
        barrier, stamp = obs.get('barrier'), obs.get('barrier_observed_at')
        event = state.event
        if event and event.anchor_at is not None and now-event.anchor_at > self.ARM_CHANGE_WINDOW_S:
            event.anchor = event.anchor_at = None
        if barrier not in ('up', 'down') or not self._fresh(stamp, now):
            state.barrier = 'unknown'
            state.up_since = state.last_reminder = None
            if state.event:
                state.event.cycle = 'waiting_down'
                state.event.up_seen = False
            return []
        previous = state.barrier_stamp
        if previous is not None and stamp <= previous:
            # Repeated results may keep a display alive until their source
            # expires, but cannot advance a timer or a barrier transition.
            return []
        gap = previous is None or stamp-previous >= self.freshness_s
        if gap:
            state.up_since = state.last_reminder = None
            if state.event:
                state.event.cycle = 'waiting_down'
                state.event.up_seen = False
        old = state.barrier
        state.barrier, state.barrier_stamp = barrier, stamp
        actions = []
        event = state.event
        if event and stamp >= event.started_at:
            position_since = obs.get('barrier_position_since', stamp)
            position_began_during_event = (_number(position_since)
                and event.started_at <= position_since <= stamp)
            if (event.anchor in ('up', 'down') and barrier != event.anchor
                    and _number(event.anchor_at) and event.anchor_at >= event.started_at
                    and position_began_during_event and event.anchor_at <= position_since
                    and not event.new_arrival):
                # A confirmed change of this lane's arm during its alarm:
                # somebody is operating the barrier for this visit.
                if barrier == 'up':
                    state.served = True
                actions.append({'type': 'raised' if barrier == 'up' else 'lowered',
                                'alarm_id': event.alarm_id, 'observed_at': stamp,
                                'origin_observed_at': event.anchor_at,
                                'alarm_started_at': event.started_at,
                                'position_since': position_since})
            event.anchor = barrier
            event.anchor_at = stamp
        state.last_known, state.last_known_at = barrier, stamp
        if barrier == 'down':
            state.up_since = state.last_reminder = None
            state.served = state.muted = False
        elif old != 'up' or state.up_since is None:
            state.up_since = stamp
        event = state.event
        if event and stamp >= event.started_at:
            if barrier == 'up':
                event.up_seen = True
            if barrier == 'down' and event.cycle == 'waiting_down':
                event.cycle = 'waiting_up'
            elif barrier == 'up' and event.cycle == 'waiting_up':
                event.cycle = 'waiting_close'
            elif barrier == 'down' and event.cycle == 'waiting_close':
                event.cycle = 'complete'
        if (state.served and not state.muted
                and state.up_since is not None and stamp-state.up_since > self.open_after_s
                and (state.last_reminder is None
                     or stamp-state.last_reminder >= self.reminder_interval_s)):
            state.last_reminder = stamp
            actions.append({'type': 'open_reminder', 'observed_at': stamp,
                            'open_s': round(stamp-state.up_since, 3)})
        return actions

    def _occupancy(self, state, obs, now, newly_bound):
        event = state.event
        occupancy, stamp = obs.get('occupancy'), obs.get('occupancy_observed_at')
        if obs.get('new_arrival') is True and event and not newly_bound:
            event.new_arrival = True
            self._reset_clear(event)
        valid = (occupancy in ('occupied', 'clear') and self._fresh(stamp, now)
                 and obs.get('model_ready') is True)
        if not valid:
            state.occupancy = 'unknown'
            self._reset_clear(event)
            return None
        previous = state.occupancy_stamp
        if previous is not None and stamp <= previous:
            return None
        state.occupancy, state.occupancy_stamp = occupancy, stamp
        if not event:
            return None
        if stamp < event.started_at:
            # A delayed model result may still be fresh but belongs to the
            # previous car. Display it without lending it to this alarm.
            self._reset_clear(event)
            return None
        if event.last_stamp is None or stamp-event.last_stamp >= self.freshness_s:
            self._reset_clear(event)
        event.last_stamp = stamp
        if occupancy == 'occupied':
            event.occupied_seen = True
            event.occupied_at = stamp
            self._reset_clear(event)
            return None
        if not event.occupied_seen or event.new_arrival:
            self._reset_clear(event)
            return None
        barrier_verified = (self._fresh(state.barrier_stamp, now)
                            and state.barrier_stamp >= event.started_at
                            and (state.barrier == 'down'
                                 or (state.barrier == 'up' and event.up_seen)))
        if not barrier_verified:
            self._reset_clear(event)
            return None
        if event.clear_since is None:
            event.clear_since = stamp
        event.clear_count += 1
        if event.clear_count >= 2 and stamp-event.clear_since >= self.clear_dwell_s:
            event.resolution_offered = True
            return {'type': 'resolve', 'alarm_id': event.alarm_id, 'observed_at': stamp,
                    'alarm_started_at': event.started_at,
                    'occupied_at': event.occupied_at,
                    'clear_since': event.clear_since,
                    'barrier_observed_at': state.barrier_stamp,
                    'barrier_state': state.barrier,
                    'reason': ('verified_service_cycle' if event.cycle == 'complete'
                               else 'verified_road_clear'),
                    'clear_s': round(stamp-event.clear_since, 3),
                    'clear_observations': event.clear_count}
        return None

    def update(self, observations, alarm=None, now=None):
        now = time.monotonic() if now is None else now
        if not _number(now):
            raise ValueError('now must be a finite monotonic timestamp')
        observations = observations if isinstance(observations, dict) else {}
        ident, alarm_route = self._alarm(alarm)
        with self._lock:
            if self._last_now is not None and now < self._last_now:
                # A delayed caller must not rewind a new event or its evidence.
                return []
            self._last_now = now
            for route in observations:
                if (isinstance(route, str) and route and len(route) <= 160
                        and route not in self._routes and len(self._routes) < self.max_routes):
                    self._routes[route] = _Route()
            actions = []
            for route, state in self._routes.items():
                target = ident if route == alarm_route else None
                newly_bound = bool(target and (not state.event or state.event.alarm_id != target))
                if target is None:
                    state.event = None
                elif newly_bound:
                    started = alarm.get('started_at')
                    started = started if _number(started) and started <= now else now
                    state.event = _Event(target, started)
                obs = observations.get(route, {})
                obs = obs if isinstance(obs, dict) else {}
                if state.event and obs.get('new_arrival') is True and not newly_bound:
                    state.event.new_arrival = True
                for action in (*self._barrier(state, obs, now),
                               self._occupancy(state, obs, now, newly_bound)):
                    if action:
                        actions.append({**action, 'route': route})
            return actions

    def mute_reminders(self, now=None):
        """The guard dismissed the left-up reminder: quiet until each arm closes."""
        now = time.monotonic() if now is None else now
        with self._lock:
            muted = []
            for route, state in self._routes.items():
                if state.barrier == 'up' and state.served and not state.muted:
                    state.muted = True
                    muted.append(route)
            return muted

    def status(self, now=None):
        now = time.monotonic() if now is None else now
        if not _number(now):
            raise ValueError('now must be a finite monotonic timestamp')
        with self._lock:
            result = {}
            for route, state in self._routes.items():
                fresh_barrier = self._fresh(state.barrier_stamp, now)
                fresh_occupancy = self._fresh(state.occupancy_stamp, now)
                event = state.event
                result[route] = {
                    'barrier': state.barrier if fresh_barrier else 'unknown',
                    'barrier_observed_at': state.barrier_stamp,
                    'occupancy': state.occupancy if fresh_occupancy else 'unknown',
                    'occupancy_observed_at': state.occupancy_stamp,
                    'open_s': (round(max(0., state.barrier_stamp-state.up_since), 3)
                               if fresh_barrier and state.up_since is not None else 0.),
                    'reminder_active': bool(fresh_barrier and state.barrier == 'up'
                                           and state.served and not state.muted
                                           and state.up_since is not None
                                           and state.barrier_stamp-state.up_since > self.open_after_s),
                    'raised_for_visit': bool(state.served),
                    'reminder_muted': bool(state.muted),
                    'alarm_id': event.alarm_id if event else None,
                    'cycle': event.cycle if event else None,
                    'up_seen_for_event': bool(event and event.up_seen),
                    'clear_s': (round(max(0., event.last_stamp-event.clear_since), 3)
                                if event and event.clear_since is not None and fresh_occupancy else 0.),
                    'blocked_by_new_arrival': bool(event and event.new_arrival),
                    'resolution_offered': bool(event and event.resolution_offered),
                }
            return {'routes': result, 'clear_dwell_s': self.clear_dwell_s,
                    'freshness_s': self.freshness_s, 'advisory_only': True}
