"""Small in-memory timing snapshots that never acquire a watcher's state lock.

Stages describe one sequential pipeline. Entering another stage closes the
previous span; a late leave for an old token cannot clear the current stage.
Frame stamps use the same monotonic clock as the reader, never wall-clock time.
No threads, image processing, logging, callbacks or file access happen here.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import math
import threading
import time


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    # A monotonic process clock cannot reasonably span thirty thousand years.
    # This also keeps every derived millisecond value finite and JSON-safe.
    return value if math.isfinite(value) and 0 <= value <= 1e12 else None


class PipelineHealth:
    """Bounded, independent telemetry for a main loop and its frame reader.

    ``with health.stage('alignment'): ...`` is the normal timing interface.
    Manual callers can use ``token = enter(name)`` followed by ``leave(token)``.
    Stages are sequential, not nested: a new entry completes the old span.
    ``measure(name)`` times independent/nested work without changing that
    sequence. ``frame_span(source_at)`` additionally retains source-correlated
    whole-frame durations, including work between named stages.

    ``frame_received(stamp)`` belongs at the actual reader receive point.
    ``frame_processed(stamp)`` belongs after successful detection processing,
    with the original receive stamp, not the completion time. Arrival age and
    processed-image age can then distinguish an upstream stall from a backlog.
    """

    def __init__(self, *, clock=time.monotonic, slow_after_s=2.0,
                 max_slow=20, max_stages=24):
        threshold = _number(slow_after_s)
        if threshold is None or threshold <= 0:
            raise ValueError('slow_after_s must be finite and positive')
        for name, value in (('max_slow', max_slow), ('max_stages', max_stages)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 128:
                raise ValueError(f'{name} must be an integer in [1, 128]')
        if not callable(clock):
            raise TypeError('clock must be callable')
        self._clock = clock
        self._lock = threading.Lock()
        self._slow_after_s = threshold
        self._max_stages = max_stages
        self._slow = deque(maxlen=max_slow)
        self._recent_frames = deque(maxlen=max_slow)
        self._measurements = {}
        self._measurement_stats = {}
        self._measurements_dropped = 0
        self._stages = {}
        self._active = None
        self._sequence = 0
        self._last_now = None
        self._started_at = None
        self._received_at = None
        self._processed_at = None
        self._completed_at = None
        self._last_processing_delay_ms = None
        self._frames_received = 0
        self._frames_processed = 0
        self._stale_dropped = 0
        self._invalid_timestamps = 0
        self._out_of_order_frames = 0
        self._clock_errors = 0

    def _read_clock(self):
        # Even a broken injected clock must not turn diagnostics into a fatal
        # exception in detection. No injected function runs under this lock.
        try:
            return _number(self._clock())
        except Exception:
            return None

    def _now_locked(self, candidate):
        if candidate is None:
            self._clock_errors += 1
            candidate = self._last_now if self._last_now is not None else 0.0
        # Concurrent readers can sample time just before another thread; never
        # let lock acquisition order make ages negative or move a span backward.
        now = max(candidate, self._last_now) if self._last_now is not None else candidate
        self._last_now = now
        if self._started_at is None:
            self._started_at = now
        return now

    def _finish_locked(self, now):
        if self._active is None:
            return
        token, name, started = self._active
        duration = max(0.0, now - started)
        milliseconds = round(duration * 1000.0, 3)
        bucket = self._stages[name]
        bucket['last_ms'] = milliseconds
        bucket['max_ms'] = max(bucket['max_ms'], milliseconds)
        bucket['count'] += 1
        if duration >= self._slow_after_s:
            self._slow.append({'stage': name, 'duration_ms': milliseconds,
                               'started_at': started, 'finished_at': now})
        self._active = None

    def enter(self, stage):
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError('stage must be a nonempty string')
        name = stage.strip()[:64]
        sampled = self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            self._finish_locked(now)
            if name not in self._stages:
                if len(self._stages) >= self._max_stages:
                    # Stage names normally come from a fixed list in watch.py.
                    # Unexpected names replace the oldest inactive bucket rather
                    # than letting a long session grow this dictionary forever.
                    self._stages.pop(next(iter(self._stages)))
                self._stages[name] = {'last_ms': None, 'max_ms': 0.0, 'count': 0}
            self._sequence += 1
            self._active = (self._sequence, name, now)
            return self._sequence

    def leave(self, token):
        sampled = self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            if (isinstance(token, bool) or not isinstance(token, int)
                    or self._active is None or token != self._active[0]):
                return False
            self._finish_locked(now)
            return True

    @contextmanager
    def stage(self, name):
        token = self.enter(name)
        try:
            yield token
        finally:
            self.leave(token)

    def begin_measurement(self, name, *, source_at=None):
        """Start an independent span; nesting cannot close the active stage."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError('measurement name must be a nonempty string')
        name = name.strip()[:64]
        sampled = self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            source = _number(source_at)
            if source_at is not None and (source is None or source > now):
                self._invalid_timestamps += 1
                source = None
            if len(self._measurements) >= self._max_stages:
                self._measurements_dropped += 1
                return None
            self._sequence += 1
            self._measurements[self._sequence] = (name, now, source)
            return self._sequence

    def end_measurement(self, token):
        sampled = self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            if isinstance(token, bool) or not isinstance(token, int):
                return False
            span = self._measurements.pop(token, None)
            if span is None:
                return False
            name, started, source = span
            duration = max(0., now-started)
            milliseconds = round(duration*1000., 3)
            if name not in self._measurement_stats:
                if len(self._measurement_stats) >= self._max_stages:
                    self._measurement_stats.pop(next(iter(self._measurement_stats)))
                self._measurement_stats[name] = {'last_ms': None, 'max_ms': 0., 'count': 0}
            bucket = self._measurement_stats[name]
            bucket.update(last_ms=milliseconds, max_ms=max(bucket['max_ms'], milliseconds),
                          count=bucket['count']+1)
            record = {'stage': name, 'span_type': 'measurement', 'duration_ms': milliseconds,
                      'started_at': started, 'finished_at': now, 'source_at': source,
                      'source_age_ms': round((now-source)*1000., 3) if source is not None else None}
            if duration >= self._slow_after_s:
                self._slow.append(record)
            if name == 'whole_frame':
                self._recent_frames.append(record)
            return True

    @contextmanager
    def measure(self, name, *, source_at=None):
        token = self.begin_measurement(name, source_at=source_at)
        try:
            yield token
        finally:
            self.end_measurement(token)

    def frame_span(self, source_at):
        return self.measure('whole_frame', source_at=source_at)

    def _frame(self, stamp, *, processed):
        value, sampled = _number(stamp), self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            if value is None or value > now:
                self._invalid_timestamps += 1
                return False
            previous = self._processed_at if processed else self._received_at
            if previous is not None and value < previous:
                self._out_of_order_frames += 1
                return False
            if processed:
                self._processed_at = value
                self._completed_at = now
                self._last_processing_delay_ms = round((now - value) * 1000.0, 3)
                self._frames_processed += 1
            else:
                self._received_at = value
                self._frames_received += 1
            return True

    def frame_received(self, stamp):
        return self._frame(stamp, processed=False)

    def frame_processed(self, stamp):
        return self._frame(stamp, processed=True)

    def record_stale_drop(self, count=1):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 2**31-1:
            return False
        with self._lock:
            self._stale_dropped += count
        return True

    def status(self):
        sampled = self._read_clock()
        with self._lock:
            now = self._now_locked(sampled)
            age = lambda stamp: None if stamp is None else round(max(0.0, now-stamp), 3)
            active = self._active
            return {
                'clock': 'monotonic',
                'current_stage': active[1] if active else 'idle',
                'stage_id': active[0] if active else None,
                'current_stage_started_at': active[2] if active else None,
                'stage_age_s': age(active[2]) if active else 0.0,
                'stage_slow': bool(active and now-active[2] >= self._slow_after_s),
                'active_slow': bool(active and now-active[2] >= self._slow_after_s),
                'slow_after_s': self._slow_after_s,
                'stages': {name: dict(values) for name, values in self._stages.items()},
                'measurements': {name: dict(values) for name, values in self._measurement_stats.items()},
                'active_measurements': [{'token': token, 'stage': name, 'started_at': started,
                                         'source_at': source, 'age_s': age(started)}
                                        for token, (name, started, source) in self._measurements.items()],
                'measurements_dropped': self._measurements_dropped,
                'measurement_slow': any(now-started >= self._slow_after_s
                                        for _, started, _ in self._measurements.values()),
                'recent_frames': [{**record, 'age_s': age(record['finished_at'])}
                                  for record in self._recent_frames],
                'recent_slow': [{**record, 'age_s': age(record['finished_at'])}
                                for record in self._slow],
                'frames_received': self._frames_received,
                'frames_processed': self._frames_processed,
                'last_received_at': self._received_at,
                'last_processed_at': self._processed_at,
                'last_received_age_s': age(self._received_at),
                'last_processed_age_s': age(self._processed_at),
                'since_last_processed_s': age(self._completed_at),
                'last_processing_delay_ms': self._last_processing_delay_ms,
                'stale_dropped': self._stale_dropped,
                'invalid_timestamps': self._invalid_timestamps,
                'out_of_order_frames': self._out_of_order_frames,
                'clock_errors': self._clock_errors,
                'uptime_s': age(self._started_at),
            }
