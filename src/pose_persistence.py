"""Latest-only persistence of verified poses, outside the detection/control lock.

One daemon belongs to the watcher process, not a service or child process.
Generation invalidation prevents an old reference's pending write publishing
after calibration, a camera remount, a profile switch, or shutdown.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time


class PoseWriter:
    def __init__(self, *, interval_s=2., clock=time.monotonic, fsync=os.fsync):
        self._condition = threading.Condition()
        self._publication = threading.Lock()
        self._clock, self._fsync = clock, fsync
        self._interval = max(0., float(interval_s))
        self._generation = 0
        self._pending = self._saved = None
        self._thread = None
        self._stopped = False
        self._writing = False
        self._error = ''
        self._writes = 0

    def start(self):
        with self._condition:
            if self._stopped or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name='pose-persistence', daemon=True)
            self._thread.start()

    def submit(self, path, matrix, saved_at):
        # Bytes detach the request from every caller-owned list/array. Neither
        # filesystem access nor a native write happens on this call's thread.
        payload = json.dumps({'current_to_reference': matrix, 'saved': saved_at},
                             allow_nan=False).encode('utf-8')
        with self._condition:
            if self._stopped:
                return False
            self._pending = (self._generation, Path(path), payload)
            self._condition.notify()
            return True

    def invalidate(self, timeout=.5):
        with self._condition:
            self._generation += 1
            self._pending = self._saved = None
            self._condition.notify_all()
        # A rename may itself block in Windows/native storage. Normal frame
        # submissions never take this mutex. Reference changes need a fence:
        # if publication is already underway they must defer, not race it.
        acquired = self._publication.acquire(timeout=max(0., float(timeout)))
        if acquired:
            self._publication.release()
        return acquired

    def saved(self):
        with self._condition:
            result = self._saved
        if result is None:
            return None
        # Payload is small immutable bytes; the returned matrix is a fresh copy.
        return {'path': result[1], **json.loads(result[2])}

    def status(self):
        with self._condition:
            alive = bool(self._thread and self._thread.is_alive())
            return {'running': alive, 'pending': self._pending is not None,
                    'writing': self._writing, 'saved': self._writes,
                    'error': self._error, 'stopped': self._stopped,
                    'shutdown_waiting_for_io': bool(self._stopped and alive)}

    def stop(self, timeout=.5):
        with self._condition:
            self._stopped = True
            self._generation += 1
            self._pending = self._saved = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.ident is not None and thread is not threading.current_thread():
            thread.join(max(0., float(timeout)))
        return not bool(thread and thread.is_alive())

    def _run(self):
        next_write = 0.
        while True:
            with self._condition:
                while not self._stopped:
                    delay = max(0., next_write-self._clock())
                    if self._pending is not None and not delay:
                        break
                    self._condition.wait(delay if self._pending is not None else None)
                if self._stopped:
                    return
                request, self._pending = self._pending, None
                self._writing = True
            try:
                self._write(request)
            except OSError as error:
                with self._condition:
                    if request[0] == self._generation:
                        self._error = f'{type(error).__name__}: {error}'[:200]
            finally:
                with self._condition:
                    self._writing = False
                    next_write = self._clock()+self._interval
                    self._condition.notify_all()

    def _write(self, request):
        generation, path, payload = request
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                self._fsync(stream.fileno())  # May block; no control/queue lock held.
            with self._publication:
                with self._condition:
                    if self._stopped or generation != self._generation:
                        return
                # No request/status/control mutex is held over native I/O.
                # Successful invalidation waits for any publication that had
                # already passed this generation check before it was canceled.
                os.replace(temporary, path)
                with self._condition:
                    if not self._stopped and generation == self._generation:
                        self._saved = request
                        self._writes += 1
                        self._error = ''
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
