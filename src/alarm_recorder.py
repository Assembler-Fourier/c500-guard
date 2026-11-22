"""A short video of every alarm: the seconds before it and the seconds after.

The watcher already makes an annotated JPEG of every frame for the dashboard
(zones, AI boxes, alarm colours). This keeps the last BEFORE seconds of those
in memory. When an alarm sounds it keeps collecting until AFTER seconds past
the latest alarm in the same clip (alarms close together share one clip, at
most MAX_S long), then a background thread turns the pictures into an MP4
with the bundled ffmpeg at below-normal priority. Only the newest KEEP clips
are kept. Nothing here can delay or silence an alarm.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import zipfile

import platform_support as ps

CLIP_NAME = re.compile(r"^\d{8}-\d{6}-[A-Za-z0-9_-]{1,40}\.mp4$")
BELOW_NORMAL = 0x00004000


def safe_id(alarm_id) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(alarm_id))[-40:] or "alarm"


class AlarmRecorder:
    MAX_S = 60.0          # one clip never grows past this, however many alarms join it
    MIN_FRAMES = 10       # fewer pictures than this is not worth a video
    MAX_FRAMES = 722      # 60 seconds at the existing 12-fps buffer allowance
    MAX_CLEAN_FRAME_BYTES = 256 * 1024
    MAX_CLEAN_BUFFER_BYTES = 16 * 1024 * 1024
    MAX_CLEAN_CLIP_BYTES = 32 * 1024 * 1024
    ENCODER_TIMEOUT_S = 90.0  # covers feeding the pipe as well as encoding

    def __init__(self, folder: Path | str, ffmpeg: str | None, *, before_s=10., after_s=20., keep=25,
                 enabled=True, clock=time.monotonic, on_saved=None, tie_child=None):
        self.folder = Path(folder)
        self.ffmpeg = ffmpeg
        self.clock = clock
        self.on_saved = on_saved          # callback(clip_name, alarm_ids) after a clip is written
        self.tie_child = tie_child        # callback(process) that ties ffmpeg to the watcher's life
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._worker = None
        self._process = None
        self._encoder_timer = None
        self._stopped = False
        self._clip = None
        self._index: dict[str, str] = {}  # alarm id -> clip name
        self.last_error = ""
        self.saved = 0
        self.clean_dropped = 0
        self.configure(before_s=before_s, after_s=after_s, keep=keep, enabled=enabled)
        self._load_index()

    # -- settings -----------------------------------------------------------
    def configure(self, *, before_s=None, after_s=None, keep=None, enabled=None) -> None:
        with self._lock:
            if before_s is not None:
                self.before_s = float(min(30., max(0., before_s)))
            if after_s is not None:
                self.after_s = float(min(60., max(5., after_s)))
            if keep is not None:
                self.keep = int(min(500, max(1, keep)))
            if enabled is not None:
                self.enabled = bool(enabled)
            frames = int(self.before_s * 12) + 2
            old = list(getattr(self, "_buffer", ()))
            self._buffer = deque(old[-frames:], maxlen=frames)
            self._buffer_clean_bytes = sum(len(item[2]) for item in self._buffer if item[2])

    # -- frames and alarms ----------------------------------------------------
    def add_frame(self, jpeg: bytes, now: float | None = None, *,
                  clean_jpeg: bytes | None = None, source_at: float | None = None) -> None:
        """Retain existing encoded bytes only; all archive work belongs to the writer.

        clean_jpeg, when supplied, is the same aligned picture before app
        overlays. source_at is its decoder receipt time on the monotonic clock,
        not camera wall time or a timestamp reconstructed from output FPS.
        """
        if not jpeg:
            return
        now = self.clock() if now is None else now
        if not self._valid_stamp(now):
            return
        source = source_at if self._valid_stamp(source_at) and source_at <= now else None
        clean, omitted = clean_jpeg, None
        if clean is not None:
            if not isinstance(clean, bytes) or not clean:
                clean, omitted = None, 'invalid_clean_jpeg'
            elif len(clean) > self.MAX_CLEAN_FRAME_BYTES:
                clean, omitted = None, 'clean_frame_byte_limit'
            elif source is None:
                clean, omitted = None, 'source_timestamp_unavailable'
        finished = None
        with self._lock:
            if self._stopped:
                return
            if len(self._buffer) == self._buffer.maxlen:
                evicted = self._buffer[0][2]
                self._buffer_clean_bytes -= len(evicted) if evicted else 0
            if clean and self._buffer_clean_bytes + len(clean) > self.MAX_CLEAN_BUFFER_BYTES:
                clean, omitted = None, 'clean_buffer_byte_limit'
            if omitted:
                self.clean_dropped += 1
            item = (now, jpeg, clean, source, omitted)
            self._buffer.append(item)
            self._buffer_clean_bytes += len(clean) if clean else 0
            clip = self._clip
            if clip is not None:
                first = clip['frames'][0][0] if clip['frames'] else clip['started_at']
                if now-first <= self.MAX_S and len(clip['frames']) < self.MAX_FRAMES:
                    if clean and clip['clean_bytes'] + len(clean) > self.MAX_CLEAN_CLIP_BYTES:
                        item = (now, jpeg, None, source, 'clean_clip_byte_limit')
                        self.clean_dropped += 1
                    clip["frames"].append(item)
                    clip['clean_bytes'] += len(item[2]) if item[2] else 0
                if (now >= clip["end_at"] or now-first >= self.MAX_S
                        or len(clip['frames']) >= self.MAX_FRAMES):
                    finished, self._clip = clip, None
        if finished is not None:
            self._submit(finished)

    def alarm(self, alarm_id, kind: str, reason: str, now: float | None = None) -> None:
        """An alarm sounded: start a clip with the pictures from just before it, or extend the current one."""
        if not alarm_id:
            return
        now = self.clock() if now is None else now
        with self._lock:
            if not self.enabled or self._stopped:
                return
            clip = self._clip
            if clip is None:
                frames = [item for item in self._buffer if now - item[0] <= self.before_s]
                first = frames[0][0] if frames else now
                self._clip = {"started_at": now, "end_at": min(first+self.MAX_S, now+self.after_s),
                              "frames": frames, 'clean_bytes': sum(len(item[2]) for item in frames if item[2]),
                              "alarms": [{"id": str(alarm_id), "kind": kind, "reason": str(reason)[:200],
                                          "offset_s": None}],
                              "wall_start": time.time() - (now - first)}
                clip = self._clip
            else:
                clip["alarms"].append({"id": str(alarm_id), "kind": kind, "reason": str(reason)[:200],
                                       "offset_s": None})
                first = clip["frames"][0][0] if clip["frames"] else clip["started_at"]
                clip["end_at"] = min(first + self.MAX_S, max(clip["end_at"], now + self.after_s))
            first = clip["frames"][0][0] if clip["frames"] else now
            clip["alarms"][-1]["offset_s"] = round(now - first, 1)

    def recording_for(self, alarm_id) -> str | None:
        with self._lock:
            name = self._index.get(str(alarm_id))
        return name if name and (self.folder / name).is_file() else None

    def pending_for(self, alarm_id) -> bool:
        with self._lock:
            clip = self._clip
            return bool(clip and any(item["id"] == str(alarm_id) for item in clip["alarms"]))

    def status(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "before_s": self.before_s, "after_s": self.after_s,
                    "keep": self.keep, "recording": self._clip is not None, "saved": self.saved,
                    "stored": len({name for name in self._index.values()}), "error": self.last_error,
                    "clean_frames_dropped": self.clean_dropped}

    @staticmethod
    def _valid_stamp(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)

    # -- writing ------------------------------------------------------------
    def _submit(self, clip) -> None:
        if len(clip["frames"]) < self.MIN_FRAMES or not self.ffmpeg:
            return
        with self._lock:
            if self._stopped:
                return
            try:
                self._jobs.put_nowait(clip)
            except queue.Full:
                self.last_error = "Too many recordings waiting; one alarm video was skipped"
                return
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._write_loop, daemon=True, name="alarm-recorder")
                self._worker.start()

    def _write_loop(self) -> None:
        while not self._stopped:
            try:
                clip = self._jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._write(clip)
            except Exception as exc:                              # noqa: BLE001 - never escape
                self.last_error = f"Alarm video not saved: {type(exc).__name__}: {exc}"[:300]
            finally:
                self._jobs.task_done()
                clip = None  # Do not retain a completed paired clip while idle.

    def _write(self, clip) -> None:
        if self._stopped:
            return
        frames = clip["frames"]
        span = max(.5, frames[-1][0] - frames[0][0])
        rate = max(1., min(15., (len(frames) - 1) / span))
        first = clip["alarms"][0]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clip["wall_start"]))
        self.folder.mkdir(parents=True, exist_ok=True)
        identifier = safe_id(first['id'])
        # A repeated job/name must not replace an older complete evidence set.
        # There is one writer, so reserving a fresh local basename is enough.
        for attempt in range(1000):
            suffix = f'_{attempt+1}' if attempt else ''
            name = f"{stamp}-{identifier[:40-len(suffix)]}{suffix}.mp4"
            target = self.folder / name
            if not any(path.exists() for path in (target, target.with_suffix('.json'),
                                                   target.with_suffix('.clean.zip'))):
                break
        else:
            raise RuntimeError('No unused recording filename available')
        temporary = target.with_name(target.stem + ".part.mp4")
        archive = target.with_suffix('.clean.zip')
        archive_temp = target.with_name(target.stem + '.part.clean.zip')
        metadata = target.with_suffix('.json')
        metadata_temp = target.with_name(target.stem + '.part.json')
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe",
                   "-framerate", f"{rate:.3f}", "-c:v", "mjpeg", "-i", "-", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   "-threads", "2", str(temporary)]
        kwargs = ps.background_subprocess_kwargs()
        if ps.WINDOWS:
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | BELOW_NORMAL
        process = None
        encoder_timer = None
        timed_out = threading.Event()
        published, completed = [], False
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, **kwargs)
            deadline = time.monotonic() + self.ENCODER_TIMEOUT_S

            def expire_encoder():
                # Own this exact child handle. A stalled stdin.write cannot
                # reach communicate's timeout; terminating its reader breaks
                # that write too. No process-name/PID search or shared service.
                if process.poll() is None:
                    timed_out.set()
                    try:
                        process.kill()
                    except OSError:
                        pass  # The child may have exited between poll and kill.

            encoder_timer = threading.Timer(self.ENCODER_TIMEOUT_S, expire_encoder)
            encoder_timer.name = 'alarm-encoder-deadline'
            encoder_timer.daemon = True
            with self._lock:
                self._process = process
                if self._stopped:
                    raise RuntimeError('stopping')
                # Publishing a started timer under the same lock as stop()
                # avoids cancelling/joining it before Thread.start finishes.
                encoder_timer.start()
                self._encoder_timer = encoder_timer
            if self.tie_child is not None:
                try:
                    self.tie_child(process)
                except Exception:                                 # noqa: BLE001
                    pass
            try:
                for item in frames:
                    if self._stopped:
                        raise RuntimeError("stopping")
                    process.stdin.write(item[1])
                process.stdin.close()
                process.stdin = None  # communicate must not flush a closed pipe.
            except (BrokenPipeError, OSError):
                pass
            if self._stopped:
                raise RuntimeError('stopping')
            try:
                _out, error = process.communicate(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                expire_encoder()
                process.communicate(timeout=5)
                raise RuntimeError("the video encoder took too long")
            if timed_out.is_set():
                raise RuntimeError("the video encoder took too long")
            if process.returncode or not temporary.is_file():
                raise RuntimeError((error or b"").decode("utf-8", "replace").strip()[-200:]
                                   or f"ffmpeg exit {process.returncode}")
            if self._stopped:
                raise RuntimeError('stopping')
            # Preserve irregular input timing; a uniform-FPS MP4 is not an
            # exact source timeline. Every rendered frame has a paired row.
            source_start = next((item[3] for item in frames if item[3] is not None), None)
            timeline = []
            for index, item in enumerate(frames):
                at, _jpeg, clean, source, omitted = item
                timeline.append({'index': index, 'preview_at': at,
                                 'preview_offset_s': at-frames[0][0], 'source_at': source,
                                 'source_offset_s': source-source_start if source is not None else None,
                                 'clean': f'clean/{index:06d}.jpg' if clean else None,
                                 'clean_omitted': omitted})
            clean_count = sum(item[2] is not None for item in frames)
            meta = {"name": name, "alarms": clip["alarms"], "seconds": round(span, 1),
                    "frames": len(frames), "fps": round(rate, 2), "before_s": self.before_s,
                    "after_s": self.after_s, "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    'frame_timing': {'clock': 'local_monotonic',
                                     'source_meaning': 'decoder_receipt',
                                     'preview_start_at': frames[0][0], 'source_start_at': source_start,
                                     'frames': timeline},
                    'clean_archive': archive.name if clean_count else None,
                    'clean_frame_space': 'aligned_before_app_overlays' if clean_count else None,
                    'clean_frames': clean_count,
                    'clean_frames_omitted': sum(item[4] is not None for item in frames)}
            encoded_meta = json.dumps(meta, indent=1, allow_nan=False).encode('utf-8')
            if clean_count:
                # JPEG is already compressed. Stored entries avoid extra
                # compression CPU and share the existing worker/retention.
                with zipfile.ZipFile(archive_temp, 'w', compression=zipfile.ZIP_STORED) as bundle:
                    bundle.writestr('manifest.json', encoded_meta)
                    for row, item in zip(timeline, frames):
                        if self._stopped:
                            raise RuntimeError('stopping')
                        if row['clean']:
                            bundle.writestr(row['clean'], item[2])
            metadata_temp.write_bytes(encoded_meta)
            if self._stopped:
                raise RuntimeError('stopping')
            # Build all three artifacts before publishing any of them. Each
            # replacement is atomic; metadata is the final completed receipt.
            os.replace(temporary, target)
            published.append(target)
            if clean_count:
                os.replace(archive_temp, archive)
                published.append(archive)
            os.replace(metadata_temp, metadata)
            published.append(metadata)
            completed = True
        finally:
            if encoder_timer is not None:
                encoder_timer.cancel()
                if encoder_timer.ident is not None:
                    encoder_timer.join()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(5)
            if process is not None:
                for pipe in (process.stdin, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            pass
            with self._lock:
                self._process = None
                self._encoder_timer = None
            for path in (temporary, archive_temp, metadata_temp):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if not completed:
                # Roll back any newly published portion; existing complete
                # recordings were never selected as publication destinations.
                for path in published:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
        with self._lock:
            for item in clip["alarms"]:
                self._index[item["id"]] = name
            self.saved += 1
            self.last_error = ""
        self._prune()
        if self.on_saved is not None:
            try:
                self.on_saved(name, [item["id"] for item in clip["alarms"]])
            except Exception:                                     # noqa: BLE001
                pass

    # Clips holding only alignment diagnostics (21 Nov 2026: 176 of 200 kept clips) may use at most this share of the
    # store and go first, so they never push real alarm evidence out. Barrier-motion diagnostics count as evidence.
    DIAGNOSTIC_SHARE = .2
    DIAGNOSTIC_PREFIX = 'diag-alignment-'

    def _prune(self) -> None:
        clips = sorted((path for path in self.folder.glob("*.mp4") if CLIP_NAME.match(path.name)),
                       key=lambda path: path.stat().st_mtime)
        with self._lock:
            known = dict(self._index)
        evidence = {name for ident, name in known.items() if not ident.startswith(self.DIAGNOSTIC_PREFIX)}
        diagnostic = {name for name in known.values() if name not in evidence}
        cap = max(1, int(self.keep * self.DIAGNOSTIC_SHARE))
        diagnostic_clips = [path for path in clips if path.name in diagnostic]
        doomed = diagnostic_clips[:-cap] if len(diagnostic_clips) > cap else []
        remaining = [path for path in clips if path not in doomed]
        if len(remaining) > self.keep:
            first = ([path for path in remaining if path.name in diagnostic]
                     + [path for path in remaining if path.name not in diagnostic])
            doomed += first[:len(remaining) - self.keep]
        for old in doomed:
            for path in (old, old.with_suffix(".json"), old.with_suffix('.clean.zip')):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        with self._lock:
            index = dict(self._index)
        # Disk checks never hold the lock used by the detector's add_frame.
        retained = {key: name for key, name in index.items() if (self.folder / name).is_file()}
        with self._lock:
            self._index = retained

    def _load_index(self) -> None:
        try:
            sidecars = sorted(self.folder.glob("*.json"), key=lambda path: path.stat().st_mtime)
        except OSError:
            return
        for path in sidecars:
            if path.name.endswith('.part.json'):
                continue
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                name = meta.get("name")
                if isinstance(name, str) and CLIP_NAME.match(name) and (self.folder / name).is_file():
                    for item in meta.get("alarms", []):
                        if isinstance(item, dict) and isinstance(item.get("id"), str):
                            self._index[item["id"]] = name
            except (OSError, ValueError, AttributeError):
                continue
        for pattern in ('*.part.mp4', '*.part.clean.zip', '*.part.json'):
            for leftover in self.folder.glob(pattern) if self.folder.is_dir() else ():
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass

    def stop(self) -> None:
        """Shutdown: drop an unfinished clip and end the encoder, so nothing is left running."""
        with self._lock:
            self._stopped = True
            self._clip = None
            self._buffer.clear()
            self._buffer_clean_bytes = 0
            while True:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except queue.Empty:
                    break
            process = self._process
            encoder_timer = self._encoder_timer
        if encoder_timer is not None:
            encoder_timer.cancel()
            encoder_timer.join()
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(5)
            except Exception:                                     # noqa: BLE001
                pass
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.)
