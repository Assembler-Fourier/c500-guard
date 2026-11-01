#!/usr/bin/env python3
"""c500-watch - a local, source-agnostic change/light watcher with a latched buzzer.

It does not care where the picture comes from. Give it an RTSP/HTTP stream, a
macOS screen rectangle or a USB/AVFoundation camera and it will:

  * raise a latched MAIN alarm when something changes inside an armed zone,
  * raise a short EARLY warning when a zone suddenly gets brighter (headlights),
  * raise a distinct FAULT alarm when the feed is lost, frozen or black,

and keep sounding until a human acknowledges it. Nothing leaves this machine.

Alarm-only. This program never operates a barrier or any other powered device.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
from io import BytesIO
import json
import math
import os
import queue
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw

import platform_support as ps
from illumination import compensated_motion_diff

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Runtime state lives outside the repo, so the checkout stays clean and
# nothing private can ever be committed or zipped up by accident.
STATE_DIR = ps.state_dir()
SOUND_DIR = STATE_DIR / "sounds"
EVENT_LOG = STATE_DIR / "events.jsonl"
ARMED_FILE = STATE_DIR / "armed.json"
DEFAULT_ZONES = REPO / "config" / "zones.json"
ALERT_DIR = STATE_DIR / "alerts"
ALERT_NAME = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}\d+-(?:early|main|fault)\.jpg$")
ALERT_KEEP = 300            # newest alarm snapshots kept on disk (about 15 MB)
LOOP_STALL_S = 20.0         # a detector loop silent this long is restarted
MAX_LIGHTING_REFERENCES = 4  # learned references for other lighting (dusk, night)
LEARN_AFTER_S = 60.0         # no strong landmark match for this long before learning
LEARN_EVERY_S = 600.0        # at most one learning attempt per ten minutes

# Every source is normalised to this working size, so zone coordinates stay
# valid when the input resolution changes.
PROC_W, PROC_H = 640, 360
FRAME_BYTES = PROC_W * PROC_H * 3

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[41;97;1m"
YELLOW = "\033[43;30;1m"
GREEN = "\033[42;30;1m"
BLUE = "\033[44;97;1m"


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

SCALE_CHAIN = (
    f"scale={PROC_W}:{PROC_H}:force_original_aspect_ratio=decrease,"
    f"pad={PROC_W}:{PROC_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
)


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        sys.exit("ffmpeg was not found on PATH. Install it (brew install ffmpeg) and retry.")
    return found


def safe_source(source: str) -> str:
    """Keep camera credentials and URL tokens out of status and diagnostics."""
    if "://" not in source:
        return source
    try:
        parts = urlsplit(source)
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1],
                           parts.path, "redacted" if parts.query else "", ""))
    except ValueError:
        return "[private stream URL]"


def build_command(source: str, fps: float) -> list[str]:
    """Translate a source string into an ffmpeg command emitting raw rgb24."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    filters = []

    if source.startswith("screen:"):
        # screen:            -> whole desktop
        # screen:x,y,w,h     -> that rectangle (all platforms)
        # screen:4@x,y,w,h   -> macOS device index plus a crop
        spec = source[len("screen:"):]
        cmd += ps.screen_input(spec, fps)
        if ps.MACOS and "@" in spec:
            x, y, w, h = (int(v) for v in spec.partition("@")[2].split(","))
            filters.append(f"crop={w}:{h}:{x}:{y}")
        filters.append(f"fps={fps}")

    elif source.startswith("cam:"):
        cmd += ps.camera_input(source[len("cam:"):], fps)
        filters.append(f"fps={fps}")

    elif source.startswith("rtsp://"):
        # No -timeout here: on the RTSP demuxer it implies listen mode and
        # breaks the connection. FrameReader's own stall watchdog covers this.
        # Slice threads do not add one decoded-frame delay per frame thread.
        # Keep probing intact: aggressive nobuffer/probe tuning delayed this
        # camera's first usable keyframe in the measured A/B test.
        cmd += ["-threads", "2", "-thread_type", "slice", "-flags", "+low_delay",
                "-rtsp_transport", "tcp", "-i", source]
        filters.append(f"fps={fps}")

    elif source.startswith(("http://", "https://", "rtmp://", "srt://", "udp://")):
        cmd += ["-i", source]
        filters.append(f"fps={fps}")

    elif source == "selftest":
        # Synthetic feed: a dark scene, a bright flash, then a moving block.
        cmd += ["-re", "-f", "lavfi", "-i",
                f"testsrc2=size={PROC_W}x{PROC_H}:rate={fps}"]
        filters.append(f"fps={fps}")

    else:  # a media file on disk
        path = Path(source).expanduser()
        if not path.exists():
            sys.exit(f"Unrecognised source and not a file on disk: {source}")
        cmd += ["-re", "-i", str(path)]
        filters.append(f"fps={fps}")

    filters.append(SCALE_CHAIN)
    cmd += ["-an", "-sn", "-vf", ",".join(filters),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return cmd


class FrameReader:
    """Runs ffmpeg and hands out fixed-size RGB frames, restarting on failure."""

    def __init__(self, source: str, fps: float, connect_grace: float = 30.0,
                 stall_after: float = 5.0):
        self.source = source
        self.fps = fps
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.dropped_frames = 0
        self.stderr_tail: list[str] = []
        self.restarts = 0
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._progress_at = 0.0
        self._got_frame = False
        self.connect_grace = max(0.1, connect_grace)
        self.stall_after = max(0.1, stall_after)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._watchdog = threading.Thread(target=self._watch_stalls, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()
        with self._proc_lock:
            proc = self._proc
        self._finish_process(proc)
        for thread in (self._thread, self._watchdog):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=3)

    @staticmethod
    def _finish_process(proc) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _watch_stalls(self) -> None:
        # Reading a pipe can block forever even while ffmpeg is still alive.
        # This separate thread terminates the child so that read() unblocks.
        while not self._stop.wait(0.1):
            with self._proc_lock:
                proc = self._proc
                limit = self.stall_after if self._got_frame else self.connect_grace
                stale = proc is not None and time.monotonic() - self._progress_at > limit
            if stale:
                self._finish_process(proc)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                line = line.replace(self.source, safe_source(self.source))
                line = re.sub(r"([a-z][a-z0-9+.-]*://)[^/\s]*@",
                              r"\1[redacted]@", line, flags=re.IGNORECASE)
                self.stderr_tail = (self.stderr_tail + [line])[-8:]

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                cmd = build_command(self.source, self.fps)
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                        creationflags=0x08000000 if ps.WINDOWS else 0)
            except (OSError, ValueError, SystemExit):
                self.stderr_tail = ["Could not start the stream decoder; check ffmpeg and source configuration."]
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            with self._proc_lock:
                self._proc = proc
                self._progress_at = time.monotonic()
                self._got_frame = False
            stderr_thread = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
            stderr_thread.start()
            got_any = False
            try:
                while not self._stop.is_set():
                    buf = proc.stdout.read(FRAME_BYTES)
                    if len(buf) != FRAME_BYTES:
                        break
                    got_any = True
                    backoff = 1.0
                    with self._proc_lock:
                        self._progress_at = time.monotonic()
                        self._got_frame = True
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                            self.dropped_frames += 1
                        except queue.Empty:
                            pass
                    self.frames.put_nowait((time.monotonic(), buf))
            finally:
                self._finish_process(proc)
                stderr_thread.join(timeout=2)
                for pipe in (proc.stdout, proc.stderr):
                    if pipe is not None:
                        pipe.close()
                with self._proc_lock:
                    if self._proc is proc:
                        self._proc = None
            if self._stop.is_set():
                return
            self.restarts += 1
            if not got_any:
                backoff = min(backoff * 2, 15.0)
            self._stop.wait(backoff)


# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------

@dataclass
class Zone:
    """One armed rectangle. Motion and light are independent trigger channels."""

    name: str
    rect: tuple[float, float, float, float]  # normalised x, y, w, h
    motion: bool = True
    light: bool = False
    severity: str = "main"          # raised when the motion channel fires
    light_severity: str = "early"   # raised when only the light channel fires
    motion_pct: float = 1.2         # % of zone pixels that must change
    light_delta: float = 10.0       # mean-luma jump that counts as a light change
    on_frames: int = 2
    off_frames: int = 8

    # runtime
    motion_active: bool = field(default=False, repr=False)
    light_active: bool = field(default=False, repr=False)
    _counts: dict = field(default_factory=lambda: {"motion_on": 0, "motion_off": 0,
                                                   "light_on": 0, "light_off": 0}, repr=False)
    last_pct: float = field(default=0.0, repr=False)
    last_delta: float = field(default=0.0, repr=False)

    @property
    def active(self) -> bool:
        return self.motion_active or self.light_active

    def slice_for(self, w: int, h: int):
        x = min(int(self.rect[0] * w), w - 1)
        y = min(int(self.rect[1] * h), h - 1)
        ww = max(1, int(self.rect[2] * w))
        hh = max(1, int(self.rect[3] * h))
        return slice(y, min(y + hh, h)), slice(x, min(x + ww, w))

    def step(self, channel: str, hit: bool) -> str:
        """Advance one channel's hysteresis. Returns "rise", "fall" or ""."""
        on_key, off_key = channel + "_on", channel + "_off"
        if hit:
            self._counts[on_key] += 1
            self._counts[off_key] = 0
        else:
            self._counts[off_key] += 1
            self._counts[on_key] = 0
        active = getattr(self, channel + "_active")
        if not active and self._counts[on_key] >= self.on_frames:
            setattr(self, channel + "_active", True)
            return "rise"
        if active and self._counts[off_key] >= self.off_frames:
            setattr(self, channel + "_active", False)
            return "fall"
        return ""


def default_zones() -> list[Zone]:
    """Whole-frame fallback so the tool is useful before zones are drawn."""
    return [Zone(name="whole view", rect=(0.0, 0.0, 1.0, 1.0),
                 motion=True, light=True, severity="main",
                 motion_pct=0.8, light_delta=10.0)]


def load_zones(path: Path) -> list[Zone]:
    if not path.exists():
        return default_zones()
    raw = json.loads(path.read_text())
    zones = []
    for item in raw.get("zones", raw if isinstance(raw, list) else []):
        item = dict(item)
        item["rect"] = tuple(float(v) for v in item["rect"])
        zones.append(Zone(**item))
    return zones or default_zones()


# --------------------------------------------------------------------------
# Alarm
# --------------------------------------------------------------------------

SOUND_VERSION = 3   # bump to force regeneration of the WAVs


def _square(phase: float) -> float:
    return 1.0 if math.sin(phase) >= 0 else -1.0


def siren_voice(phase: float) -> float:
    """Square-dominant with a second harmonic: cuts through room noise."""
    return 0.55 * _square(phase) + 0.30 * math.sin(phase) + 0.15 * math.sin(2 * phase)


def render(path: Path, seconds: float, freq_at, voice=siren_voice,
           volume: float = 0.97, fade: float = 0.004) -> None:
    """freq_at(t) -> Hz (0 for silence). Phase is accumulated, so sweeps are clean."""
    rate = 44100
    raw = []
    phase = 0.0
    for n in range(int(rate * seconds)):
        t = n / rate
        freq = freq_at(t)
        if freq <= 0:
            sample = 0.0
        else:
            phase += 2 * math.pi * freq / rate
            sample = voice(phase)
        # Short fades stop the loop seam from clicking.
        edge = min(t, seconds - t)
        raw.append(sample * min(1.0, edge / fade))
    # Normalise to the requested level so no headroom is wasted.
    peak = max((abs(v) for v in raw), default=1.0) or 1.0
    scale = volume / peak
    frames = bytearray()
    for v in raw:
        frames += struct.pack("<h", int(max(-1.0, min(1.0, v * scale)) * 32767))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def _wail(t: float, period: float = 1.6, low: float = 620.0, high: float = 1650.0) -> float:
    """Classic rising-then-falling siren sweep. Starts and ends low, so it loops seamlessly."""
    x = (t % period) / period
    tri = 2 * x if x < 0.5 else 2 * (1 - x)
    return low + (high - low) * tri


def ensure_sounds(force: bool = False) -> dict[str, Path]:
    sounds = {
        "main": SOUND_DIR / "main.wav",
        "early": SOUND_DIR / "early.wav",
        "fault": SOUND_DIR / "fault.wav",
    }
    stamp = SOUND_DIR / ".version"
    current = stamp.read_text().strip() if stamp.exists() else "0"
    if force or current != str(SOUND_VERSION) or not all(p.exists() for p in sounds.values()):
        # MAIN: continuous swept siren, no gaps, near full scale.
        render(sounds["main"], 3.2, _wail, volume=0.97)
        # EARLY: three fast rising chirps -- urgent but obviously not the siren.
        render(sounds["early"], 0.90,
               lambda t: (1150 + 800 * ((t % 0.30) / 0.13)) if (t % 0.30) < 0.13 else 0,
               volume=0.72)
        # FAULT: slow low two-tone warble; can never be mistaken for the siren.
        render(sounds["fault"], 1.60,
               lambda t: 385 if (t % 0.80) < 0.40 else 280,
               volume=0.93)
        SOUND_DIR.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(SOUND_VERSION) + "\n")
    return sounds


class Buzzer:
    """Loops a tone until told to stop. Silent whenever the system is disarmed."""

    def __init__(self, sounds: dict[str, Path], muted: bool = False):
        self.sounds = sounds
        self.muted = muted
        self.player = ps.AudioPlayer()
        self._want: str | None = None
        self._lock = threading.Lock()

    def set(self, kind: str | None) -> None:
        with self._lock:
            if kind == self._want:
                return
            self._want = kind
            if kind is None or self.muted:
                self.player.stop()
            else:
                self.player.start_loop(self.sounds[kind])

    def shutdown(self) -> None:
        with self._lock:
            self._want = None
            self.player.stop()


def notify(title: str, message: str) -> None:
    """Desktop notification, best effort. Never raised while disarmed."""
    try:
        if ps.MACOS:
            subprocess.Popen(
                ["osascript", "-e",
                 f'display notification {json.dumps(message)} with title {json.dumps(title)}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif ps.LINUX and shutil.which("notify-send"):
            subprocess.Popen(["notify-send", "-u", "critical", title, message],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Windows: deliberately no toast. A popup on a 24/7 guard machine is
        # noise, and the siren plus the web page already carry the signal.
    except OSError:
        pass


# --------------------------------------------------------------------------
# Watcher
# --------------------------------------------------------------------------

class Watcher:
    def __init__(self, args):
        self.args = args
        self.zones = load_zones(Path(args.zones))
        self.sounds = ensure_sounds()
        self.buzzer = Buzzer(self.sounds, muted=args.silent)
        self.reader = FrameReader(args.source, args.fps, args.connect_grace, args.lost_after)

        self.bg: np.ndarray | None = None
        self.started = time.monotonic()
        self.warmup_started: float | None = None
        # Seeded with the start time so a source that never connects at all
        # still raises FEED LOST instead of sitting silently at OK.
        self.last_frame_at = self.started
        # RTSP needs time to connect and wait for a keyframe. Until the first
        # frame lands, use the longer connect grace instead of the running
        # feed-lost threshold, or startup itself trips a false FAULT.
        self.first_frame_seen = False
        self.last_frame_clock = "--:--:--"
        self.last_hash: int | None = None
        self.identical_since = 0.0
        self.dark_since = 0.0
        self.frame_count = 0

        # Armed state persists across restarts and reboots, so a guard who
        # switched it off does not get an unexpected siren after a power cut.
        self._state_lock = threading.RLock()
        self.armed = self._load_armed()
        self._pending_rearm = False
        self._pending_zones = None
        self.alarm: str | None = None        # None | "early" | "main" | "fault"
        self.alarm_reason = ""
        self.alarm_since = 0.0
        self.alarm_id: str | None = None
        self._alarm_sequence = 0
        self._alarm_session = os.urandom(12).hex()
        self.ack_quiet_s = 5.0
        # Global keyboard/mouse input counts only after the siren has sounded
        # this long, so someone typing cannot silence a new alarm unheard.
        self.input_ack_min_s = 1.0
        self._ack_until = 0.0
        self._resume_alarm_check = False
        self.control_warning = ""
        self.acked = False
        self.fault_reason = ""
        self._fault_kind: str | None = None

        self._keys = None
        self.preview_jpeg: bytes | None = None
        self.scene_jpeg: bytes | None = None
        self.raw_jpeg: bytes | None = None
        self.local_pipeline_ms = 0.0
        self.processing_ms = 0.0
        self._preview_lock = threading.Lock()
        self._running = True
        self._last_status_log = 0.0
        self._last_status_key = None
        self.sleep_inhibitor = ps.SleepInhibitor()
        # Health: detector heartbeat, input-helper polls and speaker state.
        self._loop_heartbeat = time.monotonic()
        self._input_helper_seen: float | None = None
        self.audio = {"available": False, "detail": "Checking the speaker"}
        self._frame_error = ""
        self._frame_error_since = 0.0
        self._last_error_print = 0.0
        # Recent alarms with snapshots, newest first, rebuilt from the event log.
        self.recent_alerts: deque = deque(maxlen=30)
        self.alerts_version = 0
        self._snapshots: queue.Queue = queue.Queue(maxsize=16)
        self._snapshot_writer: threading.Thread | None = None
        self._load_recent_alerts()
        self.semantic = None
        model = REPO / "models" / "yolox_tiny.onnx"
        if model.is_file():
            from semantic_monitor import SemanticMonitor
            self.semantic = SemanticMonitor(self.zones, model, interval=0.5)
        self.aligner = None
        self._alignment_reference = None
        self._pending_calibration = False
        self._pending_lighting = False
        self._lighting_status = ""
        self._alignment_bad_since = 0.0
        self._alignment_blocked = False
        self._calibration_error = ""
        # (current->reference matrix, monotonic time) of the last strongly
        # verified estimate, of the matrix applied to the last accepted frame,
        # the verified pose saved across restarts, and how long a brief
        # rejection may keep using the applied matrix.
        self._alignment_good = None
        self._alignment_used = None
        self._alignment_pose = None
        self.alignment_hold_s = 10.0
        # Learned lighting references share the zone coordinate frame of the
        # main reference; index 0 is the main reference, i is extra i-1.
        self._extra_aligners: list = []
        self._extra_used: list = []
        self._preferred_aligner = 0
        self._probe_index = 0
        self._last_strong_at = 0.0
        self._last_learn_at = 0.0
        self._learning = None
        self.alignment = {"state": "disabled", "message": "Automatic alignment is not configured",
                          "calibrated": False, "confidence": None, "dx": 0, "dy": 0}
        self._load_alignment()
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _alignment_path(self):
        # Separate sources cannot silently reuse one another's reference.
        key = hashlib.sha256(self.args.source.encode()).hexdigest()[:16]
        return STATE_DIR / f"alignment-{key}.npz"

    def _feature_mask(self, frame):
        from scene_alignment import build_feature_mask
        cfg = self.alignment_config
        return build_feature_mask(frame.shape,
            include_rects=cfg.get("include_rects", [[.12, .10, .79, .82]]),
            exclude_rects=[list(z.rect) for z in self.zones] + cfg.get("exclude_rects", []),
            border_px=int(cfg.get("border_px", 8)))

    def _required_mask(self, frame):
        from scene_alignment import build_feature_mask
        return build_feature_mask(frame.shape, include_rects=[list(z.rect) for z in self.zones],
                                  exclude_rects=[], border_px=0)

    def _pose_path(self):
        return self._alignment_path().with_suffix(".pose.json")

    def _read_pose(self):
        try:
            data = json.loads(self._pose_path().read_text(encoding="utf-8"))
            matrix = np.asarray(data["current_to_reference"], dtype=np.float64)
            return matrix if matrix.shape == (2, 3) and np.isfinite(matrix).all() else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _maybe_save_pose(self, matrix, shape) -> None:
        """Persist a verified pose only when it really changes, so writes stay rare."""
        from scene_alignment import pose_gap_px
        if self._alignment_pose is not None and pose_gap_px(matrix, self._alignment_pose, shape) <= 0.5:
            return
        self._alignment_pose = np.asarray(matrix, dtype=np.float64)
        try:
            from zone_editor import _atomic_write
            _atomic_write(self._pose_path(), json.dumps({
                "current_to_reference": self._alignment_pose.round(6).tolist(),
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
        except OSError:
            pass

    def _alignment_prior(self):
        """The camera pose that weak (for example night-time) landmarks may confirm."""
        if self._alignment_good is not None:
            return self._alignment_good[0]
        if self._alignment_pose is not None:
            return self._alignment_pose
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]   # The reference's own pose.

    # -- learned lighting references -------------------------------------
    def _extra_path(self):
        return self._alignment_path().with_suffix(".lighting.npz")

    def _build_aligner(self, image):
        from scene_alignment import SceneAligner
        aligner = SceneAligner(image, self._feature_mask(image), required_mask=self._required_mask(image))
        return aligner if aligner.reference_ready else None

    def _load_lighting_references(self) -> None:
        try:
            with np.load(self._extra_path(), allow_pickle=False) as stored:
                images = [stored[key] for key in sorted(stored.files)][:MAX_LIGHTING_REFERENCES]
        except (OSError, ValueError, KeyError):
            return
        for image in images:
            if image.shape == (PROC_H, PROC_W, 3) and image.dtype == np.uint8:
                aligner = self._build_aligner(image)
                if aligner is not None:
                    self._extra_aligners.append(aligner)
                    self._extra_used.append(0.0)

    def _save_lighting_references(self) -> None:
        from zone_editor import _atomic_write
        buffer = BytesIO()
        np.savez_compressed(buffer, **{f"ref{index}": aligner.reference
                                       for index, aligner in enumerate(self._extra_aligners)})
        _atomic_write(self._extra_path(), buffer.getvalue())

    def _update_alignment(self, frame, prior):
        """Try the preferred reference, then one other in turn; a strong match wins."""
        aligners = [self.aligner] + self._extra_aligners
        preferred = self._preferred_aligner if self._preferred_aligner < len(aligners) else 0
        order = [preferred]
        if len(aligners) > 1:
            self._probe_index = (self._probe_index + 1) % len(aligners)
            if self._probe_index == preferred:
                self._probe_index = (self._probe_index + 1) % len(aligners)
            order.append(self._probe_index)
        first = weak = None
        for index in order:
            aligned, info = aligners[index].update(frame, prior=prior)
            if first is None:
                first = (aligned, info)
            if aligned is not None and not info.get("weak"):
                self._preferred_aligner = index
                if index:
                    self._extra_used[index - 1] = time.monotonic()
                return aligned, {**info, "reference_index": index}
            if aligned is not None and weak is None:
                weak = (aligned, {**info, "reference_index": index})
        return weak or first

    def _consider_learning(self, frame, aligned, info, now) -> None:
        """Learn a lighting reference when only weak evidence confirms the pose.

        The candidate is the frame warped with the verified prior, so it is in
        the zone coordinate frame. It is kept only if it then strongly aligns
        at least three of the next five frames at the verified pose.
        """
        from scene_alignment import pose_gap_px
        if self._learning is not None:
            candidate, tries, passes = self._learning
            result, status = candidate.update(frame)
            matrix = status.get("current_to_reference")
            passed = (result is not None and matrix is not None
                      and pose_gap_px(matrix, self._alignment_prior(), frame.shape) <= 0.75)
            tries, passes = tries + 1, passes + passed
            if tries < 5:
                self._learning = (candidate, tries, passes)
                return
            self._learning = None
            if passes >= 3:
                self._commit_lighting_reference(candidate, now)
                self._lighting_status = (f"Saved a lighting reference at {time.strftime('%H:%M')} "
                                         f"({len(self._extra_aligners)} stored)")
            else:
                self._lighting_status = "The view did not stay consistent; no lighting reference was saved"
            return
        agreement = info.get("static_agreement") or {}
        if (aligned is None or not info.get("weak") or info.get("pose_gap_px", 99.0) > 1.5
                or agreement.get("median_correlation", 0.0) < 0.6
                or now - self._last_strong_at < LEARN_AFTER_S or now - self._last_learn_at < LEARN_EVERY_S):
            return
        self._last_learn_at = now
        candidate = self._build_aligner(np.ascontiguousarray(aligned))
        if candidate is not None:
            self._learning = (candidate, 0, 0)

    def _commit_lighting_reference(self, candidate, now) -> None:
        if len(self._extra_aligners) >= MAX_LIGHTING_REFERENCES:
            slot = min(range(len(self._extra_used)), key=self._extra_used.__getitem__)
            self._extra_aligners[slot], self._extra_used[slot] = candidate, now
        else:
            self._extra_aligners.append(candidate)
            self._extra_used.append(now)
            slot = len(self._extra_aligners) - 1
        self._preferred_aligner = slot + 1
        try:
            self._save_lighting_references()
        except OSError:
            pass
        self.log_event("alignment", f"learned lighting reference {slot + 1} "
                                    f"({len(self._extra_aligners)} stored)")

    def _load_alignment(self):
        try:
            config = json.loads(Path(self.args.zones).read_text(encoding="utf-8-sig"))
            self.alignment_config = config.get("alignment", {}) if isinstance(config, dict) else {}
            hold = self.alignment_config.get("hold_s", 10.0)
            if isinstance(hold, (int, float)) and not isinstance(hold, bool) and math.isfinite(hold):
                self.alignment_hold_s = min(30.0, max(0.0, float(hold)))
            if not self.alignment_config.get("enabled", False):
                return
            from scene_alignment import SceneAligner
            self.aligner = SceneAligner()
            path = self._alignment_path()
            if path.is_file():
                with np.load(path, allow_pickle=False) as stored:
                    frame = stored["reference"]
                if frame.shape != (PROC_H, PROC_W, 3):
                    raise ValueError("Reference size changed; use the current view to recalibrate")
                self._alignment_reference = frame
                self.aligner.calibrate(frame, self._feature_mask(frame), required_mask=self._required_mask(frame))
                self._alignment_pose = self._read_pose()
                self._load_lighting_references()
            self.alignment = {**self.alignment, "state": "calibration_required",
                "message": "Checking scene landmarks" if path.is_file() else "Set zones, then use this view as the reference",
                "calibrated": self._alignment_reference is not None}
            self._alignment_blocked = True
        except Exception as exc:
            self.alignment = {**self.alignment, "state": "unavailable", "message": str(exc)}
            self._alignment_blocked = True

    def request_calibration(self):
        with self._state_lock:
            if self.armed:
                raise PermissionError("Turn OFF before changing the scene reference")
            if self.aligner is None:
                raise ValueError("Automatic alignment is unavailable or not enabled")
            if not self.first_frame_seen or time.monotonic() - self.last_frame_at > 2:
                raise ValueError("Wait for fresh camera frames before calibration")
            self._pending_calibration = True
            self._calibration_error = ""
            self.alignment = {**self.alignment, "state": "calibrating", "message": "Saving the current scene reference"}

    def request_lighting_reference(self):
        """Keep the current view (at the verified pose) as an extra lighting reference."""
        with self._state_lock:
            if self.armed:
                raise PermissionError("Turn OFF before saving a lighting reference")
            if self.aligner is None:
                raise ValueError("Automatic alignment is unavailable or not enabled")
            if not self.first_frame_seen or time.monotonic() - self.last_frame_at > 2:
                raise ValueError("Wait for fresh camera frames first")
            if self._learning is not None:
                raise ValueError("A lighting reference is already being checked")
            self._pending_lighting = True
            self._lighting_status = "Checking the current view as a lighting reference"

    def align_frame(self, frame):
        with self._state_lock:
            return self._align_frame_locked(frame)

    def _align_frame_locked(self, frame):
        if self.aligner is None:
            if self._alignment_blocked:
                self._alignment_bad_since = self._alignment_bad_since or time.monotonic()
            return None if self._alignment_blocked else frame
        with self._state_lock:
            if self._pending_zones is not None:
                self.zones = self._pending_zones
                self._pending_zones = None
                if self.semantic:
                    self.semantic.zones = self.zones
                self._pending_rearm = True
                if self._alignment_reference is not None:
                    self.aligner.calibrate(self._alignment_reference, self._feature_mask(self._alignment_reference),
                                           required_mask=self._required_mask(self._alignment_reference))
                    # Zone rectangles shape every mask: rebuild learned references too.
                    self._extra_aligners = [aligner for aligner in (
                        self._build_aligner(old.reference) for old in self._extra_aligners) if aligner is not None]
                    self._extra_used = [0.0] * len(self._extra_aligners)
                    self._preferred_aligner, self._learning = 0, None
            calibrate = self._pending_calibration and not self.armed
            self._pending_calibration = False
        if calibrate:
            # Preserve zone coordinates in the stabilized view when tracking.
            current, _ = self.aligner.update(frame)
            reference = current if current is not None else frame
            try:
                from scene_alignment import SceneAligner
                candidate = SceneAligner(reference, self._feature_mask(reference),
                                         required_mask=self._required_mask(reference))
                aligned, info = candidate.update(frame)
                if aligned is None:
                    raise ValueError(info.get("reason") or "Not enough static landmarks")
                path = self._alignment_path()
                buffer = BytesIO()
                np.savez_compressed(buffer, reference=reference)
                from zone_editor import _atomic_write
                _atomic_write(path, buffer.getvalue())
                self.aligner = candidate
                self._alignment_reference = reference.copy()
                # Matrices and the saved pose belong to the old reference.
                self._alignment_good = self._alignment_used = self._alignment_pose = None
                self._extra_aligners, self._extra_used = [], []
                self._preferred_aligner, self._learning = 0, None
                for stale in (self._pose_path(), self._extra_path()):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                self._pending_rearm = True
            except Exception as exc:
                self._calibration_error = str(exc)
        aligned, info = self._update_alignment(frame, self._alignment_prior())
        accepted = aligned is not None
        now = time.monotonic()
        holding = False
        if accepted:
            matrix = info.get("current_to_reference")
            if matrix is not None and not info.get("weak"):
                from scene_alignment import pose_gap_px
                if self._alignment_used is not None and pose_gap_px(matrix, self._alignment_used[0], frame.shape) > 2.0:
                    # The zones would visibly jump: relearn the background first.
                    self._pending_rearm = True
                self._last_strong_at = now
                self._alignment_good = (matrix, now)
                self._maybe_save_pose(matrix, frame.shape)
            if matrix is not None:
                self._alignment_used = (matrix, now)
            if self._alignment_blocked:
                self._pending_rearm = True
            self._alignment_bad_since = 0.0
        else:
            self._alignment_bad_since = self._alignment_bad_since or now
            # A fixed mount does not move between frames. Brief landmark loss
            # (glare, a large vehicle, decoder smear) keeps the last verified
            # transform, so detection is neither paused nor reset. Only a
            # rejection lasting longer than the hold becomes a fault.
            if (self._alignment_used is not None and not self._alignment_blocked
                    and now - self._alignment_bad_since <= self.alignment_hold_s):
                try:
                    aligned = self.aligner.warp(frame, self._alignment_used[0])
                except Exception:
                    aligned = None
                holding = aligned is not None
        self._alignment_blocked = aligned is None
        translation = info.get("translation_px") or [0, 0]
        if holding:
            translation = [self.alignment.get("dx", 0), self.alignment.get("dy", 0)]
            left = max(0.0, self.alignment_hold_s - (now - self._alignment_bad_since))
            message = (f"Holding the last verified alignment ({left:.0f} s left): "
                       + info.get("reason", "landmarks briefly unclear"))
        elif accepted and info.get("weak"):
            message = "Low light: landmarks confirm the camera has not moved; zones stay at the verified position"
        elif accepted:
            message = "Zones locked to scene landmarks" + (
                " (learned lighting reference)" if info.get("reference_index") else "")
        else:
            message = info.get("reason", "Scene alignment needs attention")
        self.alignment = {**info, "state": "tracking" if accepted else "holding" if holding else "calibration_required",
            "message": ("Reference unchanged: " + self._calibration_error) if self._calibration_error else message,
            "calibrated": bool(info.get("reference_ready")), "confidence": info.get("inlier_ratio"),
            "hold_s": self.alignment_hold_s, "dx": translation[0], "dy": translation[1],
            "references": 1 + len(self._extra_aligners), "reference_index": info.get("reference_index", 0),
            "learning": self._learning is not None or self._pending_lighting,
            "lighting_status": self._lighting_status}
        self._consider_learning(frame, aligned if accepted else None, info, now)
        if self._pending_lighting:
            # Operator request while OFF: the verified pose places this view in
            # zone coordinates even when no reference matches the current light.
            self._pending_lighting = False
            candidate = None
            if not self.armed and self._learning is None:
                try:
                    image = self.aligner.warp(frame, self._alignment_prior())
                    candidate = self._build_aligner(np.ascontiguousarray(image)) if image is not None else None
                except Exception:
                    candidate = None
            if candidate is None:
                self._lighting_status = "Not enough distinct landmarks in the current view; nothing was saved"
            else:
                self._learning, self._last_learn_at = (candidate, 0, 0), now
        return aligned

    # -- arming ----------------------------------------------------------
    def _load_armed(self) -> bool:
        try:
            value = json.loads(ARMED_FILE.read_text())["armed"]
            return value if isinstance(value, bool) else False
        except (OSError, ValueError, KeyError, TypeError):
            return False         # Initial setup stays quiet until explicitly armed.

    def set_armed(self, value: bool, via: str = "api") -> bool:
        """Returns the new state. Disarming silences everything immediately."""
        with self._state_lock:
            return self._set_armed_locked(value, via)

    def _set_armed_locked(self, value: bool, via: str) -> bool:
        value = bool(value)
        if value == self.armed:
            if self.control_warning:
                self._save_armed_locked(value, via)
            return self.armed
        self.armed = value
        self._ack_until = 0.0
        self._resume_alarm_check = False
        self.acked = False
        self._fault_kind = None
        self.fault_reason = ""
        if not value:
            # Going quiet: kill the siren and drop any latched alarm outright.
            self.alarm = None
            self.alarm_id = None
            self.alarm_reason = ""
            self.acked = False
            self.fault_reason = ""
            self.buzzer.set(None)
        else:
            # Coming back: ask the detector thread to rebuild its background at
            # a safe point. This runs on the web/hotkey thread, so it must NOT
            # touch self.bg or the zone state directly -- doing so mid-frame
            # races the detector and crashes it.
            self._pending_rearm = True
        self._save_armed_locked(value, via)
        self.log_event("armed" if value else "disarmed", f"via {via}")
        return self.armed

    def _save_armed_locked(self, value: bool, via: str) -> None:
        try:
            # Replace atomically: an interrupted write must not corrupt the
            # persisted switch position. Silence above precedes disk I/O.
            temp = ARMED_FILE.with_suffix(".tmp")
            temp.write_text(json.dumps({"armed": value,
                                       "changed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                       "via": via}) + "\n")
            ps.harden(temp)
            temp.replace(ARMED_FILE)
            self.control_warning = ""
        except OSError:
            self.control_warning = ("The current ON/OFF setting could not be saved. "
                                    "A restart may restore the previous setting. Check storage and file permissions.")

    def toggle_armed(self, via: str = "api") -> bool:
        with self._state_lock:
            return self._set_armed_locked(not self.armed, via=via)

    # -- events ----------------------------------------------------------
    def log_event(self, kind: str, detail: str, **extra) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "detail": detail,
                  "source": safe_source(self.args.source), **extra}
        try:
            with EVENT_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            # Losing a diagnostic log must never stop detection or silence control.
            pass

    def raise_alarm(self, kind: str, reason: str) -> bool:
        with self._state_lock:
            return self._raise_alarm_locked(kind, reason)

    def _expire_ack_locked(self, now: float) -> None:
        if self._ack_until and now >= self._ack_until:
            self._ack_until = 0.0
            self.acked = False
            # Recheck current conditions after the quiet period; never replay
            # an event recorded while the user asked for silence.
            self._resume_alarm_check = True
            self._fault_kind = None

    def _raise_alarm_locked(self, kind: str, reason: str) -> bool:
        now = time.monotonic()
        self._expire_ack_locked(now)
        if not self.armed or (self._pending_rearm and kind != "fault"):
            # Disarmed means silent: no tone, no notification, no event.
            return False
        if now < self._ack_until:
            return False
        rank = {"early": 1, "main": 2, "fault": 3}
        if self.alarm and not self.acked and rank[kind] <= rank.get(self.alarm, 0):
            return False
        self.alarm = kind
        self.alarm_reason = reason
        self.alarm_since = now
        self._alarm_sequence += 1
        self.alarm_id = f"{self._alarm_session}:{self._alarm_sequence}"
        self.acked = False
        self.buzzer.set(kind)
        snapshot = self._queue_snapshot(kind)
        self.log_event(kind, reason, alarm_id=self.alarm_id, snapshot=snapshot)
        self._remember_alert(time.strftime("%Y-%m-%dT%H:%M:%S"), kind, reason, self.alarm_id, snapshot)
        if kind != "early":
            notify(f"C500 watch: {kind.upper()}", reason)
        return True

    def acknowledge(self, alarm_id: str | None = None, via: str = "api") -> bool:
        with self._state_lock:
            return self._acknowledge_locked(alarm_id, via)

    def _acknowledge_locked(self, alarm_id: str | None = None, via: str = "api") -> bool:
        if not self.alarm or (alarm_id is not None and alarm_id != self.alarm_id):
            return False
        if via.startswith("input-") and time.monotonic() - self.alarm_since < self.input_ack_min_s:
            return False
        self._ack_until = time.monotonic() + self.ack_quiet_s
        acknowledged_reason, acknowledged_id = self.alarm_reason, self.alarm_id
        waited = time.monotonic() - self.alarm_since
        self.acked = True
        self.alarm = None
        self.alarm_id = None
        self.alarm_reason = ""
        self.buzzer.set(None)
        # Silencing must precede potentially slow storage or diagnostics.
        self._mark_alert_acknowledged(acknowledged_id, via, waited)
        self.log_event("ack", f"via {via}: {acknowledged_reason}",
                       alarm_id=acknowledged_id, after_s=round(waited, 1))
        return True

    def input_status(self) -> dict:
        """Small, camera-free response for the Windows input helper."""
        with self._state_lock:
            return {"armed": self.armed, "alarm_id": self.alarm_id if self.alarm else None,
                    "alarm_started_at": self.alarm_since if self.alarm else None,
                    "ack_remaining_s": round(max(0., self._ack_until-time.monotonic()), 2)}

    def note_input_helper(self) -> None:
        """The keyboard/mouse helper polls about five times a second."""
        self._input_helper_seen = time.monotonic()

    def _input_helper_status(self) -> dict:
        seen = self._input_helper_seen
        age = None if seen is None else time.monotonic() - seen
        return {"connected": age is not None and age < 5, "age_s": None if age is None else round(age, 1)}

    # -- alert history ---------------------------------------------------
    def _remember_alert(self, ts: str, kind: str, reason: str, alarm_id, snapshot, acked=None) -> None:
        date, _, clock = str(ts).partition("T")
        self.recent_alerts.appendleft({"id": alarm_id, "date": date, "time": clock, "kind": kind,
                                       "reason": reason, "snapshot": snapshot, "acked": acked})
        self.alerts_version += 1

    def _mark_alert_acknowledged(self, alarm_id, via: str, after_s: float) -> None:
        for alert in self.recent_alerts:
            if alarm_id and alert["id"] == alarm_id:
                alert["acked"] = {"via": via, "after_s": round(after_s, 1)}
                self.alerts_version += 1
                return

    def alerts(self) -> dict:
        with self._state_lock:
            return {"version": self.alerts_version,
                    "alerts": [{**alert, "acked": dict(alert["acked"]) if alert["acked"] else None}
                               for alert in self.recent_alerts]}

    def _load_recent_alerts(self) -> None:
        """After a restart, rebuild the dashboard's alarm history from the log tail."""
        try:
            with EVENT_LOG.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                start = max(0, handle.tell() - 256 * 1024)
                handle.seek(start)
                lines = handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return
        acks = {}
        for line in lines[1:] if start else lines:   # A mid-file start is a partial line.
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            kind = record.get("kind")
            if kind in ("early", "main", "fault"):
                snapshot = record.get("snapshot")
                if not (isinstance(snapshot, str) and ALERT_NAME.match(snapshot)
                        and (ALERT_DIR / snapshot).is_file()):
                    snapshot = None
                alarm_id = record.get("alarm_id") if isinstance(record.get("alarm_id"), str) else None
                self._remember_alert(record.get("ts", ""), kind, str(record.get("detail", "")),
                                     alarm_id, snapshot)
            elif kind == "ack" and isinstance(record.get("alarm_id"), str):
                acks[record["alarm_id"]] = record
        for alert in self.recent_alerts:
            ack = acks.get(alert["id"]) if alert["id"] else None
            if ack:
                via = str(ack.get("detail", "")).partition(":")[0].removeprefix("via ")
                after = ack.get("after_s")
                alert["acked"] = {"via": via, "after_s": after if isinstance(after, (int, float)) else None}

    def _queue_snapshot(self, kind: str) -> str | None:
        """Keep the picture that raised an alarm; disk I/O stays off this path."""
        with self._preview_lock:
            jpeg = self.preview_jpeg
        if not jpeg:
            return None
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._alarm_session[:6]}{self._alarm_sequence}-{kind}.jpg"
        try:
            self._snapshots.put_nowait((name, jpeg))
        except queue.Full:
            return None
        if self._snapshot_writer is None:
            self._snapshot_writer = threading.Thread(target=self._write_snapshots, daemon=True,
                                                     name="alert-snapshots")
            self._snapshot_writer.start()
        return name

    def _write_snapshots(self) -> None:
        from zone_editor import _atomic_write
        while True:
            name, jpeg = self._snapshots.get()
            try:
                ALERT_DIR.mkdir(parents=True, exist_ok=True)
                _atomic_write(ALERT_DIR / name, jpeg)
                # Names begin with the timestamp, so name order is age order.
                for old in sorted(ALERT_DIR.glob("*.jpg"))[:-ALERT_KEEP]:
                    old.unlink(missing_ok=True)
            except OSError:
                pass   # A full disk must never interfere with alarms.
            finally:
                self._snapshots.task_done()

    # -- detection -------------------------------------------------------
    def process(self, frame: np.ndarray) -> None:
        with self._state_lock:
            self._expire_ack_locked(time.monotonic())
            resume_alarm_check = self._resume_alarm_check
            self._resume_alarm_check = False
            rearm = self._pending_rearm
            self._pending_rearm = False
            if self._pending_zones is not None:
                self.zones = self._pending_zones
                self._pending_zones = None
                if self.semantic:
                    self.semantic.zones = self.zones
                rearm = True
        if resume_alarm_check and self.semantic:
            # Require fresh model observations after the quiet interval.
            self.semantic.reset()
        if rearm:
            if self.semantic:
                self.semantic.reset()
            # Consume the re-arm request here, on the detector thread only.
            self.bg = None
            for zone in self.zones:
                zone.motion_active = zone.light_active = False
                zone._counts = {k: 0 for k in zone._counts}
        gray = (frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114)
        if self.bg is None:
            self.bg = gray.copy()
            self.warmup_started = time.monotonic()
            return
        diff = compensated_motion_diff(gray, self.bg)
        warm = (time.monotonic() - self.warmup_started) < self.args.warmup
        if self.semantic:
            for kind, reason in self.semantic.update(frame, self.armed and not warm):
                self.raise_alarm(kind, reason)

        for zone in self.zones:
            ys, xs = zone.slice_for(gray.shape[1], gray.shape[0])
            sub_diff = diff[ys, xs]
            area = max(1, sub_diff.size)
            zone.last_pct = float((sub_diff > self.args.pixel_threshold).sum()) * 100.0 / area
            zone.last_delta = float(gray[ys, xs].mean() - self.bg[ys, xs].mean())

            light_hit = zone.light and zone.last_delta >= zone.light_delta
            motion_hit = zone.motion and zone.last_pct >= zone.motion_pct
            if warm:
                # Warmup must not consume an arrival's only rising edge.
                zone.motion_active = zone.light_active = False
                zone._counts = {k: 0 for k in zone._counts}
                continue
            light_edge = zone.step("light", light_hit)
            motion_edge = zone.step("motion", motion_hit)
            # During ACK quiet time edges remain observable but cannot sound.
            # At expiry, only a condition still present can alarm again.
            if light_edge == "rise" or (resume_alarm_check and light_hit and zone.light_active):
                self.raise_alarm(zone.light_severity,
                                 f"{zone.name}: brightness rose {zone.last_delta:+.0f}")
            if motion_edge == "rise" or (resume_alarm_check and motion_hit and zone.motion_active):
                self.raise_alarm(zone.severity,
                                 f"{zone.name}: {zone.last_pct:.1f}% of the zone changed")

        # Background adapts slowly, so a parked car stops re-triggering but the
        # latched alarm from its arrival stays up until a human clears it.
        alpha = self.args.adapt
        self.bg = (1.0 - alpha) * self.bg + alpha * gray

    def check_health(self, now: float) -> None:
        with self._state_lock:
            self._check_health_locked(now)

    def _check_health_locked(self, now: float) -> None:
        self._expire_ack_locked(time.monotonic())
        if not self.armed:
            self.fault_reason = ""
            self._fault_kind = None
            return
        gap = now - self.last_frame_at
        limit = self.args.lost_after if self.first_frame_seen else self.args.connect_grace
        if gap > limit:
            kind = "lost"
            self.fault_reason = (f"no usable frame for {gap:.0f}s"
                                 + ("" if self.first_frame_seen else " since startup"))
        elif self._frame_error_since and now - self._frame_error_since > 3:
            kind = "detector"
            self.fault_reason = f"{self._frame_error[:160]} (detection is not running)"
        elif self._alignment_blocked and self.first_frame_seen and self._alignment_bad_since and now - self._alignment_bad_since > 2:
            kind = "alignment"
            self.fault_reason = "Scene alignment lost; check the view and recalibrate while OFF"
        elif self.identical_since and now - self.identical_since > self.args.freeze_after:
            kind = "frozen"
            self.fault_reason = (f"byte-identical frames for "
                                 f"{now - self.identical_since:.0f}s (frozen feed)")
        elif self.dark_since and now - self.dark_since > self.args.black_after:
            kind = "black"
            self.fault_reason = f"picture black for {now - self.dark_since:.0f}s"
        else:
            self.fault_reason = ""
            self._fault_kind = None
            return
        # ACK silences this fault episode. Recovery, a different fault, or a
        # deliberate re-arm enables the next fault alarm; the status stays visible.
        if kind != self._fault_kind:
            self._fault_kind = kind
            if self.semantic:
                self.semantic.reset()
            self.raise_alarm("fault", self.fault_reason)

    # -- preview ---------------------------------------------------------
    def render_preview(self, frame: np.ndarray, *, raw: np.ndarray | None = None) -> None:
        image = Image.fromarray(frame)
        scene_buffer = BytesIO()
        image.save(scene_buffer, "JPEG", quality=85)
        raw_buffer = BytesIO()
        Image.fromarray(raw if raw is not None else frame).save(raw_buffer, "JPEG", quality=85)
        draw = ImageDraw.Draw(image)
        if self.semantic and not self._alignment_blocked:
            info = self.semantic.status()
            if info.get('age_s') is not None and info['age_s'] < 3:
                for obj in info.get('detections', []):
                    draw.rectangle(obj['bbox'], outline=(75, 190, 255), width=2)
                    draw.text((obj['bbox'][0], max(15, obj['bbox'][1]-12)),
                              f"{obj['label']} {obj['confidence']:.0%}", fill=(75, 190, 255))
        colours = [(66, 205, 190), (246, 190, 84), (123, 166, 255)]
        for index, zone in enumerate(self.zones):
            x = zone.rect[0] * PROC_W
            y = zone.rect[1] * PROC_H
            w = max(1, zone.rect[2] * PROC_W)
            h = max(1, zone.rect[3] * PROC_H)
            # Unverified alignment: grey outlines show where the zones would be.
            colour = ((150, 150, 150) if self._alignment_blocked else
                      (255, 80, 95) if zone.active else colours[index % len(colours)])
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=colour, width=2)
            label = f"{index+1}  {zone.name}"
            label_y = max(18, y-15)
            box = draw.textbbox((x+3, label_y), label)
            draw.rectangle([box[0]-3,box[1]-2,box[2]+3,box[3]+2], fill=(12, 21, 29))
            draw.text((x + 3, label_y), label, fill=colour)
        # Leave the original camera timestamp visible. A newly painted PC
        # clock must never make delayed source video appear current.
        if self._alignment_blocked:
            draw.rectangle([0, PROC_H-23, PROC_W, PROC_H], fill=(118, 66, 14))
            draw.text((7,PROC_H-18), "ALIGNMENT NEEDS ATTENTION - scene alerts paused", fill="white")
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=85)
        with self._preview_lock:
            self.preview_jpeg = buffer.getvalue()
            self.scene_jpeg = scene_buffer.getvalue()
            self.raw_jpeg = raw_buffer.getvalue()

    def snapshot_preview(self, kind="preview") -> bytes | None:
        with self._preview_lock:
            return {"preview": self.preview_jpeg, "scene": self.scene_jpeg, "raw": self.raw_jpeg}.get(kind)

    def state_dict(self) -> dict:
        with self._state_lock:
            return self._state_dict_locked()

    def _state_dict_locked(self) -> dict:
        return {
            "armed": self.armed,
            "control_warning": self.control_warning,
            "alarm": self.alarm,
            "alarm_id": self.alarm_id,
            "reason": self.alarm_reason,
            "acked": self.acked,
            "ack_quiet_s": self.ack_quiet_s,
            "input_ack_after_s": self.input_ack_min_s,
            "ack_remaining_s": round(max(0., self._ack_until-time.monotonic()), 2),
            "source": safe_source(self.args.source),
            "ml": self.semantic.status() if self.semantic else {
                "available": False, "state": "unavailable", "error": "Local model not installed"},
            "fault_reason": self.fault_reason,
            "frames": self.frame_count,
            "restarts": self.reader.restarts,
            "fps_target": self.args.fps,
            "alignment": dict(self.alignment),
            "end_to_end_delay_s": None,
            "local_pipeline_ms": round(self.local_pipeline_ms, 1),
            "processing_ms": round(self.processing_ms, 1),
            "dropped_frames": self.reader.dropped_frames,
            "warmup_remaining_s": round(max(0, self.args.warmup - (time.monotonic()-self.warmup_started)), 1) if self.armed and self.warmup_started else 0,
            "sleep_prevention": {"active": self.sleep_inhibitor.active,
                                 "detail": self.sleep_inhibitor.detail},
            "last_frame_age": round(time.monotonic() - self.last_frame_at, 1),
            "loop_age_s": round(time.monotonic() - self._loop_heartbeat, 1),
            "audio": dict(self.audio),
            "input_helper": self._input_helper_status(),
            "alerts_version": self.alerts_version,
            "server_time": time.strftime("%H:%M:%S"),
            "server_date": time.strftime("%Y-%m-%d"),
            "last_frame_clock": self.last_frame_clock,
            "uptime_s": int(time.monotonic() - self.started),
            "zones": [{"name": z.name,
                       "motion_active": z.motion_active,
                       "light_active": z.light_active,
                       "changed_pct": round(z.last_pct, 2),
                       "light_delta": round(z.last_delta, 1),
                       "motion_raises": z.severity if z.motion else None,
                       "light_raises": z.light_severity if z.light else None}
                      for z in self.zones],
            "ffmpeg_tail": self.reader.stderr_tail[-3:],
        }

    # -- main loop -------------------------------------------------------
    def _update_awake(self) -> None:
        # Windows execution-state requests belong to the thread making them.
        # Apply/release only on the detector loop, never on the HTTP thread.
        if self.armed and not self.sleep_inhibitor.active:
            self.sleep_inhibitor.__enter__()
        elif not self.armed and self.sleep_inhibitor.active:
            self.sleep_inhibitor.__exit__()

    def _note_error(self, stage: str, exc: Exception, detection: bool = True) -> None:
        """Keep running through one bad frame; a persistent error becomes a fault."""
        now = time.monotonic()
        if detection:
            self._frame_error = f"{stage} error: {type(exc).__name__}: {exc}"
            self._frame_error_since = self._frame_error_since or now
        if now - self._last_error_print >= 60 and sys.stdout:
            self._last_error_print = now
            print(f"\n{stage} error (continuing):\n{traceback.format_exc()}", flush=True)

    def _health_loop(self) -> None:
        next_audio = 0.0
        while self._running:
            now = time.monotonic()
            if now >= next_audio:
                self.audio = self._read_audio()
                next_audio = now + 15.0
            if self._watchdog_expired(now):
                return
            time.sleep(1.0)

    @staticmethod
    def _read_audio() -> dict:
        try:
            return ps.audio_output_status()
        except Exception as exc:
            return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}

    def _watchdog_expired(self, now: float) -> bool:
        """A detector loop silent for LOOP_STALL_S is handed to the supervisor."""
        stalled = now - self._loop_heartbeat
        if stalled <= LOOP_STALL_S or not self._running:
            return False
        # The stuck thread may hold the state lock, so use the buzzer directly.
        if self.armed:
            self.buzzer.set("fault")
        self.log_event("fault", f"detector loop stalled for {stalled:.0f}s; restarting the watcher")
        if sys.stdout:
            print(f"\nDetector loop stalled for {stalled:.0f}s; exiting so the supervisor restarts it.",
                  flush=True)
        time.sleep(5.0 if self.armed else 0.0)
        os._exit(3)
        return True

    def run(self) -> None:
        self._loop_heartbeat = time.monotonic()
        self.reader.start()
        threading.Thread(target=self._health_loop, daemon=True, name="health").start()
        threading.Thread(target=self._stdin_loop, daemon=True).start()
        if self.args.web:
            start_web(self, self.args.web)
        print(f"{BOLD}c500-watch{RESET}  source={safe_source(self.args.source)}  zones={len(self.zones)}"
              f"  fps={self.args.fps}")
        if self.args.web:
            print(f"Live preview and ACK button: http://127.0.0.1:{self.args.web}")
        print("Press ANY KEY to acknowledge an alarm. Ctrl-C to stop.\n"
              if sys.stdin and sys.stdin.isatty() else
              "Not a terminal: acknowledge from the web page. Ctrl-C to stop.\n")

        last_draw = 0.0
        while self._running:
            self._loop_heartbeat = time.monotonic()
            self._update_awake()
            now = time.monotonic()
            try:
                received_at, buf = self.reader.frames.get(timeout=0.25)
            except queue.Empty:
                self.check_health(now)
                self.paint_status()
                continue

            frame = np.frombuffer(buf, dtype=np.uint8).reshape(PROC_H, PROC_W, 3)
            if self.first_frame_seen and received_at - self.last_frame_at >= self.args.lost_after:
                self._pending_rearm = True
                if self.semantic:
                    self.semantic.reset()
            self.frame_count += 1
            self.last_frame_at = received_at
            self.last_frame_clock = time.strftime("%H:%M:%S")
            self.first_frame_seen = True

            digest = hash(buf)
            if digest == self.last_hash:
                self.identical_since = self.identical_since or now
            else:
                self.identical_since = 0.0
            self.last_hash = digest

            if frame.mean() < self.args.black_level:
                self.dark_since = self.dark_since or now
            else:
                self.dark_since = 0.0

            processing_started = time.monotonic()
            aligned = None
            try:
                aligned = self.align_frame(frame)
                if aligned is not None:
                    self.process(aligned.astype(np.float32))
                elif self.semantic:
                    self.semantic.reset()
                self._frame_error_since = 0.0
            except Exception as exc:
                self._note_error("Detection", exc)
            self.processing_ms = (time.monotonic() - processing_started) * 1000
            self.check_health(now)

            if now - last_draw >= 1 / min(10.0, self.args.fps):
                try:
                    self.render_preview(aligned if aligned is not None else frame, raw=frame)
                except Exception as exc:
                    self._note_error("Preview", exc, detection=False)
                self.local_pipeline_ms = (time.monotonic() - received_at) * 1000
                self.paint_status()
                last_draw = now

    def paint_status(self) -> None:
        interactive = bool(sys.stdout) and getattr(sys.stdout, 'isatty', lambda: False)()
        now = time.monotonic()
        status_key = (self.armed, self.alarm, self._fault_kind, self.reader.restarts)
        if not interactive:
            if status_key == self._last_status_key and now - self._last_status_log < 60:
                return
            self._last_status_key = status_key
            self._last_status_log = now
        if not self.armed:
            tag = f"{BLUE} OFF   {RESET}"
        elif self.alarm == "fault" or self.fault_reason:
            tag = f"{BLUE} FAULT {RESET}"
        elif self.alarm == "main":
            tag = f"{RED} ALARM {RESET}"
        elif self.alarm == "early":
            tag = f"{YELLOW} EARLY {RESET}"
        elif self.acked:
            tag = f"{GREEN} ACKED {RESET}"
        else:
            tag = f"{GREEN}  OK   {RESET}"
        zones = "  ".join(
            f"{'*' if z.active else ' '}{z.name}:{z.last_pct:5.1f}%/{z.last_delta:+5.0f}"
            for z in self.zones)
        line = (f"\r{tag} f={self.frame_count:<7d} r={self.reader.restarts} "
                f"{zones}  {self.alarm_reason[:48]:<48}")
        if sys.stdout:
            sys.stdout.write(line[:240] if interactive else line.strip()[:240] + '\n')
            sys.stdout.flush()

    def _stdin_loop(self) -> None:
        """Any single keypress acknowledges. Ctrl-C still quits."""
        with ps.KeyReader() as keys:
            self._keys = keys
            if not keys.usable:
                print(f"  keyboard acknowledge unavailable ({keys.reason})")
                return
            while self._running:
                if not keys.read_key():
                    break
                self._ack_from_keyboard()

    def _ack_from_keyboard(self) -> None:
        if not self.armed:
            print(f"\n{BLUE} system is OFF {RESET} nothing to acknowledge.")
            return
        self.acknowledge(via="terminal")
        print(f"\n{GREEN} acknowledged {RESET} alarms quiet for five seconds; detection continues.")

    def shutdown(self) -> None:
        self._running = False
        self.sleep_inhibitor.__exit__()
        if getattr(self, "_keys", None):
            self._keys.restore()
        self.buzzer.shutdown()
        self.reader.stop()
        if self.semantic:
            self.semantic.stop()


# --------------------------------------------------------------------------
# Tiny local web view (127.0.0.1 only)
# --------------------------------------------------------------------------

from dashboard_ui import PAGE


def start_web(watcher: Watcher, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):
                return self._send(403, 'text/plain', b'Local access only')
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/zones":
                from zone_editor import read_config
                from dashboard_ui import zone_editor_page
                self._send(200, "text/html; charset=utf-8", zone_editor_page(read_config(Path(watcher.args.zones))).encode())
            elif self.path.split('?')[0] in ("/frame.jpg", "/scene.jpg", "/raw.jpg"):
                kind = {"/scene.jpg": "scene", "/raw.jpg": "raw"}.get(self.path.split('?')[0], "preview")
                jpeg = watcher.snapshot_preview(kind)
                if jpeg:
                    self._send(200, "image/jpeg", jpeg)
                else:
                    self._send(503, "text/plain", b"no frame yet")
            elif self.path == "/state":
                self._send(200, "application/json",
                           json.dumps(watcher.state_dict()).encode())
            elif self.path == "/input-status":
                watcher.note_input_helper()
                self._send(200, "application/json", json.dumps(watcher.input_status()).encode())
            elif self.path == "/alerts":
                self._send(200, "application/json", json.dumps(watcher.alerts()).encode())
            elif self.path.startswith("/alerts/"):
                name = self.path[len("/alerts/"):].split("?")[0]
                path = ALERT_DIR / name
                if ALERT_NAME.match(name) and path.is_file():
                    self._send(200, "image/jpeg", path.read_bytes())
                else:
                    self._send(404, "text/plain", b"not found")
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            origin = self.headers.get('Origin')
            host = self.headers.get('Host', '')
            if host.split(':')[0] not in ('127.0.0.1', 'localhost') or (origin and origin != 'http://' + host):
                return self._send(403, 'text/plain', b'Local access only')
            if self.headers.get('X-Via') not in ('web', 'ctl', 'hotkey', 'zones', 'zone-editor', 'input-any-key', 'input-mouse'):
                return self._send(403, 'text/plain', b'Local control header required')
            if self.path == "/alignment/add-lighting":
                try:
                    watcher.request_lighting_reference()
                    return self._send(200, 'application/json', b'{"ok":true,"pending":true}')
                except PermissionError as exc:
                    return self._send(409, 'application/json', json.dumps({'error':str(exc)}).encode())
                except ValueError as exc:
                    return self._send(400, 'application/json', json.dumps({'error':str(exc)}).encode())
            if self.path == "/alignment/calibrate":
                try:
                    watcher.request_calibration()
                    return self._send(200, 'application/json', b'{"ok":true,"pending":true}')
                except PermissionError as exc:
                    return self._send(409, 'application/json', json.dumps({'error':str(exc)}).encode())
                except ValueError as exc:
                    return self._send(400, 'application/json', json.dumps({'error':str(exc)}).encode())
            if self.path == "/zones":
                from zone_editor import save_config
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if size < 1 or size > 65536:
                        raise ValueError('Invalid request size')
                    payload = json.loads(self.rfile.read(size))
                    with watcher._state_lock:
                        data = save_config(Path(watcher.args.zones), payload, armed=watcher.armed)
                        watcher._pending_zones = [Zone(**{**item, 'rect': tuple(item['rect'])}) for item in data['zones']]
                    return self._send(200, 'application/json', b'{"ok":true}')
                except (PermissionError, ValueError, TypeError, OSError) as exc:
                    return self._send(400, 'application/json', json.dumps({'error':str(exc)}).encode())
            if self.path == "/ack":
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if size < 0 or size > 256:
                        raise ValueError('Invalid acknowledgement size')
                    alarm_id = None
                    if size:
                        payload = json.loads(self.rfile.read(size))
                        if not isinstance(payload, dict) or not isinstance(payload.get('alarm_id'), str) or not 1 <= len(payload['alarm_id']) <= 128:
                            raise ValueError('An alarm identifier is required')
                        alarm_id = payload['alarm_id']
                    elif self.headers.get('X-Via', '').startswith('input-'):
                        raise ValueError('An alarm identifier is required for input acknowledgement')
                    accepted = watcher.acknowledge(alarm_id, via=self.headers.get('X-Via', 'api'))
                    self._send(200, 'application/json', json.dumps({
                        'ok': True, 'acknowledged': accepted, **watcher.input_status()}).encode())
                except (ValueError, TypeError) as exc:
                    self._send(400, 'application/json', json.dumps({'error': str(exc)}).encode())
            elif self.path in ("/arm", "/disarm", "/toggle"):
                via = self.headers.get("X-Via", "web")
                if self.path == "/toggle":
                    state = watcher.toggle_armed(via=via)
                else:
                    state = watcher.set_armed(self.path == "/arm", via=via)
                self._send(200, "application/json",
                           json.dumps({"ok": True, "armed": state, "control_warning": watcher.control_warning}).encode())
            else:
                self._send(404, "text/plain", b"not found")

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # A browser abandoning an image request is routine, not a fault;
            # printing its traceback only buries real errors in the log.
            if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    server = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_sources(_args) -> None:
    print("AVFoundation devices on this Mac (use the index with cam:N or screen:N):\n")
    proc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-f", "avfoundation",
                           "-list_devices", "true", "-i", ""],
                          capture_output=True, text=True, **ps.background_subprocess_kwargs())
    for line in (proc.stderr or "").splitlines():
        if "] [" in line or "devices:" in line:
            print("  " + line.split("] ", 1)[-1])
    print("\nOther sources: rtsp://user:pass@host/path   http://...   /path/to/clip.mp4   selftest")


def cmd_snapshot(args) -> None:
    """Grab one processed frame so zones can be measured against it."""
    cmd = [c for c in build_command(args.source, args.fps) if c != "-re"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **ps.background_subprocess_kwargs())
    deadline = time.time() + args.timeout
    last = None
    seen = 0
    # Keep the newest frame we managed to read: cameras and screen capture need
    # a moment to settle, but a short clip may end before `skip` frames arrive.
    while time.time() < deadline and seen <= args.skip:
        buf = b""
        while len(buf) < FRAME_BYTES and time.time() < deadline:
            chunk = proc.stdout.read(FRAME_BYTES - len(buf))
            if not chunk:
                break
            buf += chunk
        if len(buf) != FRAME_BYTES:
            break
        last = buf
        seen += 1
    proc.terminate()
    if last is None:
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        err = err.replace(args.source, safe_source(args.source))
        sys.exit(f"Could not read a frame from {safe_source(args.source)}.\n{err or '(ffmpeg said nothing)'}")
    frame = np.frombuffer(last, dtype=np.uint8).reshape(PROC_H, PROC_W, 3)
    image = Image.fromarray(frame).resize((PROC_W * 2, PROC_H * 2), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    # Decade grid in normalised coordinates, which is what zones.json uses.
    for i in range(1, 10):
        draw.line([(i / 10 * image.width, 0), (i / 10 * image.width, image.height)],
                  fill=(255, 200, 0), width=1)
        draw.line([(0, i / 10 * image.height), (image.width, i / 10 * image.height)],
                  fill=(255, 200, 0), width=1)
        draw.text((i / 10 * image.width + 2, 2), f"{i/10:.1f}", fill=(255, 220, 0))
        draw.text((2, i / 10 * image.height + 2), f"{i/10:.1f}", fill=(255, 220, 0))
    out = Path(args.out).expanduser()
    image.save(out)
    print(f"Wrote {out} after {seen} frame(s) "
          f"({image.width}x{image.height}, grid labelled in 0-1 zone units)")


def cmd_test_alarm(args) -> None:
    sounds = ensure_sounds()
    buzzer = Buzzer(sounds, muted=False)
    for kind in ("early", "main", "fault"):
        print(f"Playing {kind} tone for {args.seconds}s ...")
        buzzer.set(kind)
        time.sleep(args.seconds)
    buzzer.set(None)
    buzzer.shutdown()
    print("Alarm test finished. If you heard nothing, raise the volume or check the output device.")


class _Tee:
    """Write to a log file and (when there is one) the console."""

    def __init__(self, handle, console):
        self.handle, self.console = handle, console

    def write(self, text):
        try:
            self.handle.write(text)
            self.handle.flush()
        except (OSError, ValueError):
            pass
        if self.console:
            try:
                self.console.write(text)
                self.console.flush()
            except (OSError, ValueError):
                pass
        return len(text)

    def flush(self):
        for target in (self.handle, self.console):
            try:
                target and target.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return bool(self.console) and getattr(self.console, "isatty", lambda: False)()


def cmd_watch(args) -> None:
    if args.log:
        # Under Task Scheduler / launchd there is no console, so without this a
        # crash leaves nothing behind to diagnose.
        path = Path(args.log).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8", errors="replace")
        sys.stdout = _Tee(handle, sys.__stdout__)
        sys.stderr = _Tee(handle, sys.__stderr__)
        print(f"\n=== c500-guard watch started {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"source={safe_source(args.source)} ===")
    watcher = Watcher(args)

    def bye(*_a):
        watcher.shutdown()
        print("\nStopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)
    try:
        watcher.run()
    finally:
        watcher.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sources = sub.add_parser("sources", help="list capture devices on this Mac")
    sources.set_defaults(func=cmd_sources)

    snap = sub.add_parser("snapshot", help="save one frame with a 0-1 grid for drawing zones")
    snap.add_argument("source")
    snap.add_argument("--out", default=str(HERE / "snapshot.png"))
    snap.add_argument("--fps", type=float, default=5.0)
    snap.add_argument("--skip", type=int, default=8, help="frames to discard while the source settles")
    snap.add_argument("--timeout", type=float, default=25.0)
    snap.set_defaults(func=cmd_snapshot)

    test = sub.add_parser("test-alarm", help="play each tone; run this before every shift")
    test.add_argument("--seconds", type=float, default=3.0)
    test.set_defaults(func=cmd_test_alarm)

    watch = sub.add_parser("watch", help="monitor a source and sound the buzzer")
    watch.add_argument("source", help="rtsp://... | http://... | screen:4[@x,y,w,h] | cam:0 | file | selftest")
    watch.add_argument("--zones", default=str(DEFAULT_ZONES))
    watch.add_argument("--fps", type=float, default=10.0)
    watch.add_argument("--web", type=int, default=8787, help="local preview port, 0 to disable")
    watch.add_argument("--silent", action="store_true", help="visual only, no sound")
    watch.add_argument("--log", default="",
                       help="append console output to this file (use when running headless)")
    watch.add_argument("--pixel-threshold", type=float, default=22.0,
                       help="per-pixel luma change that counts as changed")
    watch.add_argument("--adapt", type=float, default=0.02,
                       help="background adaptation rate per frame (lower = longer memory)")
    watch.add_argument("--warmup", type=float, default=2.0,
                       help="seconds of silence while the background model settles")
    watch.add_argument("--lost-after", type=float, default=5.0)
    watch.add_argument("--connect-grace", type=float, default=30.0,
                       help="seconds to allow for the very first frame before calling FEED LOST")
    watch.add_argument("--freeze-after", type=float, default=20.0)
    watch.add_argument("--black-after", type=float, default=5.0)
    watch.add_argument("--black-level", type=float, default=6.0)
    watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
