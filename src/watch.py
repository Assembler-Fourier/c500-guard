#!/usr/bin/env python3
"""c500-watch - a local, source-agnostic change/light watcher with a latched buzzer.

It does not care where the picture comes from. Give it an RTSP/HTTP stream, a
macOS screen rectangle or a USB/AVFoundation camera and it will:

  * raise a latched MAIN alarm when something changes inside an armed zone,
  * raise a short EARLY warning when a zone suddenly gets brighter (headlights),
  * raise a distinct FAULT alarm when the feed is lost, frozen or black,

and keep sounding while an unanswered arrival remains. A calibrated empty-road
check can resolve that exact visit after service. Nothing leaves this machine.

Alarm-only. This program never operates a barrier or any other powered device.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
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
from urllib.parse import urlsplit, urlunsplit, parse_qs

import numpy as np
from PIL import Image, ImageDraw

import platform_support as ps
from guard_presence import GuardPresence, PresenceConfig
from power_control import request_full_off
from illumination import compensated_motion_diff, LightingMonitor
from region_geometry import rect_polygon, region_contains, polygon_mask
from approach_light import measure_local_brightness, measure_urgent_change
from alarm_audio import plan as gentle_audio_plan, ensure_sounds as ensure_gentle_sounds, limited_plan, level_ceiling, lane_profile
from visit_memory import VisitMemory
from pipeline_health import PipelineHealth
from barrier_workflow import BarrierWorkflow

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
FEEDBACK_DIR = STATE_DIR / "feedback"      # alerts the operator marked Wrong or Real
CROSS_LANE_S = 10.0                        # other-lane activity this recent labels an alarm
LET_IN_S = 20.0                            # the other lane's arm raised this recently: a car let in
MENU_BAND = (4, 14)                        # rows of the keyboard's "Live View" menu bar (640x360)
AI_SEEN_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
AI_SEEN_MIN_CONFIDENCE = 0.4
FEEDBACK_KEEP = 1500                       # examples kept per verdict
CAR_MEMORY_WAITING = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                      "<title>Vehicles</title><body style='background:#090c10;color:#eef4fb;font:15px Segoe UI,Arial,sans-serif;padding:32px'>"
                      "<h1>Vehicles</h1><p style='color:#96a2b2'>The first local history report is being prepared. "
                      "It will appear here automatically when ready.</p>"
                      "<script>let requested=false;window.addEventListener('message',event=>{"
                      "if(!requested&&event.source===parent&&event.origin===location.origin&&"
                      "event.data?.type==='barrierwatcher:report-refresh'&&typeof event.data.reportVersion==='string'&&"
                      "event.data.reportVersion){requested=true;location.reload();}});</script>")
CAR_MEMORY_HEARTBEAT_S = 90.0
CAR_MEMORY_STATUS_BYTES = 4096
RECORDINGS_DIR = STATE_DIR / "recordings"  # a short video of every alarm
ALERT_KEEP = 300            # newest alarm snapshots kept on disk (about 15 MB)
MIN_FREE_BYTES = 500 * 1024 * 1024   # evidence pictures must never fill the disk
# The camera is the operator's own: when it is taken away the whole program
# should close rather than sit blind. Long enough that a reconnect, a decoder
# restart or a slow network cannot trigger it.
CAMERA_GONE_S = 120.0
# After the guard acknowledges, that one approach stays quiet this long: a
# vehicle crossing it trips several zones, and that is one visit, not four
# alarms. Every other approach, and every fault, still sounds immediately.
VISIT_QUIET_S = 45.0
LOOP_STALL_S = 20.0         # a detector loop silent this long is restarted
REACQUIRE_FIRST_S = 5.0     # nothing has aligned since startup: look almost at once
REACQUIRE_AFTER_S = 20.0    # alignment lost this long: look for a moved camera
REACQUIRE_EVERY_S = 20.0    # and try again no more often than this
ZOOM_REACQUIRE_S = 3.0      # a zoomable (PTZ) view is moved on purpose: look again within seconds
NOISE_K = 3.0               # a light change must exceed this many flicker units
INITIAL_NOISE = 3.0         # assumed flicker before anything is measured
MIN_NOISE, MAX_NOISE = 0.8, 40.0
NOISE_ADAPT_DIVISOR = 4     # flicker is learned four times slower than the background
MAX_LIGHTING_REFERENCES = 6  # learned references for other lighting (dusk, night, wet, dawn)
LEARN_AFTER_S = 60.0         # no strong landmark match for this long before learning
LEARN_EVERY_S = 600.0        # at most one learning attempt per ten minutes
BOOTSTRAP_AFTER_S = 8.0      # nothing has aligned since start: check the scene's structure
BOOTSTRAP_EVERY_S = 15.0     # and again no more often than this
BOOTSTRAP_MIN_RESPONSE = 0.25  # same scene scores ~0.45-0.5, other views <= 0.12
BOOTSTRAP_MAX_GAP_PX = 1.5   # the structure must sit where the saved pose says

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


class CaptureMissing(OSError):
    """The configured capture device is absent; the message is shown as-is."""


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

    elif source.startswith("usbcam:"):
        # usbcam:12ab:3456 - a USB capture device found by its USB id each time
        # it is opened, so the socket it sits in and the order Windows lists
        # cameras in cannot swap it for the desk webcam. MJPEG at 1280x720 is
        # decoded cheaply and still four times the 640x360 working size.
        found = ps.find_usb_camera(source[len("usbcam:"):])
        if found is None:
            raise CaptureMissing(f"Capture stick {source[len('usbcam:'):].upper()} is not plugged in "
                                 "or is not answering.")
        _index, device = found
        cmd += ["-f", "dshow", "-rtbufsize", "8M", "-vcodec", "mjpeg",
                "-video_size", "1280x720", "-framerate", str(max(10, int(round(fps)))),
                "-i", f"video={device['path']}"]
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
        # USB capture only: was the stick there at the last (re)start? None = not a USB source.
        self.device_present: bool | None = None
        self.device_checked_at = 0.0
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
            except CaptureMissing as exc:
                self.stderr_tail = [str(exc)]
                self.device_present, self.device_checked_at = False, time.monotonic()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            except (OSError, ValueError, SystemExit, subprocess.SubprocessError):
                self.stderr_tail = ["Could not start the stream decoder; check ffmpeg and source configuration."]
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            if self.source.startswith("usbcam:"):
                self.device_present, self.device_checked_at = True, time.monotonic()
            if ps.WINDOWS and getattr(proc, "_handle", None) is not None:
                ps.normal_priority(int(proc._handle))     # the decoder must not be paged out either
                ps.tie_to_this_process(proc)              # and must never outlive the watcher
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
    """An alert region with independent geometry, object classes and trigger source."""

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
    polygon: list | None = None
    classes: list | None = None
    trigger: str = "hybrid"
    exclude_polygons: list = field(default_factory=list)
    priority: bool = False
    route_group: str = ""
    ai_confirm_observations: int = 2
    ai_confirmation_window_s: float = 0.0
    ai_fast_confirm_confidence: float = 0.0
    approach_motion: bool = False
    ai_context: bool = False
    ai_confidence: float = 0.0
    arrival_direction: list | None = None
    arrival_min_progress: float = .012
    stationary_confirm_s: float = 3.0
    confirmation_polygon: list | None = None
    route_role: str = ""
    operator_hint: str = ""
    alert_stage: str = ""
    light_polygon: list | None = None
    strict_change: bool = False
    change_threshold: float = 6.0
    # Smallest single connected patch, as % of the zone, that counts as a
    # change. None keeps the original fixed 24/9-pixel patch floor. On a clean
    # capture feed, video-compression shimmer moves hundreds of scattered
    # pixels but never forms a car-sized patch.
    connected_pct: float | None = None
    approach_motion_active: bool = field(default=False, repr=False)
    approach_metrics: dict = field(default_factory=dict, repr=False)
    # trigger "taillight" (exit entry corner): red lamp share, changed share, and the change that wakes the AI.
    taillight_red_pct: float = 2.0
    taillight_change_pct: float = 10.0
    taillight_wake_pct: float = 5.0
    # A vehicle cut off by the bottom of the picture still counts as inside this lane.
    bottom_edge_entry: bool = False
    # Weaker detections count only for a vehicle seen moving in (None = off).
    ai_confidence_moving: float | None = None

    # runtime
    motion_active: bool = field(default=False, repr=False)
    light_active: bool = field(default=False, repr=False)
    _counts: dict = field(default_factory=lambda: {"motion_on": 0, "motion_off": 0,
                                                   "light_on": 0, "light_off": 0}, repr=False)
    last_pct: float = field(default=0.0, repr=False)
    last_delta: float = field(default=0.0, repr=False)
    _notified: dict = field(default_factory=lambda: {"motion": False, "light": False}, repr=False)
    _mask_cache: dict = field(default_factory=dict, repr=False)
    _light_mask_cache: dict = field(default_factory=dict, repr=False)
    light_metrics: dict = field(default_factory=dict, repr=False)
    change_metrics: dict = field(default_factory=dict, repr=False)
    _change_previous: tuple | None = field(default=None, repr=False)

    @property
    def ai_enabled(self):
        return self.trigger in ("ai", "ai_light") or (self.trigger == "hybrid" and self.motion)

    def contains_point(self, x, y):
        return region_contains(x, y, self.polygon or rect_polygon(self.rect), self.exclude_polygons)

    def mask_for(self, w, h):
        if (w, h) not in self._mask_cache:
            self._mask_cache.clear()
            self._mask_cache[(w, h)] = polygon_mask(w, h, self.polygon or rect_polygon(self.rect), self.exclude_polygons)
        return self._mask_cache[(w, h)]

    def light_mask_for(self, w, h):
        if self.light_polygon is None:
            return self.mask_for(w, h)
        if (w, h) not in self._light_mask_cache:
            self._light_mask_cache.clear()
            self._light_mask_cache[(w, h)] = (self.mask_for(w, h)
                & polygon_mask(w, h, self.light_polygon))
        return self._light_mask_cache[(w, h)]

    @property
    def active(self) -> bool:
        return self.motion_active or self.light_active

    def slice_for(self, w: int, h: int):
        x = min(int(self.rect[0] * w), w - 1)
        y = min(int(self.rect[1] * h), h - 1)
        end_x = min(w, max(x+1, math.ceil((self.rect[0]+self.rect[2])*w)))
        end_y = min(h, max(y+1, math.ceil((self.rect[1]+self.rect[3])*h)))
        return slice(y, end_y), slice(x, end_x)

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
            self._notified[channel] = False
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
    for item in (raw if isinstance(raw, list) else raw.get("zones", [])):
        item = dict(item)
        item["rect"] = tuple(float(v) for v in item["rect"])
        zones.append(Zone(**item))
    return zones or default_zones()


# --------------------------------------------------------------------------
# Alarm
# --------------------------------------------------------------------------

SOUND_VERSION = 4


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
    return ensure_gentle_sounds(SOUND_DIR, force=force)


class Buzzer:
    """Loops a tone until told to stop. Silent whenever the system is disarmed."""

    def __init__(self, sounds: dict[str, Path], muted: bool = False):
        self.sounds = sounds
        self.muted = muted
        self.player = ps.AudioPlayer()
        self._want: str | None = None
        self._profile: str | None = "soft"
        self._playing: Path | None = None
        self._lock = threading.Lock()

    def set(self, kind: str | None) -> None:
        with self._lock:
            if kind == self._want:
                return
            self._want = kind
            if kind is None or self.muted or self._profile == "silent":
                self.player.stop()
                self._playing = None
            else:
                selected = "fault" if kind == "fault" else (self._profile or "soft")
                self._playing = self.sounds.get(selected, self.sounds[kind])
                self.player.start_loop(self._playing)

    def set_profile(self, profile: str) -> None:
        """Change the sound without creating another logical alarm."""
        with self._lock:
            self._profile = profile
            if self._want is None or self.muted:
                return
            if profile == "silent":
                if self._playing is not None:
                    self.player.stop()
                    self._playing = None
                return
            target = self.sounds.get(profile, self.sounds[self._want])
            if target != self._playing:
                self._playing = target
                self.player.start_loop(target)

    def replace_sounds(self, sounds: dict[str, Path]) -> None:
        """New sound style: a playing alarm switches to it straight away."""
        with self._lock:
            self.sounds = sounds
            if self._want is None or self._playing is None or self.muted:
                return
            selected = "fault" if self._want == "fault" else (self._profile or "soft")
            self._playing = self.sounds.get(selected, self.sounds.get(self._want))
            if self._playing is not None:
                self.player.start_loop(self._playing)

    def shutdown(self) -> None:
        with self._lock:
            self._want = None
            self._playing = None
            self.player.stop()


class AlarmLoudness:
    """Current warnings gain strength; confirmed visits have the higher ceiling.

    Speaker restoration is serialized with every write, including ACK and OFF.
    Head pose can adjust pace but can never request an immediate loud sound.
    """

    CAPS = {"early": 0.75, "fault": 0.18, "main": 0.95, "barrier_open": 0.18}
    CONFIRMATION_HOLD_S = 2.0
    # How often the speaker level is re-asserted while an alarm sounds, even
    # if the target has not moved. Without it, an alarm parked at its cap
    # stops touching the mixer and a speaker lost then goes unnoticed.
    MIXER_VERIFY_S = 2.0

    def __init__(self, presence, *, enabled: bool = True, apply_fn=None, read_fn=None,
                 clock=time.monotonic, log=None, restore_path=None, profile_fn=None):
        self.presence = presence
        self.enabled = enabled
        self.restore_path = Path(restore_path) if restore_path else (STATE_DIR / "speaker-restore.json")
        self._apply = apply_fn or ps.set_audio_output
        self._read = read_fn or ps.audio_output_status
        self.clock = clock
        self._log = log
        self._profile_fn = profile_fn or (lambda profile: None)
        # ACK/pause and the detector tick may run on different threads.
        # Finish any in-flight volume write before restoring the speaker.
        self._lock = threading.RLock()
        self.saved: dict | None = None
        self.kind: str | None = None
        self.urgent = False
        self.gentle = False
        self.started = 0.0
        self.applied: float | None = None
        self.plan = None
        self.source = "unknown"
        self.lane = ""
        self.confirmed = False
        self.evidence_fresh = True
        self.event_present = True
        self._episode_at = None
        self._episode_presence = None
        self._progress_at = None
        self._confirmed_progress_s = 0.0
        self._confirmation_hold_until = 0.0
        self._confirmation_floor = None
        self._mixer_ready = False
        self._needs_restore = False
        self._next_mixer_retry = 0.0
        self.detail = "idle"

    def begin(self, kind: str, *, urgent: bool = False, gentle: bool = False,
              source: str = "unknown", confirmed: bool = False, lane: str = "") -> None:
        with self._lock:
            if not self.enabled:
                return
            self.kind = kind
            self.lane = lane if kind in ('main', 'early') and lane in ('entrance', 'exit') else ''
            self.urgent = bool(urgent)
            # Compatibility flags never bypass the fresh-evidence audio plan.
            self.gentle = bool(gentle) and not urgent
            self.source = source
            self.confirmed = bool(confirmed)
            self.evidence_fresh = self.event_present = True
            self.started = self.clock()
            self._reset_episode(self.started)
            self.applied = None
            self._mixer_ready = False
            self._next_mixer_retry = 0.0
            if self.saved is None:
                try:
                    current = self._read()
                except Exception:
                    current = {"available": False}
                try:
                    readable = (isinstance(current, dict) and current.get("available") is True
                                and not isinstance(current.get("volume"), bool)
                                and math.isfinite(float(current["volume"]))
                                and 0.0 <= float(current["volume"]) <= 1.0
                                and isinstance(current.get("muted"), bool))
                except (ValueError, TypeError, KeyError, OverflowError):
                    readable = False
                if not readable:
                    self.detail = "Speaker level unavailable; short soft fallback only"
                    self.tick()
                    return
                # A note on disk, so a watcher killed mid-alarm still gives the
                # guard's volume back at the next start. Without it, the level
                # the alarm raised becomes the level the next alarm "restores"
                # to, and the speakers stay at full for ever.
                self.saved = self._remembered() or {"volume": float(current["volume"]),
                                                    "muted": current.get("muted")}
                self._remember(self.saved)
            self.tick()

    def evidence(self, *, fresh: bool, present: bool, confirmed: bool, source: str | None = None) -> None:
        with self._lock:
            now = self.clock()
            self._advance_progress(now)
            active = bool(fresh and present)
            previously_active = bool(self.evidence_fresh and self.event_present)
            if not active:
                # An actual loss of evidence starts a new acoustic episode on
                # return. No head-pose or confidence fluctuation takes this path.
                self._reset_episode(None)
            elif not previously_active:
                self._reset_episode(now)
            elif self.confirmed and not confirmed:
                self._confirmation_hold_until = now + self.CONFIRMATION_HOLD_S
            if (active and confirmed and not self.confirmed and previously_active
                    and self._mixer_ready and self.applied is not None
                    and self.plan is not None and self.plan.audible):
                # Recognition enriching this same ongoing warning must not
                # reset the already audible image-change reminder to a chime.
                # Retain only a level we actually set, never a requested one.
                from dataclasses import replace
                self._confirmation_floor = replace(self.plan, level=self.applied)
            self.evidence_fresh = bool(fresh)
            self.event_present = bool(present)
            self.confirmed = bool(confirmed)
            if source:
                self.source = source

    def _reset_episode(self, now, *, presence_state=None) -> None:
        """Reset on a new alarm, real evidence loss, or lost speaker control."""
        self._episode_at = now
        self._episode_presence = presence_state
        self._progress_at = now
        self._confirmed_progress_s = 0.0
        self._confirmation_hold_until = 0.0
        self._confirmation_floor = None

    def _effective_confirmation(self, now: float) -> bool:
        return bool(self.kind is not None and self.evidence_fresh and self.event_present and
                    (self.confirmed or now < self._confirmation_hold_until))

    def _advance_progress(self, now: float) -> None:
        """Count confirmed time, including at most two seconds of image-backed gap.

        Longer confidence gaps pause this clock while their fresh image warning
        uses the lower ceiling. Recognition returning resumes earned progress;
        it cannot reset a continuously observed visit to the beginning forever.
        """
        previous = self._progress_at
        self._progress_at = now
        if (previous is None or not self._mixer_ready or self._episode_at is None
                or not self.evidence_fresh or not self.event_present):
            return
        through = now if self.confirmed else min(now, self._confirmation_hold_until)
        self._confirmed_progress_s += max(0.0, through - previous)

    def _remembered(self) -> dict | None:
        """The guard's own level from an alarm that never got to finish."""
        try:
            data = json.loads(self.restore_path.read_text(encoding="utf-8"))
            volume = float(data["volume"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not 0.0 <= volume <= 1.0:
            return None
        return {"volume": volume, "muted": bool(data.get("muted", False))}

    def _remember(self, saved: dict | None) -> None:
        try:
            if saved is None:
                self.restore_path.unlink(missing_ok=True)
                return
            self.restore_path.parent.mkdir(parents=True, exist_ok=True)
            self.restore_path.write_text(json.dumps(
                {"volume": saved.get("volume"), "muted": saved.get("muted"),
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        except OSError:
            pass                    # Losing the note costs a restore, never the alarm.

    def recover_previous(self) -> dict | None:
        """Called at startup: put back a volume an interrupted alarm left raised."""
        if not self.enabled:
            return None
        with self._lock:
            saved = self._remembered() or self.saved
            if saved is None:
                return None
            result = self._safe_apply(volume=saved.get("volume"), muted=saved.get("muted"))
            if not result["applied"]:
                self.saved = saved
                self._needs_restore = True
                self._remember(saved)
                self.detail = f"Speaker restore pending: {result['detail']}"
                return None
            self.saved = None
            self._needs_restore = False
            self._mixer_ready = False
            self._reset_episode(None)
            self._remember(None)
            self.detail = f"put back {round((saved.get('volume') or 0) * 100)}% after an interrupted alarm"
            return saved

    def _safe_apply(self, **values) -> dict:
        """Mixer failures are diagnostics, never permission to use a louder file."""
        try:
            result = self._apply(**values)
        except Exception as error:
            return {"applied": False, "detail": f"Mixer unavailable ({type(error).__name__})"}
        if not isinstance(result, dict):
            return {"applied": False, "detail": "Mixer returned no confirmed result"}
        return {"applied": result.get("applied") is True,
                "detail": str(result.get("detail") or "Mixer refused the change")}

    def _fallback_plan(self, elapsed: float, presence_state: str):
        """The fixed-duration soft asset remains bounded without mixer control."""
        return limited_plan(elapsed, kind=self.kind, source=self.source,
                            presence_state=presence_state, evidence_fresh=self.evidence_fresh,
                            event_present=self.event_present)

    def _select_profile(self, profile: str) -> None:
        # Keep the policy plan's base profile for strength comparisons; only
        # playback selects a lane-specific asset. Fault/fallback stay neutral.
        self._profile_fn(lane_profile(profile, self.lane))

    def tick(self) -> None:
        with self._lock:
            if not self.enabled or self.kind is None:
                return
            now = self.clock()
            elapsed = max(0.0, now - self.started)
            self._advance_progress(now)
            if self._episode_presence is None:
                try:
                    self._episode_presence = self.presence.plan(unacknowledged_s=elapsed).state
                except Exception:
                    self._episode_presence = "unknown"
            presence_state = self._episode_presence
            effective_confirmed = self._effective_confirmation(now)
            episode_age = max(0.0, now - self._episode_at) if self._episode_at is not None else 0.0
            acoustic_age = self._confirmed_progress_s if effective_confirmed else episode_age
            if not self._mixer_ready:
                # Reacquiring mixer control must begin softly even if the
                # incident has been visible for minutes during a device fault,
                # including an unconfirmed image warning.
                acoustic_age = 0.0
            plan = gentle_audio_plan(acoustic_age, kind=self.kind, source=self.source,
                    presence_state=presence_state, evidence_fresh=self.evidence_fresh,
                    event_present=self.event_present,
                    confirmed=effective_confirmed and self.saved is not None,
                    urgent=self.urgent and self.saved is not None)
            floor = self._confirmation_floor
            if (plan.audible and effective_confirmed and self._mixer_ready
                    and self.saved is not None and floor is not None):
                from dataclasses import replace
                strength = {"silent": -1, "limited": 0, "soft": 1, "notice": 2, "wake": 3}
                if strength.get(floor.profile, -1) > strength.get(plan.profile, -1):
                    plan = replace(plan, profile=floor.profile, repeat_s=floor.repeat_s,
                                   burst_s=floor.burst_s, stage="reminder")
                plan = replace(plan, level=max(plan.level, floor.level),
                               reason="Confirmed visit continues the current reminder")
            if self.saved is None:
                plan = self._fallback_plan(elapsed, presence_state)
            target = plan.level
            self.plan = plan
            if not plan.audible:
                # Stop playback before restoring a potentially high idle volume.
                self._select_profile("silent")
                self._mixer_ready = False
                self._reset_episode(None)
                if (self.saved is not None and self._needs_restore
                        and self.clock() >= self._next_mixer_retry):
                    result = self._safe_apply(volume=self.saved.get("volume"), muted=self.saved.get("muted"))
                    if result["applied"]:
                        self._needs_restore = False
                        self.applied = None
                    else:
                        self._next_mixer_retry = self.clock() + 1.0
                        self.detail = f"Silent; speaker restore pending: {result['detail']}"
                        return
                self.detail = plan.reason
                return
            if self.saved is None:
                # The short soft waveform remains bounded even without mixer access.
                self.plan = self._fallback_plan(elapsed, presence_state)
                self._select_profile(self.plan.profile)
                return
            if not self._mixer_ready and self.clock() < self._next_mixer_retry:
                fallback = self._fallback_plan(elapsed, presence_state)
                # The fallback is always the soft asset, never its weak-notice
                # variant: the operating system level is currently unverified.
                self.plan = fallback
                self._select_profile(fallback.profile)
                return
            ceiling = level_ceiling(kind=self.kind, confirmed=effective_confirmed,
                                    urgent=self.urgent)
            if abs(target-ceiling) < 1e-9:
                # A fractional fake/real tick can land just below a milestone.
                # Normalize the endpoint so its last write is exactly the cap.
                from dataclasses import replace
                target = ceiling
                plan = replace(plan, level=ceiling)
                self.plan = plan
            at_ceiling = target == ceiling and self.applied is not None and self.applied != ceiling
            # Once parked at the cap nothing else would write to the mixer, so
            # a speaker that disappears at full volume would never be noticed.
            verify_due = self.clock() >= getattr(self, "_next_mixer_verify", 0.0)
            if (not self._mixer_ready or self.applied is None
                    or abs(target - self.applied) >= 0.02 or at_ceiling or verify_due):
                previously_ready = self._mixer_ready
                self._needs_restore = True
                self._remember(self.saved)
                result = self._safe_apply(volume=target, muted=False)
                if result["applied"]:
                    self._mixer_ready = True
                    self._next_mixer_verify = self.clock() + self.MIXER_VERIFY_S
                    self.applied = target
                    if not previously_ready:
                        self._reset_episode(self.clock(), presence_state=presence_state)
                    self.detail = f"{round(target * 100)}% - {plan.reason}"
                else:
                    self._mixer_ready = False
                    self.applied = None
                    self._reset_episode(None)
                    self._next_mixer_retry = self.clock() + 1.0
                    fallback = self._fallback_plan(elapsed, presence_state)
                    self.plan = fallback
                    self._select_profile(fallback.profile)
                    self.detail = f"{result['detail']}; short soft fallback only"
                    return
            self._select_profile(plan.profile)

    def end(self) -> None:
        with self._lock:
            self._select_profile("silent")
            self.kind = None
            self.lane = ""
            self.urgent = False
            self.gentle = False
            self.plan = None
            saved = self.saved
            self.applied = None
            self._mixer_ready = False
            self._reset_episode(None)
            if saved is None:
                self.detail = "idle"
                return
            result = self._safe_apply(volume=saved.get("volume"), muted=saved.get("muted"))
            if not result["applied"]:
                self._needs_restore = True
                self._remember(saved)
                self.detail = f"Silent; speaker restore pending: {result['detail']}"
                return
            self.saved = None
            self._needs_restore = False
            self._remember(None)
            self.detail = f"restored to {round((saved.get('volume') or 0) * 100)}%"

    def status(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "active": self.kind is not None,
                    "level_pct": None if self.applied is None else round(self.applied * 100),
                    "restores_to_pct": None if self.saved is None else round((self.saved.get("volume") or 0) * 100),
                    "detail": self.detail,
                    "reason": None if self.plan is None else self.plan.reason,
                    "profile": None if self.plan is None else lane_profile(self.plan.profile, self.lane),
                    "base_profile": None if self.plan is None else self.plan.profile,
                    "lane": self.lane,
                    "pattern": {'entrance': 'Rising tones', 'exit': 'Falling tones'}.get(self.lane, 'Standard tone'),
                    "stage": None if self.plan is None else self.plan.stage,
                    "audible": bool(self.plan and self.plan.audible),
                    "confirmed": self.confirmed,
                    "effective_confirmed": self._effective_confirmation(self.clock()),
                    "confirmed_progress_s": round(self._confirmed_progress_s, 2),
                    "presence_grace_state": self._episode_presence,
                    "mixer_ready": self._mixer_ready,
                    "restore_pending": self._needs_restore and self.kind is None,
                    "maximum_pct": 95}


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
        # The chosen sound style decides which tones are written to disk.
        self.alarm_audio_settings = self._load_audio_settings(args.zones)
        self.sounds = ensure_sounds()
        self.buzzer = Buzzer(self.sounds, muted=args.silent)
        self.reader = FrameReader(args.source, args.fps, args.connect_grace, args.lost_after)

        self.bg: np.ndarray | None = None
        # How much each pixel normally flickers. Wind in foliage, rain and
        # decoder speckle raise their own alarm bar; steady tarmac keeps full
        # sensitivity. Learned continuously, never saved.
        self.bg_noise: np.ndarray | None = None
        self.started = time.monotonic()
        self.pipeline = PipelineHealth()
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
        self.alarm_details = {}
        self.alarm_since = 0.0
        self.alarm_id: str | None = None
        self._alarm_sequence = 0
        self._alarm_session = os.urandom(12).hex()
        self.ack_quiet_s = self._load_ack_quiet(args.zones)
        self.fault_remind_s = self._load_fault_remind(args.zones)
        self.soften_let_in = self._load_soften_let_in(args.zones)
        # How sharply an unanswered reminder climbs. Tunable in zones.json
        # ("alarms.escalation"), clamped there to values that still rise.
        self.alarm_audio_settings = self._load_audio_settings(args.zones)
        self._test_tone = None
        # Local face presence may slow the reminder pace. It never proves sleep.
        self.presence = GuardPresence(self._load_presence_config(args.zones))
        self.loudness = AlarmLoudness(self.presence, enabled=not args.silent,
                                     profile_fn=getattr(self.buzzer, "set_profile", None))
        # If the last run was killed while the siren was up - a Full OFF, a
        # power cut, a crash - the speaker is still at alarm level. Put the
        # guard's own volume back before anything else happens.
        self.loudness.recover_previous()
        # Retain the old config reader for compatibility; visit identity now
        # replaces its fixed route timer and repeat alarms.
        self.visit_quiet_s = self._load_visit_quiet(args.zones)
        self._route_quiet: dict[str, tuple[float, int]] = {}
        self._route_repeats: dict[str, int] = {}
        self.visits = VisitMemory()
        # Stop the whole program when the camera itself goes away.
        self.camera_gone_s = self._load_camera_gone_limit(args.zones)
        self._stop_requested_at = 0.0
        # Global keyboard/mouse input counts only after the siren has sounded
        # this long, so someone typing cannot silence a new alarm unheard.
        self.input_ack_min_s = 1.0
        self._ack_until = 0.0
        self._resume_alarm_check = False
        self.control_warning = ""
        self.acked = False
        self.last_ack = None
        self.fault_reason = ""
        self._fault_kind: str | None = None
        self._fault_since: float | None = None      # when detection stopped (any fault kind)
        self._fault_raised_at = 0.0                 # when the last fault alarm or reminder sounded

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
        self._ai_bad_since = None
        self._light_credits = {}
        self._last_error_print = 0.0
        # Recent alarms with snapshots, newest first, rebuilt from the event log.
        self.recent_alerts: deque = deque(maxlen=80)
        self.alerts_version = 0
        self._snapshots: queue.Queue = queue.Queue(maxsize=16)
        self._snapshot_writer: threading.Thread | None = None
        self._load_recent_alerts()
        self.semantic = None
        model = REPO / "models" / "yolox_tiny.onnx"
        from secondary_analysis import SecondaryAnalyzer
        self.secondary = SecondaryAnalyzer(model, interval=1.0,
                                           learning_path=STATE_DIR / "analysis-learning.json")
        if model.is_file():
            from semantic_monitor import SemanticMonitor
            import object_detector
            object_detector.ENGINE = self._load_ai_engine(args.zones)
            self.semantic = SemanticMonitor(self.zones, model, interval=self._load_ai_interval(args.zones),
                                            context_view=self._load_ai_whole_picture(args.zones),
                                            entrance_verifier_path=(REPO / 'models' / 'yolox_s.onnx'
                                                if self._load_ai_entrance_verifier(args.zones) else None))
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
        # Set when the camera has been physically moved and the saved scene was
        # found again: every frame is mapped back into the zone coordinates.
        self._reacquire = None
        self._reacquire_info = {}
        self._last_reacquire_try = 0.0
        self._last_bootstrap_try = 0.0
        self._bootstrap_info = {}
        self._learning_bootstrap = False
        self._last_strong_at = 0.0
        self._last_learn_at = 0.0
        self._learning = None
        self.alignment = {"state": "disabled", "message": "Automatic alignment is not configured",
                          "calibrated": False, "confidence": None, "dx": 0, "dy": 0}
        self._load_alignment()
        self._service_pending = set()
        self._barrier_error = ''
        try:
            # A view other than the original monitor names its own arm lines in
            # zones.json; without them the original geometry is used.
            barrier_lines = None
            try:
                barriers_config = json.loads(Path(args.zones).read_text(encoding="utf-8-sig")).get("barriers")
                if isinstance(barriers_config, dict) and barriers_config.get("lines") is not None:
                    barrier_lines = barriers_config["lines"]
            except (OSError, ValueError, AttributeError, TypeError):
                barrier_lines = None
            self.barrier_workflow = BarrierWorkflow(args.source, STATE_DIR,
                                                   REPO / 'config' / 'barrier-seed.npz',
                                                   lines=barrier_lines)
        except (OSError, ValueError, RuntimeError) as error:
            # A missing/invalid optional calibration cannot take arrival
            # detection down. UNKNOWN is exposed instead of a guessed arm.
            self.barrier_workflow = None
            self._barrier_error = f'Barrier calibration unavailable: {error}'
        self.barrier_rules = self._load_barrier_rules(args.zones)
        self._apply_barrier_rules()
        if getattr(self, 'barrier_workflow', None) is not None:
            self.barrier_workflow.on_learned = self._barrier_learned
        from alert_learning import AlertLearning
        self.learning = AlertLearning(STATE_DIR / 'alert-learning.json')
        self.display_name = self._load_display_name(args.zones)
        self._route_activity: dict = {}   # route_group -> (monotonic time, route_role)
        from alarm_recorder import AlarmRecorder
        self.recorder = AlarmRecorder(RECORDINGS_DIR, shutil.which("ffmpeg"), on_saved=self._recording_saved,
                                      tie_child=ps.tie_to_this_process,
                                      **self._load_recording_settings(args.zones))
        self._arm_tracker = self._build_arm_tracker(args.zones)
        from arm_angle import ArmMotionRecorder
        self._arm_motion_recorder = ArmMotionRecorder(
            STATE_DIR / "barrier-motion", geometry=getattr(self._arm_tracker, 'geometry', None),
            width=PROC_W, height=PROC_H)
        self._arm_live, self._arm_last_run, self._arm_motion_acked = {}, 0., set()
        from vehicle_sightings import VehicleSightings
        self.sightings = VehicleSightings(STATE_DIR / "vehicle-sightings",
                                          enabled=self._load_sightings_enabled(args.zones))
        if self.semantic is not None:
            self.semantic.keep_frames = self.sightings.enabled
        self._settings_lock = threading.Lock()
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        ps.rotate_log(EVENT_LOG, max_bytes=5_000_000)

    def _view_key(self, view) -> str:
        identity = self.args.source + (f"|{view}" if view else "")
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    # Switching the keyboard to another saved camera is deliberate: look soon, then every 30 s.
    VIEW_SWITCH_AFTER_S, VIEW_RETRY_S = 8., 30.
    FRAMING_ESTIMATE_S = 20.

    def _zoom_tolerance(self) -> float:
        zoom = (getattr(self, "alignment_config", None) or {}).get("zoom_tolerance")
        if isinstance(zoom, (int, float)) and not isinstance(zoom, bool) and 0 < zoom <= .35:
            return float(zoom)
        return 0.

    def _view_title(self, view=None) -> str:
        view = view if view is not None else getattr(self, "view_name", None)
        return {"ptz2": "PTZ 2", "anpr": "ANPR Entrance", None: "ANPR Entrance"}.get(view, str(view))

    def _framing_status(self, locked, info, now, shape):
        """Zoomable views only: how to zoom or turn the camera back to the saved framing (for the operator)."""
        if self._zoom_tolerance() <= 0:
            return None
        import screen_registration
        label = f"the saved {self._view_title()} view"
        try:
            matrix = None
            if locked and info.get("current_to_reference") is not None:
                strict = np.vstack([np.asarray(info["current_to_reference"], dtype=np.float64), [0., 0., 1.]])
                pre = (np.asarray(self._reacquire, dtype=np.float64) if getattr(self, "_reacquire", None) is not None
                       else np.eye(3))
                matrix = strict @ pre
            else:
                seen = getattr(self, "_framing_seen", None)
                if seen and now - seen[1] <= self.FRAMING_ESTIMATE_S:
                    matrix = seen[0]
            if matrix is None:
                # Missing landmarks do not establish a zoom direction or prove
                # that the physical camera moved. Glare/occlusion can cause this.
                return {"state": "unknown", "primary": "lost", "headline": "View alignment uncertain", "fit_pct": 0,
                        "text": f"Cannot currently verify {label}. Check the live picture; do not move the camera "
                                "based on this warning alone.",
                        "short": "View alignment uncertain: checking scene landmarks"}
            guide = screen_registration.framing_guide(matrix, shape, label, zones=self._framing_zones())
            if info.get("framing_retained"):
                # The caller independently validated use of this held transform.
                # Do not pretend it is a newly measured camera position.
                return {**guide, "state": "holding", "primary": "held", "headline": "View held steady",
                        "text": "Landmarks briefly unclear; using the verified view while checking again.",
                        "short": "Verified view retained", "retained": True}
            self._framing_last = (guide, now)
            return guide
        except Exception:                                   # noqa: BLE001 - advice only, never blocks detection
            return None

    def _framing_zones(self) -> list:
        """The active alarm areas as polygons (saved-view coordinates), for the framing coach."""
        zones = []
        for zone in getattr(self, "zones", None) or []:
            try:
                polygon = getattr(zone, "polygon", None) or rect_polygon(tuple(zone.rect))
                zones.append([[float(x), float(y)] for x, y in polygon])
            except (TypeError, ValueError, AttributeError):
                continue
        return zones

    def _profile_masks(self, reference, config):
        """Landmark and required (alarm zone) masks for another view's saved profile."""
        from scene_alignment import build_feature_mask
        height, width = reference.shape[:2]
        required = np.zeros((height, width), dtype=bool)
        for zone in config.get("zones", []) if isinstance(config.get("zones"), list) else []:
            if not isinstance(zone, dict):
                continue
            polygon = zone.get("polygon")
            if not polygon and zone.get("rect"):
                polygon = rect_polygon(tuple(zone["rect"]))
            if polygon:
                required |= np.asarray(polygon_mask(width, height, polygon), dtype=bool)
        alignment = config.get("alignment", {}) or {}
        feature = build_feature_mask(reference.shape,
                                     include_rects=alignment.get("include_rects", [[.12, .10, .79, .82]]),
                                     exclude_rects=alignment.get("exclude_rects", []),
                                     border_px=int(alignment.get("border_px", 8)),
                                     include_polygon=alignment.get("scene_polygon"))
        feature[required] = 0
        return feature, required.astype(np.uint8) * 255

    def _try_other_views(self, frame, now) -> bool:
        """The picture has matched nothing for a while: is the keyboard showing another saved camera view?"""
        bad_since = getattr(self, '_alignment_bad_since', 0.) or now
        tried = getattr(self, '_view_tried_at', None)
        if now - bad_since < self.VIEW_SWITCH_AFTER_S or (tried is not None and now - tried < self.VIEW_RETRY_S):
            return False
        self._view_tried_at = now
        try:
            import screen_registration
            from scene_alignment import SceneAligner
            folder = Path(self.args.zones).resolve().parent / "views"
            current = getattr(self, 'view_name', None)
            for profile in sorted(folder.glob("*/zones.json")) if folder.is_dir() else []:
                config = json.loads(profile.read_text(encoding="utf-8-sig"))
                view = config.get("view") if isinstance(config.get("view"), str) else None
                if view == current:
                    continue
                reference_path = STATE_DIR / f"alignment-{self._view_key(view)}.npz"
                if not reference_path.is_file():
                    continue
                with np.load(reference_path, allow_pickle=False) as saved:
                    reference = saved["reference"]
                if reference.shape != frame.shape:
                    continue
                feature, required = self._profile_masks(reference, config)
                previous = self.alignment_config
                try:
                    self.alignment_config = config.get("alignment", {}) or {}
                    aligner = SceneAligner(reference, feature, config=self._aligner_config(), required_mask=required)
                finally:
                    self.alignment_config = previous
                if not aligner.reference_ready:
                    continue
                aligned, _ = aligner.update(frame)
                if aligned is None:
                    # Zoomed or turned a little (PTZ): find it by its landmarks, then confirm strictly.
                    alignment = config.get("alignment", {}) or {}
                    matrix, _ = screen_registration.estimate_transform(
                        frame, reference, required_mask=required,
                        reference_mask=screen_registration.registration_mask(
                            reference.shape, polygon=alignment.get("scene_polygon"),
                            exclude_rects=alignment.get("registration_exclude_rects", [])))
                    if matrix is not None:
                        warped = screen_registration.warp(frame, matrix, reference.shape[:2])
                        aligned, _ = aligner.update(warped, prior=[[1., 0., 0.], [0., 1., 0.]])
                if aligned is None:
                    continue
                self._switch_view(profile, self._view_title(view) if view else profile.parent.name)
                return True
        except Exception as exc:                            # noqa: BLE001 - never disturbs detection
            self._calibration_error = f"view check failed: {type(exc).__name__}: {exc}"[:200]
        return False

    def _switch_view(self, profile: Path, name: str) -> None:
        """Save the active zones back to their own profile, make the matching view active, restart."""
        from zone_editor import _atomic_write
        active = Path(self.args.zones)
        current = getattr(self, 'view_name', None)
        own = active.parent / "views" / (current or "anpr") / "zones.json"
        if own.parent.is_dir():
            _atomic_write(own, active.read_bytes())
        _atomic_write(active, profile.read_bytes())
        self.log_event("view", f"camera view changed to {name}; zones and barriers for that view loaded")
        (STATE_DIR / 'restart-request').write_text(time.strftime('%Y-%m-%dT%H:%M:%S') + '\n', encoding='utf-8')

    def _alignment_path(self):
        # Separate sources cannot silently reuse one another's reference; neither can separate camera views
        # shown through the same capture (a named view keeps its own reference).
        key = self._view_key(getattr(self, 'view_name', None))
        return STATE_DIR / f"alignment-{key}.npz"

    def _feature_mask(self, frame):
        from scene_alignment import build_feature_mask
        cfg = self.alignment_config
        mask = build_feature_mask(frame.shape,
            include_rects=cfg.get("include_rects", [[.12, .10, .79, .82]]),
            exclude_rects=cfg.get("exclude_rects", []),
            border_px=int(cfg.get("border_px", 8)),
            include_polygon=cfg.get("scene_polygon"))
        for zone in self.zones:
            mask[zone.mask_for(frame.shape[1], frame.shape[0])] = 0
        return mask

    def _required_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        for zone in self.zones:
            mask |= zone.mask_for(frame.shape[1], frame.shape[0])
        return mask.astype(np.uint8) * 255

    def _light_scene_mask(self, shape):
        polygon = self.alignment_config.get("light_scene_polygon")
        if not polygon:
            return None
        key = (shape[0], shape[1], tuple(tuple(p) for p in polygon))
        if getattr(self, "_light_scene_cache_key", None) != key:
            self._light_scene_cache_key = key
            self._light_scene_cache = polygon_mask(shape[1], shape[0], polygon)
        return self._light_scene_cache

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

    # -- recovering after the camera is physically moved --------------------
    def _reacquire_path(self):
        return self._alignment_path().with_suffix(".reacquire.json")

    def _read_reacquire(self):
        try:
            data = json.loads(self._reacquire_path().read_text(encoding="utf-8"))
            matrix = np.asarray(data["matrix"], dtype=np.float64)
            if matrix.shape == (3, 3) and np.isfinite(matrix).all():
                self._reacquire_info = {"restored": data.get("saved", ""), **data.get("report", {})}
                return matrix
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _save_reacquire(self) -> None:
        try:
            from zone_editor import _atomic_write
            if self._reacquire is None:
                self._reacquire_path().unlink(missing_ok=True)
                return
            _atomic_write(self._reacquire_path(), json.dumps({
                "matrix": np.asarray(self._reacquire, dtype=np.float64).round(8).tolist(),
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "report": {key: value for key, value in self._reacquire_info.items()
                           if isinstance(value, (int, float, str))}}).encode())
        except OSError:
            pass

    def _try_reacquire(self, camera_frame, now) -> None:
        """Find the saved scene again after the camera has been moved.

        Runs only when ordinary alignment has been lost for a while. Every
        candidate must pass the normal alignment checks in the re-mapped view,
        and no alarm zone may fall outside the new picture, so a wrong or
        partial view is refused instead of silently moving the zones.
        """
        # After a fresh start nothing has ever aligned, which is exactly the
        # case when the camera has just been set up somewhere new: react fast.
        # A zoomable (PTZ) view is zoomed or turned on purpose by the operator: look again within seconds.
        zoomable = self._zoom_tolerance() > 0
        wait = (REACQUIRE_FIRST_S if self._alignment_good is None
                else ZOOM_REACQUIRE_S if zoomable else REACQUIRE_AFTER_S)
        if (not self._alignment_bad_since or now - self._alignment_bad_since < wait
                or now - self._last_reacquire_try < (ZOOM_REACQUIRE_S if zoomable else REACQUIRE_EVERY_S)):
            return
        self._last_reacquire_try = now
        import screen_registration
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        for index, aligner in enumerate([self.aligner] + self._extra_aligners):
            reference = getattr(aligner, "reference", None)
            if reference is None:
                continue
            matrix, report = screen_registration.estimate_transform(
                camera_frame, reference, required_mask=self._required_mask(reference),
                reference_mask=screen_registration.registration_mask(
                    reference.shape, polygon=self.alignment_config.get('scene_polygon'),
                    exclude_rects=self.alignment_config.get('registration_exclude_rects', [])))
            if index == 0 and report.get("matrix") is not None:
                self._framing_seen = (report["matrix"], now)     # for the framing guide
            if matrix is None:
                self._reacquire_info = report
                continue
            warped = screen_registration.warp(camera_frame, matrix, reference.shape[:2])
            confirmed, _ = aligner.update(warped, prior=identity)
            if confirmed is None:
                self._reacquire_info = {**report, "reason": "The re-mapped view failed the usual checks"}
                continue
            moved = screen_registration.movement_px(matrix, reference.shape[:2])
            self._reacquire = matrix
            self._reacquire_info = {**report, "moved_px": round(moved, 1), "reference_index": index,
                                    "at": time.strftime("%H:%M:%S"), "reason": ""}
            # The pose and background belong to the old framing.
            self._alignment_good = self._alignment_used = None
            self._alignment_pose = None
            self._preferred_aligner = index
            self._pending_rearm = True
            self._save_reacquire()
            self.log_event("alignment", f"camera moved about {moved:.0f} px; zones re-mapped "
                                        f"automatically from {report['inliers']} landmarks")
            return

    def _try_bootstrap(self, frame, now) -> None:
        """Start up in lighting unlike every stored reference (the 13 Nov 23:50 trap).

        Landmark matching can fail completely when the light is very different
        (a first daylight start, fog, new floodlights), and then nothing ever
        locks. The picture is a digital capture of a fixed camera, so the real
        question is whether this is the same scene at the saved position. The
        structure check answers that almost independently of lighting. On a
        match the current view, warped with the saved pose, goes through the
        usual five-frame test as a new lighting reference; a different view, a
        moved camera or a blank screen does not match and stays a fault.
        """
        if (self._alignment_good is not None or self._alignment_used is not None or self._learning is not None
                or self._alignment_reference is None or not self._alignment_bad_since
                or now - self._alignment_bad_since < BOOTSTRAP_AFTER_S
                or now - self._last_bootstrap_try < BOOTSTRAP_EVERY_S):
            return
        self._last_bootstrap_try = now
        from scene_alignment import structural_match
        prior = np.asarray(self._alignment_prior(), dtype=np.float64)
        linear = prior[:, :2]
        if not np.allclose(linear, np.eye(2), atol=.01):
            self._bootstrap_info = {"reason": "The saved pose is not a plain shift; structure check skipped"}
            return
        best = None
        for index, aligner in enumerate([self.aligner] + self._extra_aligners):
            reference = getattr(aligner, "reference", None)
            if reference is None or getattr(reference, "shape", None) != frame.shape:
                continue
            result = structural_match(frame, reference, self._feature_mask(reference))
            if best is None or result["response"] > best["response"]:
                best = {**result, "reference_index": index}
        if best is None:
            return
        # prior maps current -> reference, so a matching view is shifted by -t.
        gap = float(np.hypot(best["dx"] + prior[0, 2], best["dy"] + prior[1, 2]))
        self._bootstrap_info = {"response": round(best["response"], 3), "offset_px": round(gap, 2),
                                "reference_index": best["reference_index"], "at": time.strftime("%H:%M:%S")}
        if best["response"] < BOOTSTRAP_MIN_RESPONSE or gap > BOOTSTRAP_MAX_GAP_PX:
            self._bootstrap_info["reason"] = ("The scene's structure does not match the saved view"
                                              if best["response"] < BOOTSTRAP_MIN_RESPONSE else
                                              "The scene matches but is shifted from the saved position")
            return
        try:
            aligned = self.aligner.warp(frame, prior)
            candidate = self._build_aligner(np.ascontiguousarray(aligned)) if aligned is not None else None
        except Exception:
            candidate = None
        if candidate is None:
            self._bootstrap_info["reason"] = "Structure matches, but the view has too few landmarks to learn"
            return
        self._learning, self._last_learn_at = (candidate, 0, 0), now
        self._learning_bootstrap = True
        self._lighting_status = (f"New lighting at start-up: the scene's structure matches the saved view "
                                 f"(score {best['response']:.2f}, {gap:.1f} px); checking it as a lighting reference")
        self.log_event("alignment", f"start-up structure check matched (score {best['response']:.2f}, "
                                    f"offset {gap:.1f} px); learning this lighting")

    def _structure_still_matches(self, frame, now) -> bool:
        """Past the hold: is the scene's lighting-robust structure still exactly where it was verified?

        Headlights can wash out the landmarks for longer than the hold (14 Nov 18:51, a car waiting at the
        top). A moved camera or another view fails this at once. Checked at most once a second.
        """
        if now - getattr(self, '_structure_checked_at', 0.0) < 1.0:
            return getattr(self, '_structure_ok', False)
        self._structure_checked_at = now
        self._structure_ok = False
        try:
            from scene_alignment import structural_match
            reference = self._alignment_reference
            if reference is None or getattr(reference, 'shape', None) != frame.shape or self._alignment_used is None:
                return False
            matrix = np.asarray(self._alignment_used[0], dtype=np.float64)
            if not np.allclose(matrix[:, :2], np.eye(2), atol=.01):
                return False
            result = structural_match(frame, reference, self._feature_mask(reference))
            gap = float(np.hypot(result["dx"] + matrix[0, 2], result["dy"] + matrix[1, 2]))
            self._structure_ok = result["response"] >= BOOTSTRAP_MIN_RESPONSE and gap <= BOOTSTRAP_MAX_GAP_PX
            self._structure_score = round(result["response"], 3)
        except Exception:                                  # noqa: BLE001 - then the normal fault applies
            self._structure_ok = False
        return self._structure_ok

    def _alignment_prior(self):
        """The camera pose that weak (for example night-time) landmarks may confirm."""
        if self._alignment_good is not None:
            return self._alignment_good[0]
        if self._alignment_pose is not None:
            return self._alignment_pose
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]   # The reference's own pose.

    def _reacquired_zones_visible(self, camera_frame, affine) -> bool:
        """Both remount and small-correction transforms must retain every zone."""
        if self._reacquire is None:
            return True
        if affine is None:
            return False
        import screen_registration
        matrix = np.asarray(affine, dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return False
        # Warp twice, just like the picture. An inverse second transform cannot
        # recover source pixels already clipped by the first image warp.
        import cv2
        first = screen_registration.visibility_mask(self._reacquire, camera_frame.shape[:2])
        visible = cv2.warpAffine(first, matrix, (camera_frame.shape[1], camera_frame.shape[0]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        visible = cv2.erode(visible, np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        required = self._required_mask(camera_frame) > 0
        return bool(required.any() and visible[required].all())

    # -- learned lighting references -------------------------------------
    def _extra_path(self):
        return self._alignment_path().with_suffix(".lighting.npz")

    def _aligner_config(self):
        """Alignment settings from zones.json, clamped to values that still track.

        Only the landmark budget is exposed. Measured on this camera: 1800
        features cost 39 ms a frame and 900 cost 25 ms, with the same median
        error (0.02 px) and hundreds of inliers either way. The saved
        milliseconds are what keep the detector inside its frame budget.
        """
        import dataclasses

        from scene_alignment import AlignmentConfig
        config = AlignmentConfig()
        try:
            zoom = (self.alignment_config or {}).get("zoom_tolerance")
            if isinstance(zoom, (int, float)) and not isinstance(zoom, bool) and 0 < zoom <= .35:
                # A zoom about the picture centre also moves the edges: widen scale, shift and corner
                # limits together (640x360 working size).
                config = dataclasses.replace(
                    config, max_scale_change=max(config.max_scale_change, float(zoom)),
                    max_translation_px=max(config.max_translation_px, 330. * float(zoom) + 10.),
                    max_corner_displacement_px=max(config.max_corner_displacement_px, 380. * float(zoom) + 20.),
                    # A zoom-in leaves part of the saved view outside the picture; the alarm zones must
                    # still be fully visible (required mask), which is checked separately.
                    min_valid_fraction=min(config.min_valid_fraction, (1. / (1. + float(zoom))) ** 2 - .04))
        except (TypeError, ValueError):
            pass
        try:
            requested = (self.alignment_config or {}).get("max_features")
            if requested is not None:
                config = dataclasses.replace(
                    config, max_features=int(min(3000, max(400, int(requested)))))
        except (TypeError, ValueError):
            pass
        return config

    def _build_aligner(self, image):
        from scene_alignment import SceneAligner
        aligner = SceneAligner(image, self._feature_mask(image), config=self._aligner_config(),
                               required_mask=self._required_mask(image))
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
            # A start-up candidate already passed the lighting-robust structure
            # check; at night landmarks bunch into one bright cluster, so it may
            # confirm the saved pose weakly. Ordinary learning still needs strong.
            result, status = (candidate.update(frame, prior=self._alignment_prior())
                              if getattr(self, '_learning_bootstrap', False) else candidate.update(frame))
            matrix = status.get("current_to_reference")
            passed = (result is not None and matrix is not None
                      and pose_gap_px(matrix, self._alignment_prior(), frame.shape) <= 0.75)
            tries, passes = tries + 1, passes + passed
            if tries < 5:
                self._learning = (candidate, tries, passes)
                return
            self._learning = None
            self._learning_bootstrap = False
            if passes >= 3:
                try:
                    self._commit_lighting_reference(candidate, now)
                    self._lighting_status = (f"Saved a lighting reference at {time.strftime('%H:%M')} "
                                             f"({len(self._extra_aligners)} stored)")
                except OSError:
                    self._lighting_status = "Lighting reference could not be saved; existing references retained"
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
            self._learning, self._learning_bootstrap = (candidate, 0, 0), False

    @staticmethod
    def _lighting_signature(image):
        import cv2
        gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _replacement_slot(self, candidate) -> int:
        """When full, replace the stored reference that looks most like the new one.

        Replacing the least recently used one threw away the only evening
        reference on 14 Nov 04:46 (a night of night references outvoted it),
        which is the one the next evening's start needs. Replacing the most
        similar keeps one reference per kind of lighting.
        """
        try:
            target = self._lighting_signature(candidate.reference)
            distances = [float(np.abs(self._lighting_signature(aligner.reference) - target).mean())
                         for aligner in self._extra_aligners]
            return min(range(len(distances)), key=distances.__getitem__)
        except Exception:
            return min(range(len(self._extra_used)), key=self._extra_used.__getitem__)

    def _commit_lighting_reference(self, candidate, now) -> None:
        previous = (list(self._extra_aligners), list(self._extra_used), self._preferred_aligner)
        if len(self._extra_aligners) >= MAX_LIGHTING_REFERENCES:
            slot = self._replacement_slot(candidate)
            self._extra_aligners[slot], self._extra_used[slot] = candidate, now
        else:
            self._extra_aligners.append(candidate)
            self._extra_used.append(now)
            slot = len(self._extra_aligners) - 1
        self._preferred_aligner = slot + 1
        try:
            self._save_lighting_references()
        except OSError:
            self._extra_aligners, self._extra_used, self._preferred_aligner = previous
            raise
        self.log_event("alignment", f"learned lighting reference {slot + 1} "
                                    f"({len(self._extra_aligners)} stored)")

    def _load_alignment(self):
        try:
            config = json.loads(Path(self.args.zones).read_text(encoding="utf-8-sig"))
            self.alignment_config = config.get("alignment", {}) if isinstance(config, dict) else {}
            view = config.get("view") if isinstance(config, dict) else None
            self.view_name = view if isinstance(view, str) and re.fullmatch(r"[a-z0-9_-]{1,24}", view) else None
            hold = self.alignment_config.get("hold_s", 10.0)
            if isinstance(hold, (int, float)) and not isinstance(hold, bool) and math.isfinite(hold):
                self.alignment_hold_s = min(30.0, max(0.0, float(hold)))
            if not self.alignment_config.get("enabled", False):
                return
            from scene_alignment import SceneAligner
            self.aligner = SceneAligner(config=self._aligner_config())
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
                self._reacquire = self._read_reacquire()
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
            if self.alignment.get("state") != "tracking":
                raise ValueError("Wait for verified alignment before saving lighting; use recalibration if the camera moved")
            if self._learning is not None:
                raise ValueError("A lighting reference is already being checked")
            self._pending_lighting = True
            self._lighting_status = "Checking the current view as a lighting reference"

    def align_frame(self, frame):
        with self._state_lock:
            return self._align_frame_locked(frame)

    @staticmethod
    def _keyboard_menu_visible(frame) -> bool:
        """The keyboard's own white "Live View" menu bar is across the top of the picture."""
        try:
            band = np.asarray(frame[MENU_BAND[0]:MENU_BAND[1], :, :3], dtype=np.float32).mean(axis=2)
            return bool(band.size) and float(np.mean(band >= 225)) >= 0.80 and float(band.mean()) >= 225
        except (TypeError, ValueError, IndexError):
            return False

    def _align_frame_locked(self, frame):
        if self._keyboard_menu_visible(frame):
            # Not the camera picture: never measure zones on a menu, and say so plainly.
            self._menu_since = getattr(self, '_menu_since', None) or time.monotonic()
            self.alignment = {**self.alignment, "state": "keyboard_menu",
                              "message": "The keyboard's menu is open over the camera picture; zone alerts are paused"}
            return None
        self._menu_since = None
        if self.aligner is None:
            if self._alignment_blocked:
                self._alignment_bad_since = self._alignment_bad_since or time.monotonic()
            return None if self._alignment_blocked else frame
        camera_frame = frame
        if self._reacquire is not None:
            # The camera sits somewhere new: bring its view back into the saved
            # zone coordinates before anything else looks at the picture.
            try:
                import screen_registration
                frame = screen_registration.warp(frame, self._reacquire, frame.shape[:2])
            except Exception:
                self._reacquire, frame = None, camera_frame
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
            try:
                if self._reacquire is not None:
                    current, current_info = self._update_alignment(frame, self._alignment_prior())
                    if (current is None or not self._reacquired_zones_visible(
                            camera_frame, current_info.get('current_to_reference'))):
                        raise ValueError("Verify the recovered camera position before replacing its reference")
                else:
                    current, _ = self.aligner.update(frame)
                reference = current if current is not None else frame
                from scene_alignment import SceneAligner
                candidate = SceneAligner(reference, self._feature_mask(reference),
                                         config=self._aligner_config(),
                                         required_mask=self._required_mask(reference))
                aligned, info = candidate.update(frame)
                if aligned is None:
                    raise ValueError(info.get("reason") or "Not enough static landmarks")
                if not self._reacquired_zones_visible(camera_frame, info.get('current_to_reference')):
                    raise ValueError("An alert zone is outside the recovered camera view")
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
                # The new image is already in the monitor's canonical zone
                # coordinates. Keep the raw-camera remount transform: removing
                # it here would feed the next raw frame into a different frame
                # of reference and immediately lose the newly saved alignment.
                for stale in (self._pose_path(), self._extra_path()):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                self._pending_rearm = True
            except Exception as exc:
                self._calibration_error = str(exc)
        aligned, info = self._update_alignment(frame, self._alignment_prior())
        if aligned is not None and not self._reacquired_zones_visible(camera_frame, info.get('current_to_reference')):
            aligned = None
            info = {**info, 'accepted': False, 'reason': 'An alert zone is outside the recovered camera view'}
        accepted = aligned is not None
        now = time.monotonic()
        holding = False
        if accepted:
            matrix = info.get("current_to_reference")
            if matrix is not None and not info.get("weak"):
                from scene_alignment import pose_gap_px
                if self._alignment_used is not None and pose_gap_px(matrix, self._alignment_used[0], frame.shape) > 2.0:
                    # The zones would visibly jump: relearn the background first.
                    self._alignment_rearm_reason = "Verified camera correction exceeded two pixels"
                    self._pending_rearm = True
                self._last_strong_at = now
                self._alignment_good = (matrix, now)
                self._maybe_save_pose(matrix, frame.shape)
            if matrix is not None:
                self._alignment_used = (matrix, now)
            if self._alignment_blocked:
                self._alignment_rearm_reason = "Scene alignment recovered after an unverified view"
                self._pending_rearm = True
            self._alignment_bad_since = 0.0
        else:
            self._alignment_bad_since = self._alignment_bad_since or now
            # A fixed mount does not move between frames. Brief landmark loss
            # (glare, a large vehicle, decoder smear) keeps the last verified
            # transform, so detection is neither paused nor reset. Only a
            # rejection lasting longer than the hold becomes a fault.
            if (self._alignment_used is not None and not self._alignment_blocked
                    and (now - self._alignment_bad_since <= self.alignment_hold_s
                         or self._structure_still_matches(frame, now))):
                try:
                    aligned = (self.aligner.warp(frame, self._alignment_used[0])
                               if self._reacquired_zones_visible(camera_frame, self._alignment_used[0]) else None)
                except Exception:
                    aligned = None
                holding = aligned is not None
        self._alignment_blocked = aligned is None
        if self._alignment_blocked:
            # A physically moved camera is found here, so the status published
            # below already reflects this attempt.
            self._try_reacquire(camera_frame, now)
            self._try_other_views(camera_frame, now)
            self._try_bootstrap(frame, now)
        translation = info.get("translation_px") or [0, 0]
        if holding:
            translation = [self.alignment.get("dx", 0), self.alignment.get("dy", 0)]
            left = max(0.0, self.alignment_hold_s - (now - self._alignment_bad_since))
            message = ((f"Holding the last verified alignment ({left:.0f} s left): "
                        + info.get("reason", "landmarks briefly unclear")) if left > 0 else
                       "Glare or traffic hides the landmarks; the scene's structure confirms the camera has "
                       "not moved, so zones stay at the verified position")
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
            "lighting_status": self._lighting_status,
            "reacquired": self._reacquire is not None, "reacquire": dict(self._reacquire_info),
            "bootstrap": dict(self._bootstrap_info)}
        framing_info = info
        if holding and self._alignment_used is not None:
            framing_info = {**info, "current_to_reference": self._alignment_used[0], "framing_retained": True}
        elif accepted and info.get("weak"):
            framing_info = {**info, "framing_retained": True}
        framing = self._framing_status(accepted or holding, framing_info, now, frame.shape)
        if framing is not None:
            self.alignment["framing"] = framing
            self.alignment["view"] = self._view_title()
            if framing.get("state") != "ok":
                self.alignment["message"] = f"{self.alignment['message']} \u00b7 {framing['text']}"
        self._note_alignment_transition(now)
        self._consider_learning(frame, aligned if accepted else None, info, now)
        if self._pending_lighting:
            # A steady but moved camera must not certify its own stale pose.
            self._pending_lighting = False
            candidate = None
            if not self.armed and self._learning is None and accepted:
                try:
                    image = aligned
                    candidate = self._build_aligner(np.ascontiguousarray(image)) if image is not None else None
                except Exception:
                    candidate = None
            if candidate is None:
                self._lighting_status = "Current alignment could not be verified; no lighting reference was saved"
            else:
                self._learning, self._last_learn_at = (candidate, 0, 0), now
                self._learning_bootstrap = False
        return aligned

    def _record_diagnostic_clip(self, topic, reason, now):
        """Keep bounded local video evidence without creating or acknowledging an alarm."""
        try:
            recorder = getattr(self, 'recorder', None)
            if not self.armed or recorder is None or not recorder.enabled:
                return
            stamps = getattr(self, '_diagnostic_clip_at', {})
            if now - stamps.get(topic, -math.inf) < 60.:
                return
            stamps[topic] = now
            self._diagnostic_clip_at = stamps
            ident = f"diag-{topic}-{int(time.time()*1000)}"
            recorder.alarm(ident, 'diagnostic', reason, now=now)
            self.log_event('diagnostic-video', reason, diagnostic_id=ident, topic=topic)
        except Exception:
            pass  # Recording failures never interrupt the alert path.

    def _note_alignment_transition(self, now):
        """Record brief losses too: they may recover before the fault sound threshold."""
        try:
            state = self.alignment
            key = (state.get('state'), bool(state.get('weak')), self._alignment_blocked,
                   getattr(self, 'view_name', None))
            previous = getattr(self, '_alignment_note', None)
            if previous and previous[0] != key and (key[0] != 'tracking' or key[1]):
                self._record_diagnostic_clip('alignment', state.get('message', 'Alignment uncertain'), now)
            if previous and (previous[0] == key or now-previous[1] < 1.):
                return
            self._alignment_note = (key, now)
            self.log_event('alignment-transition', state.get('message', ''),
                           previous=previous[0] if previous else None, current=key,
                           pending_reset=bool(self._pending_rearm),
                           matrix=state.get('current_to_reference'),
                           inliers=state.get('inliers'), static_agreement=state.get('static_agreement'))
        except Exception:
            pass  # Diagnostics must never gate frame processing.

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
        # Arming is a deliberate fresh start: no approach may still be inside a
        # quiet window from before, or the first vehicle after arming is held.
        self._route_quiet.clear()
        self._route_repeats.clear()
        self.visits.reset()
        if getattr(self, 'barrier_workflow', None) is not None:
            self.barrier_workflow.reset()
        if not value:
            # Going quiet: kill the siren and drop any latched alarm outright.
            self.alarm = None
            self.alarm_id = None
            self.alarm_reason = ""
            self.acked = False
            self.fault_reason = ""
            self.buzzer.set(None)
            self.loudness.end()
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

    def raise_alarm(self, kind: str, reason: str, *, details=None) -> bool:
        with self._state_lock:
            return self._raise_alarm_locked(kind, reason, details=details)

    def _expire_ack_locked(self, now: float) -> None:
        if self._ack_until and now >= self._ack_until:
            self._ack_until = 0.0
            self.acked = False
            # Recheck current conditions after the quiet period; never replay
            # an event recorded while the user asked for silence.
            self._resume_alarm_check = True

    def _raise_alarm_locked(self, kind: str, reason: str, *, details=None) -> bool:
        now = time.monotonic()
        self._expire_ack_locked(now)
        if not self.armed or (self._pending_rearm and kind != "fault"):
            # Disarmed means silent: no tone, no notification, no event.
            return False
        details = dict(details or {})
        if (kind != "fault" and "frame_received_at" in details
                and not self._local_frame_fresh(details["frame_received_at"])):
            # Recheck after obtaining the state lock: time spent waiting for
            # alignment or another operation must not freshen old evidence.
            return False
        decision = None
        if (kind != 'fault' and self.alarm in ('early','main') and self.alarm_id and not self.acked):
            decision = self.visits.continue_unanswered(self.alarm_details.get('visit_id'),self.alarm_id,
                self.alarm_details,details,self.zones,now)
        if decision is None and kind != 'fault':
            decision = self.visits.consider(details, now)
        if decision and decision.suppress:
            # A handled visit is consumed by every detector channel. A later
            # recognition enriches the original incident instead of ringing again.
            if decision.visit_id and self.alarm_details.get("visit_id") == decision.visit_id:
                if details.get("source") == "ai":
                    self.alarm_details.update(details)
                    self.alarm_details["visit_id"] = decision.visit_id
            return decision.visit_id is not None
        if now < self._ack_until:
            return False
        if (decision and self.alarm and self.alarm_id and decision.visit_id
                and decision.visit_id != self.alarm_details.get('visit_id')
                and VisitMemory.route_key(details) == VisitMemory.route_key(self.alarm_details)):
            # This different arrival may be waiting behind the current alarm.
            # Never let completion of the first car consume the second one.
            if hasattr(self, '_service_pending'):
                self._service_pending.add((self.alarm_id, VisitMemory.route_key(details)))
        rank = {"early": 1, "main": 2, "fault": 3}
        previous_resolved = False
        if self.alarm == "fault":
            previous_resolved = not bool(self.fault_reason)
        elif self.alarm:
            previous_evidence = self.visits.evidence_for_alarm(self.alarm_id, now)
            previous_resolved = (previous_evidence.get("retired") is True or
                                 (previous_evidence.get("fresh") is True
                                  and previous_evidence.get("present") is False))
        if (self.alarm and not self.acked and not previous_resolved
                and rank[kind] <= rank.get(self.alarm, 0)):
            return False
        if self.alarm in ("early", "main", "fault") and self.alarm_id and not self.acked:
            # Nobody answered the earlier alarm before this one took over: say so in its history
            # instead of leaving it "Not acknowledged" forever.
            replaced_id, replaced_reason = self.alarm_id, self.alarm_reason
            replaced_after = now - self.alarm_since
            self._mark_alert_acknowledged(replaced_id, "replaced", replaced_after)
            self.log_event("alarm-replaced", f"not answered before the next alarm: {replaced_reason}",
                           alarm_id=replaced_id, after_s=round(replaced_after, 1))
        self.alarm = kind
        self.alarm_reason = reason
        self.alarm_details = dict(details or {})
        if decision and decision.visit_id:
            self.alarm_details["visit_id"] = decision.visit_id
        self.alarm_since = now
        self._alarm_sequence += 1
        self.alarm_id = f"{self._alarm_session}:{self._alarm_sequence}"
        if decision and decision.visit_id:
            self.visits.emitted(decision.visit_id, self.alarm_id, now)
        self.acked = False
        self._test_tone = None
        if self.loudness.kind == 'barrier_open':
            self.buzzer.set(None)
            self.loudness.end()
        # Lower the mixer before the very first sample can play.
        learning = getattr(self, 'learning', None)
        learned = bool(learning is not None and kind != 'fault' and learning.reliable(details))
        if learned:
            self.alarm_details['learned_reliable'] = True
        let_in = bool((details.get("cross_lane") or {}).get("arm_raised")) and getattr(self, 'soften_let_in', True)
        if let_in:
            self.alarm_details['started_gentle'] = True
        self.loudness.begin(kind, source=details.get("source", "unknown"),
                            confirmed=details.get("source") == "ai",
                            lane=details.get("route_role", ""),
                            urgent=(bool(details.get("strict_change")) or learned) and not let_in)
        self.buzzer.set(kind)
        snapshot = self._queue_snapshot(kind)
        self.log_event(kind, reason, alarm_id=self.alarm_id, snapshot=snapshot,
                       evidence=self.alarm_details)
        if learning is not None and kind != 'fault':
            learning.note_alarm(self.alarm_id, self.alarm_details)
        self._remember_alert(time.strftime("%Y-%m-%dT%H:%M:%S"), kind, reason, self.alarm_id, snapshot,
                             evidence=self.alarm_details)
        recorder = getattr(self, 'recorder', None)
        if recorder is not None:
            recorder.alarm(self.alarm_id, kind, reason)
        if kind != "early":
            notify(f"{getattr(self, 'display_name', None) or 'BarrierWatcher'}: {kind.upper()}", reason)
        self._consider_already_opening(kind, details, now)
        return True

    def acknowledge(self, alarm_id: str | None = None, via: str = "api") -> bool:
        with self._state_lock:
            return self._acknowledge_locked(alarm_id, via)

    def _acknowledge_locked(self, alarm_id: str | None = None, via: str = "api") -> bool:
        if not self.alarm and self.loudness.kind == 'barrier_open' and (
                alarm_id is None or alarm_id == f"reminder:{getattr(self, '_reminder_seq', 0)}"):
            # Dismissing the "arm left up" knock: quiet until that arm closes.
            workflow = getattr(self, 'barrier_workflow', None)
            muted = workflow.mute_reminders() if workflow is not None else []
            self.buzzer.set(None)
            self.loudness.end()
            self.log_event("reminder-muted", f"via {via}: barrier left up", routes=muted)
            return True
        if not self.alarm or (alarm_id is not None and alarm_id != self.alarm_id):
            return False
        if via.startswith("input-") and time.monotonic() - self.alarm_since < self.input_ack_min_s:
            return False
        now = time.monotonic()
        self._ack_until = now + self.ack_quiet_s
        acknowledged_reason, acknowledged_id = self.alarm_reason, self.alarm_id
        self.visits.acknowledge(acknowledged_id, now)
        waited = now - self.alarm_since
        self.acked = True
        self.last_ack = {"via": via, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "alarm_id": acknowledged_id}
        self.alarm = None
        self.alarm_id = None
        self.alarm_reason = ""
        self.buzzer.set(None)
        self.loudness.end()
        # Silencing must precede potentially slow storage or diagnostics.
        self._mark_alert_acknowledged(acknowledged_id, via, waited)
        self.log_event("ack", f"via {via}: {acknowledged_reason}",
                       alarm_id=acknowledged_id, after_s=round(waited, 1))
        return True

    @staticmethod
    def _load_audio_settings(zones_path) -> dict:
        """Apply the escalation tuning from zones.json to the audio policy."""
        import alarm_audio
        alarm_audio.reset_configuration()
        try:
            data = json.loads(Path(zones_path).read_text(encoding="utf-8-sig"))
            alarms = data.get("alarms") or {}
            block = alarms.get("escalation")
        except (OSError, ValueError, AttributeError, TypeError):
            alarms, block = {}, None
        alarm_audio.configure(block)
        alarm_audio.set_theme(alarms.get("sound_theme") if isinstance(alarms, dict) else None)
        alarm_audio.set_start_level(alarms.get("start_level") if isinstance(alarms, dict) else None)
        return alarm_audio.settings()

    @staticmethod
    def _load_soften_let_in(zones_path) -> bool:
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("alarms") or {}).get(
                "soften_other_lane", True)
        except (OSError, ValueError, AttributeError, TypeError):
            value = True
        return value if isinstance(value, bool) else True

    @staticmethod
    def _load_fault_remind(zones_path) -> float:
        """Seconds between reminders while an acknowledged fault keeps detection paused (0 = off)."""
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("alarms") or {}).get(
                "fault_remind_s", 120.)
        except (OSError, ValueError, AttributeError, TypeError):
            value = 120.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 120.
        return 0. if value <= 0 else max(60., min(900., float(value)))

    def _fault_evidence(self, kind: str, now: float) -> dict:
        """What was wrong, kept with the fault event (it used to be empty)."""
        evidence = {"source": "fault", "fault_kind": kind,
                    "blind_s": round(now - (self._fault_since or now), 1)}
        try:
            if kind == "alignment":
                status = self.alignment or {}
                evidence["alignment"] = {key: status.get(key) for key in (
                    "state", "message", "reason", "matches", "inliers", "inlier_ratio", "weak_reasons",
                    "reference_index", "references") if key in status}
                evidence["alignment"]["bootstrap"] = dict(getattr(self, "_bootstrap_info", {}) or {})
            elif kind == "ai" and self.semantic:
                status = self.semantic.status()
                evidence["ai"] = {key: status.get(key) for key in ("state", "error", "age_s", "available")
                                  if key in status}
            elif kind == "detector":
                evidence["error"] = str(self._frame_error)[:300]
            return json.loads(json.dumps(evidence, default=str))
        except Exception:
            return {"source": "fault", "fault_kind": kind}

    @staticmethod
    def _load_ack_quiet(zones_path) -> float:
        try:
            data = json.loads(Path(zones_path).read_text(encoding="utf-8-sig"))
            value = (data.get("alarms") or {}).get("ack_quiet_s", 10.)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return 10.
            return max(5., min(30., float(value)))
        except (OSError, ValueError, AttributeError, TypeError):
            return 10.

    def _update_alarm_audio(self) -> None:
        with self._state_lock:
            self._finish_already_opening()
            if not self.alarm:
                workflow = getattr(self, 'barrier_workflow', None)
                active = (self.armed and not self._test_tone
                          and time.monotonic() >= self._ack_until
                          and not self._alignment_blocked
                          and workflow is not None
                          and self.barrier_rules.get('reminder', True)
                          and workflow.status().get('reminder_active') is True)
                if active:
                    if self.loudness.kind != 'barrier_open':
                        self._reminder_seq = getattr(self, '_reminder_seq', 0) + 1
                        self._reminder_since = time.monotonic()
                        self.loudness.begin('barrier_open', source='verified_barrier')
                        self.buzzer.set('barrier_open')
                    self.loudness.evidence(fresh=True, present=True, confirmed=False,
                                          source='verified_barrier')
                    self.loudness.tick()
                elif self.loudness.kind == 'barrier_open':
                    self.buzzer.set(None)
                    self.loudness.end()
                return
            now = time.monotonic()
            if self.alarm == "fault":
                evidence = {"fresh": True, "present": bool(self.fault_reason), "confirmed": False}
            else:
                evidence = self.visits.evidence_for_alarm(self.alarm_id, now)
                # A newly validated image warning is allowed its first pulse.
                # Subsequent repeats need the current route observation.
                if now-self.alarm_since < 1.:
                    evidence = {**evidence, "fresh": True, "present": True}
                elif (now-self.last_frame_at >= 3 or self._alignment_blocked):
                    evidence = {**evidence, "fresh": False, "present": False}
            self.loudness.evidence(fresh=evidence.get("fresh") is True,
                                  present=evidence.get("present") is True,
                                  confirmed=evidence.get("confirmed") is True,
                                  source=self.alarm_details.get("source", "unknown"))
            self.loudness.tick()

    def _observe_barrier_service(self, frame, stamp):
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is None or stamp is None:
            return
        with self._state_lock:
            alarm = ({'id': self.alarm_id, 'route': VisitMemory.route_key(self.alarm_details),
                      'started_at': self.alarm_since}
                     if self.alarm and self.alarm != 'fault' else None)
            pending = {route for ident, route in self._service_pending if ident == self.alarm_id}
            primary = self.semantic.status() if self.semantic else {}
            trusted = (not self._alignment_blocked and self.alignment.get('state') == 'tracking'
                       and self._local_frame_fresh(stamp)
                       and time.monotonic()-self.warmup_started >= self.args.warmup)
            armed = self.armed
        try:
            with self._pipeline_measure('barriers', source_at=stamp):
                intents = workflow.observe(frame, stamp, primary, self.zones, trusted,
                                           alarm, pending, enabled=armed)
            self._barrier_error = ''
            current = workflow.status()
            # Cross-check service decisions against this frame's angle reading,
            # rather than the previous frame (which may show the opposite pose).
            self._observe_arm_angles(frame, stamp, trusted, current)
            for intent in intents:
                if intent.get('type') == 'resolve':
                    self._resolve_service_visit(intent)
                elif intent.get('type') in ('raised', 'lowered'):
                    self._silence_for_barrier_change(intent)
            self._ack_on_any_arm(current, stamp)
            self._note_arm_memory(current)
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            self._barrier_error = f'Barrier check unavailable: {type(error).__name__}: {error}'
            # Swallowing the traceback leaves a fault nobody can diagnose from
            # the machine itself. Record it once a minute: enough to find the
            # cause, never enough to fill the log while the fault persists.
            now = time.monotonic()
            if now - getattr(self, '_barrier_error_logged_at', 0.0) > 60.0:
                self._barrier_error_logged_at = now
                try:
                    with (STATE_DIR / 'watcher.log').open('a', encoding='utf-8') as handle:
                        handle.write(f'\n[barrier] {self._barrier_error}\n'
                                     + traceback.format_exc() + '\n')
                except OSError:
                    pass
            workflow.reset()
            self._arm_live = {}
            tracker = getattr(self, '_arm_tracker', None)
            if tracker is not None:
                tracker.reset()

    def _resolve_service_visit(self, intent):
        """Resolve only the observed event; never start global quiet time."""
        if not isinstance(intent, dict) or intent.get('type') != 'resolve':
            return False
        with self._state_lock:
            now = time.monotonic()
            ident, route = intent.get('alarm_id'), intent.get('route')
            started, occupied, clear, barrier_at, observed = (intent.get(key) for key in
                ('alarm_started_at', 'occupied_at', 'clear_since', 'barrier_observed_at', 'observed_at'))
            if (not self.armed or self._pending_rearm or self.alarm not in ('main', 'early')
                    or not ident or ident != self.alarm_id
                    or route != VisitMemory.route_key(self.alarm_details)
                    or self._alignment_blocked or self.alignment.get('state') != 'tracking'
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in (started, occupied, clear, barrier_at, observed))
                    or started != self.alarm_since
                    or not started <= occupied <= clear <= observed
                    or not started <= barrier_at
                    or observed-clear < 2.0
                    or not self._local_frame_fresh(observed)
                    or not self._local_frame_fresh(barrier_at)
                    or not self._alarm_lane()
                    or intent.get('barrier_state') not in ('up', 'down')
                    or self._tracker_agrees(self._alarm_lane(), intent.get('barrier_state')) is False
                    or self._competing_visit_blocks_arm_ack()):
                return False
            workflow = getattr(self, 'barrier_workflow', None)
            if workflow is None or workflow.status().get('trusted') is not True:
                return False
            # These calls are deliberately separate from _acknowledge_locked:
            # a second lane/car must remain immediately eligible to alert.
            self.buzzer.set(None)
            self.loudness.end()
            self.visits.acknowledge(ident, now)
            waited, reason = now-self.alarm_since, self.alarm_reason
            self.alarm = self.alarm_id = None
            self.alarm_reason = ''
            self.alarm_details = {}
            self.acked = False
            workflow.resolved(intent)
            self._mark_alert_acknowledged(ident, 'auto-road-clear', waited)
            presence = self.presence.status() if hasattr(self.presence, 'status') else {}
            activity = presence.get('desk_activity', {})
            self.log_event('auto-resolved', reason, alarm_id=ident, route=route,
                           after_s=round(waited, 1), evidence=intent,
                           desk_movement=activity.get('state') == 'movement')
            return True

    def _silence_for_barrier_change(self, intent):
        """The guard moved this lane's arm during its alarm: they are dealing
        with the car, so the siren stops. Raising and lowering each have their
        own switch in Settings. Only this visit is settled; another lane or a
        new car can still alert at once (no global quiet time)."""
        if not isinstance(intent, dict) or intent.get('type') not in ('raised', 'lowered'):
            return False
        lowered = intent.get('type') == 'lowered'
        if not self.barrier_rules.get('ack_on_lower' if lowered else 'ack_on_raise', True):
            return False
        with self._state_lock:
            now = time.monotonic()
            ident, route = intent.get('alarm_id'), intent.get('route')
            arm = self._alarm_lane()
            origin, bound_start, position_since, observed = (intent.get(key) for key in
                ('origin_observed_at', 'alarm_started_at', 'position_since', 'observed_at'))
            if (not self.armed or self._pending_rearm or self.alarm not in ('main', 'early')
                    or not ident or ident != self.alarm_id
                    or route != VisitMemory.route_key(self.alarm_details)
                    or self._alignment_blocked or self.alignment.get('state') != 'tracking'
                    or not self._local_frame_fresh(intent.get('observed_at'))
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in (origin, bound_start, position_since, observed))
                    or bound_start != self.alarm_since
                    or not self.alarm_since <= origin <= position_since <= observed
                    or observed-origin > self.ARM_CHANGE_WINDOW_S
                    or not arm
                    or self._tracker_agrees(arm, 'down' if lowered else 'up') is False
                    or self._competing_visit_blocks_arm_ack()
                    or self._arm_motion_predates_alarm(arm)):
                return False
            workflow = getattr(self, 'barrier_workflow', None)
            if workflow is None or workflow.status().get('trusted') is not True:
                return False
            self.buzzer.set(None)
            self.loudness.end()
            self.visits.acknowledge(ident, now)
            waited, reason = now - self.alarm_since, self.alarm_reason
            self.alarm = self.alarm_id = None
            self.alarm_reason = ''
            self.alarm_details = {}
            self.acked = True
            word = 'lowered' if lowered else 'raised'
            workflow.resolved({**intent, 'reason': f'barrier_{word}'})
            self._mark_alert_acknowledged(ident, f'barrier-{word}', waited)
            self.log_event('auto-barrier-down' if lowered else 'auto-barrier-up', f'{reason} (arm {word})',
                           alarm_id=ident, route=route, after_s=round(waited, 1), evidence=intent)
            return True

    _silence_for_raised_barrier = _silence_for_barrier_change

    ARM_CHANGE_WINDOW_S = 30.0   # a moving arm reads "unknown"; its old position counts this long

    def _alarm_lane(self):
        """Only a vehicle lane can be served by a physical barrier."""
        role = self.alarm_details.get('route_role')
        if role:
            return role if role in ('exit', 'entrance') else None
        route = VisitMemory.route_key(self.alarm_details)
        return 'exit' if route.endswith('exit') else 'entrance' if route.endswith('entrance') else None

    # -- exit corner early evidence and "already opening" (16 Nov 2026 audit) --------------------------------------
    AI_BOOST_S = 4.0              # look at the lanes every frame this long after the exit corner changes
    LANE_SEEN_GAP_S = 10.0        # lane evidence older than this belongs to an earlier vehicle
    PRE_RISE_S = 4.0              # an arm that began rising this shortly before an alarm is already serving it
    SERVED_CHIRP_S = 1.2          # such an alarm sounds this long (a chirp), then stops by itself

    def _measure_taillights(self, zone, frame, shape) -> dict:
        from exit_entry import measure_taillights
        try:
            metrics = measure_taillights(frame, self.bg, zone.mask_for(shape[1], shape[0]),
                                         red_pct=float(zone.taillight_red_pct),
                                         change_pct=float(zone.taillight_change_pct),
                                         wake_pct=float(zone.taillight_wake_pct))
        except Exception:                                   # noqa: BLE001 - one bad frame never stops detection
            return {"red_pct": 0.0, "changed_pct": 0.0, "hit": False, "wake": False, "lane_seen": False}
        now = time.monotonic()
        semantic = getattr(self, 'semantic', None)
        if metrics.get("wake") and semantic is not None and hasattr(semantic, "boost"):
            semantic.boost(self.AI_BOOST_S)
        if metrics.get("lane_seen"):
            lanes = getattr(self, '_lane_seen', None)
            if lanes is None:
                lanes = self._lane_seen = {}
            role = zone.route_role or "exit"
            seen = lanes.get(role)
            if not seen or now - seen[1] > self.LANE_SEEN_GAP_S:
                seen = [now, now]
            seen[1] = now
            lanes[role] = seen
        return metrics

    def _note_arm_rise(self, arm, reading) -> None:
        """Remember when each arm last began to rise (monotonic frame time)."""
        if (reading.get('motion') != 'rising' or reading.get('ok') is not True
                or reading.get('position_conflict') is True):
            return
        started = reading.get('motion_started')
        if isinstance(started, bool) or not isinstance(started, (int, float)) or not math.isfinite(started):
            return
        rises = getattr(self, '_recent_rise', None)
        if rises is None:
            rises = self._recent_rise = {}
        rises[arm] = float(started)

    def _vehicle_seen_since(self, lane, details, now):
        """When this lane's vehicle was first seen: exit-corner tail lights or its AI track (None if unknown)."""
        times = []
        seen = (getattr(self, '_lane_seen', None) or {}).get(lane)
        if seen and now - seen[1] <= self.LANE_SEEN_GAP_S:
            times.append(seen[0])
        first = (details or {}).get('first_observed_at')
        if isinstance(first, (int, float)) and not isinstance(first, bool) and math.isfinite(first):
            times.append(float(first))
        return min(times) if times else None

    def _consider_already_opening(self, kind, details, now) -> None:
        """The guard is already raising this lane's barrier for the vehicle that caused this alarm: let it chirp,
        then stop by itself (logged) instead of ringing until the arm comes down again."""
        self._served_pending = None
        rules = self.barrier_rules
        if (kind not in ('main', 'early') or not rules.get('ack_on_movement', True)
                or not rules.get('ack_on_raise', True) or not rules.get('ack_already_opening', True)):
            return
        lane = self._alarm_lane()
        started = (getattr(self, '_recent_rise', None) or {}).get(lane) if lane else None
        if started is None or not 0 <= now - started <= self.PRE_RISE_S:
            return
        if self._alignment_blocked or (self.alignment or {}).get('state') != 'tracking':
            return
        seen = self._vehicle_seen_since(lane, details, now)
        if seen is None or seen > started - .2:
            return          # not shown to be this vehicle: the rise may belong to the one before
        if self._competing_visit_blocks_arm_ack():
            return
        self._served_pending = (self.alarm_id, now + self.SERVED_CHIRP_S, lane, started)
        self.alarm_details['already_opening'] = {'arm': lane, 'rise_before_alarm_s': round(now - started, 2),
                                                 'seen_before_rise_s': round(started - seen, 2)}

    def _finish_already_opening(self) -> None:
        with self._state_lock:
            pending = getattr(self, '_served_pending', None)
            if not pending:
                return
            ident, due, lane, started = pending
            if self.alarm_id != ident or self.alarm not in ('main', 'early') or self.acked:
                self._served_pending = None
                return
            now = time.monotonic()
            finite = lambda value: (isinstance(value, (int, float)) and not isinstance(value, bool)
                                    and math.isfinite(value))
            if not all(finite(value) for value in (due, started, self.alarm_since)):
                self._served_pending = None
                return
            if now < due:
                return
            # The delayed chirp is only an offered service decision. A newer
            # lane/arm observation, competing car or setting can invalidate it.
            # Cancellation leaves the original alarm sounding and unanswered.
            self._served_pending = None
            rules = self.barrier_rules
            if (not self.armed or self._pending_rearm or self._alarm_lane() != lane
                    or self._alignment_blocked or (self.alignment or {}).get('state') != 'tracking'
                    or not all(rules.get(key, True) for key in ('ack_on_movement', 'ack_on_raise', 'ack_already_opening'))
                    or not 0 <= self.alarm_since-started <= self.PRE_RISE_S
                    or not 0 <= now-due < 3.
                    or (getattr(self, '_recent_rise', None) or {}).get(lane) != started
                    or not finite(self.last_frame_at) or not self._local_frame_fresh(self.last_frame_at, now)
                    or self._competing_visit_blocks_arm_ack()):
                return
            seen = self._vehicle_seen_since(lane, self.alarm_details, now)
            if not finite(seen) or seen > started-.2:
                return
            reading = (getattr(self, '_arm_live', None) or {}).get(lane) or {}
            observed, wall = reading.get('at'), reading.get('at_wall')
            if (reading.get('ok') is not True or reading.get('position_conflict') is True
                    or not finite(observed) or not self.alarm_since <= observed <= self.last_frame_at
                    or not 0 <= now-observed < 1.5 or not finite(wall) or not 0 <= time.time()-wall < 1.5):
                return
            rising = reading.get('motion') == 'rising' and reading.get('motion_started') == started
            raised = reading.get('motion') != 'lowering' and self._tracker_agrees(lane, 'up') is True
            if not (rising or raised):
                return
            route = VisitMemory.route_key(self.alarm_details)
            evidence = self.visits.evidence_for_alarm(ident, now)
            if (evidence.get('visit_id') != self.alarm_details.get('visit_id')
                    or evidence.get('route') != route or evidence.get('acknowledged') is True
                    or evidence.get('retired') is True):
                return
            workflow = getattr(self, 'barrier_workflow', None)
            if workflow is not None:
                status = workflow.status()
                if not isinstance(status, dict) or status.get('trusted') is not True:
                    return
                service = ((status.get('service') or {}).get('routes') or {}).get(route)
                if service and service.get('alarm_id') != ident:
                    return
            reason, waited = self.alarm_reason, now-self.alarm_since
            self.buzzer.set(None)
            self.loudness.end()
            self.visits.acknowledge(ident, now)
            self.alarm = self.alarm_id = None
            self.alarm_reason = ''
            self.alarm_details = {}
            self.acked = True
            self._mark_alert_acknowledged(ident, 'barrier-already-opening', waited)
            self.log_event('auto-barrier-up', f'{reason} ({lane} arm was already rising {now-started:.1f} s before: '
                           'the guard had already let it through)', alarm_id=ident, route=route, arm=lane,
                           after_s=round(waited, 1), already_opening=True)

    def _arm_matches_alarm(self, arm) -> bool:
        """An arm in another lane cannot establish that this visitor was served."""
        role = self._alarm_lane()
        return arm == role and role is not None

    def _competing_visit_blocks_arm_ack(self) -> bool:
        route = VisitMemory.route_key(self.alarm_details)
        if (self.alarm_id, route) in getattr(self, '_service_pending', set()):
            return True
        workflow = getattr(self, 'barrier_workflow', None)
        status = workflow.status() if workflow is not None else {}
        if not isinstance(status, dict):
            return False
        current = (status.get('service') or {}).get('routes', {}).get(route, {})
        # The workflow keeps this veto between AI frames. A camera-only tick
        # must not forget that another visit was waiting in this lane.
        return (current.get('alarm_id') == self.alarm_id
                and current.get('blocked_by_new_arrival') is True)

    def _arm_motion_predates_alarm(self, arm) -> bool:
        reading = (getattr(self, '_arm_live', None) or {}).get(arm) or {}
        started = reading.get('motion_started')
        return (reading.get('motion') in ('rising', 'lowering')
                and isinstance(started, (int, float)) and not isinstance(started, bool)
                and math.isfinite(started) and started < self.alarm_since)

    def _ack_on_any_arm(self, status, stamp) -> bool:
        """A matching lane's newly observed movement may settle its current visit."""
        now = time.monotonic()
        changes = []
        known = getattr(self, '_arm_last_known', None)
        if known is None:
            known = self._arm_last_known = {}
        if not isinstance(status, dict) or status.get('trusted') is not True:
            known.clear()
            return False
        for row in (status or {}).get('barriers', []) if isinstance(status, dict) else []:
            if not isinstance(row, dict):
                continue
            arm, state = row.get('id'), row.get('state')
            if state not in ('up', 'down') or not row.get('fresh') or not row.get('calibrated'):
                continue
            observed = row.get('observed_at')
            if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                    or not math.isfinite(observed) or observed > stamp
                    or not self._local_frame_fresh(observed)):
                continue
            last = known.get(arm)
            if last and observed <= last[1]:
                continue
            if self._tracker_agrees(arm, state) is False:
                continue        # the live angle says it did not move (e.g. a car under a raised arm): keep the old state
            known[arm] = (state, observed)
            if last and last[0] != state and now - last[1] <= self.ARM_CHANGE_WINDOW_S:
                changes.append((arm, state, last[1], row.get('position_since')))
                if state == 'up':
                    raised = getattr(self, '_arm_raised_at', None)
                    if raised is None:
                        raised = self._arm_raised_at = {}
                    raised[arm] = now
        if not changes or not self.barrier_rules.get('ack_any_arm', True):
            return False
        for arm, state, old_seen_at, position_since in changes:
            lowered = state == 'down'
            if not self.barrier_rules.get('ack_on_lower' if lowered else 'ack_on_raise', True):
                continue
            with self._state_lock:
                if (not self.armed or self._pending_rearm or self.alarm not in ('main', 'early')
                        or not self.alarm_id or self._alignment_blocked
                        or self.alignment.get('state') != 'tracking'
                        or old_seen_at < self.alarm_since
                        or isinstance(position_since, bool) or not isinstance(position_since, (int, float))
                        or not math.isfinite(position_since)
                        or not old_seen_at <= position_since <= stamp
                        or not self._arm_matches_alarm(arm)
                        or self._tracker_agrees(arm, state) is False
                        or self._arm_motion_predates_alarm(arm)
                        or self._competing_visit_blocks_arm_ack()
                        or not self._local_frame_fresh(stamp)):
                    continue
                if (status or {}).get('trusted') is not True:
                    continue
                ident, reason, waited = self.alarm_id, self.alarm_reason, now - self.alarm_since
                route = VisitMemory.route_key(self.alarm_details)
                self.buzzer.set(None)
                self.loudness.end()
                self.visits.acknowledge(ident, now)
                self.alarm = self.alarm_id = None
                self.alarm_reason = ''
                self.alarm_details = {}
                self.acked = True
                word = 'lowered' if lowered else 'raised'
                self._mark_alert_acknowledged(ident, f'barrier-{word}', waited)
                self.log_event('auto-barrier-down' if lowered else 'auto-barrier-up',
                               f'{reason} ({arm} arm {word}; matching lane)', alarm_id=ident, route=route,
                               arm=arm, after_s=round(waited, 1), any_arm=True)
                return True
        return False

    # -- which lane caused it? (label only, never silences) ------------------
    def _note_ai_activity(self, primary, now) -> None:
        """Vehicles/people the AI sees in any zone mark that lane active and count as "real" for learning."""
        try:
            self._note_ai_activity_unsafe(primary, now)
        except Exception:                                  # noqa: BLE001 - evidence only, never blocks detection
            pass

    def _note_ai_activity_unsafe(self, primary, now) -> None:
        detections = primary.get("detections") if isinstance(primary, dict) else None
        if not isinstance(detections, list) or not detections:
            return
        zones = {zone.name: zone for zone in self.zones}
        routes = {}
        for item in detections:
            if not isinstance(item, dict) or item.get("label") not in AI_SEEN_LABELS:
                continue
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)) or confidence < AI_SEEN_MIN_CONFIDENCE:
                continue
            for name in (item.get("region_status") or {}):
                zone = zones.get(name)
                if zone is not None and zone.route_group:
                    routes[zone.route_group] = zone.route_role
        learning = getattr(self, "learning", None)
        for route, role in routes.items():
            self._route_activity[route] = (now, role)
            if learning is not None:
                learning.note_recognised(route)

    def _alarm_context(self, zone, primary, now) -> dict:
        """Extra evidence for a light/change alarm: other-lane activity and what the AI saw.

        Labels only: any error here returns nothing, so it can never delay or stop an alarm.
        """
        try:
            return self._alarm_context_unsafe(zone, primary, now)
        except Exception:                                  # noqa: BLE001
            return {}

    def _alarm_context_unsafe(self, zone, primary, now) -> dict:
        context = {}
        if zone.route_role in ("exit", "entrance"):
            own = self._own_arm_open(zone.route_role, now)
            if own:
                context["own_arm_up"] = own
            other_arm = "entrance" if zone.route_role == "exit" else "exit"
            raised_at = (getattr(self, "_arm_raised_at", None) or {}).get(other_arm)
            if raised_at is not None and 0 <= now - raised_at <= LET_IN_S:
                ago = now - raised_at
                context["cross_lane"] = {
                    "route": other_arm, "role": other_arm, "seconds_ago": round(ago, 1), "arm_raised": True,
                    "text": (f"Probably the vehicle let in at the {other_arm} barrier (its arm went up {ago:.0f} s "
                             f"earlier). It still sounded; lowering that arm stops it.")}
                return {**context, **self._ai_seen(primary)}
            others = [(route, now - at, role) for route, (at, role) in self._route_activity.items()
                      if route != zone.route_group and role in ("exit", "entrance") and 0 <= now - at <= CROSS_LANE_S]
            if others:
                route, ago, role = min(others, key=lambda item: item[1])
                side = "coming in" if role == "entrance" else "going out"
                context["cross_lane"] = {"route": route, "role": role, "seconds_ago": round(ago, 1),
                                         "text": f"Possibly the lights of a vehicle {side} on the other lane "
                                                 f"({ago:.0f} s earlier). It still sounded; mark it Wrong or Real."}
        return {**context, **self._ai_seen(primary)}

    OPEN_ARM_MIN_S = 10.0   # an arm raised only just before the alarm is the normal "let through"

    def _own_arm_open(self, arm, now):
        """This lane's arm was already up when its alarm started (left open): label it, change nothing."""
        last = (getattr(self, "_arm_last_known", None) or {}).get(arm)
        if not last or last[0] != "up" or not 0 <= now - last[1] <= self.ARM_CHANGE_WINDOW_S:
            return None
        raised = (getattr(self, "_arm_raised_at", None) or {}).get(arm)
        up_for = round(now - raised, 1) if raised is not None and raised <= now else None
        if up_for is not None and up_for < self.OPEN_ARM_MIN_S:
            return None
        held = f" (up {up_for:.0f} s)" if up_for is not None else ""
        return {"arm": arm, "up_for_s": up_for,
                "text": (f"The {arm} barrier was already open{held}, so the vehicle could drive straight "
                         f"through. It still sounded; no arm moved, so a key or click answers it.")}

    @staticmethod
    def _ai_seen(primary) -> dict:
        seen = []
        detections = primary.get("detections") if isinstance(primary, dict) else None
        for item in (detections if isinstance(detections, list) else [])[:6]:
            if isinstance(item, dict) and item.get("label"):
                confidence = item.get("confidence")
                seen.append({"label": str(item["label"]),
                             "confidence": round(float(confidence), 2) if isinstance(confidence, (int, float)) else None,
                             "zones": {str(k): str(v) for k, v in (item.get("region_status") or {}).items()}})
        return {"ai_seen": seen} if seen else {}

    # -- operator reports: "Wrong alarm" / "Real" --------------------------
    def label_alert(self, alarm_id, verdict, via: str = 'web') -> dict:
        """The operator says a recent alert was wrong or real (or takes it back).

        The verdict is shown in Activity, replaces the automatic guess in alert
        learning, and the picture plus evidence are kept in the feedback folder
        for later review and training. It never silences anything by itself.
        """
        if verdict not in ('wrong', 'real', 'clear'):
            raise ValueError('The report must be wrong, real or clear.')
        if not isinstance(alarm_id, str) or not 1 <= len(alarm_id) <= 128:
            raise ValueError('An alarm identifier is required.')
        with self._state_lock:
            alert = next((item for item in self.recent_alerts if item.get('id') == alarm_id), None)
            if alert is None:
                raise LookupError('That alert is no longer in the recent list.')
            if alert.get('kind') == 'fault':
                raise ValueError('System faults cannot be marked wrong or real.')
            alert['feedback'] = None if verdict == 'clear' else {'verdict': verdict,
                                                                 'at': time.strftime('%H:%M:%S')}
            self.alerts_version += 1
            evidence = dict(alert.get('evidence') or {})
            snapshot, reason = alert.get('snapshot'), str(alert.get('reason') or '')
            stamp = f"{alert.get('date', '')}T{alert.get('time', '')}"
            kind = alert.get('kind')
        learned = None
        if getattr(self, 'learning', None) is not None:
            try:
                learned = self.learning.note_feedback(alarm_id, verdict, evidence)
            except ValueError:
                learned = None
        self._keep_feedback_example(alarm_id, verdict, kind, snapshot, evidence, stamp, reason)
        self.log_event('feedback', f'{verdict}: {reason}'[:300], alarm_id=alarm_id, verdict=verdict, via=via,
                       alert_at=stamp, alert_kind=kind, snapshot=snapshot, evidence=evidence)
        return {'ok': True, 'alarm_id': alarm_id, 'verdict': verdict, 'learning': learned}

    def _keep_feedback_example(self, alarm_id, verdict, kind, snapshot, evidence, stamp, reason) -> None:
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', alarm_id)[:64]
        try:
            for folder in ('wrong', 'real'):
                for old in (FEEDBACK_DIR / folder).glob(safe + '.*'):
                    old.unlink(missing_ok=True)
            if verdict == 'clear':
                return
            target = FEEDBACK_DIR / verdict
            target.mkdir(parents=True, exist_ok=True)
            if isinstance(snapshot, str) and ALERT_NAME.match(snapshot) and (ALERT_DIR / snapshot).is_file():
                shutil.copyfile(ALERT_DIR / snapshot, target / f'{safe}.jpg')
            recorder = getattr(self, 'recorder', None)
            clip = recorder.recording_for(alarm_id) if recorder is not None else None
            if clip:
                shutil.copyfile(RECORDINGS_DIR / clip, target / f'{safe}.mp4')
            (target / f'{safe}.json').write_text(json.dumps({
                'alarm_id': alarm_id, 'verdict': verdict, 'alert_at': stamp, 'kind': kind, 'reason': reason,
                'snapshot': snapshot, 'marked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'evidence': evidence},
                indent=1, default=str), encoding='utf-8')
            kept = sorted(target.glob('*.json'), key=lambda item: item.stat().st_mtime)
            for old in kept[:-FEEDBACK_KEEP]:
                old.unlink(missing_ok=True)
                old.with_suffix('.jpg').unlink(missing_ok=True)
                old.with_suffix('.mp4').unlink(missing_ok=True)
        except OSError as exc:
            self._note_error('Saving an alert report', exc)

    @staticmethod
    def _load_display_name(zones_path) -> str:
        from settings_store import DEFAULT_NAME, NAME_PATTERN
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("system") or {}).get("name")
        except (OSError, ValueError, AttributeError, TypeError):
            value = None
        value = " ".join(value.split()) if isinstance(value, str) else ""
        return value if NAME_PATTERN.fullmatch(value) else DEFAULT_NAME

    def _barrier_learned(self, entry: dict) -> None:
        """Every self-learned barrier example is logged with the picture it came from."""
        picture = None
        with self._preview_lock:
            jpeg = self.preview_jpeg
        if jpeg:
            folder = STATE_DIR / "barrier-learning"
            picture = f"{time.strftime('%Y%m%d-%H%M%S')}-{entry.get('arm')}-{entry.get('position')}.jpg"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / picture).write_bytes(jpeg)
                for old in sorted(folder.glob("*.jpg"))[:-40]:
                    old.unlink(missing_ok=True)
            except OSError:
                picture = None
        self.log_event("barrier-learned", f"{entry.get('arm')} arm {entry.get('position')}: learned a new lighting "
                                          f"example (match was {entry.get('match')})", picture=picture, entry=entry)

    def undo_barrier_learning(self, via: str = 'web') -> dict:
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is None or not hasattr(workflow, 'undo_last_learning'):
            return {'ok': False, 'reason': 'Barrier detection is not running'}
        result = workflow.undo_last_learning()
        if result.get('ok'):
            self.log_event('barrier-learning-undo', f"undid {result.get('undone') or result.get('restored')}", via=via)
        return result

    def learning_status(self) -> dict:
        from zone_editor import read_config
        status = self.learning.status(read_config(Path(self.args.zones))) if getattr(self, 'learning', None) else {}
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is not None and hasattr(workflow, 'learned'):
            status['barrier_learning'] = list(workflow.learned)[-8:]
            status['barrier_learning_skips'] = [{'arm': key[0], 'position': key[1], **value}
                                                for key, value in getattr(workflow, 'learning_skips', {}).items()]
            path = getattr(getattr(workflow, 'monitor', None), 'storage_path', None)
            status['barrier_undo_available'] = bool(isinstance(path, (str, os.PathLike)) and any(Path(path).parent.glob(
                Path(path).name + '.before-learn-*')))
        return status

    # -- live settings ---------------------------------------------------
    @staticmethod
    def _load_barrier_rules(zones_path) -> dict:
        rules = {'ack_on_raise': True, 'ack_on_lower': True, 'ack_any_arm': True, 'ack_on_movement': True,
                 'reminder': True,
                 'reminder_after_s': 15., 'reminder_every_s': 15.}
        try:
            block = json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("barriers") or {}
        except (OSError, ValueError, AttributeError, TypeError):
            block = {}
        for key in ('ack_on_raise', 'ack_on_lower', 'ack_any_arm', 'ack_on_movement', 'reminder'):
            if isinstance(block.get(key), bool):
                rules[key] = block[key]
        for key in ('reminder_after_s', 'reminder_every_s'):
            value = block.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                rules[key] = min(600., max(15., float(value)))
        return rules

    def _apply_barrier_rules(self) -> None:
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is not None:
            workflow.cycles.open_after_s = self.barrier_rules['reminder_after_s']
            workflow.cycles.reminder_interval_s = self.barrier_rules['reminder_every_s']

    @staticmethod
    def _load_ai_engine(zones_path) -> str:
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("ai") or {}).get("engine")
        except (OSError, ValueError, AttributeError, TypeError):
            value = None
        return value if value in ("graphics", "processor") else "graphics"

    def _feed_sightings(self, primary) -> None:
        """Clean crops of followed vehicles for the car memory. Never affects detection or alarms."""
        sightings = getattr(self, 'sightings', None)
        if sightings is None or not sightings.enabled or self.semantic is None or self._alignment_blocked:
            return
        try:
            stamp = primary.get('observed_at') if isinstance(primary, dict) else None
            frame = self.semantic.frame_at(stamp) if stamp is not None else None
            sightings.observe(primary.get('detections') or [], stamp, frame, self.zones)
        except Exception:                                   # noqa: BLE001 - evidence only
            pass

    @staticmethod
    def _load_sightings_enabled(zones_path) -> bool:
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("memory") or {}).get(
                "vehicle_sightings", True)
        except (OSError, ValueError, AttributeError, TypeError):
            value = True
        return value if isinstance(value, bool) else True

    @staticmethod
    def _load_ai_whole_picture(zones_path) -> bool:
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("ai") or {}).get(
                "whole_picture", True)
        except (OSError, ValueError, AttributeError, TypeError):
            value = True
        return value if isinstance(value, bool) else True

    @staticmethod
    def _load_ai_entrance_verifier(zones_path) -> bool:
        try:
            block = json.loads(Path(zones_path).read_text(encoding='utf-8-sig')).get('ai') or {}
            return block.get('entrance_verifier') is True and block.get('whole_picture', True) is True
        except (OSError, ValueError, AttributeError, TypeError):
            return False

    @staticmethod
    def _load_ai_interval(zones_path) -> float:
        try:
            value = (json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("ai") or {}).get("interval_s", .5)
        except (OSError, ValueError, AttributeError, TypeError):
            value = .5
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return .5
        return min(2., max(.25, float(value)))

    def settings_snapshot(self) -> dict:
        import settings_store
        from zone_editor import read_config
        import alarm_audio
        config = read_config(Path(self.args.zones))
        return {"sections": settings_store.describe(config),
                "live": {"sound": alarm_audio.settings(), "barrier_rules": dict(self.barrier_rules),
                         "recordings": self.recorder.status() if getattr(self, 'recorder', None) else None,
                         "ai_interval_s": getattr(self.semantic, 'interval', None) if self.semantic else None,
                         "armed": self.armed},
                "learning": self.learning_status() if hasattr(self, 'learning_status') else None,
                "name": getattr(self, 'display_name', None) or 'BarrierWatcher'}

    def update_settings(self, changes: dict) -> dict:
        """Validate, back up, save and apply live. Zone outlines never change here."""
        import settings_store
        from zone_editor import read_config, validate_config, _atomic_write
        with self._settings_lock:
            path = Path(self.args.zones)
            original = read_config(path)
            updated, changed = settings_store.apply_changes(original, changes)
            updated = validate_config(updated, updated)
            stamp = time.strftime('%Y%m%dT%H%M%S')
            backups = sorted(path.parent.glob(path.name + '.settings-*'))
            for old in backups[:-20]:
                old.unlink(missing_ok=True)
            _atomic_write(path.with_name(f'{path.name}.settings-{stamp}-{os.getpid()}'), path.read_bytes())
            _atomic_write(path, (json.dumps(updated, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            self._apply_live_settings(updated, changed)
            if 'system.name' in changed:
                self.display_name = self._load_display_name(path)
        self.log_event('settings', ', '.join(f'{key}={changes[key]}' for key in changed)[:400])
        return {"ok": True, "changed": changed}

    def _apply_live_settings(self, config: dict, changed: list) -> None:
        import alarm_audio
        args_zones = self.args.zones
        if any(key.startswith('alarms.') for key in changed):
            theme_before = alarm_audio.theme()
            self.alarm_audio_settings = self._load_audio_settings(args_zones)
            self.ack_quiet_s = self._load_ack_quiet(args_zones)
            self.fault_remind_s = self._load_fault_remind(args_zones)
            self.soften_let_in = self._load_soften_let_in(args_zones)
            if alarm_audio.theme() != theme_before:
                sounds = ensure_sounds(force=True)
                replace = getattr(self.buzzer, 'replace_sounds', None)
                if replace is not None:
                    replace(sounds)
                self.sounds = sounds
        if any(key.startswith('barriers.') for key in changed):
            self.barrier_rules = self._load_barrier_rules(args_zones)
            self._apply_barrier_rules()
        if any(key in changed for key in ('ai.interval_s', 'ai.whole_picture')) and self.semantic is not None:
            self.semantic.interval = self._load_ai_interval(args_zones)
            self.semantic.context_view = self._load_ai_whole_picture(args_zones)
            if getattr(self, 'sightings', None) is not None:
                self.sightings.enabled = self._load_sightings_enabled(args_zones)
                self.semantic.keep_frames = self.sightings.enabled
        if 'alignment.hold_s' in changed:
            hold = (config.get('alignment') or {}).get('hold_s', 10.)
            self.alignment_hold_s = min(30., max(0., float(hold)))
        if 'power.stop_when_camera_gone_s' in changed:
            self.camera_gone_s = self._load_camera_gone_limit(args_zones)
        if any(key.startswith('recordings.') for key in changed) and getattr(self, 'recorder', None) is not None:
            self.recorder.configure(**self._load_recording_settings(args_zones))
        if any(key.startswith('zone:') for key in changed):
            with self._state_lock:
                self._pending_zones = [Zone(**{**item, 'rect': tuple(item['rect'])}) for item in config['zones']]

    def request_refresh(self, via: str = 'web') -> None:
        """Restart the watcher cleanly: the supervisor starts a fresh one at once."""
        self.log_event('refresh', f'system refresh requested via {via}')
        try:
            (STATE_DIR / 'restart-request').write_text(time.strftime('%Y-%m-%dT%H:%M:%S') + '\n', encoding='utf-8')
        except OSError:
            pass

        def stop_soon():
            time.sleep(.6)          # let the HTTP reply reach the page first
            self._running = False
        threading.Thread(target=stop_soon, daemon=True, name='refresh').start()

    def barrier_status(self):
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is None:
            return {'configured': False, 'barriers': [], 'error': getattr(self, '_barrier_error', '')}
        status = {**workflow.status(), 'error': self._barrier_error, **self._arm_memory_status()}
        live, rows = getattr(self, '_arm_live', None) or {}, []
        for row in status.get('barriers', []):
            reading = live.get(row.get('id')) if isinstance(row, dict) else None
            recent = reading and 0 <= time.time() - reading.get('at_wall', 0) < 1.5
            if recent and reading.get('position_conflict') is True and row.get('state') in ('up', 'down'):
                row = {**row, 'classifier_state': row['state'], 'state': 'unknown',
                       'confidence': 0., 'active_line': None, 'candidate_position': None,
                       'live_line': None, 'live_angle': None, 'live_motion': None,
                       'reason': 'Arm angle disagrees with calibrated position'}
            elif recent and reading.get('line'):
                row = {**row, 'live_line': reading['line'], 'live_angle': reading.get('theta'),
                       'live_motion': reading.get('motion')}
            rows.append(row)
        status['barriers'] = rows
        return status

    @staticmethod
    def _build_arm_tracker(zones_path):
        """Live arm angles from the arms' display geometry (fitted to the real arms) or their calibrated lines."""
        try:
            from arm_angle import ArmAngleTracker, geometry_from_config, geometry_from_lines
            block = json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("barriers") or {}
            saved, lines = block.get("display_geometry") or {}, block.get("lines") or {}
            geometry = {}
            for arm in ("exit", "entrance"):
                if isinstance(saved.get(arm), dict):
                    geometry[arm] = geometry_from_config(saved[arm], PROC_W, PROC_H)
                elif isinstance(lines.get(arm), dict) and lines[arm].get("down") and lines[arm].get("up"):
                    geometry[arm] = geometry_from_lines(lines[arm], PROC_W, PROC_H)
            return ArmAngleTracker(geometry, PROC_W, PROC_H) if geometry else None
        except Exception:                                   # noqa: BLE001 - display/early-ack aid only
            return None

    def _observe_arm_angles(self, frame, stamp, trusted, status) -> None:
        """About 6 times a second on the clean aligned frame: each arm's live angle, for the on-screen line and to
        stop an alarm as soon as an arm starts moving. Never raises; the strict reading stays the authority."""
        tracker = getattr(self, '_arm_tracker', None)
        if tracker is None or not trusted or frame is None:
            if tracker is not None:
                tracker.reset()
            self._arm_live = {}
            self._arm_last_run = 0.
            return
        now = time.monotonic()
        if now - getattr(self, '_arm_last_run', 0.) < .15:
            return
        self._arm_last_run = now
        try:
            out = tracker.update(frame, stamp)
            strict = {row.get('id'): row for row in (status or {}).get('barriers', []) if isinstance(row, dict)}
            for arm, reading in out.items():
                row = strict.get(arm) or {}
                sure = row.get('state') if row.get('fresh') and row.get('state') in ('up', 'down') else None
                reading['strict'] = sure
                reading['position_conflict'] = False
                if sure is not None and reading.get('ok') is True and reading.get('theta') is not None:
                    if abs(reading['theta'] - (0. if sure == 'down' else 90.)) > 20:
                        reading['line'] = None
                        reading['position_conflict'] = True  # Neither reading can prove a position.
            self._arm_live = {arm: {**reading, 'at_wall': time.time()} for arm, reading in out.items()}
            recorder = getattr(self, '_arm_motion_recorder', None)
            if recorder is not None:
                recorder.add(frame, stamp, out)
            for arm, reading in out.items():
                if reading.get('motion'):
                    self._note_arm_rise(arm, reading)
                    if self.alarm not in ('main', 'early') and reading.get('ok') is True:
                        self._record_diagnostic_clip('barrier-motion',
                            f"{arm} arm {reading['motion']} without an active vehicle alarm", now)
                    self._ack_on_arm_motion(arm, reading)
        except Exception:                                   # noqa: BLE001
            # An unreadable update must break the movement sequence. Otherwise
            # pre-fault angles could be reported as a newly observed movement.
            tracker.reset()
            self._arm_live = {}

    def _ack_on_arm_motion(self, arm, reading) -> bool:
        """Stop the current alarm as soon as an arm starts rising (the guard let the car through) or lowering."""
        rules = self.barrier_rules
        motion, started = reading.get('motion'), reading.get('motion_started')
        observed = reading.get('at')
        if (not rules.get('ack_on_movement', True) or motion not in ('rising', 'lowering')
                or reading.get('ok') is not True
                or reading.get('position_conflict') is True
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in (started, observed))
                or not 0 <= observed-started <= (3. if motion == 'rising' else 8.)):
            return False
        if not rules.get('ack_on_raise' if motion == 'rising' else 'ack_on_lower', True):
            return False
        start_state = 'down' if motion == 'rising' else 'up'
        if reading.get('strict') == start_state:
            return False                                    # the strict reading still sees it there: not moving
        memory = (getattr(self, '_arm_memory', None) or {}).get(arm)
        if not memory or memory.get('state') != start_state:
            return False                                    # e.g. an arm held open cannot "rise"
        key = (arm, motion, round(float(started), 2))
        if key in self._arm_motion_acked:
            return False
        with self._state_lock:
            if (not 0 <= time.monotonic()-observed < 1.5
                    or not self.armed or self._pending_rearm or self.alarm not in ('main', 'early') or not self.alarm_id
                    or self._alignment_blocked or self.alignment.get('state') != 'tracking'
                    or started < self.alarm_since
                    or self._competing_visit_blocks_arm_ack()):
                return False
            if not self._arm_matches_alarm(arm):
                return False
            self._arm_motion_acked.add(key)
            ident, reason, waited = self.alarm_id, self.alarm_reason, time.monotonic() - self.alarm_since
            route = VisitMemory.route_key(self.alarm_details)
            self.buzzer.set(None)
            self.loudness.end()
            self.visits.acknowledge(ident, time.monotonic())
            self.alarm = self.alarm_id = None
            self.alarm_reason = ''
            self.alarm_details = {}
            self.acked = True
            word = 'raised' if motion == 'rising' else 'lowered'
            self._mark_alert_acknowledged(ident, f'barrier-{word}', waited)
            self.log_event('auto-barrier-up' if motion == 'rising' else 'auto-barrier-down',
                           f'{reason} ({arm} arm {motion}: stopped as it started moving)', alarm_id=ident,
                           route=route, arm=arm, after_s=round(waited, 1), early=True)
            return True

    ARM_CHANGE_HOLD_S = 1.0

    def _tracker_agrees(self, arm, state):
        """True/False when the live angle tracker (smoothed, fresh) agrees the arm is up/down; None if it has no
        reading. Two independent methods must agree before an arm counts as moved."""
        reading = (getattr(self, '_arm_live', None) or {}).get(arm)
        if not reading:
            return None
        age = time.time() - reading.get('at_wall', 0)
        # Smoothing intentionally holds the previous angle across a brief raw
        # disagreement. That display history cannot overrule current contrary
        # evidence or silence an alarm while the position is shown as Unknown.
        if 0 <= age < 1.5 and reading.get('position_conflict') is True:
            return False
        if not 0 <= age <= 1.0 or reading.get('shown') is None:
            return None
        shown = float(reading['shown'])
        return shown <= 25. if state == 'down' else shown >= 65. if state == 'up' else None

    def _arm_change_confirmed(self, arm, state, now) -> bool:
        """A strict up/down change counts when the tracker agrees, or (no tracker reading) after holding 1 s."""
        agrees = self._tracker_agrees(arm, state)
        if agrees is False:
            self._arm_pending.pop(arm, None)
            return False
        if agrees:
            return True
        pending = self._arm_pending.get(arm)
        if not pending or pending[0] != state:
            self._arm_pending[arm] = (state, now)
            return False
        return now - pending[1] >= self.ARM_CHANGE_HOLD_S

    def _note_arm_memory(self, status) -> None:
        """Each arm's last confirmed position and since when; every real change is logged ("arm").

        Display and history only. The strict reading (which drops to unknown while glare, a vehicle or the
        moving arm hides it) still decides every acknowledgement.
        """
        try:
            memory = getattr(self, '_arm_memory', None)
            if memory is None:
                memory = self._arm_memory = {}
            if not hasattr(self, '_arm_pending'):
                self._arm_pending = {}
                self._arm_history = deque(maxlen=20)
            now, wall = time.monotonic(), time.time()
            for row in (status or {}).get('barriers', []) if isinstance(status, dict) else []:
                if not isinstance(row, dict):
                    continue
                arm, state = row.get('id'), row.get('state')
                if (arm not in ('exit', 'entrance') or state not in ('up', 'down')
                        or not row.get('fresh') or not row.get('calibrated')):
                    continue
                previous = memory.get(arm)
                if previous is not None and previous['state'] == state:
                    previous['seen'] = now
                    self._arm_pending.pop(arm, None)
                    continue
                if previous is not None and not self._arm_change_confirmed(arm, state, now):
                    continue
                memory[arm] = {'state': state, 'since': now, 'since_wall': wall, 'seen': now}
                if previous is not None:
                    held = round(now - previous['since'], 1)
                    self._arm_history.append({'arm': arm, 'state': state, 'previous_for_s': held,
                                              'at': time.strftime('%H:%M:%S', time.localtime(wall))})
                    self.log_event('arm', f"{arm.title()} barrier {state} (was {previous['state']} for {held:.0f} s)",
                                   arm=arm, state=state, previous_for_s=held)
        except Exception:                                   # noqa: BLE001 - display and history only
            pass

    def _arm_memory_status(self) -> dict:
        now, rows = time.monotonic(), {}
        for arm, item in dict(getattr(self, '_arm_memory', None) or {}).items():
            try:
                rows[arm] = {'state': item['state'], 'for_s': round(now - item['since'], 1),
                             'since': time.strftime('%H:%M:%S', time.localtime(item['since_wall'])),
                             'seen_ago_s': round(now - item['seen'], 1)}
            except (KeyError, TypeError, ValueError):
                continue
        return {'memory': rows, 'history': list(getattr(self, '_arm_history', ()))[-12:]}

    def calibrate_barrier(self, payload):
        workflow = getattr(self, 'barrier_workflow', None)
        if workflow is None:
            raise ValueError('Barrier monitoring is unavailable')
        with self._state_lock:
            if payload.get('action') in ('capture', 'capture-road'):
                primary = self.semantic.status() if self.semantic else {}
                if (self._alignment_blocked or self.alignment.get('state') != 'tracking'
                        or primary.get('state') != 'ready' or primary.get('detections')):
                    raise ValueError('Capture requires verified alignment and clear lanes')
        result = workflow.calibrate(payload)
        if result.get('ok'):
            self.log_event('barrier-calibration', f"{payload.get('id')}: {payload.get('action')} {payload.get('position', '')}")
        return result

    @staticmethod
    def _load_visit_quiet(zones_path) -> float:
        """Seconds one approach stays quiet after the guard acknowledges it.

        0 disables the grouping entirely and every zone alarms on its own again.
        """
        try:
            config = json.loads(Path(zones_path).read_text(encoding="utf-8-sig"))
            value = float((config.get("alarms") or {}).get("visit_quiet_s", VISIT_QUIET_S))
        except (OSError, ValueError, AttributeError, TypeError):
            return VISIT_QUIET_S
        if value <= 0:
            return 0.0
        return min(600.0, max(5.0, value))

    def _visit_status(self) -> dict:
        return self.visits.status(time.monotonic())

    @staticmethod
    def _route_key(details) -> str:
        """Which approach an alarm belongs to; the zone itself when it has none."""
        details = details or {}
        return str(details.get("route_group") or details.get("zone_name") or "")

    @staticmethod
    def _load_presence_config(zones_path) -> PresenceConfig:
        """Webcam presence settings; any problem leaves the safe defaults in place."""
        try:
            config = json.loads(Path(zones_path).read_text(encoding="utf-8-sig"))
            return PresenceConfig.from_dict(config.get("presence"))
        except (OSError, ValueError, AttributeError, TypeError):
            return PresenceConfig()

    @staticmethod
    def _load_camera_gone_limit(zones_path) -> float:
        """Seconds without the camera before the program closes; 0 disables it."""
        try:
            config = json.loads(Path(zones_path).read_text(encoding="utf-8-sig"))
            value = (config.get("power") or {}).get("stop_when_camera_gone_s", CAMERA_GONE_S)
        except (OSError, ValueError, AttributeError, TypeError):
            return CAMERA_GONE_S
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return CAMERA_GONE_S
        if value <= 0:
            return 0.0
        return float(min(3600.0, max(30.0, value)))

    def _request_full_stop(self, reason: str) -> None:
        """Close the whole program: the camera is gone, not merely slow."""
        now = time.monotonic()
        if self._stop_requested_at and now - self._stop_requested_at < 60:
            return
        self._stop_requested_at = now
        # Dispatch before optional device cleanup: a blocked desk camera or
        # speaker must not prevent the whole-program stop controller starting.
        # That controller requests disarm before its bounded forced cleanup.
        self.log_event("power", f"stopping the whole program: {reason}")
        try:
            from power_control import request_full_off
            request_full_off()
            self.buzzer.set(None)
        except Exception as exc:
            self.control_warning = 'Automatic Full OFF failed. Use BarrierWatcher OFF on the desktop (or F4) and check shutdown status.'
            self.log_event("power", f"automatic stop could not start: {type(exc).__name__}: {exc}")

    def _noise_grid(self, columns: int = 32, rows: int = 18):
        noise = self.bg_noise
        if noise is None:
            return None
        try:
            import cv2
            small = cv2.resize(np.asarray(noise, dtype=np.float32), (columns, rows),
                               interpolation=cv2.INTER_AREA)
            return [[round(float(value), 1) for value in row] for row in small]
        except Exception:
            return None

    def _stick_still_connected(self) -> bool:
        """True only when the USB capture stick was found at the reader's latest (re)start."""
        reader = getattr(self, 'reader', None)
        return (getattr(reader, 'device_present', None) is True
                and str(getattr(self.args, 'source', '')).startswith('usbcam:'))

    def camera_gone_status(self, now: float) -> dict:
        """How long the camera has been missing, and how long until it closes."""
        gap = now - self.last_frame_at
        missing = gap if self.first_frame_seen and gap > self.args.lost_after else 0.0
        connected = self._stick_still_connected() if missing else None
        stays_on = bool(connected and self.camera_gone_s and missing > self.camera_gone_s)
        return {"missing_s": round(missing, 1), "limit_s": self.camera_gone_s,
                "stick_connected": connected, "staying_on": stays_on,
                "stopping_in_s": (None if stays_on else round(max(0.0, self.camera_gone_s - missing), 1)
                                  if missing and self.camera_gone_s else None)}

    def play_test_tone(self, seconds: float = 2.0, *, lane: str = "") -> bool:
        """Sound the siren briefly so the speaker can be checked before a shift.

        A real alarm always owns the buzzer: the test refuses to start during
        one, and never silences an alarm that begins while it is playing.
        """
        if not isinstance(lane, str) or lane not in ('', 'entrance', 'exit'):
            raise ValueError('Choose entrance or exit for a lane sound test')
        with self._state_lock:
            if self.alarm:
                raise PermissionError("Acknowledge the current alarm before testing the siren")
            if self._test_tone is not None:
                raise PermissionError('A sound test is already playing')
            token = object()
            self._test_tone = token
            self.loudness.begin("early", source="test", confirmed=False, lane=lane)
            self.buzzer.set("main")
        self.log_event("test", f"{lane or 'standard'} attention tone test for {seconds:.0f}s", lane=lane)

        def finish():
            time.sleep(max(.2, seconds))
            with self._state_lock:
                if self._test_tone is not token or self.alarm:
                    return
                self._test_tone = None
                self.buzzer.set(None)
                self.loudness.end()

        threading.Thread(target=finish, daemon=True, name="siren-test").start()
        return True

    def input_status(self) -> dict:
        """Small, camera-free response for the Windows input helper."""
        with self._state_lock:
            reminder = not self.alarm and self.loudness.kind == 'barrier_open'
            reminder_id = f"reminder:{getattr(self, '_reminder_seq', 0)}" if reminder else None
            return {"armed": self.armed,
                    "alarm_id": self.alarm_id if self.alarm else reminder_id,
                    "alarm_started_at": (self.alarm_since if self.alarm else
                                         getattr(self, '_reminder_since', None) if reminder else None),
                    "ack_quiet_s": self.ack_quiet_s,
                    "ack_remaining_s": round(max(0., self._ack_until-time.monotonic()), 2)}

    def note_input_helper(self) -> None:
        """The keyboard/mouse helper polls about five times a second."""
        self._input_helper_seen = time.monotonic()

    def _input_helper_status(self) -> dict:
        seen = self._input_helper_seen
        age = None if seen is None else time.monotonic() - seen
        return {"connected": age is not None and age < 5, "age_s": None if age is None else round(age, 1)}

    # -- alert history ---------------------------------------------------
    def _remember_alert(self, ts: str, kind: str, reason: str, alarm_id, snapshot, acked=None, evidence=None) -> None:
        date, _, clock = str(ts).partition("T")
        self.recent_alerts.appendleft({"id": alarm_id, "date": date, "time": clock, "kind": kind,
                                       "reason": reason, "snapshot": snapshot, "acked": acked,
                                       "evidence": dict(evidence or {}), "feedback": None})
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
                    "alerts": [{**alert, "acked": dict(alert["acked"]) if alert["acked"] else None,
                                "feedback": dict(alert["feedback"]) if alert.get("feedback") else None,
                                **self._recording_fields(alert.get("id"))}
                               for alert in self.recent_alerts]}

    def _recording_fields(self, alarm_id) -> dict:
        recorder = getattr(self, 'recorder', None)
        if recorder is None or not alarm_id:
            return {"recording": None, "recording_pending": False}
        name = recorder.recording_for(alarm_id)
        return {"recording": name, "recording_pending": name is None and recorder.pending_for(alarm_id)}

    def _recording_saved(self, name: str, alarm_ids) -> None:
        """A clip was written: the Activity page can now play it."""
        ids = set(alarm_ids)
        with self._state_lock:
            self.alerts_version += 1
            marked = [(alert["id"], alert["feedback"]["verdict"]) for alert in self.recent_alerts
                      if alert.get("id") in ids and alert.get("feedback")]
        self.log_event("recording", f"alarm video saved: {name}", clip=name, alarm_ids=sorted(ids))
        for alarm_id, verdict in marked:
            self._copy_feedback_clip(alarm_id, verdict, name)

    def _copy_feedback_clip(self, alarm_id, verdict, name) -> None:
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', alarm_id)[:64]
        try:
            target = FEEDBACK_DIR / verdict
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RECORDINGS_DIR / name, target / f'{safe}.mp4')
        except OSError as exc:
            self._note_error('Saving a reported alarm video', exc)

    @staticmethod
    def _load_recording_settings(zones_path) -> dict:
        settings = {"enabled": True, "before_s": 10., "after_s": 20., "keep": 25}
        try:
            block = json.loads(Path(zones_path).read_text(encoding="utf-8-sig")).get("recordings") or {}
        except (OSError, ValueError, AttributeError, TypeError):
            block = {}
        if isinstance(block.get("enabled"), bool):
            settings["enabled"] = block["enabled"]
        for key, low, high in (("before_s", 0., 20.), ("after_s", 10., 50.), ("keep", 5, 200)):
            value = block.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                settings[key] = min(high, max(low, value))
        settings["keep"] = int(settings["keep"])
        return settings

    # Automatic answers are logged under their own kinds; restore them too (a restart showed 11
    # barrier acknowledgements as "Not acknowledged" on 14 Nov 22:15).
    RESTORED_ACK_VIA = {"auto-barrier-up": "barrier-raised", "auto-barrier-down": "barrier-lowered",
                        "auto-resolved": "auto-road-clear", "alarm-replaced": "replaced"}

    def _load_recent_alerts(self) -> None:
        """After a restart, rebuild the dashboard's alarm history from the log tail."""
        try:
            with EVENT_LOG.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                start = max(0, handle.tell() - 1024 * 1024)
                handle.seek(start)
                lines = handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return
        acks, reports = {}, {}
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
                                     alarm_id, snapshot, evidence=record.get("evidence") if isinstance(record.get("evidence"),dict) else None)
            elif (kind == "ack" or kind in self.RESTORED_ACK_VIA) and isinstance(record.get("alarm_id"), str):
                acks[record["alarm_id"]] = record
            elif (kind == "feedback" and isinstance(record.get("alarm_id"), str)
                    and record.get("verdict") in ("wrong", "real", "clear")):
                reports[record["alarm_id"]] = record
        for alert in self.recent_alerts:
            ack = acks.get(alert["id"]) if alert["id"] else None
            if ack:
                via = (self.RESTORED_ACK_VIA.get(ack.get("kind"))
                       or str(ack.get("detail", "")).partition(":")[0].removeprefix("via "))
                after = ack.get("after_s")
                alert["acked"] = {"via": via, "after_s": after if isinstance(after, (int, float)) else None}
            report = reports.get(alert["id"]) if alert["id"] else None
            if report and report.get("verdict") in ("wrong", "real"):
                alert["feedback"] = {"verdict": report["verdict"], "at": str(report.get("ts", ""))[11:19]}

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
                if shutil.disk_usage(ALERT_DIR).free < MIN_FREE_BYTES:
                    for old in sorted(ALERT_DIR.glob("*.jpg"))[:60]:
                        old.unlink(missing_ok=True)
                    if shutil.disk_usage(ALERT_DIR).free < MIN_FREE_BYTES:
                        continue      # Detection matters more than evidence.
                _atomic_write(ALERT_DIR / name, jpeg)
                # Names begin with the timestamp, so name order is age order.
                for old in sorted(ALERT_DIR.glob("*.jpg"))[:-ALERT_KEEP]:
                    old.unlink(missing_ok=True)
            except OSError:
                pass   # A full disk must never interfere with alarms.
            finally:
                self._snapshots.task_done()

    # -- detection -------------------------------------------------------
    @staticmethod
    def _local_frame_fresh(received_at, now=None) -> bool:
        """Three-second local age limit; None supports direct synthetic callers."""
        if received_at is None:
            return True
        if (isinstance(received_at, bool) or not isinstance(received_at, (int, float))
                or not math.isfinite(received_at)):
            return False
        moment = time.monotonic() if now is None else now
        return 0 <= moment-received_at < 3.0

    def _accept_local_frame(self, received_at, now=None) -> bool:
        fresh = self._local_frame_fresh(received_at, now)
        if not fresh and getattr(self, "pipeline", None) is not None:
            self.pipeline.record_stale_drop()
        return fresh

    def process(self, frame: np.ndarray, received_at=None) -> bool:
        if not self._accept_local_frame(received_at):
            return False
        with self._state_lock:
            if not self._accept_local_frame(received_at):
                return False
            if hasattr(self, '_service_pending'):
                self._service_pending.clear()
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
        if rearm:
            if getattr(self, '_approach_warnings', None) is not None:
                self._approach_warnings.reset()
            alignment_reason = getattr(self, '_alignment_rearm_reason', None)
            if alignment_reason:
                self._alignment_rearm_reason = None
                self.log_event('alignment-reset', alignment_reason,
                               warmup_s=self.args.warmup, view=getattr(self, 'view_name', None))
            if self.semantic:
                self.semantic.reset()
            # Consume the re-arm request here, on the detector thread only.
            self.bg = self.bg_noise = None
            self._light_credits.clear()
            for zone in self.zones:
                zone.motion_active = zone.light_active = False
                zone._change_previous = None
                zone._counts = {k: 0 for k in zone._counts}
                zone._notified = {"motion": False, "light": False}
        if not hasattr(self, "lighting_monitor"):
            self.lighting_monitor = LightingMonitor()
        self.lighting_monitor.update(np.asarray(frame, dtype=np.uint8), scene_rect=(.15, .14, .72, .53))
        if not self._accept_local_frame(received_at):
            return False
        gray = (frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114)
        if not self._accept_local_frame(received_at):
            return False
        if self.bg is None:
            self.bg = gray.copy()
            # Start cautious: assume a little flicker everywhere until measured.
            self.bg_noise = np.full_like(gray, INITIAL_NOISE)
            self.warmup_started = time.monotonic()
            return True
        diff = compensated_motion_diff(gray, self.bg)
        if not self._accept_local_frame(received_at):
            return False
        warm = (time.monotonic() - self.warmup_started) < self.args.warmup
        light_candidates, motion_candidates = [], []
        for zone in self.zones:
            ys, xs = zone.slice_for(gray.shape[1], gray.shape[0])
            region = zone.mask_for(gray.shape[1], gray.shape[0])[ys, xs]
            # The urgent road channel measures both light and dark changes.
            # Spatial/noise validation below rejects unsupported flicker.
            sub_diff = (np.abs(gray[ys, xs] - self.bg[ys, xs])[region]
                        if zone.strict_change else diff[ys, xs][region])
            area = max(1, sub_diff.size)
            threshold = zone.change_threshold if zone.strict_change else self.args.pixel_threshold
            if not zone.strict_change:
                # Do not briefly publish raw speckle counts for an urgent zone
                # while its filtered measurement is still being calculated.
                zone.last_pct = float((sub_diff > threshold).sum()) * 100.0 / area
            zone.last_delta = float((gray[ys, xs]-self.bg[ys, xs])[region].mean()) if region.any() else 0.0
            light_hit = zone.trigger != "ai" and zone.light and zone.last_delta >= zone.light_delta
            if zone.trigger == "ai_light" and zone.light:
                zone.light_metrics = measure_local_brightness(gray, self.bg,
                    zone.light_mask_for(gray.shape[1], gray.shape[0]),
                    scene_mask=self._light_scene_mask(gray.shape), min_delta=zone.light_delta,
                    noise=self.bg_noise, noise_k=NOISE_K)
                if not self._accept_local_frame(received_at):
                    return False
                zone.last_delta = zone.light_metrics["positive_delta"]
                light_hit = zone.light_metrics["hit"]
                if light_hit and zone.light_metrics.get("support_bounds"):
                    self._note_highlight(zone, zone.light_metrics["support_bounds"], gray.shape, "light")
            motion_hit = zone.trigger not in ("ai", "ai_light") and zone.motion and zone.last_pct >= zone.motion_pct
            if zone.trigger == "taillight":
                # Exit entry corner: lit red tail lights moving in (white headlights of entering cars do not).
                zone.light_metrics = self._measure_taillights(zone, frame, gray.shape)
                zone.last_delta = zone.light_metrics.get("red_pct", 0.0)
                zone.last_pct = zone.light_metrics.get("changed_pct", 0.0)
                light_hit, motion_hit = bool(zone.light_metrics.get("hit")), False
                if light_hit and zone.light_metrics.get("support_bounds"):
                    self._note_highlight(zone, zone.light_metrics["support_bounds"], gray.shape, "light")
            if zone.strict_change and zone.motion:
                zone.change_metrics = measure_urgent_change(gray, self.bg,
                    zone.mask_for(gray.shape[1], gray.shape[0]),
                    scene_mask=self._light_scene_mask(gray.shape),
                    threshold=zone.change_threshold, min_patch_pct=zone.motion_pct,
                    min_connected_pct=zone.connected_pct,
                    noise=self.bg_noise)
                measured_at = time.monotonic()
                if not self._accept_local_frame(received_at, measured_at):
                    return False
                zone.last_pct = zone.change_metrics["changed_pct"]
                motion_hit = zone.change_metrics["hit"]
                # Confirmation spacing belongs to the observed frames, not to
                # completion times which can bunch up after a processing stall.
                moment = measured_at if received_at is None else received_at
                bounds = zone.change_metrics["support_bounds"]
                previous = zone._change_previous
                weak = zone.change_metrics["weak_hit"]
                repeated = False
                if weak and bounds and previous and 0 < moment-previous[0] <= 1.0:
                    x,y,w,h = bounds
                    px,py,pw,ph = previous[1]
                    overlap = max(0,min(x+w,px+pw)-max(x,px))*max(0,min(y+h,py+ph)-max(y,py))
                    repeated = overlap >= .3 * min(w*h,pw*ph)
                zone._change_previous = (moment, bounds) if weak and bounds and not warm else None
                zone.change_metrics["confirmation_frames"] = 1 if motion_hit else 2
                motion_hit = motion_hit or repeated
                zone.change_metrics["confirmed"] = bool(motion_hit)
                metrics = zone.change_metrics
                if metrics.get("support_bounds") and (metrics.get("hit") or metrics.get("weak_hit")):
                    self._note_highlight(zone, metrics["support_bounds"], gray.shape, "change")
            if not self._accept_local_frame(received_at):
                return False
            if warm:
                # Warmup must not consume an arrival's only rising edge.
                zone.motion_active = zone.light_active = False
                zone._counts = {k: 0 for k in zone._counts}
                continue
            if (light_hit or motion_hit) and zone.route_group:
                self._route_activity[zone.route_group] = (time.monotonic(), zone.route_role)
            zone.step("light", light_hit)
            zone.step("motion", motion_hit)
            # A pending episode survives quiet time, but an accepted episode
            # remains consumed until the channel actually clears.
            if light_hit and zone.light_active and not zone._notified["light"]:
                light_candidates.append(zone)
            if motion_hit and zone.motion_active and not zone._notified["motion"]:
                motion_candidates.append(zone)

        now = time.monotonic()
        primary = self.semantic.status() if self.semantic else {}
        if not self._accept_local_frame(received_at):
            return False
        frame_details = {} if received_at is None else {"frame_received_at": received_at}
        approach_warnings = self._measure_approach_warnings(frame, primary, received_at, warm=warm)
        if not self._accept_local_frame(received_at):
            return False
        self.visits.observe(now, self.zones, primary,
                            frame_fresh=not warm and not self._alignment_blocked,
                            width=frame.shape[1], height=frame.shape[0])
        # Fast corner warnings still precede inference, with the same soft onset.
        for zone in motion_candidates:
            if not self._accept_local_frame(received_at):
                return False
            if not zone.strict_change:
                continue
            zone._notified["motion"] = self.raise_alarm(zone.severity,
                f"{zone.name}: immediate change alarm ({zone.last_pct:.2f}% changed)",
                details={"source": "motion", "strict_change": True, **frame_details,
                         "zone_name": zone.name, "route_group": zone.route_group,
                         "route_role": zone.route_role, "operator_hint": zone.operator_hint,
                         "changed_pct": zone.last_pct, "pixel_threshold": zone.change_threshold,
                         "confirmation_frames": zone.change_metrics.get("confirmation_frames", zone.on_frames),
                         "measurement": dict(zone.change_metrics),
                         **self._alarm_context(zone, primary, now)})
        for zone, detector, motion, proof in approach_warnings:
            if not self._accept_local_frame(received_at):
                return False
            details = {"source": "motion", "fusion": "forward motion and vehicle candidate",
                       "zone_name": zone.name, "route_group": zone.route_group,
                       "route_role": zone.route_role, "alert_stage": zone.alert_stage,
                       "operator_hint": zone.operator_hint, "direction_state": "toward",
                       "bbox_normalized": proof['bbox_normalized'], "label": proof['label'],
                       "observed_at": motion['observed_at'], "corroboration": proof,
                       "measurement": dict(motion), **frame_details,
                       **self._alarm_context(zone, primary, now)}
            if self.raise_alarm(zone.severity, f"{zone.name}: vehicle-shaped movement towards the entrance; advance warning",
                                details=details):
                detector.mark_emitted(motion['observed_at'])
        # Deliver the urgent road warning before any synchronous AI/tracker
        # work, so its latency or exception cannot prevent that siren.
        if not self._accept_local_frame(received_at):
            return False
        if self.semantic:
            events = (self.semantic.update(frame, self.armed and not warm)
                      if received_at is None else
                      self.semantic.update(frame, self.armed and not warm, stamp=received_at))
        else:
            events = []
        if not self._accept_local_frame(received_at):
            return False
        primary = self.semantic.status() if self.semantic else {}
        self._feed_sightings(primary)
        if not self._accept_local_frame(received_at):
            return False
        self._note_ai_activity(primary, time.monotonic())
        self.visits.observe(time.monotonic(), self.zones, primary,
                            frame_fresh=not warm and not self._alignment_blocked,
                            width=frame.shape[1], height=frame.shape[0])
        for kind, reason, event_id in events:
            context = self.semantic.event_context(event_id)
            if getattr(self, 'learning', None) is not None:
                self.learning.note_recognised((context or {}).get('route_group'))
            observed_at = (context or {}).get('observed_at', received_at)
            if not self._local_frame_fresh(observed_at):
                continue
            evidence_stamp = {} if observed_at is None else {'frame_received_at': observed_at}
            if self.raise_alarm(kind, reason, details={"source": "ai", **(context or {}), **evidence_stamp}):
                self.semantic.mark_emitted(event_id)
        for zone in sorted(light_candidates, key=lambda item: not item.priority):
            if not self._accept_local_frame(received_at):
                return False
            cause = "localized light change" if zone.trigger == "ai_light" else "brightness rose"
            text = f"{zone.name}: {cause} {zone.last_delta:+.1f}; approach warning"
            if zone.trigger == "taillight":
                text = (f"{zone.name}: vehicle tail lights entering the lane "
                        f"({zone.light_metrics.get('red_pct', 0):.1f}% red, "
                        f"{zone.light_metrics.get('changed_pct', 0):.0f}% changed); approach warning")
            zone._notified["light"] = self.raise_alarm(zone.light_severity,
                    text,
                    details={"source":"light", "zone_name":zone.name, **frame_details,
                             "trigger":zone.trigger,
                             "route_group":zone.route_group, "route_role":zone.route_role,
                             "operator_hint":zone.operator_hint,
                             "alert_stage":zone.alert_stage,
                             "measurement":dict(zone.light_metrics),
                             **self._alarm_context(zone, primary, time.monotonic())})
        for zone in motion_candidates:
            if not self._accept_local_frame(received_at):
                return False
            if zone.strict_change:
                continue
            zone._notified["motion"] = self.raise_alarm(zone.severity,
                f"{zone.name}: {zone.last_pct:.1f}% of the zone changed",
                details={"source":"motion", "zone_name":zone.name, **frame_details,
                         "route_group":zone.route_group, "route_role":zone.route_role,
                         **self._alarm_context(zone, primary, time.monotonic())})
        self._observe_barrier_service(frame, received_at)
        self._update_alarm_audio()
        if not self._accept_local_frame(received_at):
            return False

        # Background adapts slowly, so a parked car stops re-triggering but the
        # latched alarm from its arrival stays up until a human clears it.
        alpha = self.args.adapt
        deviation = np.abs(gray - self.bg)
        # Float32 recurrence can creep slightly above 255 at saturated pixels.
        self.bg = np.clip((1.0 - alpha) * self.bg + alpha * gray, 0, 255)
        # Learn the flicker more slowly than the background, so one bright
        # arrival cannot teach the scene to ignore that spot.
        if self.bg_noise is None:
            self.bg_noise = np.full_like(gray, INITIAL_NOISE)
        slow = alpha / NOISE_ADAPT_DIVISOR
        self.bg_noise = np.clip((1.0 - slow) * self.bg_noise + slow * deviation,
                                MIN_NOISE, MAX_NOISE)
        return True

    def _measure_approach_warnings(self, frame, primary, received_at, *, warm):
        """Two independent cues; plain movement never raises this alarm."""
        if not any(getattr(zone, 'approach_motion', False) for zone in self.zones):
            if getattr(self, '_approach_warnings', None) is not None:
                self._approach_warnings.reset()
            return []
        from approach_warning import ApproachWarnings
        if getattr(self, '_approach_warnings', None) is None:
            self._approach_warnings = ApproachWarnings()
        enabled = (self.armed and not warm and not self._alignment_blocked
                   and self._local_frame_fresh(received_at))
        stamp = time.monotonic() if received_at is None else received_at
        return self._approach_warnings.update(frame, self.zones, primary, stamp, enabled=enabled)

    def check_health(self, now: float) -> None:
        with self._state_lock:
            self._check_health_locked(now)
        if getattr(self, 'learning', None) is not None:
            self.learning.tick()

    def _check_health_locked(self, now: float) -> None:
        self._expire_ack_locked(time.monotonic())
        gap = now - self.last_frame_at
        # Whole-program camera power policy applies while alerts are paused too.
        # Pausing sound must not disable removal of the camera/input processes.
        if self.first_frame_seen and self.camera_gone_s and gap > self.camera_gone_s:
            if self._stick_still_connected():
                # Frozen or blank source with the stick plugged in: keep alarming.
                if not getattr(self, '_kept_on_logged', False):
                    self._kept_on_logged = True
                    self.log_event("power", f"no picture for {gap:.0f}s but the capture stick is still "
                                            "connected; staying ON and alarming (pull the stick or use OFF to stop)")
            else:
                self._request_full_stop(f"camera offline for {gap:.0f}s")
        elif gap < 5:
            self._kept_on_logged = False
        if not self.armed:
            self.fault_reason = ""
            self._fault_kind = None
            self._fault_since = None
            self._ai_bad_since = None
            return
        ai_required = any(z.trigger in ("ai", "ai_light") for z in self.zones)
        ml = self.semantic.status() if ai_required and self.semantic else {}
        age = ml.get("age_s")
        ai_ready = (ml.get("available") and ml.get("state") == "ready"
                    and isinstance(age, (int, float)) and math.isfinite(age) and 0 <= age < 3)
        if ai_ready:
            self._ai_was_ready = True
        self._ai_state_now = ml.get("state")
        if ai_required and not ai_ready:
            if self._ai_bad_since is None:
                self._ai_bad_since = now
        else:
            self._ai_bad_since = None
        gap = now - self.last_frame_at
        limit = self.args.lost_after if self.first_frame_seen else self.args.connect_grace
        # The camera worked and then stopped. A reconnect or a slow network
        # recovers well inside this window; an unplugged camera does not.
        if gap > limit:
            kind = "lost"
            self.fault_reason = (f"no usable frame for {gap:.0f}s"
                                 + ("" if self.first_frame_seen else " since startup"))
        elif self._frame_error_since and now - self._frame_error_since > 3:
            kind = "detector"
            self.fault_reason = f"{self._frame_error[:160]} (detection is not running)"
        elif getattr(self, '_menu_since', None) and now - self._menu_since > 2:
            kind = "keyboard_menu"
            self.fault_reason = ("The keyboard's menu is open over the camera picture: zone alerts are paused. "
                                 "Close the menu and show the camera picture full-screen")
        elif (self._alignment_blocked and self.first_frame_seen and self._alignment_bad_since
              and now - self._alignment_bad_since > (
                  (6 if self._zoom_tolerance() > 0 else 2) if self._alignment_used is not None else 12)):
            # Just after a start nothing can be held yet; a passing car's headlights may hide the
            # landmarks for a few seconds (14 Nov 21:36). Allow 12 s to lock before calling it a fault.
            kind = "alignment"
            framing = (self.alignment or {}).get("framing") or {}
            self.fault_reason = ("Scene alignment lost: zone alerts are paused. Check the keyboard shows "
                                 "a saved camera view (ANPR Entrance or PTZ 2) full-screen"
                                 + (f". {framing['text']}" if framing.get("text") else ""))
        elif self.identical_since and now - self.identical_since > self.args.freeze_after:
            kind = "frozen"
            self.fault_reason = (f"byte-identical frames for "
                                 f"{now - self.identical_since:.0f}s (frozen feed)")
        elif self.dark_since and now - self.dark_since > self.args.black_after:
            kind = "black"
            self.fault_reason = f"picture black for {now - self.dark_since:.0f}s"
        elif (self.first_frame_seen and self._ai_bad_since is not None
              and now - self._ai_bad_since > 5 and now - self.started > 15
              and not (not getattr(self, '_ai_was_ready', False) and now - self.started < 75
                       and getattr(self, '_ai_state_now', None) == 'loading')):
            kind = "ai"
            self.fault_reason = "Object detection unavailable; AI regions cannot report arrivals"
        else:
            self.fault_reason = ""
            self._fault_kind = None
            self._fault_since = None
            return
        if self._fault_since is None:
            self._fault_since = now
        # ACK silences this fault episode. Recovery, a different fault, or a
        # deliberate re-arm enables the next fault alarm; the status stays visible.
        if kind != self._fault_kind:
            if self.raise_alarm("fault", self.fault_reason, details=self._fault_evidence(kind, now)):
                self._fault_kind = kind
                self._fault_raised_at = now
                if self.semantic:
                    self.semantic.reset()
        elif (self.fault_remind_s and self.alarm is None
              and now - self._fault_raised_at >= self.fault_remind_s):
            # 13 Nov 22:53: an alignment fault acknowledged after 2 s left the
            # system blind for 13 minutes without another sound. Remind gently
            # (faults start at the low level) until detection is back.
            blind = now - self._fault_since
            reason = (f"Still not watching after {max(1, round(blind / 60))} min: {self.fault_reason}")
            if self.raise_alarm("fault", reason, details=self._fault_evidence(kind, now)):
                self._fault_raised_at = now

    # -- preview ---------------------------------------------------------
    HIGHLIGHT_S = 1.5        # a changed patch stays outlined this long after its last measurement
    HIGHLIGHT_COLOURS = {'alert': (255, 80, 95), 'zone': (255, 190, 70), 'seen': (75, 190, 255),
                         'person': (190, 150, 255)}

    @staticmethod
    def _note_highlight(zone, bounds, shape, kind) -> None:
        """Remember where in this zone the picture changed, to outline it (display only)."""
        try:
            height, width = shape[:2]
            x, y, w, h = (float(v) for v in bounds)
            if width <= 0 or height <= 0 or w <= 0 or h <= 0:
                return
            box = [round(max(0., x / width), 4), round(max(0., y / height), 4),
                   round(min(1., (x + w) / width), 4), round(min(1., (y + h) / height), 4)]
            zone._highlight = (time.monotonic(), box, kind)
        except (TypeError, ValueError):
            pass

    def _change_boxes(self) -> list:
        """Changed patches measured in the last moment, per zone (normalised x1, y1, x2, y2)."""
        if self._alignment_blocked:
            return []
        now, boxes = time.monotonic(), []
        for zone in self.zones:
            mark = getattr(zone, '_highlight', None)
            if mark and 0 <= now - mark[0] <= self.HIGHLIGHT_S:
                boxes.append({'zone': zone.name, 'box': list(mark[1]), 'kind': mark[2],
                              'alarm': bool(getattr(zone, 'active', False))})
        return boxes[:8]

    @staticmethod
    def _overlay_objects(observed) -> list:
        """Objects the AI is following, ready to outline: normalised box, recent path, zone and state."""
        objects = []
        for obj in list(observed or [])[:12]:
            try:
                box = obj.get('box_n')
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                confirmed = [str(name) for name in obj.get('confirmed_regions') or []]
                regions = [str(name) for name in obj.get('regions') or []]
                zone = (confirmed or regions or [''])[0]
                objects.append({
                    'id': obj.get('track_id'), 'label': str(obj.get('label', ''))[:20],
                    'confidence': round(float(obj.get('confidence', 0.)), 2),
                    'box': [round(float(v), 4) for v in box],
                    'state': 'alert' if confirmed else 'zone' if regions else 'seen',
                    'zone': zone, 'status': str((obj.get('region_status') or {}).get(zone, ''))[:40],
                    'trail': [[round(float(x), 4), round(float(y), 4)]
                              for x, y in list(obj.get('trail') or [])[-10:]]})
            except (TypeError, ValueError, AttributeError):
                continue
        return objects

    def _draw_objects(self, draw, objects) -> None:
        for obj in objects:
            key = 'person' if obj['label'] == 'person' and obj['state'] == 'seen' else obj['state']
            colour = self.HIGHLIGHT_COLOURS.get(key, (75, 190, 255))
            x1, y1, x2, y2 = (obj['box'][0] * PROC_W, obj['box'][1] * PROC_H,
                              obj['box'][2] * PROC_W, obj['box'][3] * PROC_H)
            trail = [(x * PROC_W, y * PROC_H) for x, y in obj['trail']]
            if len(trail) >= 2:
                draw.line(trail, fill=colour, width=2)
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=2 if obj['state'] == 'seen' else 3)
            text = f"{obj['label'].upper()} {obj['confidence']:.0%}" + (
                f" #{obj['id']}" if obj['id'] is not None else "")
            size = draw.textbbox((0, 0), text)
            tall = size[3] - size[1] + 4
            top = y1 - tall - 1 if y1 > tall + 16 else y2 + 1
            draw.rectangle([x1, top, x1 + size[2] - size[0] + 6, top + tall], fill=colour)
            draw.text((x1 + 3, top + 1), text, fill=(10, 12, 14))

    @staticmethod
    def _draw_change_boxes(draw, boxes) -> None:
        for item in boxes:
            colour = (255, 80, 95) if item['alarm'] else (255, 190, 70)
            x1, y1, x2, y2 = (item['box'][0] * PROC_W, item['box'][1] * PROC_H,
                              item['box'][2] * PROC_W, item['box'][3] * PROC_H)
            arm = max(4., min(14., (x2 - x1) / 3, (y2 - y1) / 3))
            for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                draw.line([(cx, cy), (cx + dx * arm, cy)], fill=colour, width=3)
                draw.line([(cx, cy), (cx, cy + dy * arm)], fill=colour, width=3)

    def render_preview(self, frame: np.ndarray, *, raw: np.ndarray | None = None,
                       received_at: float | None = None) -> None:
        image = Image.fromarray(frame)
        scene_buffer = BytesIO()
        image.save(scene_buffer, "JPEG", quality=85)
        raw_buffer = BytesIO()
        Image.fromarray(raw if raw is not None else frame).save(raw_buffer, "JPEG", quality=85)
        draw = ImageDraw.Draw(image)
        colours = [(66, 205, 190), (246, 190, 84), (123, 166, 255)]
        label_boxes = []
        def label(text, left, top, colour):
            # Close entrance outlines must not cover one another's labels.
            size = draw.textbbox((0, 0), text)
            left = min(max(3, left), PROC_W-(size[2]-size[0])-4)
            for offset in (0, -15, 15, -30, 30, -45, 45, 60):
                y = min(PROC_H-16, max(18, top+offset))
                box = draw.textbbox((left, y), text)
                padded = [box[0]-3, box[1]-2, box[2]+3, box[3]+2]
                if not any(padded[0] < old[2] and padded[2] > old[0]
                           and padded[1] < old[3] and padded[3] > old[1]
                           for old in label_boxes):
                    break
            label_boxes.append(padded)
            draw.rectangle(padded, fill=(12, 21, 29))
            draw.text((left, y), text, fill=colour)
        for index, zone in enumerate(self.zones):
            x = zone.rect[0] * PROC_W
            y = zone.rect[1] * PROC_H
            w = max(1, zone.rect[2] * PROC_W)
            h = max(1, zone.rect[3] * PROC_H)
            # Unverified alignment: grey outlines show where the zones would be.
            route_colour = {"exit": (66, 205, 190), "entrance": (246, 190, 84),
                            "pedestrian": (188, 157, 255)}.get(zone.route_role, colours[index % len(colours)])
            colour = ((150, 150, 150) if self._alignment_blocked else
                      (255, 80, 95) if zone.active else route_colour)
            points = [(px*PROC_W, py*PROC_H) for px, py in (zone.polygon or rect_polygon(zone.rect))]
            draw.line(points + [points[0]], fill=colour, width=2)
            if zone.confirmation_polygon:
                gate = [(px*PROC_W, py*PROC_H) for px, py in zone.confirmation_polygon]
                draw.line(gate + [gate[0]], fill=(255, 225, 135), width=3)
                label("TOP ALERT" if zone.alert_stage == "advance" else "WAIT ALERT",
                      min(p[0] for p in gate)+3, max(p[1] for p in gate)+2,
                      (255, 225, 135))
            for hole in zone.exclude_polygons:
                excluded = [(px*PROC_W, py*PROC_H) for px, py in hole]
                draw.line(excluded + [excluded[0]], fill=(230, 115, 170), width=2)
            label(f"{index+1}  {zone.name}", x+3, y-15, colour)
        # Highlights go on top of the zone outlines: what changed, then what the AI is following.
        if not self._alignment_blocked:
            self._draw_change_boxes(draw, self._change_boxes())
            if self.semantic:
                info = self.semantic.status()
                if info.get('age_s') is not None and info['age_s'] < 3:
                    self._draw_objects(draw, self._overlay_objects(info.get('detections', [])))
        # Leave the original camera timestamp visible. A newly painted PC
        # clock must never make delayed source video appear current.
        framing = (getattr(self, "alignment", None) or {}).get("framing") or {}
        if framing.get("state") in ("adjust", "unknown"):
            # Zoomable view: show where the saved framing is and how to get back to it.
            outline = framing.get("outline")
            if self._alignment_blocked and outline:
                points = [(x * PROC_W, y * PROC_H) for x, y in outline]
                draw.line(points + points[:1], fill=(255, 214, 10), width=2)
            # Explanatory advice belongs outside the road picture in the UI.
            # The short fault strip below still exposes an actual detection pause.
        if self._alignment_blocked:
            draw.rectangle([0, PROC_H-23, PROC_W, PROC_H], fill=(118, 66, 14))
            draw.text((7,PROC_H-18), "ALIGNMENT NEEDS ATTENTION - scene alerts paused", fill="white")
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=85)
        annotated = buffer.getvalue()
        with self._preview_lock:
            self.preview_jpeg = annotated
            self.scene_jpeg = scene_buffer.getvalue()
            self.raw_jpeg = raw_buffer.getvalue()
        recorder = getattr(self, 'recorder', None)
        if recorder is not None:
            recorder.add_frame(annotated, clean_jpeg=scene_buffer.getvalue(), source_at=received_at)

    def snapshot_preview(self, kind="preview") -> bytes | None:
        with self._preview_lock:
            return {"preview": self.preview_jpeg, "scene": self.scene_jpeg, "raw": self.raw_jpeg}.get(kind)

    def analysis_status(self) -> dict:
        """Second-view evidence is advisory and cannot acknowledge an arrival."""
        return self.secondary.status()

    def _submit_analysis(self, frame, received_at: float, *, aligned: bool) -> None:
        # A latest-only worker owns enhancement/inference. This call only queues
        # a bounded frame and metadata after the immediate detector has run.
        with self._state_lock:
            alarm = {"kind": self.alarm, "alarm_id": self.alarm_id,
                     "reason": self.alarm_reason, **self.alarm_details} if self.alarm else None
            primary = self.semantic.status() if self.semantic else {}
            alignment = {**self.alignment,
                         "scene_polygon": self.alignment_config.get("scene_polygon"),
                         "frame_aligned": aligned,
                         "armed": self.armed}
            self.secondary.submit(frame, self.zones, primary, alignment, alarm, stamp=received_at)

    def state_dict(self) -> dict:
        with self._state_lock:
            return self._state_dict_locked()

    def _state_dict_locked(self) -> dict:
        ml = self.semantic.status() if self.semantic else {
            "available": False, "state": "unavailable", "error": "Local model not installed"}
        observed = ml.get("detections", []) if (ml.get("age_s") is not None and ml["age_s"] < 3
                                                  and not self._alignment_blocked) else []
        return {
            "armed": self.armed,
            "barriers": self.barrier_status(),
            "control_warning": self.control_warning,
            "alarm": self.alarm,
            "alarm_id": self.alarm_id,
            "reason": self.alarm_reason,
            "alarm_source": self.alarm_details.get("source") if self.alarm else None,
            "alarm_evidence": dict(self.alarm_details) if self.alarm else {},
            "acked": self.acked,
            "last_ack": dict(self.last_ack) if self.last_ack else None,
            "ack_quiet_s": self.ack_quiet_s,
            "input_ack_after_s": self.input_ack_min_s,
            "ack_remaining_s": round(max(0., self._ack_until-time.monotonic()), 2),
            "source": safe_source(self.args.source),
            "camera_view": self._view_title(),
            "ml": ml,
            "objects": self._overlay_objects(observed),
            "change_boxes": self._change_boxes(),
            "ui_version": ui_version(getattr(self, "display_name", None)),
            "sightings": self.sightings.status() if getattr(self, "sightings", None) else None,
            "presence": self.presence.status(),
            "loudness": self.loudness.status(),
            "visit_grouping": self._visit_status(),
            "lighting": self.lighting_monitor.status() if hasattr(self, "lighting_monitor") else {},
            "alert_policy": f"One alert per observed visit; acknowledgement adds {self.ack_quiet_s:g} seconds of quiet",
            "fault_reason": self.fault_reason,
            "blind_for_s": (round(time.monotonic() - self._fault_since, 1)
                            if getattr(self, "_fault_since", None) is not None and self.armed else None),
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
            "camera_gone": self.camera_gone_status(time.monotonic()),
            # A coarse picture of the learned flicker, so the dashboard can
            # show which parts of the view the watcher deliberately ignores.
            "noise_grid": self._noise_grid(),
            "audio": dict(self.audio),
            "input_helper": self._input_helper_status(),
            "alerts_version": self.alerts_version,
            "server_time": time.strftime("%H:%M:%S"),
            "server_date": time.strftime("%Y-%m-%d"),
            "last_frame_clock": self.last_frame_clock,
            "uptime_s": int(time.monotonic() - self.started),
            "zones": [{"name": z.name,
                       "trigger": z.trigger, "classes": z.classes,
                       "priority": z.priority, "route_group": z.route_group,
                       "route_role": z.route_role, "operator_hint": z.operator_hint,
                       "alert_stage": z.alert_stage,
                       "confirmation_polygon": z.confirmation_polygon,
                       "light_polygon": z.light_polygon,
                       "ai_confirm_observations": z.ai_confirm_observations,
                       "ai_confirmation_window_s": z.ai_confirmation_window_s,
                       "ai_fast_confirm_confidence": z.ai_fast_confirm_confidence,
                       "approach_motion": z.approach_motion,
                       "ai_context": z.ai_context,
                       "approach_metrics": z.approach_metrics,
                       "ai_confidence": z.ai_confidence,
                       "arrival_direction": z.arrival_direction,
                       "light_metrics": z.light_metrics,
                       "polygon": z.polygon, "rect": z.rect,
                       "ai_active": any(z.name in item.get("confirmed_regions", []) for item in observed),
                       "ai_objects": sum(z.name in item.get("regions", []) for item in observed),
                       "motion_active": z.motion_active,
                       "light_active": z.light_active,
                       "strict_change": z.strict_change,
                       "change_metrics": z.change_metrics,
                       "change_threshold": z.change_threshold if z.strict_change else None,
                       "changed_pct": round(z.last_pct, 2),
                       "light_delta": round(z.last_delta, 1),
                       # The levels each reading is measured against, so the
                       # dashboard can show how close a zone is to alarming.
                       "light_threshold": z.light_delta if z.light and z.trigger != "ai" else None,
                       "motion_threshold": z.motion_pct if z.motion and z.trigger not in ("ai", "ai_light") else None,
                       "motion_raises": z.severity if z.motion and z.trigger not in ("ai", "ai_light") else None,
                       "light_raises": z.light_severity if z.light and z.trigger != "ai" else None}
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

    def _pipeline_stage(self, name):
        pipeline = getattr(self, "pipeline", None)
        return pipeline.stage(name) if pipeline is not None else nullcontext()

    def _pipeline_measure(self, name, *, source_at=None):
        pipeline = getattr(self, "pipeline", None)
        return pipeline.measure(name, source_at=source_at) if pipeline is not None else nullcontext()

    def _pipeline_frame(self, source_at):
        pipeline = getattr(self, "pipeline", None)
        return pipeline.frame_span(source_at) if pipeline is not None else nullcontext()

    def diagnostics(self) -> dict:
        """Readable even while alignment holds the main state/control lock.

        Decoder progress is measured separately from the last frame dequeued
        by the main loop. No device call, filesystem access or inference is
        performed here. The optional stack lists code locations, never locals.
        """
        now = time.monotonic()
        pipeline = getattr(self, "pipeline", None)
        report = pipeline.status() if pipeline is not None else {}
        reader = getattr(self, "reader", None)
        progress = getattr(reader, "_progress_at", None)
        got_frame = getattr(reader, "_got_frame", False)
        report["capture"] = {
            "decoder_running": getattr(reader, "_proc", None) is not None,
            "last_frame_age_s": (round(max(0., now-progress), 3)
                                 if got_frame and isinstance(progress, (float, int)) else None),
            "restarts": getattr(reader, "restarts", 0),
            "dropped_frames": getattr(reader, "dropped_frames", 0),
        }
        report["loop_heartbeat_age_s"] = round(max(0., now-getattr(self, "_loop_heartbeat", now)), 3)
        report["main_stack"] = []
        if report.get("stage_age_s", 0) >= 2.0 or report.get("measurement_slow"):
            frame = sys._current_frames().get(threading.main_thread().ident)
            try:
                for _ in range(12):
                    if frame is None:
                        break
                    report["main_stack"].append({"file": Path(frame.f_code.co_filename).name,
                                                  "function": frame.f_code.co_name,
                                                  "line": frame.f_lineno})
                    frame = frame.f_back
            finally:
                del frame
        return report

    def _audio_health_loop(self) -> None:
        # Native speaker queries can block in a device driver. Keep them off
        # the watchdog thread, so detector-stall recovery remains independent.
        next_audio = 0.0
        while self._running:
            now = time.monotonic()
            if now >= next_audio:
                self.audio = self._read_audio()
                next_audio = now + 15.0
            time.sleep(1.0)

    def _health_loop(self) -> None:
        while self._running:
            now = time.monotonic()
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
        if (STATE_DIR / 'power-off').exists():
            return
        self._loop_heartbeat = time.monotonic()
        self.reader.start()
        self.presence.start()
        self.secondary.start()
        threading.Thread(target=self._health_loop, daemon=True, name="health").start()
        threading.Thread(target=self._audio_health_loop, daemon=True, name="audio-health").start()
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
            if (STATE_DIR / 'power-off').exists():
                break
            self._loop_heartbeat = time.monotonic()
            self._update_awake()
            now = time.monotonic()
            # A ringing alarm climbs while nobody answers it.
            if self.alarm or self.loudness.kind == 'barrier_open':
                self._update_alarm_audio()
            try:
                with self._pipeline_stage('frame_wait'):
                    received_at, buf = self.reader.frames.get(timeout=0.25)
            except queue.Empty:
                self.check_health(time.monotonic())
                self.paint_status()
                continue

            # A local queue item can age while another operation stalls. Never
            # process it as a new observation or let it displace newer evidence.
            if getattr(self, "pipeline", None) is not None:
                self.pipeline.frame_received(received_at)
            if not self._accept_local_frame(received_at):
                self.check_health(time.monotonic())
                continue
            with self._pipeline_frame(received_at):
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
                    with self._pipeline_stage('alignment'):
                        aligned = self.align_frame(frame)
                    if aligned is not None and self._accept_local_frame(received_at):
                        with self._pipeline_stage('detection'):
                            completed = self.process(aligned.astype(np.float32), received_at=received_at)
                        if completed is not False and self._local_frame_fresh(received_at):
                            if getattr(self, "pipeline", None) is not None:
                                self.pipeline.frame_processed(received_at)
                    elif aligned is None and self.semantic:
                        self.semantic.reset()
                        if getattr(self, '_approach_warnings', None) is not None:
                            self._approach_warnings.reset()
                    self._frame_error_since = 0.0
                except Exception as exc:
                    self._note_error("Detection", exc)
                self.processing_ms = (time.monotonic() - processing_started) * 1000
                self.check_health(time.monotonic())
                try:
                    self._submit_analysis(aligned if aligned is not None else frame,
                                          received_at, aligned=aligned is not None)
                except Exception as exc:
                    self._note_error("Second-view analysis", exc, detection=False)

                if now - last_draw >= 0.85 / min(10.0, self.args.fps):   # allow camera frame-timing jitter
                    try:
                        with self._pipeline_stage('preview'):
                            self.render_preview(aligned if aligned is not None else frame, raw=frame,
                                                received_at=received_at if aligned is not None else None)
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
            sys.stdout.write(line[:240] if interactive
                             else time.strftime('%Y-%m-%d %H:%M:%S ') + line.strip()[:240] + '\n')
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
        print(f"\n{GREEN} acknowledged {RESET} alarms quiet for {self.ack_quiet_s:g} seconds; detection continues.")

    def shutdown(self) -> None:
        self._running = False
        if getattr(self, 'learning', None) is not None:
            try:
                self.learning.save(force=True)
            except Exception:
                pass
        # Restore sound before other cleanup, and attempt every component even
        # when one cleanup fails. The power controller still has a bounded
        # forced-stop fallback for a blocked native call.
        cleanup = [("silence", lambda: self.buzzer.set(None)),
                   ("volume", self.loudness.end), ("buzzer", self.buzzer.shutdown),
                   ("desk camera", self.presence.stop), ("reader", self.reader.stop),
                   ("sleep prevention", self.sleep_inhibitor.__exit__)]
        if getattr(self, "_keys", None):
            cleanup.append(("terminal keys", self._keys.restore))
        if self.semantic:
            cleanup.append(("detector", self.semantic.stop))
        if getattr(self, "secondary", None):
            cleanup.append(("second-view analysis", self.secondary.stop))
        if getattr(self, "recorder", None):
            cleanup.append(("alarm videos", self.recorder.stop))
        if getattr(self, "sightings", None):
            cleanup.append(("vehicle sightings", self.sightings.flush))
        for name, stop in cleanup:
            try:
                stop()
            except Exception as exc:
                try:
                    self._note_error(f"Shutdown {name}", exc)
                except Exception:
                    pass


# --------------------------------------------------------------------------
# Tiny local web view (127.0.0.1 only)
# --------------------------------------------------------------------------

from dashboard_ui import PAGE, dashboard_page, rebrand


_UI_VERSIONS: dict = {}


def ui_version(name) -> str:
    """A short fingerprint of the dashboard page this program serves; changes with every page update."""
    if name not in _UI_VERSIONS:
        _UI_VERSIONS[name] = hashlib.sha1(dashboard_page(name).encode('utf-8')).hexdigest()[:12]
    return _UI_VERSIONS[name]


def car_memory_status(watcher, *, now=None) -> dict:
    """Small, read-only health metadata; reports and vehicle records are never parsed here."""
    now = time.time() if now is None else now
    folder = STATE_DIR / 'car-memory'
    result = dict(available=False, report_version=None, generated_at=None, age_s=None,
                  updater_alive=False, updater_state='unknown', updater_age_s=None,
                  last_success_at=None, model_state='unknown', sightings_enabled=None,
                  sightings_error=False, stale=True)

    def timestamp(value):
        return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 < value <= now else None)

    try:
        report = folder / 'report.html'
        stat = report.stat()
        generated = timestamp(stat.st_mtime)
        if report.is_file() and stat.st_size > 0 and generated is not None:
            result.update(available=True, report_version=f'{stat.st_mtime_ns:x}:{stat.st_size:x}',
                          generated_at=generated, age_s=round(now-generated, 3))
    except OSError:
        pass
    try:
        with (folder / 'worker-status.json').open('rb') as handle:
            raw = handle.read(CAR_MEMORY_STATUS_BYTES + 1)
        if len(raw) > CAR_MEMORY_STATUS_BYTES:
            raise ValueError('Oversized worker status')
        worker = json.loads(raw)
        if not isinstance(worker, dict):
            raise ValueError('Invalid worker status')
        observed = timestamp(worker.get('updated_at'))
        state = worker.get('state')
        if state in ('ready', 'updating', 'error', 'stopped'):
            result['updater_state'] = state
        model = worker.get('model_state')
        if model in ('ready', 'classical', 'not_loaded'):
            result['model_state'] = model
        result['last_success_at'] = timestamp(worker.get('last_success_at'))
        if observed is not None:
            age = now-observed
            result['updater_age_s'] = round(age, 3)
            result['updater_alive'] = age < CAR_MEMORY_HEARTBEAT_S
    except (OSError, ValueError, TypeError):
        pass
    sightings = getattr(watcher, 'sightings', None)
    if sightings is not None:
        enabled = getattr(sightings, 'enabled', None)
        result['sightings_enabled'] = enabled if isinstance(enabled, bool) else None
        result['sightings_error'] = bool(getattr(sightings, 'last_error', ''))
    result['stale'] = (not result['available'] or not result['updater_alive']
                       or result['updater_state'] not in ('ready', 'updating')
                       or result['sightings_enabled'] is not True or result['sightings_error'])
    return result


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
            if self.path.split("?")[0] == "/":
                name = getattr(watcher, 'display_name', None)
                page = dashboard_page(name)
                page = page.replace('</head>', f'<meta name="bw-version" content="{ui_version(name)}"></head>', 1)
                if "app=1" in urlsplit(self.path).query:      # stay in app mode when moving between pages
                    page = page.replace('href="/settings"', 'href="/settings?app=1"')
                self._send(200, "text/html; charset=utf-8", page.encode())
            elif self.path == "/zones":
                from zone_editor import read_config
                from dashboard_ui import zone_editor_page
                self._send(200, "text/html; charset=utf-8",
                           rebrand(zone_editor_page(read_config(Path(watcher.args.zones))),
                                   getattr(watcher, 'display_name', None)).encode())
            elif self.path.split('?')[0] in ("/frame.jpg", "/scene.jpg", "/raw.jpg"):
                kind = {"/scene.jpg": "scene", "/raw.jpg": "raw"}.get(self.path.split('?')[0], "preview")
                jpeg = watcher.snapshot_preview(kind)
                if jpeg:
                    self._send(200, "image/jpeg", jpeg)
                else:
                    self._send(503, "text/plain", b"no frame yet")
            elif self.path.split('?')[0] == "/guard-cam.jpg":
                # The desk camera, annotated. Loopback only, never stored.
                jpeg = watcher.presence.preview_jpeg()
                if jpeg:
                    self._send(200, "image/jpeg", jpeg)
                else:
                    self._send(503, "text/plain", b"no webcam picture")
            elif self.path.split("?")[0] == "/car-memory/status":
                self._send(200, "application/json", json.dumps(car_memory_status(watcher)).encode())
            elif self.path.split("?")[0] == "/car-memory":
                try:
                    body = (STATE_DIR / "car-memory" / "report.html").read_bytes()
                except OSError:
                    body = CAR_MEMORY_WAITING.encode()
                self._send(200, "text/html; charset=utf-8", body)
            elif self.path.split("?")[0] == "/settings":
                from settings_ui import PAGE as SETTINGS_PAGE
                import html as _html
                name = _html.escape(getattr(watcher, 'display_name', None) or 'BarrierWatcher')
                page = SETTINGS_PAGE.replace('<title>Settings</title>', f'<title>{name} / Settings</title>', 1)
                if "app=1" in urlsplit(self.path).query:
                    page = page.replace('<a class="back" href="/">', '<a class="back" href="/?app=1">', 1)
                if "embedded=1" in urlsplit(self.path).query:      # shown inside the app: its rail is the way back
                    page = page.replace('<a class="back" href="/">← Dashboard</a>', '', 1)
                self._send(200, "text/html; charset=utf-8", page.encode())
            elif self.path == "/settings.json":
                self._send(200, "application/json", json.dumps(watcher.settings_snapshot()).encode())
            elif self.path == "/state":
                self._send(200, "application/json",
                           json.dumps(watcher.state_dict()).encode())
            elif self.path == "/diagnostics":
                self._send(200, "application/json",
                           json.dumps(watcher.diagnostics()).encode())
            elif self.path == "/analysis":
                self._send(200, "application/json", json.dumps(watcher.analysis_status()).encode())
            elif self.path.split('?')[0] == "/analysis/frame.jpg":
                query = parse_qs(urlsplit(self.path).query)
                view = query.get("view", ["enhanced"])[0]
                if view not in ("enhanced", "crop", "edges", "change", "landmarks", "rectified"):
                    return self._send(400, "text/plain", b"Unknown analysis view")
                jpeg = watcher.secondary.image(view)
                self._send(200 if jpeg else 503, "image/jpeg" if jpeg else "text/plain",
                           jpeg or b"Analysis image unavailable")
            elif self.path == "/input-status":
                watcher.note_input_helper()
                self._send(200, "application/json", json.dumps(watcher.input_status()).encode())
            elif self.path == "/alerts":
                self._send(200, "application/json", json.dumps(watcher.alerts()).encode())
            elif self.path.startswith("/recordings/"):
                from alarm_recorder import CLIP_NAME
                name = self.path[len("/recordings/"):].split("?")[0]
                recorder = getattr(watcher, "recorder", None)
                path = recorder.folder / name if recorder is not None and CLIP_NAME.match(name) else None
                if path is None or not path.is_file():
                    return self._send(404, "text/plain", b"not found")
                size = path.stat().st_size
                start, end, partial = 0, size - 1, False
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
                if match and (match.group(1) or match.group(2)):
                    partial = True
                    if match.group(1):
                        start = int(match.group(1))
                        end = min(size - 1, int(match.group(2))) if match.group(2) else size - 1
                    else:
                        start = max(0, size - int(match.group(2)))
                    if start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                with path.open("rb") as handle:
                    handle.seek(start)
                    body = handle.read(end - start + 1)
                try:
                    self.send_response(206 if partial else 200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(len(body)))
                    if partial:
                        self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{size}")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                except (ConnectionError, OSError):
                    pass              # the player stopped reading; nothing to clean up
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
            # Read the whole body before any answer. Replying first (403, 400 "too large") left
            # the client still sending, and Windows then reset the connection mid-request.
            try:
                length = max(0, int(self.headers.get('Content-Length', '0') or 0))
            except ValueError:
                length = 0
            body = b''
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                if len(body) < 1_048_576:
                    body += chunk[:1_048_576 - len(body)]
                remaining -= len(chunk)
            self.rfile = BytesIO(body)
            origin = self.headers.get('Origin')
            host = self.headers.get('Host', '')
            if host.split(':')[0] not in ('127.0.0.1', 'localhost') or (origin and origin != 'http://' + host):
                return self._send(403, 'text/plain', b'Local access only')
            if self.headers.get('X-Via') not in ('web', 'ctl', 'hotkey', 'zones', 'zone-editor', 'input-any-key', 'input-mouse'):
                return self._send(403, 'text/plain', b'Local control header required')
            if self.path == '/power/off':
                if self.headers.get('X-Via') not in ('web', 'ctl'):
                    return self._send(403, 'text/plain', b'Explicit power control required')
                try:
                    request_full_off()
                except RuntimeError as exc:
                    return self._send(503, 'application/json', json.dumps({'ok': False, 'error': str(exc)}).encode())
                watcher.set_armed(False, via='full-off')
                return self._send(202, 'application/json', json.dumps({
                    'ok': True, 'status': 'stopping',
                    'message': 'Full shutdown requested. Open the BarrierWatcher app to start it again.'}).encode())
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
            if self.path == "/settings":
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if size < 2 or size > 16384:
                        raise ValueError('Invalid request size')
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected a settings object')
                    result = watcher.update_settings(payload.get('changes'))
                    return self._send(200, 'application/json', json.dumps(result).encode())
                except (ValueError, TypeError, OSError) as exc:
                    return self._send(400, 'application/json', json.dumps({'ok': False, 'error': str(exc)}).encode())
            if self.path == "/barrier-learning/undo":
                if self.headers.get('X-Via') not in ('web', 'ctl'):
                    return self._send(403, 'text/plain', b'Explicit request required')
                result = watcher.undo_barrier_learning(self.headers.get('X-Via', 'web'))
                return self._send(200 if result.get('ok') else 409, 'application/json', json.dumps(result).encode())
            if self.path == "/refresh":
                watcher.request_refresh(self.headers.get('X-Via', 'web'))
                return self._send(200, 'application/json', b'{"ok":true,"restarting":true}')
            if self.path == "/test-tone":
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 <= size <= 128:
                        raise ValueError('Invalid sound test request size')
                    payload = json.loads(self.rfile.read(size)) if size else {}
                    if not isinstance(payload, dict) or set(payload) - {'lane'}:
                        raise ValueError('Invalid sound test request')
                    watcher.play_test_tone(lane=payload.get('lane', ''))
                    return self._send(200, 'application/json', b'{"ok":true,"played":true}')
                except PermissionError as exc:
                    return self._send(409, 'application/json', json.dumps({'error': str(exc)}).encode())
                except (ValueError, TypeError) as exc:
                    return self._send(400, 'application/json', json.dumps({'error': str(exc)}).encode())
            if self.path == "/desk/calibrate":
                # Ten seconds of the guard looking at the screen teaches the
                # watcher what "looking at the screen" means for this person.
                started = watcher.presence.start_calibration(10.0)
                return self._send(200 if started else 409, 'application/json',
                                  json.dumps({"ok": started, "seconds": 10,
                                              "error": None if started else "Desk camera is switched off"}).encode())
            if self.path == '/barrier-calibration':
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 1 <= size <= 1024:
                        raise ValueError('Invalid calibration request size')
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected a calibration object')
                    result = watcher.calibrate_barrier(payload)
                    return self._send(200 if result.get('ok') else 409, 'application/json',
                                      json.dumps(result).encode())
                except (ValueError, TypeError, OSError) as exc:
                    return self._send(400, 'application/json', json.dumps({'ok': False, 'error': str(exc)}).encode())
            if self.path == '/feedback':
                if self.headers.get('X-Via') not in ('web', 'ctl'):
                    return self._send(403, 'text/plain', b'Explicit report required')
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 2 <= size <= 512:
                        raise ValueError('Invalid report size')
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected a report object')
                    result = watcher.label_alert(payload.get('alarm_id'), payload.get('verdict'),
                                                 via=self.headers.get('X-Via', 'web'))
                    return self._send(200, 'application/json', json.dumps(result).encode())
                except LookupError as exc:
                    return self._send(404, 'application/json', json.dumps({'ok': False, 'error': str(exc)}).encode())
                except (ValueError, TypeError) as exc:
                    return self._send(400, 'application/json', json.dumps({'ok': False, 'error': str(exc)}).encode())
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
    if (STATE_DIR / 'power-off').exists():
        print('BarrierWatcher is fully OFF. Open the BarrierWatcher app (desktop or Start menu) to start it.')
        return
    # Task Scheduler starts us at below-normal CPU and low memory priority; on a
    # PC short of RAM that let Windows page the watcher out and freeze it.
    priority = ps.normal_priority()
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
    print(f"Process priority normal={priority['priority']} memory={priority['memory']} "
          f"no-throttling={priority['throttling']}", flush=True)
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
