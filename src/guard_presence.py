"""Estimate desk presence and head orientation for alarm pacing.

The barrier camera decides *whether* to alarm. This decides *how hard* the alarm
has to push to reach the person sitting at the desk:

  watching   a light buzz is enough - they can already see the screen
  present    they are there but not looking: start moderate, climb
  away       nobody at the desk: start low, climb to full
YuNet provides face landmarks, not eyelid measurements. Its head orientation
must never be treated as closed eyes or proof of sleep. The optional closed-eye
tracker is reserved for explicit measurements from a future supported source;
neither installed detector supplies those measurements.

Everything here fails towards being *louder*. A camera that is covered, dark,
broken or absent reports ``unknown``, which is treated exactly like ``away``:
the alarm still rings and still climbs to full. Nothing in this file can make
the system quieter than its floor, and nothing here can stop an alarm.

No webcam image is ever written to disk or sent anywhere. Frames live in memory
only, for the moment it takes to look for a face; the dashboard preview is
served on the loopback interface to the person sitting at the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time

from desk_activity import DeskActivity


def _state_dir() -> Path:
    """Where private state lives; matches platform_support without importing it."""
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "c500-guard"
    return Path.home() / ".local" / "state" / "c500-guard"


# A face smaller than this fraction of the frame width is somebody walking past
# in the background, not the guard at the desk.
MIN_FACE_FRACTION = 0.10
# Detector sizes. Small frames keep a face check well under 15 ms per look.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECT_WIDTH = 320


@dataclass
class PresenceConfig:
    """Timings and volumes, all overridable from zones.json."""

    enabled: bool = True
    device_index: int = 0
    # When set (e.g. "322e:2333"), the desk webcam is found by its USB id each
    # time it is opened and device_index is ignored. A second USB camera (the
    # barrier capture stick) can then never be opened by mistake.
    device_usb_id: str = ""
    checks_per_second: float = 4.0
    # How fresh each piece of evidence has to be for the state it supports.
    watching_grace_s: float = 4.0
    present_grace_s: float = 15.0
    away_after_s: float = 30.0
    # Eyes closed this long with the face still visible means drowsy; drowsy
    # this long means asleep.
    drowsy_after_s: float = 10.0
    asleep_after_s: float = 60.0
    eyes_window_s: float = 8.0
    eyes_closed_ratio: float = 0.2
    # Movement is a secondary filter for explicit closed-eye measurements.
    # Stillness alone never supplies that measurement. The installed face
    # detectors report eye state unknown. Movement uses face-widths per second.
    still_below: float = 0.06
    # Volume floor: the alarm never goes below this, whatever the camera thinks.
    floor: float = 0.25
    # An unacknowledged alarm climbs even when the guard appears to be watching.
    escalate_unacked_s: float = 30.0
    # When nobody is attending, weak evidence is worth the full siren.
    escalate_weak_when_unattended: bool = True
    levels: dict = field(default_factory=lambda: {
        # state: (start level, peak level, seconds to climb)
        "watching": (0.30, 0.30, 0.0),
        "present": (0.45, 1.00, 12.0),
        "away": (0.30, 1.00, 18.0),
        "unknown": (0.30, 1.00, 18.0),
        "drowsy": (0.75, 1.00, 4.0),
        "asleep": (1.00, 1.00, 0.0),
    })

    @classmethod
    def from_dict(cls, data: dict | None) -> "PresenceConfig":
        config = cls()
        if not isinstance(data, dict):
            return config
        numbers = {
            "checks_per_second": (0.5, 15.0), "watching_grace_s": (1.0, 60.0),
            "present_grace_s": (2.0, 300.0), "away_after_s": (5.0, 900.0),
            "drowsy_after_s": (2.0, 600.0), "asleep_after_s": (5.0, 3600.0),
            "eyes_window_s": (2.0, 120.0), "eyes_closed_ratio": (0.0, 1.0),
            "still_below": (0.0, 5.0),
            "floor": (0.0, 1.0), "escalate_unacked_s": (0.0, 600.0),
        }
        for key, (low, high) in numbers.items():
            if key in data:
                try:
                    setattr(config, key, min(high, max(low, float(data[key]))))
                except (TypeError, ValueError):
                    pass
        for flag in ("enabled", "escalate_weak_when_unattended"):
            if flag in data:
                setattr(config, flag, bool(data[flag]))
        if "device_index" in data:
            try:
                config.device_index = max(0, int(data["device_index"]))
            except (TypeError, ValueError):
                pass
        if data.get("device_usb_id"):
            try:
                import platform_support as ps
                config.device_usb_id = ":".join(ps.parse_usb_id(data["device_usb_id"]))
            except (ImportError, ValueError):
                # An unreadable id must not fall back to "camera 0": that could
                # be the capture stick. Presence stays unavailable instead.
                config.device_usb_id = "invalid"
        levels = data.get("levels")
        if isinstance(levels, dict):
            merged = dict(config.levels)
            for state, value in levels.items():
                if state in merged and isinstance(value, (list, tuple)) and len(value) == 3:
                    try:
                        start, peak, ramp = (float(v) for v in value)
                    except (TypeError, ValueError):
                        continue
                    start = min(1.0, max(0.0, start))
                    peak = min(1.0, max(start, peak))
                    merged[state] = (start, peak, max(0.0, ramp))
            config.levels = merged
        # An asleep guard must never be given a quieter plan than an awake one.
        config.levels["asleep"] = (max(config.levels["asleep"][0], config.levels["watching"][0]),
                                   config.levels["asleep"][1], config.levels["asleep"][2])
        return config


@dataclass(frozen=True)
class Reading:
    """One webcam reading with separate head orientation and eyelid evidence.

    ``eyes`` is an explicit open/closed measurement, never detector failure or
    head pose. ``attention`` is only an estimate of facing the screen.
    """

    at: float
    face: bool
    eyes: bool | None = None
    frontal: bool = True
    detail: str = ""
    # Face box (x, y, w, h) in the captured frame, used to measure stillness.
    box: tuple | None = None
    attention: bool | None = None


@dataclass(frozen=True)
class LoudnessPlan:
    """How loud, how fast. ``level_at`` is what the siren loop asks for."""

    state: str
    start: float
    peak: float
    ramp_s: float
    reason: str

    def level_at(self, seconds_ringing: float) -> float:
        if self.ramp_s <= 0 or seconds_ringing <= 0:
            return self.start
        share = min(1.0, seconds_ringing / self.ramp_s)
        return self.start + (self.peak - self.start) * share


class PresenceTracker:
    """Turns a stream of webcam readings into one of six honest states.

    Pure logic: every timestamp is passed in, so a whole night of behaviour can
    be replayed in a test in milliseconds.
    """

    STATES = ("watching", "present", "drowsy", "asleep", "away", "unknown")

    def __init__(self, config: PresenceConfig | None = None):
        self.config = config or PresenceConfig()
        self.last_face_at: float | None = None
        self.last_eyes_at: float | None = None
        self.last_attention_at: float | None = None
        self.eye_history: list[tuple[float, bool]] = []
        self.box_history: list[tuple[float, float, float, float]] = []   # at, cx, cy, width
        self.state = "unknown"
        self.state_since: float | None = None
        self.closed_since: float | None = None
        self.last_reading_at: float | None = None
        self.camera_ok = False
        self.detail = "not started"

    def observe(self, reading: Reading) -> str:
        self.last_reading_at = reading.at
        self.camera_ok = True
        if reading.detail:
            self.detail = reading.detail
        if reading.face:
            self.last_face_at = reading.at
            attentive = reading.attention
            if attentive is None and reading.eyes is True and reading.frontal:
                attentive = True
            if attentive:
                self.last_attention_at = reading.at
            if reading.box:
                x, y, w, h = (float(v) for v in reading.box)
                self.box_history.append((reading.at, x + w / 2, y + h / 2, max(1.0, w)))
        if reading.face and reading.eyes is not None:
            self.eye_history.append((reading.at, bool(reading.eyes)))
            if reading.eyes:
                self.last_eyes_at = reading.at
                self.closed_since = None
            elif self.closed_since is None:
                self.closed_since = reading.at
        else:
            # Unknown eyelids break a measured closure. A previous observation
            # cannot turn later pose-only readings into evidence of sleep.
            self.closed_since = None
            self.eye_history.clear()
        window = self.config.eyes_window_s
        self.eye_history = [row for row in self.eye_history if reading.at - row[0] <= window]
        self.box_history = [row for row in self.box_history if reading.at - row[0] <= window]
        return self._settle(reading.at)

    def head_movement(self) -> float | None:
        """Face-widths per second the head has moved. None when not measurable."""
        if len(self.box_history) < 3:
            return None
        span = self.box_history[-1][0] - self.box_history[0][0]
        if span <= 0:
            return None
        travel = 0.0
        for (_, x1, y1, w1), (_, x2, y2, _) in zip(self.box_history, self.box_history[1:]):
            travel += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / w1
        return travel / span

    def camera_failed(self, at: float, detail: str) -> str:
        """A covered lens, a dark room or an unplugged camera: never a quiet alarm."""
        self.camera_ok = False
        self.detail = detail
        self.last_reading_at = at
        # Recovery must establish attention/sleep from fresh camera evidence.
        self.last_face_at = self.last_eyes_at = self.closed_since = None
        self.last_attention_at = None
        self.eye_history.clear()
        self.box_history.clear()
        return self._set_state("unknown", at)

    def open_eye_ratio(self) -> float | None:
        if not self.eye_history:
            return None
        open_count = sum(1 for _, seen in self.eye_history if seen)
        return open_count / len(self.eye_history)

    def _settle(self, now: float) -> str:
        config = self.config
        face_age = None if self.last_face_at is None else now - self.last_face_at
        attention_age = None if self.last_attention_at is None else now - self.last_attention_at
        ratio = self.open_eye_ratio()

        if face_age is None or face_age > config.away_after_s:
            return self._set_state("away", now)
        if attention_age is not None and attention_age <= config.watching_grace_s:
            return self._set_state("watching", now)
        # Only a measured run of closed-eye readings enters the legacy closure
        # states. Head pose and missing eye detections supply no such evidence.
        closed_for = 0.0 if self.closed_since is None else now - self.closed_since
        movement = self.head_movement()
        # Stillness can support measured closure; it cannot replace it.
        still = movement is not None and movement <= config.still_below
        eyes_look_shut = (ratio is not None and ratio <= config.eyes_closed_ratio
                          and closed_for >= config.drowsy_after_s and still)
        if eyes_look_shut and face_age <= config.present_grace_s:
            if closed_for >= config.drowsy_after_s + config.asleep_after_s:
                return self._set_state("asleep", now)
            return self._set_state("drowsy", now)
        if face_age <= config.present_grace_s:
            return self._set_state("present", now)
        return self._set_state("away", now)

    def _set_state(self, state: str, now: float) -> str:
        if state != self.state:
            self.state = state
            self.state_since = now
        elif self.state_since is None:
            self.state_since = now
        return state

    # -- what the rest of the program asks for ----------------------------

    @property
    def unattended(self) -> bool:
        """True when nobody is reliably watching the screen right now."""
        return self.state in ("away", "asleep", "drowsy", "unknown")

    def plan(self, *, unacknowledged_s: float = 0.0) -> LoudnessPlan:
        config = self.config
        state = self.state if self.state in config.levels else "unknown"
        start, peak, ramp = config.levels[state]
        reason = {
            "watching": "you are looking at the screen",
            "present": "you are at the desk but not looking",
            "away": "nobody at the desk",
            "unknown": "cannot see the desk",
            "drowsy": "eyes closed",
            "asleep": "asleep",
        }[state]
        # A light buzz that is being ignored stops being light.
        if (state == "watching" and config.escalate_unacked_s
                and unacknowledged_s >= config.escalate_unacked_s):
            fallback = config.levels["away"]
            start, peak, ramp = max(start, fallback[0]), max(peak, fallback[1]), fallback[2]
            reason = "not acknowledged - climbing anyway"
        floor = config.floor
        return LoudnessPlan(state=state, start=min(1.0, max(floor, start)),
                            peak=min(1.0, max(floor, peak)), ramp_s=ramp, reason=reason)

    def status(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        plan = self.plan()
        age = None if self.last_reading_at is None else round(now - self.last_reading_at, 1)
        return {
            "state": self.state,
            "for_s": None if self.state_since is None else round(now - self.state_since, 1),
            "camera_ok": self.camera_ok,
            "detail": self.detail,
            "last_look_age_s": age,
            "face_age_s": None if self.last_face_at is None else round(now - self.last_face_at, 1),
            "eyes_age_s": None if self.last_eyes_at is None else round(now - self.last_eyes_at, 1),
            "attention_age_s": None if self.last_attention_at is None else round(now - self.last_attention_at, 1),
            "open_eye_ratio": None if self.open_eye_ratio() is None else round(self.open_eye_ratio(), 2),
            "head_movement": None if self.head_movement() is None else round(self.head_movement(), 3),
            "unattended": self.unattended,
            "alarm_plan": {"start_pct": round(plan.start * 100), "peak_pct": round(plan.peak * 100),
                           "ramp_s": plan.ramp_s, "reason": plan.reason},
        }


@dataclass
class DeskBaseline:
    """What this guard's face measures like when they are looking at the screen.

    Generic thresholds cannot know how somebody sits: how far back, how square
    to the screen, how the webcam is angled. So the numbers are learned from ten
    seconds of the real person at their real desk, and everything after that is
    judged relative to them.
    """

    eye_gap: float = 0.0
    sideways: float = 0.0
    pitch: float = 0.0
    samples: int = 0
    at: str = ""

    @property
    def learned(self) -> bool:
        return self.samples >= 12 and self.eye_gap > 0 and self.pitch > 0

    def as_dict(self) -> dict:
        return {"eye_gap": round(self.eye_gap, 2), "sideways": round(self.sideways, 3),
                "pitch": round(self.pitch, 3), "samples": self.samples, "at": self.at,
                "learned": self.learned}

    @classmethod
    def from_dict(cls, data: dict | None) -> "DeskBaseline":
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(eye_gap=float(data.get("eye_gap", 0)), sideways=float(data.get("sideways", 0)),
                       pitch=float(data.get("pitch", 0)), samples=int(data.get("samples", 0)),
                       at=str(data.get("at", "")))
        except (TypeError, ValueError):
            return cls()


# Tolerances around the learned baseline. Wide on purpose: being wrong towards
# "not watching" only makes the alarm louder, which is the safe direction.
SIDEWAYS_TOLERANCE = 0.22        # extra head turn allowed, in eye-widths
PITCH_TOLERANCE = 0.80           # head may drop to this share of its usual pitch
MIN_FACE_SCORE = 0.70
# Fallback thresholds, used only until a baseline is learned.
DEFAULT_SIDEWAYS_LIMIT = 0.35
DEFAULT_PITCH_FLOOR = 0.70


def face_geometry(points) -> dict:
    """Yaw and pitch of a face, from five landmarks, in eye-gap units.

    points: right eye, left eye, nose, right mouth corner, left mouth corner.
    Scaling everything by the distance between the eyes makes the numbers
    independent of how close the guard is sitting.
    """
    right_eye, left_eye, nose, right_mouth, left_mouth = (tuple(float(v) for v in p) for p in points)
    gap = ((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2) ** 0.5
    if gap <= 0:
        return {"eye_gap": 0.0, "sideways": 1.0, "pitch": 0.0}
    eye_middle = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    mouth_middle = ((right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2)
    return {
        "eye_gap": gap,
        # Nose off the centre line between the eyes: the head turned left/right.
        "sideways": abs(nose[0] - eye_middle[0]) / gap,
        # Eyes to mouth, which foreshortens as the head tips down.
        "pitch": (mouth_middle[1] - eye_middle[1]) / gap,
    }


def looking_at_screen(geometry: dict, score: float, baseline: DeskBaseline) -> bool:
    """Is this face turned towards the screen? Judged against the learned baseline."""
    if score < MIN_FACE_SCORE or geometry.get("eye_gap", 0) <= 0:
        return False
    sideways, pitch = geometry["sideways"], geometry["pitch"]
    if baseline.learned:
        return (sideways <= baseline.sideways + SIDEWAYS_TOLERANCE
                and pitch >= baseline.pitch * PITCH_TOLERANCE)
    return sideways <= DEFAULT_SIDEWAYS_LIMIT and pitch >= DEFAULT_PITCH_FLOOR


class YuNetLook:
    """Face detection with five landmarks for approximate head orientation.

    Chosen over the Haar eye cascades after measuring both on the real desk
    camera: the cascades found the face but almost never the eyes, so a guard
    looking straight at the screen was being reported as 'not looking'.
    """

    def __init__(self, model_path: Path, baseline: DeskBaseline | None = None):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (DETECT_WIDTH, 240), 0.6, 0.3, 5000)
        self.baseline = baseline or DeskBaseline()
        self.last_box: tuple[int, int, int, int] | None = None
        self.last_geometry: dict = {}
        self.last_score = 0.0
        self.name = "YuNet"

    def look(self, frame, at: float) -> Reading:
        cv2 = self.cv2
        height, width = frame.shape[:2]
        scale = DETECT_WIDTH / float(width)
        small = cv2.resize(frame, (DETECT_WIDTH, int(round(height * scale))))
        if float(small.max()) < 8:
            self.last_box, self.last_geometry, self.last_score = None, {}, 0.0
            return Reading(at=at, face=False, eyes=None, detail="camera covered or dark")
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.detector.detect(small)
        if found is None or not len(found):
            self.last_box, self.last_geometry, self.last_score = None, {}, 0.0
            return Reading(at=at, face=False, eyes=None, detail="no face in view")
        row = max(found, key=lambda item: item[2] * item[3])
        x, y, w, h = (float(v) for v in row[:4])
        if w < small.shape[1] * MIN_FACE_FRACTION:
            self.last_box, self.last_geometry, self.last_score = None, {}, 0.0
            return Reading(at=at, face=False, eyes=None, detail="face too far away")
        points = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
        geometry = face_geometry(points)
        score = float(row[14])
        attentive = looking_at_screen(geometry, score, self.baseline)
        self.last_box = tuple(int(v / scale) for v in (x, y, w, h))
        self.last_geometry, self.last_score = geometry, score
        return Reading(at=at, face=True, eyes=None, attention=attentive,
                       frontal=attentive, box=self.last_box,
                       detail="facing screen (head pose)" if attentive else "face turned away or down; eye state unknown")


class FaceLook:
    """Haar face + eye detection. Small, CPU-cheap, and honest about failure."""

    def __init__(self):
        import cv2                                    # imported late: tests never need it

        self.cv2 = cv2
        path = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(path + "haarcascade_frontalface_alt2.xml")
        self.profile = cv2.CascadeClassifier(path + "haarcascade_profileface.xml")
        # The eyeglasses cascade also finds bare eyes, and the guard may wear glasses.
        self.eyes = cv2.CascadeClassifier(path + "haarcascade_eye_tree_eyeglasses.xml")
        self.last_box: tuple[int, int, int, int] | None = None

    def look(self, frame, at: float) -> Reading:
        cv2 = self.cv2
        height, width = frame.shape[:2]
        scale = DETECT_WIDTH / float(width) if width > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if float(gray.max()) < 8:
            self.last_box = None
            return Reading(at=at, face=False, eyes=None, detail="camera covered or dark")
        gray = cv2.equalizeHist(gray)
        minimum = int(small.shape[1] * MIN_FACE_FRACTION)
        faces = self.face.detectMultiScale(gray, 1.15, 5, minSize=(minimum, minimum))
        frontal = len(faces) > 0
        if not frontal:
            faces = self.profile.detectMultiScale(gray, 1.15, 4, minSize=(minimum, minimum))
        if not len(faces):
            self.last_box = None
            return Reading(at=at, face=False, eyes=None, detail="no face in view")
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        self.last_box = tuple(int(v / scale) for v in (x, y, w, h))
        if not frontal:
            # A profile face cannot show two eyes; claiming 'eyes closed' there
            # would invent sleep out of a head turned sideways.
            return Reading(at=at, face=True, eyes=None, frontal=False, detail="looking away",
                           box=self.last_box)
        upper = gray[y:y + int(h * 0.62), x:x + w]
        seen = self.eyes.detectMultiScale(upper, 1.1, 6,
                                          minSize=(max(8, int(w * 0.12)), max(8, int(w * 0.12))))
        # An eye cascade locates an eye-like area; a miss is not a closed eye.
        return Reading(at=at, face=True, eyes=None,
                       attention=True if len(seen) else None, frontal=True, box=self.last_box,
                       detail="frontal face and eye regions" if len(seen) else "face in view; eye state unknown")


class GuardPresence:
    """Runs the webcam in its own thread and keeps the tracker up to date."""

    MAX_READING_AGE = 3.0

    def __init__(self, config: PresenceConfig | None = None, *, clock=time.monotonic,
                 model_path: Path | None = None, baseline_path: Path | None = None):
        self.config = config or PresenceConfig()
        self.tracker = PresenceTracker(self.config)
        self.clock = clock
        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet.onnx")
        self.baseline_path = Path(baseline_path) if baseline_path else (
            _state_dir() / "desk-baseline.json")
        self.baseline = self._read_baseline()
        self.detector_name = "not started"
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._preview: bytes | None = None
        self._preview_at = 0.0
        self._calibrate_until = 0.0
        self._calibration: list[dict] = []
        self._looker = None
        self.frames = 0
        self._desk_activity = DeskActivity()

    # -- learning this guard's own "looking at the screen" -----------------

    def _read_baseline(self) -> DeskBaseline:
        try:
            return DeskBaseline.from_dict(json.loads(self.baseline_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return DeskBaseline()

    def start_calibration(self, seconds: float = 10.0) -> bool:
        """Learn from the guard looking at the screen for a few seconds."""
        if not self.config.enabled:
            return False
        with self._lock:
            self._calibrate_until = self.clock() + max(3.0, min(60.0, seconds))
            self._calibration = []
        return True

    @property
    def calibrating(self) -> bool:
        return self.clock() < self._calibrate_until

    def _collect_calibration(self, geometry: dict, score: float) -> None:
        if not geometry or score < MIN_FACE_SCORE or geometry.get("eye_gap", 0) <= 0:
            return
        with self._lock:
            self._calibration.append(geometry)

    def _finish_calibration(self) -> None:
        with self._lock:
            samples, self._calibration = self._calibration, []
            self._calibrate_until = 0.0
        if len(samples) < 12:
            return                       # too little to trust; keep what we had

        def at_rank(key: str, share: float) -> float:
            ordered = sorted(sample[key] for sample in samples)
            return ordered[min(len(ordered) - 1, int(len(ordered) * share))]

        # Not the middle: a glance away mid-calibration would loosen the test and
        # make a real alarm quieter. Taking a squarer turn and a taller tilt than
        # average keeps "looking at the screen" strict, so mistakes are loud.
        baseline = DeskBaseline(eye_gap=at_rank("eye_gap", .5), sideways=at_rank("sideways", .35),
                                pitch=at_rank("pitch", .65), samples=len(samples),
                                at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.baseline = baseline
        if self._looker is not None:
            self._looker.baseline = baseline
        try:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self.baseline_path.write_text(json.dumps(baseline.as_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass                         # a lost baseline only costs accuracy

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="guard-presence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        # A blocked native read needs the enclosing watcher's forced-stop path.
        # Keep its identity visible, and prevent start() from opening a duplicate.
        if thread is None or not thread.is_alive():
            self._thread = None
        self._camera_failed("webcam stopped")

    def _expire_locked(self, now: float) -> None:
        last = self.tracker.last_reading_at
        if (self.tracker.camera_ok and last is not None
                and now - last >= self.MAX_READING_AGE):
            self.tracker.camera_failed(now, "webcam observations are stale")
            self.tracker.last_reading_at = last
            self._preview = None
            self._desk_activity.invalidate('camera_observations_stale')

    def _camera_failed(self, detail: str) -> None:
        with self._lock:
            self.tracker.camera_failed(self.clock(), detail)
            self._preview = None
            self._desk_activity.invalidate('camera_unavailable')

    # -- what the watcher asks --------------------------------------------

    @property
    def unattended(self) -> bool:
        with self._lock:
            self._expire_locked(self.clock())
            return self.tracker.unattended

    def plan(self, *, unacknowledged_s: float = 0.0) -> LoudnessPlan:
        with self._lock:
            self._expire_locked(self.clock())
            return self.tracker.plan(unacknowledged_s=unacknowledged_s)

    def status(self) -> dict:
        with self._lock:
            now = self.clock()
            self._expire_locked(now)
            data = self.tracker.status(now)
            # Presence reports observations; the alarm controller owns loudness.
            data.pop("alarm_plan", None)
            data["enabled"] = self.config.enabled
            data["looks"] = self.frames
            data["running"] = bool(self._thread and self._thread.is_alive())
            data["detector"] = self.detector_name
            data["attention_method"] = "head pose" if self.detector_name == "YuNet" else "face visibility"
            data["sleep_detection_supported"] = False
            data["eye_openness_supported"] = False
            data["calibrating"] = self.calibrating
            data["baseline"] = self.baseline.as_dict()
            data["desk_activity"] = self._desk_activity.status(now)
            looker = self._looker
            if self.tracker.camera_ok and looker is not None and getattr(looker, "last_geometry", None):
                geometry = looker.last_geometry
                data["face_geometry"] = {"sideways": round(geometry.get("sideways", 0), 3),
                                         "pitch": round(geometry.get("pitch", 0), 3),
                                         "eye_gap_px": round(geometry.get("eye_gap", 0), 1),
                                         "score": round(getattr(looker, "last_score", 0.0), 2)}
            return data

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            now = self.clock()
            self._expire_locked(now)
            if not self.tracker.camera_ok or now - self._preview_at >= self.MAX_READING_AGE:
                return None
            return self._preview

    # -- the thread itself -------------------------------------------------

    def _build_looker(self):
        """Landmarks when the model is installed, Haar cascades when it is not."""
        if self.model_path.is_file():
            try:
                looker = YuNetLook(self.model_path, self.baseline)
                self.detector_name = looker.name
                self._looker = looker
                return looker
            except Exception:                            # noqa: BLE001 - fall back, never fail
                pass
        looker = FaceLook()
        self.detector_name = "Haar cascades"
        self._looker = looker
        return looker

    def _device_index(self):
        """The desk webcam's DirectShow index, or None when it cannot be proven."""
        usb_id = self.config.device_usb_id
        if not usb_id:
            return self.config.device_index
        if usb_id == "invalid":
            return None
        try:
            import platform_support as ps
            found = ps.find_usb_camera(usb_id)
        except Exception:                                # noqa: BLE001 - listing failed
            return None
        return None if found is None else found[0]

    def _loop(self) -> None:
        try:
            import cv2
        except ImportError as exc:                       # pragma: no cover - install issue
            self._camera_failed(f"OpenCV missing ({exc})")
            return
        looker = None
        capture = None
        interval = 1.0 / max(0.5, self.config.checks_per_second)
        opened_at = 0.0
        while self._running:
            try:
                if capture is None or not capture.isOpened():
                    if self.clock() - opened_at < 5.0:
                        time.sleep(0.5)
                        continue
                    opened_at = self.clock()
                    index = self._device_index()
                    if index is None:
                        self._camera_failed("desk webcam not found")
                        continue
                    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    if not capture.isOpened():
                        self._camera_failed("webcam not available")
                        capture = None
                        continue
                    looker = looker or self._build_looker()
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._camera_failed("webcam gave no picture")
                    capture.release()
                    capture = None
                    continue
                self.frames += 1
                reading = looker.look(frame, self.clock())
                if not self._running:
                    break
                self._observe_desk(frame, reading)
                with self._lock:
                    self.tracker.observe(reading)
                if self.calibrating:
                    self._collect_calibration(getattr(looker, "last_geometry", {}),
                                              getattr(looker, "last_score", 0.0))
                elif self._calibrate_until:
                    self._finish_calibration()
                self._store_preview(cv2, frame, looker.last_box, reading)
            except Exception as exc:                     # noqa: BLE001 - never kill the watcher
                self._camera_failed(f"webcam error: {type(exc).__name__}")
                if capture is not None:
                    capture.release()
                capture = None
            time.sleep(interval)
        if capture is not None:
            capture.release()

    def _observe_desk(self, frame, reading: Reading) -> None:
        """Advisory lower-view motion on the existing capture thread only."""
        try:
            self._desk_activity.update(frame, reading.at, now=self.clock(),
                                       face_box=reading.box,
                                       camera_ready='covered' not in reading.detail.lower())
        except Exception:
            # Optional movement diagnostics cannot stop presence or sound.
            self._desk_activity.invalidate('motion_measurement_unavailable')

    def _store_preview(self, cv2, frame, box, reading: Reading) -> None:
        """One small annotated JPEG in memory, for the dashboard. Never saved."""
        now = self.clock()
        if now - self._preview_at < 0.5:
            return
        shot = cv2.resize(frame, (320, 240))
        if box:
            x, y, w, h = (int(v * 320 / frame.shape[1]) for v in box)
            colour = (0, 220, 0) if reading.attention else (0, 170, 255)
            cv2.rectangle(shot, (x, y), (x + w, y + h), colour, 2)
        ok, buffer = cv2.imencode(".jpg", shot, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self._lock:
                self._preview = buffer.tobytes()
                self._preview_at = now
