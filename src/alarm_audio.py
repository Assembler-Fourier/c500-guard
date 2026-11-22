"""Alternating attention pulses and an evidence-gated escalation policy.

This module never opens an audio device or changes the system volume. The
watcher applies a plan while retaining ownership of speaker restoration and
alarm acknowledgement. A volume fraction is a software setting, not a dB
measurement or a promise about the loudness of a particular speaker.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import wave


SOUND_VERSION = "attention-pulses-4-lanes"
SAMPLE_RATE = 24000
START_LEVEL = 0.30
MAX_LEVEL = 0.95
WEAK_LEVEL = 0.75
CONFIRMED_RISE_S = 8.0
IMAGE_RISE_S = 6.0
# A verified immediate change on the bottom corner is not a flicker: cars
# arrive there at speed and stop at a barrier somebody lifts by hand. It
# gets its own tier - stronger than an unidentified change, still short of
# a recognised vehicle, and it reaches its cap sooner than either.
URGENT_LEVEL = 0.88
URGENT_RISE_S = 7.0

_TIERS = {
    "confirmed": (CONFIRMED_RISE_S, MAX_LEVEL, 3.2),
    "urgent": (URGENT_RISE_S, URGENT_LEVEL, 3.0),
    "image": (IMAGE_RISE_S, WEAK_LEVEL, 2.6),
}
_LIMITS = {"rise_s": (1.0, 600.0), "level": (0.20, 1.00), "shape": (0.5, 8.0)}
_settings = {name: values for name, values in _TIERS.items()}
# Where every alarm starts before it climbs. zones.json alarms.start_level
# overrides it (the operator asked for a 10% start on 14 Nov 2026).
START_LIMITS = (0.05, 0.60)
_start = [START_LEVEL]


def set_start_level(value) -> float:
    """Starting volume for every alarm, clamped and never above any ceiling."""
    number = _number(value)
    if number is None or isinstance(value, bool):
        number = START_LEVEL
    low, high = START_LIMITS
    ceiling = min(level for _rise, level, _shape in _settings.values())
    _start[0] = min(max(low, min(high, number)), ceiling)
    return _start[0]


def start_level() -> float:
    return _start[0]


def level_ceiling(*, kind="main", confirmed=False, urgent=False) -> float:
    """Effective endpoint of the selected policy, including the hard limit.

    The mixer controller uses this same endpoint, so the urgent tier and any
    locally configured ceiling are respected (fixed caps ignored them).
    """
    if kind in ("fault", "barrier_open"):
        return .18
    tier = "confirmed" if confirmed else ("urgent" if urgent else "image")
    return min(MAX_LEVEL, max(0., _settings[tier][1]))


def configure(values) -> dict:
    """Apply escalation tuning from zones.json; bad values are ignored.

    Each tier may set rise_s, level and shape. Everything is clamped, and
    weaker evidence can never end up louder or faster than stronger
    evidence, so a typo cannot silence the alarm or invert the grading.
    """
    applied = {}
    if isinstance(values, dict):
        for tier in _TIERS:
            block = values.get(tier)
            if not isinstance(block, dict):
                continue
            rise, level, shape = _settings[tier]
            for key, current in (("rise_s", rise), ("level", level), ("shape", shape)):
                number = _number(block.get(key))
                if number is None or isinstance(block.get(key), bool):
                    continue
                low, high = _LIMITS[key]
                value = min(high, max(low, number))
                if key == "rise_s":
                    rise = value
                elif key == "level":
                    level = value
                else:
                    shape = value
            _settings[tier] = (rise, level, shape)
            applied[tier] = {"rise_s": rise, "level": level, "shape": shape}
    confirmed_level = _settings["confirmed"][1]
    urgent = _settings["urgent"]
    _settings["urgent"] = (urgent[0], min(urgent[1], confirmed_level), urgent[2])
    image = _settings["image"]
    _settings["image"] = (image[0], min(image[1], _settings["urgent"][1]), image[2])
    return applied


def reset_configuration() -> None:
    """Return to the built-in tiers (used by tests and by a config reload)."""
    _settings.clear()
    _settings.update(_TIERS)
    _start[0] = START_LEVEL
    set_theme(DEFAULT_THEME)
    set_reminder_period(_REMINDER_DEFAULT)


def settings() -> dict:
    """The tuning currently in force, for the dashboard and diagnostics."""
    result = {tier: {"rise_s": values[0], "level": values[1], "shape": values[2]}
              for tier, values in _settings.items()}
    result["start_level"] = _start[0]
    result["sound_theme"] = _theme[0]
    return result
LIMITED_SOUND_LIMIT_S = 12.0


@dataclass(frozen=True)
class AudioPlan:
    level: float
    profile: str
    stage: str
    reason: str
    audible: bool
    repeat_s: float = 0.0
    burst_s: float = 0.0

    @property
    def sound(self) -> str | None:
        """Compatibility name for players that use None to mean stop."""
        return self.profile if self.audible else None

    @property
    def volume(self) -> float:
        return self.level


# Each tuple is (loop seconds, (start seconds, duration seconds, Hz) ...).
# The pulse begins at the start of its file: no buffer-sized waiting period.
# Silence belongs to the asset so even a slow watcher never makes a continuous
# tone. Alternating pitches with a few smooth harmonics make the reminder
# distinct from a notification chime without square waves or sharp attacks.
PROFILES = {
    "limited": (6.0, ((0.0, 0.16, 520.0),)),
    "soft": (3.0, ((0.0, 0.30, 880.0), (0.52, 0.30, 620.0))),
    "notice": (2.8, ((0.0, 0.60, 880.0), (0.95, 0.60, 620.0))),
    "wake": (3.0, ((0.0, 0.85, 880.0), (1.15, 0.85, 620.0))),
    "fault": (30.0, ((0.0, 0.16, 330.0),)),
    "barrier_open": (15.0, ((0.0, 0.16, 180.0), (0.38, 0.16, 150.0))),
}
PROFILE_PEAKS = {"limited": 0.22, "soft": 0.50, "notice": 0.70, "wake": 0.85,
                 "fault": 0.20, "barrier_open": 0.20}
PROFILE_HARMONICS = {
    "limited": ((1, 1.0),), "fault": ((1, 1.0),), "barrier_open": ((1, 1.0),),
    "soft": ((1, 1.0), (3, .12)),
    "notice": ((1, 1.0), (3, .18), (5, .04)),
    "wake": ((1, 1.0), (3, .22), (5, .08)),
}

# Sound styles for the three attention stages. 'pulses' is the original sound.
# Only soft/notice/wake change; the fault, limited and barrier reminders keep
# their own distinct sounds whatever style is chosen.
THEMES = {
    "pulses": {"label": "Pulses (original)", "envelope": "flat",
               "soft": (3.0, ((0.0, 0.30, 880.0), (0.52, 0.30, 620.0))),
               "notice": (2.8, ((0.0, 0.60, 880.0), (0.95, 0.60, 620.0))),
               "wake": (3.0, ((0.0, 0.85, 880.0), (1.15, 0.85, 620.0))),
               "harmonics": {"soft": ((1, 1.0), (3, .12)), "notice": ((1, 1.0), (3, .18), (5, .04)),
                             "wake": ((1, 1.0), (3, .22), (5, .08))}},
    "chime": {"label": "Chime (ding-dong)", "envelope": "bell",
              "soft": (3.2, ((0.0, 0.90, 784.0), (0.55, 1.10, 659.0))),
              "notice": (2.8, ((0.0, 0.90, 784.0), (0.55, 1.10, 659.0))),
              "wake": (2.6, ((0.0, 0.70, 880.0), (0.45, 0.70, 698.0), (0.90, 1.00, 587.0))),
              "harmonics": {name: ((1, 1.0), (2, .30), (3, .12), (4, .05))
                            for name in ("soft", "notice", "wake")}},
    "beeps": {"label": "Beeps (short)", "envelope": "flat",
              "soft": (2.6, ((0.0, 0.12, 1000.0), (0.22, 0.12, 1000.0))),
              "notice": (2.2, ((0.0, 0.12, 1000.0), (0.22, 0.12, 1000.0), (0.44, 0.12, 1000.0))),
              "wake": (1.8, ((0.0, 0.12, 1100.0), (0.20, 0.12, 1100.0), (0.40, 0.12, 1100.0),
                             (0.60, 0.12, 1100.0))),
              "harmonics": {name: ((1, 1.0), (3, .20)) for name in ("soft", "notice", "wake")}},
    "low": {"label": "Low tone (softer)", "envelope": "flat",
            "soft": (3.2, ((0.0, 0.45, 440.0), (0.65, 0.45, 330.0))),
            "notice": (2.8, ((0.0, 0.60, 440.0), (0.85, 0.60, 330.0))),
            "wake": (2.6, ((0.0, 0.80, 466.0), (1.00, 0.80, 349.0))),
            "harmonics": {name: ((1, 1.0), (2, .15), (3, .08)) for name in ("soft", "notice", "wake")}},
}
DEFAULT_THEME = "pulses"
_theme = [DEFAULT_THEME]
LANES = {"entrance": "Rising tones", "exit": "Falling tones"}
_ATTENTION_PROFILES = ("soft", "notice", "wake")


def lane_profile(profile: str, lane: str | None) -> str:
    """Select a prebuilt lane sound without changing the policy or theme.

    Only arrival attention sounds have a direction. Faults, barrier-open
    reminders and the mixer-failure fallback retain their neutral profiles.
    Reapplying this function is safe, including when the active lane changes.
    """
    base = {"main": "soft", "early": "soft"}.get(profile, profile)
    for prefix in LANES:
        if base in {f"{prefix}_{name}" for name in _ATTENTION_PROFILES}:
            base = base[len(prefix) + 1:]
            break
    if base in _ATTENTION_PROFILES and isinstance(lane, str) and lane in LANES:
        return f"{lane}_{base}"
    return base


def _set_lane_profiles(profile: str) -> None:
    period, pulses = PROFILES[profile]
    pitches = sorted(pulse[2] for pulse in pulses)
    if pitches[0] == pitches[-1]:
        # The short-beep theme originally repeats one pitch. A narrow range
        # gives it a direction while retaining its pulse count and timing.
        carrier = pitches[0]
        pitches = [carrier * (.9 + .2 * index / (len(pulses) - 1))
                   for index in range(len(pulses))]
    for lane in LANES:
        name = lane_profile(profile, lane)
        frequencies = pitches if lane == "entrance" else list(reversed(pitches))
        PROFILES[name] = (period, tuple((start, duration, frequency)
            for (start, duration, _old), frequency in zip(pulses, frequencies)))
        PROFILE_PEAKS[name] = PROFILE_PEAKS[profile]
        PROFILE_HARMONICS[name] = PROFILE_HARMONICS[profile]


_REMINDER_DEFAULT = 15.0
_reminder_period = [_REMINDER_DEFAULT]


def reminder_period() -> float:
    return _reminder_period[0]


def set_reminder_period(seconds) -> float:
    """The arm-left-up beep repeats with its sound file, so the file follows the configured interval (10..600 s)."""
    value = _number(seconds)
    period = _REMINDER_DEFAULT if value is None or isinstance(seconds, bool) else min(600.0, max(10.0, float(value)))
    PROFILES["barrier_open"] = (period, PROFILES["barrier_open"][1])
    _reminder_period[0] = period
    return period


def set_theme(name) -> str:
    """Select a sound style. Unknown names keep the original pulses."""
    chosen = name if isinstance(name, str) and name in THEMES else DEFAULT_THEME
    spec = THEMES[chosen]
    for profile in _ATTENTION_PROFILES:
        PROFILES[profile] = spec[profile]
        PROFILE_HARMONICS[profile] = spec["harmonics"][profile]
        _set_lane_profiles(profile)
    _theme[0] = chosen
    return chosen


# Prebuild both lane descriptions even if no configured theme is loaded.
# ensure_sounds renders every profile once; changing lane only selects a file.
set_theme(DEFAULT_THEME)


def theme() -> str:
    return _theme[0]


def _number(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _silent(stage: str, reason: str) -> AudioPlan:
    return AudioPlan(0.0, "silent", stage, reason, False)


def _audible(profile: str, level: float, stage: str, reason: str) -> AudioPlan:
    period, pulses = PROFILES[profile]
    return AudioPlan(min(MAX_LEVEL, max(0.0, level)), profile, stage,
                     reason, True, period, sum(pulse[1] for pulse in pulses))


def _ramp(seconds: float, *, confirmed: bool = True, urgent: bool = False) -> float:
    """A normalized exponential rise: prompt growth, exact bounded endpoint.

    Confirmed: 30% initially, about 67% at 2s, 84% at 4s, 95% at 8s.
    Image-only: 30% initially, about 58% at 2s, 70% at 4s, 75% at 6s.
    Normalization avoids waiting forever for an asymptotic ceiling. ``expm1``
    also preserves a smooth first increment for very small elapsed times.
    """
    duration, ceiling, shape = _settings["confirmed"] if confirmed else (
        _settings["urgent"] if urgent else _settings["image"])
    start = min(_start[0], ceiling)
    if seconds <= 0.0:
        return start
    if seconds >= duration:
        return ceiling
    progress = math.expm1(-shape * seconds / duration) / math.expm1(-shape)
    return start + (ceiling - start) * progress


def plan(elapsed_s: float, *, kind: str = "main", source: str = "unknown",
         presence_state: str = "unknown", evidence_fresh: bool = True,
         event_present: bool = True, acknowledged: bool = False,
         quiet_remaining_s: float = 0.0, enabled: bool = True,
         confirmed: bool = False, urgent: bool = False,
         pedestrian_latched: bool = False) -> AudioPlan:
    """Return an alarm plan without side effects.

    ``confirmed`` must come from current vehicle/person evidence, never from
    the face camera, a strict-change flag, or a bright patch by itself. Fresh
    evidence and continued event presence are required for every repeat and
    every escalation. Missing evidence stops sound while the watcher may keep
    the visual incident open. The initial short pulse needs no AI confirmation.

    ``pedestrian_latched`` is an explicit manual-acknowledgement incident;
    it may repeat after evidence clears without claiming current occupancy.

    Presence does not change timing or strength. It cannot acknowledge a
    visit, increase an evidence grade, or prevent an unanswered reminder.
    ``barrier_open`` is a quiet separate reminder; the caller must only mark
    it present while the current barrier reading is known and open, and must
    give arrival alarms priority over that reminder.
    """
    if not enabled:
        return _silent("disabled", "Alerts are paused")
    if acknowledged:
        return _silent("acknowledged", "This visit has been acknowledged")
    quiet = _number(quiet_remaining_s)
    if quiet is None:
        return _silent("quiet", "Waiting for a valid acknowledgement timer")
    if quiet > 0:
        return _silent("quiet", "Acknowledgement quiet period")
    elapsed = _number(elapsed_s)
    if elapsed is None:
        return _silent("unavailable", "Alarm timer is unavailable")
    elapsed = max(0.0, elapsed)
    # A latched, validated pedestrian incident needs operator attention; it
    # does not claim that the person is still in the picture. ACK/quiet/OFF
    # and the separate speaker-failure fallback remain authoritative.
    if pedestrian_latched and kind in ("main", "early"):
        ceiling = level_ceiling(confirmed=True)
        start = min(.60, ceiling)
        fraction = math.expm1(-1.2 * min(elapsed, 4.) / 4.) / math.expm1(-1.2)
        level = start + (ceiling-start) * fraction
        return _audible("notice" if elapsed < 2. else "wake", level,
                        "pedestrian_ack", "Pedestrian alert; acknowledge to stop")
    if not event_present:
        return _silent("cleared", "The current event is no longer present")
    if not evidence_fresh:
        return _silent("stale", "Waiting for fresh evidence; incident remains visible")
    if kind not in ("main", "early", "fault", "barrier_open"):
        return _silent("unavailable", "No recognised alarm is active")
    if kind == "fault":
        return _audible("fault", 0.18, "fault", "Short system reminder every 30 seconds")
    if kind == "barrier_open":
        return _audible("barrier_open", 0.18, "barrier_open",
                        "Barrier remains open; quiet double reminder every 15 seconds")

    age = elapsed
    if not confirmed and not urgent:
        level = _ramp(age, confirmed=False)
        profile = "soft" if age < 1.5 else "notice"
        cap = round(_settings["image"][1] * 100)
        reason = f"Current image change remains; periodic reminders capped at {cap}%"
        return _audible(profile, level, "unconfirmed", reason)

    if not confirmed:
        # Verified immediate change at the barrier corner: a driver is
        # waiting, so this climbs on the confirmed pattern schedule.
        level = _ramp(age, confirmed=False, urgent=True)
        cap = round(_settings["urgent"][1] * 100)
        if age < 1.5:
            return _audible("soft", level, "urgent", "Immediate change at the barrier corner")
        if age < 3.5:
            return _audible("notice", level, "urgent",
                            "Immediate change still present at the corner")
        return _audible("wake", level, "urgent",
                        f"Vehicle waiting at the corner; climbing to {cap}%")

    level = _ramp(age)
    if age < 1.5:
        return _audible("soft", level, "soft", "Arrival attention tone")
    if age < 3.5:
        return _audible("notice", level, "reminder", "Visit still present and not acknowledged")
    return _audible("wake", level, "persistent",
                    "Continued confirmed visit; stronger alternating reminders, capped at 95%")


def limited_plan(elapsed_s: float, *, kind: str = "main", source: str = "unknown",
                 presence_state: str = "unknown", evidence_fresh: bool = True,
                 event_present: bool = True, acknowledged: bool = False,
                 quiet_remaining_s: float = 0.0, enabled: bool = True) -> AudioPlan:
    """Keep the old brief fallback when speaker volume cannot be controlled.

    This is intentionally separate from unconfirmed image evidence. A current
    light warning may keep reminding while mixer control works. Mixer failure
    must never inherit that persistent or increasingly forceful sound policy.
    """
    result = plan(elapsed_s, kind=kind, source=source, presence_state=presence_state,
                  evidence_fresh=evidence_fresh, event_present=event_present,
                  acknowledged=acknowledged, quiet_remaining_s=quiet_remaining_s,
                  enabled=enabled, confirmed=False)
    if not result.audible or kind == "fault":
        return result
    if kind == "barrier_open":
        return _silent("limited", "Speaker control unavailable; barrier reminder remains visual")
    if float(elapsed_s) >= LIMITED_SOUND_LIMIT_S:
        return _silent("limited", "Speaker control unavailable; short fallback finished")
    return _audible("limited", .12, "limited", "Speaker control unavailable; short fallback only")


def pcm_samples(profile: str, *, sample_rate: int = SAMPLE_RATE) -> array:
    """Render signed 16-bit mono PCM for validation or WAV creation only."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown alarm profile: {profile}")
    if not isinstance(sample_rate, int) or sample_rate < 8000 or sample_rate > 96000:
        raise ValueError("sample_rate must be an integer between 8000 and 96000")
    period, pulses = PROFILES[profile]
    samples = array("h", [0]) * round(period * sample_rate)
    # Mixer failures select only 'limited', never these stronger attention
    # assets. Digital peaks remain below full scale independently of OS volume.
    amplitude = PROFILE_PEAKS[profile] * 32767
    harmonics = PROFILE_HARMONICS[profile]
    harmonic_weight = sum(weight for _multiple, weight in harmonics)
    for start, duration, frequency in pulses:
        first = round(start * sample_rate)
        count = round(duration * sample_rate)
        # Raised-cosine edges have zero value and envelope slope. The attention
        # profiles keep a steady middle instead of fading like a sweet chime.
        attack = max(1, round(.025 * sample_rate))
        release = max(1, round(.035 * sample_rate))
        for offset in range(count):
            if profile in ("limited", "fault", "barrier_open"):
                share = offset / max(1, count - 1)
                envelope = math.sin(math.pi * share) ** 2
            elif THEMES[_theme[0]]["envelope"] == "bell":
                # Struck-bell shape: a 6 ms attack, then an exponential decay
                # that has faded to silence by the end of the note.
                rise = min(1.0, offset / max(1, round(.006 * sample_rate)))
                decay = math.exp(-4.5 * offset / max(1, count))
                tail = min(1.0, (count - 1 - offset) / max(1, round(.02 * sample_rate)))
                envelope = (0.5 - 0.5 * math.cos(math.pi * rise)) * decay * tail
            else:
                rise = min(1.0, offset / attack)
                fall = min(1.0, (count - 1 - offset) / release)
                envelope = (0.5 - 0.5 * math.cos(math.pi * rise)) * (
                    0.5 - 0.5 * math.cos(math.pi * fall))
            phase = 2.0 * math.pi * frequency * offset / sample_rate
            carrier = sum(weight * math.sin(multiple * phase)
                          for multiple, weight in harmonics) / harmonic_weight
            samples[first + offset] = round(amplitude * envelope * carrier)
    return samples


def ensure_sounds(directory: Path | str, force: bool = False) -> dict[str, Path]:
    """Create versioned pulse assets. Does not play them or touch volume."""
    directory = Path(directory)
    paths = {name: directory / f"attention-{name}.wav" for name in PROFILES}
    stamp = directory / ".attention-version"
    version = SOUND_VERSION if _theme[0] == DEFAULT_THEME else f"{SOUND_VERSION}:{_theme[0]}"
    if _reminder_period[0] != _REMINDER_DEFAULT:
        version += f":reminder-{_reminder_period[0]:g}s"
    try:
        current = stamp.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        current = ""
    if force or current != version or not all(path.is_file() for path in paths.values()):
        directory.mkdir(parents=True, exist_ok=True)
        for profile, path in paths.items():
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            try:
                pcm = pcm_samples(profile)
                if sys.byteorder != "little":
                    pcm.byteswap()
                with wave.open(str(temporary), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(SAMPLE_RATE)
                    output.writeframes(pcm.tobytes())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        stamp.write_text(version + "\n", encoding="ascii")
    # Main/early use the short first attention pattern. The watcher must select
    # 'limited' explicitly if it cannot verify control of the speaker level.
    return {**paths, "main": paths["soft"], "early": paths["soft"]}
