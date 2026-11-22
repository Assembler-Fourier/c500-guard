"""Live-tunable settings: one registry, validated, saved to zones.json, applied without restart.

Every setting has a plain label, a safe range and one place in zones.json. The
settings page reads describe() and posts {key: value} changes; apply_changes()
returns a new, validated configuration and never touches zone geometry.
"""
from __future__ import annotations

import copy
import math
import re

DEFAULT_NAME = "BarrierWatcher"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .&_-]{0,23}")

# Shapes for the three tiers by climb style (higher = more of the rise early).
CLIMBS = {"gentle": 1.2, "steady": 2.0, "quick": 3.2}
TIERS = (("confirmed", "Recognised vehicle or person"),
         ("urgent", "Car at the exit corner"),
         ("image", "Unrecognised change or headlights"))

STATIC = [
    {"id": "sound", "title": "Sound", "items": [
        {"key": "alarms.sound_theme", "label": "Sound style", "kind": "choice",
         "choices": [["pulses", "Pulses (original)"], ["chime", "Chime (ding-dong)"],
                     ["beeps", "Beeps (short)"], ["low", "Low tone (softer)"]],
         "default": "pulses", "help": "Use Test sound to hear it."},
        {"key": "alarms.start_level", "label": "Starting volume", "kind": "percent",
         "min": 5, "max": 60, "step": 1, "default": 0.10,
         "help": "Every alarm begins this loud, then climbs while nobody answers."},
        {"key": "alarms.climb", "label": "How it climbs", "kind": "choice",
         "choices": [["gentle", "Gentle (slow at first)"], ["steady", "Steady"],
                     ["quick", "Quick (most of the rise early)"]], "default": "gentle"},
    ] + [item for tier, label in TIERS for item in (
        {"key": f"alarms.escalation.{tier}.level", "label": f"{label}: top volume", "kind": "percent",
         "min": 20, "max": 100, "step": 1,
         "default": {"confirmed": .95, "urgent": .88, "image": .75}[tier]},
        {"key": f"alarms.escalation.{tier}.rise_s", "label": f"{label}: seconds to reach top",
         "kind": "seconds", "min": 1, "max": 120, "step": 1,
         "default": {"confirmed": 8., "urgent": 7., "image": 6.}[tier]},
    )] + [
        {"key": "alarms.soften_other_lane", "label": "Start gentler just after a car is let in on the other lane",
         "kind": "bool", "default": True,
         "help": "Its headlights often flood the exit corner. The alarm still sounds, and lowering that arm stops it."},
        {"key": "alarms.ack_quiet_s", "label": "Quiet after you acknowledge", "kind": "seconds",
         "min": 5, "max": 30, "step": 1, "default": 10.,
         "help": "Only that visit stays quiet; other lanes still alert."},
    ]},
    {"id": "barriers", "title": "Barriers", "items": [
        {"key": "barriers.ack_on_raise", "label": "Raising an arm stops its lane's alarm", "kind": "bool",
         "default": True},
        {"key": "barriers.ack_on_lower", "label": "Lowering an arm stops its lane's alarm", "kind": "bool",
         "default": True, "help": "For when the arms are held open and you close one for a car."},
        {"key": "barriers.ack_on_movement", "label": "Stop the alarm as soon as an arm starts moving",
         "kind": "bool", "default": True,
         "help": "Raising an arm means you let the car in or out, so its alarm stops within about a second of the arm starting to lift, instead of when it is fully up. A real movement is required (the arm seen passing through the angles in between), so a vehicle passing a raised arm cannot cause it."},
        {"key": "barriers.ack_any_arm", "label": "This lane's arm reaching up or down acknowledges its alarm",
         "kind": "bool", "default": True,
         "help": "Both lanes: e.g. raising the entrance arm for a lorry whose lights set off an exit alarm. "
                 "Uses the raising/lowering switches above."},
        {"key": "barriers.reminder", "label": "Remind me when a barrier is left up", "kind": "bool",
         "default": True, "help": "A short beep once an arm has been up this long, then again at the interval below, "
                                  "until it goes down. Any key silences it until that arm closes. The Live page has "
                                  "the same switch (bell) on each lane."},
        {"key": "barriers.reminder_after_s", "label": "Beep after the arm has been up for", "kind": "seconds",
         "min": 15, "max": 600, "step": 5, "default": 60.},
        {"key": "barriers.reminder_every_s", "label": "Then beep every", "kind": "seconds",
         "min": 10, "max": 600, "step": 5, "default": 10.},
    ]},
    {"id": "recordings", "title": "Alarm videos", "items": [
        {"key": "recordings.enabled", "label": "Record a video of every alarm", "kind": "bool", "default": True,
         "help": "Shows what happened around each alarm. Open an alert on the Activity page to play it."},
        {"key": "recordings.before_s", "label": "Seconds kept from before the alarm", "kind": "seconds",
         "min": 0, "max": 20, "step": 1, "default": 10.},
        {"key": "recordings.after_s", "label": "Seconds recorded after the alarm", "kind": "seconds",
         "min": 10, "max": 50, "step": 1, "default": 20.},
        {"key": "recordings.keep", "label": "Videos kept (the oldest is removed)", "kind": "number",
         "min": 5, "max": 200, "step": 5, "default": 25.},
    ]},
    {"id": "system", "title": "Detection and system", "items": [
        {"key": "system.name", "label": "System name", "kind": "text", "max_length": 24,
         "default": DEFAULT_NAME,
         "help": "Shown on the dashboard, here and in pop-up notifications. Reload the dashboard to see it."},
        {"key": "ai.engine", "label": "AI runs on", "kind": "choice",
         "choices": [["graphics", "Intel graphics (3x faster, less CPU)"], ["processor", "Processor (original)"]],
         "default": "graphics",
         "help": "Takes effect after Refresh system. If the graphics ever fail it switches to the processor by itself."},
        {"key": "ai.whole_picture", "label": "AI also watches the whole picture", "kind": "bool",
         "default": True,
         "help": "Besides the two lane close-ups, the AI checks the whole view (about 10 ms more on the graphics), so a person crossing the road near the camera is followed and outlined. Takes effect after Refresh system."},
        {"key": "memory.vehicle_sightings", "label": "Keep clean pictures of each vehicle for the car memory",
         "kind": "bool", "default": True,
         "help": "Up to four small pictures of every vehicle that goes in or out, for the separate car memory (same car in, same car out). No number plates are read. Deleted automatically after 12 hours. Takes effect after Refresh system."},
        {"key": "ai.interval_s", "label": "AI looks at the picture every", "kind": "seconds",
         "min": 0.25, "max": 2.0, "step": 0.05, "default": 0.5,
         "help": "Shorter is quicker to confirm cars but uses more of the PC."},
        {"key": "alignment.hold_s", "label": "Keep watching while traffic hides the scene for", "kind": "seconds",
         "min": 0, "max": 30, "step": 1, "default": 30.},
        {"key": "alarms.fault_remind_s", "label": "While detection is paused, remind me every",
         "kind": "seconds", "min": 0, "max": 900, "step": 30, "default": 120.,
         "help": "After you acknowledge a fault (for example alignment lost), it sounds again until detection is back. 0 turns this off; otherwise at least 60 seconds."},
        {"key": "power.stop_when_camera_gone_s", "label": "Switch everything off when the picture is gone for",
         "kind": "seconds", "min": 0, "max": 3600, "step": 10, "default": 120.,
         "help": "0 turns this off. Otherwise at least 30 seconds."},
    ]},
]

ZONE_FIELDS = (
    # field, label, kind, min, max, step, applies-when
    ("connected_pct", "Smallest car-sized change (% of zone)", "number", 1, 60, 0.5,
     lambda z: bool(z.get("strict_change"))),
    ("light_delta", "Headlight threshold (brightness levels)", "number", 1, 80, 0.5,
     lambda z: bool(z.get("light"))),
    ("ai_confidence", "AI must be at least this sure", "percent", 30, 95, 1,
     lambda z: z.get("trigger") in ("ai", "ai_light")),
)


def _get(config, path, default=None):
    node = config
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set(config, path, value):
    node = config
    for part in path[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[path[-1]] = value


def _climb_of(config):
    shapes = [_get(config, ("alarms", "escalation", tier, "shape")) for tier, _ in TIERS]
    if not all(isinstance(s, (int, float)) for s in shapes):
        return None
    average = sum(shapes) / len(shapes)
    return min(CLIMBS, key=lambda name: abs(CLIMBS[name] - average))


def _zone_items(config):
    items = []
    for zone in config.get("zones", []):
        for field, label, kind, low, high, step, applies in ZONE_FIELDS:
            if not applies(zone):
                continue
            items.append({"key": f"zone:{zone['name']}:{field}", "label": label, "kind": kind,
                          "min": low, "max": high, "step": step, "zone": zone["name"]})
    return items


def describe(config: dict) -> list[dict]:
    """Sections with each setting's current value from this configuration."""
    sections = copy.deepcopy(STATIC)
    for section in sections:
        for item in section["items"]:
            item["value"] = value_of(config, item["key"], item.get("default"))
    sections.append({"id": "zones", "title": "Zone sensitivity",
                     "items": [{**item, "value": value_of(config, item["key"])}
                               for item in _zone_items(config)]})
    return sections


def value_of(config, key, default=None):
    if key == "alarms.climb":
        return _climb_of(config) or default
    if key.startswith("zone:"):
        _, name, field = key.split(":", 2)
        for zone in config.get("zones", []):
            if zone.get("name") == name:
                return zone.get(field, default)
        return default
    return _get(config, tuple(key.split(".")), default)


def _definition(config, key):
    for section in STATIC:
        for item in section["items"]:
            if item["key"] == key:
                return item
    for item in _zone_items(config):
        if item["key"] == key:
            return item
    raise ValueError(f"Unknown setting: {key}")


def _coerce(item, raw):
    kind = item["kind"]
    if kind == "bool":
        if not isinstance(raw, bool):
            raise ValueError(f"{item['label']} must be on or off.")
        return raw
    if kind == "text":
        if not isinstance(raw, str):
            raise ValueError(f"{item['label']} must be text.")
        value = " ".join(raw.split())
        if not NAME_PATTERN.fullmatch(value):
            raise ValueError(f"{item['label']}: use 1 to 24 letters, numbers, spaces, dots, dashes or &.")
        return value
    if kind == "choice":
        allowed = [choice[0] for choice in item["choices"]]
        if raw not in allowed:
            raise ValueError(f"{item['label']} must be one of: {', '.join(allowed)}.")
        return raw
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"{item['label']} must be a number.")
    value = float(raw)
    # Percent settings travel as fractions (0.10) but their limits are in %.
    shown = value * 100 if item["kind"] == "percent" else value
    if not item["min"] - 1e-9 <= shown <= item["max"] + 1e-9:
        unit = "%" if item["kind"] == "percent" else (" s" if item["kind"] == "seconds" else "")
        raise ValueError(f"{item['label']} must be between {item['min']}{unit} and {item['max']}{unit}.")
    if item["key"] == "power.stop_when_camera_gone_s" and 0 < value < 30:
        raise ValueError("Switch-off time must be 0 (off) or at least 30 seconds.")
    if item["key"] == "alarms.fault_remind_s" and 0 < value < 60:
        raise ValueError("Fault reminders must be 0 (off) or at least 60 seconds apart.")
    return value


def apply_changes(config: dict, changes: dict) -> tuple[dict, list[str]]:
    """Return a changed copy of the configuration and the keys that changed."""
    if not isinstance(changes, dict) or not changes or len(changes) > 64:
        raise ValueError("Send between 1 and 64 setting changes.")
    updated = copy.deepcopy(config)
    changed = []
    for key, raw in changes.items():
        if not isinstance(key, str):
            raise ValueError("Setting names must be text.")
        item = _definition(updated, key)
        value = _coerce(item, raw)
        if key == "alarms.climb":
            for tier, _label in TIERS:
                _set(updated, ("alarms", "escalation", tier, "shape"), CLIMBS[value])
        elif key.startswith("zone:"):
            _, name, field = key.split(":", 2)
            for zone in updated["zones"]:
                if zone.get("name") == name:
                    zone[field] = value
        else:
            _set(updated, tuple(key.split(".")), value)
        changed.append(key)
    # Weaker evidence may never be set louder than stronger evidence.
    levels = {tier: _get(updated, ("alarms", "escalation", tier, "level")) for tier, _ in TIERS}
    if all(isinstance(v, (int, float)) for v in levels.values()):
        if not levels["image"] <= levels["urgent"] <= levels["confirmed"]:
            raise ValueError("Top volumes must go: unrecognised change ≤ exit corner ≤ recognised vehicle.")
        start = _get(updated, ("alarms", "start_level"))
        if isinstance(start, (int, float)) and start > levels["image"]:
            raise ValueError("The starting volume cannot be above the lowest top volume.")
    return updated, changed
