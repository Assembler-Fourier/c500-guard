"""Local zone editor and strict, atomic configuration updates.

The HTTP handler must hold the watcher's state lock while checking armed state,
saving, and replacing the in-memory zones. No camera reader is created here.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from region_geometry import validate_polygon


FIELDS = {"name", "rect", "motion", "light", "severity", "light_severity",
          "motion_pct", "light_delta", "on_frames", "off_frames", "polygon",
          "exclude_polygons", "classes", "trigger", "priority", "route_group",
          "ai_confirm_observations", "ai_confirmation_window_s", "ai_fast_confirm_confidence", "ai_confidence", "arrival_direction",
          "arrival_min_progress", "stationary_confirm_s", "confirmation_polygon",
          "route_role", "operator_hint", "alert_stage", "light_polygon",
          "strict_change", "change_threshold", "connected_pct",
          "taillight_red_pct", "taillight_change_pct", "taillight_wake_pct", "bottom_edge_entry", "bottom_edge_y",
          "headlight_pct", "headlight_max_pct", "headlight_white", "headlight_learn_s",
          "ai_confidence_moving", "ai_moving_entry_polygon", "approach_motion", "ai_context",
          "ignore_from_routes", "toward_direction"}

ALLOWED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def _number(value, label: str, low: float, high: float, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value) or not low <= value <= high or (positive and value <= 0):
        raise ValueError(f"{label} must be {'greater than 0 and ' if positive else ''}between {low} and {high}.")
    return value


def validate_config(payload: dict, original: dict | None = None) -> dict:
    """Validate editable fields and retain the file's existing top-level metadata."""
    if not isinstance(payload, dict) or not isinstance(payload.get("zones"), list):
        raise ValueError("Configuration must contain a zones list.")
    if not 1 <= len(payload["zones"]) <= 32:
        raise ValueError("Keep between 1 and 32 zones.")
    result = copy.deepcopy(original or {})
    result["zones"] = copy.deepcopy(payload["zones"])
    names = set()
    for index, zone in enumerate(result["zones"], 1):
        label = f"Zone {index}"
        if not isinstance(zone, dict) or set(zone) - FIELDS:
            raise ValueError(f"{label} contains unsupported fields.")
        name = zone.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError(f"{label} needs a label of 1–80 characters.")
        name = name.strip()
        if name.casefold() in names:
            raise ValueError("Zone labels must be unique.")
        names.add(name.casefold())
        zone["name"] = name
        rect = zone.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError(f"{label} rectangle must contain x, y, width, height.")
        x, y, width, height = [_number(v, f"{label} rectangle", 0, 1) for v in rect]
        if width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
            raise ValueError(f"{label} rectangle must have positive dimensions and stay inside the image.")
        zone["rect"] = [x, y, width, height]
        if "polygon" in zone:
            zone["polygon"] = validate_polygon(zone["polygon"], f"{label} outline")
            if any(px < x - 1e-9 or py < y - 1e-9 or
                   px > x + width + 1e-9 or py > y + height + 1e-9
                   for px, py in zone["polygon"]):
                raise ValueError(f"{label} outline must stay inside its rectangle.")
        if "exclude_polygons" in zone:
            exclusions = zone["exclude_polygons"]
            if not isinstance(exclusions, list) or len(exclusions) > 16:
                raise ValueError(f"{label} can contain up to 16 ignored areas.")
            zone["exclude_polygons"] = [
                validate_polygon(points, f"{label} ignored area {index + 1}")
                for index, points in enumerate(exclusions)]
        if zone.get("confirmation_polygon") is not None:
            zone["confirmation_polygon"] = validate_polygon(
                zone["confirmation_polygon"], f"{label} audible confirmation area")
            if any(px < x - 1e-9 or py < y - 1e-9 or
                   px > x + width + 1e-9 or py > y + height + 1e-9
                   for px, py in zone["confirmation_polygon"]):
                raise ValueError(f"{label} confirmation area must stay inside its rectangle.")
        if zone.get("light_polygon") is not None:
            zone["light_polygon"] = validate_polygon(zone["light_polygon"], f"{label} light area")
            if any(px < x - 1e-9 or py < y - 1e-9 or
                   px > x + width + 1e-9 or py > y + height + 1e-9
                   for px, py in zone["light_polygon"]):
                raise ValueError(f"{label} light area must stay inside its rectangle.")
        if zone.get("ai_moving_entry_polygon") is not None:
            # A clipped vehicle can first appear just outside the alert zone;
            # the entry origin is therefore bounded by the image, not its zone.
            zone["ai_moving_entry_polygon"] = validate_polygon(
                zone["ai_moving_entry_polygon"], f"{label} weak vehicle entry area")
            if zone.get("ai_confidence_moving") is None or zone.get("arrival_direction") is None:
                raise ValueError(f"{label} weak vehicle entry area requires a moving confidence floor and direction.")
        if "route_role" in zone and zone["route_role"] not in ("", "exit", "entrance", "pedestrian"):
            raise ValueError(f"{label} route role must be exit, entrance, or pedestrian.")
        if "alert_stage" in zone and zone["alert_stage"] not in ("", "advance", "barrier"):
            raise ValueError(f"{label} alert stage must be advance or barrier.")
        if "operator_hint" in zone and (not isinstance(zone["operator_hint"], str)
                or len(zone["operator_hint"]) > 240):
            raise ValueError(f"{label} operator hint must be text of at most 240 characters.")
        if "trigger" in zone and zone["trigger"] not in ("ai", "ai_light", "motion", "hybrid", "taillight",
                                                         "headlights"):
            raise ValueError(f"{label} alert source must be ai, ai_light, motion, hybrid, taillight or headlights.")
        for field, low, high in (("headlight_pct", .1, 50), ("headlight_max_pct", 1, 100),
                                 ("headlight_white", 150, 255), ("headlight_learn_s", 10, 3600)):
            if field in zone and zone[field] is not None:
                _number(zone[field], f"{label} {field}", low, high)
        for field, low, high in (("taillight_red_pct", .1, 50), ("taillight_change_pct", 1, 100),
                                 ("taillight_wake_pct", .5, 100), ("ai_confidence_moving", .05, .95)):
            if field in zone and zone[field] is not None:
                _number(zone[field], f"{label} {field}", low, high)
        if "bottom_edge_entry" in zone and not isinstance(zone["bottom_edge_entry"], bool):
            raise ValueError(f"{label} bottom_edge_entry must be true or false.")
        if "bottom_edge_y" in zone:
            _number(zone["bottom_edge_y"], f"{label} bottom edge", .95, 1.)
        if "priority" in zone and not isinstance(zone["priority"], bool):
            raise ValueError(f"{label} priority must be true or false.")
        routes = zone.get("ignore_from_routes")
        if routes is not None and (not isinstance(routes, list) or len(routes) > 8 or any(
                not isinstance(r, str) or not r or len(r) > 80 for r in routes)):
            raise ValueError(f"{label} ignore_from_routes must be a list of route group names.")
        toward = zone.get("toward_direction")
        if toward is not None:
            if not isinstance(toward, (list, tuple)) or len(toward) != 2:
                raise ValueError(f"{label} toward direction must contain dx and dy.")
            for value in toward:
                _number(value, f"{label} toward direction", -1, 1)
            if math.hypot(*toward) < .01:
                raise ValueError(f"{label} toward direction must not be zero.")
        if "ai_context" in zone and not isinstance(zone["ai_context"], bool):
            raise ValueError(f"{label} AI context must be true or false.")
        if "route_group" in zone and (not isinstance(zone["route_group"], str) or len(zone["route_group"]) > 80):
            raise ValueError(f"{label} route group must be text of at most 80 characters.")
        if "ai_confirm_observations" in zone and (type(zone["ai_confirm_observations"]) is not int
                or not 2 <= zone["ai_confirm_observations"] <= 8):
            raise ValueError(f"{label} AI confirmation must be an integer from 2 to 8.")
        for field, low, high in (("ai_confidence", 0, 1), ("ai_confirmation_window_s", 0, 1.5), ("arrival_min_progress", .001, .5),
                                 ("stationary_confirm_s", 0, 30)):
            if field in zone:
                _number(zone[field], f"{label} {field}", low, high)
        if "ai_fast_confirm_confidence" in zone:
            fast = _number(zone["ai_fast_confirm_confidence"], f"{label} strong confirmation", 0, 1)
            if fast and (fast < .60 or fast < zone.get("ai_confidence", 0)
                         or zone.get("alert_stage") != "barrier" or not zone.get("confirmation_polygon")
                         or zone.get("ai_confirm_observations", 2) < 3):
                raise ValueError(f"{label} strong confirmation requires a barrier core, three ordinary observations and a confidence of at least .60 and the ordinary floor.")
        direction = zone.get("arrival_direction")
        if "approach_motion" in zone and not isinstance(zone['approach_motion'], bool):
            raise ValueError(f"{label} approach motion must be true or false.")
        if zone.get('approach_motion') and (zone.get('alert_stage') != 'advance'
                or zone.get('route_role') != 'entrance' or not zone.get('route_group')
                or zone.get('trigger') != 'ai' or not zone.get('polygon') or direction is None):
            raise ValueError(f"{label} approach motion requires an entrance advance AI region with a direction.")
        if direction is not None:
            if not isinstance(direction, (list, tuple)) or len(direction) != 2:
                raise ValueError(f"{label} arrival direction must contain dx and dy.")
            for value in direction:
                _number(value, f"{label} arrival direction", -1, 1)
            if math.hypot(*direction) < .01:
                raise ValueError(f"{label} arrival direction must not be zero.")
        if "classes" in zone:
            classes = zone["classes"]
            if (not isinstance(classes, list) or len(classes) > len(ALLOWED_CLASSES)
                    or any(not isinstance(value, str) or value not in ALLOWED_CLASSES
                           for value in classes) or len(classes) != len(set(classes))):
                raise ValueError(f"{label} object types must be unique supported labels.")
            if not classes and zone.get("trigger", "hybrid") not in ("motion", "taillight", "headlights"):
                raise ValueError(f"{label} needs at least one object type for AI alerts.")
        for field in ("motion", "light", "strict_change"):
            if field in zone and not isinstance(zone[field], bool):
                raise ValueError(f"{label} {field} must be true or false.")
        for field in ("severity", "light_severity"):
            if field in zone and zone[field] not in ("early", "main", "fault"):
                raise ValueError(f"{label} {field} must be early, main, or fault.")
        if zone.get("strict_change") and (zone.get("trigger") != "motion" or not zone.get("motion", True)
                                          or zone.get("on_frames", 2) != 1):
            raise ValueError(f"{label} strict change requires a one-frame motion alarm.")
        for field, maximum in (("motion_pct", 100), ("light_delta", 255), ("change_threshold", 255),
                               ("connected_pct", 100)):
            if field in zone:
                _number(zone[field], f"{label} {field}", 0, maximum, positive=True)
        for field in ("on_frames", "off_frames"):
            if field in zone and (type(zone[field]) is not int or not 1 <= zone[field] <= 1000):
                raise ValueError(f"{label} {field} must be an integer from 1 to 1000.")
    return result


def read_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        raw = {"zones": raw}
    return validate_config(raw, raw)


def _atomic_write(path: Path, contents: bytes) -> None:
    filename = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            filename = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, path)
    finally:
        if filename is not None:
            filename.unlink(missing_ok=True)


def save_config(path: Path, payload: dict, *, armed: bool) -> dict:
    """Save only while disarmed, retaining an exact dated backup before replacement."""
    if armed:
        raise PermissionError("Disarm the watcher before saving zone changes.")
    original = read_config(path)
    updated = validate_config(payload, original)
    contents = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}-{uuid.uuid4().hex[:8]}")
    _atomic_write(backup, path.read_bytes())
    _atomic_write(path, contents)
    return updated


def render_page(path: Path) -> str:
    """Use the same editor for direct rendering and the live dashboard route."""
    from dashboard_ui import zone_editor_page
    return zone_editor_page(read_config(path))
