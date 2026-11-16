"""Clean pictures of every vehicle the AI follows through the gate, for the separate car memory.

Enable only with appropriate local authorization and retention settings. For each
vehicle track that passes through an entrance or exit zone, up to four clean crops (from the un-annotated aligned
frame) and a small description are written when the track ends:
    <state>/vehicle-sightings/<YYYYmmdd-HHMMSS>-<track>/meta.json + crop_0.jpg ...
No number plates are read. Everything older than ``keep_hours`` (12) is deleted. Display/evidence only: nothing here
can raise, delay or silence an alarm, and any error is swallowed.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image

VEHICLES = frozenset({"car", "bus", "truck", "motorcycle"})


class VehicleSightings:
    def __init__(self, folder, *, keep_hours: float = 12., max_crops: int = 4, finish_after_s: float = 4.,
                 min_width: int = 36, min_confidence: float = .45, enabled: bool = True,
                 clock=time.monotonic, wall=time.time):
        self.folder = Path(folder)
        self.keep_hours = float(keep_hours)
        self.max_crops = int(max_crops)
        self.finish_after_s = float(finish_after_s)
        self.min_width = int(min_width)
        self.min_confidence = float(min_confidence)
        self.enabled = bool(enabled)
        self.clock, self.wall = clock, wall
        self._tracks: dict = {}
        self._last_stamp = None
        self._last_purge = 0.
        self.written = 0
        self.diagnostics_written = 0
        self.last_error = ""

    # -- feeding ---------------------------------------------------------------
    def observe(self, objects, stamp, frame, zones) -> None:
        """``objects``: the tracker's observed list for the frame taken at ``stamp``; ``frame``: that RGB frame."""
        try:
            self._observe(objects, stamp, frame, zones)
        except Exception as exc:  # noqa: BLE001 - never disturbs detection
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]

    def _observe(self, objects, stamp, frame, zones) -> None:
        now = self.clock()
        if self.enabled and stamp is not None and stamp != self._last_stamp and frame is not None:
            objects = [o for o in (objects or []) if isinstance(o, dict)]
            if all("track_id" in o for o in objects):          # the tracker has numbered this result
                self._last_stamp = stamp
                roles = {getattr(z, "name", ""): getattr(z, "route_role", "") for z in zones or []}
                brightness = float(np.asarray(frame).mean())
                for obj in objects:
                    self._add(obj, frame, roles, brightness, now)
        self._finish(now, zones)
        if now - self._last_purge > 600:
            self._last_purge = now
            self.purge()

    def _add(self, obj, frame, roles, brightness, now) -> None:
        if obj.get("label") not in VEHICLES or float(obj.get("confidence", 0)) < self.min_confidence:
            return
        box = obj.get("box_n")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return
        record = self._tracks.get(obj["track_id"])
        if record is None:
            if len(self._tracks) >= 50:
                return
            record = self._tracks[obj["track_id"]] = {
                "first": self.wall(), "labels": {}, "grounds": [], "roles": set(), "zones": set(), "crops": [],
                "brightness": [], "path": [], "observations": []}
        record["last"] = now
        record["last_wall"] = self.wall()
        record["labels"][obj["label"]] = record["labels"].get(obj["label"], 0) + 1
        ground = ((box[0] + box[2]) / 2, box[3])
        record.setdefault("first_ground", ground)
        record["grounds"].append([round(ground[0], 4), round(ground[1], 4)])
        del record["grounds"][:-60]
        here = None
        for name in obj.get("regions") or []:
            record["zones"].add(str(name))
            if roles.get(name) in ("entrance", "exit"):
                record["roles"].add(roles[name])
                here = "exit" if "exit" in (here, roles[name]) else roles[name]
        record["path"].append(here)                       # the lane role at each observation, in order
        if here is not None:
            record.setdefault("first_role", here)
        del record["path"][:-60]
        record["brightness"].append(brightness)
        # Keep the evidence which actually reached the tracker, including why
        # it did not alert. Ground points alone cannot distinguish an incorrect
        # zone from a confidence rejection or an identity jump. This is bounded,
        # local diagnostic metadata under the same retention as the crops.
        names = [str(name)[:80] for name in (obj.get("regions") or [])[:16]]
        statuses = obj.get("region_status") or {}
        record["observations"].append({
            "at": round(record["last_wall"], 3),
            "bbox": [round(float(value), 6) for value in box],
            "confidence": round(float(obj["confidence"]), 4),
            "regions": names,
            "region_status": {name: str(statuses[name])[:80] for name in names if name in statuses},
        })
        del record["observations"][:-60]
        height, width = frame.shape[:2]
        x1, y1 = int(max(0, box[0] * width)), int(max(0, box[1] * height))
        x2, y2 = int(min(width, box[2] * width)), int(min(height, box[3] * height))
        if x2 - x1 < self.min_width or y2 - y1 < 16 or not self.whole_vehicle((x1, y1, x2, y2), width, height):
            return
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        # Prefer large AND clear views: headlight glare (blown-out pixels) hides the vehicle's colour and shape.
        blown = float((crop.max(axis=2) > 245).mean())
        quality = (x2 - x1) * (y2 - y1) * max(0., 1. - 3. * blown)
        crops = record["crops"]
        if quality > 0 and (len(crops) < self.max_crops or quality > min(c[0] for c in crops)):
            crops.append((quality, crop.copy()))
            crops.sort(key=lambda c: -c[0])
            del crops[self.max_crops:]

    @staticmethod
    def whole_vehicle(box, width, height, edge=3) -> bool:
        """Not a slice at the picture's edge, and not touching the camera icon in the lower left of the view."""
        x1, y1, x2, y2 = box
        if x1 <= edge or x2 >= width - edge or y2 >= height - edge:
            return False
        icon = (0.02 * width, 0.63 * height, 0.134 * width, 0.82 * height)
        return min(x2, icon[2]) <= max(x1, icon[0]) or min(y2, icon[3]) <= max(y1, icon[1])

    def _finish(self, now, zones) -> None:
        for track_id in [t for t, r in self._tracks.items() if now - r["last"] > self.finish_after_s]:
            record = self._tracks.pop(track_id)
            if record["roles"] and record.get("observations"):
                self._write(track_id, record, diagnostic_only=not record["crops"] or self.fragment(record))

    @staticmethod
    def fragment(record) -> bool:
        """A moment of a car the tracker briefly lost (one picture, under a second): the full sighting has it."""
        return len(record["crops"]) < 2 and record["last_wall"] - record["first"] < 1.

    @staticmethod
    def direction(record) -> str:
        """Order matters: a car leaving goes through the exit lanes and then drives up the top road (which the
        entrance zones cover); a car coming in comes down the top road. 15 Nov evening: 15 of 17 leaving cars were
        "unclear" under the old any-zone rule."""
        grounds = record["grounds"]
        dy = grounds[-1][1] - record.get("first_ground", grounds[0])[1] if len(grounds) >= 2 else 0.
        path = [role for role in record.get("path", []) if role]
        first = record.get("first_role", path[0] if path else None)
        roles = record.get("roles", set(path))
        if "exit" in roles and (dy < -.02 or (first == "exit" and "entrance" in roles)):
            return "out"                                  # moved away, or went from the exit lanes up the top road
        if roles and "exit" not in roles and dy < -.05:
            return "out"                                  # moving away up the top road, after the exit barrier
        if "entrance" in roles and "exit" not in roles and dy > .02:
            return "in"
        return "unclear"

    def _write(self, track_id, record, diagnostic_only=False) -> None:
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(record["first"])) + f"-{int(track_id)}"
        target = self.folder / name
        # Tracker IDs restart with the detector. Preserve an earlier completed
        # record if another session happens to reuse its second/id pair.
        if target.exists():
            target = self.folder / (name + "-" + uuid.uuid4().hex[:8])
        target.mkdir(parents=True, exist_ok=True)
        # A rejected crop/one-frame fragment still explains a missed early
        # arrival. Keep bounded metadata under the same retention, no extra
        # images, and explicitly exclude it from vehicle-memory counts.
        crops = [] if diagnostic_only else record["crops"]
        for index, (_area, crop) in enumerate(crops):
            Image.fromarray(crop).save(target / f"crop_{index}.jpg", quality=88)
        labels = record["labels"]
        meta = {"at": round((record["first"] + record["last_wall"]) / 2, 2), "first": round(record["first"], 2),
                "last": round(record["last_wall"], 2), "track_id": int(track_id),
                "label": max(labels, key=labels.get), "direction": self.direction(record),
                "zones": sorted(record["zones"]), "grounds": record["grounds"][::max(1, len(record["grounds"]) // 20)],
                "first_ground": list(record.get("first_ground", record["grounds"][0])),
                "scene_brightness": round(float(np.median(record["brightness"])), 1), "crops": len(crops),
                "diagnostic_only": bool(diagnostic_only),
                "diagnostic_reason": ("no usable crop" if not record["crops"] else "brief track fragment") if diagnostic_only else "",
                "observations": record.get("observations", [])}
        # Publish metadata last and atomically: a reader never consumes a
        # partially written completed-sighting manifest.
        temporary = target / "meta.tmp"
        temporary.write_text(json.dumps(meta), encoding="utf-8")
        temporary.replace(target / "meta.json")
        if diagnostic_only:
            self.diagnostics_written += 1
        else:
            self.written += 1

    # -- retention -----------------------------------------------------------------
    def purge(self) -> int:
        """Delete sightings older than keep_hours."""
        cutoff = self.wall() - self.keep_hours * 3600
        removed = 0
        if not self.folder.is_dir():
            return 0
        for entry in self.folder.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    def flush(self) -> None:
        """Shutdown: write what is complete enough, so a vehicle in view at OFF is not lost."""
        try:
            for track_id in list(self._tracks):
                record = self._tracks.pop(track_id)
                if record["roles"] and record.get("observations"):
                    self._write(track_id, record, diagnostic_only=not record["crops"] or self.fragment(record))
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict:
        return {"enabled": self.enabled, "following": len(self._tracks), "written": self.written,
                "diagnostics_written": self.diagnostics_written,
                "keep_hours": self.keep_hours, "error": self.last_error}
