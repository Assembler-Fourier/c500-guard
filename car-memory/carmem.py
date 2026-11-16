"""Car memory: remember each vehicle that comes IN and recognise the same vehicle when it goes OUT.

Separate from BarrierWatcher: it reads local sighting files. Operation requires a local
AUTHORISATION or AUTHORIZATION note. Appearance matches are suggestions for review,
not exact identity, number-plate recognition or a count of vehicles on site.

Inputs:
  * clean vehicle sightings written by BarrierWatcher (%LOCALAPPDATA%\\c500-guard\\vehicle-sightings), when present:
    every followed vehicle, clean crops;
  * otherwise / also the 30-s alarm videos (annotated, alarm-only).
Appearance signature per vehicle:
  * learned: Intel vehicle-attributes-recognition-barrier-0042 (colour 7 + type 4 probabilities) and
    vehicle-reid-0001 (OSNet, 512-d appearance fingerprint), run with OpenVINO on the processor;
  * classical: hue/value histograms on lit paint, mean Lab colour, colour name, night/infrared flags.

Commands:
  selftest   synthetic cars; no real footage
  bench CLIP timing on one alarm video; stores nothing
  run        build the memory and data/report.html
  calibrate  same-vehicle vs different-vehicle similarity on stored sightings, to set thresholds

Privacy: local only; data older than --keep-hours (12) is deleted at every start, including the report.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"
AUTH_DIRS = (HERE, HERE.parent)
SITE = Path(r"C:\c500-guard")
STATE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "c500-guard"
PUBLISHED = STATE / "car-memory" / "report.html"      # served by BarrierWatcher at /car-memory
MODEL = SITE / "models" / "yolox_tiny.onnx"
FFMPEG = SITE / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
VEHICLES = {"car", "bus", "truck", "motorcycle"}
HUE_BINS = 12
VALUE_BINS = 4
LEARNED_COLOURS = ("white", "grey", "yellow", "red", "green", "blue", "black")
LEARNED_TYPES = ("car", "van", "truck", "bus")


# ------------------------------------------------------------------------------------------------ authorisation
def find_authorisation():
    """(path, text) of the authorisation note: AUTHORISATION/AUTHORIZATION*.txt here or in the parent folder."""
    for folder in AUTH_DIRS:
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            if re.match(r"^authori[sz]ation.*\.txt$", name, re.I):
                text = (folder / name).read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return folder / name, " ".join(text.split())[:300]
    return None, ""


# ------------------------------------------------------------------------------------------------ learned models
class VehicleModels:
    """Intel colour/type + appearance-fingerprint models on the processor (OpenVINO). Missing files: classical only."""

    def __init__(self, folder: Path = MODELS, threads: int = 2):
        self.available, self.error = False, ""
        try:
            sys.path.append(str(SITE / "tools" / "openvino"))
            import openvino as ov
            core = ov.Core()
            attr = core.read_model(str(folder / "vehicle-attributes-recognition-barrier-0042.xml"))
            reid = core.read_model(str(folder / "osnet_ain_x1_0_vehicle_reid.onnx"))
            reid.reshape({reid.inputs[0]: [1, 3, 208, 208]})
            config = {"INFERENCE_NUM_THREADS": threads}
            self._attr = core.compile_model(attr, "CPU", config).create_infer_request()
            self._reid = core.compile_model(reid, "CPU", config).create_infer_request()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - the classical signature still works
            self.error = f"{type(exc).__name__}: {exc}"[:200]

    def attributes(self, crop_rgb):
        """(colour probabilities x7, type probabilities x4). Input: BGR 72x72, 0-255 (mean is inside the model)."""
        x = cv2.resize(crop_rgb, (72, 72), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        out = self._attr.infer({0: np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)})
        values = list(out.values())
        colour, kind = (values[0], values[1]) if values[0].size == 7 else (values[1], values[0])
        return colour.reshape(-1), kind.reshape(-1)

    def embed(self, crop_rgb):
        """512-d appearance fingerprint, unit length. Input: RGB 208x208 (the model normalises itself)."""
        x = cv2.resize(crop_rgb, (208, 208), interpolation=cv2.INTER_AREA)
        vector = list(self._reid.infer({0: np.ascontiguousarray(x.transpose(2, 0, 1)[None],
                                                                   dtype=np.float32)}).values())[0].reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


# ------------------------------------------------------------------------------------------- appearance signature
def body_region(crop: np.ndarray) -> np.ndarray:
    """The painted body: the middle of the box (skips windows/roof, road/shadow and drawn box outlines)."""
    h, w = crop.shape[:2]
    return crop[int(h * .35):max(int(h * .35) + 1, int(h * .85)), int(w * .15):max(int(w * .15) + 1, int(w * .85))]


def colour_name(hue: float, saturation: float, value: float, chroma_share: float) -> str:
    if chroma_share < .3:
        return "black" if value < 55 else "dark grey" if value < 110 else "silver/grey" if value < 175 else "white"
    if hue < 10 or hue >= 170:
        return "red"
    return ("orange" if hue < 22 else "yellow" if hue < 35 else "green" if hue < 85 else "blue" if hue < 130
            else "purple")


def suppress_lights(crop_rgb: np.ndarray) -> np.ndarray:
    """Replace lamp pixels (bright saturated red tail/brake lights, blown-out headlights) with the crop's typical
    paint colour. Seen from behind at night, tail lights otherwise make every car "red"; from the front, headlights
    make every car "white"."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_lamp = ((h < 10) | (h >= 140)) & (s > 80) & (v > 110)         # lamp and its red/pink glow
    blown = v > 245
    lamps = cv2.dilate((red_lamp | blown).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if not lamps.any() or lamps.mean() > .6:
        return crop_rgb
    out = crop_rgb.copy()
    out[lamps] = np.median(crop_rgb[~lamps].reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def is_night(at: float | None, scene_brightness: float) -> bool:
    """Night by the clock (UK: about 16:30-07:30 in winter, 21:00-05:00 in summer), or a dark picture."""
    if scene_brightness < 70:
        return True
    if at is None:
        return False
    t = time.localtime(at)
    hour = t.tm_hour + t.tm_min / 60
    dusk, dawn = {1: (16.5, 8), 2: (17.5, 7.25), 3: (18.5, 6.25), 4: (20, 6.25), 5: (21, 5.25), 6: (21.5, 4.75),
                  7: (21.25, 5), 8: (20.5, 5.75), 9: (19.25, 6.75), 10: (18.25, 7.5), 11: (16.5, 7.5),
                  12: (16, 8)}[t.tm_mon]
    return hour >= dusk or hour < dawn


def usable_crop(box, width=640, height=360, edge=3) -> bool:
    """A whole vehicle, not a slice at the picture's edge or under the camera icon (drawn at the lower left)."""
    x1, y1, x2, y2 = box
    if x1 <= edge or x2 >= width - edge or y2 >= height - edge or x2 - x1 < 36 or y2 - y1 < 20:
        return False
    icon = (0.02 * width, 0.63 * height, 0.134 * width, 0.82 * height)
    overlap = max(0., min(x2, icon[2]) - max(x1, icon[0])) * max(0., min(y2, icon[3]) - max(y1, icon[1]))
    return overlap == 0                                                  # the drawn icon is never a vehicle


def signature(crop_rgb: np.ndarray, scene_brightness: float, models: VehicleModels | None = None,
              at: float | None = None) -> dict | None:
    """Signature of one vehicle crop (RGB uint8). None when there is too little usable paint."""
    crop_rgb = suppress_lights(crop_rgb)
    body = body_region(crop_rgb)
    if body.size == 0:
        return None
    hsv = cv2.cvtColor(body, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    usable = (hsv[:, 2] > 25) & (hsv[:, 2] < 245)            # no deep shadow, no blown headlight
    if usable.sum() < 40:
        return None
    pixels = hsv[usable]
    # Hue only on reasonably lit paint: in dark pixels noise gives random hues (black would read "blue").
    chroma = (pixels[:, 1] >= 50) & (pixels[:, 2] >= 60)
    share = float(chroma.mean())
    hue_hist = np.histogram(pixels[chroma, 0], bins=HUE_BINS, range=(0, 180), weights=pixels[chroma, 1])[0]
    value_hist = np.histogram(pixels[~chroma, 2], bins=VALUE_BINS, range=(0, 256))[0].astype(np.float32)
    hue_hist = hue_hist / max(hue_hist.sum(), 1e-9) * share
    value_hist = value_hist / max(value_hist.sum(), 1e-9) * (1 - share)
    vector = np.concatenate([hue_hist, value_hist])
    vector = vector / max(vector.sum(), 1e-9)
    lab = cv2.cvtColor(body, cv2.COLOR_RGB2LAB).reshape(-1, 3)[usable].astype(np.float32).mean(axis=0)
    hue = float(np.median(pixels[chroma, 0])) if chroma.any() else 0.
    night = is_night(at, scene_brightness)
    strong_paint = (share >= .35 and 22 <= hue < 130
                    and float(np.median(pixels[chroma, 1])) >= 90 if chroma.any() else False)
    if night and not strong_paint:
        # At night tail-light glow turns paint red/pink/purple on this camera, so only the tone is honest.
        # A strong yellow, green or blue under the white floodlights (an ambulance, a liveried van) still counts.
        value = float(np.median(pixels[:, 2]))
        name = "dark" if value < 70 else "mid-tone" if value < 150 else "light"
    else:
        name = colour_name(hue, float(np.median(pixels[:, 1])), float(np.median(pixels[:, 2])), share)
    result = {"hist": [round(float(v), 5) for v in vector], "lab": [round(float(v), 2) for v in lab],
              "colour": name, "night": night, "colourless": bool(float(pixels[:, 1].mean()) < 12),
              "pixels": int(usable.sum()), "width": int(crop_rgb.shape[1]), "height": int(crop_rgb.shape[0])}
    if models is not None and models.available and min(crop_rgb.shape[:2]) >= 24:
        colour_p, type_p = models.attributes(crop_rgb)
        result["colour_p"] = [round(float(v), 4) for v in colour_p]
        result["type_p"] = [round(float(v), 4) for v in type_p]
        result["emb"] = [round(float(v), 5) for v in models.embed(crop_rgb)]
    return result


def combine(signatures: list[dict]) -> dict | None:
    """Several frames of the same vehicle make one steadier signature."""
    signatures = [s for s in signatures if s]
    if not signatures:
        return None
    weights = np.array([s["pixels"] for s in signatures], np.float32)
    hist = np.average(np.array([s["hist"] for s in signatures]), axis=0, weights=weights)
    lab = np.average(np.array([s["lab"] for s in signatures]), axis=0, weights=weights)
    names = [s["colour"] for s in signatures]
    out = {"hist": [round(float(v), 5) for v in hist / max(hist.sum(), 1e-9)],
           "lab": [round(float(v), 2) for v in lab], "colour": max(set(names), key=names.count),
           "night": sum(s["night"] for s in signatures) > len(signatures) / 2,
           "colourless": sum(s.get("colourless", False) for s in signatures) > len(signatures) / 2,
           "frames": len(signatures), "pixels": int(weights.sum()),
           "size": round(float(np.median([s.get("width", 0) * s.get("height", 0) for s in signatures])) ** .5, 1)}
    learned = [s for s in signatures if "emb" in s]
    if learned:
        emb = np.mean([s["emb"] for s in learned], axis=0)
        out["emb"] = [round(float(v), 5) for v in emb / max(float(np.linalg.norm(emb)), 1e-9)]
        out["colour_p"] = [round(float(v), 4) for v in np.mean([s["colour_p"] for s in learned], axis=0)]
        out["type_p"] = [round(float(v), 4) for v in np.mean([s["type_p"] for s in learned], axis=0)]
        out["learned_colour"] = LEARNED_COLOURS[int(np.argmax(out["colour_p"]))]
        out["learned_type"] = LEARNED_TYPES[int(np.argmax(out["type_p"]))]
    return out


# ------------------------------------------------------------------------------------------------------ matching
NEIGHBOURS = {frozenset(p) for p in (("black", "dark grey"), ("dark grey", "silver/grey"), ("silver/grey", "white"),
                                     ("red", "orange"), ("orange", "yellow"), ("blue", "purple"),
                                     ("dark", "mid-tone"), ("mid-tone", "light"))}
# Appearance-fingerprint cosine: below LOW nothing alike, above HIGH as alike as one car seen twice. Defaults until
# `calibrate` has measured this site; data/calibration.json overrides them.
REID_LOW, REID_HIGH = .35, .80


def _bc(p, q) -> float:
    p, q = np.clip(np.asarray(p, float), 0, None), np.clip(np.asarray(q, float), 0, None)
    return float(np.sum(np.sqrt(p / max(p.sum(), 1e-9) * q / max(q.sum(), 1e-9))))


def reid_scale():
    try:
        saved = json.loads((DATA / "calibration.json").read_text(encoding="utf-8"))
        return float(saved["reid_low"]), float(saved["reid_high"])
    except (OSError, ValueError, KeyError, TypeError):
        return REID_LOW, REID_HIGH


def similarity(a: dict, b: dict, scale=None) -> float:
    """0..1 from colour histogram, Lab colour, colour name, type and (when both have it) the learned fingerprint."""
    hist = _bc(a["hist"], b["hist"])
    la, lb = np.array(a["lab"]), np.array(b["lab"])
    lab_sim = max(0., 1 - math.dist((la[0] / 2, la[1], la[2]), (lb[0] / 2, lb[1], lb[2])) / 60)
    name = 1. if a["colour"] == b["colour"] else .5 if frozenset((a["colour"], b["colour"])) in NEIGHBOURS else 0.
    kind = 1. if a.get("type", "car") == b.get("type", "car") else .3
    if "emb" in a and "emb" in b:
        low, high = scale or reid_scale()
        cosine = float(np.dot(a["emb"], b["emb"]))
        reid = min(1., max(0., (cosine - low) / max(high - low, 1e-6)))
        colour_p = _bc(a["colour_p"], b["colour_p"])
        type_p = _bc(a["type_p"], b["type_p"])
        if a.get("night") or b.get("night"):
            # Learned colour is unreliable at night; lean on the fingerprint, type and the lamp-free colour.
            return round(.45 * reid + .20 * hist + .10 * lab_sim + .05 * colour_p + .10 * type_p + .10 * name, 4)
        return round(.40 * reid + .15 * hist + .10 * lab_sim + .15 * colour_p + .10 * type_p + .10 * name, 4)
    return round(.45 * hist + .25 * lab_sim + .15 * name + .15 * kind, 4)


STRONG, POSSIBLE, MARGIN = .80, .65, .08


def pair_exits(entries: list[dict], exits: list[dict], window_s: float = 12 * 3600,
               min_stay_s: float = 60.) -> list[dict]:
    """Pair each exit with an earlier, still-unmatched entry. Honest bands: strong / possible / none.

    Only a STRONG match (high score, clearly ahead of the runner-up, same light conditions, real colour) consumes
    an entry; a possible match is shown for the operator to judge and leaves the entry available. An entry less
    than ``min_stay_s`` before the exit is not a candidate: a car that turns round is not "in, then out".
    """
    used, results = set(), []
    for out in sorted(exits, key=lambda e: e["at"]):
        candidates = sorted(((similarity(out, entry), entry["id"]) for entry in entries
                             if entry["id"] not in used and min_stay_s <= out["at"] - entry["at"] <= window_s),
                            reverse=True)
        best, runner = (candidates + [(0., None), (0., None)])[:2]
        score, entry_id = best
        margin = round(score - runner[0], 4)
        entry = next((e for e in entries if e["id"] == entry_id), None)
        light_differs = entry is not None and (entry.get("night") != out.get("night")
                                               or entry.get("colourless") or out.get("colourless"))
        if entry_id is not None and score >= STRONG and margin >= MARGIN and not light_differs:
            band = "strong"
            used.add(entry_id)
        elif entry_id is not None and score >= POSSIBLE:
            band = "possible"
        else:
            band, entry_id = "none", None
        results.append({"out": out["id"], "in": entry_id, "score": score, "margin": margin, "band": band,
                        "light_differs": bool(light_differs)})
    return results


# ------------------------------------------------------------------------------------------- tracks and direction
def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def build_tracks(frames: list[tuple[float, list[dict]]], max_gap_s: float = 1.5) -> list[dict]:
    """Greedy IoU tracking of vehicle detections across sampled frames: [(time_s, detections)]."""
    tracks, live = [], []
    for at, detections in frames:
        live = [t for t in live if at - t["last"] <= max_gap_s]
        free = list(range(len(detections)))
        for track in sorted(live, key=lambda t: -t["last"]):
            best = max(free, key=lambda i: iou(track["boxes"][-1], detections[i]["bbox"]), default=None)
            if best is not None and iou(track["boxes"][-1], detections[best]["bbox"]) >= .2:
                d = detections[best]
                track["boxes"].append(d["bbox"]), track["times"].append(at), track["labels"].append(d["label"])
                track["last"] = at
                free.remove(best)
        for i in free:
            d = detections[i]
            track = {"boxes": [d["bbox"]], "times": [at], "labels": [d["label"]], "last": at}
            tracks.append(track)
            live.append(track)
    return [t for t in tracks if len(t["boxes"]) >= 3]


def point_in(point, polygon) -> bool:
    x, y = point
    hit = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def direction_of(grounds: list, zones: list[dict]) -> str:
    """'in' / 'out' / 'unclear' from the path, in order: a car leaving goes through the exit lanes and then up the
    top road (which the entrance zones cover); a car coming in comes DOWN the top road."""
    path = []
    for g in grounds:
        roles = {z.get("route_role") for z in zones if point_in(g, z.get("polygon") or [])}
        path.append("exit" if "exit" in roles else "entrance" if "entrance" in roles else None)
    path = [p for p in path if p]
    first = path[0] if path else None
    dy = grounds[-1][1] - grounds[0][1] if len(grounds) >= 2 else 0.
    if "exit" in path and (dy < -.02 or (first == "exit" and "entrance" in path)):
        return "out"                      # moved away, or went from the exit lanes up the top road
    if path and "exit" not in path and dy < -.05:
        return "out"
    if "entrance" in path and "exit" not in path and dy > .02:
        return "in"
    return "unclear"


def direction(track: dict, zones: list[dict], width=640, height=360) -> str:
    return direction_of([((b[0] + b[2]) / 2 / width, b[3] / height) for b in track["boxes"]], zones)


# ------------------------------------------------------------------------------------------------- storage/report
def connect() -> sqlite3.Connection:
    DATA.mkdir(exist_ok=True)
    db = sqlite3.connect(DATA / "memory.db")
    db.execute("""CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY, at REAL, direction TEXT, type TEXT,
                  colour TEXT, night INTEGER, signature TEXT, thumb BLOB, clip TEXT, created REAL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS suggestions (id INTEGER PRIMARY KEY, created REAL, out_id INTEGER,
                  in_id INTEGER, band TEXT, score REAL, margin REAL, decided TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS processed_sources
                  (source TEXT PRIMARY KEY, event_at REAL NOT NULL, fingerprint TEXT NOT NULL DEFAULT '')""")
    if 'fingerprint' not in {row[1] for row in db.execute('PRAGMA table_info(processed_sources)')}:
        db.execute("ALTER TABLE processed_sources ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''")
    db.execute("CREATE TABLE IF NOT EXISTS report_state (id INTEGER PRIMARY KEY CHECK (id=1), pending INTEGER NOT NULL)")
    db.execute("INSERT OR IGNORE INTO report_state(id,pending) VALUES(1,1)")
    return db


def purge(db: sqlite3.Connection, keep_hours: float) -> int:
    """Delete sightings whose EVENT is older than the window (not just old rows), their suggestions, and the last
    report (it holds thumbnails and is rebuilt by every run)."""
    cutoff = time.time() - keep_hours * 3600
    removed = db.execute("DELETE FROM sightings WHERE at < ? OR created < ?", (cutoff, cutoff)).rowcount
    db.execute("DELETE FROM suggestions WHERE created < ? OR out_id NOT IN (SELECT id FROM sightings)", (cutoff,))
    db.execute("DELETE FROM processed_sources WHERE event_at < ?", (cutoff,))
    if removed:
        db.execute("UPDATE report_state SET pending=1 WHERE id=1")
    db.commit()
    # Keep the last valid report available while adding new sightings. Expired
    # content is removed immediately; otherwise publication replaces atomically.
    if removed:
        (DATA / "report.html").unlink(missing_ok=True)
        PUBLISHED.unlink(missing_ok=True)
    return removed


def thumbnail(crop_rgb: np.ndarray, width=240) -> bytes:
    h, w = crop_rgb.shape[:2]
    scale = min(1., width / max(w, 1), 180. / max(h, 1))
    small = cv2.resize(crop_rgb, (max(1,int(w*scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def describe(row: dict) -> str:
    # Intel's colour model learned front views by day; at night (often seen from behind, tail-light glow) the
    # classical colour, measured with the lamps removed, is the better guess.
    kind = row.get("learned_type") or row["type"]
    if row.get("night"):
        tone = row["colour"] in ("dark", "mid-tone", "light")
        return f"{row['colour']} {kind}" + (" (night: exact colour unsure)" if tone else "")
    colour = row.get("learned_colour") or row["colour"]
    return f"{colour} {kind}"


def write_report(rows: list[dict], pairs: list[dict], out: Path, authorisation: str = "", *, keep_hours=12.) -> Path:
    """Build a local read-only report and publish it atomically."""
    # The launcher uses Python -I: load the trusted adjacent renderer explicitly.
    import importlib.util
    spec = importlib.util.spec_from_file_location('carmem_report_template', HERE / 'report_template.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    page = renderer.render_report(rows, pairs, time.time(), keep_hours, authorisation)
    temporary = out.with_suffix('.tmp')
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(out)
    return out


# ------------------------------------------------------------------------------------------------- inputs
def video_frames(path: Path, fps: float = 4.):
    raw = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale=640:360",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    size = 640 * 360 * 3
    for i in range(len(raw) // size):
        yield i / fps, np.frombuffer(raw[i * size:(i + 1) * size], np.uint8).reshape(360, 640, 3)


def detector():
    sys.path.insert(0, str(SITE / "src"))                          # read-only use of the site's detector
    import object_detector
    return object_detector.ObjectDetector(str(MODEL), confidence=.4, engine="processor", threads=2)


def analyse_clip(path: Path, zones: list[dict], model, models: VehicleModels | None = None) -> list[dict]:
    """Vehicle sightings (direction, type, signature, thumbnail) in one recorded alarm video."""
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    start = os.path.getmtime(path) - float(meta.get("seconds", 30))
    frames, images = [], {}
    for at, frame in video_frames(path):
        frames.append((at, [d for d in model.detect(frame) if d["label"] in VEHICLES]))
        images[at] = frame
    sightings = []
    for track in build_tracks(frames):
        whole = [i for i in range(len(track["boxes"])) if usable_crop(track["boxes"][i])]
        order = sorted(whole, key=lambda i: -(track["boxes"][i][2] - track["boxes"][i][0])
                       * (track["boxes"][i][3] - track["boxes"][i][1]))[:5]
        at = start + track["times"][len(track["times"]) // 2]
        crops = []
        for i in order:
            frame = images[track["times"][i]]
            x1, y1, x2, y2 = (int(v) for v in track["boxes"][i])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                crops.append((crop, float(frame.mean())))
        sighting = sighting_from_crops(crops, models, at)
        if sighting:
            labels = track["labels"]
            sightings.append({"at": at,
                              "direction": direction(track, zones), "type": max(set(labels), key=labels.count),
                              "clip": path.name, **sighting})
    return sightings


def sighting_from_crops(crops, models, at=None):
    sig = combine([signature(crop, brightness, models, at) for crop, brightness in crops])
    if sig is None:
        return None
    return {"signature": sig, "thumb": thumbnail(crops[0][0])}


def source_fingerprint(path):
    """Metadata/crop versions, without opening image pixels."""
    files = [path] + sorted(path.parent.glob('crop_*.jpg')) if path.name == 'meta.json' else [path]
    parts=[]
    for item in files:
        info=item.stat()
        parts.append(f'{item.name}:{info.st_mtime_ns}:{info.st_size}')
    return '|'.join(parts)


def read_sightings_folder(folder: Path, zones: list[dict], models, cutoff: float,
                         known_sources=(), processed=None, processed_versions=None) -> list[dict]:
    """Clean sightings written by BarrierWatcher: <folder>/<name>/meta.json + crop_*.jpg (one per vehicle)."""
    found = []
    known_sources = set(known_sources)
    for meta_path in sorted(folder.glob("*/meta.json")) if folder.is_dir() else []:
        try:
            source = "live:" + meta_path.parent.name
            if source in known_sources:
                continue  # Skip before loading images or running either learned model.
            fingerprint = source_fingerprint(meta_path)
            if (processed_versions or {}).get(source) == fingerprint:
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if float(meta.get("at", 0)) < cutoff or meta.get("label") not in VEHICLES:
                continue
            if meta.get("diagnostic_only"):
                if processed is not None:
                    processed[source] = (float(meta['at']),fingerprint)
                continue
            crops = []
            images=sorted(meta_path.parent.glob("crop_*.jpg"))[:5]
            expected=min(5,int(meta.get('crops',len(images))))
            if expected < 1 or len(images) < expected:
                continue  # Incomplete external/older writer: retry when crops arrive.
            for image in images[:expected]:
                bgr = cv2.imread(str(image))
                if bgr is not None:
                    crops.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), float(meta.get("scene_brightness", 100))))
            if len(crops) < expected:
                continue
            loaded_models = models() if callable(models) else models
            sighting = sighting_from_crops(crops, loaded_models, float(meta["at"]))
            if processed is not None:
                processed[source] = (float(meta['at']),fingerprint)
            if not sighting:
                continue
            way = meta.get("direction")
            if way not in ("in", "out"):                      # older sightings: decide from the path, in order
                way = direction_of([tuple(g) for g in meta.get("grounds", [])], zones)
            found.append({"at": float(meta["at"]), "direction": way, "type": meta["label"],
                          "clip": source, **sighting})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


# ----------------------------------------------------------------------------------------------------------- CLI
def below_normal_priority() -> bool:
    try:
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p               # a 64-bit pseudo-handle, not an int
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        return bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000))
    except Exception:  # noqa: BLE001
        return False


def load_rows(db):
    rows = []
    for r in db.execute("SELECT id, at, direction, type, colour, night, signature, thumb, clip FROM sightings"):
        sig = json.loads(r[6])
        rows.append({**sig, "id": r[0], "at": r[1], "direction": r[2], "type": r[3], "colour": r[4],
                     "night": bool(r[5]), "thumb": r[7], "clip": r[8]})
    return rows


def cmd_bench(args):
    below_normal_priority()
    zones = json.loads(Path(args.zones).read_text(encoding="utf-8-sig"))["zones"]
    model, models = detector(), VehicleModels()
    started = time.perf_counter()
    sightings = analyse_clip(Path(args.clip), zones, model, models)
    took = time.perf_counter() - started
    counts = {}
    for s in sightings:
        counts[s["direction"]] = counts.get(s["direction"], 0) + 1
    print(f"detector {model.provider}; learned models {'on' if models.available else 'off: ' + models.error}; "
          f"{took:.1f} s for one clip; vehicles by direction {counts}; nothing stored")


def cmd_run(args):
    path, authorisation = find_authorisation()
    if path is None:
        print("LOCKED. Building a memory of real vehicles needs written authorisation from the site (the data"
              " controller); see README.md. Put it in AUTHORISATION.txt next to carmem.py or in its parent folder.")
        return 2
    below_normal_priority()
    db = connect()
    heartbeat = getattr(args, '_heartbeat', lambda *a, **kw: None)
    models = None
    def load_models():
        nonlocal models
        if models is None:
            heartbeat('updating', model_state='loading')
            models = VehicleModels()
            heartbeat('updating', model_state='ready' if models.available else 'classical')
        return models
    try:
        removed = purge(db, args.keep_hours)
        zones = json.loads(Path(args.zones).read_text(encoding="utf-8-sig"))["zones"]
        cutoff = time.time() - args.keep_hours * 3600
        known = {r[0] for r in db.execute("SELECT DISTINCT clip FROM sightings")}
        processed_versions=dict(db.execute("SELECT source,fingerprint FROM processed_sources"))
        new, processed = [], {}
        if args.source in ("sightings", "both"):
            new += read_sightings_folder(Path(args.sightings), zones, load_models, cutoff, known, processed,processed_versions)
        if args.source in ("clips", "both"):
            model = None
            for clip in sorted(Path(args.recordings).glob("*.mp4")):
                if clip.stat().st_mtime < cutoff or clip.name.endswith(".part.mp4") or clip.name in known:
                    continue
                fingerprint=source_fingerprint(clip)
                if processed_versions.get(clip.name)==fingerprint:
                    continue
                heartbeat('updating')
                model = model or detector()
                new += analyse_clip(clip, zones, model, load_models())
                processed[clip.name] = (clip.stat().st_mtime,fingerprint)
        for s in new:
            sig = s["signature"]
            db.execute("INSERT INTO sightings (at, direction, type, colour, night, signature, thumb, clip, created)"
                       " VALUES (?,?,?,?,?,?,?,?,?)", (s["at"], s["direction"], s["type"], sig["colour"],
                                                     int(sig["night"]), json.dumps(sig), s["thumb"], s["clip"],
                                                     time.time()))
        db.executemany("INSERT OR REPLACE INTO processed_sources (source,event_at,fingerprint) VALUES (?,?,?)",
                       [(source,at,fingerprint) for source,(at,fingerprint) in processed.items()])
        # Persist publication debt with the data, before committing it. A failed
        # render/copy must retry even when its sources are already imported.
        if (new or removed or not getattr(args, '_maintenance', False)
                or not (DATA/'report.html').exists() or not PUBLISHED.exists()):
            db.execute("UPDATE report_state SET pending=1 WHERE id=1")
        db.commit()
        # Idle polling must not make users lose report position or reload models.
        if (getattr(args, '_maintenance', False) and not new and not removed
                and (DATA/'report.html').exists() and PUBLISHED.exists()
                and not db.execute("SELECT pending FROM report_state WHERE id=1").fetchone()[0]):
            return 0
        rows = load_rows(db)
        pairs = pair_exits([r for r in rows if r["direction"] == "in"], [r for r in rows if r["direction"] == "out"],
                           window_s=args.keep_hours * 3600)
        db.execute("DELETE FROM suggestions WHERE decided IS NULL")
        decided = {r[0] for r in db.execute("SELECT out_id FROM suggestions WHERE decided IS NOT NULL")}
        db.executemany("INSERT INTO suggestions (created, out_id, in_id, band, score, margin) VALUES (?,?,?,?,?,?)",
                       [(time.time(), p["out"], p["in"], p["band"], p["score"], p["margin"])
                        for p in pairs if p['out'] not in decided])
        db.commit()
        report = write_report(rows, pairs, DATA / "report.html", authorisation, keep_hours=args.keep_hours)
        PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
        tmp = PUBLISHED.with_suffix(".tmp")
        tmp.write_bytes(report.read_bytes())
        os.replace(tmp, PUBLISHED)
        db.execute("UPDATE report_state SET pending=0 WHERE id=1")
        db.commit()
        print(f"{len(new)} new sightings; {len(rows)} in memory; {removed} expired; report updated")
        if getattr(args, "open", False):
            os.startfile(str(report))
        return 0
    finally:
        db.close()


def system_up(timeout=3.) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/state", timeout=timeout):
            return True
    except OSError:
        return False


def source_snapshot(args):
    """Cheap completed-file inventory; never opens images or loads models."""
    paths = []
    if args.source in ('sightings', 'both'):
        paths.extend(Path(args.sightings).glob('*/meta.json'))
    if args.source in ('clips', 'both'):
        paths.extend(p for p in Path(args.recordings).glob('*.mp4') if not p.name.endswith('.part.mp4'))
    rows = []
    for path in paths:
        try:
            rows.append((str(path),source_fingerprint(path)))
        except OSError:
            continue
    return tuple(sorted(rows))


def write_worker_status(state, **fields):
    """Atomic, nonidentifying heartbeat for the read-only dashboard endpoint."""
    payload = {'updated_at': time.time(), 'state': state, **fields}
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    temporary = PUBLISHED.parent / 'worker-status.tmp'
    temporary.write_text(json.dumps(payload), encoding='utf-8')
    temporary.replace(PUBLISHED.parent / 'worker-status.json')


def cmd_watch(args):
    """Poll completed metadata every10s; analyse only new sources. No idle model loads."""
    import gc
    try:                                                        # one updater at a time (app launcher or by hand)
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        args._mutex = kernel.CreateMutexW(None, False, r"Local\BarrierWatcher.CarMemory")
        if ctypes.get_last_error() == 183:
            print("the car memory is already updating")
            return 0
    except (OSError, AttributeError):
        pass
    below_normal_priority()
    down_since = None
    previous_sources = None
    last_maintenance = 0.
    last_success = None
    model_state = 'not_loaded'
    def heartbeat(state, **fields):
        nonlocal model_state
        model_state = fields.get('model_state', model_state)
        write_worker_status(state, last_success_at=last_success, model_state=model_state)
    while True:
        if system_up():
            down_since = None
            try:
                sources = source_snapshot(args)
                if sources != previous_sources or time.time()-last_maintenance >= max(30.,args.every):
                    heartbeat('updating')
                    result = cmd_run(argparse.Namespace(**{**vars(args), "open": False,
                        '_maintenance': previous_sources is not None, '_heartbeat': heartbeat}))
                    if result:
                        heartbeat('error')
                    else:
                        previous_sources = sources
                        last_maintenance = last_success = time.time()
                        heartbeat('ready')
                    gc.collect()
                else:
                    heartbeat('ready')
            except Exception as exc:  # noqa: BLE001 - try again next round
                print("update failed:", type(exc).__name__, exc)
                heartbeat('error')
        else:
            heartbeat('stopped')
            down_since = down_since or time.time()
            if time.time() - down_since > 120:
                print("BarrierWatcher is off; car memory stops too")
                return 0
        time.sleep(10.)


def cmd_calibrate(args):
    """Fingerprint similarity for the same vehicle (two halves of one sighting) vs different vehicles."""
    db = connect()
    rows = [r for r in load_rows(db) if "emb" in r]
    if len(rows) < 4:
        print("need at least 4 stored sightings with fingerprints; run first")
        return 1
    embs = np.array([r["emb"] for r in rows])
    cos = embs @ embs.T
    different = cos[np.triu_indices(len(rows), 1)]
    print(f"different vehicles: n={different.size} median {np.median(different):.2f} "
          f"95th {np.percentile(different, 95):.2f} max {different.max():.2f}")
    low, high = float(np.percentile(different, 50)), float(min(.95, np.percentile(different, 99) + .1))
    (DATA / "calibration.json").write_text(json.dumps({"reid_low": round(low, 3), "reid_high": round(high, 3),
                                                       "measured": time.strftime("%Y-%m-%d %H:%M"),
                                                       "pairs": int(different.size)}), encoding="utf-8")
    print(f"saved reid_low {low:.2f} (typical different vehicles) reid_high {high:.2f}")
    return 0


def synthetic_car(rgb, night=False, seed=0, size=(90, 150)):
    """A box-shaped 'car' crop: painted body, dark windows, road below, optional sodium-light cast and noise."""
    rng = np.random.default_rng(seed)
    h, w = size
    crop = np.zeros((h, w, 3), np.float32)
    crop[:] = (70, 70, 72)                                     # road
    crop[int(h * .15):int(h * .9), int(w * .05):int(w * .95)] = rgb
    crop[int(h * .15):int(h * .38), int(w * .2):int(w * .8)] = (25, 28, 32)    # windows
    if night:
        crop = crop * .55 + np.array((40, 22, 0))              # dim, orange-ish street light
    crop += rng.normal(0, 6, crop.shape)
    return np.clip(crop, 0, 255).astype(np.uint8)


def cmd_selftest(_args):
    paints = {"red": (170, 30, 35), "blue": (30, 60, 160), "white": (225, 225, 220), "black": (30, 30, 32),
              "silver": (165, 168, 172), "green": (40, 120, 60)}
    entries, exits, n = [], [], 0
    for i, rgb in enumerate(paints.values()):
        n += 1
        sig = combine([signature(synthetic_car(rgb, False, seed=s), 120.) for s in range(3)])
        entries.append({"id": n, "at": 1000. + i, "type": "car", **sig})
    for i, name in enumerate(paints):
        n += 1
        sig = combine([signature(synthetic_car(paints[name], False, seed=50 + s, size=(80, 140)), 120.)
                       for s in range(3)])
        exits.append({"id": n, "at": 5000. + i, "type": "car", "truth": name, **sig})
    pairs = pair_exits(entries, exits)
    names = {e["id"]: list(paints)[k] for k, e in enumerate(entries)}
    correct = sum(1 for p, e in zip(pairs, exits) if p["in"] is not None and names[p["in"]] == e["truth"])
    print("colours seen:", [e["colour"] for e in entries])
    print(f"synthetic day pairs: {correct}/{len(exits)} correct;", "bands:", [p["band"] for p in pairs])
    night_in = [{"id": 100 + k, "at": 1000. + k, "type": "car",
                 **combine([signature(synthetic_car(rgb, True, seed=s), 50.) for s in range(3)])}
                for k, rgb in enumerate(paints.values())]
    night_out = [{"id": 200 + k, "at": 5000. + k, "type": "car", "truth": 100 + k,
                  **combine([signature(synthetic_car(rgb, True, seed=70 + s, size=(80, 140)), 50.)
                             for s in range(3)])} for k, rgb in enumerate(paints.values())]
    night_pairs = pair_exits(night_in, night_out)
    night_ok = sum(1 for p, e in zip(night_pairs, night_out) if p["in"] == e["truth"])
    print(f"synthetic night pairs: {night_ok}/{len(night_out)} correct;", "bands:", [p["band"] for p in night_pairs])
    mixed = pair_exits(entries, night_out)
    print("day-in / night-out bands:", [p["band"] for p in mixed])
    ok = (correct == len(exits) and night_ok >= len(night_out) - 1
          and "black" in [e["colour"] for e in entries] and all(p["band"] != "strong" for p in mixed))
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    bench = sub.add_parser("bench")
    bench.add_argument("clip")
    bench.add_argument("--zones", default=str(SITE / "config" / "zones.json"))
    run = sub.add_parser("run")
    run.add_argument("--recordings", default=str(STATE / "recordings"))
    run.add_argument("--sightings", default=str(STATE / "vehicle-sightings"))
    run.add_argument("--source", choices=("both", "sightings", "clips"), default="both")
    run.add_argument("--zones", default=str(SITE / "config" / "zones.json"))
    run.add_argument("--keep-hours", type=float, default=12.)
    run.add_argument("--open", action="store_true", help="open the report afterwards")
    sub.add_parser("calibrate")
    watch = sub.add_parser("watch")
    watch.add_argument("--every", type=float, default=300.)
    watch.add_argument("--recordings", default=str(STATE / "recordings"))
    watch.add_argument("--sightings", default=str(STATE / "vehicle-sightings"))
    watch.add_argument("--source", choices=("both", "sightings", "clips"), default="sightings")
    watch.add_argument("--zones", default=str(SITE / "config" / "zones.json"))
    watch.add_argument("--keep-hours", type=float, default=12.)
    args = parser.parse_args(argv)
    return {"selftest": cmd_selftest, "bench": cmd_bench, "run": cmd_run, "calibrate": cmd_calibrate,
            "watch": cmd_watch}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
