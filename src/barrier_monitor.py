"""Template-based barrier readings and a frozen empty-road comparison.

Geometry is calibrated for the current 640x360 monitor view. This module does
not infer landmarks, move barriers or learn an empty road from missing AI boxes.
Unknown observations are never reported as down or clear.
"""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np


WIDTH, HEIGHT = 640, 360
MAX_AGE_S = 3.0
STABLE_S = 0.30
IDS = ("exit", "entrance")
POSITIONS = ("down", "up")
LINES = {
    "exit": {"down": [[258/640, 61/360], [359/640, 69/360]],
             "up": [[361/640, 14/360], [360/640, 67/360]]},
    "entrance": {"down": [[334/640, 82/360], [487/640, 98/360]],
                 "up": [[330/640, 14/360], [330/640, 79/360]]},
}
FEATURE_SHAPE = (64, 15)
# One template per position cannot describe an arm that is floodlit at night and
# daylit in the afternoon: the same arm scores three times the allowed error
# across those two eras. Several calibrated variants are kept per position and
# the closest one wins, exactly as the scene aligner keeps lighting references.
MAX_VARIANTS = 4
TEMPLATE_SHAPE = (2, 2, MAX_VARIANTS, 2, *FEATURE_SHAPE)
LEGACY_TEMPLATE_SHAPE = (2, 2, 2, *FEATURE_SHAPE)
# An arm is recognised by the SHAPE of its profile, not by absolute pixel
# distance. Measured on this gate with night references: in the evening the
# correct position scored 0.31 and 0.43 by distance - past the 0.30 gate, so
# both arms read "unknown" - while correlation scored 0.84 and 0.73 against
# 0.15 and 0.00 for the wrong position. Correlation removes each strip's own
# gain and offset, which is exactly what a change of light does to a picture.
MIN_CORRELATION = 0.55
MIN_CORRELATION_MARGIN = 0.25
# Correlation reads through a change of light, but not through a view that has
# no light left in it. Measured across every labelled capture and the live feed,
# a usable view of this monitor scores 43-47 here; a fifth of that brightness
# scores 9.5, a third 16.6. Twenty sits below every real view by more than half
# and above every adverse one.
MIN_VIEW_CONTRAST = 20.0


def _gray(frame):
    image = np.asarray(frame)
    if (image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32
            or not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all()):
        raise ValueError("A finite RGB camera frame is required")
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(image, (WIDTH, HEIGHT)), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _strip(gray, line):
    first, last = np.asarray(line, np.float32) * (WIDTH, HEIGHT)
    delta = last-first
    perpendicular = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    centers = first + np.linspace(.08, .90, 64)[:, None]*delta
    coordinates = centers[:, None, :] + np.arange(-7, 8)[None, :, None]*perpendicular
    patch = cv2.remap(gray, coordinates[:, :, 0].astype(np.float32),
                      coordinates[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    # Subtract neighbouring background at each point along the arm, then
    # normalize contrast. Uniform brightness changes do not create arm edges.
    feature = patch-patch[:, [0, 1, 13, 14]].mean(axis=1, keepdims=True)
    contrast = float(np.std(feature))
    return feature/max(12., contrast), contrast


def validate_lines(lines):
    """Arm lines for another view: {exit|entrance: {down|up: [[x, y], [x, y]]}} in 0..1."""
    if lines is None:
        return LINES
    if not isinstance(lines, dict) or set(lines) != set(IDS):
        raise ValueError("Barrier lines need exactly an exit and an entrance entry")
    checked = {}
    for identity in IDS:
        positions = lines[identity]
        if not isinstance(positions, dict) or set(positions) != set(POSITIONS):
            raise ValueError(f"The {identity} barrier needs a down and an up line")
        checked[identity] = {}
        for position in POSITIONS:
            line = positions[position]
            try:
                points = [[float(x), float(y)] for x, y in line]
            except (TypeError, ValueError):
                raise ValueError(f"{identity} {position} line must be two [x, y] points") from None
            if (len(points) != 2 or not all(math.isfinite(v) and 0. <= v <= 1. for point in points for v in point)
                    or math.dist(points[0], points[1]) < .03):
                raise ValueError(f"{identity} {position} line must be two distinct points inside the picture")
            checked[identity][position] = points
    return checked


def _features(gray, identity, lines=None):
    lines = LINES if lines is None else lines
    values = [_strip(gray, lines[identity][position]) for position in POSITIONS]
    return np.stack([value[0] for value in values]), min(value[1] for value in values)


def _correlation(stored, live):
    """How alike are these two profiles, ignoring brightness and contrast?

    1.0 is the same shape, 0.0 unrelated. Flat input correlates with nothing, so
    it scores 0 rather than dividing by zero.
    """
    first = stored - stored.mean()
    second = live - live.mean()
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float((first * second).sum() / scale) if scale > 1e-9 else 0.0


def _position_evidence(present, absent):
    """Fraction of the available resemblance gained over the calibrated background.

    UP and DOWN sample different strips, whose backgrounds can have very
    different correlations. Compare each strip with its own absent-arm view
    first. A perfect arm-present match scores one; an unchanged background
    scores zero or less. A low absolute match still cannot prove a position.
    """
    if present < MIN_CORRELATION:
        return -1.0
    return float(np.clip((present-absent)/max(1e-6, 1.-absent), -1., 1.))


def _photometric_indices(reference, mask):
    """The reference pixels a photometric fit samples (None: too little detail). Depends only on reference + mask."""
    selected = np.asarray(mask, bool) & (reference > 12) & (reference < 238)
    indices = np.flatnonzero(selected)
    if len(indices) < 100:
        return None
    return indices[::max(1, len(indices)//4096)]


def _photometric(reference, current, mask, indices=False):
    """Robust gain/offset fit; return unknown for adverse exposure changes.

    ``indices``: the result of _photometric_indices(reference, mask) when the caller already has it (None means
    "too little detail"); omitted, it is computed here. The fit itself is unchanged.
    """
    if indices is False:
        indices = _photometric_indices(reference, mask)
    if indices is None:
        return None, "Insufficient reference detail", None
    original = reference.ravel()[indices].astype(np.float64)
    observed = current.ravel()[indices].astype(np.float64)
    keep = np.ones(len(indices), bool)
    gain, offset = 1., 0.
    for _ in range(4):
        x, y = original[keep], observed[keep]
        variance = float(np.var(x))
        if len(x) < 80 or variance < 20:
            return None, "Reference has too little contrast", None
        gain = float(np.mean((x-x.mean())*(y-y.mean()))/variance)
        offset = float(np.median(y-gain*x))
        residual = np.abs(observed-(original*gain+offset))
        keep = residual <= max(3., float(np.quantile(residual, .75)))
    if not (.55 <= gain <= 1.8 and abs(offset) <= 45):
        return None, "Exposure changed beyond calibration", (gain, offset)
    clipped = float(np.mean((observed <= 2) | (observed >= 253)))
    if clipped > .18:
        return None, "Camera view is clipped or obscured", (gain, offset)
    return (current-offset)/gain, "Exposure matched", (gain, offset)


class BarrierMonitor:
    SHARED_GRAY = True   # update()/road_clear() accept the frame's grey image computed once by the caller

    @staticmethod
    def prepare_gray(frameRGB):
        """The grey working image update() and road_clear() use, or None when the frame is unusable."""
        try:
            return _gray(frameRGB)
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _usable_gray(gray):
        return (isinstance(gray, np.ndarray) and gray.shape == (HEIGHT, WIDTH)
                and gray.dtype == np.float32)

    def _indices_for(self, key, reference, mask_source, mask):
        cached = self._sample_cache.get(key)
        if cached is not None and cached[0] is reference and cached[1] is mask_source:
            return cached[2]
        indices = _photometric_indices(reference, mask)
        self._sample_cache[key] = (reference, mask_source, indices)
        if len(self._sample_cache) > 32:
            self._sample_cache.pop(next(iter(self._sample_cache)))
        return indices

    def __init__(self, storage_path=None, seed_path=None, source_key="", clock=time.monotonic, lines=None):
        self.lines = validate_lines(lines)
        self.storage_path = Path(storage_path) if storage_path else None
        self.source_key = str(source_key)
        self.clock = clock
        self._lock = threading.RLock()
        self._templates = np.zeros(TEMPLATE_SHAPE, np.float32)
        self._counts = np.zeros((2, 2, MAX_VARIANTS), np.int32)
        self._reference = None
        self._road_reference = []
        self._frames = deque(maxlen=9)
        # Work that depends only on fixed inputs, reused while those inputs are the same objects.
        self._sample_cache = {}       # key -> (reference, mask_source, indices)
        self._road_mask_cache = {}    # key -> (route_mask, lines, prepared_mask)
        self._last_at = None
        self._aligned = False
        self._load_reason = "Barrier references are not calibrated"
        self._exposure_note = ""
        self._observations = {identity: self._blank() for identity in IDS}
        for path in (self.storage_path, Path(seed_path) if seed_path else None):
            if path is not None and self._load(path):
                break

    @staticmethod
    def _blank(reason="Waiting for a current barrier image"):
        return {"state": "unknown", "confidence": 0., "reason": reason,
                "candidate": None, "candidate_at": None, "candidate_count": 0,
                "up_at": None, "observed_at": None}

    def _load(self, path):
        try:
            with np.load(path, allow_pickle=False) as saved:
                version = int(saved["version"].item())
                if version not in (1, 2) or str(saved["source_key"].item()) != self.source_key:
                    raise ValueError("Reference belongs to another camera view")
                templates = saved["templates"].astype(np.float32)
                counts = saved["counts"].astype(np.int32)
                reference = saved["reference"].astype(np.float32)
                road = saved["road_reference"].astype(np.float32)
                if version == 1 and templates.shape == LEGACY_TEMPLATE_SHAPE and counts.shape == (2, 2):
                    # An earlier single-variant calibration keeps its value: it
                    # becomes the first lighting variant of the new store.
                    grown = np.zeros(TEMPLATE_SHAPE, np.float32)
                    grown[:, :, 0] = templates
                    grown_counts = np.zeros((2, 2, MAX_VARIANTS), np.int32)
                    grown_counts[:, :, 0] = counts
                    templates, counts = grown, grown_counts
                if road.ndim == 2 and road.shape == (HEIGHT, WIDTH):
                    road = road[None, ...]          # one frozen empty road, pre-variants
                road_ok = (road.size == 0
                           or (road.ndim == 3 and road.shape[1:] == (HEIGHT, WIDTH)
                               and 1 <= road.shape[0] <= MAX_VARIANTS))
                if (templates.shape != TEMPLATE_SHAPE or counts.shape != (2, 2, MAX_VARIANTS)
                        or reference.shape not in ((0,), (HEIGHT, WIDTH)) or not road_ok
                        or not all(np.isfinite(a).all() for a in (templates, reference, road))
                        or np.any(counts < 0) or np.any(counts > 10000)):
                    raise ValueError("Invalid barrier reference data")
                self._templates, self._counts = templates, counts
                self._reference = reference if reference.size else None
                self._road_reference = [row for row in road] if road.size else []
                self._load_reason = "Calibrated references loaded"
                return True
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            self._load_reason = "Reference unavailable or bound to another camera view"
            return False

    def _save(self, path=None):
        path = Path(path) if path else self.storage_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(output, version=np.array(2), source_key=np.array(self.source_key),
                                    templates=self._templates, counts=self._counts,
                                    reference=self._reference if self._reference is not None else np.array([]),
                                    road_reference=(np.stack(self._road_reference) if self._road_reference else np.array([])))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _fresh(self):
        return (self._aligned and self._last_at is not None
                and -.25 <= self.clock()-self._last_at <= MAX_AGE_S)

    def _unknown(self, reason="Waiting for a current barrier image"):
        self._observations = {identity: self._blank(reason) for identity in IDS}

    def update(self, frameRGB, stamp, aligned=True, gray=None):
        with self._lock:
            try:
                stamp = float(stamp)
                if not math.isfinite(stamp) or not -.25 <= self.clock()-stamp <= MAX_AGE_S:
                    raise ValueError("Camera image is stale")
                if not aligned:
                    raise ValueError("Camera alignment is unavailable")
                gray = gray if self._usable_gray(gray) else _gray(frameRGB)
            except (ValueError, TypeError, OverflowError):
                self._aligned = False
                self._frames.clear()
                self._unknown("Fresh aligned camera evidence is required")
                return self.status()
            if self._last_at is not None and stamp <= self._last_at:
                return self.status()
            if self._last_at is None or stamp-self._last_at > MAX_AGE_S or not self._aligned:
                self._unknown()
                self._frames.clear()
            self._last_at, self._aligned = stamp, True
            self._exposure_note = ""
            # A view that is crushed to black, blown out to white or physically
            # obscured carries no arm to read, whatever the templates say. This
            # is checked on its own, so that a merely brighter or darker view -
            # which correlation handles - is never mistaken for a blind one.
            window = gray[40:295, 90:530]
            if window.size:
                blind = float(np.mean((window <= 2) | (window >= 253)))
                if blind > .18 or float(window.std()) < MIN_VIEW_CONTRAST:
                    self._frames.clear()
                    self._unknown("Camera view is clipped or obscured")
                    return self.status()
            normalized = gray
            if self._reference is not None:
                mask = np.zeros((HEIGHT, WIDTH), bool)
                mask[40:295, 90:530] = True
                matched, reason, _fit = _photometric(
                    self._reference, gray, mask,
                    self._indices_for("view", self._reference, "view-window", mask))
                if matched is None:
                    # The arms are recognised by correlation, which already
                    # ignores gain and offset, so a failed exposure fit must not
                    # blind them. It must also not discard the frames, or a view
                    # that has changed brightness could never be recalibrated:
                    # the arms would stay unknown and capture would have nothing
                    # to work from. The absolute empty-road comparison keeps its
                    # own exposure gate, where it is genuinely required.
                    self._exposure_note = reason
                else:
                    normalized = matched
            self._frames.append((stamp, gray))
            for index, identity in enumerate(IDS):
                calibrated = self._counts[index] > 0
                if not np.all(np.any(calibrated, axis=1)):
                    self._observations[identity] = self._blank(self._load_reason)
                    continue
                feature, contrast = _features(normalized, identity, self.lines)
                # Each position gets its best arm-present lighting match on
                # its own strip. The opposite-position calibrations supply
                # the matching strip's background comparison below.
                evidence = np.full(2, -1.0)
                for position in range(2):
                    for variant in range(MAX_VARIANTS):
                        if not calibrated[position, variant]:
                            continue
                        stored = self._templates[index, position, variant][position]
                        evidence[position] = max(evidence[position],
                                                 _correlation(stored, feature[position]))
                # The same strip can contain a post, tree or reflection when
                # the arm is in the opposite position. A changed light must
                # not turn that stationary background into a raised arm merely
                # because the DOWN strip no longer matches its old lighting.
                # Compare like with like: the winning strip must resemble an
                # arm-present calibration more than any arm-absent calibration.
                backgrounds = np.array([
                    max((_correlation(self._templates[index, 1-position, variant][position], feature[position])
                         for variant in range(MAX_VARIANTS) if calibrated[1-position, variant]), default=-1.)
                    for position in range(2)])
                discriminative = np.array([_position_evidence(evidence[position], backgrounds[position])
                                           for position in range(2)])
                winner = max(range(2), key=lambda position: (discriminative[position], evidence[position]))
                best, other = float(evidence[winner]), float(evidence[1-winner])
                background = float(backgrounds[winner])
                support, opposite = float(discriminative[winner]), float(discriminative[1-winner])
                # Both the absolute match and the background-relative evidence
                # must pass the existing confidence floor. The existing margin
                # now compares two meaningful arm-presence scores, rather than
                # unrelated brightness patterns on horizontal/vertical strips.
                valid = (contrast >= 3 and best >= MIN_CORRELATION
                         and support >= MIN_CORRELATION
                         and support-opposite >= MIN_CORRELATION_MARGIN)
                if not valid:
                    reason = ("Arm strips have too little contrast" if contrast < 3 else
                              "No calibrated arm position matches" if max(evidence) < MIN_CORRELATION else
                              "Arm strip matches the opposite-position background" if support < MIN_CORRELATION else
                              "UP and DOWN evidence conflict")
                    observation = self._blank(reason)
                    observation.update(best=best, other=other, background=background,
                                       support=support, opposite_support=opposite, observed_at=stamp)
                    self._observations[identity] = observation
                    continue
                state = POSITIONS[winner]
                observation = self._observations[identity]
                if observation["candidate"] != state:
                    observation = self._blank("Confirming stable arm position")
                    observation.update(candidate=state, candidate_at=stamp)
                    self._observations[identity] = observation
                observation["candidate_count"] += 1
                observation["observed_at"] = stamp
                # How far past the accepting threshold this reading sits, and by
                # how clear a margin over the opposite position.
                observation["best"] = best
                observation["other"] = other
                observation["background"] = background
                observation["support"] = support
                observation["opposite_support"] = opposite
                observation["confidence"] = float(np.clip(
                    min((best-MIN_CORRELATION)/(1.-MIN_CORRELATION),
                        (support-MIN_CORRELATION)/(1.-MIN_CORRELATION),
                        (support-opposite)/max(1e-6, 1.-MIN_CORRELATION_MARGIN)), 0., 1.))
                if observation["candidate_count"] >= 2 and stamp-observation["candidate_at"] >= STABLE_S-1e-9:
                    if observation["state"] != state:
                        observation["up_at"] = stamp if state == "up" else None
                    observation["state"] = state
                    observation["reason"] = "Calibrated arm position confirmed"
            return self.status()

    def status(self):
        with self._lock:
            fresh = self._fresh()
            barriers = []
            for index, identity in enumerate(IDS):
                observation = self._observations[identity]
                state = observation["state"] if fresh else "unknown"
                # Calibrated means every position has at least one variant.
                calibrated = bool(np.all(np.any(self._counts[index] > 0, axis=1)))
                barriers.append({"id": identity, "name": f"{identity.title()} barrier", "state": state,
                                 "confidence": observation["confidence"] if fresh else 0.,
                                 "up_for_s": max(0., self.clock()-observation["up_at"]) if state == "up" else 0.,
                                 "fresh": fresh, "calibrated": calibrated,
                                 "samples": {position: int(self._counts[index, j].sum())
                                             for j, position in enumerate(POSITIONS)},
                                 "lighting_variants": {position: int(np.count_nonzero(self._counts[index, j]))
                                                       for j, position in enumerate(POSITIONS)},
                                 "lines": {key: [point[:] for point in value] for key, value in self.lines[identity].items()},
                                 "active_line": self.lines[identity][state] if state in POSITIONS else None,
                                 "route_group": identity, "observed_at": observation["observed_at"],
                                 "position_since": observation["candidate_at"],
                                 "candidate_position": observation["candidate"] if fresh else None,
                                 "confirmation_s": (round(max(0., observation["observed_at"]-observation["candidate_at"]), 3)
                                                    if fresh and observation["candidate_at"] is not None else 0.),
                                 "match": observation.get("best") if fresh else None,
                                 "opposite_match": observation.get("other") if fresh else None,
                                 "background_match": observation.get("background") if fresh else None,
                                 "discriminative_match": observation.get("support") if fresh else None,
                                 "opposite_discriminative_match": observation.get("opposite_support") if fresh else None,
                                 "reason": observation["reason"] if fresh else "Waiting for fresh aligned barrier evidence"})
            return {"configured": bool(np.all(np.any(self._counts > 0, axis=2))), "barriers": barriers}

    def learning_conditions(self, identity, max_age=None):
        """Light on this arm over the last few seconds: glare share and brightness swing.

        Used only by self-learning. Headlights flood the arm strips with
        near-white pixels and make their brightness jump; an example learned
        then describes the glare, not the arm.
        """
        with self._lock:
            if identity not in IDS:
                return {"glare": 1.0, "light_range": 255.0, "frames": 0}
            limit = MAX_AGE_S if max_age is None else max_age
            frames = [gray for at, gray in self._frames if self.clock() - at <= limit]
            lines = self.lines[identity]
        glare, means = 0.0, []
        for gray in frames:
            values = []
            for position in POSITIONS:
                first, last = np.asarray(lines[position], np.float32) * (WIDTH, HEIGHT)
                delta = last - first
                length = float(np.linalg.norm(delta))
                if length < 1:
                    continue
                perpendicular = np.array([-delta[1], delta[0]]) / length
                centers = first + np.linspace(.08, .90, 64)[:, None] * delta
                points = centers[:, None, :] + np.arange(-10, 11)[None, :, None] * perpendicular
                patch = cv2.remap(gray, points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32),
                                  cv2.INTER_NEAREST)
                values.append(patch)
            if not values:
                continue
            strip = np.concatenate([value.ravel() for value in values])
            glare = max(glare, float(np.mean(strip >= 240)))
            means.append(float(strip.mean()))
        return {"glare": round(glare, 4), "light_range": round(max(means) - min(means), 1) if means else 0.0,
                "frames": len(frames)}

    def reload(self) -> bool:
        """Read the stored references again (after an undo restored an older file)."""
        with self._lock:
            if self.storage_path is None:
                return False
            self._templates = np.zeros(TEMPLATE_SHAPE, np.float32)
            self._counts = np.zeros((2, 2, MAX_VARIANTS), np.int32)
            loaded = self._load(self.storage_path)
            self._observations = {identity: self._blank("References restored; checking new observations")
                                  for identity in IDS}
            return loaded

    def capture(self, identity, position):
        with self._lock:
            if identity not in IDS or position not in POSITIONS:
                return {"ok": False, "reason": "Unknown barrier or position"}
            frames = [(at, gray) for at, gray in self._frames if self.clock()-at <= MAX_AGE_S]
            if not self._fresh() or len(frames) < 5 or frames[-1][0]-frames[0][0] < STABLE_S:
                return {"ok": False, "reason": "Keep the labelled barrier still in five fresh aligned frames"}
            features = [_features(gray, identity, self.lines) for _at, gray in frames]
            template = np.median(np.stack([item[0] for item in features]), axis=0).astype(np.float32)
            spread = max(float(np.mean(np.abs(item[0]-template))) for item in features)
            if min(item[1] for item in features) < 3 or spread > .12:
                return {"ok": False, "reason": "Arm image is moving or has insufficient contrast"}
            i, j = IDS.index(identity), POSITIONS.index(position)
            opposite = self._counts[i, 1-j] > 0
            if np.any(opposite):
                # The new reference must differ from the opposite position in
                # every lighting variant already stored for it, or up and down
                # could be confused under one of those conditions.
                separation = np.mean(np.abs(self._templates[i, 1-j]-template[None, ...]), axis=(2, 3))
                if np.any(separation[opposite] < .18):
                    return {"ok": False, "reason": "Up and down must show distinctly different arm positions"}
            slots = self._counts[i, j]
            if np.any(slots == 0):
                slot = int(np.argmax(slots == 0))
            else:
                # All variants in use: replace the one this view resembles most,
                # which is the same lighting era being recalibrated.
                distance = np.mean(np.abs(self._templates[i, j]-template[None, ...]), axis=(1, 2, 3))
                slot = int(np.argmin(distance))
            old_template, old_count = self._templates[i, j, slot].copy(), self._counts[i, j, slot]
            self._templates[i, j, slot], self._counts[i, j, slot] = template, len(frames)
            try:
                self._save()
            except OSError:
                self._templates[i, j, slot], self._counts[i, j, slot] = old_template, old_count
                return {"ok": False, "reason": "Reference could not be saved"}
            self._observations[identity] = self._blank("Reference saved; checking new observations")
            stored = int(np.count_nonzero(self._counts[i, j]))
            return {"ok": True, "status": self.status(),
                    "reason": f"Saved {identity} {position} reference "
                              f"({stored} lighting variant{'s' if stored != 1 else ''} stored)"}

    def capture_road(self):
        """Freeze what an empty road looks like in the light of this moment.

        The caller asserts the lanes are clear; this only checks that the view
        is fresh, still and aligned. Each capture is an additional reference, so
        a night reference and an evening one can both say "the road is clear".
        """
        with self._lock:
            frames = [(at, gray) for at, gray in self._frames if self.clock()-at <= MAX_AGE_S]
            if not self._fresh() or len(frames) < 5 or frames[-1][0]-frames[0][0] < STABLE_S:
                return {"ok": False, "reason": "Keep the clear road still in five fresh aligned frames"}
            stack = np.stack([gray for _at, gray in frames])
            frozen = np.median(stack, axis=0).astype(np.float32)
            # Pixel differences alone cannot separate a moving arm from this
            # camera's speckle: measured on the real gate, a still minute shows
            # patches up to 187 px and a moving arm only 248 px. The arms
            # themselves are the reliable witness - one in motion reads unknown -
            # so require them settled, and keep a wide pixel guard for something
            # gross like a vehicle crossing the view.
            unsettled = [identity for identity in IDS
                         if self._observations[identity]["state"] not in POSITIONS]
            if unsettled:
                return {"ok": False,
                        "reason": "Wait until both arms are settled and readable "
                                  f"({', '.join(unsettled)} still unknown)"}
            worst = 0
            for layer in (np.abs(stack-frozen) > 20).astype(np.uint8):
                count, _labels, statistics, _centroids = cv2.connectedComponentsWithStats(layer, 8)
                worst = max(worst, int(statistics[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0)
            if worst > 600:
                return {"ok": False,
                        "reason": f"Something is still moving in view ({worst} px patch)"}
            previous = list(self._road_reference)
            references = previous + [frozen]
            if len(references) > MAX_VARIANTS:
                # Replace the closest stored era rather than the oldest: that is
                # the one being re-frozen under today's conditions.
                distance = [float(np.median(np.abs(reference-frozen))) for reference in previous]
                references = previous[:]
                references[int(np.argmin(distance))] = frozen
            self._road_reference = references
            try:
                self._save()
            except OSError:
                self._road_reference = previous
                return {"ok": False, "reason": "Empty-road reference could not be saved"}
            return {"ok": True, "status": self.status(),
                    "reason": f"Saved an empty-road reference ({len(references)} lighting "
                              f"variant{'s' if len(references) != 1 else ''} stored)"}

    def reset(self, identity):
        with self._lock:
            if identity not in IDS:
                return {"ok": False, "reason": "Unknown barrier"}
            index = IDS.index(identity)
            old_counts = self._counts[index].copy()
            self._counts[index] = 0
            try:
                self._save()
            except OSError:
                self._counts[index] = old_counts
                return {"ok": False, "reason": "Reference reset could not be saved"}
            self._observations[identity] = self._blank("Barrier references reset")
            return {"ok": True, "reason": "Barrier references reset", "status": self.status()}

    def road_clear(self, frameRGB, route_mask, gray=None, cache_key=None):
        with self._lock:
            result = {"clear": False, "matched_fraction": 0., "reason": "Frozen empty-road reference unavailable"}
            if not self._road_reference or not self._fresh():
                return result
            try:
                current = gray if self._usable_gray(gray) else _gray(frameRGB)
                cached = self._road_mask_cache.get(cache_key) if cache_key is not None else None
                if cached is not None and cached[0] is route_mask and cached[1] is self.lines:
                    mask = cached[2]
                else:
                    mask = np.asarray(route_mask)
                    if mask.ndim != 2 or not np.isfinite(mask).all():
                        raise ValueError("Invalid route mask")
                    mask = cv2.resize((mask != 0).astype(np.uint8), (WIDTH, HEIGHT),
                                      interpolation=cv2.INTER_NEAREST)
                    for lines in self.lines.values():
                        for line in lines.values():
                            points = np.round(np.asarray(line)*(WIDTH, HEIGHT)).astype(int)
                            cv2.line(mask, tuple(points[0]), tuple(points[1]), 0, 19)
                    mask = mask.astype(bool)
                    if cache_key is not None:
                        self._road_mask_cache[cache_key] = (route_mask, self.lines, mask)
            except (ValueError, TypeError, cv2.error):
                result["reason"] = "Valid current road image and mask required"
                return result
            if np.count_nonzero(mask) < 100:
                result["reason"] = "Route mask has too little visible road"
                return result
            # An empty road at midnight and the same empty road at dusk are not
            # the same picture. Each frozen reference is tried and the best
            # match speaks: one era of light must not make "clear" impossible.
            best = None
            for reference in self._road_reference:
                normalized, reason, fit = _photometric(
                    reference, current, mask,
                    self._indices_for(("road", cache_key, id(reference)), reference, mask, mask)
                    if cache_key is not None else False)
                if normalized is None or fit is None or not (.75 <= fit[0] <= 1.35 and abs(fit[1]) <= 25):
                    reason = reason if normalized is None else "Road illumination is outside the empty reference"
                    if best is None:
                        best = {"clear": False, "matched_fraction": 0., "reason": reason}
                    continue
                changed = (np.abs(normalized-reference) > 20) & mask
                matched = 1.-np.count_nonzero(changed)/np.count_nonzero(mask)
                _count, _labels, statistics, _centroids = cv2.connectedComponentsWithStats(
                    changed.astype(np.uint8), 8)
                largest = int(statistics[1:, cv2.CC_STAT_AREA].max()) if len(statistics) > 1 else 0
                clear = bool(matched >= .97 and largest <= 20)
                candidate = {"clear": clear, "matched_fraction": float(matched),
                             "largest_change_px": largest,
                             "reason": "Road matches the frozen empty reference" if clear
                                       else "Visible change remains on the road"}
                if best is None or candidate["matched_fraction"] > best.get("matched_fraction", 0.):
                    best = candidate
                if clear:
                    break
            return best or result


def create_seed(path, both_down_frames, exit_up_frames, entrance_up_frames, source_key=""):
    """Write a seed only from explicitly labelled RGB calibration frames.

    The caller asserts that both_down_frames show an empty road. Later captures
    and observed AI absence never update this fixed empty-road reference.
    """
    groups = {"down": [_gray(frame) for frame in both_down_frames],
              "exit": [_gray(frame) for frame in exit_up_frames],
              "entrance": [_gray(frame) for frame in entrance_up_frames]}
    if any(len(frames) < 3 for frames in groups.values()):
        raise ValueError("At least three labelled frames are required for each calibration")
    monitor = BarrierMonitor(source_key=source_key)
    monitor._reference = np.median(np.stack(groups["down"]), axis=0).astype(np.float32)
    monitor._road_reference = [monitor._reference.copy()]
    for index, identity in enumerate(IDS):
        for j, position in enumerate(POSITIONS):
            values = [_features(gray, identity) for gray in groups["down" if position == "down" else identity]]
            if min(item[1] for item in values) < 3:
                raise ValueError("Calibration has insufficient arm contrast")
            template = np.median(np.stack([item[0] for item in values]), axis=0)
            if max(float(np.mean(np.abs(item[0]-template))) for item in values) > .12:
                raise ValueError("Labelled barrier was moving during calibration")
            # A seed provides the first lighting variant; later captures under
            # other light fill the remaining slots.
            monitor._templates[index, j, 0] = template
            monitor._counts[index, j, 0] = len(values)
        separation = np.mean(np.abs(monitor._templates[index, 0, 0]-monitor._templates[index, 1, 0]),
                             axis=(1, 2))
        if np.any(separation < .18):
            raise ValueError("Up and down references must differ on both arm strips")
    monitor._save(path)
    return monitor.status()
