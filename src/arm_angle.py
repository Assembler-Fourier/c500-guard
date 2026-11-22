"""Follow each barrier arm at any angle, including while it moves (15 Nov 2026, user request).

A barrier arm is a rigid bar turning about its pivot. Seen by the camera, its tip moves along the image of a circle,
which (for an object this small and far away) is the ellipse
    tip(theta) = pivot + cos(theta) * (down_tip - pivot) + sin(theta) * (up_tip - pivot),   theta 0..90 degrees.
For every candidate angle we measure how much the line from the pivot looks like the arm: a thin bar brighter than
both sides, with red reflective stripes the sides do not have, allowing a pixel or two of misalignment. The best
angle, kept steady over time, gives the arm's live position for the on-screen line and tells whether it is rising
or lowering. Measured on the 14 Nov videos: the exit arm rises 0->90 degrees in about 1 s and lowers in about 4.5 s;
a passing ambulance did not move the reading. The strict up/down reading (barrier_monitor) remains the authority
for "up"/"down" and for the barrier learning; this adds the positions in between.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

ANGLES = np.linspace(0., 90., 19)          # 5-degree steps
SAMPLES = 32                                # points along the arm
SIDE_PX = 4.0                               # how far either side the background is measured
SHIFTS = np.array([-2., -1., 0., 1., 2.])   # tolerated sideways misalignment, pixels


def geometry_from_lines(lines: dict, width: int = 640, height: int = 360) -> dict:
    """Pivot and the two tips (pixels) from the calibrated down/up lines of one arm (the shared end is the pivot)."""
    down = [np.array(p, float) * (width, height) for p in lines["down"]]
    up = [np.array(p, float) * (width, height) for p in lines["up"]]
    best = min(((i, j) for i in range(2) for j in range(2)), key=lambda ij: np.linalg.norm(down[ij[0]] - up[ij[1]]))
    pivot = (down[best[0]] + up[best[1]]) / 2
    return {"pivot": pivot, "down_tip": down[1 - best[0]], "up_tip": up[1 - best[1]]}


def geometry_from_config(saved: dict, width: int = 640, height: int = 360) -> dict:
    """A saved display geometry (normalised pivot/down_tip/up_tip) in pixels."""
    return {key: np.array(saved[key], float) * (width, height) for key in ("pivot", "down_tip", "up_tip")}


def segment(geometry: dict, theta_deg: float, start=.12, end=.97):
    """Points along the arm at angle theta (0 = down, 90 = up), skipping the pivot housing."""
    t = math.radians(theta_deg)
    pivot = geometry["pivot"]
    tip = pivot + math.cos(t) * (geometry["down_tip"] - pivot) + math.sin(t) * (geometry["up_tip"] - pivot)
    fractions = np.linspace(start, end, SAMPLES)[:, None]
    return pivot + fractions * (tip - pivot), tip


def _channels(frame_rgb):
    frame = np.asarray(frame_rgb).astype(np.float32)
    if frame.ndim == 2:
        return frame, np.zeros_like(frame)
    luma = frame.mean(axis=2)
    redness = frame[:, :, 0] - (frame[:, :, 1] + frame[:, :, 2]) / 2      # the arm's red reflective stripes
    return luma, redness


def _gather(channel, points):
    h, w = channel.shape
    xs = np.clip(np.rint(points[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(points[..., 1]).astype(int), 0, h - 1)
    return channel[ys, xs]


def _scores(luma, redness, geometry, angles) -> np.ndarray:
    """Vectorised: every angle and sideways shift at once (about a millisecond per arm)."""
    pivot = geometry["pivot"]
    radians = np.radians(np.asarray(angles, float))[:, None]
    tips = pivot + np.cos(radians) * (geometry["down_tip"] - pivot) + np.sin(radians) * (geometry["up_tip"] - pivot)
    fractions = np.linspace(.12, .97, SAMPLES)[None, :, None]
    points = pivot + fractions * (tips[:, None, :] - pivot)                         # (A, S, 2)
    direction = tips - pivot
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)       # (A, 2)
    shifted = points[:, None] + SHIFTS[None, :, None, None] * normal[:, None, None, :]   # (A, K, S, 2)
    side = normal[:, None, None, :] * SIDE_PX
    centre = _gather(luma, shifted)
    left, right = _gather(luma, shifted + side), _gather(luma, shifted - side)
    # Median and 80th percentile along the arm by partial sorting (much cheaper than np.median/percentile).
    bar = np.partition(np.minimum(centre - left, centre - right), SAMPLES // 2, axis=2)[..., SAMPLES // 2]
    red_c = _gather(redness, shifted)
    red_s = np.maximum(_gather(redness, shifted + side), _gather(redness, shifted - side))
    k80 = int(SAMPLES * .8)
    stripes = np.partition(red_c - red_s, k80, axis=2)[..., k80]
    return (bar + .5 * np.maximum(stripes, 0.)).max(axis=1)


def arm_scores(frame_rgb: np.ndarray, geometry: dict, angles=ANGLES) -> np.ndarray:
    """How arm-like the line is at every candidate angle."""
    luma, redness = _channels(frame_rgb)
    return _scores(luma, redness, geometry, angles)


def refine_geometry(geometry: dict, down_frame=None, up_frame=None, reach: int = 6) -> dict:
    """Nudge the pivot and the two tips (a few pixels) so the lines sit exactly on the arm in pictures where its
    position is known. Used once, at calibration; the result is saved."""
    pictures = [(0., _channels(down_frame)) if down_frame is not None else None,
                (90., _channels(up_frame)) if up_frame is not None else None]
    pictures = [p for p in pictures if p]

    def fit(g):
        return sum(float(_scores(luma, red, g, [theta])[0]) for theta, (luma, red) in pictures)

    best = {k: np.array(v, float) for k, v in geometry.items()}
    best_fit = fit(best)
    for _round in range(3):
        for key in ("pivot", "down_tip", "up_tip"):
            for dx in range(-reach, reach + 1, 2):
                for dy in range(-reach, reach + 1, 2):
                    trial = {k: v.copy() for k, v in best.items()}
                    trial[key] = best[key] + (dx, dy)
                    score = fit(trial)
                    if score > best_fit + .5:
                        best, best_fit = trial, score
        reach = max(1, reach // 2)
    return best


def display_barrier_row(row: dict, reading: dict | None, geometry: dict | None, *,
                        reading_fresh: bool = False, width: int = 640, height: int = 360) -> dict:
    """Resolve presentation only; never change recognition or movement evidence.

    Recognition strips deliberately sample only part of an arm and may extend
    into its background. They are not display geometry. A transient angle
    failure must not switch a settled arm back to those different endpoints.
    Unknown remains unknown; only a fresh measured movement may supply its
    provisional line. The caller still owns all alarm/acknowledgement rules.
    """
    result = {**row, 'active_line': None, 'live_line': None,
              'live_angle': None, 'live_motion': None}
    reading = reading if isinstance(reading, dict) else {}

    def valid_line(line):
        if not isinstance(line, (list, tuple)) or len(line) != 2:
            return False
        try:
            values = np.asarray(line, dtype=float)
            return (values.shape == (2, 2) and np.isfinite(values).all()
                    and np.all((values >= 0) & (values <= 1))
                    and np.linalg.norm(values[1]-values[0]) > .001)
        except (TypeError, ValueError, OverflowError):
            return False

    fresh = row.get('fresh') is True and row.get('calibrated') is True
    known = fresh and row.get('state') in ('up', 'down')
    conflict = reading_fresh and reading.get('position_conflict') is True
    if conflict:
        if row.get('state') in ('up', 'down'):
            # Preserve the raw classifier claim for diagnostics even when it
            # lacks fresh calibration proof; the presented state stays Unknown.
            result['classifier_state'] = row['state']
        result.update(state='unknown', confidence=0., candidate_position=None,
                      reason='Arm angle disagrees with calibrated position')
        return result
    if known:
        theta = 0. if row['state'] == 'down' else 90.
        line = None
        if geometry is not None:
            try:
                scale = np.asarray((width, height), dtype=float)
                points = np.asarray([geometry['down_tip' if theta == 0 else 'up_tip'],
                                     geometry['pivot']], dtype=float)
                if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError('Invalid display dimensions')
                if points.shape != (2, 2) or not np.isfinite(points).all():
                    raise ValueError('Invalid display geometry')
                candidate = np.round(points / scale, 4).tolist()
                if valid_line(candidate):
                    line = candidate
            except (KeyError, TypeError, ValueError, OverflowError):
                pass  # Invalid configured geometry never borrows sampling strips.
        elif valid_line(row.get('active_line')):
            # Legacy callers without display geometry retain their endpoint.
            line = [list(point) for point in row['active_line']]
        result.update(active_line=line, live_line=line, live_angle=theta if line else None)
        return result
    result.update(state='unknown', confidence=0., candidate_position=None)
    if (fresh and reading_fresh and reading.get('ok') is True
            and reading.get('motion') in ('rising', 'lowering')
            and valid_line(reading.get('line'))):
        shown = reading.get('shown', reading.get('theta'))
        if (not isinstance(shown, bool) and isinstance(shown, (int, float))
                and math.isfinite(shown) and 0 <= shown <= 90):
            result.update(live_line=[list(point) for point in reading['line']],
                          live_angle=shown, live_motion=reading['motion'])
    return result


class ArmAngleTracker:
    """Live angle per arm, whether the reading can be trusted, and whether the arm is rising or lowering."""

    RISE_FROM, RISE_TO = 10., 25.          # degrees: left the down position and is clearly lifting
    LOWER_FROM, LOWER_TO = 80., 65.
    MAX_OBSERVATION_GAP_S = .75           # normally sampled every .15 s; unseen motion is not a trajectory

    def __init__(self, geometry_by_arm: dict, width: int = 640, height: int = 360, min_score: float = 6.,
                 history_s: float = 6.):
        self.geometry = geometry_by_arm
        self.size = (width, height)
        self.min_score = min_score
        self.history_s = history_s
        self.reset()

    def reset(self) -> None:
        """Forget continuity when the view or its arm readings cannot be trusted."""
        self._history = {arm: [] for arm in self.geometry}          # (stamp, theta) of trusted readings
        self._motion_started = {arm: None for arm in self.geometry}
        self._shown = {arm: None for arm in self.geometry}       # smoothed angle for the screen
        self.latest: dict = {}
        self._last_stamp = None

    def update(self, frame_rgb: np.ndarray, stamp: float) -> dict:
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            self.reset()
            return {}
        if self._last_stamp is not None and not 0 < stamp - self._last_stamp <= self.MAX_OBSERVATION_GAP_S:
            self.reset()
        self._last_stamp = stamp
        luma, redness = _channels(frame_rgb)
        out = {}
        for arm, geometry in self.geometry.items():
            scores = _scores(luma, redness, geometry, ANGLES)
            k = int(np.argmax(scores))
            best = float(scores[k])
            others = np.delete(scores, range(max(0, k - 2), min(len(scores), k + 3)))
            runner = float(others.max()) if others.size else 0.
            ok = best >= self.min_score and best - runner >= .25 * best
            history = [h for h in self._history[arm] if stamp - h[0] <= self.history_s]
            theta = float(ANGLES[k])
            if ok:
                history.append((stamp, theta))
            else:
                history = []                   # never renew an earlier movement with an unreadable arm
                self._shown[arm] = None
            self._history[arm] = history
            motion, started = self._motion(history)
            shown = self._smooth(arm, history, theta if ok else None)
            pivot = geometry["pivot"]
            tip = segment(geometry, shown if shown is not None else theta)[1]
            line = [[round(float(np.clip(p[0] / self.size[0], 0, 1)), 4), round(float(np.clip(p[1] / self.size[1], 0, 1)), 4)]
                    for p in (tip, pivot)]
            out[arm] = {"theta": theta if ok else None, "fraction_up": round(theta / 90., 3) if ok else None,
                        "shown": shown, "score": round(best, 1), "ok": bool(ok),
                        "line": line if ok and shown is not None else None,
                        "motion": motion, "motion_started": started, "at": stamp}
        self.latest = out
        return out

    def _smooth(self, arm, history, theta):
        """The angle shown on screen: follows real movement at once (small steps), but a jump of more than
        40 degrees (a car passing the arm) only once three readings in a row agree with it."""
        shown = self._shown.get(arm)
        if theta is not None:
            if shown is None or abs(theta - shown) <= 40:
                shown = theta
            else:
                recent = [h[1] for h in history[-3:]]
                if len(recent) == 3 and all(abs(r - theta) <= 15 for r in recent):
                    shown = theta
            self._shown[arm] = shown
        return shown

    MID_LOW, MID_HIGH = 12., 78.             # a real movement passes through these angles
    RISE_WITHIN_S, LOWER_WITHIN_S = 3., 8.   # measured: up in about 1 s, down in about 4.5 s

    def _motion(self, history):
        """('rising' | 'lowering' | None, when it started). A real movement: the arm was seen at its starting position
        in two consecutive readings, then at least two readings in between (a real lift was read 15, 20, 25, 35,
        75 degrees), then past the threshold in the last two, moving one way all along, quickly enough. A vehicle
        passing along the arm can make it look briefly down or up, but jumps straight across without the angles in
        between, so it never counts."""
        if len(history) < 6:
            return None, None
        thetas = [h[1] for h in history]
        for kind, ready, at_start, step_ok, within in (
                ("rising", min(thetas[-2:]) >= self.RISE_TO, lambda v: v <= self.RISE_FROM, lambda d: d >= -5,
                 self.RISE_WITHIN_S),
                ("lowering", max(thetas[-2:]) <= self.LOWER_TO, lambda v: v >= self.LOWER_FROM, lambda d: d <= 5,
                 self.LOWER_WITHIN_S)):
            if not ready:
                continue
            for i in range(len(thetas) - 3, 0, -1):
                if at_start(thetas[i]) and at_start(thetas[i - 1]):
                    path = thetas[i:]
                    middle = [v for v in path if self.MID_LOW < v < self.MID_HIGH]
                    if (len(middle) >= 2 and all(step_ok(d) for d in np.diff(path))
                            and history[-1][0] - history[i][0] <= within):
                        return kind, history[i][0]
                    break
        return None, None


class ArmMotionRecorder:
    """Keeps clean pictures of every arm movement (the barrier area, with the angle of each frame), so the arm
    model can be checked and improved from real movements. Last ``keep`` movements are kept."""

    ROI = (.28, 0., .86, .36)                       # legacy ANPR fallback, normalised

    def __init__(self, folder, keep: int = 40, before_s: float = 1.5, after_s: float = 5., wall=time.time,
                 geometry=None, width: int = 640, height: int = 360):
        self.folder = Path(folder)
        self.keep, self.before_s, self.after_s = keep, before_s, after_s
        self.wall = wall
        self.roi = self.roi_from_geometry(geometry, width, height)
        self._buffer = []                               # (stamp, crop, angles)
        self._active = None                             # {"name", "until", "frames"}
        self.saved = 0

    @classmethod
    def roi_from_geometry(cls, geometry, width=640, height=360, margin_px=12.) -> tuple:
        """Bounds of the arms' entire sweep in the clean aligned view, with space around the bars.

        Geometry uses the tracker's pixel coordinates. Intermediate angles can extend beyond both endpoint
        tips, so include each coordinate's analytic extremum rather than just the up/down bounding box.
        Missing geometry retains the legacy crop; malformed geometry keeps the full frame for diagnosis.
        """
        if not geometry:
            return cls.ROI
        try:
            size = np.asarray((width, height), dtype=float)
            if not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("Invalid frame size")
            points = []
            for arm in geometry.values():
                pivot, down, up = (np.asarray(arm[key], dtype=float)
                                   for key in ("pivot", "down_tip", "up_tip"))
                if any(p.shape != (2,) or not np.isfinite(p).all() for p in (pivot, down, up)):
                    raise ValueError("Invalid arm geometry")
                a, b = down - pivot, up - pivot
                angles = [0., math.pi / 2]
                for axis in range(2):
                    extreme = math.atan2(b[axis], a[axis]) % math.pi
                    if 0 < extreme < math.pi / 2:
                        angles.append(extreme)
                points.append(pivot)
                points.extend(pivot + math.cos(theta) * a + math.sin(theta) * b for theta in angles)
            points = np.asarray(points)
            low = np.clip((points.min(axis=0) - margin_px) / size, 0., 1.)
            high = np.clip((points.max(axis=0) + margin_px) / size, 0., 1.)
            if np.any(high <= low):
                raise ValueError("Arm geometry is outside the frame")
            return tuple(float(v) for v in (*low, *high))
        except (AttributeError, KeyError, TypeError, ValueError):
            return (0., 0., 1., 1.)

    def add(self, frame_rgb, stamp, angles: dict) -> None:
        try:
            h, w = frame_rgb.shape[:2]
            x1, y1 = math.floor(self.roi[0] * w), math.floor(self.roi[1] * h)
            x2, y2 = math.ceil(self.roi[2] * w), math.ceil(self.roi[3] * h)
            # The watcher works on a float copy of the frame; pictures are saved as ordinary 8-bit images.
            crop = np.clip(np.asarray(frame_rgb[y1:y2, x1:x2]), 0, 255).astype(np.uint8)
            summary = {arm: {"theta": a.get("theta"), "score": a.get("score"), "motion": a.get("motion")}
                       for arm, a in angles.items()}
            self._buffer = [b for b in self._buffer if stamp - b[0] <= self.before_s] + [(stamp, crop, summary)]
            moving = [arm for arm, a in angles.items() if a.get("motion")]
            if self._active is None and moving:
                motion = angles[moving[0]]["motion"]
                name = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.wall())) + f"-{moving[0]}-{motion}"
                self._active = {"name": name, "until": stamp + self.after_s, "frames": list(self._buffer)}
            elif self._active is not None:
                self._active["frames"].append((stamp, crop, summary))
                if stamp >= self._active["until"]:
                    self._write(self._active)
                    self._active = None
        except Exception:  # noqa: BLE001 - a diagnostic recording never disturbs detection
            self._active = None

    def _write(self, record) -> None:
        from PIL import Image
        target = self.folder / record["name"]
        target.mkdir(parents=True, exist_ok=True)
        for empty in [p for p in self.folder.iterdir() if p.is_dir() and p != target and not any(p.iterdir())]:
            shutil.rmtree(empty, ignore_errors=True)          # left by a failed write
        first = record["frames"][0][0]
        index = []
        for i, (stamp, crop, summary) in enumerate(record["frames"]):
            Image.fromarray(crop).save(target / f"frame_{i:03d}.jpg", quality=90)
            index.append({"frame": i, "t": round(stamp - first, 3), "arms": summary})
        (target / "angles.json").write_text(json.dumps({"roi": self.roi, "frames": index}), encoding="utf-8")
        self.saved += 1
        folders = sorted(p for p in self.folder.iterdir() if p.is_dir())
        for old in folders[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
