"""Forced realignment: get the guard seeing again after the camera has moved.

Why this exists
---------------
The guard refuses to run when it cannot match the live picture against its saved
scene reference, and there is no way out of that from the normal controls:

  * `request_calibration()` in watch.py needs the watcher running, disarmed AND
    receiving fresh frames.
  * But the ON sequence gives the camera 75 seconds to become "healthy", and a
    lost alignment never becomes healthy, so ON rolls itself back and kills the
    bridge and the LAN relay.

So the picture you need in order to recalibrate disappears before you can
recalibrate with it. This tool breaks that deadlock by bringing up ONLY the
camera bridge - no watcher, no health gate, no rollback - capturing the current
view, and writing the reference files directly in the format watch.py reads.

What it does
------------
  --inspect (default)  Bring the camera up, capture the current view, measure it
                       against every saved reference, and report exactly which
                       check fails and by how much. Changes nothing.
  --force              Do the realignment and commit it.

--force picks its own repair, and says which it used:

  remap    The saved scene is still recognisable, the camera has just moved.
           Writes a fresh reacquire matrix so the zones follow the scene to its
           new place on screen. Zones keep covering the same real-world spots.

  reseed   The saved scene is no longer recognisable at all. The old reference
           is useless, so the current view becomes the new reference. Zones stay
           where they are in the frame, which means THEY MUST BE CHECKED - they
           may no longer sit over the right parts of the road.

Safety
------
  * Refuses to run while the guard is armed.
  * Every file it replaces is copied to <name>.before-realign-<stamp> first.
  * The power-off state is captured up front and restored in a finally block,
    so the program is left exactly as OFF or ON as it was found.
  * Nothing is committed until the captured view passes usability checks (not
    black, not frozen, enough landmarks) and, for remap, until the re-mapped
    result passes the watcher's own alignment checks.
  * --inspect writes nothing at all.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import platform_support as ps  # noqa: E402
# The watcher's own zone loader, frame geometry and scale chain, so a reference
# written here is byte-for-byte the kind of picture watch.py will compare
# against. Importing it is safe: its module level only defines constants.
import watch  # noqa: E402

REPO = HERE.parent
STATE_DIR = ps.state_dir()
ZONES = watch.DEFAULT_ZONES
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
BRIDGE = HERE / "bridge.py"
SOURCE = "rtsp://127.0.0.1:8554/c500"
PROC_W, PROC_H = watch.PROC_W, watch.PROC_H
SCALE_CHAIN = watch.SCALE_CHAIN

POWER_OFF = STATE_DIR / "power-off"
SERVICE_PAUSED = STATE_DIR / "service-paused"
ARMED = STATE_DIR / "armed.json"

# A capture must clear these before it is allowed to become a reference.
MIN_CONTRAST = 12.0          # flat grey or a black screen has almost none
MIN_LANDMARKS = 150          # ORB keypoints in the usable area
LIVENESS_S = 12.0            # long enough that the clock in the picture must tick


_transcript = None


def log(message=""):
    """Print, and also keep a transcript: this usually runs in an elevated
    window whose output the caller cannot capture."""
    print(message, flush=True)
    if _transcript is not None:
        try:
            _transcript.write(message + "\n")
            _transcript.flush()
        except OSError:
            pass


def stamp():
    return time.strftime("%Y%m%dT%H%M%S")


# ---------------------------------------------------------------- power state

class PowerState:
    """Capture the OFF markers, clear them, and put them back exactly."""

    def __init__(self):
        self.saved = {}

    def __enter__(self):
        for path in (POWER_OFF, SERVICE_PAUSED):
            self.saved[path] = path.read_bytes() if path.is_file() else None
            if path.is_file():
                path.unlink()
        was_off = any(value is not None for value in self.saved.values())
        log(f"    program was {'OFF' if was_off else 'ON'}; "
            f"markers cleared for the repair and will be restored")
        return self

    def __exit__(self, *_):
        for path, value in self.saved.items():
            try:
                if value is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(value)
            except OSError as error:
                log(f"    WARNING: could not restore {path.name}: {error}")
        log("    power markers restored to how they were found")
        return False


def guard_is_armed() -> bool:
    try:
        return bool(json.loads(ARMED.read_text(encoding="utf-8")).get("armed"))
    except (OSError, ValueError, AttributeError):
        return False


# --------------------------------------------------------------- camera bring-up

def run(command, timeout=180):
    done = subprocess.run([str(c) for c in command], capture_output=True,
                          text=True, timeout=timeout)
    return done.returncode, (done.stdout or "").strip(), (done.stderr or "").strip()


def bridge_up() -> bool:
    code, out, err = run([sys.executable, BRIDGE, "start"])
    if code:
        log(f"    bridge start failed ({code}): {(out + ' ' + err).strip()[:300]}")
        return False
    log("    camera bridge and LAN relay started")
    return True


def bridge_down() -> None:
    run([sys.executable, BRIDGE, "stop"])
    log("    camera bridge stopped")


def capture(seconds=LIVENESS_S, timeout=180):
    """Grab genuine consecutive frames, in the watcher's exact frame geometry.

    Deliberately no fps filter: fps= DUPLICATES frames when the camera runs
    slower than the requested rate, which would make a live feed look frozen and
    hide real motion. Frames are taken at whatever rate the camera sends.
    """
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
               "-i", SOURCE, "-t", str(seconds), "-vf", SCALE_CHAIN,
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run([str(c) for c in command], capture_output=True,
                         timeout=timeout).stdout
    size = PROC_W * PROC_H * 3
    return [np.frombuffer(raw[i * size:(i + 1) * size], np.uint8).reshape(PROC_H, PROC_W, 3)
            for i in range(len(raw) // size)]


def wait_for_frames(limit_s=120.0, want=8):
    began = time.time()
    while time.time() - began < limit_s:
        frames = capture()
        if len(frames) >= want:
            log(f"    captured {len(frames)} frames after {time.time() - began:.0f}s")
            return frames
        time.sleep(3.0)
    return []


def usable(frames):
    """Is this picture fit to become a reference? Returns (ok, reasons, facts)."""
    import cv2
    reasons, facts = [], {}
    if len(frames) < 4:
        return False, ["The camera produced almost no frames"], {"frames": len(frames)}
    stack = np.stack(frames).astype(np.float32)
    steady = np.median(stack, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(steady, cv2.COLOR_RGB2GRAY)

    facts["frames"] = len(frames)
    facts["brightness"] = round(float(gray.mean()), 1)
    facts["contrast"] = round(float(gray.std()), 1)

    # Consecutive frames are NOT a liveness test on this camera. Starved of
    # bitrate on a static night scene, its encoder emits skip frames that
    # reproduce the previous picture exactly, so neighbouring frames are
    # genuinely bit-identical for seconds at a time. What cannot stay frozen is
    # the clock burnt into the NVR picture, so count how many DISTINCT frames
    # appear across the whole capture window instead.
    distinct = len({frame.tobytes() for frame in frames})
    spread = float(np.abs(stack.max(axis=0) - stack.min(axis=0)).mean())
    facts["distinct_frames"] = f"{distinct}/{len(frames)}"
    facts["change_across_window"] = round(spread, 5)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=8, nlevels=10)
    keypoints = orb.detect(gray, None)
    facts["landmarks"] = 0 if keypoints is None else len(keypoints)

    if facts["contrast"] < MIN_CONTRAST:
        reasons.append(f"The picture is nearly flat (contrast {facts['contrast']}, "
                       f"need {MIN_CONTRAST}). Is the monitor on and showing the cameras?")
    if facts["landmarks"] < MIN_LANDMARKS:
        reasons.append(f"Only {facts['landmarks']} landmarks in view (need {MIN_LANDMARKS}). "
                       f"The camera may not be pointed at the monitor.")
    if distinct < 2:
        reasons.append(f"Not one pixel changed anywhere in {LIVENESS_S:.0f} seconds. Even the "
                       f"clock in the picture should tick, so the feed is frozen.")
    return (not reasons), reasons, facts, steady


# ------------------------------------------------------------------- measuring

def load_config():
    try:
        return json.loads(ZONES.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not read {ZONES}: {error}")


def required_mask(zones):
    """The union of every alarm zone, exactly as watch._required_mask builds it."""
    mask = np.zeros((PROC_H, PROC_W), dtype=bool)
    for zone in zones:
        mask |= zone.mask_for(PROC_W, PROC_H)
    return mask.astype(np.uint8) * 255


def stored_references():
    """The main scene reference and every learned lighting reference."""
    import hashlib
    key = hashlib.sha256(SOURCE.encode()).hexdigest()[:16]
    main = STATE_DIR / f"alignment-{key}.npz"
    extra = main.with_suffix(".lighting.npz")
    found = []
    if main.is_file():
        try:
            with np.load(main, allow_pickle=False) as stored:
                found.append(("main reference", stored["reference"]))
        except (OSError, ValueError, KeyError):
            pass
    if extra.is_file():
        try:
            with np.load(extra, allow_pickle=False) as stored:
                for name in sorted(stored.files):
                    found.append((f"lighting {name}", stored[name]))
        except (OSError, ValueError, KeyError):
            pass
    return main, extra, found


def measure(current, references, zone_mask):
    """Compare the current view with every reference, strict then relaxed."""
    import screen_registration as sr

    # Tiers loosen ONE idea at a time, weakest relaxation first, so the repair
    # that gets used is the least permissive one that works - and so the report
    # names the single check that actually stood in the way.
    #
    # supported_hull / supported_span are the spatial spread of the agreeing
    # landmarks. Moving the camera closer means only part of the old view is
    # still on screen, so the agreement is genuinely clustered even when it is
    # overwhelming. Relaxing spread is safe ONLY because every candidate is
    # afterwards re-checked by the watcher's own aligner in verify().
    tiers = [
        ("as the guard requires", sr.ReacquireConfig()),
        ("relaxed landmarks", sr.ReacquireConfig(min_inliers=30, min_inlier_ratio=0.40)),
        ("allowing clustered landmarks", sr.ReacquireConfig(
            min_supported_hull=0.05, min_supported_span=0.25)),
        ("allowing tight clusters", sr.ReacquireConfig(
            min_inliers=30, min_inlier_ratio=0.40,
            min_supported_hull=0.02, min_supported_span=0.12)),
        ("also relaxing the fit", sr.ReacquireConfig(
            min_inliers=30, min_inlier_ratio=0.40,
            min_supported_hull=0.02, min_supported_span=0.12,
            max_median_error_px=4.0)),
        ("ignoring zone visibility", sr.ReacquireConfig(
            min_inliers=25, min_inlier_ratio=0.35,
            min_supported_hull=0.01, min_supported_span=0.08,
            max_median_error_px=5.0,
            min_zone_coverage=0.0, min_view_coverage=0.20)),
    ]
    results = []
    for index, (label, reference) in enumerate(references):
        row = {"label": label, "index": index, "tiers": []}
        for rank, (tier_name, config) in enumerate(tiers):
            matrix, report = sr.estimate_transform(
                current, reference, config=config, required_mask=zone_mask)
            row["tiers"].append({"tier": tier_name, "ok": matrix is not None,
                                 "report": report,
                                 "matrix": None if matrix is None else matrix.tolist()})
            if matrix is not None:
                row["best"] = {"tier": tier_name, "rank": rank, "report": report,
                               "matrix": matrix.tolist(), "reference": reference,
                               "moved_px": round(float(sr.movement_px(matrix, reference.shape[:2])), 1)}
                break
        results.append(row)
    return results


def choose(results):
    """The candidate accepted under the STRICTEST rules wins, not the first one
    found. A reference that needs no relaxation is worth far more than an
    earlier one that only survives a loosened spread check."""
    workable = [row for row in results if row.get("best")]
    if not workable:
        return None
    return min(workable, key=lambda row: (row["best"]["rank"],
                                          -row["best"]["report"].get("inliers", 0)))


def describe(results):
    """Show every tier, not just the strict one: when a repair is refused it
    matters exactly which check refused it and by how much."""
    for row in results:
        first = row["tiers"][0]["report"]
        log(f"\n    {row['label']}: {first.get('matches', 0)} matches, "
            f"{first.get('inliers', 0)} inliers, ratio {first.get('inlier_ratio', 0)}")
        for tier in row["tiers"]:
            report = tier["report"]
            spread = report.get("distribution") or {}
            span = spread.get("supported_span") or []
            detail = (f"inliers {report.get('inliers')}, "
                      f"hull {round(spread.get('supported_hull', 0.0), 4)}, "
                      f"span {[round(v, 3) for v in span]}, "
                      f"median {report.get('median_error_px')}, "
                      f"zones {report.get('zone_coverage')}, view {report.get('view_coverage')}")
            if tier["ok"]:
                log(f"      {'PASS':<5} {tier['tier']:<30} {detail}")
            else:
                log(f"      {'no':<5} {tier['tier']:<30} {report.get('reason') or 'no match'}")
                log(f"            {'':<30} {detail}")


# ------------------------------------------------------------------ committing

def aim(references, zone_mask, alignment_config, main_path, minutes):
    """Live aiming help while the camera is moved back into place.

    The guard tracks landmarks OUTSIDE the alarm zones - the monitor surround,
    the desk, the screen beside it. If the camera is so close that it sees only
    the barrier picture, those landmarks are simply not in view and NO amount of
    threshold loosening can recover the old zones. Moving the camera back until
    the surround is visible again restores everything with no reconfiguration,
    and this loop says how close you are while you do it.
    """
    import screen_registration as sr

    log("\n  AIMING MODE")
    log("  Move the camera slowly back until it sees the whole monitor again,")
    log("  including a little of the bezel and desk around the picture.")
    log("  The score below rises as the view matches the saved scene.")
    log("  It locks by itself the moment it fits. Ctrl-C to stop.\n")

    strict = sr.ReacquireConfig()
    best_ever = 0.0
    deadline = time.time() + minutes * 60.0
    while time.time() < deadline:
        frames = capture(seconds=2.5, timeout=90)
        if len(frames) < 2:
            log("    ... waiting for the picture")
            time.sleep(1.0)
            continue
        live = np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)

        best = None
        for label, reference in references:
            matrix, report = sr.estimate_transform(
                live, reference, config=strict, required_mask=zone_mask)
            score = report.get("inliers", 0)
            if best is None or score > best[2].get("inliers", 0):
                best = (label, reference, report, matrix)
        label, reference, report, matrix = best
        inliers = report.get("inliers", 0)
        matches = report.get("matches", 0)

        if matrix is not None:
            log(f"    FIT  {label}: {matches} matches, {inliers} inliers - checking it holds ...")
            good, detail = verify_remap(reference, matrix, zone_mask, alignment_config)
            if good:
                moved = round(float(sr.movement_px(matrix, reference.shape[:2])), 1)
                log(f"    {detail}")
                commit_remap(main_path, matrix, report, moved)
                log(f"\n  LOCKED against the {label} ({moved} px). The alarm zones are kept.")
                log("  Turn the guard on with the BarrierWatcher app.")
                return True
            log(f"    not yet: {detail}")
        else:
            # Report the WEAKEST requirement, because that is the one the camera
            # position has to fix. Landmark spread is usually it: a camera too
            # close sees only the picture, and the surround it needs is off view.
            spread = report.get("distribution") or {}
            hull = float(spread.get("supported_hull", 0.0))
            span = spread.get("supported_span") or [0.0, 0.0]
            checks = {
                "landmarks": inliers / strict.min_inliers,
                "agreement": report.get("inlier_ratio", 0.0) / strict.min_inlier_ratio,
                "spread across": hull / strict.min_supported_hull,
                "spread sideways": (span[0] if span else 0.0) / strict.min_supported_span,
                "spread up/down": (span[1] if len(span) > 1 else 0.0) / strict.min_supported_span,
            }
            weakest = min(checks, key=checks.__getitem__)
            score = max(0.0, min(1.0, checks[weakest]))
            filled = int(round(score * 30))
            trend = "warmer" if score > best_ever + 0.01 else (
                "colder" if score < best_ever - 0.01 else "same")
            log(f"    |{'#' * filled}{'.' * (30 - filled)}| {score * 100:3.0f}%  "
                f"{matches:>3} matches {inliers:>3} inliers   "
                f"held back by: {weakest}   ({trend})")
            best_ever = max(best_ever, score)
    log("\n  Time is up; the alignment never fitted. Nothing was changed.")
    return False


def back_up(path):
    if path.is_file():
        target = path.with_name(f"{path.name}.before-realign-{stamp()}")
        shutil.copy2(path, target)
        log(f"    backed up {path.name} -> {target.name}")
        return target
    return None


def commit_reseed(main_path, extra_path, steady):
    """The saved scene is unrecognisable: make the current view the reference."""
    from zone_editor import _atomic_write
    back_up(main_path)
    buffer = BytesIO()
    np.savez_compressed(buffer, reference=steady)
    _atomic_write(main_path, buffer.getvalue())
    log(f"    wrote a new scene reference from the current view")

    # Everything below described the OLD framing and must not survive.
    for path in (main_path.with_suffix(".reacquire.json"),
                 main_path.with_suffix(".pose.json"),
                 extra_path):
        if path.is_file():
            back_up(path)
            path.unlink()
            log(f"    removed {path.name} (it described the old framing)")


def commit_remap(main_path, matrix, report, moved_px):
    """The scene is recognisable: point the zones at its new place on screen."""
    from zone_editor import _atomic_write
    path = main_path.with_suffix(".reacquire.json")
    back_up(path)
    _atomic_write(path, json.dumps({
        "matrix": np.asarray(matrix, dtype=np.float64).round(8).tolist(),
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report": {**{k: v for k, v in report.items() if isinstance(v, (int, float, str))},
                   "moved_px": moved_px, "forced": 1,
                   "at": time.strftime("%H:%M:%S")}}).encode())
    log(f"    wrote a forced re-map ({moved_px} px) so the zones follow the scene")
    # A pose from the old framing would fight the new mapping.
    pose = main_path.with_suffix(".pose.json")
    if pose.is_file():
        back_up(pose)
        pose.unlink()
        log(f"    removed {pose.name} (it described the old framing)")


def build_aligner(reference, zone_mask, config):
    """The aligner exactly as watch._build_aligner constructs it."""
    import scene_alignment as sa
    mask = sa.build_feature_mask(
        reference.shape,
        include_rects=config.get("include_rects", [[.12, .10, .79, .82]]),
        exclude_rects=config.get("exclude_rects", []),
        border_px=int(config.get("border_px", 8)),
        include_polygon=config.get("scene_polygon"))
    mask[zone_mask.astype(bool)] = 0
    return sa.SceneAligner(reference, mask, required_mask=zone_mask)


def fresh_frame():
    frames = capture()
    if len(frames) < 4:
        return None
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def verify_reseed(main_path, zone_mask, config):
    """Read the new reference back as watch.py would, and align a fresh frame."""
    try:
        with np.load(main_path, allow_pickle=False) as stored:
            reference = stored["reference"]
    except (OSError, ValueError, KeyError) as error:
        return False, f"the new reference could not be read back: {error}"
    if reference.shape != (PROC_H, PROC_W, 3):
        return False, f"the new reference has the wrong shape {reference.shape}"
    live = fresh_frame()
    if live is None:
        return False, "no frames available to verify with"
    aligner = build_aligner(reference, zone_mask, config)
    if not aligner.reference_ready:
        return False, "the new reference does not hold enough stable landmarks"
    aligned, info = aligner.update(live)
    if aligned is None:
        return False, f"a fresh frame did not align: {info.get('reason') or info}"
    return True, (f"a fresh frame aligned with {info.get('inliers')} inliers, "
                  f"median error {info.get('median_error_px')} px")


def verify_remap(reference, matrix, zone_mask, config):
    """Re-run the watcher's own acceptance test on the re-mapped picture.

    This is the same two-step check _try_reacquire applies: warp a genuinely
    fresh frame through the candidate matrix, then require it to satisfy the
    full strict alignment rules at the reference's own pose. A matrix fitted
    from clustered landmarks cannot survive this unless it is actually right.
    """
    import screen_registration as sr
    live = fresh_frame()
    if live is None:
        return False, "no frames available to verify with"
    warped = sr.warp(live, np.asarray(matrix, dtype=np.float64), reference.shape[:2])
    aligner = build_aligner(reference, zone_mask, config)
    if not aligner.reference_ready:
        return False, "the matched reference does not hold enough stable landmarks"
    aligned, info = aligner.update(warped, prior=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if aligned is None:
        return False, (f"the re-mapped picture failed the watcher's own checks: "
                       f"{info.get('reason') or info}")
    return True, (f"a fresh re-mapped frame passed the full alignment checks with "
                  f"{info.get('inliers')} inliers, median error "
                  f"{info.get('median_error_px')} px")


# ------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true",
                      help="measure and report only; change nothing (default)")
    mode.add_argument("--force", action="store_true",
                      help="perform the realignment and commit it")
    mode.add_argument("--aim", action="store_true",
                      help="live aiming help: keeps measuring while you move the "
                           "camera, and locks the alignment in the moment it fits")
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="how long --aim keeps watching (default 20)")
    parser.add_argument("--reseed", action="store_true",
                        help="with --force, always take the current view as the new "
                             "reference even if a re-map would work")
    parser.add_argument("--images", default=str(REPO / "docs"),
                        help="where to write the before/after pictures")
    args = parser.parse_args()
    doing = "force" if args.force else ("aim" if args.aim else "inspect")

    global _transcript
    try:
        _transcript = (STATE_DIR / "realign.log").open("w", encoding="utf-8")
    except OSError:
        _transcript = None

    if guard_is_armed():
        log("The guard is ARMED. Turn it OFF first, then run this again.")
        return 2

    images = Path(args.images)
    images.mkdir(parents=True, exist_ok=True)

    log(f"Forced realignment - {doing}")
    log(f"  state directory : {STATE_DIR}")
    main_path, extra_path, references = stored_references()
    log(f"  saved references: {len(references)}")

    config = load_config()
    zones = watch.load_zones(ZONES)
    zone_mask = required_mask(zones)
    log(f"  alarm zones     : {len(zones)}")

    with PowerState():
        try:
            log("\n  bringing the camera up (bridge only - no watcher, no health gate)")
            if not bridge_up():
                return 1
            if doing == "aim":
                return 0 if aim(references, zone_mask, config.get("alignment", {}),
                                main_path, args.minutes) else 1

            frames = wait_for_frames()
            if not frames:
                log("    no frames at all. The camera or the relay is not delivering.")
                return 1

            ok, reasons, facts, steady = usable(frames)
            log(f"\n  the current view: {facts['frames']} frames, "
                f"brightness {facts['brightness']}, contrast {facts['contrast']}, "
                f"{facts['landmarks']} landmarks")
            log(f"    distinct frames {facts['distinct_frames']}, "
                f"change across the window {facts['change_across_window']}")
            from PIL import Image
            Image.fromarray(steady).save(images / "realign-current-view.png")
            log(f"    saved {images / 'realign-current-view.png'}")
            if references:
                Image.fromarray(references[0][1]).save(images / "realign-saved-reference.png")
                log(f"    saved {images / 'realign-saved-reference.png'}")

            if not ok:
                log("\n  THIS PICTURE CANNOT BE USED AS A REFERENCE:")
                for reason in reasons:
                    log(f"    - {reason}")
                log("\n  Fix what the camera is looking at, then run this again.")
                return 1

            results = measure(steady, references, zone_mask) if references else []
            if results:
                describe(results)
            else:
                log("\n    there are no saved references at all; a reseed is the only option")

            recoverable = choose(results)
            alignment_config = config.get("alignment", {})

            if doing == "inspect":
                log("\n  Inspection only - nothing was changed.")
                if recoverable and not args.reseed:
                    best = recoverable["best"]
                    log(f"  A re-map WOULD work against the {recoverable['label']}: "
                        f"{best['moved_px']} px moved, accepted {best['tier']}, "
                        f"zone coverage {best['report'].get('zone_coverage')}.")
                    log("  Run again with --force to apply it. The alarm zones would be kept.")
                else:
                    log("  No saved reference can be recovered; --force would reseed from "
                        "the current view, and the alarm zones would then need checking.")
                return 0

            log("")
            if recoverable and not args.reseed:
                best = recoverable["best"]
                log(f"  repairing by RE-MAP against the {recoverable['label']} "
                    f"({best['moved_px']} px moved, accepted {best['tier']})")
                good, detail = verify_remap(best["reference"], best["matrix"],
                                            zone_mask, alignment_config)
                log(f"  pre-commit check: {'PASSED' if good else 'FAILED'} - {detail}")
                if not good:
                    log("  Nothing was changed. The re-map did not survive the watcher's own "
                        "checks, so committing it would have left the zones pointing at the "
                        "wrong places.")
                    return 1
                commit_remap(main_path, best["matrix"], best["report"], best["moved_px"])
                outcome = "remap"
            else:
                log("  repairing by RESEED: the saved scene is not recognisable, so the "
                    "current view becomes the new reference")
                commit_reseed(main_path, extra_path, steady)
                outcome = "reseed"

            if outcome == "remap":
                good, detail = verify_remap(recoverable["best"]["reference"],
                                            recoverable["best"]["matrix"],
                                            zone_mask, alignment_config)
            else:
                good, detail = verify_reseed(main_path, zone_mask, alignment_config)
            log(f"\n  verification: {'PASSED' if good else 'FAILED'} - {detail}")
            if not good:
                log("  The repair did not verify. The backups next to each file end in "
                    "'.before-realign-<time>' if you need to put something back.")
                return 1

            log(f"\n  Done ({outcome}).")
            if outcome == "reseed":
                log("  IMPORTANT: the camera is looking at a different scene than before, so")
                log("  the alarm zones are almost certainly in the wrong places now. Check")
                log("  them in the zone editor before relying on the alarm.")
            log("  Turn the guard on with the BarrierWatcher app.")
            return 0
        finally:
            bridge_down()


if __name__ == "__main__":
    raise SystemExit(main())
