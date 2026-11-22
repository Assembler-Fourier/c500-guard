"""A car leaving through the exit is not a top-road arrival (21 Nov 2026). Real PTZ 2 zone shapes, synthetic boxes."""
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_watch import watch  # noqa: E402
from object_tracking import ObjectVisitTracker  # noqa: E402
from zone_editor import validate_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config" / "views" / "ptz2" / "zones.json"


def zones(option=True):
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    picked = []
    for item in config["zones"]:
        if item["name"] not in ("Exit approach", "Top road first sighting"):
            continue
        item = {**item, "rect": tuple(item["rect"])}
        item.pop("ignore_from_routes", None)
        item.pop("toward_direction", None)
        if option and item["name"] == "Top road first sighting":
            item.update(ignore_from_routes=["campus-exit"], toward_direction=[1.0, 0.5])
        picked.append(watch.Zone(**item))
    return picked


def box(gx, gy, w=.08, h=.06, conf=.7):
    return {"label": "car", "confidence": conf,
            "bbox": [(gx - w / 2) * 640, (gy - h) * 360, (gx + w / 2) * 640, gy * 360]}


def path(points, steps_per_leg=6):
    out = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for i in range(steps_per_leg):
            f = i / steps_per_leg
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    out.append(points[-1])
    return out


def first_sighting_events(zone_list, ground_points, conf=.7, step=.25):
    tracker, now, fired = ObjectVisitTracker(), 100., []
    for gx, gy in ground_points:
        now += step
        events, _ = tracker.update([box(gx, gy, conf=conf)], zone_list, 640, 360, now)
        fired += [text for _sev, text, _id in events if text.startswith("Top road first sighting")]
    return fired


# Up the exit lane, past the exit barrier, then left along the top road (how departing cars leave PTZ 2).
DEPARTING = path([(.30, .86), (.40, .70), (.50, .55), (.56, .45), (.55, .36), (.50, .335), (.44, .32)])
# An arrival from the left end of the top road, moving right towards the entrance lane.
ARRIVING = path([(.42, .31), (.50, .33), (.60, .36), (.70, .40), (.78, .43)])


class DepartingExitTests(unittest.TestCase):
    def setUp(self):
        if not CONFIG.is_file():
            self.skipTest("PTZ 2 profile not present")
        names = {z.name for z in zones()}
        if names != {"Exit approach", "Top road first sighting"}:
            self.skipTest("PTZ 2 zones renamed")

    def test_without_the_option_the_departing_car_raises_a_first_sighting(self):
        # The measured false alarm: the same car is announced again as an arrival.
        self.assertTrue(first_sighting_events(zones(option=False), DEPARTING))

    def test_departing_exit_car_is_not_an_arrival(self):
        self.assertEqual(first_sighting_events(zones(), DEPARTING), [])

    def test_real_arrival_from_the_left_still_alarms(self):
        self.assertTrue(first_sighting_events(zones(), ARRIVING))
        self.assertTrue(first_sighting_events(zones(option=False), ARRIVING))

    def test_an_exit_identity_moving_towards_the_entrance_still_alarms(self):
        # If the tracker ever handed an exit car's identity to an arriving car, the arrival must still sound.
        swapped = path([(.30, .86), (.40, .70), (.50, .55), (.56, .45), (.55, .37), (.62, .37), (.70, .40),
                        (.78, .43)])
        self.assertTrue(first_sighting_events(zones(), swapped))

    def test_settings_are_validated(self):
        base = {"name": "Z", "rect": [.1, .1, .2, .2], "classes": ["car"], "trigger": "ai"}
        validate_config({"zones": [{**base, "ignore_from_routes": ["campus-exit"], "toward_direction": [1, .5]}]})
        for bad in ({"ignore_from_routes": "campus-exit"}, {"ignore_from_routes": [""]}, {"toward_direction": [0, 0]},
                    {"toward_direction": [2, 0]}, {"toward_direction": [1]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_config({"zones": [{**base, **bad}]})


if __name__ == "__main__":
    unittest.main()
