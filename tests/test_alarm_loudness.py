"""A volume tick may not outlive acknowledgement's speaker restoration."""
from pathlib import Path
import sys
import threading
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_watch import watch
from guard_presence import LoudnessPlan


class AlarmLoudnessRaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="alarm-race-", dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.restore_path = Path(temporary.name) / "speaker-restore.json"

    def test_urgent_change_starts_soft_even_if_camera_fails(self):
        from unittest.mock import Mock
        presence = Mock()
        presence.plan.side_effect = AssertionError("urgent alerts do not depend on webcam state")
        writes = []

        def apply(**values):
            writes.append(values)
            return {"applied": True}

        loudness = watch.AlarmLoudness(
            presence, apply_fn=apply,
            read_fn=lambda: {"available": True, "volume": .35, "muted": True},
            restore_path=self.restore_path, clock=lambda: 0.)
        loudness.begin("main", urgent=True)
        self.assertEqual(writes[0], {"volume": .30, "muted": False})
        loudness.tick()
        self.assertTrue(loudness.plan.audible)
        self.assertEqual(loudness.plan.profile, "soft")
        self.assertLessEqual(loudness.applied, .75)
        loudness.end()
        self.assertEqual(writes[-1], {"volume": .35, "muted": True})

    def test_end_waits_for_inflight_tick_and_restores_last(self):
        entered, release, ended = (threading.Event() for _ in range(3))
        clock = {"now": 0.0}
        block_write = {"enabled": False}
        writes = []
        errors = []

        class Presence:
            def plan(self, **kwargs):
                return LoudnessPlan("away", .3, 1.0, 20.0, "nobody at desk")

        def apply(**values):
            if block_write["enabled"] and values.get("muted") is False:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release volume write")
            writes.append(values)
            return {"applied": True}

        presence = Presence()
        loudness = watch.AlarmLoudness(
            presence, clock=lambda: clock["now"], apply_fn=apply,
            read_fn=lambda: {"available": True, "volume": .2, "muted": True},
            restore_path=self.restore_path)
        loudness.begin("main")
        block_write["enabled"] = True
        clock["now"] = 10.0

        def tick():
            try:
                loudness.tick()
            except Exception as exc:
                errors.append(exc)

        def end():
            try:
                loudness.end()
            except Exception as exc:
                errors.append(exc)
            finally:
                ended.set()

        worker, acknowledger = threading.Thread(target=tick), threading.Thread(target=end)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            acknowledger.start()
            self.assertFalse(ended.wait(.05), "restoration must follow the pending write")
        finally:
            release.set()
            worker.join(2)
            if acknowledger.ident is not None:
                acknowledger.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(acknowledger.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(writes[-1], {"volume": .2, "muted": True})
        self.assertFalse(loudness.status()["active"])
        self.assertIsNone(loudness.saved)
        count = len(writes)
        loudness.tick()
        self.assertEqual(len(writes), count)


if __name__ == "__main__":
    unittest.main()
