"""The verified-corner tier, and the rule that evidence grading cannot invert.

Cars come down the bottom corner fast and stop at a barrier the guard lifts by
hand. A verified immediate change there is stronger evidence than a flicker, so
it climbs faster and further - but it must never overtake a recognised vehicle,
and a configuration typo must never be able to silence anything.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import alarm_audio  # noqa: E402
from alarm_audio import MAX_LEVEL, START_LEVEL, URGENT_LEVEL, WEAK_LEVEL, plan  # noqa: E402


class UrgentTierTests(unittest.TestCase):
    def setUp(self):
        alarm_audio.reset_configuration()
        self.addCleanup(alarm_audio.reset_configuration)

    def level(self, seconds, **kwargs):
        return plan(seconds, **kwargs).level

    def test_it_sits_between_a_flicker_and_a_recognised_vehicle(self):
        for seconds in (0, .5, 1, 2, 4, 6, 8, 20, 600):
            image = self.level(seconds, confirmed=False)
            urgent = self.level(seconds, confirmed=False, urgent=True)
            confirmed = self.level(seconds, confirmed=True)
            self.assertGreaterEqual(urgent, image, f"at {seconds}s")
            self.assertLessEqual(urgent, confirmed, f"at {seconds}s")

    def test_it_reaches_a_commanding_level_within_seconds(self):
        self.assertAlmostEqual(self.level(0, confirmed=False, urgent=True), START_LEVEL)
        self.assertGreater(self.level(2, confirmed=False, urgent=True), .60)
        self.assertGreater(self.level(4, confirmed=False, urgent=True), .75)
        self.assertAlmostEqual(self.level(7, confirmed=False, urgent=True), URGENT_LEVEL)
        self.assertAlmostEqual(self.level(600, confirmed=False, urgent=True), URGENT_LEVEL)

    def test_it_never_asks_for_the_confirmed_maximum(self):
        for seconds in (0, 1, 5, 30, 600):
            self.assertLess(self.level(seconds, confirmed=False, urgent=True), MAX_LEVEL)

    def test_it_climbs_without_a_step(self):
        levels = [self.level(t / 20, confirmed=False, urgent=True) for t in range(400)]
        self.assertEqual(levels, sorted(levels), "it never gets quieter while ignored")
        self.assertLess(max(b - a for a, b in zip(levels, levels[1:])), 0.03)

    def test_it_uses_the_strong_pattern_quickly(self):
        self.assertEqual(plan(0.5, confirmed=False, urgent=True).profile, "soft")
        self.assertEqual(plan(2.0, confirmed=False, urgent=True).profile, "notice")
        self.assertEqual(plan(5.0, confirmed=False, urgent=True).profile, "wake")

    def test_an_ordinary_image_change_is_unaffected(self):
        for seconds in (0, 2, 4, 6, 600):
            self.assertLessEqual(self.level(seconds, confirmed=False), WEAK_LEVEL)
        self.assertNotEqual(plan(30, confirmed=False).profile, "wake")

    def test_every_silence_rule_still_wins_over_the_new_tier(self):
        for changes in ({"acknowledged": True}, {"enabled": False}, {"event_present": False},
                        {"evidence_fresh": False}, {"quiet_remaining_s": 3}):
            result = plan(5, confirmed=False, urgent=True, **changes)
            self.assertFalse(result.audible, changes)
            self.assertEqual(result.level, 0)


class EscalationSettingTests(unittest.TestCase):
    def setUp(self):
        alarm_audio.reset_configuration()
        self.addCleanup(alarm_audio.reset_configuration)

    def test_tuning_is_applied(self):
        alarm_audio.configure({"urgent": {"level": .8, "rise_s": 4.0}})
        self.assertAlmostEqual(alarm_audio.settings()["urgent"]["level"], .8)
        self.assertAlmostEqual(plan(4, confirmed=False, urgent=True).level, .8)

    def test_weaker_evidence_can_never_be_configured_louder(self):
        alarm_audio.configure({"image": {"level": 1.0}, "urgent": {"level": 1.0},
                               "confirmed": {"level": .5}})
        levels = alarm_audio.settings()
        self.assertLessEqual(levels["urgent"]["level"], levels["confirmed"]["level"])
        self.assertLessEqual(levels["image"]["level"], levels["urgent"]["level"])

    def test_absurd_values_are_clamped_not_obeyed(self):
        alarm_audio.configure({"confirmed": {"level": 99, "rise_s": -5, "shape": 500}})
        confirmed = alarm_audio.settings()["confirmed"]
        self.assertLessEqual(confirmed["level"], 1.0)
        self.assertGreaterEqual(confirmed["rise_s"], 1.0)
        self.assertLessEqual(confirmed["shape"], 8.0)
        self.assertTrue(plan(2, confirmed=True).audible)

    def test_nonsense_input_leaves_the_defaults(self):
        for payload in (None, "fast", 5, [], {"confirmed": "loud"}, {"nope": {"level": .1}}):
            alarm_audio.reset_configuration()
            alarm_audio.configure(payload)
            self.assertAlmostEqual(alarm_audio.settings()["confirmed"]["level"], MAX_LEVEL)

    def test_a_reset_restores_the_built_in_tiers(self):
        alarm_audio.configure({"image": {"level": .25}})
        alarm_audio.reset_configuration()
        self.assertAlmostEqual(alarm_audio.settings()["image"]["level"], WEAK_LEVEL)

    def test_the_barrier_reminder_is_quiet_and_slow(self):
        reminder = plan(0, kind="barrier_open")
        self.assertTrue(reminder.audible)
        self.assertLessEqual(reminder.level, 0.2)
        self.assertEqual(reminder.repeat_s, 15.0)
        self.assertLess(reminder.burst_s, 0.4)
        for seconds in (0, 30, 600):
            self.assertEqual(plan(seconds, kind="barrier_open").level, reminder.level,
                             "the barrier reminder never escalates")


if __name__ == "__main__":
    unittest.main()
