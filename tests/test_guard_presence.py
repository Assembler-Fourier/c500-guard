"""Presence evidence, gentle alarm pacing and exact speaker restoration.

Head pose cannot force a loud onset. A current confirmed visit may build up
gradually; missing evidence stops sound while the visual incident remains.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard_presence import (DeskBaseline, GuardPresence, LoudnessPlan, PresenceConfig,  # noqa: E402
                            PresenceTracker, Reading, face_geometry, looking_at_screen)
import watch  # noqa: E402


def landmarks(*, turn=0.0, drop=1.0, gap=40.0):
    """Five face landmarks: eyes level, nose turned by `turn`, mouth `drop` below."""
    left_eye, right_eye = (100.0, 100.0), (100.0 + gap, 100.0)
    middle = (left_eye[0] + right_eye[0]) / 2
    nose = (middle + turn * gap, 100.0 + drop * gap * .55)
    mouth_y = 100.0 + drop * gap
    return [left_eye, right_eye, nose, (middle - gap * .35, mouth_y), (middle + gap * .35, mouth_y)]


def feed(tracker, *, start=0.0, seconds=1.0, step=0.25, face=True, eyes=None, frontal=True,
         drift=0.0):
    """Drive the tracker with steady readings and return the final clock.

    ``drift`` moves the face box between looks. ``eyes=False`` in these pure
    tracker tests represents an explicit closure measurement, not hidden eyes.
    """
    now, moved = start, 0.0
    end = start + seconds
    while now <= end + 1e-9:
        moved += drift
        box = (100 + moved, 80 + moved, 120, 120) if face else None
        tracker.observe(Reading(at=now, face=face, eyes=eyes, frontal=frontal, box=box))
        now += step
    return end


class PresenceStateTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PresenceTracker(PresenceConfig())

    def test_face_with_open_eyes_is_watching(self):
        feed(self.tracker, seconds=2.0, face=True, eyes=True)
        self.assertEqual(self.tracker.state, "watching")
        self.assertFalse(self.tracker.unattended)

    def test_face_without_eyes_briefly_is_present_not_asleep(self):
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        feed(self.tracker, start=end + .25, seconds=6.0, face=True, eyes=False)
        self.assertEqual(self.tracker.state, "present")

    def test_eyes_closed_long_enough_is_drowsy_then_asleep(self):
        config = PresenceConfig()
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        end = feed(self.tracker, start=end + .25, seconds=config.drowsy_after_s + 1, face=True, eyes=False)
        self.assertEqual(self.tracker.state, "drowsy")
        feed(self.tracker, start=end + .25, seconds=config.asleep_after_s + 1, face=True, eyes=False)
        self.assertEqual(self.tracker.state, "asleep")
        self.assertTrue(self.tracker.unattended)

    def test_no_face_becomes_away(self):
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        feed(self.tracker, start=end + .25, seconds=self.tracker.config.away_after_s + 2, face=False)
        self.assertEqual(self.tracker.state, "away")
        self.assertTrue(self.tracker.unattended)

    def test_profile_face_never_invents_sleep(self):
        """A head turned sideways hides the eyes; that is not a closed eye."""
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        feed(self.tracker, start=end + .25, seconds=120.0, face=True, eyes=None, frontal=False)
        self.assertEqual(self.tracker.state, "present")
        self.assertNotIn(self.tracker.state, ("drowsy", "asleep"))

    def test_head_down_but_moving_is_working_not_sleeping(self):
        """Reading a phone hides the eyes exactly like sleep. Movement tells them apart."""
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        feed(self.tracker, start=end + .25, seconds=90.0, face=True, eyes=False, drift=3.0)
        self.assertEqual(self.tracker.state, "present")
        self.assertGreater(self.tracker.head_movement(), self.tracker.config.still_below)

    def test_legacy_closure_state_requires_explicit_measurements_and_stillness(self):
        end = feed(self.tracker, seconds=1.0, face=True, eyes=True)
        feed(self.tracker, start=end + .25, seconds=90.0, face=True, eyes=False, drift=0.0)
        self.assertEqual(self.tracker.state, "asleep")
        self.assertEqual(self.tracker.head_movement(), 0.0)

    def test_movement_is_measured_per_face_width_not_per_pixel(self):
        """Sitting closer must not look like restlessness."""
        near, far = PresenceTracker(PresenceConfig()), PresenceTracker(PresenceConfig())
        for index in range(8):
            at = index * .25
            near.observe(Reading(at=at, face=True, eyes=False, box=(index * 4, 0, 200, 200)))
            far.observe(Reading(at=at, face=True, eyes=False, box=(index * 2, 0, 100, 100)))
        self.assertAlmostEqual(near.head_movement(), far.head_movement(), places=6)

    def test_camera_failure_is_unknown_and_unattended(self):
        feed(self.tracker, seconds=2.0, face=True, eyes=True)
        self.tracker.camera_failed(10.0, "camera covered or dark")
        self.assertEqual(self.tracker.state, "unknown")
        self.assertTrue(self.tracker.unattended)
        self.assertFalse(self.tracker.camera_ok)

    def test_waking_up_returns_to_watching(self):
        end = feed(self.tracker, seconds=80.0, face=True, eyes=False)
        self.assertIn(self.tracker.state, ("drowsy", "asleep"))
        feed(self.tracker, start=end + .25, seconds=1.0, face=True, eyes=True)
        self.assertEqual(self.tracker.state, "watching")


class LoudnessPlanTests(unittest.TestCase):
    def test_watching_is_a_light_buzz_and_asleep_is_immediate_full(self):
        tracker = PresenceTracker(PresenceConfig())
        feed(tracker, seconds=2.0, face=True, eyes=True)
        light = tracker.plan()
        self.assertEqual(light.state, "watching")
        self.assertLessEqual(light.peak, 0.35)
        self.assertEqual(light.level_at(60.0), light.start)

        tracker.state = "asleep"
        loud = tracker.plan()
        self.assertEqual(loud.start, 1.0)
        self.assertEqual(loud.level_at(0.0), 1.0)

    def test_absent_guard_ramps_from_low_to_full(self):
        tracker = PresenceTracker(PresenceConfig())
        tracker.state = "away"
        plan = tracker.plan()
        self.assertLess(plan.start, 0.5)
        self.assertEqual(plan.peak, 1.0)
        self.assertGreater(plan.level_at(plan.ramp_s / 2), plan.start)
        self.assertEqual(plan.level_at(plan.ramp_s * 2), 1.0)

    def test_unknown_camera_is_never_quieter_than_away(self):
        config = PresenceConfig()
        away = PresenceTracker(config)
        away.state = "away"
        unknown = PresenceTracker(config)
        unknown.state = "unknown"
        for seconds in (0.0, 5.0, 30.0):
            self.assertGreaterEqual(unknown.plan().level_at(seconds), away.plan().level_at(seconds))

    def test_ignored_light_buzz_climbs_anyway(self):
        tracker = PresenceTracker(PresenceConfig())
        feed(tracker, seconds=2.0, face=True, eyes=True)
        quiet = tracker.plan(unacknowledged_s=0.0)
        climbing = tracker.plan(unacknowledged_s=tracker.config.escalate_unacked_s + 1)
        self.assertGreater(climbing.peak, quiet.peak)
        self.assertEqual(climbing.peak, 1.0)

    def test_no_state_can_fall_below_the_floor(self):
        config = PresenceConfig.from_dict({"floor": .4, "levels": {"watching": [0.01, 0.01, 0]}})
        tracker = PresenceTracker(config)
        tracker.state = "watching"
        self.assertGreaterEqual(tracker.plan().start, .4)

    def test_level_at_interpolates_and_clamps(self):
        plan = LoudnessPlan("away", .25, 1.0, 10.0, "test")
        self.assertAlmostEqual(plan.level_at(0), .25)
        self.assertAlmostEqual(plan.level_at(5), .625)
        self.assertAlmostEqual(plan.level_at(10), 1.0)
        self.assertAlmostEqual(plan.level_at(999), 1.0)


class PresenceConfigTests(unittest.TestCase):
    def test_nonsense_values_are_clamped_not_obeyed(self):
        config = PresenceConfig.from_dict({
            "away_after_s": -50, "asleep_after_s": 99999, "eyes_closed_ratio": 5,
            "device_index": -3, "checks_per_second": 0})
        self.assertGreaterEqual(config.away_after_s, 5.0)
        self.assertLessEqual(config.asleep_after_s, 3600.0)
        self.assertLessEqual(config.eyes_closed_ratio, 1.0)
        self.assertEqual(config.device_index, 0)
        self.assertGreaterEqual(config.checks_per_second, 0.5)

    def test_missing_or_broken_config_keeps_defaults(self):
        self.assertEqual(PresenceConfig.from_dict(None).levels, PresenceConfig().levels)
        self.assertEqual(PresenceConfig.from_dict("nonsense").levels, PresenceConfig().levels)
        self.assertEqual(PresenceConfig.from_dict({"levels": {"asleep": "loud"}}).levels["asleep"],
                         PresenceConfig().levels["asleep"])

    def test_asleep_can_never_be_configured_quieter_than_watching(self):
        config = PresenceConfig.from_dict({"levels": {"asleep": [0.05, 0.05, 0], "watching": [0.4, 0.4, 0]}})
        self.assertGreaterEqual(config.levels["asleep"][0], config.levels["watching"][0])


class FaceGeometryTests(unittest.TestCase):
    def test_geometry_is_scale_free(self):
        """Sitting closer must not change the measured pose."""
        near = face_geometry(landmarks(turn=.1, drop=1.0, gap=80))
        far = face_geometry(landmarks(turn=.1, drop=1.0, gap=20))
        self.assertAlmostEqual(near["sideways"], far["sideways"], places=6)
        self.assertAlmostEqual(near["pitch"], far["pitch"], places=6)
        self.assertGreater(near["eye_gap"], far["eye_gap"])

    def test_a_square_face_is_looking_at_the_screen(self):
        self.assertTrue(looking_at_screen(face_geometry(landmarks()), .95, DeskBaseline()))

    def test_a_turned_head_is_not(self):
        self.assertFalse(looking_at_screen(face_geometry(landmarks(turn=.6)), .95, DeskBaseline()))

    def test_a_dropped_head_is_not(self):
        self.assertFalse(looking_at_screen(face_geometry(landmarks(drop=.4)), .95, DeskBaseline()))

    def test_a_weak_detection_is_never_treated_as_attention(self):
        self.assertFalse(looking_at_screen(face_geometry(landmarks()), .2, DeskBaseline()))

    def test_a_learned_baseline_accepts_this_guard_s_own_posture(self):
        """Somebody who always sits slightly turned is still 'looking at the screen'."""
        habitual = face_geometry(landmarks(turn=.50))
        self.assertFalse(looking_at_screen(habitual, .95, DeskBaseline()))
        learned = DeskBaseline(eye_gap=habitual["eye_gap"], sideways=habitual["sideways"],
                               pitch=habitual["pitch"], samples=40, at="2026-09-12T19:00:00")
        self.assertTrue(learned.learned)
        self.assertTrue(looking_at_screen(habitual, .95, learned))
        self.assertFalse(looking_at_screen(face_geometry(landmarks(turn=.9)), .95, learned))

    def test_a_thin_calibration_is_not_trusted(self):
        self.assertFalse(DeskBaseline(eye_gap=40, sideways=.1, pitch=1.0, samples=4).learned)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "desk-baseline.json"
        self.presence = GuardPresence(PresenceConfig(), baseline_path=self.path)

    def test_calibration_learns_medians_and_saves_them(self):
        self.presence.start_calibration(10.0)
        self.assertTrue(self.presence.calibrating)
        for turn in [.1, .12, .11, .13, .1, .12, .11, .1, .12, .11, .13, .12, .9]:
            self.presence._collect_calibration(face_geometry(landmarks(turn=turn)), .95)
        self.presence._finish_calibration()
        self.assertTrue(self.presence.baseline.learned)
        # A glance away during calibration must not loosen the test.
        self.assertLessEqual(self.presence.baseline.sideways, .12)
        self.assertGreaterEqual(self.presence.baseline.sideways, .10)
        self.assertTrue(self.path.is_file())
        self.assertTrue(GuardPresence(PresenceConfig(), baseline_path=self.path).baseline.learned)

    def test_too_few_samples_leaves_the_old_baseline_alone(self):
        self.presence.baseline = DeskBaseline(eye_gap=40, sideways=.1, pitch=1.0, samples=50, at="x")
        self.presence.start_calibration(10.0)
        self.presence._collect_calibration(face_geometry(landmarks()), .95)
        self.presence._finish_calibration()
        self.assertEqual(self.presence.baseline.samples, 50)

    def test_a_weak_face_is_never_calibrated_from(self):
        self.presence.start_calibration(10.0)
        for _ in range(20):
            self.presence._collect_calibration(face_geometry(landmarks()), .2)
        self.presence._finish_calibration()
        self.assertFalse(self.presence.baseline.learned)

    def test_calibration_is_refused_when_the_desk_camera_is_off(self):
        off = GuardPresence(PresenceConfig.from_dict({"enabled": False}), baseline_path=self.path)
        self.assertFalse(off.start_calibration())


class FakeSpeaker:
    """Stands in for the machine's speaker, and records every change."""

    def __init__(self, volume=.62, muted=False, readable=True):
        self.volume, self.muted, self.readable = volume, muted, readable
        self.writes = []

    def read(self):
        if not self.readable:
            return {"available": False, "detail": "no speaker"}
        return {"available": True, "volume": self.volume, "muted": self.muted, "device": "test"}

    def apply(self, volume=None, muted=None):
        if volume is not None:
            self.volume = volume
        if muted is not None:
            self.muted = muted
        self.writes.append((volume, muted))
        return {"applied": True}


class FakePresence:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, unacknowledged_s=0.0):
        return self._plan


class AlarmLoudnessTests(unittest.TestCase):
    def build(self, plan, speaker=None, kind="main"):
        import tempfile
        speaker = speaker or FakeSpeaker()
        clock = {"now": 0.0}
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.restore_path = Path(folder.name) / "speaker-restore.json"
        profiles = []
        loudness = watch.AlarmLoudness(FakePresence(plan), apply_fn=speaker.apply,
                                       read_fn=speaker.read, clock=lambda: clock["now"],
                                       restore_path=self.restore_path, profile_fn=profiles.append)
        loudness.test_profiles = profiles
        return loudness, speaker, clock

    def test_a_watcher_killed_mid_alarm_gives_the_volume_back_at_the_next_start(self):
        """Full OFF, a crash or a power cut must not leave the speakers at full."""
        plan = LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep")
        loudness, speaker, _ = self.build(plan)
        loudness.begin("main")
        self.assertAlmostEqual(speaker.volume, .30)
        self.assertTrue(self.restore_path.is_file(), "the guard's level must be written down")

        # The process dies here: no end(), no restore. A new watcher starts.
        successor = watch.AlarmLoudness(FakePresence(plan), apply_fn=speaker.apply,
                                        read_fn=speaker.read, restore_path=self.restore_path)
        recovered = successor.recover_previous()
        self.assertAlmostEqual(speaker.volume, .62)
        self.assertEqual(round(recovered["volume"], 2), .62)
        self.assertFalse(self.restore_path.exists())

    def test_the_level_an_alarm_raised_never_becomes_the_level_to_restore(self):
        """The cascade that leaves a machine at 100% for ever."""
        plan = LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep")
        loudness, speaker, _ = self.build(plan)
        loudness.begin("main")                     # 62% remembered, speaker now 30%
        # Killed. The next alarm sees the temporary level, but the note wins.
        successor = watch.AlarmLoudness(FakePresence(plan), apply_fn=speaker.apply,
                                        read_fn=speaker.read, restore_path=self.restore_path)
        successor.begin("main")
        successor.end()
        self.assertAlmostEqual(speaker.volume, .62)

    def test_a_damaged_note_is_ignored_rather_than_obeyed(self):
        loudness, speaker, _ = self.build(LoudnessPlan("away", .3, 1.0, 5.0, "x"))
        self.restore_path.write_text('{"volume": 42}', encoding="utf-8")
        self.assertIsNone(loudness.recover_previous())
        self.restore_path.write_text("not json at all", encoding="utf-8")
        self.assertIsNone(loudness.recover_previous())
        self.assertEqual(speaker.writes, [])

    def test_nothing_to_recover_changes_nothing(self):
        loudness, speaker, _ = self.build(LoudnessPlan("away", .3, 1.0, 5.0, "x"))
        self.assertIsNone(loudness.recover_previous())
        self.assertEqual(speaker.writes, [])

    def test_alarm_sets_the_start_level_and_restores_afterwards(self):
        plan = LoudnessPlan("away", .3, 1.0, 20.0, "nobody at the desk")
        loudness, speaker, clock = self.build(plan)
        loudness.begin("main")
        self.assertAlmostEqual(speaker.volume, .30)
        self.assertFalse(speaker.muted)
        loudness.end()
        self.assertAlmostEqual(speaker.volume, .62)

    def test_confirmed_current_visit_climbs_gradually_and_is_capped(self):
        plan = LoudnessPlan("away", .3, 1.0, 20.0, "nobody at the desk")
        loudness, speaker, clock = self.build(plan)
        loudness.begin("main", source="ai", confirmed=True)
        clock["now"] = 10.0
        loudness.evidence(fresh=True, present=True, confirmed=True, source="ai")
        loudness.tick()
        self.assertAlmostEqual(speaker.volume, .95)
        clock["now"] = 45.0
        loudness.evidence(fresh=True, present=True, confirmed=True, source="ai")
        loudness.tick()
        self.assertAlmostEqual(speaker.volume, .95)
        self.assertTrue(loudness.status()['audible'])

    def test_a_muted_speaker_is_unmuted_for_the_alarm(self):
        speaker = FakeSpeaker(volume=.1, muted=True)
        loudness, speaker, _ = self.build(LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep"), speaker)
        loudness.begin("main")
        self.assertFalse(speaker.muted)
        self.assertAlmostEqual(speaker.volume, .30)
        loudness.end()
        self.assertTrue(speaker.muted)          # put back exactly as found
        self.assertAlmostEqual(speaker.volume, .1)

    def test_weak_evidence_is_capped_so_a_maybe_never_blasts(self):
        loudness, speaker, _ = self.build(LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep"))
        loudness.begin("early")
        self.assertLessEqual(speaker.volume, watch.AlarmLoudness.CAPS["early"])

    def test_unsupported_sleep_state_never_forces_full_volume_onset(self):
        loudness, speaker, _ = self.build(LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep"))
        loudness.begin("main", source="ai", confirmed=True)
        self.assertAlmostEqual(speaker.volume, .30)
        self.assertEqual(loudness.test_profiles[-1], 'soft')

    def test_disappeared_evidence_stops_sound_even_when_operator_is_away(self):
        plan = LoudnessPlan("away", .3, 1.0, 10.0, "nobody")
        loudness, speaker, clock = self.build(plan)
        loudness.begin("main", source="ai", confirmed=True)
        clock["now"] = 45.0
        loudness.evidence(fresh=True, present=True, confirmed=True, source="ai")
        loudness.tick()
        self.assertAlmostEqual(speaker.volume, .95)
        loudness.evidence(fresh=True, present=False, confirmed=False)
        loudness.tick()
        self.assertFalse(loudness.status()['audible'])
        self.assertEqual(loudness.test_profiles[-1], 'silent')
        self.assertAlmostEqual(speaker.volume, .62)

    def test_unreadable_speaker_changes_nothing_at_all(self):
        speaker = FakeSpeaker(readable=False)
        loudness, speaker, _ = self.build(LoudnessPlan("asleep", 1.0, 1.0, 0.0, "asleep"), speaker)
        loudness.begin("main")
        loudness.tick()
        loudness.end()
        self.assertEqual(speaker.writes, [], "never guess a volume we could not read")

    def test_disabled_controller_touches_nothing(self):
        speaker = FakeSpeaker()
        loudness = watch.AlarmLoudness(FakePresence(LoudnessPlan("asleep", 1.0, 1.0, 0.0, "x")),
                                       enabled=False, apply_fn=speaker.apply, read_fn=speaker.read)
        loudness.begin("main")
        loudness.tick()
        loudness.end()
        self.assertEqual(speaker.writes, [])

    def test_a_repeat_is_a_light_buzz_while_the_guard_is_at_the_desk(self):
        watching = LoudnessPlan("watching", .30, .30, 0.0, "you are looking at the screen")
        loudness, speaker, clock = self.build(watching)
        loudness.presence = FakePresence(LoudnessPlan("present", .45, 1.0, 12.0, "at the desk"))
        loudness.begin("main", gentle=True)
        self.assertLessEqual(speaker.volume, .31, "a repeat they already know about stays light")

    def test_absent_or_unknown_operator_still_gets_a_gentle_first_pulse(self):
        """A new visit can wake gradually without a face-state volume jump."""
        for state, start in (("away", .30), ("asleep", 1.0), ("unknown", .30)):
            with self.subTest(state=state):
                plan = LoudnessPlan(state, start, 1.0, 0.0 if state == "asleep" else 18.0, state)
                loudness, speaker, clock = self.build(plan)
                loudness.begin("main", gentle=True, source="ai", confirmed=True)
                self.assertAlmostEqual(speaker.volume, .30)
                self.assertEqual(loudness.test_profiles[-1], 'soft')

    def test_current_unconfirmed_notice_continues_at_moderate_cap_then_stops_on_departure(self):
        present = LoudnessPlan("present", .45, 1.0, 12.0, "at the desk")
        loudness, speaker, clock = self.build(present)
        loudness.begin("main", gentle=True, source="light")
        self.assertLessEqual(speaker.volume, .31)
        clock["now"] = 31.0
        loudness.tick()
        self.assertTrue(loudness.status()['audible'])
        self.assertAlmostEqual(speaker.volume, .75)
        self.assertEqual(loudness.test_profiles[-1], 'notice')
        loudness.evidence(fresh=True, present=False, confirmed=False)
        loudness.tick()
        self.assertFalse(loudness.status()['audible'])
        self.assertEqual(loudness.test_profiles[-1], 'silent')
        self.assertAlmostEqual(speaker.volume, .62, msg='idle volume is restored only after silence')

    def test_legacy_urgent_flag_does_not_bypass_gentle_evidence_policy(self):
        loudness, speaker, _ = self.build(LoudnessPlan("present", .45, 1.0, 12.0, "at the desk"))
        loudness.begin("main", urgent=True, gentle=True)
        self.assertAlmostEqual(speaker.volume, .30)
        self.assertEqual(loudness.test_profiles[-1], 'soft')

    def test_end_is_safe_to_call_twice(self):
        loudness, speaker, _ = self.build(LoudnessPlan("away", .3, 1.0, 5.0, "x"))
        loudness.begin("main")
        loudness.end()
        before = len(speaker.writes)
        loudness.end()
        self.assertEqual(len(speaker.writes), before)


class GuardPresenceLifecycleTests(unittest.TestCase):
    def test_disabled_presence_never_starts_a_thread(self):
        presence = GuardPresence(PresenceConfig.from_dict({"enabled": False}))
        presence.start()
        self.assertFalse(presence.status()["running"])
        presence.stop()

    def test_status_is_json_safe_before_any_frame(self):
        import json
        presence = GuardPresence(PresenceConfig.from_dict({"enabled": False}))
        json.dumps(presence.status())
        self.assertEqual(presence.status()["state"], "unknown")
        self.assertTrue(presence.status()["unattended"])


if __name__ == "__main__":
    unittest.main()
