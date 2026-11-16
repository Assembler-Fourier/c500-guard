"""Buzzer/mixer/ACK integration with fake speakers and isolated restore files."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from test_watch import watch, watcher
from guard_presence import LoudnessPlan
import alarm_audio


class GentleAlarmIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gentle-integration-", dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.restore_path = Path(temporary.name) / "speaker-restore.json"
        self.now = 100.0
        self.trace = []
        self.mixer = {"available": True, "volume": .91, "muted": True}
        self.apply_result = {"applied": True}
        self.apply_error = None
        self.presence_state = "unknown"
        self.presence = types.SimpleNamespace(plan=lambda **_: LoudnessPlan(
            self.presence_state, 1., 1., 0., "Legacy volume values must be ignored"))
        self.player = types.SimpleNamespace(
            start_loop=lambda path: self.trace.append(("play", path.name)),
            stop=lambda: self.trace.append(("stop",)),
        )
        paths = {key: Path(f"{key}.wav") for key in ("limited", "soft", "notice", "wake", "fault")}
        paths.update(main=paths["soft"], early=paths["soft"])
        with mock.patch.object(watch.ps, "AudioPlayer", return_value=self.player):
            self.buzzer = watch.Buzzer(paths)
        self.loudness = watch.AlarmLoudness(
            self.presence, apply_fn=self.apply, read_fn=lambda: dict(self.mixer),
            clock=lambda: self.now, restore_path=self.restore_path,
            profile_fn=self.profile)

    def apply(self, **values):
        self.trace.append(("volume", values["volume"], values["muted"]))
        if self.apply_error:
            raise self.apply_error
        if self.apply_result.get("applied"):
            self.mixer.update(values)
        return dict(self.apply_result)

    def profile(self, name):
        self.trace.append(("profile", name))
        self.buzzer.set_profile(name)

    def start(self, *, kind="main", source="ai", confirmed=True):
        self.loudness.begin(kind, source=source, confirmed=confirmed)
        self.buzzer.set(kind)

    def advance(self, seconds, *, fresh=True, present=True, confirmed=True):
        self.now += seconds
        self.loudness.evidence(fresh=fresh, present=present, confirmed=confirmed)
        self.loudness.tick()

    def test_first_sample_plays_after_mixer_is_lowered(self):
        self.start()
        volume = next(i for i, item in enumerate(self.trace) if item[0] == "volume")
        sample = next(i for i, item in enumerate(self.trace) if item[0] == "play")
        self.assertLess(volume, sample)
        self.assertEqual(self.trace[volume], ("volume", .30, False))
        self.assertEqual(self.trace[sample], ("play", "soft.wav"))
        remembered = json.loads(self.restore_path.read_text())
        self.assertEqual(remembered["volume"], .91)
        self.assertTrue(remembered["muted"])

    def _assert_policy_endpoint(self, *, tier, duration, ceiling, interval, configured=False):
        alarm_audio.reset_configuration()
        self.addCleanup(alarm_audio.reset_configuration)
        if configured:
            alarm_audio.configure({tier: {"rise_s": duration, "level": ceiling}})
        confirmed, urgent = tier == 'confirmed', tier == 'urgent'
        start = self.now
        self.loudness.begin('main', source='ai' if confirmed else 'motion',
                            confirmed=confirmed, urgent=urgent)
        tick = 1
        while tick * interval < duration:
            self.now = start + tick * interval
            self.loudness.evidence(fresh=True, present=True, confirmed=confirmed)
            self.loudness.tick()
            tick += 1
        self.now = start + duration
        self.loudness.evidence(fresh=True, present=True, confirmed=confirmed)
        self.loudness.tick()
        self.assertEqual(self.loudness.plan.level, ceiling)
        self.assertEqual(self.loudness.applied, ceiling,
                         'The endpoint must not wait for the periodic mixer verification')
        self.assertEqual(self.mixer['volume'], ceiling)

    def test_urgent_endpoint_bypasses_small_mixer_write_deadband(self):
        self._assert_policy_endpoint(tier='urgent', duration=7., ceiling=.88, interval=.25)

    def test_configured_confirmed_endpoint_uses_configured_ceiling(self):
        self._assert_policy_endpoint(tier='confirmed', duration=8., ceiling=.83,
                                     interval=.1, configured=True)

    def test_configured_image_endpoint_uses_configured_ceiling(self):
        self._assert_policy_endpoint(tier='image', duration=6., ceiling=.62,
                                     interval=.1, configured=True)

    def test_watcher_lowering_precedes_actual_alarm_playback(self):
        obj = watcher()
        obj.buzzer, obj.loudness = self.buzzer, self.loudness
        with mock.patch.object(watch, "notify"), mock.patch.object(watch.time, "monotonic", return_value=self.now):
            self.assertTrue(obj.raise_alarm("main", "Arrival", details={"source": "ai"}))
        events = [item for item in self.trace if item[0] in ("volume", "play")]
        self.assertEqual(events[:2], [("volume", .30, False), ("play", "soft.wav")])

    def test_web_ack_stops_before_restore_and_retains_ten_seconds(self):
        obj = watcher()
        obj.buzzer, obj.loudness = self.buzzer, self.loudness
        obj.ack_quiet_s = 10.
        with mock.patch.object(watch, "notify"), mock.patch.object(watch.time, "monotonic", return_value=self.now):
            self.assertTrue(obj.raise_alarm("main", "Arrival", details={"source": "ai"}))
            alarm_id = obj.alarm_id
            self.trace.clear()
            self.assertTrue(obj.acknowledge(alarm_id, via="web"))
            self.assertAlmostEqual(obj._ack_until-self.now, 10.)
        stopped = next(i for i, item in enumerate(self.trace) if item[0] == "stop")
        restore = next(i for i, item in enumerate(self.trace) if item == ("volume", .91, True))
        self.assertLess(stopped, restore)
        self.assertIsNone(obj.alarm)
        self.assertIsNone(self.buzzer._playing)
        self.assertFalse(self.restore_path.exists())

    def test_stale_audio_stops_before_idle_volume_is_restored(self):
        self.start()
        self.advance(40)
        self.assertEqual(self.loudness.plan.profile, "wake")
        self.trace.clear()
        self.advance(.1, fresh=False)
        stop = next(i for i, item in enumerate(self.trace) if item[0] == "stop")
        restore = next(i for i, item in enumerate(self.trace) if item == ("volume", .91, True))
        self.assertLess(stop, restore)
        self.assertIsNone(self.buzzer._playing)
        self.assertFalse(self.loudness.status()["audible"])

    def test_return_after_stale_evidence_begins_soft_again(self):
        self.start()
        self.advance(40)
        self.advance(.1, fresh=False)
        self.trace.clear()
        self.advance(30)
        self.assertEqual(self.loudness.plan.profile, "soft")
        self.assertEqual(self.loudness.applied, .30)
        audible = [item for item in self.trace if item[0] == "play"]
        self.assertEqual(audible, [("play", "soft.wav")])

    def test_late_confirmation_keeps_current_reminder_then_increases(self):
        self.start(source="light", confirmed=False)
        self.advance(60, confirmed=False)
        self.assertTrue(self.loudness.status()["audible"])
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.trace.clear()
        self.advance(1, confirmed=True)
        self.assertEqual(self.loudness.plan.profile, "notice")
        self.assertEqual(self.loudness.applied, .75)
        self.assertEqual(self.buzzer._playing.name, "notice.wav")
        self.assertFalse([item for item in self.trace if item[0] == "play"],
                         "An evidence upgrade must not restart the current pulse")
        self.advance(2)
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.advance(2)
        self.assertGreater(self.loudness.applied, .83)
        self.assertLess(self.loudness.applied, .87)
        self.assertEqual(self.buzzer._playing.name, "wake.wav")
        self.advance(4)
        self.assertEqual(self.loudness.applied, .95)

    def test_continuity_floor_clears_on_stale_or_departed_evidence(self):
        for lost in ({"fresh": False}, {"present": False}):
            with self.subTest(lost=lost):
                self.start(source="light", confirmed=False)
                self.advance(30, confirmed=False)
                self.advance(1)
                self.assertAlmostEqual(self.loudness.applied, .75)
                self.advance(.1, **lost)
                self.assertIsNone(self.buzzer._playing)
                self.advance(10)
                self.assertEqual(self.loudness.applied, .30)
                self.assertEqual(self.buzzer._playing.name, "soft.wav")
                self.loudness.end()

    def test_continuity_floor_does_not_cross_new_alarm_or_ack(self):
        self.start(source="light", confirmed=False)
        self.advance(30, confirmed=False)
        self.advance(1)
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.loudness.end()
        self.start()
        self.assertEqual(self.loudness.applied, .30)
        self.assertEqual(self.buzzer._playing.name, "soft.wav")

    def test_mixer_recovery_discards_confirmation_continuity_floor(self):
        self.start(source="light", confirmed=False)
        self.advance(30, confirmed=False)
        self.advance(1)
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.apply_result = {"applied": False}
        self.advance(10)
        self.assertIsNone(self.buzzer._playing)
        self.apply_result = {"applied": True}
        self.advance(2)
        self.assertEqual(self.loudness.applied, .30)
        self.assertEqual(self.buzzer._playing.name, "soft.wav")

    def test_unknown_face_and_strict_change_never_request_full_volume(self):
        self.loudness.begin("main", urgent=True, source="motion", confirmed=False)
        self.buzzer.set("main")
        for delay in (2, 2, 4, 2, 4, 60):
            self.advance(delay, confirmed=False)
        changes = [item[1] for item in self.trace if item[0] == "volume" and item[2] is False]
        self.assertTrue(changes)
        # The verified corner tier is stronger than a mere image change, and
        # still strictly below a recognised vehicle. It never asks for full.
        from alarm_audio import MAX_LEVEL, URGENT_LEVEL
        self.assertLessEqual(max(changes), URGENT_LEVEL)
        self.assertLess(max(changes), MAX_LEVEL)
        self.assertLess(max(changes), 1.0)

    def test_mixer_refusal_never_selects_louder_waveform(self):
        self.apply_result = {"applied": False, "detail": "Mixer unavailable"}
        self.start()
        self.advance(40)
        self.assertNotIn(("play", "wake.wav"), self.trace)
        self.assertNotEqual(self.loudness.status()["profile"], "wake")
        self.assertIn(self.buzzer._playing.name if self.buzzer._playing else None, ("limited.wav", None))

    def test_mixer_exception_keeps_process_alive_and_sound_soft(self):
        self.apply_error = OSError("Mixer disconnected")
        self.start()
        self.advance(40)
        self.assertNotIn(("play", "wake.wav"), self.trace)
        self.assertNotEqual(self.loudness.status()["profile"], "wake")

    def test_failed_mid_alarm_change_never_keeps_stronger_waveform(self):
        self.start()
        self.advance(8)
        self.assertEqual(self.buzzer._playing.name, "wake.wav")
        self.apply_result = {"applied": False, "detail": "Mixer disconnected"}
        # Once the new fast ramp reaches its cap, unchanged ticks perform no
        # mixer write. Trigger a real requested downgrade after a confidence
        # gap and after the 12-second fallback window has expired.
        self.advance(5, confirmed=False)
        self.trace.clear()
        self.advance(2.1, confirmed=False)
        self.assertIsNone(self.buzzer._playing)
        self.assertFalse(self.loudness.status()["audible"])
        self.assertNotIn(("play", "wake.wav"), self.trace)

    def test_mixer_recovery_restarts_confirmed_ramp_from_soft(self):
        self.apply_result = {"applied": False}
        self.start()
        self.advance(40)
        self.apply_result = {"applied": True}
        self.trace.clear()
        self.advance(2)
        self.assertEqual(self.loudness.applied, .30)
        self.assertEqual(self.buzzer._playing.name, "soft.wav")
        self.advance(4)
        self.assertGreater(self.loudness.applied, .83)
        self.assertLess(self.loudness.applied, .87)

    def test_refused_mixer_status_matches_soft_fallback_at_ten_seconds(self):
        self.apply_result = {"applied": False}
        self.start()
        self.advance(10)
        self.assertEqual(self.loudness.status()["profile"], "limited")
        self.assertEqual(self.buzzer._playing.name, "limited.wav")
        self.assertIsNone(self.loudness.status()["level_pct"])

    def test_invalid_reported_idle_volume_is_never_used_for_restoration(self):
        self.mixer["volume"] = float("nan")
        self.start()
        self.loudness.end()
        self.assertFalse([item for item in self.trace if item[0] == "volume"])
        self.assertFalse(self.restore_path.exists())

    def test_unreadable_mixer_keeps_short_fallback_then_silence(self):
        self.mixer["available"] = False
        self.start()
        self.assertFalse([item for item in self.trace if item[0] == "volume"])
        self.assertEqual(self.buzzer._playing.name, "limited.wav")
        self.advance(40)
        self.assertIsNone(self.buzzer._playing)
        self.assertNotIn(("play", "wake.wav"), self.trace)

    def test_failed_end_restore_retains_recovery_receipt(self):
        self.start()
        self.apply_result = {"applied": False, "detail": "Mixer unavailable"}
        self.loudness.end()
        self.assertIsNone(self.buzzer._playing)
        self.assertTrue(self.restore_path.exists())
        self.apply_result = {"applied": True}
        restored = self.loudness.recover_previous()
        self.assertEqual(restored, {"volume": .91, "muted": True})
        self.assertEqual(self.mixer["volume"], .91)
        self.assertFalse(self.restore_path.exists())

    def test_failed_startup_restore_keeps_recovery_receipt(self):
        self.restore_path.write_text(json.dumps({"volume": .37, "muted": True}))
        self.apply_result = {"applied": False, "detail": "Mixer unavailable"}
        self.loudness.recover_previous()
        self.assertTrue(self.restore_path.exists())
        self.apply_result = {"applied": True}
        restored = self.loudness.recover_previous()
        self.assertEqual(restored, {"volume": .37, "muted": True})

    def test_restore_exception_preserves_receipt_and_silence(self):
        self.start()
        self.apply_error = OSError("Speaker removed")
        self.loudness.end()
        self.assertIsNone(self.buzzer._playing)
        self.assertTrue(self.restore_path.exists())
        self.assertTrue(self.loudness.status()["restore_pending"])
        self.assertIsNone(self.loudness.recover_previous())
        self.assertTrue(self.restore_path.exists())

    def test_constant_profile_does_not_restart_a_pulse_every_tick(self):
        self.start()
        for _ in range(12):
            self.advance(.1)
        starts = [item for item in self.trace if item[0] == "play"]
        self.assertEqual(starts, [("play", "soft.wav")])
        self.advance(.31)
        for _ in range(10):
            self.advance(.1)
        starts = [item for item in self.trace if item[0] == "play"]
        self.assertEqual(starts, [("play", "soft.wav"), ("play", "notice.wav")])

    def test_current_image_signal_keeps_moderate_reminders_past_twelve_seconds(self):
        self.start(source="light", confirmed=False)
        self.advance(30, confirmed=False)
        self.assertTrue(self.loudness.status()["audible"])
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.assertEqual(self.buzzer._playing.name, "notice.wav")
        self.advance(120, confirmed=False)
        self.assertTrue(self.loudness.status()["audible"])
        self.assertAlmostEqual(self.loudness.applied, .75)
        self.advance(.1, fresh=False, confirmed=False)
        self.assertFalse(self.loudness.status()["audible"])
        self.assertIsNone(self.buzzer._playing)

    def test_looking_at_screen_does_not_prevent_escalation_when_unanswered(self):
        self.presence_state = "watching"
        self.start()
        self.advance(2)
        self.assertGreater(self.loudness.applied, .65)
        self.assertLess(self.loudness.applied, .70)
        self.advance(2)
        self.assertGreater(self.loudness.applied, .83)
        self.assertLess(self.loudness.applied, .87)
        self.assertEqual(self.buzzer._playing.name, "wake.wav")
        self.advance(4)
        self.assertEqual(self.loudness.applied, .95)

    def test_presence_transitions_never_reduce_an_ongoing_confirmed_episode(self):
        rank = {"soft": 0, "notice": 1, "wake": 2}
        for initial in ("unknown", "watching", "present", "away", "asleep"):
            with self.subTest(initial=initial):
                self.presence_state = initial
                self.start()
                levels, patterns = [self.loudness.applied], [rank[self.loudness.plan.profile]]
                for index in range(16):
                    self.presence_state = ("watching", "away", "present", "asleep", "unknown")[index % 5]
                    self.advance(.5)
                    levels.append(self.loudness.applied)
                    patterns.append(rank[self.loudness.plan.profile])
                self.assertEqual(levels, sorted(levels))
                self.assertEqual(patterns, sorted(patterns))
                self.assertEqual(self.loudness.applied, .95)
                self.loudness.end()

    def test_brief_confirmation_flicker_cannot_reset_escalation_forever(self):
        self.start()
        levels = []
        for _ in range(40):
            for seconds, confirmed in ((2., True), (.1, False), (.1, True)):
                self.advance(seconds, confirmed=confirmed)
                levels.append(self.loudness.applied)
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(self.loudness.applied, .95)
        self.assertEqual(self.buzzer._playing.name, "wake.wav")

    def test_longer_confidence_gap_caps_sound_and_pauses_earned_progress(self):
        self.start()
        self.advance(3)
        self.advance(.1, confirmed=False)
        self.advance(2.1, confirmed=False)
        paused_progress = self.loudness.status()["confirmed_progress_s"]
        self.assertAlmostEqual(paused_progress, 5.1, places=1)
        self.assertFalse(self.loudness.status()["effective_confirmed"])
        self.assertLessEqual(self.loudness.applied, .75)
        self.advance(30, confirmed=False)
        self.assertEqual(self.loudness.status()["confirmed_progress_s"], paused_progress)
        self.assertEqual(self.loudness.applied, .75)
        self.advance(.1, confirmed=True)
        self.assertGreaterEqual(self.loudness.applied, .75)
        self.assertLess(self.loudness.applied, .95)
        self.advance(3, confirmed=True)
        self.assertEqual(self.loudness.applied, .95)

    def test_stale_or_departed_evidence_stops_even_during_confirmation_hold(self):
        for lost in ({"fresh": False}, {"present": False}):
            with self.subTest(lost=lost):
                self.start()
                self.advance(35)
                self.advance(.1, confirmed=False)
                self.assertTrue(self.loudness.status()["effective_confirmed"])
                self.advance(.1, confirmed=False, **lost)
                self.assertFalse(self.loudness.status()["audible"])
                self.assertIsNone(self.buzzer._playing)
                self.advance(5, confirmed=False)
                self.assertEqual(self.loudness.applied, .30)
                self.assertEqual(self.buzzer._playing.name, "soft.wav")
                self.loudness.end()

    def test_unconfirmed_mixer_recovery_starts_at_thirty_not_seventy_five_percent(self):
        self.apply_result = {"applied": False}
        self.start(source="light", confirmed=False)
        self.advance(40, confirmed=False)
        self.assertIsNone(self.buzzer._playing)
        self.apply_result = {"applied": True}
        self.advance(2, confirmed=False)
        self.assertEqual(self.loudness.applied, .30)
        self.assertEqual(self.buzzer._playing.name, "soft.wav")
        self.advance(15, confirmed=False)
        self.assertEqual(self.loudness.applied, .75)

    def test_unconfirmed_warning_after_stale_starts_a_new_audio_episode(self):
        self.start(source="light", confirmed=False)
        self.advance(40, confirmed=False)
        self.assertEqual(self.loudness.applied, .75)
        self.advance(.1, fresh=False, confirmed=False)
        self.assertIsNone(self.buzzer._playing)
        self.advance(5, confirmed=False)
        self.assertEqual(self.loudness.applied, .30)
        self.assertEqual(self.buzzer._playing.name, "soft.wav")

    def test_small_ticks_still_reach_the_exact_confirmed_ceiling(self):
        self.start()
        for _ in range(80):
            self.advance(.1)
        self.assertEqual(self.loudness.plan.level, .95)
        self.assertEqual(self.loudness.applied, .95)
        self.assertEqual(self.loudness.status()["level_pct"], 95)

    def test_silent_profile_also_prevents_new_logical_kind_from_playing(self):
        self.buzzer.set_profile("silent")
        self.buzzer.set("main")
        self.assertIsNone(self.buzzer._playing)
        self.assertFalse([item for item in self.trace if item[0] == "play"])


if __name__ == "__main__":
    unittest.main()
