"""No audio devices are opened: validate policy and generated PCM offline."""

from array import array
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alarm_audio import (MAX_LEVEL, WEAK_LEVEL, PROFILES, PROFILE_PEAKS, SAMPLE_RATE, SOUND_VERSION,
                         ensure_sounds, pcm_samples, plan, limited_plan)


class GentlePolicyTests(unittest.TestCase):
    def test_new_arrival_is_brief_and_moderate_for_every_presence_state(self):
        for state in ("watching", "present", "away", "unknown", "drowsy", "asleep", "bad"):
            with self.subTest(state=state):
                result = plan(0, confirmed=True, presence_state=state)
                self.assertEqual(result.profile, "soft")
                self.assertAlmostEqual(result.level, 0.30)
                self.assertLess(result.burst_s, 0.7)
                self.assertGreaterEqual(result.repeat_s, 3)

    def test_unconfirmed_current_evidence_keeps_reminding_at_moderate_cap(self):
        for source in ("motion", "light", "strict_change", "ai", "unknown"):
            for elapsed in (0, 4, 8, 11.9, 12, 30, 1000):
                with self.subTest(source=source, elapsed=elapsed):
                    result = plan(elapsed, source=source)
                    self.assertLessEqual(result.level, 0.75)
                    self.assertTrue(result.audible)
                    self.assertNotEqual(result.profile, "wake")
        for seconds, level in ((0, .30), (6, .75), (15, .75), (600, .75)):
            self.assertAlmostEqual(plan(seconds, confirmed=False).level, level)
        self.assertGreaterEqual(plan(2, source="light").level, .57)
        self.assertLessEqual(plan(2, source="light").level, .60)
        self.assertGreaterEqual(plan(4, source="light").level, .69)
        self.assertLessEqual(plan(4, source="light").level, .72)

    def test_confirmed_ramp_is_fast_continuous_and_bounded(self):
        points = ((0, 0.30), (8, .95), (20, .95), (600, .95))
        for seconds, level in points:
            self.assertAlmostEqual(plan(seconds, confirmed=True).level, level)
        levels = [plan(t / 10, confirmed=True).level for t in range(1000)]
        self.assertEqual(levels, sorted(levels))
        self.assertLess(max(b - a for a, b in zip(levels, levels[1:])), 0.027)
        self.assertLessEqual(max(levels), MAX_LEVEL)
        self.assertGreaterEqual(plan(2, confirmed=True).level, .65)
        self.assertLessEqual(plan(2, confirmed=True).level, .70)
        self.assertGreaterEqual(plan(4, confirmed=True).level, .83)
        self.assertLessEqual(plan(4, confirmed=True).level, .87)

    def test_presence_never_delays_or_reduces_either_alarm_ramp(self):
        for confirmed in (True, False):
            for seconds in (0, .1, 1, 2, 4, 6, 8, 60):
                ordinary = plan(seconds, confirmed=confirmed)
                for state in ("watching", "present", "unknown", "away", "drowsy", "asleep", "bad"):
                    self.assertEqual(plan(seconds, confirmed=confirmed, presence_state=state), ordinary)

    def test_exponential_rises_remain_continuous_at_caps_and_start(self):
        for confirmed, endpoint, ceiling in ((True, 8., MAX_LEVEL), (False, 6., WEAK_LEVEL)):
            with self.subTest(confirmed=confirmed):
                self.assertAlmostEqual(plan(1e-8, confirmed=confirmed).level, .30, places=7)
                self.assertAlmostEqual(plan(endpoint - 1e-8, confirmed=confirmed).level, ceiling, places=7)
                self.assertEqual(plan(endpoint, confirmed=confirmed).level, ceiling)
                self.assertEqual(plan(1e300, confirmed=confirmed).level, ceiling)
                levels = [plan(t / 100, confirmed=confirmed).level for t in range(1001)]
                self.assertEqual(levels, sorted(levels))

    def test_stronger_patterns_arrive_in_seconds_without_waiting_for_late_milestones(self):
        for confirmed in (False, True):
            self.assertEqual(plan(0, confirmed=confirmed).profile, "soft")
            self.assertEqual(plan(1.49, confirmed=confirmed).profile, "soft")
            self.assertEqual(plan(1.5, confirmed=confirmed).profile, "notice")
        self.assertEqual(plan(3.5, confirmed=True).profile, "wake")
        self.assertEqual(plan(3.5, confirmed=False).profile, "notice")

    def test_unknown_or_sleep_labels_do_not_make_alarms_louder(self):
        for seconds in (0, 4, 8, 20, 40):
            ordinary = plan(seconds, confirmed=True)
            for state in ("unknown", "drowsy", "asleep", "away"):
                self.assertEqual(plan(seconds, confirmed=True, presence_state=state), ordinary)

    def test_left_or_stale_event_never_keeps_audible_reminders(self):
        for elapsed in (0, 0.5, 8, 50):
            for changes in ({"event_present": False}, {"evidence_fresh": False}):
                with self.subTest(elapsed=elapsed, changes=changes):
                    result = plan(elapsed, confirmed=True, **changes)
                    self.assertFalse(result.audible)
                    self.assertIsNone(result.sound)
                    self.assertEqual(result.level, 0)

    def test_acknowledgement_always_stops_every_sound(self):
        for kind in ("main", "early", "fault", "barrier_open"):
            for elapsed in (0, 1000):
                for changes in ({"acknowledged": True}, {"quiet_remaining_s": 0.01}, {"enabled": False}):
                    self.assertFalse(plan(elapsed, kind=kind, confirmed=True, **changes).audible)

    def test_faults_are_short_infrequent_and_never_escalate(self):
        for seconds in (0, 8, 40, 5000):
            result = plan(seconds, kind="fault", confirmed=True, presence_state="asleep")
            self.assertEqual(result.level, 0.18)
            self.assertEqual(result.repeat_s, 30)
            self.assertLess(result.burst_s, 0.2)

    def test_open_barrier_has_a_quiet_double_thump_without_escalation(self):
        for elapsed in (0, 2, 8, 60, 5000):
            for confirmed in (True, False):
                result = plan(elapsed, kind="barrier_open", confirmed=confirmed,
                              presence_state="asleep")
                self.assertEqual(result.profile, "barrier_open")
                self.assertEqual(result.level, .18)
                self.assertEqual(result.repeat_s, 15)
                self.assertLessEqual(result.burst_s, .32)
                self.assertEqual(len(PROFILES[result.profile][1]), 2)

    def test_barrier_reminder_requires_current_open_evidence(self):
        for elapsed in (0, 5000):
            for changes in ({"event_present": False}, {"evidence_fresh": False}):
                self.assertFalse(plan(elapsed, kind="barrier_open", **changes).audible)

    def test_barrier_reminder_is_silent_when_mixer_control_is_unavailable(self):
        for elapsed in (0, 1, 11, 12, 60, 5000):
            result = limited_plan(elapsed, kind="barrier_open")
            self.assertFalse(result.audible)
            self.assertEqual(result.level, 0)

    def test_confirmation_allows_stronger_profile_than_image_change_alone(self):
        weaker = plan(16, source="light", confirmed=False)
        self.assertTrue(weaker.audible)
        self.assertAlmostEqual(weaker.level, .75)
        result = plan(16, source="light", confirmed=True)
        self.assertTrue(result.audible)
        self.assertEqual(result.profile, "wake")
        self.assertGreater(result.level, weaker.level)

    def test_mixer_fallback_remains_short_and_low_even_as_alarm_policy_increases(self):
        for presence in ("watching", "present", "unknown", "asleep"):
            for elapsed in (0, 4, 11.9, 12, 20, 1000):
                result = limited_plan(elapsed, presence_state=presence)
                self.assertEqual(result.audible, elapsed < 12)
                self.assertLessEqual(result.level, .12)
                if result.audible:
                    self.assertEqual(result.profile, "limited")
                    self.assertLessEqual(result.burst_s, .16)

    def test_quiet_ack_stale_and_departure_also_stop_mixer_fallback(self):
        for changes in ({"acknowledged": True}, {"quiet_remaining_s": 10},
                        {"evidence_fresh": False}, {"event_present": False}, {"enabled": False}):
            self.assertFalse(limited_plan(0, **changes).audible)

    def test_fault_fallback_stays_low_and_infrequent(self):
        result = limited_plan(500, kind="fault")
        self.assertTrue(result.audible)
        self.assertEqual(result.profile, "fault")
        self.assertLessEqual(result.level, .18)
        self.assertEqual(result.repeat_s, 30.)

    def test_invalid_inputs_do_not_produce_loud_tones(self):
        for invalid in (None, "not-a-number", math.nan, math.inf, -math.inf):
            self.assertFalse(plan(invalid, confirmed=True).audible)
            self.assertFalse(plan(50, quiet_remaining_s=invalid, confirmed=True).audible)
        self.assertFalse(plan(50, kind="unknown", confirmed=True).audible)
        self.assertAlmostEqual(plan(-100, confirmed=True).level, 0.30)


class GentleWaveformTests(unittest.TestCase):
    def test_every_profile_keeps_silent_gaps_and_does_not_clip(self):
        for profile, (period, pulses) in PROFILES.items():
            with self.subTest(profile=profile):
                samples = pcm_samples(profile)
                self.assertEqual(len(samples), round(period * SAMPLE_RATE))
                peak = max(abs(x) for x in samples)
                self.assertGreater(peak, 1000)
                self.assertLessEqual(peak, PROFILE_PEAKS[profile] * 32767 + 1)
                self.assertGreater(sum(x == 0 for x in samples) / len(samples), 0.40)
                self.assertEqual(samples[0], 0)
                self.assertEqual(samples[-1], 0)
                for start, duration, _frequency in pulses:
                    first = round(start * SAMPLE_RATE)
                    count = round(duration * SAMPLE_RATE)
                    self.assertEqual(samples[first], 0)
                    self.assertEqual(samples[first + count - 1], 0)
                    self.assertLess(duration, 1.0, "Even the strongest pulse must leave a gap")

    def test_no_square_wave_or_abrupt_click_edges(self):
        for profile in PROFILES:
            samples = pcm_samples(profile)
            difference = max(abs(b - a) for a, b in zip(samples, samples[1:]))
            self.assertLess(difference, 0.37 * 32767)
            # A rounded pulse has many amplitudes, unlike a two-level square.
            self.assertGreater(len(set(samples)), 1000)

    def test_attention_strength_increases_in_energy_and_duration(self):
        energy = []
        duration = []
        for profile in ("limited", "soft", "notice", "wake"):
            samples = pcm_samples(profile)
            energy.append(sum(float(sample) ** 2 for sample in samples) / len(samples))
            duration.append(sum(pulse[1] for pulse in PROFILES[profile][1]))
        self.assertEqual(energy, sorted(energy))
        self.assertEqual(duration, sorted(duration))
        self.assertGreater(energy[1], energy[0] * 10)

    def test_attention_tones_have_rounded_edges_and_a_sustained_middle(self):
        for profile in ("soft", "notice", "wake"):
            samples = pcm_samples(profile)
            head = max(abs(v) for v in samples[:round(.005 * SAMPLE_RATE)])
            middle = max(abs(v) for v in samples[round(.05 * SAMPLE_RATE):round(.1 * SAMPLE_RATE)])
            self.assertLess(head, middle * .15)
            self.assertGreater(middle, .25 * 32767)

    def test_barrier_thumps_use_low_pitch_rounded_envelopes_and_long_silence(self):
        period, pulses = PROFILES["barrier_open"]
        samples = pcm_samples("barrier_open")
        self.assertEqual(period, 15)
        self.assertEqual(len(pulses), 2)
        for start, duration, frequency in pulses:
            self.assertLessEqual(frequency, 200)
            first = round(start * SAMPLE_RATE)
            head = max(abs(v) for v in samples[first:first + round(.005 * SAMPLE_RATE)])
            middle = max(abs(v) for v in samples[first + round(.05 * SAMPLE_RATE):
                                                  first + round(.1 * SAMPLE_RATE)])
            self.assertLess(head, middle * .02)
            self.assertLessEqual(duration, .16)
        self.assertGreater(sum(sample == 0 for sample in samples) / len(samples), .97)

    def test_wav_files_match_assets_and_legacy_names_start_soft(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = ensure_sounds(temporary)
            self.assertEqual(paths["main"], paths["soft"])
            self.assertEqual(paths["early"], paths["soft"])
            self.assertEqual((Path(temporary) / ".attention-version").read_text().strip(), SOUND_VERSION)
            self.assertIn("limited", paths)
            self.assertIn("barrier_open", paths)
            timestamps = {name: path.stat().st_mtime_ns for name, path in paths.items()}
            for profile in PROFILES:
                with wave.open(str(paths[profile]), "rb") as output:
                    self.assertEqual(output.getnchannels(), 1)
                    self.assertEqual(output.getsampwidth(), 2)
                    self.assertEqual(output.getframerate(), SAMPLE_RATE)
                    self.assertEqual(output.getnframes(), round(PROFILES[profile][0] * SAMPLE_RATE))
                    self.assertGreater(len(output.readframes(100)), 0)
            ensure_sounds(temporary)
            self.assertEqual({name: path.stat().st_mtime_ns for name, path in paths.items()}, timestamps)
            paths["wake"].unlink()
            self.assertTrue(ensure_sounds(temporary)["wake"].exists())
            self.assertFalse(list(Path(temporary).glob("*.tmp")))

    def test_invalid_profile_or_sample_rate_rejected(self):
        for rate in (0, 1000, 24000.0, 200000):
            with self.assertRaises(ValueError):
                pcm_samples("soft", sample_rate=rate)
        with self.assertRaises(ValueError):
            pcm_samples("wail")


if __name__ == "__main__":
    unittest.main()
