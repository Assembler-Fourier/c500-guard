"""Lane identity in generated PCM, without opening an audio device."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import alarm_audio as audio


class LaneAudioTests(unittest.TestCase):
    def setUp(self):
        audio.reset_configuration()

    def tearDown(self):
        audio.reset_configuration()

    def test_mapping_preserves_neutral_profiles_and_handles_lane_switches(self):
        self.assertEqual(audio.LANES,
                         {"entrance": "Rising tones", "exit": "Falling tones"})
        for profile in ("soft", "notice", "wake"):
            for lane in audio.LANES:
                chosen = audio.lane_profile(profile, lane)
                self.assertEqual(chosen, f"{lane}_{profile}")
                self.assertIn(chosen, audio.PROFILES)
                self.assertEqual(audio.lane_profile(chosen, lane), chosen)
                self.assertEqual(audio.lane_profile(chosen, None), profile)
                self.assertEqual(audio.lane_profile(chosen, "exit"), f"exit_{profile}")
            for lane in (None, "", "pedestrian", "entrance_approach"):
                self.assertEqual(audio.lane_profile(profile, lane), profile)
        for profile in ("silent", "limited", "fault", "barrier_open", "unknown"):
            for lane in audio.LANES:
                self.assertEqual(audio.lane_profile(profile, lane), profile)
        for alias in ("main", "early"):
            self.assertEqual(audio.lane_profile(alias, "entrance"), "entrance_soft")

    def test_lane_selection_preserves_evidence_policy_and_escalation(self):
        audio.set_start_level(.10)
        for seconds in (0, .5, 1.49, 1.5, 3.49, 3.5, 8, 20):
            for evidence in ({}, {"confirmed": True}, {"urgent": True},
                             {"evidence_fresh": False}, {"acknowledged": True},
                             {"kind": "fault"}, {"kind": "barrier_open"}):
                original = audio.plan(seconds, **evidence)
                for lane in audio.LANES:
                    selected = replace(original,
                                       profile=audio.lane_profile(original.profile, lane))
                    self.assertEqual(replace(selected, profile=original.profile), original)
                    if selected.audible:
                        period, pulses = audio.PROFILES[selected.profile]
                        self.assertEqual(selected.repeat_s, period)
                        self.assertEqual(selected.burst_s, sum(pulse[1] for pulse in pulses))
        self.assertEqual(audio.plan(0, confirmed=True).level, .10)
        self.assertEqual(audio.lane_profile(audio.limited_plan(0).profile, "exit"), "limited")

    def test_every_theme_preserves_timing_gain_and_digital_energy(self):
        for theme in audio.THEMES:
            audio.set_theme(theme)
            for profile in ("soft", "notice", "wake"):
                period, pulses = audio.PROFILES[profile]
                baseline = audio.pcm_samples(profile)
                baseline_energy = sum(value * value for value in baseline)
                for lane in audio.LANES:
                    with self.subTest(theme=theme, profile=profile, lane=lane):
                        chosen = audio.lane_profile(profile, lane)
                        lane_period, lane_pulses = audio.PROFILES[chosen]
                        self.assertEqual(lane_period, period)
                        self.assertEqual([(p[0], p[1]) for p in lane_pulses],
                                         [(p[0], p[1]) for p in pulses])
                        self.assertEqual(audio.PROFILE_PEAKS[chosen], audio.PROFILE_PEAKS[profile])
                        self.assertEqual(audio.PROFILE_HARMONICS[chosen],
                                         audio.PROFILE_HARMONICS[profile])
                        pcm = audio.pcm_samples(chosen)
                        self.assertEqual(len(pcm), len(baseline))
                        self.assertLessEqual(max(map(abs, pcm)), audio.PROFILE_PEAKS[chosen] * 32767 + 1)
                        self.assertEqual(pcm[0], 0)
                        self.assertEqual(pcm[-1], 0)
                        # Pitch changes have negligible numerical effects on
                        # energy; the gain, duration and envelope stay intact.
                        ratio = sum(value * value for value in pcm) / baseline_energy
                        self.assertAlmostEqual(ratio, 1.0, delta=.005)
                        # Check the same smooth attack for every tone, including
                        # later chime notes which overwrite the preceding tail.
                        for start, _duration, _frequency in lane_pulses:
                            first = round(start * audio.SAMPLE_RATE)
                            head = pcm[first:first + round(.001 * audio.SAMPLE_RATE)]
                            body = pcm[first + round(.025 * audio.SAMPLE_RATE):
                                       first + round(.035 * audio.SAMPLE_RATE)]
                            self.assertLess(max(map(abs, head)), max(map(abs, body)) * .12)

    def test_pcm_pitch_direction_is_distinct_in_each_theme_and_stage(self):
        def measured_frequency(pcm, start):
            # The dominant fundamental's upward zero crossings are measured
            # from PCM, independently of the profile's frequency metadata.
            first = round((start + .04) * audio.SAMPLE_RATE)
            last = round((start + .08) * audio.SAMPLE_RATE)
            segment = pcm[first:last]
            crossings = sum(a <= 0 < b for a, b in zip(segment, segment[1:]))
            return crossings * audio.SAMPLE_RATE / (len(segment) - 1)

        for theme in audio.THEMES:
            audio.set_theme(theme)
            for profile in ("soft", "notice", "wake"):
                rendered = {}
                for lane in audio.LANES:
                    chosen = audio.lane_profile(profile, lane)
                    pcm = audio.pcm_samples(chosen)
                    pulses = audio.PROFILES[chosen][1]
                    measured = [measured_frequency(pcm, pulse[0]) for pulse in pulses]
                    with self.subTest(theme=theme, profile=profile, lane=lane):
                        for actual, pulse in zip(measured, pulses):
                            self.assertAlmostEqual(actual, pulse[2], delta=30)
                        if lane == "entrance":
                            self.assertLess(measured[0], measured[-1] - 70)
                        else:
                            self.assertGreater(measured[0], measured[-1] + 70)
                    rendered[lane] = pcm.tobytes()
                self.assertNotEqual(rendered["entrance"], rendered["exit"])

    def test_neutral_and_default_waveforms_remain_byte_identical(self):
        # Captured from the production snapshot before adding lane variants.
        expected = {
            "limited": "56b477269462b1417d9489203c170cbc9121e231b388a506bf66c4ca8cdf409c",
            "fault": "c090625eadce7f2c692d207392563a6307cf5247c7f6963d6aaea26ad75340e1",
            "barrier_open": "34af21d7635fed3a52d2ef3fdd049736774614c2737d9f74c354cd9428dbcd7f",
            "soft": "08ae68b28ff656c6b881c944cbd9c4746a66123e2bc2b32ad8399f23ad673ab2",
            "notice": "265fcf48a24d209fbc452da2e37b3e81e2307ae9330fc310263174c19c83876f",
            "wake": "bdc0adc2b03f44aafebd824b7b7eb599dd73f799a69508face2ab2050af4decc",
        }
        for theme in audio.THEMES:
            audio.set_theme(theme)
            profiles = expected if theme == audio.DEFAULT_THEME else ("limited", "fault", "barrier_open")
            for profile in profiles:
                pcm = audio.pcm_samples(profile)
                if sys.byteorder != "little":
                    pcm.byteswap()
                self.assertEqual(hashlib.sha256(pcm.tobytes()).hexdigest(), expected[profile])

    def test_assets_prebuild_both_lanes_and_are_reused_without_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio.set_theme("low")
            paths = audio.ensure_sounds(directory)
            for lane in audio.LANES:
                for profile in ("soft", "notice", "wake"):
                    chosen = audio.lane_profile(profile, lane)
                    with wave.open(str(paths[chosen]), "rb") as sound:
                        self.assertEqual(sound.getnchannels(), 1)
                        self.assertEqual(sound.getsampwidth(), 2)
                        self.assertEqual(sound.getframerate(), audio.SAMPLE_RATE)
                        self.assertEqual(sound.getnframes(),
                                         round(audio.PROFILES[chosen][0] * audio.SAMPLE_RATE))
            mtimes = {name: path.stat().st_mtime_ns for name, path in paths.items()}
            with patch.object(audio, "pcm_samples", side_effect=AssertionError("unexpected regeneration")):
                for lane in ("entrance", "exit", "entrance"):
                    self.assertTrue(paths[audio.lane_profile("soft", lane)].is_file())
                self.assertEqual(audio.ensure_sounds(directory), paths)
            self.assertEqual(mtimes, {name: path.stat().st_mtime_ns for name, path in paths.items()})
            before = paths["entrance_soft"].read_bytes()
            audio.set_theme("beeps")
            regenerated = audio.ensure_sounds(directory)
            self.assertNotEqual(regenerated["entrance_soft"].read_bytes(), before)
            self.assertEqual((directory / ".attention-version").read_text().strip(),
                             f"{audio.SOUND_VERSION}:beeps")
            self.assertFalse(list(directory.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
