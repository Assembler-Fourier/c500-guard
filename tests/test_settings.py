"""Live settings: validation, saving, live application, sound styles, refresh."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from test_watch import watch, watcher
import alarm_audio
import settings_store

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'config' / 'views' / 'anpr' / 'zones.json').read_text(encoding='utf-8-sig'))


class RegistryTests(unittest.TestCase):
    def test_every_setting_has_a_value_and_a_label(self):
        sections = settings_store.describe(config())
        keys = [item['key'] for section in sections for item in section['items']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn('alarms.start_level', keys)
        self.assertIn('barriers.ack_on_lower', keys)
        self.assertTrue(any(key.startswith('zone:') for key in keys))
        for section in sections:
            for item in section['items']:
                self.assertTrue(item['label'])

    def test_changes_are_validated(self):
        base = config()
        for changes in ({'alarms.start_level': .9}, {'alarms.sound_theme': 'siren'},
                        {'barriers.ack_on_raise': 'yes'}, {'power.stop_when_camera_gone_s': 10},
                        {'nonsense.key': 1}, {}, {'alarms.escalation.image.level': 1.0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                settings_store.apply_changes(base, changes)

    def test_valid_changes_land_in_the_right_place(self):
        updated, changed = settings_store.apply_changes(config(), {
            'alarms.start_level': .1, 'alarms.sound_theme': 'chime', 'alarms.climb': 'gentle',
            'barriers.ack_on_lower': False, 'zone:Top road arrival:light_delta': 18})
        self.assertEqual(.1, updated['alarms']['start_level'])
        self.assertEqual('chime', updated['alarms']['sound_theme'])
        self.assertEqual(1.2, updated['alarms']['escalation']['image']['shape'])
        self.assertFalse(updated['barriers']['ack_on_lower'])
        top = next(z for z in updated['zones'] if z['name'] == 'Top road arrival')
        self.assertEqual(18, top['light_delta'])
        self.assertEqual(5, len(changed))
        # Geometry is never touched.
        self.assertEqual([z['polygon'] for z in config()['zones']], [z['polygon'] for z in updated['zones']])

    def test_weaker_evidence_cannot_be_set_louder(self):
        with self.assertRaises(ValueError):
            settings_store.apply_changes(config(), {'alarms.escalation.urgent.level': 1.0,
                                                    'alarms.escalation.confirmed.level': .5})


class AudioTests(unittest.TestCase):
    def tearDown(self):
        alarm_audio.reset_configuration()

    def test_start_level_is_where_every_alarm_begins(self):
        alarm_audio.set_start_level(.10)
        self.assertAlmostEqual(.10, alarm_audio.plan(0, kind='main', confirmed=True).level)
        self.assertGreater(alarm_audio.plan(4, kind='main', confirmed=True).level, .10)

    def test_start_level_is_clamped_below_every_ceiling(self):
        self.assertEqual(.05, alarm_audio.set_start_level(.0))
        self.assertLessEqual(alarm_audio.set_start_level(.99), alarm_audio.settings()['image']['level'])

    def test_each_theme_renders_distinct_audible_sound(self):
        rendered = {}
        for name in alarm_audio.THEMES:
            alarm_audio.set_theme(name)
            samples = alarm_audio.pcm_samples('soft')
            self.assertGreater(max(abs(v) for v in samples), 1000, name)
            rendered[name] = bytes(samples)
        self.assertEqual(len(rendered), len(set(rendered.values())))

    def test_unknown_theme_keeps_original_pulses(self):
        self.assertEqual('pulses', alarm_audio.set_theme('siren'))

    def test_theme_change_regenerates_the_sound_files(self):
        with tempfile.TemporaryDirectory() as folder:
            alarm_audio.set_theme('pulses')
            first = alarm_audio.ensure_sounds(folder)['soft'].read_bytes()
            alarm_audio.set_theme('beeps')
            second = alarm_audio.ensure_sounds(folder)['soft'].read_bytes()
        self.assertNotEqual(first, second)

    def test_urgent_ceiling_is_respected(self):
        self.assertAlmostEqual(alarm_audio.settings()['urgent']['level'],
                               alarm_audio.level_ceiling(kind='main', urgent=True))
        self.assertEqual(.18, alarm_audio.level_ceiling(kind='fault'))


class LiveSettingsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.zones = Path(self.folder.name) / 'zones.json'
        shutil.copyfile(ROOT / 'config' / 'views' / 'anpr' / 'zones.json', self.zones)
        self.obj = watcher()
        self.obj.args.zones = str(self.zones)
        self.addCleanup(alarm_audio.reset_configuration)

    def test_saved_and_applied_without_restart(self):
        with mock.patch.object(watch, 'ensure_sounds', return_value=self.obj.sounds):
            result = self.obj.update_settings({'alarms.start_level': .1, 'barriers.ack_on_lower': False,
                                               'alarms.ack_quiet_s': 20, 'zone:Top road arrival:light_delta': 22})
        self.assertTrue(result['ok'])
        saved = json.loads(self.zones.read_text(encoding='utf-8'))
        self.assertEqual(.1, saved['alarms']['start_level'])
        self.assertAlmostEqual(.1, alarm_audio.start_level())
        self.assertFalse(self.obj.barrier_rules['ack_on_lower'])
        self.assertEqual(20., self.obj.ack_quiet_s)
        top = next(z for z in self.obj._pending_zones if z.name == 'Top road arrival')
        self.assertEqual(22, top.light_delta)
        self.assertTrue(list(self.zones.parent.glob('zones.json.settings-*')))

    def test_bad_change_saves_nothing(self):
        before = self.zones.read_bytes()
        with self.assertRaises(ValueError):
            self.obj.update_settings({'alarms.start_level': 5})
        self.assertEqual(before, self.zones.read_bytes())

    def test_lowering_switch_is_obeyed(self):
        self.obj.barrier_rules['ack_on_lower'] = False
        self.obj.alarm = 'main'
        self.assertFalse(self.obj._silence_for_barrier_change({'type': 'lowered', 'alarm_id': 'x'}))
        self.assertEqual('main', self.obj.alarm)

    def test_site_config_starts_alarms_at_ten_percent_and_climbs_gently(self):
        settings = self.obj._load_audio_settings(ROOT / 'config' / 'views' / 'anpr' / 'zones.json')
        self.assertAlmostEqual(.10, settings['start_level'])
        self.assertAlmostEqual(.10, alarm_audio.plan(0, kind='main', confirmed=True).level)
        self.assertEqual(1.2, settings['confirmed']['shape'])

    def test_snapshot_lists_sections(self):
        snapshot = self.obj.settings_snapshot()
        self.assertTrue(snapshot['sections'])
        self.assertIn('barrier_rules', snapshot['live'])


class RefreshTests(unittest.TestCase):
    def test_refresh_writes_the_request_and_stops_the_loop(self):
        obj = watcher()
        obj._running = True
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(watch, 'STATE_DIR', Path(folder)), \
             mock.patch.object(watch.time, 'sleep'):
            obj.request_refresh('test')
            for _ in range(50):
                if not obj._running:
                    break
                watch.time.sleep(0)
            self.assertTrue((Path(folder) / 'restart-request').exists())
        import time as real_time
        deadline = real_time.monotonic() + 3
        while obj._running and real_time.monotonic() < deadline:
            real_time.sleep(.05)
        self.assertFalse(obj._running)


if __name__ == '__main__':
    unittest.main()
