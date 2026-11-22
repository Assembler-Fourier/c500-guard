"""Barrier left up: the Live timer, last vehicle seen, the bell switch and the one-minute reminder (21 Nov 2026)."""
import json
import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from test_watch import watch, watcher        # puts src/ on the path first
import alarm_audio
import settings_store
import test_lane_dashboard_ui as lanes


class LiveLaneTimerTests(unittest.TestCase):
    """Fake DOM, same harness as the lane tiles."""

    def harness(self):
        base = lanes.LaneDashboardUiTests('test_barrier_rows_open_barrier_setup_and_ask_for_it_only_when_unknown')
        base.setUpClass()
        source = base.routes_harness()
        hook = " return [];\n}};"
        self.assertIn(hook, source)
        source = source.replace(hook, " if(selector==='[data-bell]')return [...html.matchAll(/data-bell=\"([a-z]+)\"/g)]"
                                      ".map(m=>{const b=button({bell:m[1]});nodes.set('bell-click-'+m[1],b);return b});\n"
                                      " return [];\n}};\nconst currentState=()=>state;let state=null;")
        return base, source

    def test_up_timer_last_seen_bell_and_alerting(self):
        base, source = self.harness()
        base.js(source + r"""
state=current;
current.barriers.left_up={entrance:{up_s:75.4,muted:false}};
current.barriers.reminder={enabled:true,after_s:60,every_s:10,sounding:true};
current.lanes={entrance:{last_seen_s:130},exit:{last_seen_s:42}};
renderRoutes(current,{...healthy});
assert.equal($('lane-entrance-arm').textContent,'UP 1:15');
assert.match($('lane-entrance-arm').className,/\balerting\b/,'due, not muted and sounding: the pill pulses');
assert.equal($('lane-exit-arm').textContent,'DOWN','no timer while the arm is down');
assert.equal($('lane-entrance-count').textContent,'last car 2m ago');
assert.equal($('lane-exit-count').textContent,'last car 42s ago');
assert.equal($('lane-pedestrian-count').textContent,'no person yet');
assert.equal($('lane-entrance-bell').ariaPressed,'true');
assert.match($('lane-entrance-bell').title,/ON: short beep after 1 min up, then every 10 s/);
assert.match($('lane-entrance-bell').innerHTML,/data-icon="bell"/);
// Muted by a key press: timer stays, pulse stops.
current.barriers.left_up.entrance.muted=true;renderRoutes(current,{...healthy});
assert.equal($('lane-entrance-arm').textContent,'UP 1:15');assert.doesNotMatch($('lane-entrance-arm').className,/alerting/);
// The bell asks the server to switch the reminder off (and back on).
nodes.get('bell-click-entrance').listeners.click();
assert.deepEqual(calls.at(-1),['settings',{changes:{'barriers.reminder':false}}]);
current.barriers.reminder.enabled=false;renderRoutes(current,{...healthy});
assert.equal($('lane-entrance-bell').ariaPressed,'false');assert.match($('lane-entrance-bell').className,/\boff\b/);
assert.match($('lane-entrance-bell').innerHTML,/data-icon="bell-off"/);
nodes.get('bell-click-exit').listeners.click();
assert.deepEqual(calls.at(-1),['settings',{changes:{'barriers.reminder':true}}]);
// Over an hour, and cars in view win over the last-seen line.
current.barriers.left_up.entrance.up_s=3725;current.zones.push({name:'x',route_role:'entrance'});
renderRoutes(current,{...healthy});assert.equal($('lane-entrance-arm').textContent,'UP 1h 02m');
""")

    def test_unknown_or_unset_arm_never_shows_a_timer_and_bell_waits_for_connection(self):
        base, source = self.harness()
        base.js(source + r"""
current.barriers.left_up={entrance:{up_s:300,muted:false}};
current.barriers.reminder={enabled:true,after_s:60,every_s:10,sounding:true};
const unknown=structuredClone(current);unknown.barriers.barriers[1].fresh=false;renderRoutes(unknown,healthy);
assert.equal($('lane-entrance-arm').textContent,'UNKNOWN');assert.doesNotMatch($('lane-entrance-arm').className,/alerting/);
connectionOk=false;renderedRoutes='';renderRoutes(current,healthy);
assert.equal($('lane-entrance-bell').disabled,true);
""")

    def test_page_wiring(self):
        page = lanes.freshness.page_source()
        self.assertIn("'alignment/calibrate','settings']", page)
        self.assertIn('<symbol id="i-bell"', page)
        self.assertIn('<symbol id="i-bell-off"', page)
        self.assertIn(".arm-pill.up.alerting", page)
        self.assertIn("toast(payload.changes['barriers.reminder']?'Barrier-up reminder on':'Barrier-up reminder off')", page)


class ReminderSoundTests(unittest.TestCase):
    def tearDown(self):
        alarm_audio.reset_configuration()

    def test_period_is_clamped_and_reset(self):
        self.assertEqual(alarm_audio.set_reminder_period(10), 10.)
        self.assertEqual(alarm_audio.PROFILES['barrier_open'][0], 10.)
        self.assertEqual(alarm_audio.set_reminder_period(3), 10.)
        self.assertEqual(alarm_audio.set_reminder_period(9999), 600.)
        self.assertEqual(alarm_audio.set_reminder_period(True), 15.)
        self.assertEqual(alarm_audio.set_reminder_period(float('nan')), 15.)
        alarm_audio.set_reminder_period(20)
        alarm_audio.reset_configuration()
        self.assertEqual(alarm_audio.reminder_period(), 15.)
        self.assertEqual(alarm_audio.PROFILES['barrier_open'][0], 15.)

    def test_ten_second_file_is_versioned_and_keeps_the_same_soft_knock(self):
        pulses = alarm_audio.PROFILES['barrier_open'][1]
        with tempfile.TemporaryDirectory() as folder:
            alarm_audio.ensure_sounds(folder)
            default = (Path(folder) / '.attention-version').read_text().strip()
            alarm_audio.set_reminder_period(10)
            paths = alarm_audio.ensure_sounds(folder)
            self.assertEqual((Path(folder) / '.attention-version').read_text().strip(), default + ':reminder-10s')
            with wave.open(str(paths['barrier_open']), 'rb') as output:
                self.assertEqual(output.getnframes(), round(10 * alarm_audio.SAMPLE_RATE))
        self.assertEqual(alarm_audio.PROFILES['barrier_open'][1], pulses)
        self.assertTrue(all(frequency <= 200 for _start, _duration, frequency in pulses))


class ReminderSettingsTests(unittest.TestCase):
    def test_defaults_and_limits(self):
        items = {item['key']: item for group in settings_store.STATIC for item in group['items']}
        self.assertEqual(items['barriers.reminder_after_s']['default'], 60.)
        self.assertEqual(items['barriers.reminder_every_s']['default'], 10.)
        self.assertEqual(items['barriers.reminder_every_s']['min'], 10)

    def test_rules_load_one_minute_and_ten_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps({'barriers': {}}), encoding='utf-8')
            rules = watch.Watcher._load_barrier_rules(path)
            self.assertEqual((rules['reminder_after_s'], rules['reminder_every_s']), (60., 10.))
            path.write_text(json.dumps({'barriers': {'reminder_after_s': 5, 'reminder_every_s': 2}}), encoding='utf-8')
            rules = watch.Watcher._load_barrier_rules(path)
            self.assertEqual((rules['reminder_after_s'], rules['reminder_every_s']), (15., 10.))


class LeftUpTimerTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.barrier_rules = {**self.obj.barrier_rules, 'reminder': True, 'reminder_after_s': 60.,
                                  'reminder_every_s': 10.}
        self.now = time.monotonic()

    def tearDown(self):
        alarm_audio.reset_configuration()

    def memory(self, state='up', up_for=70., seen_ago=1.):
        self.obj._arm_memory = {'exit': {'state': state, 'since': self.now - up_for, 'since_wall': time.time() - up_for,
                                         'seen': self.now - seen_ago}}

    def test_only_a_confirmed_arm_still_seen_up_counts(self):
        self.memory()
        rows = self.obj._left_up_arms(self.now)
        self.assertAlmostEqual(rows['exit']['up_s'], 70., places=3)
        self.assertTrue(self.obj._left_up_due(self.now))
        self.memory(up_for=40.)
        self.assertFalse(self.obj._left_up_due(self.now), 'under a minute')
        self.memory(seen_ago=8.)
        self.assertEqual(self.obj._left_up_arms(self.now), {}, 'not seen up lately (picture lost or unknown)')
        self.memory(state='down')
        self.assertEqual(self.obj._left_up_arms(self.now), {})

    def test_reminder_sounds_when_armed_and_a_key_silences_it_until_the_arm_closes(self):
        self.memory()
        self.obj.armed, self.obj._test_tone, self.obj._ack_until = True, False, 0.
        self.obj._alignment_blocked = False
        self.obj.alarm = None
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'reminder_active': False}
        self.obj.barrier_workflow.mute_reminders.return_value = []
        self.obj.loudness = mock.Mock(kind=None)
        self.obj._update_alarm_audio()
        self.obj.loudness.begin.assert_called_once_with('barrier_open', source='verified_barrier')
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')
        self.obj.loudness.kind = 'barrier_open'
        self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
        self.assertTrue(self.obj._left_up_arms()['exit']['muted'])
        self.assertFalse(self.obj._left_up_due())
        # The arm went down and up again: a new up-period, so the reminder can come back.
        self.obj._arm_memory['exit']['since'] = time.monotonic() - 65
        self.assertTrue(self.obj._left_up_due())

    def test_switch_off_or_disarmed_keeps_it_quiet(self):
        self.memory()
        self.obj._test_tone, self.obj._ack_until, self.obj._alignment_blocked, self.obj.alarm = False, 0., False, None
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'reminder_active': False}
        for armed, enabled in ((False, True), (True, False)):
            with self.subTest(armed=armed, enabled=enabled):
                self.obj.armed = armed
                self.obj.barrier_rules['reminder'] = enabled
                self.obj.loudness = mock.Mock(kind=None)
                self.obj._update_alarm_audio()
                self.obj.loudness.begin.assert_not_called()

    def test_status_reports_timer_switch_and_last_seen(self):
        self.memory()
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted': True}
        self.obj.loudness = mock.Mock(kind='barrier_open')
        self.obj.alarm = None
        status = self.obj.barrier_status()
        self.assertEqual(status['left_up']['exit']['muted'], False)
        self.assertGreaterEqual(status['left_up']['exit']['up_s'], 70.)
        self.assertEqual(status['reminder'], {'enabled': True, 'after_s': 60., 'every_s': 10., 'sounding': True})
        self.obj.zones = [watch.Zone(name='Exit approach', rect=(.1, .1, .2, .2), route_role='exit'),
                          watch.Zone(name='Path', rect=(.5, .5, .2, .2), route_role='pedestrian')]
        self.obj._note_lane_seen({'detections': [{'label': 'car', 'regions': ['Exit approach']}]})
        self.obj._note_lane_seen({'detections': [{'label': 'car', 'regions': ['Somewhere else']}]})
        self.obj._note_lane_seen(None)
        lanes_seen = self.obj._lane_seen_status()
        self.assertEqual(set(lanes_seen), {'exit'})
        self.assertLess(lanes_seen['exit']['last_seen_s'], 2.)


class SharedViewSettingTests(unittest.TestCase):
    def test_switch_is_copied_to_every_saved_view_and_nothing_else_changes(self):
        root = Path(__file__).resolve().parents[1] / 'config'
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'config'
            (config / 'views' / 'anpr').mkdir(parents=True)
            (config / 'views' / 'ptz2').mkdir(parents=True)
            shutil.copy(root / 'views' / 'anpr' / 'zones.json', config / 'views' / 'anpr' / 'zones.json')
            shutil.copy(root / 'views' / 'ptz2' / 'zones.json', config / 'views' / 'ptz2' / 'zones.json')
            shutil.copy(root / 'views' / 'ptz2' / 'zones.json', config / 'zones.json')
            before = {name: json.loads((config / 'views' / name / 'zones.json').read_text(encoding='utf-8'))
                      for name in ('anpr', 'ptz2')}
            obj = watcher(config / 'zones.json')
            try:
                obj.update_settings({'barriers.reminder': False})
                for name in ('anpr', 'ptz2'):
                    after = json.loads((config / 'views' / name / 'zones.json').read_text(encoding='utf-8'))
                    self.assertIs(after['barriers']['reminder'], False)
                    after['barriers']['reminder'] = before[name]['barriers']['reminder']
                    self.assertEqual(after, before[name], f'{name}: only the reminder switch changed')
                self.assertIs(json.loads((config / 'zones.json').read_text(encoding='utf-8'))['barriers']['reminder'],
                              False)
                obj.update_settings({'alarms.ack_quiet_s': 12})
                anpr = json.loads((config / 'views' / 'anpr' / 'zones.json').read_text(encoding='utf-8'))
                self.assertEqual(anpr.get('alarms', {}).get('ack_quiet_s'), before['anpr'].get('alarms', {}).get('ack_quiet_s'),
                                 'other settings stay per view')
            finally:
                alarm_audio.reset_configuration()


if __name__ == '__main__':
    unittest.main()
