"""Real watcher methods; synthetic configs, fake buzzer, scratch-only persistence."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
import wave
from unittest import mock

from test_watch import watch, watcher, FakeBuzzer
import alarm_audio
import settings_store
import zone_editor


CONFIG = {'view': 'view-a', 'zones': [
    {'name': 'Synthetic exit', 'rect': [.1, .1, .3, .3], 'route_role': 'exit'}],
    'barriers': {'reminder': True, 'reminder_after_s': 60., 'reminder_every_s': 10.},
    'alarms': {'sound_theme': 'pulses', 'ack_quiet_s': 10.}}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='reminder-interactions-')
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(alarm_audio.reset_configuration)
        self.root = Path(self.temp.name)
        self.path = self.root / 'zones.json'
        self.path.write_text(json.dumps(CONFIG), encoding='utf-8')
        self.obj = watcher(self.path)
        self.obj._mark_alert_acknowledged = mock.Mock()
        self.obj._alignment_blocked = False
        self.obj.armed, self.obj._test_tone, self.obj._ack_until = True, False, 0.
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.learned = []
        self.obj.barrier_workflow.learning_skips = {}
        self.obj.barrier_workflow.monitor.storage_path = None
        self.obj.barrier_workflow.mute_reminders.return_value = []
        self.obj.barrier_workflow.status.return_value = {'trusted': True, 'reminder_active': False}
        self.obj.loudness = mock.Mock(kind=None)
        self.obj.loudness.end.side_effect = lambda: setattr(self.obj.loudness, 'kind', None)
        self.obj.loudness.begin.side_effect = lambda kind, **kw: setattr(self.obj.loudness, 'kind', kind)
        self.clock = mock.patch.object(watch.time, 'monotonic', return_value=1000.).start()
        self.wall = mock.patch.object(watch.time, 'time', return_value=2000.).start()
        self.addCleanup(mock.patch.stopall)

    def at(self, instant):
        self.clock.return_value = instant
        self.wall.return_value = instant + 1000.

    def memory(self, arm, since=900., seen=1000.):
        if not hasattr(self.obj, '_arm_memory'):
            self.obj._arm_memory = {}
        self.obj._arm_memory[arm] = {'state': 'up', 'since': since, 'since_wall': since + 1000., 'seen': seen}

    def observe(self, arm, state='up', conflict=False, tracker=True):
        now = self.clock.return_value
        self.obj._arm_live = ({arm: {'ok': True, 'theta': 0. if conflict or state == 'down' else 90.,
            'shown': 0. if conflict or state == 'down' else 90., 'position_conflict': conflict,
            'at': now, 'at_wall': now + 1000.}} if tracker else {})
        status = {'trusted': True, 'reminder_active': False, 'barriers': [
            {'id': arm, 'state': state, 'fresh': True, 'calibrated': True, 'observed_at': now}]}
        self.obj.barrier_workflow.status.return_value = status
        self.obj._note_arm_memory(status)
        return status

    def views(self):
        result = []
        for name in ('view-a', 'view-b', 'view-c'):
            path = self.root / 'views' / name / 'zones.json'
            path.parent.mkdir(parents=True)
            data = copy.deepcopy(CONFIG)
            data['view'] = name
            data['zones'][0]['name'] = 'Synthetic ' + name
            data['barriers']['ack_on_raise'] = name != 'view-b'
            data['alarms']['ack_quiet_s'] = 12. if name == 'view-b' else 10.
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            result.append(path)
        return result

    def update(self, changes):
        with mock.patch.object(watch, 'ensure_sounds', return_value={}):
            return self.obj.update_settings(changes)


class AudioResetTests(Fixture):
    def duration(self):
        with wave.open(str(self.obj.sounds['barrier_open']), 'rb') as output:
            return output.getnframes() / output.getframerate()

    def test_sound_style_save_preserves_ten_second_wav(self):
        alarm_audio.set_reminder_period(10.)
        self.obj.sounds = alarm_audio.ensure_sounds(self.root / 'sounds')
        with mock.patch.object(watch, 'ensure_sounds', side_effect=lambda force=False:
                alarm_audio.ensure_sounds(self.root / 'sounds', force=force)):
            result = self.obj.update_settings({'alarms.sound_theme': 'chime'})
        self.assertTrue(result['ok'])
        self.assertEqual(self.duration(), 10.)
        self.assertEqual(alarm_audio.reminder_period(), 10.)
        self.assertEqual(self.obj.barrier_rules['reminder_every_s'], 10.)

    def test_every_alarm_setting_preserves_period_before_asset_regeneration(self):
        changes = {item['key']: item['default'] for section in settings_store.STATIC
                   for item in section['items'] if item['key'].startswith('alarms.')}
        for key, value in changes.items():
            with self.subTest(key=key):
                alarm_audio.set_reminder_period(10.)
                self.assertTrue(self.update({key: value})['ok'])
                self.assertEqual(alarm_audio.reminder_period(), 10.)
        self.obj.sounds = alarm_audio.ensure_sounds(self.root / 'sounds', force=True)
        self.assertEqual(self.duration(), 10.)

    def test_constructing_watcher_keeps_module_period_matching_written_asset(self):
        with mock.patch.object(watch, 'ensure_sounds', side_effect=lambda:
                alarm_audio.ensure_sounds(self.root / 'sounds')), \
                mock.patch.object(watch, 'Buzzer', FakeBuzzer), \
                mock.patch('semantic_monitor.SemanticMonitor', return_value=None), \
                mock.patch.object(watch.Watcher, '_load_armed', return_value=True):
            self.obj = watch.Watcher(self.obj.args)
        self.assertEqual(self.duration(), 10.)
        self.assertEqual(alarm_audio.reminder_period(), 10.)

    def test_combined_sound_and_period_change_generates_matching_asset(self):
        with mock.patch.object(watch, 'ensure_sounds', side_effect=lambda force=False:
                alarm_audio.ensure_sounds(self.root / 'sounds', force=force)):
            self.obj.update_settings({'alarms.sound_theme': 'beeps', 'barriers.reminder_every_s': 20.})
        self.assertEqual((self.duration(), alarm_audio.reminder_period()), (20., 20.))

    def test_combined_volume_and_period_change_regenerates_without_theme_change(self):
        alarm_audio.set_reminder_period(10.)
        self.obj.sounds = alarm_audio.ensure_sounds(self.root / 'sounds')
        with mock.patch.object(watch, 'ensure_sounds', side_effect=lambda force=False:
                alarm_audio.ensure_sounds(self.root / 'sounds', force=force)) as generate:
            self.obj.update_settings({'alarms.start_level': .15, 'barriers.reminder_every_s': 20.})
        self.assertEqual((self.duration(), alarm_audio.reminder_period()), (20., 20.))
        self.assertEqual(generate.call_count, 1)


class MultiArmReminderTests(Fixture):
    def test_other_arm_ack_during_unknown_preserves_earlier_muted_period(self):
        for first, second in (('exit', 'entrance'), ('entrance', 'exit')):
            with self.subTest(first=first):
                self.at(1000.)
                self.obj._arm_memory, self.obj._left_up_muted = {}, {}
                self.memory(first)
                self.obj._update_alarm_audio()
                self.assertEqual(self.obj.buzzer.want, 'barrier_open')
                self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
                self.at(1010.)
                self.memory(second, since=910., seen=1010.)
                self.obj._update_alarm_audio()
                self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
                self.at(1011.)
                self.observe(first)
                self.obj._update_alarm_audio()
                self.assertTrue(self.obj._left_up_arms()[first]['muted'])
                self.assertIsNone(self.obj.buzzer.want)

    def test_confirmed_close_rearms_only_that_arm(self):
        for arm in ('exit', 'entrance'):
            self.memory(arm)
        self.obj._update_alarm_audio()
        self.assertTrue(self.obj.acknowledge())
        self.at(1001.)
        self.observe('exit', 'down')
        self.at(1002.)
        self.observe('exit', 'up')
        self.at(1062.)
        self.observe('exit')
        self.observe('entrance')
        rows = self.obj._left_up_arms()
        self.assertFalse(rows['exit']['muted'])
        self.assertTrue(rows['entrance']['muted'])
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')

    def test_conflicting_angle_neither_refreshes_memory_nor_sounds(self):
        self.memory('exit', seen=999.)
        for instant in (1000., 1000.2, 1001., 1005., 1010., 2000.):
            with self.subTest(instant=instant):
                self.at(instant)
                self.observe('exit', conflict=True)
                self.assertEqual(self.obj.barrier_status()['barriers'][0]['state'], 'unknown')
                self.assertFalse(self.obj._left_up_due())
                self.obj._update_alarm_audio()
                self.assertIsNone(self.obj.buzzer.want)
                self.assertEqual(self.obj._arm_memory['exit']['seen'], 999.)

    def test_conflict_stops_running_reminder_and_requires_new_consistent_observation(self):
        self.memory('exit')
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')
        self.at(1000.1)
        self.observe('exit', conflict=True)
        self.obj._update_alarm_audio()
        self.assertIsNone(self.obj.buzzer.want)
        self.at(1002.)
        self.obj._arm_live = {}  # conflict expiry alone is no fresh positive evidence
        self.assertFalse(self.obj._left_up_due())
        self.observe('exit', tracker=False)
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')
        self.assertEqual(self.obj._arm_memory['exit']['since'], 900.)

    def test_first_conflicting_up_is_not_a_confirmed_up_period(self):
        self.observe('exit', conflict=True)
        self.assertNotIn('exit', getattr(self.obj, '_arm_memory', {}))
        self.at(1100.)
        self.observe('exit')
        self.assertFalse(self.obj._left_up_due())
        self.assertEqual(self.obj._arm_memory['exit']['since'], 1100.)

    def test_served_arm_reminder_cannot_bypass_conflict_or_muted_period(self):
        self.memory('exit')
        status = self.observe('exit', conflict=True)
        route = watch.VisitMemory.route_key(self.obj.zones[0])
        status.update(reminder_active=True, service={'routes': {route: {'reminder_active': True}}})
        self.obj._update_alarm_audio()
        self.assertIsNone(self.obj.buzzer.want)
        self.at(1000.1)
        self.observe('exit')
        self.obj.barrier_workflow.status.return_value = status
        self.obj._left_up_muted = {'exit': 900.}
        self.obj._update_alarm_audio()
        self.assertIsNone(self.obj.buzzer.want)

    def test_one_conflicting_arm_does_not_suppress_other_eligible_reminder(self):
        self.memory('exit')
        self.memory('entrance')
        self.observe('exit', conflict=True)
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')

    def test_served_arm_timer_keeps_working_but_cannot_borrow_other_arm_evidence(self):
        self.memory('exit', since=970.)  # general memory is not old enough yet
        self.observe('exit')
        route = watch.VisitMemory.route_key(self.obj.zones[0])
        status = self.obj.barrier_workflow.status.return_value
        status.update(reminder_active=True, service={'routes': {route: {'reminder_active': True}}})
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')
        self.at(1000.1)
        self.memory('entrance', since=970., seen=1000.1)
        self.observe('exit', conflict=True)
        self.obj.barrier_workflow.status.return_value = status
        self.obj._update_alarm_audio()
        self.assertIsNone(self.obj.buzzer.want)

    def test_reminder_gates_resume_only_after_pause_alignment_and_ack_quiet_clear(self):
        for armed, aligned, quiet in itertools.product((False, True), (False, True), (False, True)):
            with self.subTest(armed=armed, aligned=aligned, quiet=quiet):
                self.memory('exit')
                self.obj.armed = armed
                self.obj._alignment_blocked = not aligned
                self.obj._ack_until = 1010. if quiet else 0.
                self.obj._update_alarm_audio()
                self.assertEqual(self.obj.buzzer.want, 'barrier_open' if armed and aligned and not quiet else None)
        self.at(1010.)
        self.observe('exit')
        self.obj._update_alarm_audio()
        self.assertEqual(self.obj.buzzer.want, 'barrier_open')

    def test_unavailable_tracker_still_allows_strict_up_without_changing_ack_rules(self):
        self.memory('exit', seen=990.)
        self.observe('exit', tracker=False)
        self.assertTrue(self.obj._left_up_due())
        self.assertEqual(self.obj.ack_quiet_s, 10.)
        self.assertEqual(self.obj.input_ack_min_s, 1.)

    def test_unknown_and_conflict_ack_interleavings_preserve_mutes_and_rearm(self):
        # Independent arms can lose/recover evidence in either order. Repeated ACKs
        # and gate changes must not erase the original arm's mute.
        for first, second in (('exit', 'entrance'), ('entrance', 'exit')):
            for operations in itertools.permutations(('unknown', 'conflict', 'other_ack')):
                with self.subTest(first=first, operations=operations):
                    self.at(1000.)
                    self.obj._arm_memory, self.obj._left_up_muted, self.obj._arm_live = {}, {}, {}
                    self.memory(first)
                    self.obj._update_alarm_audio()
                    self.assertTrue(self.obj.acknowledge())
                    for step, operation in enumerate(operations):
                        self.at(1010. + step * 10.)
                        if operation == 'other_ack':
                            self.memory(second, since=910., seen=self.clock.return_value)
                            self.obj._update_alarm_audio()
                            self.assertTrue(self.obj.acknowledge())
                        else:
                            self.observe(first, 'unknown' if operation == 'unknown' else 'up',
                                         conflict=operation == 'conflict')
                    self.at(1041.)
                    self.observe(first)
                    self.observe(second)
                    self.obj._update_alarm_audio()
                    self.assertTrue(self.obj._left_up_arms()[first]['muted'])
                    self.assertTrue(self.obj._left_up_arms()[second]['muted'])
                    self.assertIsNone(self.obj.buzzer.want)


class ProfilePersistenceTests(Fixture):
    def test_partial_profile_write_returns_actionable_failure_and_same_value_retry_repairs(self):
        views = self.views()
        write = zone_editor._atomic_write
        for failing in views:
            with self.subTest(failing=failing.parent.name):
                self.update({'barriers.reminder': True})
                def fail_one(path, content):
                    if Path(path) == failing:
                        raise OSError('synthetic profile fault')
                    return write(path, content)
                with mock.patch.object(zone_editor, '_atomic_write', side_effect=fail_one):
                    result = self.update({'barriers.reminder': False})
                self.assertIs(result['ok'], False)
                self.assertIs(result['partial'], True)
                self.assertIs(result['applied'], True)
                self.assertIn(failing.parent.name, result['error'])
                self.assertEqual([row['view'] for row in result['view_sync']['failed']], [failing.parent.name])
                self.assertFalse(json.loads(self.path.read_text())['barriers']['reminder'])
                self.assertTrue(json.loads(failing.read_text())['barriers']['reminder'])
                retry = self.update({'barriers.reminder': False})
                self.assertIs(retry['ok'], True)
                for profile in views:
                    self.assertFalse(json.loads(profile.read_text())['barriers']['reminder'])

    def test_view_switch_preserves_active_shared_choices_even_if_profile_is_stale(self):
        views = self.views()
        original = json.loads(views[1].read_text())
        active = json.loads(self.path.read_text())
        active['barriers'].update(reminder=False, reminder_after_s=75., reminder_every_s=20.)
        self.path.write_text(json.dumps(active), encoding='utf-8')
        # Represents a fresh process after a prior partial write: no in-memory queue.
        obj = watcher(self.path)
        with mock.patch.object(watch, 'STATE_DIR', self.root):
            obj._switch_view(views[1], 'Synthetic second view')
        result = json.loads(self.path.read_text())
        self.assertEqual({key: result['barriers'][key] for key in ('reminder', 'reminder_after_s', 'reminder_every_s')},
                         {'reminder': False, 'reminder_after_s': 75., 'reminder_every_s': 20.})
        for key in ('reminder', 'reminder_after_s', 'reminder_every_s'):
            result['barriers'][key] = original['barriers'][key]
        self.assertEqual(result, original, 'all nonshared profile settings and zones stay exact')
        self.assertTrue((self.root / 'restart-request').exists())

    def test_failed_active_save_does_not_change_other_profiles_or_runtime(self):
        views = self.views()
        before = {path: path.read_bytes() for path in [self.path, *views]}
        write = zone_editor._atomic_write
        def fail_active(path, content):
            if Path(path) == self.path:
                raise OSError('synthetic active fault')
            return write(path, content)
        with mock.patch.object(zone_editor, '_atomic_write', side_effect=fail_active):
            with self.assertRaises(OSError):
                self.update({'barriers.reminder': False})
        self.assertTrue(self.obj.barrier_rules['reminder'])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_snapshot_keeps_partial_retry_and_later_shared_save_repairs_all_choices(self):
        views = self.views()
        initial_revision = self.obj.state_dict()['view_sync']['revision']
        self.at(initial_revision + 1.)
        write = zone_editor._atomic_write
        def fail_one(path, content):
            if Path(path) == views[1]:
                raise OSError('synthetic delayed view')
            return write(path, content)
        with mock.patch.object(zone_editor, '_atomic_write', side_effect=fail_one):
            result = self.update({'barriers.reminder': False})
        snapshot = self.obj.settings_snapshot()
        self.assertEqual(snapshot['view_sync'], result['view_sync'])
        self.assertGreater(snapshot['view_sync']['revision'], initial_revision)
        self.assertEqual(self.obj.state_dict()['view_sync'], result['view_sync'])
        self.assertIs(snapshot['view_sync']['pending']['barriers.reminder'], False)
        self.at(initial_revision + 2.)
        repaired = self.update({'barriers.reminder_after_s': 75.})
        self.assertTrue(repaired['ok'])
        self.assertEqual(repaired['view_sync']['pending'], {})
        self.assertGreater(repaired['view_sync']['revision'], snapshot['view_sync']['revision'])
        self.assertEqual(self.obj.settings_snapshot()['view_sync']['pending'], {})
        saved = json.loads(views[1].read_text())['barriers']
        self.assertEqual((saved['reminder'], saved['reminder_after_s']), (False, 75.))

    def test_concurrent_view_change_defers_without_inverting_settings_and_state_locks(self):
        views = self.views()
        config = json.loads(self.path.read_text())
        config['zones'][0]['light'] = True
        self.path.write_text(json.dumps(config), encoding='utf-8')
        self.obj._invalidate_pose_writes = mock.Mock()
        state_held, settings_held, attempt_switch = threading.Event(), threading.Event(), threading.Event()
        write = zone_editor._atomic_write
        errors, deferred = [], []
        def delayed_active(path, content):
            if Path(path) == self.path:
                settings_held.set()
            return write(path, content)
        def save():
            try:
                self.update({'barriers.reminder': False, 'zone:Synthetic exit:light_delta': 18})
            except Exception as exc:
                errors.append(exc)
        def switch():
            try:
                # This is the real alignment caller's lock order. The save holds
                # settings while needing state to publish its changed zone.
                with self.obj._state_lock:
                    state_held.set()
                    if not attempt_switch.wait(2.):
                        raise RuntimeError('synthetic scheduling timeout')
                    try:
                        self.obj._switch_view(views[1], 'Synthetic second view')
                    except OSError as exc:
                        deferred.append(str(exc))
            except Exception as exc:
                errors.append(exc)
        with mock.patch.object(zone_editor, '_atomic_write', side_effect=delayed_active), \
                mock.patch.object(watch, 'STATE_DIR', self.root):
            saving = threading.Thread(target=save, daemon=True)
            switching = threading.Thread(target=switch, daemon=True)
            switching.start()
            try:
                self.assertTrue(state_held.wait(2.))
                saving.start()
                self.assertTrue(settings_held.wait(2.), repr(errors))
            finally:
                attempt_switch.set()
                switching.join(2.)
                if saving.ident is not None:
                    saving.join(2.)
        self.assertFalse(saving.is_alive())
        self.assertFalse(switching.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(deferred), 1)
        self.assertIn('retry', deferred[0])
        self.obj._invalidate_pose_writes.assert_not_called()
        self.assertFalse((self.root / 'restart-request').exists())
        self.assertEqual(json.loads(self.path.read_text())['view'], 'view-a')
        with mock.patch.object(watch, 'STATE_DIR', self.root), self.obj._state_lock:
            self.obj._switch_view(views[1], 'Synthetic second view')
        self.obj._invalidate_pose_writes.assert_called_once()
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['view'], 'view-b')
        self.assertIs(saved['barriers']['reminder'], False)


class AcceleratedTwoArmTests(Fixture):
    def test_ten_thousand_complete_two_arm_reminder_cycles(self):
        class FakeLoudness:
            kind = None
            def begin(self, kind, **kwargs):
                self.kind = kind
            def end(self):
                self.kind = None
            def evidence(self, **kwargs):
                pass
            def tick(self):
                pass

        self.obj.loudness = FakeLoudness()
        self.obj.log_event = lambda *args, **kwargs: None
        latest = {'trusted': True, 'reminder_active': False, 'barriers': []}
        self.obj.barrier_workflow = types.SimpleNamespace(status=lambda: latest, mute_reminders=lambda: [])

        def reading(arm, state, conflict=False):
            now = self.clock.return_value
            angle = 0. if conflict or state == 'down' else 90.
            self.obj._arm_live[arm] = {'ok': True, 'shown': angle, 'theta': angle,
                'position_conflict': conflict, 'at': now, 'at_wall': now + 1000.}
            latest['barriers'] = [{'id': arm, 'state': state, 'fresh': True,
                                  'calibrated': True, 'observed_at': now}]
            self.obj._note_arm_memory(latest)

        def sound(expected):
            self.obj._update_alarm_audio()
            self.assertEqual(self.obj.buzzer.want, 'barrier_open' if expected else None)

        # Avoid collecting hundreds of thousands of mock clock calls; this is a
        # fully accelerated clock and never sleeps or touches the real speaker.
        with mock.patch.object(watch.time, 'monotonic', new=lambda: self.clock.return_value), \
                mock.patch.object(watch.time, 'time', new=lambda: self.wall.return_value):
            for cycle in range(10_000):
                start = 1000. + cycle * 200.
                first, second = ('exit', 'entrance') if cycle % 2 == 0 else ('entrance', 'exit')
                self.at(start)
                for arm in (first, second):
                    reading(arm, 'down')
                self.at(start + 1.)
                reading(first, 'up')
                self.at(start + 10.)
                reading(second, 'up')
                self.at(start + 61.)
                reading(first, 'up')
                sound(True)
                self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
                self.at(start + 70.)
                reading(first, 'unknown')
                reading(second, 'up')
                sound(True)
                self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
                self.at(start + 71.)
                reading(first, 'up')
                sound(False)  # original mute survived another arm's acknowledgement
                self.assertTrue(self.obj._left_up_arms()[first]['muted'])
                self.at(start + 72.)
                reading(first, 'up', conflict=True)
                sound(False)
                self.at(start + 73.)
                reading(first, 'down')
                self.at(start + 74.)
                reading(first, 'up')
                self.at(start + 134.)
                reading(first, 'up')
                sound(True)  # a genuine down/up starts a new reminder period
                self.obj.barrier_rules['reminder'] = False
                sound(False)
                self.obj.barrier_rules['reminder'] = True
                self.obj.armed = False
                sound(False)
                self.obj.armed, self.obj._alignment_blocked = True, True
                sound(False)
                self.obj._alignment_blocked = False
                self.obj._ack_until = start + 144.
                self.at(start + 143.999)
                reading(first, 'up')
                sound(False)
                self.at(start + 144.)
                reading(first, 'up')
                sound(True)
                self.assertTrue(self.obj.acknowledge(None, via='input-any-key'))
                sound(False)


if __name__ == '__main__':
    unittest.main()
