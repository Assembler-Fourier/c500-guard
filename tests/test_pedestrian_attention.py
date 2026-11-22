"""Pedestrian attention is latched; ordinary vehicle/evidence rules stay intact."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from test_watch import watch, watcher
import alarm_audio


PERSON = dict(source='ai', label='person', route_role='pedestrian',
              route_group='campus-pedestrian', zone_name='Pedestrian crossing',
              track_id=8, confidence=.72, observations=2)


class PedestrianAttentionTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.now = 100.
        self.clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now).start()
        mock.patch.object(watch, 'notify').start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(alarm_audio.reset_configuration)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.writes = []
        self.obj.loudness = watch.AlarmLoudness(
            types.SimpleNamespace(plan=lambda **_: types.SimpleNamespace(state='present')),
            read_fn=lambda: dict(available=True, volume=.25, muted=True),
            apply_fn=lambda **kw: self.writes.append(kw) or dict(applied=True),
            restore_path=Path(self.temp.name)/'restore.json', clock=lambda: self.now)

    def start(self):
        self.assertTrue(self.obj.raise_alarm('main', 'Pedestrian crossing: person', details=PERSON))

    def test_confirmed_person_starts_strong_and_keeps_sound_after_departure(self):
        self.start()
        self.assertTrue(self.obj.alarm_details['manual_ack_required'])
        self.assertEqual(self.obj.loudness.plan.profile, 'notice')
        self.assertAlmostEqual(self.writes[0]['volume'], .60)
        self.assertFalse(self.writes[0]['muted'])
        ident = self.obj.alarm_id
        self.obj.visits.evidence_for_alarm = mock.Mock(return_value={
            'fresh': False, 'present': False, 'confirmed': False, 'retired': True})
        for elapsed in (1, 2, 4, 40, 600):
            self.now = 100.+elapsed
            self.obj._update_alarm_audio()
            self.assertEqual(self.obj.alarm_id, ident)
            self.assertTrue(self.obj.loudness.plan.audible)
            self.assertLessEqual(self.obj.loudness.plan.level, .95)
            self.assertFalse(self.obj.loudness.event_present, 'Do not invent continued visibility')
        self.assertEqual(self.obj.loudness.plan.profile, 'wake')
        self.assertAlmostEqual(self.obj.loudness.plan.level, .95)

    def test_person_takes_priority_over_vehicle_then_cannot_be_replaced(self):
        self.assertTrue(self.obj.raise_alarm('main', 'Vehicle', details={
            'source':'ai', 'label':'car', 'route_role':'exit', 'track_id':1}))
        self.start()
        ident = self.obj.alarm_id
        self.now += 5
        self.obj.visits.evidence_for_alarm = mock.Mock(return_value={'retired':True})
        for kind in ('main', 'early', 'fault'):
            self.assertFalse(self.obj.raise_alarm(kind, 'Other event', details={
                'source':'ai', 'label':'car', 'route_role':'exit', 'track_id':2}))
            self.assertEqual(self.obj.alarm_id, ident)
        self.assertIsNone(self.obj._alarm_lane(), 'Barrier movement cannot service a pedestrian')
        self.assertFalse(self.obj._arm_matches_alarm('exit'))
        self.assertFalse(self.obj._arm_matches_alarm('entrance'))

    def test_confirmed_person_takes_audible_priority_over_fault_without_hiding_health(self):
        self.obj.fault_reason = 'Capture interrupted'
        self.assertTrue(self.obj.raise_alarm('fault', self.obj.fault_reason))
        self.start()
        self.assertEqual(self.obj.fault_reason, 'Capture interrupted')
        self.assertTrue(self.obj.alarm_details['manual_ack_required'])

    def test_ack_keeps_one_second_input_guard_and_exact_ten_second_quiet(self):
        self.start()
        ident = self.obj.alarm_id
        self.now = 100.99
        self.assertFalse(self.obj.acknowledge(ident, via='input-any-key'))
        self.now = 101.
        self.assertTrue(self.obj.acknowledge(ident, via='input-mouse'))
        self.assertIsNone(self.obj.buzzer.want)
        self.assertFalse(self.obj.loudness.pedestrian_latched)
        self.assertEqual(self.writes[-1], {'volume':.25, 'muted':True})
        for instant in (101.1, 105., 110.99):
            self.now = instant
            self.assertFalse(self.obj.raise_alarm('main', 'Another person', details={**PERSON,'track_id':9}))
        self.assertEqual(self.obj._ack_until, 111.)
        self.now = 111.
        self.assertTrue(self.obj.raise_alarm('main', 'New exit car', details={
            'source':'ai','label':'car','route_role':'exit','track_id':99}))

    def test_off_cancels_latch_and_restores_speaker(self):
        self.start()
        self.obj.set_armed(False)
        self.assertFalse(self.obj.armed)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertFalse(self.obj.loudness.pedestrian_latched)
        self.assertEqual(self.writes[-1], {'volume':.25, 'muted':True})

    def test_unconfirmed_wrong_class_and_vehicle_lane_do_not_latch(self):
        for changes in ({'source':'light'}, {'label':'motorcycle'}, {'route_role':'exit'}):
            with self.subTest(changes=changes):
                obj = watcher()
                obj.raise_alarm('main', 'Not a confirmed pedestrian route',
                                details={**PERSON, **changes, 'manual_ack_required':True})
                self.assertNotIn('manual_ack_required', obj.alarm_details)

    def test_latched_plan_respects_ack_off_quiet_and_configured_ceiling(self):
        for changes in ({'enabled':False}, {'acknowledged':True}, {'quiet_remaining_s':10}):
            self.assertFalse(alarm_audio.plan(600,pedestrian_latched=True,**changes).audible)
        alarm_audio.configure({'confirmed':{'rise_s':8,'level':.77}})
        values=[alarm_audio.plan(t/10,pedestrian_latched=True,evidence_fresh=False,event_present=False).level for t in range(61)]
        self.assertEqual(values,sorted(values))
        self.assertEqual(values[0],.60)
        self.assertAlmostEqual(values[40],.77)
        self.assertLessEqual(max(values),.77)

    def test_mixer_failure_does_not_unlock_uncontrolled_strong_audio(self):
        self.obj.loudness._read=lambda: {'available':False}
        self.start()
        self.assertLessEqual(self.obj.loudness.plan.level,.12)
        self.assertNotEqual(self.obj.loudness.plan.profile,'wake')
        self.now += 30
        self.obj._update_alarm_audio()
        self.assertFalse(self.obj.loudness.plan.audible)
        self.assertEqual(self.writes,[])


if __name__ == '__main__':
    unittest.main()
