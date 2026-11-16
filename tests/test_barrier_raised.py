"""Raising a lane's arm during its alarm stops the siren; a key dismisses the left-up knock."""
import time
import unittest
from unittest import mock

from test_watch import watch, watcher
from visit_memory import VisitMemory


class RaisedArmTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.armed = True
        self.obj._pending_rearm = False
        self.obj._alignment_blocked = False
        self.obj.alignment = {**self.obj.alignment, 'state': 'tracking'}
        self.obj.alarm, self.obj.alarm_id = 'main', 'visit:1'
        self.obj.alarm_since = time.monotonic() - 4
        self.obj.alarm_reason = 'Exit approach: car arriving'
        self.obj.alarm_details = {'zone_name': 'Exit approach', 'route_group': 'campus-exit',
                                  'route_role': 'exit', 'source': 'ai'}
        self.route = VisitMemory.route_key(self.obj.alarm_details)
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted': True}
        self.obj.loudness = mock.Mock(kind='main')
        fresh = mock.patch.object(self.obj, '_local_frame_fresh', return_value=True)
        fresh.start()
        self.addCleanup(fresh.stop)

    def intent(self, **change):
        return {'type': 'raised', 'alarm_id': 'visit:1', 'route': self.route,
                'alarm_started_at': self.obj.alarm_since,
                'origin_observed_at': self.obj.alarm_since + 1,
                'position_since': self.obj.alarm_since + 2,
                'observed_at': time.monotonic(), **change}

    def test_old_or_unbound_transition_never_silences_new_alarm(self):
        for change in ({'origin_observed_at': self.obj.alarm_since - .1},
                       {'position_since': self.obj.alarm_since - .1},
                       {'alarm_started_at': self.obj.alarm_since - 1},
                       {'origin_observed_at': None}, {'position_since': float('nan')},
                       {'position_since': True}):
            with self.subTest(change=change):
                self.assertFalse(self.obj._silence_for_raised_barrier(self.intent(**change)))
                self.assertEqual('main', self.obj.alarm)

    def test_known_older_motion_or_distinct_pending_car_blocks_auto_ack(self):
        self.obj._arm_live = {'exit': {'motion': 'lowering', 'motion_started': self.obj.alarm_since-.2}}
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))
        self.obj._arm_live = {}
        self.obj._service_pending.add(('visit:1', self.route))
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))

    def test_raised_arm_stops_the_siren_for_this_visit_only(self):
        self.assertTrue(self.obj._silence_for_raised_barrier(self.intent()))
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.obj.loudness.end.assert_called_once()
        self.assertLess(self.obj._ack_until, time.monotonic())        # no global quiet: other lane still alerts
        kinds = [call.args[0] for call in self.obj.log_event.call_args_list]
        self.assertIn('auto-barrier-up', kinds)

    def test_other_lane_old_alarm_or_unverified_view_does_not_silence(self):
        for change in ({'route': 'campus-entrance|entrance'}, {'alarm_id': 'visit:0'}):
            with self.subTest(change=change):
                self.assertFalse(self.obj._silence_for_raised_barrier(self.intent(**change)))
                self.assertEqual('main', self.obj.alarm)
        self.obj.alignment = {**self.obj.alignment, 'state': 'holding'}
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))
        self.obj.alignment = {**self.obj.alignment, 'state': 'tracking'}
        self.obj.barrier_workflow.status.return_value = {'trusted': False}
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))
        self.assertEqual('main', self.obj.alarm)

    def test_disarmed_or_fault_is_never_silenced_by_an_arm(self):
        self.obj.alarm = 'fault'
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))
        self.obj.alarm, self.obj.armed = 'main', False
        self.assertFalse(self.obj._silence_for_raised_barrier(self.intent()))


class LeftUpReminderTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.alarm = None
        self.obj.loudness = mock.Mock(kind='barrier_open')
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.mute_reminders.return_value = ['campus-exit|exit']
        self.obj._reminder_seq, self.obj._reminder_since = 3, time.monotonic() - 2

    def test_key_press_sees_and_dismisses_the_knock(self):
        status = self.obj.input_status()
        self.assertEqual('reminder:3', status['alarm_id'])
        self.assertIsNotNone(status['alarm_started_at'])
        self.assertTrue(self.obj.acknowledge('reminder:3', via='input-any-key'))
        self.obj.barrier_workflow.mute_reminders.assert_called_once()
        self.obj.loudness.end.assert_called_once()
        self.assertIsNone(self.obj.buzzer.want)

    def test_stale_reminder_id_is_ignored(self):
        self.assertFalse(self.obj.acknowledge('reminder:2', via='input-any-key'))
        self.obj.barrier_workflow.mute_reminders.assert_not_called()

    def test_nothing_to_dismiss_when_quiet(self):
        self.obj.loudness = mock.Mock(kind=None)
        self.assertIsNone(self.obj.input_status()['alarm_id'])
        self.assertFalse(self.obj.acknowledge(None, via='api'))


if __name__ == '__main__':
    unittest.main()
