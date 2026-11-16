"""An acknowledged fault must not leave the guard watching a blind system in silence."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from test_watch import watch, watcher
import settings_store


class FaultReminderTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = mock.Mock()
        self.obj.semantic.status.return_value = {'available': True, 'state': 'ready', 'age_s': .1}
        mock.patch.object(watch, 'notify').start()
        self.now = time.monotonic()
        self.obj.first_frame_seen, self.obj.last_frame_at = True, self.now + 1e6   # frames keep arriving
        self.obj._alignment_blocked, self.obj._alignment_bad_since = True, self.now - 3
        self.obj._alignment_used = ([[1., 0., 0.], [0., 1., 0.]], self.now - 10)   # it had locked before

    def tearDown(self):
        mock.patch.stopall()

    def faults(self):
        return [call for call in self.obj.log_event.call_args_list if call.args[0] == 'fault']

    def test_acknowledged_alignment_fault_reminds_until_detection_is_back(self):
        self.obj.check_health(self.now)
        self.assertEqual('fault', self.obj.alarm)
        first = self.faults()[-1]
        self.assertEqual('alignment', first.kwargs['evidence']['fault_kind'])
        self.assertIn('saved camera view (ANPR Entrance or PTZ 2) full-screen', first.args[1])
        self.obj.acknowledge()
        self.obj._ack_until = 0.     # the 10 s quiet period has long passed by then
        self.obj.check_health(self.now + 60)
        self.assertIsNone(self.obj.alarm, 'no reminder before the interval')
        self.obj.check_health(self.now + 121)
        self.assertEqual('fault', self.obj.alarm)
        self.assertIn('Still not watching after 2 min', self.faults()[-1].args[1])
        self.assertEqual(121., self.faults()[-1].kwargs['evidence']['blind_s'])
        self.obj.acknowledge()
        self.obj._alignment_blocked = False
        self.obj.check_health(self.now + 400)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(2, len(self.faults()))
        self.assertIsNone(self.obj._fault_since)

    def test_an_unanswered_fault_is_not_re_raised(self):
        self.obj.check_health(self.now)
        alarm_id = self.obj.alarm_id
        self.obj.check_health(self.now + 500)
        self.assertEqual(alarm_id, self.obj.alarm_id)
        self.assertEqual(1, len(self.faults()))

    def test_reminders_can_be_switched_off_and_never_sound_while_paused(self):
        self.obj.fault_remind_s = 0.
        self.obj.check_health(self.now)
        self.obj.acknowledge()
        self.obj._ack_until = 0.
        self.obj.check_health(self.now + 1000)
        self.assertIsNone(self.obj.alarm)
        self.obj.fault_remind_s = 120.
        self.obj.armed = False
        self.obj.check_health(self.now + 2000)
        self.assertIsNone(self.obj.alarm)

    def test_just_after_a_start_the_lock_gets_twelve_seconds(self):
        self.obj._alignment_used = None
        self.obj.check_health(self.now)                    # 3 s without a lock: not yet a fault
        self.assertIsNone(self.obj.alarm)
        self.obj._alignment_bad_since = self.now - 13
        self.obj.check_health(self.now)
        self.assertEqual('fault', self.obj.alarm)

    def test_setting_is_validated_and_loaded(self):
        item = settings_store._definition({}, 'alarms.fault_remind_s')
        self.assertEqual(0., settings_store._coerce(item, 0))
        self.assertEqual(180., settings_store._coerce(item, 180))
        with self.assertRaises(ValueError):
            settings_store._coerce(item, 30)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            for stored, expected in ((None, 120.), (0, 0.), (45, 60.), (5000, 900.), ('x', 120.)):
                path.write_text(json.dumps({'alarms': {} if stored is None else {'fault_remind_s': stored}}))
                self.assertEqual(expected, watch.Watcher._load_fault_remind(path), stored)


if __name__ == '__main__':
    unittest.main()
