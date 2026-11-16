"""The Activity history stays honest.

* Alarms answered automatically (barrier up/down, road cleared) keep that answer after a restart
  (14 Nov 22:15: 11 barrier acknowledgements came back as "Not acknowledged").
* An alarm nobody answered before a newer one took over says so (22:08:47 ambulance -> 22:11:38 car).
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_watch import watch, watcher


class AlertHistoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)
        mock.patch.object(watch, 'ALERT_DIR', self.base / 'alerts').start()
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.obj.recent_alerts.clear()

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()

    def restore(self, records):
        log = self.base / 'events.jsonl'
        log.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        with mock.patch.object(watch, 'EVENT_LOG', log):
            self.obj._load_recent_alerts()
        return {a['id']: a['acked'] for a in self.obj.alerts()['alerts']}

    def test_automatic_answers_come_back_after_a_restart(self):
        def alarm(n):
            return {'ts': f'2026-11-14T21:0{n}:00', 'kind': 'main', 'detail': f'alarm {n}', 'alarm_id': f'a{n}'}
        acked = self.restore([
            alarm(1), {'ts': '2026-11-14T21:01:06', 'kind': 'auto-barrier-up', 'alarm_id': 'a1', 'after_s': 5.8,
                       'detail': 'Exit corner (arm raised)'},
            alarm(2), {'ts': '2026-11-14T21:02:03', 'kind': 'auto-barrier-down', 'alarm_id': 'a2', 'after_s': 3.0,
                       'detail': 'Exit corner (exit arm lowered; any-arm rule)'},
            alarm(3), {'ts': '2026-11-14T21:03:09', 'kind': 'auto-resolved', 'alarm_id': 'a3', 'after_s': 9.0,
                       'detail': 'Exit corner'},
            alarm(4), {'ts': '2026-11-14T21:06:00', 'kind': 'alarm-replaced', 'alarm_id': 'a4', 'after_s': 171.0,
                       'detail': 'not answered before the next alarm: alarm 4'},
            alarm(5), {'ts': '2026-11-14T21:05:06', 'kind': 'ack', 'alarm_id': 'a5', 'after_s': 6.3,
                       'detail': 'via input-mouse: alarm 5'},
            alarm(6),
        ])
        self.assertEqual({'via': 'barrier-raised', 'after_s': 5.8}, acked['a1'])
        self.assertEqual({'via': 'barrier-lowered', 'after_s': 3.0}, acked['a2'])
        self.assertEqual({'via': 'auto-road-clear', 'after_s': 9.0}, acked['a3'])
        self.assertEqual({'via': 'replaced', 'after_s': 171.0}, acked['a4'])
        self.assertEqual({'via': 'input-mouse', 'after_s': 6.3}, acked['a5'])
        self.assertIsNone(acked['a6'])

    def test_a_newer_alarm_marks_the_unanswered_one_replaced(self):
        self.assertTrue(self.obj.raise_alarm('main', 'ambulance leaving'))
        first = self.obj.alarm_id
        with mock.patch.object(self.obj.visits, 'evidence_for_alarm', return_value={'retired': True}):
            self.assertTrue(self.obj.raise_alarm('main', 'next car'))
        second = self.obj.alarm_id
        self.assertNotEqual(first, second)
        acked = {a['id']: a['acked'] for a in self.obj.alerts()['alerts']}
        self.assertEqual('replaced', acked[first]['via'])
        self.assertIsNone(acked[second])
        kinds = [call.args[0] for call in self.obj.log_event.call_args_list]
        self.assertIn('alarm-replaced', kinds)
        self.assertLess(kinds.index('alarm-replaced'), kinds.index('main', kinds.index('alarm-replaced')))
        replaced = next(c for c in self.obj.log_event.call_args_list if c.args[0] == 'alarm-replaced')
        self.assertEqual(first, replaced.kwargs['alarm_id'])

    def test_a_cleared_fault_replaced_by_an_alarm_says_so(self):
        self.obj.fault_reason = 'Scene alignment lost'
        self.assertTrue(self.obj.raise_alarm('fault', 'Scene alignment lost'))
        fault = self.obj.alarm_id
        self.obj.fault_reason = ''                     # the fault cleared by itself
        self.assertTrue(self.obj.raise_alarm('main', 'car at the exit'))
        acked = {a['id']: a['acked'] for a in self.obj.alerts()['alerts']}
        self.assertEqual('replaced', acked[fault]['via'])

    def test_an_answered_alarm_is_never_marked_replaced(self):
        self.assertTrue(self.obj.raise_alarm('main', 'first car'))
        first = self.obj.alarm_id
        self.assertTrue(self.obj.acknowledge(first, via='web'))
        self.obj._ack_until = 0
        self.assertTrue(self.obj.raise_alarm('main', 'second car'))
        acked = {a['id']: a['acked'] for a in self.obj.alerts()['alerts']}
        self.assertEqual('web', acked[first]['via'])
        self.assertNotIn('alarm-replaced', [call.args[0] for call in self.obj.log_event.call_args_list])

    def test_a_still_present_alarm_is_not_replaced_by_an_equal_one(self):
        self.assertTrue(self.obj.raise_alarm('main', 'first car'))
        first = self.obj.alarm_id
        with mock.patch.object(self.obj.visits, 'evidence_for_alarm', return_value={'fresh': True, 'present': True}):
            self.assertFalse(self.obj.raise_alarm('main', 'second car'))
        self.assertEqual(first, self.obj.alarm_id)
        self.assertNotIn('alarm-replaced', [call.args[0] for call in self.obj.log_event.call_args_list])


if __name__ == '__main__':
    unittest.main()
