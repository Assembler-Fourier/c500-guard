"""Expose the reason for accepted silence without recording keyboard contents."""
import unittest
from unittest import mock

from test_watch import watcher


class AckDiagnosticsTests(unittest.TestCase):
    def test_only_an_accepted_acknowledgement_updates_the_receipt(self):
        obj = watcher()
        obj._queue_snapshot = mock.Mock(return_value=None)
        obj._remember_alert = mock.Mock()
        obj._mark_alert_acknowledged = mock.Mock()
        obj.raise_alarm('main', 'synthetic car', details={'source': 'ai'})
        ident = obj.alarm_id
        self.assertFalse(obj.acknowledge('wrong-event', via='web'))
        self.assertIsNone(obj.last_ack)
        obj.input_ack_min_s = 0
        self.assertTrue(obj.acknowledge(ident, via='input-any-key'))
        receipt = obj.state_dict()['last_ack']
        self.assertEqual({'via', 'at', 'alarm_id'}, set(receipt))
        self.assertEqual('input-any-key', receipt['via'])
        self.assertEqual(ident, receipt['alarm_id'])
        self.assertRegex(receipt['at'], r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')
        self.assertFalse(obj.acknowledge(ident, via='input-mouse'))
        self.assertEqual(receipt, obj.state_dict()['last_ack'])


if __name__ == '__main__':
    unittest.main()
