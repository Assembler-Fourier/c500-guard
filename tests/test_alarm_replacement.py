"""Resolved incidents cannot silently block a later vehicle needing attention."""
import unittest
from unittest import mock

from test_watch import watch, watcher


class AlarmReplacementTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.obj.zones = []
        self.obj.first_frame_seen = True
        self.clock = mock.patch.object(watch.time, 'monotonic', return_value=100.).start()
        mock.patch.object(watch, 'notify').start()

    def tearDown(self):
        mock.patch.stopall()

    def details(self, track=1):
        return {'source': 'ai', 'zone_name': 'exit', 'route_group': 'campus-exit',
                'route_role': 'exit', 'track_id': track, 'label': 'car',
                'bbox_normalized': [.1 + .3*(track-1), .3, .3 + .3*(track-1), .6],
                'observed_at': self.clock.return_value}

    def observe(self, tracks=(), *, age=0.):
        now = self.clock.return_value
        self.obj.last_frame_at = now
        objects = [{**self.details(track), 'regions': ['exit'],
                    'confirmed_regions': ['exit']} for track in tracks]
        self.obj.visits.observe(now, [{'name': 'exit', 'route_group': 'campus-exit',
                                      'route_role': 'exit'}],
            {'available': True, 'state': 'ready', 'age_s': age, 'detections': objects},
            frame_fresh=True)

    def start_lost_fault(self):
        self.obj.last_frame_at = self.clock.return_value - 1.
        self.obj.check_health(self.clock.return_value)
        self.assertEqual(self.obj.alarm, 'fault')
        return self.obj.alarm_id

    def test_new_vehicle_replaces_recovered_unacknowledged_fault(self):
        fault_id = self.start_lost_fault()
        self.clock.return_value = 101.
        self.observe([1])
        self.obj.check_health(101.)
        self.assertEqual(self.obj.fault_reason, '')
        self.assertTrue(self.obj.raise_alarm('main', 'new car', details=self.details()))
        self.assertEqual(self.obj.alarm, 'main')
        self.assertEqual(self.obj.buzzer.want, 'main')
        self.assertNotEqual(self.obj.alarm_id, fault_id)

    def test_ongoing_fault_retains_priority_over_vehicle(self):
        fault_id = self.start_lost_fault()
        self.clock.return_value = 101.
        self.obj.check_health(101.)
        self.assertTrue(self.obj.fault_reason)
        self.assertFalse(self.obj.raise_alarm('main', 'new car', details=self.details()))
        self.assertEqual(self.obj.alarm, 'fault')
        self.assertEqual(self.obj.alarm_id, fault_id)

    def test_acknowledged_fault_quiet_applies_after_recovery(self):
        self.start_lost_fault()
        self.assertTrue(self.obj.acknowledge())
        self.clock.return_value = 103.
        self.observe([1])
        self.obj.check_health(103.)
        self.assertFalse(self.obj.raise_alarm('main', 'car during quiet', details=self.details()))
        self.assertIsNone(self.obj.alarm)
        self.clock.return_value = 110.
        self.observe([1])
        self.assertTrue(self.obj.raise_alarm('main', 'car still waiting', details=self.details()))
        self.assertEqual(self.obj.alarm, 'main')

    def test_second_vehicle_stays_eligible_while_first_is_still_present(self):
        self.observe([1])
        self.assertTrue(self.obj.raise_alarm('main', 'first car', details=self.details()))
        first_id = self.obj.alarm_id
        self.clock.return_value = 101.
        self.observe([1, 2])
        self.assertFalse(self.obj.raise_alarm('main', 'second car', details=self.details(2)))
        self.assertEqual(self.obj.alarm_id, first_id)
        self.clock.return_value = 102.
        self.assertTrue(self.obj.acknowledge(first_id))
        self.clock.return_value = 112.
        self.observe([1, 2])
        self.assertTrue(self.obj.raise_alarm('main', 'second car still waiting', details=self.details(2)))
        self.assertNotEqual(self.obj.alarm_id, first_id)
        self.assertEqual(self.obj.alarm_details['track_id'], 2)

    def test_departed_vehicle_does_not_block_a_new_car(self):
        self.observe([1])
        self.obj.raise_alarm('main', 'first car', details=self.details())
        first_id = self.obj.alarm_id
        self.clock.return_value = 104.
        self.observe([2])
        current = self.obj.visits.evidence_for_alarm(first_id, 104.)
        self.assertTrue(current['fresh'])
        self.assertIs(current['present'], False)
        self.assertTrue(self.obj.raise_alarm('main', 'second car', details=self.details(2)))
        self.assertNotEqual(self.obj.alarm_id, first_id)

    def test_unknown_old_visit_presence_does_not_authorize_replacement(self):
        self.observe([1])
        self.obj.raise_alarm('main', 'first car', details=self.details())
        first_id = self.obj.alarm_id
        self.clock.return_value = 104.
        self.observe([], age=5.)
        current = self.obj.visits.evidence_for_alarm(first_id, 104.)
        self.assertFalse(current['fresh'])
        self.assertIsNone(current['present'])
        self.assertFalse(self.obj.raise_alarm('main', 'second car', details=self.details(2)))
        self.assertEqual(self.obj.alarm_id, first_id)


if __name__ == '__main__':
    unittest.main()
