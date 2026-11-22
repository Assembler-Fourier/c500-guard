"""Entrance light fallback honors controls and wakes AI without changing acknowledgement rules."""
import unittest

import test_entrance_headlights as fixtures

lamp_pair = fixtures.lamp_pair


class HeadlightIntegrationTests(unittest.TestCase):
    setUp = fixtures.HeadlightAlarmTests.setUp
    deliver = fixtures.HeadlightAlarmTests.deliver

    def test_disabled_headlight_channel_does_not_alarm_or_wake_ai(self):
        self.lane.light = False
        self.deliver(self.road, 2)
        self.deliver(lamp_pair(self.road), 10)
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.lane.light_active)
        self.obj.semantic.boost.assert_not_called()
        self.assertTrue(self.lane.light_metrics['disabled'])

    def test_disabled_taillight_channel_does_not_alarm_or_wake_ai(self):
        self.lane.trigger = 'taillight'
        self.lane.light = False
        self.lane.route_role = 'exit'
        self.lane.route_group = 'campus-exit'
        self.deliver(self.road, 2)
        red = lamp_pair(self.road, value=(255., 170., 70.), r=.035)
        # This input is a valid tail-light trigger when enabled.
        self.deliver(red, 10)
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.lane.light_active)
        self.obj.semantic.boost.assert_not_called()
        self.assertFalse(getattr(self.obj, '_lane_seen', None))
        self.lane.light = True
        self.deliver(red, 5)
        self.assertEqual(self.obj.alarm, 'main')

    def test_valid_entrance_bloom_wakes_ai_before_confirmation(self):
        self.deliver(self.road, 2)
        self.obj.semantic.boost.assert_not_called()
        self.deliver(lamp_pair(self.road))
        self.obj.semantic.boost.assert_called_once_with(self.obj.AI_BOOST_S)
        self.assertIsNone(self.obj.alarm)
        self.deliver(lamp_pair(self.road), 4)
        self.assertEqual(self.obj.alarm, 'main')
        # Anonymous light is not new identity proof for automatic service.
        self.assertFalse(getattr(self.obj, '_lane_seen', None))
        self.assertNotIn('already_opening', self.obj.alarm_details)

    def test_rejected_flood_and_tiny_glint_do_not_wake_ai(self):
        self.deliver(self.road, 2)
        self.deliver(self.road * 0 + 252, 6)
        self.deliver(self.road, 2)
        self.deliver(lamp_pair(self.road, r=.003), 6)
        self.obj.semantic.boost.assert_not_called()
        self.assertIsNone(self.obj.alarm)

    def test_ai_boost_failure_does_not_drop_headlight_alarm(self):
        self.obj.semantic.boost.side_effect = RuntimeError('isolated AI wake failure')
        self.deliver(self.road, 2)
        self.deliver(lamp_pair(self.road), 5)
        self.assertEqual(self.obj.alarm, 'main')
        self.assertIsNotNone(self.lane._headlight_usual)


if __name__ == '__main__':
    unittest.main()
