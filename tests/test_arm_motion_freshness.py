"""A real current arm movement may resolve a visit; missing/stale motion may not."""
import time
import unittest
from unittest import mock

from test_watch import watcher


class ArmMotionFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj._alignment_blocked = False
        self.obj.alignment = {'state': 'tracking'}
        self.obj._pending_rearm = False
        self.obj.barrier_rules = {'ack_on_movement': True, 'ack_on_raise': True,
                                  'ack_on_lower': True, 'ack_any_arm': True}
        self.obj._arm_memory = {'exit': {'state': 'down'}}
        self.obj._arm_motion_acked = set()
        self.obj.alarm = 'main'
        self.obj.alarm_id = 'fresh-motion-test'
        self.obj.alarm_since = time.monotonic()-1.
        self.obj.alarm_reason = 'Test exit visit'
        self.obj.alarm_details = {'route_role': 'exit', 'route_group': 'campus-exit'}
        self.obj._mark_alert_acknowledged = mock.Mock()
        self.obj.loudness = mock.Mock()

    def reading(self, **updates):
        now = time.monotonic()
        return {'motion': 'rising', 'motion_started': now-.5, 'at': now,
                'ok': True, 'strict': None, **updates}

    def test_current_valid_movement_resolves_only_current_visit(self):
        before_quiet = self.obj._ack_until
        self.assertTrue(self.obj._ack_on_arm_motion('exit', self.reading()))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(before_quiet, self.obj._ack_until)
        self.obj._mark_alert_acknowledged.assert_called_once()

    def test_unreadable_frame_cannot_replay_previous_rising_event(self):
        self.assertFalse(self.obj._ack_on_arm_motion('exit', self.reading(ok=False, theta=None)))
        self.assertEqual('fresh-motion-test', self.obj.alarm_id)
        self.obj.loudness.end.assert_not_called()

    def test_even_recent_prealarm_motion_and_other_lane_cannot_acknowledge(self):
        self.assertFalse(self.obj._ack_on_arm_motion('exit', self.reading(
            motion_started=self.obj.alarm_since-.01)))
        self.obj._arm_memory['entrance'] = {'state': 'down'}
        self.assertFalse(self.obj._ack_on_arm_motion('entrance', self.reading()))
        self.assertEqual('fresh-motion-test', self.obj.alarm_id)

    def test_competing_visit_veto_survives_a_frame_without_an_ai_event(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'service': {'routes': {
            'campus-exit': {'alarm_id': self.obj.alarm_id, 'blocked_by_new_arrival': True}}}}
        self.obj._service_pending.clear()
        self.assertFalse(self.obj._ack_on_arm_motion('exit', self.reading()))
        self.assertEqual('fresh-motion-test', self.obj.alarm_id)

    def test_old_future_and_invalid_source_timestamps_cannot_acknowledge(self):
        now = time.monotonic()
        for updates in ({'at': now-2., 'motion_started': now-2.5},
                        {'at': now+1.}, {'at': None}, {'at': True}, {'at': float('nan')},
                        {'motion_started': float('nan')}, {'motion_started': True},
                        {'motion_started': now+1.}, {'motion_started': now-9.}):
            with self.subTest(updates=updates):
                self.assertFalse(self.obj._ack_on_arm_motion('exit', self.reading(**updates)))
                self.assertEqual('fresh-motion-test', self.obj.alarm_id)

    def test_alignment_loss_discards_angles_and_display_evidence(self):
        tracker = self.obj._arm_tracker = mock.Mock()
        self.obj._arm_live = {'exit': self.reading()}
        self.obj._observe_arm_angles(None, time.monotonic(), False, {})
        tracker.reset.assert_called_once()
        tracker.update.assert_not_called()
        self.assertEqual({}, self.obj._arm_live)

    def test_tracker_failure_discards_its_previous_movement(self):
        tracker = self.obj._arm_tracker = mock.Mock()
        tracker.update.side_effect = ValueError('unreadable frame')
        self.obj._arm_live = {'exit': self.reading()}
        self.obj._arm_last_run = 0.
        self.obj._observe_arm_angles(object(), time.monotonic(), True, {})
        tracker.reset.assert_called_once()
        self.assertEqual({}, self.obj._arm_live)

    def test_calibration_busy_cannot_bypass_workflow_trust_via_angle_tracker(self):
        now = time.monotonic()
        self.obj.last_frame_at = now
        self.obj._semantic_warmup_remaining = mock.Mock(return_value=0.)
        self.obj._local_frame_fresh = mock.Mock(return_value=True)
        self.obj.semantic = mock.Mock()
        self.obj.semantic.status.return_value = {'state': 'ready'}
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.observe.return_value = []
        self.obj.barrier_workflow.status.return_value = {
            'trusted': False, 'busy': True, 'barriers': [], 'service': {'routes': {}}}
        self.obj._arm_tracker = mock.Mock()
        self.obj._arm_tracker.update.return_value = {'exit': self.reading()}
        self.obj._arm_live = {'exit': self.reading()}
        self.obj._arm_last_run = 0.
        self.obj._ack_on_any_arm = mock.Mock()
        self.obj._note_arm_memory = mock.Mock()
        self.obj._observe_barrier_service(object(), now)
        self.obj._arm_tracker.update.assert_not_called()
        self.obj._arm_tracker.reset.assert_called_once()
        self.assertEqual({}, self.obj._arm_live)
        self.assertEqual('fresh-motion-test', self.obj.alarm_id)
        self.obj._mark_alert_acknowledged.assert_not_called()

    def test_verified_workflow_still_runs_current_angle_tracker(self):
        now = time.monotonic()
        self.obj._semantic_warmup_remaining = mock.Mock(return_value=0.)
        self.obj._local_frame_fresh = mock.Mock(return_value=True)
        self.obj.semantic = mock.Mock()
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.observe.return_value = []
        self.obj.barrier_workflow.status.return_value = {'trusted': True, 'barriers': []}
        self.obj._arm_tracker = mock.Mock()
        self.obj._arm_tracker.update.return_value = {}
        self.obj._arm_last_run = 0.
        self.obj._ack_on_any_arm = mock.Mock()
        self.obj._note_arm_memory = mock.Mock()
        frame = object()
        self.obj._observe_barrier_service(frame, now)
        self.obj._arm_tracker.update.assert_called_once_with(frame, now)


if __name__ == '__main__':
    unittest.main()
