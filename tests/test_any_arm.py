"""Only the matching barrier moving after the alarm can acknowledge it; long glare keeps the verified pose."""
import time
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher
from test_scene_alignment import textured_scene


def status(exit_state='down', entrance_state='down', trusted=True):
    stamp = time.monotonic()
    return {'trusted': trusted, 'barriers': [
        {'id': 'exit', 'state': exit_state, 'fresh': True, 'calibrated': True,
         'position_since': stamp, 'observed_at': stamp},
        {'id': 'entrance', 'state': entrance_state, 'fresh': True, 'calibrated': True,
         'position_since': stamp, 'observed_at': stamp}]}


class AnyArmTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.obj.armed = True
        self.obj._alignment_blocked = False
        self.obj.alignment = {'state': 'tracking'}
        mock.patch.object(watch, 'notify').start()
        mock.patch.object(self.obj, '_local_frame_fresh', return_value=True).start()
        self.obj._ack_on_any_arm(status(), time.monotonic())          # both arms seen down before the alarm
        self.obj.raise_alarm('main', 'Exit approach: localized light change +24.4; approach warning',
                             details={'source': 'light', 'zone_name': 'Exit approach', 'route_group': 'campus-exit',
                                      'route_role': 'exit'})
        self.assertEqual('main', self.obj.alarm)

    def tearDown(self):
        mock.patch.stopall()

    def kinds(self):
        return [call.args[0] for call in self.obj.log_event.call_args_list]

    def test_other_lanes_arm_cannot_acknowledge_the_exit_alarm(self):
        self.obj._ack_on_any_arm(status(), time.monotonic())
        self.obj._ack_on_any_arm(status(entrance_state='unknown'), time.monotonic())
        self.assertFalse(self.obj._ack_on_any_arm(status(entrance_state='up'), time.monotonic()))
        self.assertEqual('main', self.obj.alarm)

    def test_matching_arm_observed_down_after_alarm_then_up_acknowledges(self):
        self.obj._ack_on_any_arm(status(), time.monotonic())
        self.assertTrue(self.obj._ack_on_any_arm(status(exit_state='up'), time.monotonic()))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual('auto-barrier-up', self.kinds()[-1])
        self.assertEqual('exit', self.obj.log_event.call_args.kwargs['arm'])

    def test_previous_car_lowering_cannot_acknowledge_a_new_alarm(self):
        self.obj._arm_last_known['exit'] = ('up', self.obj.alarm_since - .1)
        self.assertFalse(self.obj._ack_on_any_arm(status(), time.monotonic()))
        self.assertEqual('main', self.obj.alarm)

    def test_processing_time_cannot_refresh_a_prealarm_source_image(self):
        self.obj._arm_last_known.clear()
        old = status()
        for row in old['barriers']:
            row['observed_at'] = self.obj.alarm_since-.2
        self.obj._ack_on_any_arm(old, time.monotonic())
        self.assertFalse(self.obj._ack_on_any_arm(status(exit_state='up'), time.monotonic()))
        self.assertEqual('main', self.obj.alarm)

    def test_latched_competing_visit_blocks_settled_arm_between_ai_frames(self):
        self.obj._ack_on_any_arm(status(), time.monotonic())
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'service': {'routes': {
            'campus-exit': {'alarm_id': self.obj.alarm_id, 'blocked_by_new_arrival': True}}}}
        self.obj._service_pending.clear()
        self.assertFalse(self.obj._ack_on_any_arm(status(exit_state='up'), time.monotonic()))
        self.assertEqual('main', self.obj.alarm)

    def test_matching_arm_lowering_with_current_origin_acknowledges(self):
        self.obj._arm_last_known['exit'] = ('up', self.obj.alarm_since)
        self.assertTrue(self.obj._ack_on_any_arm(status(), time.monotonic()))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual('auto-barrier-down', self.kinds()[-1])

    def test_rule_switches_untrusted_readings_and_faults_are_respected(self):
        self.obj.barrier_rules['ack_any_arm'] = False
        self.obj._ack_on_any_arm(status(entrance_state='up'), time.monotonic())
        self.assertEqual('main', self.obj.alarm)
        self.obj.barrier_rules['ack_any_arm'] = True
        self.obj.barrier_rules['ack_on_lower'] = False
        self.obj._ack_on_any_arm(status(entrance_state='down'), time.monotonic())   # lowering switched off
        self.assertEqual('main', self.obj.alarm)
        self.obj._ack_on_any_arm(status(entrance_state='up', trusted=False), time.monotonic())
        self.assertEqual('main', self.obj.alarm)

    def test_an_arm_that_changed_long_before_the_alarm_does_not_count(self):
        self.obj._arm_last_known['entrance'] = ('down', self.obj.alarm_since - 20)
        self.obj._ack_on_any_arm({'trusted': True, 'barriers': [
            {'id': 'entrance', 'state': 'up', 'fresh': True, 'calibrated': True, 'position_since': time.monotonic()}]}, time.monotonic())
        self.assertEqual('main', self.obj.alarm)

    def test_fault_alarms_are_never_silenced_by_an_arm(self):
        self.obj.alarm = 'fault'
        self.obj._ack_on_any_arm(status(entrance_state='up'), time.monotonic())
        self.assertEqual('fault', self.obj.alarm)


class GlareHoldTests(unittest.TestCase):
    def test_structure_keeps_the_pose_after_the_hold_and_refuses_a_moved_view(self):
        obj = watcher()
        scene = textured_scene(seed=11)
        obj._alignment_reference = scene
        obj._alignment_used = (np.array([[1., 0., 0.], [0., 1., 0.]]), 0.)
        bright = np.clip(scene.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        self.assertTrue(obj._structure_still_matches(bright, 100.))
        moved = np.roll(scene, 7, axis=1)
        self.assertFalse(obj._structure_still_matches(moved, 102.))
        self.assertFalse(obj._structure_still_matches(np.zeros_like(scene), 104.))


if __name__ == '__main__':
    unittest.main()
