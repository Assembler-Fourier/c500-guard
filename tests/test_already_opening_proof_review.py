"""Delayed arm service decisions must revalidate current evidence; fake outputs."""
import unittest
from unittest import mock

import test_exit_early as exit_cases
from test_watch import watch


class AlreadyOpeningProofTests(unittest.TestCase):
    def setUp(self):
        self.fixture = exit_cases.AlreadyOpeningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        wall = mock.patch.object(watch.time, 'time', return_value=1000.)
        wall.start(); self.addCleanup(wall.stop)
        self.obj = self.fixture.obj
        self.obj._lane_seen = {'exit':[497.,499.7]}
        self.obj._recent_rise = {'exit':499.}
        self.fixture.alarm()
        self.ident = self.obj.alarm_id
        self.visit_id = self.obj.alarm_details['visit_id']
        self.fixture.now = 501.3
        self.obj.last_frame_at = self.fixture.now
        self.obj._arm_live = {'exit':{'ok':True,'motion':'rising','motion_started':499.,
            'shown':45.,'at':self.fixture.now,'at_wall':1000.,'position_conflict':False}}
        acknowledge = mock.patch.object(self.obj.visits, 'acknowledge', wraps=self.obj.visits.acknowledge)
        self.acknowledge = acknowledge.start(); self.addCleanup(acknowledge.stop)

    def finish(self):
        self.obj._finish_already_opening()

    def assert_keeps_alarm(self):
        self.finish()
        self.assertEqual(self.ident,self.obj.alarm_id)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual('main',self.obj.buzzer.want)
        self.assertFalse(self.obj.acked)
        self.assertIsNone(self.obj._served_pending)
        self.acknowledge.assert_not_called()
        self.obj.loudness.end.assert_not_called()

    def test_fresh_same_rising_motion_preserves_valid_short_chirp(self):
        self.finish()
        self.assertIsNone(self.obj.alarm)
        self.acknowledge.assert_called_once_with(self.ident,501.3)

    def test_fresh_raised_position_after_same_rise_also_serves_visit(self):
        self.obj._arm_live['exit'].update(motion=None,shown=90.,theta=90.)
        self.finish()
        self.assertIsNone(self.obj.alarm)
        self.acknowledge.assert_called_once()

    def test_conflicting_raw_angle_cannot_be_hidden_by_smoothed_up_position(self):
        self.obj._arm_live['exit'].update(position_conflict=True,theta=0.,shown=90.)
        self.assert_keeps_alarm()

    def test_missing_or_stale_arm_reading_cannot_complete_delayed_service(self):
        self.obj._arm_live = {}
        self.assert_keeps_alarm()

    def test_pre_alarm_frame_processed_late_cannot_complete_service(self):
        self.obj._arm_live['exit']['at'] = 499.9
        self.assert_keeps_alarm()

    def test_future_arm_reading_cannot_complete_service(self):
        self.obj._arm_live['exit']['at'] = 501.4
        self.assert_keeps_alarm()

    def test_source_camera_staleness_cancels_chirp_without_acknowledging(self):
        self.obj.last_frame_at = 497.
        self.assert_keeps_alarm()

    def test_old_wall_clock_angle_cache_cannot_complete_service(self):
        self.obj._arm_live['exit']['at_wall'] = 998.4
        self.assert_keeps_alarm()

    def test_lowering_or_a_different_rising_movement_does_not_serve_this_proof(self):
        self.obj._arm_live['exit'].update(motion='lowering',shown=90.)
        self.assert_keeps_alarm()

    def test_new_rise_replaces_original_service_proof(self):
        self.obj._recent_rise['exit'] = 500.5
        self.assert_keeps_alarm()

    def test_competing_arrival_seen_during_chirp_vetoes_acknowledgement(self):
        self.obj._service_pending.add((self.ident,'campus-exit'))
        self.assert_keeps_alarm()

    def test_workflow_competing_arrival_veto_persists_between_ai_frames(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted':True,'service':{'routes':{
            'campus-exit':{'alarm_id':self.ident,'blocked_by_new_arrival':True}}}}
        self.assert_keeps_alarm()

    def test_current_service_identity_must_still_match_alarm(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted':True,'service':{'routes':{
            'campus-exit':{'alarm_id':'another-alarm'}}}}
        self.assert_keeps_alarm()

    def test_untrusted_current_service_cannot_complete_chirp(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted':False}
        self.assert_keeps_alarm()

    def test_same_lane_but_replaced_route_cannot_inherit_old_service(self):
        self.obj.alarm_details['route_group'] = 'another-exit'
        self.assert_keeps_alarm()

    def test_disabling_automatic_ack_during_chirp_keeps_alarm_sounding(self):
        self.obj.barrier_rules['ack_already_opening'] = False
        self.assert_keeps_alarm()

    def test_new_alignment_uncertainty_cancels_chirp(self):
        self.obj._alignment_blocked = True
        self.assert_keeps_alarm()

    def test_pending_camera_rearm_cancels_chirp(self):
        self.obj._pending_rearm = True
        self.assert_keeps_alarm()

    def test_retired_visit_cannot_be_acknowledged_by_old_pending_service(self):
        self.obj.visits._visits[self.visit_id].retired_at = 501.
        self.assert_keeps_alarm()

    def test_late_timer_cannot_replay_old_service_with_new_fresh_pixels(self):
        self.fixture.now = 505.
        self.obj.last_frame_at = 505.
        self.obj._arm_live['exit']['at'] = 505.
        self.assert_keeps_alarm()


if __name__ == '__main__':unittest.main()
