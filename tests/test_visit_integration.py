"""Cross-channel visit handling through the real watcher and semantic tracker.

All frames and model outputs are synthetic. No worker, speaker, camera or live
HTTP endpoint is started. Detector outputs carry real tracker identities and
region visits rather than pre-decided fake event IDs.
"""
import unittest
from unittest import mock

import test_approach_runtime as approach_runtime


class VisitIntegrationTests(unittest.TestCase):
    setUp = approach_runtime.ApproachRuntimeTests.setUp
    deliver = approach_runtime.ApproachRuntimeTests.deliver
    light_alarm = approach_runtime.ApproachRuntimeTests.light_alarm

    def test_late_ai_after_twelve_seconds_inherits_first_light_ack(self):
        self.light_alarm()
        first_visit = self.obj.alarm_details['visit_id']
        self.assertTrue(self.obj.acknowledge())
        for _ in range(60):
            self.deliver(frame=self.changed)
        with mock.patch.object(self.semantic, 'mark_emitted', wraps=self.semantic.mark_emitted) as mark:
            self.deliver([self.first], frame=self.changed)
            self.deliver([self.first], frame=self.changed)
            mark.assert_called_once()
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        visits = self.obj.visits.status(self.now)
        self.assertEqual(1, visits['active_visits'])
        self.assertEqual(first_visit, visits['routes'][0]['visit_id'])
        self.assertTrue(visits['routes'][0]['acknowledged'])
        self.assertTrue(visits['routes'][0]['confirmed'])

    def test_unacknowledged_light_gets_ai_evidence_without_restarting_sound(self):
        self.light_alarm()
        alarm_id = self.obj.alarm_id
        alarm_since = self.obj.alarm_since
        for _ in range(40):
            self.deliver(frame=self.changed)
        with mock.patch.object(self.obj.loudness, 'begin', wraps=self.obj.loudness.begin) as begin:
            self.deliver([self.first], frame=self.changed)
            self.deliver([self.first], frame=self.changed)
            begin.assert_not_called()
        self.assertEqual(alarm_id, self.obj.alarm_id)
        self.assertEqual(alarm_since, self.obj.alarm_since)
        self.assertEqual('ai', self.obj.alarm_details['source'])
        self.assertEqual(1, self.obj._alarm_sequence)
        evidence = self.obj.visits.evidence_for_alarm(alarm_id, self.now)
        self.assertTrue(evidence['fresh'])
        self.assertTrue(evidence['present'])
        self.assertTrue(evidence['confirmed'])

    def test_second_car_is_not_consumed_during_global_quiet_then_sounds(self):
        self.light_alarm()
        self.obj.acknowledge()
        deadline = self.obj._ack_until
        with mock.patch.object(self.semantic, 'mark_emitted', wraps=self.semantic.mark_emitted) as mark:
            self.deliver([self.first, self.second], frame=self.changed)
            self.deliver([self.first, self.second], frame=self.changed)
            self.assertEqual(1, mark.call_count, 'Only the first car inherits the light warning')
            while self.now + .2 < deadline:
                self.deliver([self.first, self.second], frame=self.changed)
                self.assertIsNone(self.obj.alarm)
            self.assertEqual(1, mark.call_count, 'The second car must remain pending during quiet')
            self.deliver([self.first, self.second], after=deadline-self.now+.001, frame=self.changed)
            self.assertEqual(2, mark.call_count)
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, self.obj._alarm_sequence)
        self.assertEqual(2, self.obj.alarm_details['track_id'])

    def test_new_car_that_leaves_during_quiet_is_not_replayed(self):
        self.light_alarm()
        self.obj.acknowledge()
        self.deliver([self.first, self.second], frame=self.changed)
        self.deliver([self.first, self.second], frame=self.changed)
        for _ in range(60):
            self.deliver([self.first], frame=self.changed)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertEqual(1, self.semantic._tracker.track_count)

    def test_tracker_loss_and_nearby_reidentification_keeps_same_ack(self):
        self.deliver([self.first])
        self.deliver([self.first])
        self.assertEqual('main', self.obj.alarm)
        old_track = self.obj.alarm_details['track_id']
        old_visit = self.obj.alarm_details['visit_id']
        self.obj.acknowledge()
        # More than tracker.lost_after, less than VisitMemory.clear_dwell_s.
        for _ in range(26):
            self.deliver()
        self.deliver([self.first])
        self.deliver([self.first])
        detections = self.semantic.status()['detections']
        self.assertNotEqual(old_track, detections[0]['track_id'])
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        active = self.obj.visits.status(self.now)['routes']
        self.assertEqual(1, len(active))
        self.assertEqual(old_visit, active[0]['visit_id'])
        self.assertTrue(active[0]['acknowledged'])

    def test_fresh_clear_ends_visit_and_a_return_gets_new_alarm(self):
        self.deliver([self.first])
        self.deliver([self.first])
        original = self.obj.alarm_details['visit_id']
        self.obj.acknowledge()
        for _ in range(60):
            self.deliver()
        self.assertEqual(0, self.obj.visits.status(self.now)['active_visits'])
        self.deliver([self.first])
        self.deliver([self.first])
        self.assertEqual('main', self.obj.alarm)
        self.assertNotEqual(original, self.obj.alarm_details['visit_id'])
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_model_gap_does_not_make_an_acknowledged_car_new(self):
        self.deliver([self.first])
        self.deliver([self.first])
        self.obj.acknowledge()
        self.semantic._status.update(available=False, state='retrying', updated_at=None)
        for _ in range(60):
            self.deliver(ai_ready=False)
        self.assertEqual(1, self.obj.visits.status(self.now)['acknowledged_visits'])
        # After a long occlusion, geometry cannot safely establish identity. A
        # current distinct track may alert; the outage itself never means clear.
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_motion_and_lights_on_an_acknowledged_car_do_not_create_new_incidents(self):
        self.deliver([self.first])
        self.deliver([self.first])
        self.obj.acknowledge()
        for _ in range(60):
            self.deliver([self.first], frame=self.changed)
        self.assertTrue(self.zone._notified['light'])
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_new_car_replaces_old_unacknowledged_incident_after_observed_clear(self):
        self.deliver([self.first])
        self.deliver([self.first])
        old_id = self.obj.alarm_id
        for _ in range(40):
            self.deliver()
        self.assertEqual(old_id, self.obj.alarm_id, 'Old incident can remain visible until acknowledged')
        self.assertTrue(self.obj.visits.evidence_for_alarm(old_id, self.now)['retired'])
        self.deliver([self.second])
        self.deliver([self.second])
        self.assertEqual('main', self.obj.alarm)
        self.assertNotEqual(old_id, self.obj.alarm_id)
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_retired_unacknowledged_incident_does_not_block_traffic_minutes_later(self):
        self.deliver([self.first])
        self.deliver([self.first])
        old_id = self.obj.alarm_id
        for _ in range(80):
            self.deliver(after=1.)
        self.assertTrue(self.obj.visits.evidence_for_alarm(old_id, self.now)['retired'])
        self.deliver([self.second])
        self.deliver([self.second])
        self.assertNotEqual(old_id, self.obj.alarm_id)
        self.assertEqual(2, self.obj._alarm_sequence)


if __name__ == '__main__':
    unittest.main()
