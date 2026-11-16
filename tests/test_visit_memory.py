"""Observed traffic scenarios for the cross-channel acknowledgement policy."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from visit_memory import VisitMemory


EXIT = 'campus-exit'
ENTRANCE = 'campus-entrance'
BOX = [.15, .25, .3, .5]
OTHER = [.35, .2, .48, .4]


def zones(signal=False, other_signal=False):
    return [SimpleNamespace(name='Exit corner', route_group=EXIT, route_role='exit',
                            motion_active=signal, light_active=False),
            SimpleNamespace(name='Exit barrier', route_group=EXIT, route_role='exit',
                            motion_active=False, light_active=False),
            SimpleNamespace(name='Top road', route_group=ENTRANCE, route_role='entrance',
                            motion_active=False, light_active=other_signal),
            SimpleNamespace(name='Footpath', route_group='', route_role='pedestrian',
                            motion_active=False, light_active=False)]


def ai(track=1, bbox=None, route=EXIT, zone='Exit barrier', label='car'):
    return {'source': 'ai', 'track_id': track, 'bbox_normalized': bbox or BOX,
            'route_group': route, 'zone_name': zone, 'label': label,
            'route_role': 'pedestrian' if label == 'person' else 'exit',
            'direction_state': 'toward'}


def sighting(track=1, bbox=None, regions=None, label='car'):
    return {'track_id': track, 'label': label, 'bbox_normalized': bbox or BOX,
            'regions': regions if regions is not None else ['Exit barrier'],
            'confirmed_regions': regions if regions is not None else ['Exit barrier']}


def ml(*objects, **overrides):
    return {'available': True, 'state': 'ready', 'age_s': 0.,
            'detections': list(objects), **overrides}


def light(route=EXIT, zone='Exit corner', **extra):
    return {'source': 'light', 'route_group': route, 'zone_name': zone,
            'measurement': {'hit': True}, **extra}


class VisitMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = VisitMemory()

    def announce(self, details, now=0, alarm='a'):
        decision = self.memory.consider(details, now)
        self.assertFalse(decision.suppress)
        self.assertTrue(self.memory.emitted(decision.visit_id, alarm, now))
        return decision

    def test_same_car_through_multiple_regions_one_warning(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(12, zones(), ml(sighting()))
        same = self.memory.consider(ai(zone='Exit approach'), 12)
        self.assertEqual(first.visit_id, same.visit_id)
        self.assertTrue(same.suppress)
        self.assertEqual(same.reason, 'acknowledged visit')

    def test_warning_is_not_replayed_while_original_alarm_unacknowledged(self):
        first = self.announce(ai())
        second = self.memory.consider(ai(zone='Exit approach'), 20)
        self.assertTrue(second.suppress)
        self.assertEqual(first.visit_id, second.visit_id)

    def test_light_then_ai_after_five_seconds_inherits_ack(self):
        first = self.announce(light())
        self.memory.acknowledge('a', 1)
        for now in range(2, 15):
            self.memory.observe(now, zones(signal=True), ml())
        second = self.memory.consider(ai(), 15)
        self.assertTrue(second.suppress)
        self.assertEqual(first.visit_id, second.visit_id)
        self.assertTrue(self.memory.active_evidence(first.visit_id, 15)['confirmed'])

    def test_strict_motion_light_and_ai_share_one_visit(self):
        first = self.announce(light(source='motion', strict_change=True,
                                    measurement={'hit': False, 'weak_hit': True, 'confirmed': True}))
        self.memory.acknowledge('a', 1)
        later_light = self.memory.consider(light(zone='Exit approach'), 11)
        self.assertTrue(later_light.suppress)
        later_ai = self.memory.consider(ai(), 12)
        self.assertEqual(first.visit_id, later_ai.visit_id)
        self.assertTrue(later_ai.suppress)

    def test_separate_simultaneous_car_is_not_hidden_by_route_ack(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(12, zones(), ml(sighting(), sighting(2, OTHER)))
        second = self.memory.consider(ai(2, OTHER), 12)
        self.assertFalse(second.suppress)
        self.assertNotEqual(first.visit_id, second.visit_id)

    def test_overlapping_visible_cars_remain_distinct(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        overlapping = [.16, .26, .31, .51]
        self.memory.observe(12, zones(), ml(sighting(), sighting(2, overlapping)))
        second = self.memory.consider(ai(2, overlapping), 12)
        self.assertFalse(second.suppress)
        self.assertNotEqual(first.visit_id, second.visit_id)

    def test_provisional_warning_can_credit_only_first_confirmed_car(self):
        first = self.announce(light())
        self.memory.acknowledge('a', 1)
        self.memory.observe(11, zones(signal=True), ml(sighting(), sighting(2, OTHER)))
        self.assertTrue(self.memory.consider(ai(), 11).suppress)
        second = self.memory.consider(ai(2, OTHER), 11)
        self.assertFalse(second.suppress)
        self.assertNotEqual(first.visit_id, second.visit_id)

    def test_pending_new_car_survives_global_quiet_rejection(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(4, zones(), ml(sighting(), sighting(2, OTHER)))
        pending = self.memory.consider(ai(2, OTHER), 4)
        # Alarm owner rejects during its global quiet window; no emitted call.
        self.assertFalse(pending.suppress)
        self.memory.observe(12, zones(), ml(sighting(), sighting(2, OTHER)))
        retry = self.memory.consider(ai(2, OTHER), 12)
        self.assertFalse(retry.suppress)
        self.assertEqual(pending.visit_id, retry.visit_id)

    def test_stationary_acknowledged_car_has_no_fixed_timeout(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        for now in (10, 45, 120, 600, 3600):
            self.memory.observe(now, zones(), ml(sighting()))
            self.assertTrue(self.memory.consider(ai(), now).suppress)
        self.assertEqual(self.memory.status(3600)['acknowledged_visits'], 1)
        self.assertTrue(self.memory.active_evidence(first.visit_id, 3600)['present'])

    def test_five_second_detector_gap_does_not_end_visit(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        for now in (2, 3, 5, 7):
            self.memory.observe(now, zones(), ml())
        self.memory.observe(7.5, zones(), ml(sighting()))
        self.assertTrue(self.memory.consider(ai(), 7.5).suppress)

    def test_track_reidentification_after_blink_retains_ack(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(3, zones(), ml())
        self.memory.observe(5, zones(), ml(sighting(9, [.16, .25, .31, .5])))
        reacquired = self.memory.consider(ai(9, [.16, .25, .31, .5]), 5)
        self.assertTrue(reacquired.suppress)
        self.assertEqual(first.visit_id, reacquired.visit_id)

    def test_distinct_distant_new_track_does_not_inherit_ack(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(4, zones(), ml(sighting(2, OTHER)))
        self.assertFalse(self.memory.consider(ai(2, OTHER), 4).suppress)

    def test_pedestrian_and_car_do_not_reidentify_as_each_other(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(4, zones(), ml(sighting(2, label='person')))
        self.assertFalse(self.memory.consider(ai(2, label='person'), 4).suppress)

    def test_car_bus_label_flicker_can_reidentify_same_vehicle(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(4, zones(), ml(sighting(2, label='bus')))
        self.assertTrue(self.memory.consider(ai(2, label='bus'), 4).suppress)

    def test_leave_then_return_after_observed_clear_gets_new_visit(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml(sighting(regions=[])))
        for now in range(3, 9):
            self.memory.observe(now, zones(), ml(sighting(regions=[])))
        self.memory.observe(9, zones(), ml(sighting()))
        returned = self.memory.consider(ai(), 9)
        self.assertFalse(returned.suppress)
        self.assertNotEqual(first.visit_id, returned.visit_id)

    def test_signal_episode_rearms_only_after_clear_dwell(self):
        first = self.announce(light())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml())
        self.memory.observe(4, zones(signal=True), ml())
        self.assertTrue(self.memory.consider(light(), 4).suppress)
        self.memory.observe(5, zones(), ml())
        for now in range(6, 12):
            self.memory.observe(now, zones(), ml())
        next_visit = self.memory.consider(light(), 12)
        self.assertFalse(next_visit.suppress)
        self.assertNotEqual(first.visit_id, next_visit.visit_id)

    def test_ml_outage_never_proves_car_left(self):
        first = self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml())
        self.memory.observe(20, zones(), ml(available=False, state='retrying'))
        self.memory.observe(40, zones(), ml(available=False, state='retrying'))
        self.memory.observe(41, zones(), ml(sighting()))
        self.assertTrue(self.memory.consider(ai(), 41).suppress)
        self.assertEqual(self.memory.status(41)['active_visits'], 1)

    def test_stale_ai_does_not_prove_clear(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml(age_s=3))
        self.memory.observe(100, zones(), ml(age_s=50))
        self.assertTrue(self.memory.consider(ai(), 101).suppress)

    def test_camera_outage_does_not_prove_clear(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml(), frame_fresh=False)
        self.memory.observe(100, zones(), ml(), frame_fresh=False)
        self.assertTrue(self.memory.consider(ai(), 101).suppress)

    def test_worker_raw_detections_without_regions_do_not_prove_clear(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        raw = {'bbox': [96, 90, 192, 180], 'label': 'car', 'confidence': .8}
        self.memory.observe(2, zones(), ml(raw))
        self.memory.observe(100, zones(), ml(raw))
        self.assertTrue(self.memory.consider(ai(), 101).suppress)

    def test_parking_only_event_is_excluded(self):
        event = ai()
        event['route_role'] = 'parking'
        decision = self.memory.consider(event, 0)
        self.assertTrue(decision.suppress)
        self.assertIsNone(decision.visit_id)
        self.assertEqual(self.memory.status(0)['active_visits'], 0)

    def test_departing_vehicle_is_excluded(self):
        event = ai()
        event['direction_state'] = 'away'
        self.assertTrue(self.memory.consider(event, 0).suppress)

    def test_rejected_exposure_does_not_create_visit(self):
        for measurement in ({'hit': False}, {'exposure_only': True}, {'rejected': True}):
            self.assertTrue(self.memory.consider(light(measurement=measurement), 0).suppress)
        self.assertEqual(self.memory.status(0)['active_visits'], 0)

    def test_global_brightness_offset_is_not_itself_a_veto(self):
        self.assertFalse(self.memory.consider(light(measurement={'hit': True, 'global_offset': 10}), 0).suppress)

    def test_other_route_remains_independent(self):
        self.announce(light())
        self.memory.acknowledge('a', 1)
        self.assertFalse(self.memory.consider(light(route=ENTRANCE, zone='Top road'), 2).suppress)

    def test_evidence_becomes_confirmed_when_later_ai_matches_warning(self):
        first = self.announce(light())
        self.memory.observe(1, zones(signal=True), ml())
        initial = self.memory.evidence_for_alarm('a', 1)
        self.assertTrue(initial['present'])
        self.assertFalse(initial['confirmed'])
        self.memory.observe(2, zones(), ml(sighting()))
        self.memory.consider(ai(), 2)
        evidence = self.memory.evidence_for_alarm('a', 2)
        self.assertTrue(evidence['confirmed'])
        self.assertTrue(evidence['fresh'])
        self.assertEqual(evidence['visit_id'], first.visit_id)
        self.assertEqual(evidence['bbox_normalized'], BOX)

    def test_audio_evidence_distinguishes_clear_and_unknown(self):
        first = self.announce(ai())
        self.memory.observe(1, zones(), ml(sighting()))
        self.assertTrue(self.memory.active_evidence(first.visit_id, 1)['present'])
        self.memory.observe(5, zones(), ml())
        self.assertFalse(self.memory.active_evidence(first.visit_id, 5)['present'])
        self.memory.observe(6, zones(), ml(available=False))
        self.assertIsNone(self.memory.active_evidence(first.visit_id, 8)['present'])

    def test_evidence_cannot_stay_fresh_if_detector_stops_calling_observe(self):
        first = self.announce(ai())
        self.memory.observe(1, zones(signal=True), ml(sighting()))
        evidence = self.memory.active_evidence(first.visit_id, 20)
        self.assertFalse(evidence['fresh'])
        self.assertIsNone(evidence['present'])

    def test_dictionary_zones_and_pixel_bboxes_are_supported(self):
        self.announce(ai())
        dictionary_zones = [vars(zone) for zone in zones()]
        obj = sighting()
        del obj['bbox_normalized']
        obj['bbox'] = [96, 90, 192, 180]
        self.memory.observe(1, dictionary_zones, ml(obj))
        self.assertEqual(self.memory.evidence_for_alarm('a', 1)['bbox_normalized'], BOX)

    def test_stale_thread_timestamp_does_not_erase_recent_ack(self):
        self.announce(ai(), now=10)
        self.memory.acknowledge('a', 11)
        self.memory.observe(10.99, zones(), ml(sighting()))
        self.assertTrue(self.memory.consider(ai(), 12).suppress)

    def test_capacity_exhaustion_does_not_silence_new_object(self):
        memory = VisitMemory(max_visits=1)
        first = memory.consider(ai(), 0)
        memory.emitted(first.visit_id, 'a', 0)
        memory.observe(1, zones(), ml(sighting(), sighting(2, OTHER)))
        second = memory.consider(ai(2, OTHER), 1)
        self.assertFalse(second.suppress)
        self.assertIsNone(second.visit_id)

    def test_expired_visits_are_bounded_and_do_not_block_capacity(self):
        memory = VisitMemory(max_visits=1)
        first = memory.consider(ai(), 0)
        memory.emitted(first.visit_id, 'a', 0)
        memory.acknowledge('a', 0)
        memory.observe(1, zones(), ml())
        for now in range(2, 8):
            memory.observe(now, zones(), ml())
        second = memory.consider(ai(2, OTHER), 8)
        self.assertFalse(second.suppress)
        self.assertIsNotNone(second.visit_id)

    def test_unscoped_events_are_not_blanket_suppressed(self):
        decision = self.memory.consider({'source': 'fault'}, 0)
        self.assertFalse(decision.suppress)
        self.assertIsNone(decision.visit_id)

    def test_unobserved_time_does_not_count_as_continuously_clear(self):
        self.announce(ai())
        self.memory.acknowledge('a', 1)
        self.memory.observe(2, zones(), ml())
        self.memory.observe(50, zones(), ml())
        self.assertTrue(self.memory.consider(ai(), 51).suppress)

    def test_new_light_does_not_make_old_ai_confirmation_fresh(self):
        first = self.announce(ai())
        self.memory.observe(1, zones(), ml(sighting()))
        self.memory.observe(5, zones(signal=True), ml())
        self.memory.consider(light(), 5)
        evidence = self.memory.active_evidence(first.visit_id, 5)
        self.assertTrue(evidence['present'])
        self.assertTrue(evidence['recognized'])
        self.assertFalse(evidence['confirmed'])

    def test_below_confidence_or_departing_track_does_not_refresh_confirmation(self):
        first = self.announce(ai())
        obj = sighting()
        obj['confirmed_regions'] = []
        obj['region_status'] = {'Exit barrier': 'below confidence threshold'}
        self.memory.observe(5, zones(), ml(obj))
        evidence = self.memory.active_evidence(first.visit_id, 5)
        self.assertTrue(evidence['present'])
        self.assertFalse(evidence['confirmed'])

    def test_ai_confirmation_age_uses_observation_not_later_processing_time(self):
        event = {**ai(), 'observed_at': 1.}
        first = self.announce(event, now=3.5)
        self.memory.observe(3.5, zones(), ml(age_s=2.5))
        self.assertTrue(self.memory.active_evidence(first.visit_id, 3.5)['confirmed'])
        self.assertFalse(self.memory.active_evidence(first.visit_id, 4.1)['confirmed'])

    def test_stale_or_future_ai_candidate_cannot_create_fresh_visit(self):
        for observed in (1., 5., float('nan'), True):
            event = {**ai(), 'observed_at': observed}
            decision = self.memory.consider(event, 4.)
            self.assertTrue(decision.suppress)
            self.assertIsNone(decision.visit_id)
        self.assertEqual(0, self.memory.status(4.)['active_visits'])

    def test_retired_alarm_evidence_is_clear_then_unknown_if_feed_stops(self):
        first = self.announce(ai())
        for now in range(1, 8):
            self.memory.observe(now, zones(), ml())
        clear = self.memory.evidence_for_alarm('a', 7)
        self.assertTrue(clear['retired'])
        self.assertTrue(clear['fresh'])
        self.assertIs(clear['present'], False)
        self.memory.observe(8, zones(), ml(available=False))
        unknown = self.memory.evidence_for_alarm('a', 8)
        self.assertTrue(unknown['retired'])
        self.assertFalse(unknown['fresh'])
        self.assertIsNone(unknown['present'])

    def test_retired_unacknowledged_alarm_keeps_clear_evidence_until_replaced(self):
        first = self.announce(ai())
        for now in range(1, 80):
            self.memory.observe(now, zones(), ml())
        old = self.memory.evidence_for_alarm('a', 79)
        self.assertTrue(old['retired'])
        self.assertIs(old['present'], False)
        self.assertEqual(first.visit_id, old['visit_id'])
        second = self.memory.consider(ai(2, OTHER), 80)
        self.memory.emitted(second.visit_id, 'b', 80)
        self.memory.observe(81, zones(), ml(sighting(2, OTHER)))
        self.assertNotIn(first.visit_id, self.memory._visits)

    def test_acknowledging_retired_alarm_releases_pinned_history(self):
        first = self.announce(ai())
        for now in range(1, 80):
            self.memory.observe(now, zones(), ml())
        self.memory.acknowledge('a', 80)
        self.memory.observe(81, zones(), ml())
        self.assertNotIn(first.visit_id, self.memory._visits)

    def test_reset_invalidates_old_visits_without_reusing_ids(self):
        first = self.announce(ai())
        self.memory.reset()
        second = self.memory.consider(ai(), 1)
        self.assertFalse(second.suppress)
        self.assertNotEqual(first.visit_id, second.visit_id)

    def test_wrong_alarm_acknowledgement_does_not_consume_new_visit(self):
        self.announce(ai())
        self.assertEqual(self.memory.acknowledge('wrong', 1), 0)
        self.assertEqual(self.memory.status(1)['acknowledged_visits'], 0)

    def test_bad_numbers_do_not_enter_bbox_memory(self):
        first = self.announce(ai(bbox=[math.nan, .1, .2, .3]))
        self.assertIsNone(self.memory.active_evidence(first.visit_id, 0)['bbox_normalized'])
        for bad in (math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                self.memory.consider(ai(), bad)


if __name__ == '__main__':
    unittest.main()
