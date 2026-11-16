"""Observed traffic sequences, visit delivery and region/class semantics."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from object_tracking import ObjectVisitTracker, TARGET_LABELS


def zone(**overrides):
    values = dict(name='Gate lane', rect=(.1, .1, .8, .8), ai_enabled=True,
                  motion=False, classes=list(TARGET_LABELS), severity='main')
    values.update(overrides)
    return SimpleNamespace(**values)


def detection(x=20, y=20, width=12, height=20, label='car'):
    return dict(label=label, confidence=.9, bbox=[x, y, x+width, y+height])


class VisitTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectVisitTracker()
        self.zones = [zone()]
        self.now = 100.
        self.car = detection()

    def step(self, objects, *, after=.5):
        self.now += after
        return self.tracker.update(objects, self.zones, 100, 100, self.now)

    def accept(self, events):
        for _, _, event_id in events:
            self.assertTrue(self.tracker.mark_emitted(event_id))

    def confirmed(self, objects=None):
        objects = [self.car] if objects is None else objects
        self.assertEqual([], self.step(objects)[0])
        events, observed = self.step(objects)
        self.assertTrue(events)
        return events, observed

    def test_two_fresh_observations_confirm_arrival(self):
        events, observed = self.confirmed()
        self.assertEqual('main', events[0][0])
        self.assertIn('Gate lane: AI confirmed car arrival', events[0][1])
        self.assertNotIn('object 1', events[0][1])
        self.assertEqual(['Gate lane'], observed[0]['confirmed_regions'])

    def test_confirmation_needs_consecutive_observations(self):
        self.step([self.car])
        self.step([])
        self.assertEqual([], self.step([self.car])[0])
        self.assertEqual(1, len(self.step([self.car])[0]))

    def test_stationary_vehicle_does_not_repeat_after_five_second_quiet(self):
        events, first = self.confirmed()
        self.accept(events)
        for _ in range(30):
            events, objects = self.step([self.car])
            self.assertEqual([], events)
            self.assertEqual(first[0]['track_id'], objects[0]['track_id'])

    def test_second_vehicle_alarms_while_first_remains(self):
        events, first = self.confirmed()
        self.accept(events)
        second = detection(x=60)
        self.assertEqual([], self.step([self.car, second])[0])
        events, observed = self.step([self.car, second])
        self.assertEqual(1, len(events))
        self.assertEqual(2, len({obj['track_id'] for obj in observed}))
        self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_two_arrivals_in_same_region_produce_two_events(self):
        events, observed = self.confirmed([self.car, detection(x=60)])
        self.assertEqual(2, len(events))
        self.assertEqual(2, len({event[2] for event in events}))

    def test_detections_reordering_does_not_swap_stationary_identities(self):
        second = detection(x=60)
        events, observed = self.confirmed([self.car, second])
        self.accept(events)
        by_x = {obj['bbox'][0]: obj['track_id'] for obj in observed}
        events, observed = self.step([second, self.car])
        self.assertEqual([], events)
        self.assertEqual(by_x, {obj['bbox'][0]: obj['track_id'] for obj in observed})

    def test_vehicle_label_flicker_does_not_create_another_visit(self):
        events, first = self.confirmed()
        self.accept(events)
        for label in ['truck', 'bus', 'car', 'truck', 'car']:
            events, observed = self.step([detection(label=label)])
            self.assertEqual([], events)
            self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_person_does_not_steal_car_identity_at_same_position(self):
        events, first = self.confirmed()
        self.accept(events)
        events, observed = self.step([detection(label='person')])
        self.assertEqual([], events)
        self.assertNotEqual(first[0]['track_id'], observed[0]['track_id'])
        self.assertEqual(1, len(self.step([detection(label='person')])[0]))

    def test_brief_detector_loss_retains_consumed_stationary_visit(self):
        events, first = self.confirmed()
        self.accept(events)
        for _ in range(7):
            self.assertEqual([], self.step([])[0])
        events, observed = self.step([self.car])
        self.assertEqual([], events)
        self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_long_departure_expires_identity_and_requires_fresh_pair(self):
        events, first = self.confirmed()
        self.accept(events)
        self.step([], after=5.01)
        events, observed = self.step([self.car])
        self.assertEqual([], events)
        self.assertNotEqual(first[0]['track_id'], observed[0]['track_id'])
        self.assertEqual(1, len(self.step([self.car])[0]))

    def test_visible_region_exit_then_return_is_a_new_visit(self):
        self.zones = [zone(rect=(.1, .1, .23, .8))]
        events, first = self.confirmed()
        self.accept(events)
        # Same tracked car moves gradually across the right-hand region edge.
        for x in [25, 30, 31, 32]:
            self.step([detection(x=x)])
        for x in [29, 25]:
            events, observed = self.step([detection(x=x)])
        self.assertEqual([], events)
        events, observed = self.step([self.car])
        self.assertEqual(1, len(events))
        self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_boundary_jitter_does_not_rearm_an_accepted_visit(self):
        self.zones = [zone(rect=(.1, .1, .23, .8))]
        events, _ = self.confirmed()
        self.accept(events)
        for x in [30, 20, 29, 30, 20, 31, 20]:
            self.assertEqual([], self.step([detection(x=x)])[0])

    def test_pending_event_retries_until_alarm_owner_accepts_it(self):
        events, _ = self.confirmed()
        first_id = events[0][2]
        for _ in range(12):  # A five-second quiet interval cannot lose the arrival.
            pending, _ = self.step([self.car])
            self.assertEqual(first_id, pending[0][2])
        self.accept(pending)
        self.assertEqual([], self.step([self.car])[0])

    def test_pending_second_vehicle_waits_while_first_alarm_is_active(self):
        events, _ = self.confirmed()
        self.accept(events)
        second = detection(x=60)
        self.step([self.car, second])
        pending, _ = self.step([self.car, second])
        self.assertEqual(1, len(pending))
        event_id = pending[0][2]
        for _ in range(12):
            pending, _ = self.step([self.car, second])
            self.assertEqual(event_id, pending[0][2])
        self.accept(pending)
        self.assertEqual([], self.step([self.car, second])[0])

    def test_pending_event_is_not_delivered_for_absent_object(self):
        events, _ = self.confirmed()
        self.assertEqual([], self.step([])[0])
        self.assertEqual([], self.step([], after=6)[0])
        self.assertFalse(self.tracker.mark_emitted(events[0][2]))

    def test_pending_event_is_not_delivered_after_object_leaves_region(self):
        self.zones = [zone(rect=(.1, .1, .23, .8))]
        events, _ = self.confirmed()
        for x in [30, 31, 32]:
            self.assertEqual([], self.step([detection(x=x)])[0])
        self.assertFalse(self.tracker.mark_emitted(events[0][2]))

    def test_reset_invalidates_delivery_id_and_needs_two_new_observations(self):
        events, _ = self.confirmed()
        old_id = events[0][2]
        self.tracker.reset()
        self.assertFalse(self.tracker.mark_emitted(old_id))
        events, _ = self.confirmed()
        self.assertNotEqual(old_id, events[0][2])

    def test_person_only_region_ignores_vehicles_even_if_motion_is_disabled(self):
        self.zones = [zone(classes=['person'], motion=False)]
        for _ in range(3):
            self.assertEqual([], self.step([self.car])[0])
        self.assertEqual([], self.step([detection(label='person')])[0])
        self.assertEqual(1, len(self.step([detection(label='person')])[0]))

    def test_motion_only_region_does_not_generate_ai_alerts(self):
        self.zones = [zone(ai_enabled=False, motion=True)]
        for _ in range(3):
            self.assertEqual([], self.step([self.car])[0])

    def test_each_region_has_an_independent_class_allowlist(self):
        self.zones = [zone(name='Walkway', classes=['person']),
                      zone(name='Gate lane', classes=['car', 'truck', 'bus'])]
        events, observed = self.confirmed([self.car, detection(x=60, label='person')])
        self.assertEqual(2, len(events))
        self.assertEqual(['Gate lane'], observed[0]['regions'])
        self.assertEqual(['Walkway'], observed[1]['regions'])

    def test_ground_point_keeps_car_parked_below_region_outside(self):
        # Box centre is in the traffic region; wheels are in adjacent parking.
        self.zones = [zone(rect=(.1, .1, .8, .35))]
        for _ in range(3):
            self.assertEqual([], self.step([detection(y=25, height=30)])[0])

    def test_exact_polygon_and_exclusion_predicate_is_used(self):
        # A triangular road region with a rectangular parking cutout.
        def contains(x, y):
            return .1 <= y <= .9 and .1 <= x <= y and not (.2 <= x <= .4 and y >= .6)
        self.zones = [zone(contains_point=contains)]
        excluded = detection(x=20, y=50, height=20)
        outside_polygon = detection(x=60, y=20)
        for _ in range(3):
            self.assertEqual([], self.step([excluded, outside_polygon])[0])
        self.assertEqual([], self.step([self.car])[0])
        self.assertEqual(1, len(self.step([self.car])[0]))

    def test_identity_is_resolution_independent(self):
        events, first = self.confirmed()
        self.accept(events)
        scaled = {**self.car, 'bbox': [value * 2 for value in self.car['bbox']]}
        events, observed = self.tracker.update([scaled], self.zones, 200, 200, self.now+.5)
        self.assertEqual([], events)
        self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_invalid_boxes_are_ignored_without_breaking_valid_detections(self):
        invalid = [{}, {'label': 'cat', 'bbox': [20, 20, 30, 40]},
                   {'label': 'car', 'bbox': [20, 20, math.nan, 40]},
                   {'label': 'car', 'bbox': [20, 20, 10, 40]},
                   {'label': 'car', 'bbox': [20, 20, 30]},
                   {'label': 'car', 'bbox': None}]
        events, observed = self.confirmed(invalid + [self.car])
        self.assertEqual(1, len(events))
        self.assertEqual(1, len(observed))

    def test_track_and_detection_work_is_bounded(self):
        tracker = ObjectVisitTracker(max_detections=4, max_tracks=3)
        many = [detection(x=10+index*15, width=4) for index in range(5)]
        _, observed = tracker.update(many, self.zones, 100, 100, 100)
        self.assertLessEqual(len(observed), 3)
        self.assertLessEqual(tracker.track_count, 3)

    def test_backwards_time_starts_a_new_generation(self):
        events, _ = self.confirmed()
        self.accept(events)
        self.assertEqual([], self.tracker.update([self.car], self.zones, 100, 100, 50)[0])

    def test_confirmation_count_and_confidence_are_independent_per_region(self):
        self.zones = [zone(name='Sensitive', ai_confirm_observations=2, ai_confidence=.60),
                      zone(name='Strict', ai_confirm_observations=3, ai_confidence=.80)]
        uncertain = {**self.car, 'confidence': .65}
        self.step([uncertain])
        events, observed = self.step([uncertain])
        self.assertEqual(1, len(events))
        self.assertIn('Sensitive', events[0][1])
        self.assertEqual('below confidence threshold', observed[0]['region_status']['Strict'])
        self.accept(events)
        for _ in range(2):
            self.assertEqual([], self.step([self.car])[0])
        events, _ = self.step([self.car])
        self.assertEqual(1, len(events))
        self.assertIn('in 3 observations', events[0][1])

    def test_low_confidence_dip_breaks_confirmation_but_retains_accepted_visit(self):
        self.zones = [zone(ai_confirm_observations=3, ai_confidence=.8)]
        low = {**self.car, 'confidence': .6}
        self.step([self.car])
        self.step([self.car])
        self.step([low])
        for _ in range(2):
            self.assertEqual([], self.step([self.car])[0])
        events, first = self.step([self.car])
        self.accept(events)
        for _ in range(10):
            self.assertEqual([], self.step([low])[0])
        events, last = self.step([self.car])
        self.assertEqual([], events)
        self.assertEqual(first[0]['track_id'], last[0]['track_id'])

    def test_configured_confidence_does_not_accept_missing_or_invalid_scores(self):
        self.zones = [zone(ai_confidence=.6)]
        missing = {key: value for key, value in self.car.items() if key != 'confidence'}
        for confidence in [math.nan, math.inf, -.1, 1.1, None, 'unknown']:
            for _ in range(3):
                self.assertEqual([], self.step([{**self.car, 'confidence': confidence}])[0])
        for _ in range(3):
            self.assertEqual([], self.step([missing])[0])

    def test_toward_motion_confirms_without_waiting_for_stationary_timeout(self):
        self.zones = [zone(arrival_direction=[1, -1], arrival_min_progress=.012)]
        self.step([detection(x=20, y=30)])
        events, objects = self.step([detection(x=22, y=28)])
        self.assertEqual(1, len(events))
        self.assertEqual('toward', objects[0]['direction']['Gate lane'])

    def test_departing_car_does_not_alert_even_when_it_stops(self):
        self.zones = [zone(arrival_direction=[1, 0], stationary_confirm_s=3.)]
        for x in [45, 42, 39, 36, 33, 30]:
            self.assertEqual([], self.step([detection(x=x)])[0])
        for _ in range(12):
            events, observed = self.step([detection(x=30)])
            self.assertEqual([], events)
        self.assertEqual('away', observed[0]['direction']['Gate lane'])

    def test_car_first_seen_stationary_gets_conservative_fallback(self):
        self.zones = [zone(arrival_direction=[1, 0], stationary_confirm_s=3.)]
        for _ in range(6):
            self.assertEqual([], self.step([self.car])[0])
        events, observed = self.step([self.car])
        self.assertEqual(1, len(events))
        self.assertEqual('stationary', observed[0]['direction']['Gate lane'])

    def test_sideways_motion_cannot_use_stationary_fallback(self):
        self.zones = [zone(arrival_direction=[1, 0], stationary_confirm_s=3.)]
        for y in [20, 23, 26, 29, 32, 35, 38, 41]:
            self.assertEqual([], self.step([detection(y=y)])[0])
        for _ in range(8):
            self.assertEqual([], self.step([detection(y=41)])[0])

    def test_bbox_size_changes_cannot_use_stationary_fallback(self):
        self.zones = [zone(arrival_direction=[1, 0], stationary_confirm_s=3.)]
        self.step([self.car])
        changing = detection(x=17, width=18)  # Same ground point, changing scale.
        for _ in range(10):
            self.assertEqual([], self.step([changing])[0])

    def test_route_group_emits_one_event_for_overlapping_regions(self):
        self.zones = [zone(name='Entry', route_group='west'),
                      zone(name='Barrier', route_group='west')]
        events, _ = self.confirmed()
        self.assertEqual(1, len(events))
        context = self.tracker.event_context(events[0][2], self.zones)
        self.assertEqual('Entry', context['zone_name'])
        self.assertEqual('west', context['route_group'])
        self.assertEqual(1, context['track_id'])
        self.assertEqual([.2,.2,.32,.4], context['bbox_normalized'])
        self.assertEqual(.9, context['confidence'])
        self.assertEqual('car', context['label'])
        self.assertEqual('unrestricted', context['direction_state'])
        self.assertEqual(2, context['observations'])
        self.assertEqual(self.now, context['observed_at'])
        self.accept(events)
        self.assertIsNone(self.tracker.event_context(events[0][2], self.zones))
        self.assertEqual([], self.step([self.car])[0])
        self.assertFalse(self.tracker.mark_emitted(events[0][2]))

    def test_one_route_visit_stays_consumed_across_adjacent_regions(self):
        self.zones = [zone(name='Entry', rect=(.1,.1,.30,.8), route_group='west'),
                      zone(name='Barrier', rect=(.35,.1,.40,.8), route_group='west')]
        events, first = self.confirmed()
        self.accept(events)
        for x in [25, 30, 35, 40, 45, 50, 50, 50]:
            events, observed = self.step([detection(x=x)])
            self.assertEqual([], events)
            self.assertEqual(first[0]['track_id'], observed[0]['track_id'])
        self.assertEqual('already alerted', observed[0]['region_status']['Barrier'])

    def test_new_vehicle_in_same_route_has_its_own_pending_event(self):
        self.zones = [zone(name='Entry', route_group='west'),
                      zone(name='Barrier', route_group='west')]
        events, _ = self.confirmed()
        self.accept(events)
        second = detection(x=60)
        self.step([self.car, second])
        events, _ = self.step([self.car, second])
        self.assertEqual(1, len(events))
        context = self.tracker.event_context(events[0][2], self.zones)
        self.assertEqual(2, context['track_id'])
        for _ in range(12):
            pending, _ = self.step([self.car, second])
            self.assertEqual(events[0][2], pending[0][2])

    def test_three_visible_observations_outside_whole_route_allow_return_visit(self):
        self.zones = [zone(rect=(.1,.1,.23,.8), route_group='west')]
        events, first = self.confirmed()
        self.accept(events)
        for x in [25, 30, 31, 32, 29, 25]:
            self.step([detection(x=x)])
        events, observed = self.step([self.car])
        self.assertEqual(1, len(events))
        self.assertEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_short_frame_interval_cannot_teleport_an_accepted_identity(self):
        events, first = self.confirmed()
        self.accept(events)
        events, observed = self.step([detection(x=33)], after=.01)
        self.assertEqual([], events)
        self.assertNotEqual(first[0]['track_id'], observed[0]['track_id'])

    def test_motion_prediction_preserves_two_vehicles_crossing(self):
        # Keep each car's velocity when the model lists them in another order.
        _, observed = self.step([detection(x=20, width=8), detection(x=50, width=8)])
        left, right = observed[0]['track_id'], observed[1]['track_id']
        self.step([detection(x=30, width=8), detection(x=40, width=8)])
        _, observed = self.step([detection(x=30, width=8), detection(x=40, width=8)])
        by_x = {item['bbox'][0]: item['track_id'] for item in observed}
        self.assertEqual(left, by_x[40])
        self.assertEqual(right, by_x[30])


if __name__ == '__main__':
    unittest.main()
