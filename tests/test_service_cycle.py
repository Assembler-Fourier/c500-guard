import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from service_cycle import ServiceCycleMonitor


EXIT = 'campus-exit'
ENTRY = 'campus-entrance'
ALARM = {'id': 'event-a', 'route': EXIT}


def observation(stamp, occupancy='occupied', barrier='down', **extra):
    return {'barrier': barrier, 'barrier_observed_at': stamp,
            'occupancy': occupancy, 'occupancy_observed_at': stamp,
            'model_ready': True, 'new_arrival': False, **extra}


class ServiceCycleTests(unittest.TestCase):
    def setUp(self):
        self.monitor = ServiceCycleMonitor()

    def update(self, stamp, occupancy='occupied', barrier='down', **extra):
        return self.monitor.update({EXIT: observation(stamp, occupancy, barrier, **extra)},
                                   ALARM, now=stamp)

    def test_verified_open_close_cycle_and_clear_targets_original_event_until_applied(self):
        self.assertEqual([], self.update(0))
        self.assertEqual(['raised'], [a['type'] for a in self.update(1, barrier='up')])
        self.assertEqual(['lowered'], [a['type'] for a in self.update(2, 'clear')])
        self.assertEqual([], self.update(3, 'clear'))
        actions = self.update(4, 'clear')
        self.assertEqual([{'type': 'resolve', 'alarm_id': 'event-a', 'route': EXIT,
                          'observed_at': 4, 'reason': 'verified_service_cycle',
                          'alarm_started_at': 0, 'occupied_at': 1,
                          'clear_since': 2, 'barrier_observed_at': 4, 'barrier_state': 'down',
                          'clear_s': 2., 'clear_observations': 3}], actions)
        retry = self.update(5, 'clear')
        self.assertEqual('event-a', retry[0]['alarm_id'])
        self.assertEqual([], self.monitor.update({EXIT: observation(6, 'clear')}, None, 6))

    def test_clear_only_fallback_requires_original_occupied_observation(self):
        for stamp in (0, 1, 2, 3):
            self.assertEqual([], self.update(stamp, 'clear'))
        self.update(4, 'occupied')
        self.update(5, 'clear')
        self.update(6, 'clear')
        self.assertEqual('verified_road_clear', self.update(7, 'clear')[0]['reason'])

    def test_unknown_barrier_cannot_complete_even_fresh_clear_road(self):
        self.update(0)
        for stamp in (1, 2, 3, 4):
            self.assertEqual([], self.update(stamp, 'clear', 'unknown'))
        self.assertEqual([], self.update(5, 'clear'))
        self.assertEqual([], self.update(6, 'clear'))
        self.assertEqual('resolve', self.update(7, 'clear')[0]['type'])

    def test_car_can_depart_while_witnessed_barrier_remains_up(self):
        self.update(0)
        self.update(1, 'occupied', 'up')
        self.update(2, 'clear', 'up')
        self.update(3, 'clear', 'up')
        action = self.update(4, 'clear', 'up')[0]
        self.assertEqual('verified_road_clear', action['reason'])
        self.assertTrue(self.monitor.status(4)['routes'][EXIT]['up_seen_for_event'])

    def test_up_result_before_alarm_cannot_be_borrowed_by_later_event(self):
        self.monitor.update({EXIT: observation(0, barrier='up')}, None, 0)
        self.monitor.update({EXIT: observation(.1, barrier='up', barrier_observed_at=0)}, ALARM, .1)
        for stamp in (.2, 1.2, 2.2):
            obs = observation(stamp, 'clear', 'up', barrier_observed_at=0)
            self.assertEqual([], self.monitor.update({EXIT: obs}, ALARM, stamp))
        self.assertFalse(self.monitor.status(2.2)['routes'][EXIT]['up_seen_for_event'])

    def test_open_close_alone_never_resolves_occupied_road(self):
        for stamp, barrier in enumerate(('down', 'up', 'down', 'down', 'down')):
            self.assertEqual([], [a for a in self.update(stamp, 'occupied', barrier)
                                  if a['type'] not in ('raised', 'lowered')])

    def test_occupied_interrupts_clear_dwell(self):
        self.update(0)
        self.update(1, 'clear')
        self.update(2, 'occupied')
        self.update(3, 'clear')
        self.assertEqual([], self.update(4, 'clear'))
        self.assertEqual('resolve', self.update(5, 'clear')[0]['type'])

    def test_unknown_and_model_not_ready_interrupt_clear_dwell(self):
        for extra in ({'occupancy': 'unknown'}, {'model_ready': False}, {'model_ready': 'ready'}):
            with self.subTest(extra=extra):
                self.setUp()
                self.update(0)
                self.update(1, 'clear')
                self.monitor.update({EXIT: {**observation(2, 'clear'), **extra}}, ALARM, 2)
                self.update(3, 'clear')
                self.assertEqual([], self.update(4, 'clear'))
                self.assertEqual('resolve', self.update(5, 'clear')[0]['type'])

    def test_duplicate_clear_inference_cannot_accumulate_elapsed_dwell(self):
        self.update(0)
        clear = {EXIT: observation(1, 'clear')}
        for now in (1, 1.5, 2, 2.5, 3):
            self.assertEqual([], self.monitor.update(clear, ALARM, now))
        self.assertEqual(0., self.monitor.status(3)['routes'][EXIT]['clear_s'])
        self.assertEqual('resolve', self.update(3.1, 'clear')[0]['type'])

    def test_source_gap_of_three_seconds_breaks_clear_continuity(self):
        self.update(0)
        self.update(1, 'clear')
        self.assertEqual([], self.update(4, 'clear'))
        self.assertEqual([], self.update(5, 'clear'))
        self.assertEqual('resolve', self.update(6, 'clear')[0]['type'])

    def test_invalid_stale_future_nonfinite_and_boolean_source_never_clear(self):
        for bad in (None, -10, 5, math.nan, math.inf, True, '2'):
            with self.subTest(stamp=bad):
                self.setUp()
                self.update(0)
                self.update(1, 'clear')
                invalid = observation(2, 'clear', occupancy_observed_at=bad)
                self.assertEqual([], self.monitor.update({EXIT: invalid}, ALARM, 2))
                self.assertEqual('unknown', self.monitor.status(2)['routes'][EXIT]['occupancy'])
                self.assertEqual([], self.update(3, 'clear'))
                self.assertEqual([], self.update(4, 'clear'))

    def test_competing_new_arrival_blocks_old_event_even_after_road_clears(self):
        self.update(0)
        self.update(1, 'clear')
        self.assertEqual([], self.update(2, new_arrival=True))
        for stamp in (3, 4, 5, 6):
            self.assertEqual([], self.update(stamp, 'clear'))
        self.assertTrue(self.monitor.status(6)['routes'][EXIT]['blocked_by_new_arrival'])

    def test_initial_arrival_pulse_belongs_to_initial_alarm(self):
        self.update(0, new_arrival=True)
        self.update(1, 'clear')
        self.update(2, 'clear')
        self.assertEqual('resolve', self.update(3, 'clear')[0]['type'])

    def test_other_route_activity_does_not_clear_or_block_original_route(self):
        for stamp in range(4):
            actions = self.monitor.update({
                EXIT: observation(stamp, 'occupied' if stamp == 0 else 'clear'),
                ENTRY: observation(stamp, 'occupied', new_arrival=True)}, ALARM, stamp)
        self.assertEqual(EXIT, actions[0]['route'])
        self.assertEqual('event-a', actions[0]['alarm_id'])

    def test_replacement_alarm_on_same_route_cannot_inherit_old_clear_or_cycle(self):
        self.update(0)
        self.update(1, barrier='up')
        self.update(2, 'clear')
        other = {'id': 'event-b', 'route': EXIT}
        for stamp in (3, 4, 5):
            self.assertEqual([], self.monitor.update({EXIT: observation(stamp, 'clear')}, other, stamp))
        self.assertEqual('event-b', self.monitor.status(5)['routes'][EXIT]['alarm_id'])

    def test_replacement_alarm_on_other_route_and_removed_alarm_are_never_resolved(self):
        self.update(0)
        self.update(1, 'clear')
        for stamp, current in ((2, {'id': 'event-b', 'route': ENTRY}), (3, None)):
            self.assertEqual([], self.monitor.update({EXIT: observation(stamp, 'clear')}, current, stamp))
            self.assertIsNone(self.monitor.status(stamp)['routes'][EXIT]['alarm_id'])

    def test_missing_route_invalidates_clear_and_up_continuity(self):
        self.update(0, barrier='up')
        self.update(1, 'clear', 'up')
        self.assertEqual([], self.monitor.update({}, ALARM, 2))
        status = self.monitor.status(2)['routes'][EXIT]
        self.assertEqual('unknown', status['barrier'])
        self.assertEqual('unknown', status['occupancy'])
        self.assertEqual(0., status['clear_s'])
        self.assertEqual([], self.update(3, 'clear', 'up'))

    def raise_for_alarm(self, at=0):
        """Alarm on the exit lane with its arm down, then the guard raises it."""
        self.assertEqual([], self.monitor.update({EXIT: observation(at, barrier='down')}, ALARM, at))
        actions = self.monitor.update({EXIT: observation(at + 1, barrier='up')}, ALARM, at + 1)
        self.monitor.update({EXIT: observation(at + 2, barrier='up')}, None, at + 2)   # alarm silenced
        return actions

    def test_raising_the_arm_during_its_alarm_is_reported_once(self):
        actions = self.raise_for_alarm()
        self.assertEqual([{'type': 'raised', 'alarm_id': 'event-a', 'observed_at': 1, 'route': EXIT,
                           'origin_observed_at': 0, 'alarm_started_at': 0, 'position_since': 1}], actions)
        self.assertTrue(self.monitor.status(2)['routes'][EXIT]['raised_for_visit'])

    def test_arm_already_up_when_the_alarm_starts_is_not_an_answer(self):
        # Arms held open all evening: a raised arm cannot mean "I am dealing with it".
        for stamp in range(6):
            actions = self.monitor.update({EXIT: observation(stamp, barrier='up')}, ALARM, stamp)
            self.assertNotIn('raised', [action['type'] for action in actions])

    def test_other_lane_arm_does_not_answer_this_alarm(self):
        self.monitor.update({ENTRY: observation(0, barrier='down')}, ALARM, 0)
        actions = self.monitor.update({ENTRY: observation(1, barrier='up')}, ALARM, 1)
        self.assertEqual([], actions)

    def unknown(self, stamp, alarm=ALARM):
        return self.monitor.update({EXIT: observation(stamp, barrier='unknown')}, alarm, stamp)

    def test_raise_is_seen_through_the_unknown_readings_of_a_moving_arm(self):
        self.assertEqual([], self.update(0))                 # arm down when the alarm starts
        for stamp in (1, 2, 3):                              # swinging: neither template matches
            self.assertEqual([], self.unknown(stamp))
        actions = self.update(4, barrier='up')
        self.assertEqual([{'type': 'raised', 'alarm_id': 'event-a', 'observed_at': 4, 'route': EXIT,
                           'origin_observed_at': 0, 'alarm_started_at': 0, 'position_since': 4}], actions)

    def test_lower_is_seen_through_unknown_readings(self):
        self.assertEqual([], self.update(0, barrier='up'))   # held open when the car came
        self.unknown(1); self.unknown(2)
        self.assertEqual(['lowered'], [a['type'] for a in self.update(3, barrier='down')])

    def test_arm_already_moving_before_the_alarm_cannot_answer_it(self):
        self.monitor.update({EXIT: observation(0, barrier='down')}, None, 0)   # no alarm yet
        for stamp in (1, 2, 3):
            self.unknown(stamp, alarm=None)
        self.unknown(4)                                       # alarm starts while the arm moves
        self.assertEqual([], self.update(5, barrier='up'))

    def test_previous_car_lowering_cannot_answer_a_new_alarm(self):
        self.monitor.update({EXIT: observation(0, barrier='up')}, None, 0)
        alarm = {'id': 'new-car', 'route': EXIT, 'started_at': 1.}
        self.monitor.update({EXIT: observation(1, barrier='unknown')}, alarm, 1)
        self.assertEqual([], self.monitor.update(
            {EXIT: observation(1.7, barrier='down')}, alarm, 1.7))

    def test_replacement_alarm_discards_previous_car_arm_anchor(self):
        self.update(0, barrier='up')
        self.unknown(.5)
        replacement = {'id': 'event-b', 'route': EXIT, 'started_at': 1.}
        self.assertEqual([], self.monitor.update(
            {EXIT: observation(1.7, barrier='down')}, replacement, 1.7))

    def test_delayed_stable_confirmation_cannot_freshen_pre_alarm_position_change(self):
        alarm = {'id': 'new-car', 'route': EXIT, 'started_at': 1.}
        self.monitor.update({EXIT: observation(1.1, barrier='up')}, alarm, 1.1)
        self.assertEqual([], self.monitor.update({EXIT: observation(
            1.7, barrier='down', barrier_position_since=.9)}, alarm, 1.7))

    def test_current_alarm_movement_proof_is_bound_to_its_origin_and_start(self):
        alarm = {'id': 'new-car', 'route': EXIT, 'started_at': 1.}
        self.monitor.update({EXIT: observation(1.1, barrier='up')}, alarm, 1.1)
        self.monitor.update({EXIT: observation(1.2, barrier='unknown')}, alarm, 1.2)
        actions = self.monitor.update({EXIT: observation(
            1.7, barrier='down', barrier_position_since=1.3)}, alarm, 1.7)
        self.assertEqual(1, len(actions))
        self.assertEqual('lowered', actions[0]['type'])
        self.assertEqual('new-car', actions[0]['alarm_id'])
        self.assertEqual(1.1, actions[0]['origin_observed_at'])
        self.assertEqual(1., actions[0]['alarm_started_at'])
        self.assertEqual(1.3, actions[0]['position_since'])

    def test_invalid_position_start_proof_cannot_answer_alarm(self):
        for bad in (None, True, math.nan, math.inf, '1', -1, 3):
            with self.subTest(bad=bad):
                self.setUp()
                self.update(0)
                self.assertEqual([], self.update(1, barrier='up', barrier_position_since=bad))

    def test_pending_distinct_arrival_blocks_same_frame_arm_resolution(self):
        self.update(0, barrier='up')
        self.assertEqual([], self.update(1, barrier='down', new_arrival=True))
        self.assertTrue(self.monitor.status(1)['routes'][EXIT]['blocked_by_new_arrival'])

    def test_old_arm_position_does_not_count(self):
        self.monitor.update({EXIT: observation(0, barrier='down')}, None, 0)
        self.unknown(30)                                      # alarm 30 s later; anchor too old
        self.assertEqual([], self.update(31, barrier='up'))

    def test_each_change_is_reported_once(self):
        self.update(0)
        self.assertEqual(['raised'], [a['type'] for a in self.update(1, barrier='up')])
        for stamp in (2, 3, 4):
            self.assertEqual([], [a for a in self.update(stamp, barrier='up') if a['type'] == 'raised'])

    def test_open_reminders_after_fifteen_seconds_and_each_fifteen_after(self):
        self.raise_for_alarm()
        emitted = []
        for stamp in range(3, 50):
            emitted.extend(self.monitor.update({EXIT: observation(stamp, barrier='up')}, None, stamp))
        self.assertEqual([17, 32, 47], [action['observed_at'] for action in emitted])
        self.assertTrue(all(action['type'] == 'open_reminder' and action['route'] == EXIT
                            and 'alarm_id' not in action for action in emitted))
        self.assertTrue(self.monitor.status(49)['routes'][EXIT]['reminder_active'])
        self.assertFalse(self.monitor.status(52)['routes'][EXIT]['reminder_active'])

    def test_arm_held_open_without_a_visit_never_knocks(self):
        emitted = []
        for stamp in range(120):
            emitted.extend(self.monitor.update({EXIT: observation(stamp, barrier='up')}, None, stamp))
        self.assertEqual([], emitted)
        self.assertFalse(self.monitor.status(119)['routes'][EXIT]['reminder_active'])

    def test_dismissed_reminder_stays_quiet_until_the_arm_closes(self):
        self.raise_for_alarm()
        for stamp in range(3, 20):
            self.monitor.update({EXIT: observation(stamp, barrier='up')}, None, stamp)
        self.assertTrue(self.monitor.status(19)['routes'][EXIT]['reminder_active'])
        self.assertEqual([EXIT], self.monitor.mute_reminders(19))
        quiet = []
        for stamp in range(20, 80):
            quiet.extend(self.monitor.update({EXIT: observation(stamp, barrier='up')}, None, stamp))
        self.assertEqual([], quiet)
        self.assertFalse(self.monitor.status(79)['routes'][EXIT]['reminder_active'])
        # Closing the arm clears the dismissal and the "served" flag.
        self.monitor.update({EXIT: observation(80, barrier='down')}, None, 80)
        status = self.monitor.status(80)['routes'][EXIT]
        self.assertFalse(status['reminder_muted'])
        self.assertFalse(status['raised_for_visit'])

    def test_open_timer_uses_source_not_polling_time_and_stops_unknown_down(self):
        for stamp in range(17):
            self.monitor.update({EXIT: observation(stamp, barrier='up')}, None, stamp)
        self.assertEqual([], self.monitor.update({EXIT: observation(16, barrier='up')}, None, 17))
        self.assertEqual(16., self.monitor.status(17)['routes'][EXIT]['open_s'])
        for state in ('unknown', 'down'):
            self.monitor.update({EXIT: observation(18, barrier=state)}, None, 18)
            self.assertFalse(self.monitor.status(18)['routes'][EXIT]['reminder_active'])
        self.assertEqual([], self.monitor.update({EXIT: observation(19, barrier='up')}, None, 19))

    def test_large_gap_in_up_readings_cannot_be_assumed_continuous(self):
        self.monitor.update({EXIT: observation(0, barrier='up')}, None, 0)
        self.assertEqual([], self.monitor.update({EXIT: observation(20, barrier='up')}, None, 20))
        self.assertEqual(0., self.monitor.status(20)['routes'][EXIT]['open_s'])

    def test_off_reset_discards_pending_resolution_and_open_timer(self):
        self.update(0)
        self.update(1, 'clear', 'up')
        self.monitor.reset()
        self.assertEqual({}, self.monitor.status(2)['routes'])
        self.assertEqual([], self.update(3, 'clear'))

    def test_reordered_old_caller_cannot_replace_alarm_or_rewind_evidence(self):
        self.update(10)
        other = {'id': 'old-event', 'route': EXIT}
        self.assertEqual([], self.monitor.update({EXIT: observation(9, 'clear')}, other, 9))
        self.assertEqual('event-a', self.monitor.status(10)['routes'][EXIT]['alarm_id'])

    def test_bounds_and_returned_status_are_independent(self):
        monitor = ServiceCycleMonitor(max_routes=1)
        monitor.update({EXIT: observation(0), ENTRY: observation(0)}, ALARM, 0)
        snapshot = monitor.status(0)
        self.assertEqual([EXIT], list(snapshot['routes']))
        snapshot['routes'][EXIT]['alarm_id'] = 'tampered'
        self.assertEqual('event-a', monitor.status(0)['routes'][EXIT]['alarm_id'])
        for kwargs in ({'clear_dwell_s': 1}, {'freshness_s': math.inf},
                       {'open_after_s': 5}, {'reminder_interval_s': True}, {'max_routes': 0}):
            with self.assertRaises(ValueError):
                ServiceCycleMonitor(**kwargs)


if __name__ == '__main__':
    unittest.main()
