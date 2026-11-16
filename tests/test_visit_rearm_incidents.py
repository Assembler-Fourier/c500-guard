"""Configured PTZ visit/quiet scenarios with real tracking and a fake alarm output.

No model, camera, speaker, HTTP service or live state is used. Invented boxes
exercise the current configured polygons; this is not an exact incident replay.
"""
from pathlib import Path
import unittest
from unittest import mock

from object_tracking import ObjectVisitTracker
from region_geometry import point_in_polygon
from test_watch import watch, watcher


def object_at(x, y, label='car'):
    width, height = (.025, .09) if label == 'person' else (.07, .12)
    return {'label': label, 'confidence': .90,
            'bbox': [(x-width/2)*640, (y-height)*360, (x+width/2)*640, y*360]}


class ConfiguredVisitRearmTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher(zones_path=Path(__file__).resolve().parents[1] / 'config' / 'zones.json')
        self.obj._pending_rearm = False
        self.obj._queue_snapshot = mock.Mock(return_value=None)
        self.tracker = ObjectVisitTracker()
        self.last_objects = []
        self.accepted = []
        self.entrance = object_at(.755, .49)
        self.second_entrance = object_at(.835, .505)
        # These invented cars exercise independent visits inside the physical
        # stop-line core, not arrivals on the excluded upper departure branch.
        waiting = next(z for z in self.obj.zones if z.name == 'Entrance waiting')
        for vehicle in (self.entrance, self.second_entrance):
            x1, _, x2, bottom = vehicle['bbox']
            ground = ((x1+x2)/1280, bottom/360)
            self.assertTrue(waiting.contains_point(*ground))
            self.assertTrue(point_in_polygon(*ground, waiting.confirmation_polygon),
                            'Synthetic entrance fixtures must be in the confirmation core')
        self.exit = object_at(.51, .59)
        self.person = object_at(.20, .60, 'person')

    def deliver(self, objects=(), after=.2):
        self.now += after
        events, observed = self.tracker.update(objects, self.obj.zones, 640, 360, self.now)
        self.last_objects = observed
        ml = {'available': True, 'state': 'ready', 'age_s': 0., 'detections': observed}
        self.obj.visits.observe(self.now, self.obj.zones, ml)
        self.obj._expire_ack_locked(self.now)
        for kind, reason, event_id in events:
            context = self.tracker.event_context(event_id, self.obj.zones)
            sequence = self.obj._alarm_sequence
            if self.obj.raise_alarm(kind, reason, details={'source': 'ai', **context}):
                self.tracker.mark_emitted(event_id)
            if self.obj._alarm_sequence != sequence:
                self.accepted.append(dict(self.obj.alarm_details))
        return observed

    def hold(self, objects=(), seconds=4.):
        for _ in range(round(seconds/.2)):
            self.deliver(objects)

    def test_minutes_old_entrance_ack_cannot_hide_a_new_entrance_car(self):
        self.hold([self.entrance])
        self.assertEqual('main', self.obj.alarm)
        prior = dict(self.obj.alarm_details)
        self.assertTrue(self.obj.acknowledge())
        self.hold(seconds=12.)
        self.assertEqual(0, self.obj.visits.status(self.now)['active_visits'])
        self.deliver(after=180.)
        self.hold([self.entrance])
        self.assertEqual('main', self.obj.alarm)
        self.assertNotEqual(prior['track_id'], self.obj.alarm_details['track_id'])
        self.assertNotEqual(prior['visit_id'], self.obj.alarm_details['visit_id'])
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_same_waiting_car_remains_acknowledged_after_quiet(self):
        self.hold([self.entrance])
        self.assertTrue(self.obj.acknowledge())
        self.hold([self.entrance], seconds=20.)
        self.assertEqual(0, self.obj._ack_until)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_second_visible_entrance_car_survives_first_ack_and_quiet(self):
        self.hold([self.entrance, self.second_entrance])
        first = dict(self.obj.alarm_details)
        self.assertEqual('main', self.obj.alarm)
        self.assertTrue(self.obj.acknowledge())
        self.hold([self.entrance, self.second_entrance], seconds=11.)
        self.assertEqual('main', self.obj.alarm)
        self.assertNotEqual(first['track_id'], self.obj.alarm_details['track_id'])
        self.assertNotEqual(first['visit_id'], self.obj.alarm_details['visit_id'])
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_exit_entrance_and_pedestrian_stay_independently_pending(self):
        objects = [self.exit, self.entrance, self.person]
        self.hold(objects)
        for _ in range(2):
            self.assertEqual('main', self.obj.alarm)
            self.assertTrue(self.obj.acknowledge())
            self.hold(objects, seconds=11.)
        self.assertEqual(3, self.obj._alarm_sequence)
        self.assertEqual({'exit', 'entrance', 'pedestrian'},
                         {event['route_role'] for event in self.accepted})
        self.assertEqual(3, len({event['visit_id'] for event in self.accepted}))

    def test_person_does_not_inherit_acknowledged_car_identity(self):
        self.hold([self.entrance])
        car_id = self.obj.alarm_details['visit_id']
        self.assertTrue(self.obj.acknowledge())
        self.hold([self.entrance, self.person], seconds=11.)
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual('person', self.obj.alarm_details['label'])
        self.assertNotEqual(car_id, self.obj.alarm_details['visit_id'])

    def test_remembered_raised_entrance_does_not_reject_a_new_alarm(self):
        # raise_alarm has no barrier-up rejection: old arm memory may affect
        # presentation, but cannot consume an otherwise accepted new visit.
        self.obj._arm_memory = {'entrance': {'state': 'up', 'since': self.now-200,
                                            'seen_at': self.now-100}}
        self.hold([self.entrance])
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual('entrance', self.obj.alarm_details['route_role'])
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_incoming_car_gets_advance_warning_before_brief_gate_confidence_dip(self):
        # Invented sampling of an incoming turn. One weak gate observation
        # must not erase an advance warning already accepted on the top road.
        points = [(.48, .33), (.51, .35), (.57, .365), (.63, .385),
                  (.67, .405), (.70, .425), (.73, .445), (.75, .485),
                  (.755, .505), (.75, .54), (.76, .56)]
        for index, point in enumerate(points):
            obj = object_at(*point)
            if index == 7:
                obj['confidence'] = .55
            self.deliver([obj], after=.6)
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertEqual('Top road arrival', self.obj.alarm_details['zone_name'])
        self.assertEqual('entrance', self.obj.alarm_details['route_role'])

    def test_one_hundred_varied_visits_rearm_once_and_keep_memory_bounded(self):
        # Each cycle supplies continuing observations, including an observed
        # clear interval; a clock jump alone must never manufacture a departure.
        labels = ('car', 'truck', 'bus', 'person')
        announced_ids = set()
        track_ids = set()
        peak_remembered = 0
        for visit_index in range(100):
            label = labels[visit_index % len(labels)]
            centre = ((.20, .60) if label == 'person' else
                      (.755, .49) if visit_index % 2 == 0 else (.51, .59))
            sequence_before = self.obj._alarm_sequence
            for sample in range(22):
                # Sub-pixel deterministic detector jitter keeps a stationary
                # waiting object realistic without inventing approach motion.
                dx = ((sample + visit_index) % 3 - 1) * .0002
                dy = ((sample * 2 + visit_index) % 3 - 1) * .0002
                self.deliver([object_at(centre[0]+dx, centre[1]+dy, label)])
                self.assertLessEqual(self.obj._alarm_sequence, sequence_before+1)
                self.assertLessEqual(self.tracker.track_count, 1)
                self.assertLessEqual(self.obj.visits.status(self.now)['active_visits'], 1)
            self.assertEqual(sequence_before+1, self.obj._alarm_sequence, visit_index)
            self.assertEqual('main', self.obj.alarm)
            self.assertEqual(label, self.obj.alarm_details['label'])
            announced_ids.add(self.obj.alarm_details['visit_id'])
            track_ids.add(self.obj.alarm_details['track_id'])
            self.assertTrue(self.obj.acknowledge())
            # Keep seeing that same car/person during quiet: no second alarm.
            for _ in range(3):
                self.deliver([object_at(*centre, label)])
                self.assertIsNone(self.obj.alarm)
            for _ in range(24):
                self.deliver(after=.5)
                self.assertIsNone(self.obj.alarm)
                self.assertEqual(sequence_before+1, self.obj._alarm_sequence)
            self.assertEqual(0, self.tracker.track_count)
            self.assertEqual(0, self.obj.visits.status(self.now)['active_visits'])
            self.assertEqual(0, self.obj._ack_until)
            peak_remembered = max(peak_remembered, len(self.obj.visits._visits))
            self.assertLessEqual(len(self.obj.visits._visits), 8,
                                 'Retired visits must expire, not grow with traffic volume')
        self.assertEqual(100, self.obj._alarm_sequence)
        self.assertEqual(100, len(announced_ids))
        self.assertEqual(100, len(track_ids))
        self.assertLess(peak_remembered, 100)
        # A batch of visible objects outside every configured alarm region
        # remains silent even after all previous acknowledgement quiet expires.
        parking = [object_at(.94, .80, 'car'), object_at(.94, .95, 'bus'),
                   object_at(.95, .60, 'person')]
        for _ in range(20):
            observed = self.deliver(parking, after=.5)
            self.assertEqual(3, len(observed))
            self.assertTrue(all(not obj['regions'] for obj in observed))
            self.assertIsNone(self.obj.alarm)
            self.assertEqual(100, self.obj._alarm_sequence)


if __name__ == '__main__':
    unittest.main()
