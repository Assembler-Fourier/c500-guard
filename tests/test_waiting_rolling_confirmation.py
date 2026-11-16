"""Stop-line rolling confirmation with real tracker and fake alarm services.

Seven anonymized retained observations reproduce the eligible boxes of the
missed entrance visit. Intermediate empty results are an explicit modeled
counterfactual: the sightings file does not retain those inference packets.
No model, camera, speaker or production state is used.
"""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

from test_watch import watch, watcher
from object_tracking import ObjectVisitTracker
from test_visit_grouping import RecordingLoudness


CONFIG = Path(__file__).resolve().parents[1] / 'config/zones.json'
RECORDED = (
    (0., .6136, [.72325, .38203, .82784, .51625]),
    (.277, .5819, [.72223, .38831, .82756, .51908]),
    (.817, .5518, [.71852, .40669, .82620, .54611]),
    (1.551, .8596, [.72575, .41131, .81434, .55294]),
    (1.851, .5258, [.71483, .40925, .81909, .56158]),
    (2.806, .5403, [.71981, .42269, .82625, .56731]),
    (3.312, .5178, [.71831, .42297, .82525, .56644]),
)


def detection(confidence=.55, bbox=None):
    box = bbox or [.72325, .38203, .82784, .51625]
    return {'label': 'car', 'confidence': confidence,
            'bbox': [box[0]*640, box[1]*360, box[2]*640, box[3]*360]}


def waiting(window):
    original = next(z for z in watch.load_zones(CONFIG) if z.name == 'Entrance waiting')
    return replace(original, ai_confirmation_window_s=window)


class WaitingRollingTrackerTests(unittest.TestCase):
    def setup_tracker(self, window=1.5):
        self.zone = waiting(window)
        self.tracker = ObjectVisitTracker()
        self.events = []

    def step(self, at, objects=()):
        events, seen = self.tracker.update(list(objects), [self.zone], 640, 360, 100.+at)
        for event in events:
            self.events.append((at, self.tracker.event_context(event[2], [self.zone])))
            self.tracker.mark_emitted(event[2])
        return seen

    def recorded_with_modeled_gaps(self):
        previous = None
        for at, confidence, box in RECORDED:
            if previous is not None and at-previous > .4:
                self.step((previous+at)/2.)
            self.step(at, [detection(confidence, box)])
            previous = at

    def test_zero_window_control_misses_intermittent_recorded_boxes(self):
        self.setup_tracker(0.)
        self.recorded_with_modeled_gaps()
        self.assertEqual([], self.events)

    def test_window_confirms_third_recorded_box_despite_modeled_miss(self):
        self.setup_tracker()
        self.recorded_with_modeled_gaps()
        self.assertEqual(1, len(self.events))
        self.assertAlmostEqual(.817, self.events[0][0])
        self.assertEqual(3, self.events[0][1]['observations'])
        self.assertNotEqual('strong_pair', self.events[0][1].get('confirmation_mode'))

    def test_two_normal_observations_never_count_as_three(self):
        self.setup_tracker()
        self.step(0., [detection(.50)])
        self.step(.2)
        self.step(.4, [detection(.55)])
        self.assertEqual([], self.events)
        self.step(.6, [detection(.50)])
        self.assertEqual(1, len(self.events))
        self.assertEqual(3, self.events[0][1]['observations'])

    def test_below_floor_box_is_not_a_confirmation(self):
        self.setup_tracker()
        for at, confidence in ((0., .55), (.3, .49), (.6, .55), (.9, .4999)):
            self.step(at, [detection(confidence)])
        self.assertEqual([], self.events)
        self.step(1.2, [detection(.50)])
        self.assertEqual(1, len(self.events))

    def test_old_observation_expires_before_new_confirmation(self):
        self.setup_tracker()
        for at in (0., .5, 1.501):
            self.step(at, [detection()])
        self.assertEqual([], self.events)
        self.step(1.6, [detection()])
        self.assertEqual(1, len(self.events))

    def test_single_false_detection_followed_by_empty_frames_never_alarms(self):
        self.setup_tracker()
        self.step(0., [detection()])
        for n in range(1, 13):
            self.step(n*.25)
        self.assertEqual([], self.events)

    def test_repeated_source_timestamp_cannot_multiply_evidence(self):
        self.setup_tracker()
        for _ in range(5):
            self.step(0., [detection()])
        self.assertEqual([], self.events)
        self.step(.25, [detection()])
        self.assertEqual([], self.events)

    def test_leaving_confirmation_polygon_clears_earlier_samples(self):
        self.setup_tracker()
        self.step(0., [detection()])
        self.step(.25, [detection()])
        self.step(.5, [detection(bbox=[.72325, .34, .82784, .475])])
        self.step(.75, [detection()])
        self.assertEqual([], self.events)

    def test_two_simultaneous_vehicles_do_not_pool_confirmations(self):
        self.setup_tracker()
        first = detection(bbox=[.67, .40, .73, .52])
        second = detection(bbox=[.80, .40, .86, .52])
        self.step(0., [first, second])
        self.step(.25, [first])
        self.step(.5, [second])
        self.assertEqual([], self.events)
        self.step(.75, [first, second])
        self.assertEqual(2, len(self.events))
        self.assertEqual(2, len({context['track_id'] for _, context in self.events}))

    def test_strong_pair_keeps_two_distinct_sources_and_original_threshold(self):
        self.setup_tracker()
        self.assertEqual(.6, self.zone.ai_fast_confirm_confidence)
        self.step(0., [detection(.61)])
        self.step(.25, [detection(.60)])
        self.assertEqual(1, len(self.events))
        context = self.events[0][1]
        self.assertEqual(2, context['observations'])
        self.assertEqual('strong_pair', context['confirmation_mode'])
        self.assertEqual(2, len(context['strong_confirmation']))


class WaitingRollingWatcherTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        for patch in (mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now),
                      mock.patch.object(watch, 'notify')):
            patch.start()
            self.addCleanup(patch.stop)
        self.obj = watcher(zones_path=CONFIG)
        self.obj.zones = [waiting(1.5)]
        self.obj._pending_rearm = False
        self.obj.ack_quiet_s = 10.
        self.obj._queue_snapshot = mock.Mock(return_value=None)
        self.obj._remember_alert = mock.Mock()
        self.obj._mark_alert_acknowledged = mock.Mock()
        self.obj.loudness = RecordingLoudness()
        self.tracker = ObjectVisitTracker()

    def step(self, at, objects=()):
        self.now = 100.+at
        events, seen = self.tracker.update(list(objects), self.obj.zones, 640, 360, self.now)
        self.obj.visits.observe(self.now, self.obj.zones,
            {'available': True, 'state': 'ready', 'age_s': 0., 'detections': seen})
        self.obj._expire_ack_locked(self.now)
        for kind, reason, ident in events:
            context = self.tracker.event_context(ident, self.obj.zones)
            if self.obj.raise_alarm(kind, reason, details={'source': 'ai', **context}):
                self.tracker.mark_emitted(ident)

    def initial(self):
        for at in (0., .3, .6):
            self.step(at, [detection()])
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertTrue(self.obj.acknowledge(via='web'))

    def test_acknowledged_same_car_is_not_replayed_after_quiet(self):
        self.initial()
        for n in range(1, 49):
            self.step(.6+n*.25, [detection()] if n % 3 else [])
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_distinct_waiting_follower_survives_ack_quiet(self):
        self.initial()
        follower = detection(bbox=[.805,.40,.865,.52])
        for n in range(1, 49):
            self.step(.6+n*.25, [detection(), follower])
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, self.obj._alarm_sequence)
        self.assertEqual(2, self.obj.alarm_details['track_id'])

    def test_fully_clear_then_new_car_gets_new_alarm(self):
        self.initial()
        for n in range(1, 49):
            self.step(.6+n*.25)
        self.assertEqual(0, self.obj.visits.status(self.now)['active_visits'])
        for at in (13., 13.3, 13.6):
            self.step(at, [detection()])
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, self.obj._alarm_sequence)


if __name__ == '__main__':
    unittest.main()
