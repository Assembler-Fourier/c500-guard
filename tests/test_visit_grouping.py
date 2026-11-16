"""An acknowledged visit is consumed; a different observed car remains eligible."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_watch import watch, watcher

EXIT_CORNER = {'source': 'motion', 'route_group': 'campus-exit', 'zone_name': 'Exit corner'}
EXIT_APPROACH = {'source': 'light', 'route_group': 'campus-exit', 'zone_name': 'Exit approach'}
ENTRANCE = {'source': 'light', 'route_group': 'campus-entrance', 'zone_name': 'Top road arrival'}
FOOTPATH = {'source': 'ai', 'track_id': 3, 'label': 'person', 'zone_name': 'Pedestrian footpath'}


class RecordingLoudness:
    def __init__(self):
        self.calls = []
        # The watcher asks which sound is currently owned, so the barrier
        # reminder can stand aside for a real alarm.
        self.kind = None

    def begin(self, kind, **settings):
        self.kind = kind
        self.calls.append({'kind': kind, **settings})

    def evidence(self, **settings):
        pass

    def tick(self):
        pass

    def end(self):
        self.kind = None

    def status(self):
        return {'enabled': True, 'active': False}


class VisitGroupingTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher()
        self.obj.armed = True
        self.obj._pending_rearm = False
        self.obj.ack_quiet_s = 10.
        self.events = []
        self.obj.log_event = lambda kind, detail, **extra: self.events.append((kind, detail, extra))
        self.obj._queue_snapshot = lambda kind: None
        self.obj._remember_alert = lambda *a, **k: None
        self.obj._mark_alert_acknowledged = lambda *a, **k: None
        self.loudness = RecordingLoudness()
        self.obj.loudness = self.loudness

    def raise_alarm(self, kind='main', details=None):
        return self.obj.raise_alarm(kind, 'traffic', details=details)

    def acknowledge(self):
        return self.obj.acknowledge(via='web')

    def test_cross_zone_repeat_is_consumed_without_another_tone_or_incident(self):
        self.assertTrue(self.raise_alarm(details=EXIT_CORNER))
        self.assertTrue(self.acknowledge())
        self.now += 11
        self.assertTrue(self.raise_alarm(details=EXIT_APPROACH), 'True tells the channel its episode was consumed')
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertEqual(1, len(self.loudness.calls))
        self.assertEqual(1, len([event for event in self.events if event[0] == 'main']))

    def test_fixed_time_does_not_make_continuing_visit_new(self):
        self.raise_alarm(details=EXIT_CORNER)
        self.acknowledge()
        for after in (11, 46, 120, 600):
            self.now = 1000 + after
            self.assertTrue(self.raise_alarm(details=EXIT_APPROACH))
            self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, self.obj._alarm_sequence)

    def test_new_route_waits_for_exact_ten_second_quiet_deadline(self):
        self.raise_alarm(details=EXIT_CORNER)
        self.acknowledge()
        self.now += 9.999
        self.assertFalse(self.raise_alarm(details=ENTRANCE))
        self.assertIsNone(self.obj.alarm)
        self.now = 1010.
        self.assertTrue(self.raise_alarm(details=ENTRANCE))
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, len(self.loudness.calls))

    def test_late_confirmed_car_inherits_acknowledged_image_warning(self):
        self.raise_alarm(kind='early', details=EXIT_CORNER)
        self.acknowledge()
        self.now += 12
        self.assertTrue(self.raise_alarm(details={**EXIT_APPROACH, 'source': 'ai', 'track_id': 1,
                                                'label': 'car', 'bbox_normalized': [.1, .2, .3, .4]}))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1, len(self.loudness.calls))

    def test_late_ai_enriches_current_incident_without_restarting_alarm(self):
        self.raise_alarm(details=EXIT_CORNER)
        alarm_id = self.obj.alarm_id
        self.now += 12
        self.assertTrue(self.raise_alarm(details={**EXIT_APPROACH, 'source': 'ai', 'track_id': 1,
                                                'label': 'car', 'bbox_normalized': [.1, .2, .3, .4]}))
        self.assertEqual(alarm_id, self.obj.alarm_id)
        self.assertEqual('ai', self.obj.alarm_details['source'])
        self.assertEqual(1, len(self.loudness.calls))

    def test_different_confirmed_car_remains_eligible_on_same_road(self):
        first = {**EXIT_APPROACH, 'source': 'ai', 'track_id': 1,
                 'label': 'car', 'bbox_normalized': [.1, .2, .3, .4]}
        second = {**first, 'track_id': 2, 'bbox_normalized': [.5, .2, .7, .4]}
        self.raise_alarm(details=first)
        self.acknowledge()
        self.now += 3
        self.assertFalse(self.raise_alarm(details=second))
        self.now += 7
        self.assertTrue(self.raise_alarm(details=second))
        self.assertEqual(2, self.obj._alarm_sequence)
        self.assertEqual(2, self.obj.alarm_details['track_id'])

    def test_unscoped_pedestrian_zone_has_independent_visit(self):
        self.raise_alarm(details=FOOTPATH)
        self.acknowledge()
        self.now += 10
        self.assertTrue(self.raise_alarm(details=FOOTPATH))
        self.assertIsNone(self.obj.alarm)
        self.assertTrue(self.raise_alarm(details=ENTRANCE))
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_fault_is_independent_of_traffic_visit_memory(self):
        self.raise_alarm(details=EXIT_CORNER)
        self.acknowledge()
        self.now += 10
        self.assertTrue(self.raise_alarm(kind='fault', details=EXIT_CORNER))
        self.assertEqual('fault', self.obj.alarm)

    def test_explicit_rearm_creates_fresh_visit(self):
        self.raise_alarm(details=EXIT_CORNER)
        self.acknowledge()
        self.obj.set_armed(False)
        self.obj.set_armed(True)
        self.obj._pending_rearm = False
        self.now += 1
        self.assertTrue(self.raise_alarm(details=EXIT_CORNER))
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual(2, self.obj._alarm_sequence)

    def test_status_reports_acknowledged_visit_and_suppressed_duplicate(self):
        self.raise_alarm(details=EXIT_CORNER)
        self.acknowledge()
        self.now += 10
        self.raise_alarm(details=EXIT_APPROACH)
        status = self.obj._visit_status()
        self.assertEqual(1, status['acknowledged_visits'])
        self.assertEqual('campus-exit', status['routes'][0]['route'])
        self.assertEqual(1, status['held_alarms'])
        self.assertEqual(6., status['clear_dwell_s'])


class AcknowledgementSettingTests(unittest.TestCase):
    def load(self, payload):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            return watch.Watcher._load_ack_quiet(path)

    def test_default_is_ten_seconds(self):
        self.assertEqual(10., self.load({'zones': []}))

    def test_values_are_clamped(self):
        self.assertEqual(5., self.load({'alarms': {'ack_quiet_s': 1}}))
        self.assertEqual(30., self.load({'alarms': {'ack_quiet_s': 99999}}))

    def test_invalid_values_keep_default(self):
        for value in ('soon', True, None, float('nan'), float('inf')):
            self.assertEqual(10., self.load({'alarms': {'ack_quiet_s': value}}))
        self.assertEqual(10., self.load({'alarms': 'none'}))

    def test_missing_file_keeps_default(self):
        self.assertEqual(10., watch.Watcher._load_ack_quiet(Path('no-such-file.json')))


if __name__ == '__main__':
    unittest.main()
