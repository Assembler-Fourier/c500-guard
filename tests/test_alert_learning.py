"""Learning from the system's own evidence: what is real, what to tighten, which arms drifted."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_watch import watch, watcher  # noqa: F401 - also puts src on the path
from alert_learning import AlertLearning, FOLLOW_S, MIN_EVIDENCE
from barrier_workflow import BarrierWorkflow

LIGHT = {'source': 'light', 'zone_name': 'Top road arrival', 'route_group': 'campus-entrance'}
CORNER = {'source': 'motion', 'strict_change': True, 'zone_name': 'Exit corner - immediate',
          'route_group': 'campus-exit'}


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.learning = AlertLearning(None, clock=self.clock)

    def alert(self, details, recognised_route=None, after=3.):
        self.learning.note_alarm('a', details)
        self.clock.now += after
        if recognised_route:
            self.learning.note_recognised(recognised_route)
        self.clock.now += FOLLOW_S
        self.learning.tick()

    def row(self, key):
        return next(r for r in self.learning.status()['routes'] if r['key'] == key)

    def test_alert_followed_by_recognition_on_same_route_counts_as_real(self):
        self.alert(LIGHT, 'campus-entrance')
        self.assertEqual((1, 1), (self.row('Top road arrival|light')['followed'],
                                  self.row('Top road arrival|light')['alerts']))

    def test_other_route_or_late_recognition_does_not_count(self):
        self.alert(LIGHT, 'campus-exit')
        self.alert(LIGHT, 'campus-entrance', after=FOLLOW_S + 5)
        self.assertEqual(0, self.row('Top road arrival|light')['followed'])
        self.assertEqual(2, self.row('Top road arrival|light')['alerts'])

    def test_ai_alerts_are_real_by_definition(self):
        self.learning.note_alarm('a', {'source': 'ai', 'zone_name': 'Exit approach', 'route_group': 'campus-exit'})
        self.assertEqual(1, self.row('Exit approach|ai')['followed'])

    def test_reliable_only_after_enough_evidence(self):
        for _ in range(MIN_EVIDENCE - 1):
            self.alert(CORNER, 'campus-exit')
        self.assertFalse(self.learning.reliable(CORNER))
        self.alert(CORNER, 'campus-exit')
        self.assertTrue(self.learning.reliable(CORNER))
        self.assertFalse(self.learning.reliable({'source': 'ai', 'zone_name': 'x'}))

    def test_rarely_real_light_kind_gets_a_tightening_suggestion(self):
        for index in range(MIN_EVIDENCE):
            self.alert(LIGHT, 'campus-entrance' if index < 2 else None)
        config = {'zones': [{'name': 'Top road arrival', 'light': True, 'light_delta': 20}]}
        suggestions = self.learning.suggestions(config)
        self.assertEqual(1, len(suggestions))
        self.assertEqual('zone:Top road arrival:light_delta', suggestions[0]['key'])
        self.assertGreater(suggestions[0]['value'], 20)
        self.assertFalse(self.learning.reliable(LIGHT))

    def test_mostly_real_kind_gets_no_suggestion(self):
        for _ in range(MIN_EVIDENCE):
            self.alert(LIGHT, 'campus-entrance')
        config = {'zones': [{'name': 'Top road arrival', 'light': True, 'light_delta': 20}]}
        self.assertEqual([], self.learning.suggestions(config))

    def test_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learning.json'
            first = AlertLearning(path, clock=self.clock)
            first.note_alarm('a', LIGHT)
            first.note_recognised('campus-entrance')
            self.clock.now += FOLLOW_S + 1
            first.tick()
            first.save(force=True)
            second = AlertLearning(path, clock=self.clock)
            row = next(r for r in second.status()['routes'] if r['key'] == 'Top road arrival|light')
            self.assertEqual((1, 1), (row['followed'], row['alerts']))

    def test_corrupt_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learning.json'
            path.write_text('{"kinds": {"x": {"alerts": -1}}}', encoding='utf-8')
            self.assertEqual([], AlertLearning(path).status()['routes'])


class WatcherLearningTests(unittest.TestCase):
    def test_alerts_are_recorded_and_reliable_kinds_climb_faster(self):
        obj = watcher()
        obj.loudness = mock.Mock()
        obj.learning = mock.Mock()
        obj.learning.reliable.return_value = True
        obj.armed = True
        self.assertTrue(obj.raise_alarm('main', 'Top road arrival: light', details=dict(LIGHT)))
        obj.learning.note_alarm.assert_called_once()
        self.assertTrue(obj.loudness.begin.call_args.kwargs['urgent'])
        self.assertTrue(obj.alarm_details['learned_reliable'])

    def test_unproven_kinds_keep_their_normal_tier(self):
        obj = watcher()
        obj.loudness = mock.Mock()
        obj.learning = mock.Mock()
        obj.learning.reliable.return_value = False
        obj.armed = True
        obj.raise_alarm('main', 'Top road arrival: light', details=dict(LIGHT))
        self.assertFalse(obj.loudness.begin.call_args.kwargs['urgent'])


class BarrierSelfLearningTests(unittest.TestCase):
    def workflow(self, best, confidence=.8, stable_since=0.):
        monitor = mock.Mock()
        monitor._observations = {'exit': {'best': best, 'candidate_at': stable_since}}
        monitor.capture.return_value = {'ok': True}
        clock = Clock()
        flow = BarrierWorkflow('test', tempfile.gettempdir(), None, monitor=monitor, clock=clock)
        flow._backup_references = mock.Mock(return_value=Path('synthetic-backup'))
        status = {'barriers': [{'id': 'exit', 'state': 'down', 'fresh': True, 'calibrated': True,
                                'confidence': confidence}]}
        return flow, monitor, clock, status

    def test_drifted_but_confident_arm_is_learned_once_per_half_hour(self):
        flow, monitor, clock, status = self.workflow(.86)
        flow._learn_arms(status, clock())
        flow._learn_arms(status, clock() + 60)
        monitor.capture.assert_called_once_with('exit', 'down')
        flow._learn_arms(status, clock() + 1801)
        self.assertEqual(2, monitor.capture.call_count)
        self.assertEqual('down', flow.learned[-1]['position'])

    def test_drifting_arm_at_clear_confidence_is_learned_once_stable_for_a_minute(self):
        flow, monitor, clock, status = self.workflow(.74, confidence=.65, stable_since=1000.)
        flow._learn_arms(status, 1030.)            # only 30 s in this position
        monitor.capture.assert_not_called()
        flow._learn_arms(status, 1061.)
        monitor.capture.assert_called_once_with('exit', 'down')

    def test_good_match_or_low_confidence_is_not_learned(self):
        for best, confidence in ((.95, .9), (.86, .3)):
            with self.subTest(best=best, confidence=confidence):
                flow, monitor, clock, status = self.workflow(best, confidence)
                flow._learn_arms(status, clock())
                monitor.capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
