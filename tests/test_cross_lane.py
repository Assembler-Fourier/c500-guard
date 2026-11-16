"""Alarms after other-lane activity are labelled (never silenced); AI sightings count as real for learning."""
import time
import unittest
from unittest import mock

from test_watch import watch, watcher
from alert_learning import AlertLearning, FOLLOW_S, OPERATOR_WRONG_MIN
import dashboard_ui


class CrossLaneTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.exit = next(z for z in self.obj.zones if z.route_role == 'exit')
        self.entrance = next(z for z in self.obj.zones if z.route_role == 'entrance')
        self.now = time.monotonic()

    def test_exit_alarm_just_after_entrance_activity_is_labelled(self):
        self.obj._route_activity = {self.entrance.route_group: (self.now - 3, 'entrance')}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.assertEqual('entrance', context['cross_lane']['role'])
        self.assertIn('coming in', context['cross_lane']['text'])

    def test_old_same_lane_or_footpath_activity_is_not_a_label(self):
        for activity in ({self.entrance.route_group: (self.now - 15, 'entrance')},
                         {self.exit.route_group: (self.now - 1, 'exit')},
                         {'Pedestrian footpath': (self.now - 1, 'pedestrian')}):
            with self.subTest(activity=activity):
                self.obj._route_activity = activity
                self.assertNotIn('cross_lane', self.obj._alarm_context(self.exit, {}, self.now))

    def test_ai_sightings_mark_the_lane_and_teach_learning(self):
        self.obj.learning = mock.Mock()
        primary = {'detections': [
            {'label': 'truck', 'confidence': .62, 'region_status': {self.entrance.name: 'tracking approach'}},
            {'label': 'car', 'confidence': .2, 'region_status': {self.exit.name: 'below confidence threshold'}},
            {'label': 'dog', 'confidence': .9, 'region_status': {self.exit.name: 'confirmed'}}]}
        self.obj._note_ai_activity(primary, self.now)
        self.assertEqual({self.entrance.route_group}, set(self.obj._route_activity))
        self.obj.learning.note_recognised.assert_called_once_with(self.entrance.route_group)
        context = self.obj._alarm_context(self.exit, primary, self.now + 1)
        self.assertIn('cross_lane', context)
        self.assertEqual(['truck', 'car', 'dog'], [item['label'] for item in context['ai_seen']])

    def test_broken_ai_status_never_stops_an_alarm(self):
        self.assertEqual({}, self.obj._alarm_context(self.exit, mock.Mock(), self.now))
        self.obj._note_ai_activity(mock.Mock(), self.now)          # no exception
        self.obj._note_ai_activity({'detections': [None, 'x', {'label': 'car'}]}, self.now)


class LearningAdviceTests(unittest.TestCase):
    def test_wrong_marks_after_other_lane_activity_get_advice_not_a_threshold(self):
        learning = AlertLearning(None)
        details = {'source': 'light', 'zone_name': 'Exit approach', 'route_group': 'campus-exit',
                   'cross_lane': {'role': 'entrance'}}
        for index in range(OPERATOR_WRONG_MIN):
            learning.note_feedback(f'a{index}', 'wrong', details)
        config = {'zones': [{'name': 'Exit approach', 'light': True, 'light_delta': 10}]}
        suggestion = learning.suggestions(config)[0]
        self.assertIsNone(suggestion['key'])
        self.assertTrue(suggestion['advice'])
        self.assertIn('other lane', suggestion['text'])

    def test_pending_alert_keeps_its_label_when_marked(self):
        clock = [100.]
        learning = AlertLearning(None, clock=lambda: clock[0])
        learning.note_alarm('a1', {'source': 'light', 'zone_name': 'Exit approach', 'route_group': 'campus-exit',
                                   'cross_lane': {'role': 'entrance'}})
        learning.note_feedback('a1', 'wrong')
        clock[0] += FOLLOW_S + 1
        learning.tick()
        row = learning._kinds['Exit approach|light']
        self.assertTrue(row['labels']['a1']['cross'])


class DashboardLabelTests(unittest.TestCase):
    def test_activity_shows_the_label(self):
        page = dashboard_ui.dashboard_page('Gatewatch')
        self.assertIn('Other lane?', page)
        self.assertIn('event-dialog-cross', page)


if __name__ == '__main__':
    unittest.main()
