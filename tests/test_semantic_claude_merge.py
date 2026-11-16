"""Preserve the 01:40 production additions alongside bounded diagnostic work."""
import unittest
from unittest.mock import Mock, patch

import test_semantic_latest_pending as pending_helpers


class SemanticMergeTests(unittest.TestCase):
    make_monitor = pending_helpers.LatestPendingTests.make_monitor
    frame = staticmethod(pending_helpers.LatestPendingTests.frame)
    send = pending_helpers.LatestPendingTests.send

    def test_moving_floor_is_preserved_separately_from_diagnostic_and_display_floors(self):
        monitor = self.make_monitor(interval=.5)
        zone = monitor.zones[0]
        zone.ai_confidence = .6
        zone.ai_confidence_moving = .45
        zone.approach_motion = True
        self.assertEqual(.45, monitor._candidate_confidence())
        self.assertEqual(.6, monitor._display_floor())
        self.assertEqual(.3, monitor._inference_confidence(monitor._candidate_confidence()))
        rows = [{'confidence': .4, 'raw_confidence': .4},
                {'confidence': .5, 'raw_confidence': .5},
                {'confidence': .7, 'raw_confidence': .7}]
        self.assertEqual(rows[1:], monitor._normal_detections(rows, monitor._candidate_confidence()))
        effective = monitor._tracking_zones()[0]
        self.assertEqual(.6, effective.ai_confidence)
        self.assertEqual(.45, effective.ai_confidence_moving)

    def test_boosted_cadence_can_replace_pending_work_then_expires(self):
        monitor = self.make_monitor(interval=.5)
        self.send(monitor, 100., 1)
        with patch('semantic_monitor.time.monotonic', return_value=100.):
            monitor.boost(4.)
        self.send(monitor, 100.11, 2)
        self.assertEqual(1, monitor._pipeline['inputs_submitted'])
        self.send(monitor, 100.13, 3)
        self.assertEqual(2, monitor._pipeline['inputs_submitted'])
        self.assertEqual(1, monitor._pipeline['pending_replaced'])
        self.assertEqual(100.13, monitor._input.get_nowait()[2])
        self.send(monitor, 103.9, 4)
        self.send(monitor, 104.03, 5)
        self.assertEqual(3, monitor._pipeline['inputs_submitted'], 'ordinary cadence resumes after boost')
        self.assertEqual(.5, monitor._current_interval(104.03))

    def test_unconfirmed_moving_only_results_are_hidden_but_confirmed_results_remain(self):
        monitor = self.make_monitor()
        monitor.zones[0].ai_confidence = .6
        monitor.zones[0].ai_confidence_moving = .45
        observed = [{'confidence': .5, 'confirmed_regions': []},
                    {'confidence': .5, 'confirmed_regions': ['Gate']},
                    {'confidence': .7, 'confirmed_regions': []}]
        monitor._tracker = Mock(update=Mock(return_value=(['event'], observed)))
        self.assertEqual(['event'], monitor._confirm([], 100, 100, 100.))
        self.assertEqual(observed[1:], monitor._status['detections'])


if __name__ == '__main__':
    unittest.main()
