"""Faults found on the running gate, kept as tests.

A barrier reading that has never been taken carries ``observed_at = None``. The
whole barrier check used to raise a TypeError on that, once per frame, which
silently disabled every barrier feature: auto-acknowledgement, the open-barrier
reminder and the dashboard readout. Nothing in the alarm path noticed, because
the exception was caught and its traceback discarded.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import barrier_workflow  # noqa: E402


class PedestrianArmStampTests(unittest.TestCase):
    """The exact expression that failed, isolated."""

    @staticmethod
    def earliest(arms):
        stamps = [value for value in (a.get("observed_at") for a in arms)
                  if isinstance(value, (int, float)) and not isinstance(value, bool)]
        return min(stamps, default=0.)

    def test_two_unread_arms_do_not_raise(self):
        arms = [{"state": "unknown", "observed_at": None},
                {"state": "unknown", "observed_at": None}]
        self.assertEqual(self.earliest(arms), 0.)

    def test_one_read_and_one_unread_arm_uses_the_real_stamp(self):
        arms = [{"state": "down", "observed_at": 1234.5}, {"state": "unknown", "observed_at": None}]
        self.assertEqual(self.earliest(arms), 1234.5)

    def test_the_earliest_reading_wins(self):
        arms = [{"state": "down", "observed_at": 900.0}, {"state": "down", "observed_at": 950.0}]
        self.assertEqual(self.earliest(arms), 900.0)

    def test_a_boolean_is_not_a_timestamp(self):
        self.assertEqual(self.earliest([{"observed_at": True}, {"observed_at": None}]), 0.)

    def test_a_missing_key_is_handled_like_an_unread_arm(self):
        self.assertEqual(self.earliest([{"state": "unknown"}, {"observed_at": None}]), 0.)


class SourceSanityTests(unittest.TestCase):
    def test_the_workflow_no_longer_defaults_a_present_none(self):
        """dict.get(key, default) does not replace a value that is present and None."""
        source = Path(barrier_workflow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("a.get('observed_at', 0.)", source)
        # Footpaths no longer synthesize a barrier from those two stamps;
        # end-to-end lane tests cover unread arms and footpath isolation.
        self.assertNotIn("min(stamps, default=0.)", source)


class CurrentVisitMovementTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.rows = []
        monitor = SimpleNamespace(update=lambda *args, **kwargs: None,
                                  status=lambda: {'barriers': self.rows})
        self.workflow = barrier_workflow.BarrierWorkflow(
            'synthetic', '.', None, monitor=monitor, clock=lambda: self.now)
        self.zone = SimpleNamespace(name='Entrance', route_role='entrance',
                                    route_group='campus-entrance', motion_active=False,
                                    light_active=False)
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def step(self, stamp, state, alarm=None, position_since=None):
        self.now = stamp
        self.rows = [{'id': 'entrance', 'route_group': 'entrance', 'state': state,
                      'fresh': True, 'calibrated': True, 'observed_at': stamp,
                      'position_since': stamp if position_since is None else position_since}]
        primary = {'available': True, 'state': 'ready', 'observed_at': stamp,
                   'detections': [{'regions': ['Entrance']}]}
        return self.workflow.observe(self.frame, stamp, primary, [self.zone], True, alarm)

    def test_previous_car_lowering_does_not_resolve_new_car(self):
        self.step(100., 'up')
        alarm = {'id': 'second-car', 'route': 'campus-entrance', 'started_at': 101.}
        self.assertEqual([], self.step(101., 'unknown', alarm))
        self.assertEqual([], self.step(101.7, 'down', alarm, position_since=100.9))

    def test_candidate_timestamp_survives_workflow_and_blocks_late_confirmation(self):
        alarm = {'id': 'second-car', 'route': 'campus-entrance', 'started_at': 101.}
        self.step(101.1, 'up', alarm)
        self.assertEqual([], self.step(101.7, 'down', alarm, position_since=100.9))

    def test_fresh_current_visit_change_keeps_timestamp_proof(self):
        alarm = {'id': 'second-car', 'route': 'campus-entrance', 'started_at': 101.}
        self.step(101.1, 'up', alarm)
        self.step(101.2, 'unknown', alarm)
        actions = self.step(101.7, 'down', alarm, position_since=101.3)
        self.assertEqual([{'type': 'lowered', 'alarm_id': 'second-car',
                           'route': 'campus-entrance', 'observed_at': 101.7,
                           'origin_observed_at': 101.1, 'alarm_started_at': 101.,
                           'position_since': 101.3}], actions)


if __name__ == "__main__":
    unittest.main()
