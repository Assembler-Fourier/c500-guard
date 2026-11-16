"""A barrier must be matched to the approach it controls.

Found on the running gate: the monitor was reading both arms correctly ("up",
confidence 0.56 and 0.81) while every route still reported barrier "unknown".
Barriers are named for the movement they control ("exit"); zones name their
approach ("campus-exit"), so the two never met and every barrier feature -
auto-acknowledgement, the open-barrier reminder, the dashboard readout - stayed
silent with nothing reporting a fault.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from barrier_workflow import BarrierWorkflow  # noqa: E402

match = BarrierWorkflow._route_for


class RouteMappingTests(unittest.TestCase):
    ROUTES = ("campus-exit", "campus-entrance", "Pedestrian footpath")

    def test_the_real_campus_names_are_matched(self):
        self.assertEqual(match("exit", self.ROUTES), "campus-exit")
        self.assertEqual(match("entrance", self.ROUTES), "campus-entrance")

    def test_an_exact_name_wins(self):
        routes = ("exit", "campus-exit")
        self.assertEqual(match("exit", routes), "exit")

    def test_case_does_not_matter(self):
        self.assertEqual(match("Exit", ("Campus-Exit",)), "Campus-Exit")

    def test_an_ambiguous_match_is_refused(self):
        """Two approaches ending the same way must not silently pick one."""
        self.assertIsNone(match("exit", ("north-exit", "south-exit")))

    def test_an_unmatched_barrier_is_refused(self):
        self.assertIsNone(match("service", self.ROUTES))
        self.assertIsNone(match("exit", ("Pedestrian footpath",)))

    def test_rubbish_input_is_refused(self):
        for identity in (None, "", 5, True, ["exit"]):
            self.assertIsNone(match(identity, self.ROUTES))
        self.assertIsNone(match("exit", ()))
        self.assertIsNone(match("exit", (None, 7)))

    def test_a_footpath_is_never_given_an_arm_of_its_own(self):
        self.assertIsNone(match("exit", ("Pedestrian footpath",)))
        self.assertIsNone(match("entrance", ("Pedestrian footpath",)))


if __name__ == "__main__":
    unittest.main()
